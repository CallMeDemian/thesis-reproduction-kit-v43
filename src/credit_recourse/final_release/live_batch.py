"""Explicitly gated live Batch transport for V4.3 Final Plan-3."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Mapping

from .common import ContractError, file_hash, load_json, utc_now, write_json
from .ledger import (
    append_event,
    begin_shard_submission,
    bind_remote_job,
    connect,
    record_shard_cancellation_provenance,
    reset_shard_after_confirmed_remote_absence,
)
from .executor import load_release
from .retry_policy import (
    CancellationOrigin,
    is_control_plane_transport_exception,
    normalize_provider_job_state,
)


LIVE_SENTINEL = "I_APPROVE_LLM_V43_FINAL_PLAN3_BATCH"


def assert_live_authorized(
    *,
    approved: bool,
    release: Mapping[str, Any],
    attempt_index: int | None = None,
) -> None:
    if attempt_index is not None and int(attempt_index) not in {1, 2, 3}:
        raise ContractError("attempt_index must be 1..3")
    pilot = release.get("run_kind") == "engineering_pilot"
    env_name = "CREDIT_RECOURSE_ENABLE_LLM_V43_PILOT" if pilot else "CREDIT_RECOURSE_ENABLE_LLM_V43_FINAL_PLAN3"
    sentinel = "I_APPROVE_LLM_V43_COMMON_PILOT" if pilot else LIVE_SENTINEL
    if not approved or os.environ.get(env_name) != sentinel:
        raise ContractError(
            f"Live submission requires --live-api-approved and the {'pilot' if pilot else 'production'} sentinel"
        )
    if release.get("status") not in {"PILOT_FROZEN_READY_WAVE1", "FROZEN_READY_WAVE1"}:
        raise ContractError("Release is not frozen for live Batch submission")
    missing = [name for name in ("OPENAI_API_KEY",) if not os.environ.get(name)]
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        missing.append("GEMINI_API_KEY/GOOGLE_API_KEY")
    sdks = []
    for module in ("openai", "google.genai"):
        try:
            installed = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError):
            installed = False
        if not installed:
            sdks.append(module)
    if missing or sdks:
        raise ContractError(f"Provider preflight failed: missing_keys={missing}, missing_sdks={sdks}")


def _openai_submit(path: Path, release_hash: str, wave: int, shard_id: str | None = None) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    with path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    metadata = {"protocol_release_hash": release_hash, "wave": str(wave)}
    if shard_id:
        metadata["shard_id"] = shard_id[:64]
    job = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=metadata,
    )
    return {"provider": "openai", "batch_job_id": job.id, "input_file_id": uploaded.id, "state": str(job.status)}


def _google_submit(path: Path, model: str, display_name: str) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    uploaded = client.files.upload(
        file=path,
        config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
    )
    job = client.batches.create(model=model, src=uploaded.name, config={"display_name": display_name})
    return {"provider": "google", "batch_job_id": job.name, "input_file_id": uploaded.name, "state": str(job.state)}


def _find_remote_jobs(provider: str, correlation_key: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if provider == "openai":
        from openai import OpenAI

        page = OpenAI().batches.list(limit=100)
        while True:
            for job in page:
                metadata = dict(getattr(job, "metadata", None) or {})
                if str(metadata.get("shard_id") or "") == correlation_key:
                    matches.append(
                        {
                            "provider": "openai",
                            "batch_job_id": str(job.id),
                            "input_file_id": str(getattr(job, "input_file_id", "") or ""),
                            "state": str(getattr(job, "status", "")),
                        }
                    )
            if not getattr(page, "has_next_page", lambda: False)():
                break
            page = page.get_next_page()
    elif provider == "google":
        from google import genai

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        expected_name = f"llm-v43-{correlation_key}"
        for job in client.batches.list(config={"page_size": 100}):
            if str(getattr(job, "display_name", "") or "") == expected_name:
                state = str(getattr(getattr(job, "state", None), "name", getattr(job, "state", "")))
                matches.append(
                    {"provider": "google", "batch_job_id": str(job.name), "input_file_id": "", "state": state}
                )
    else:
        raise ContractError(f"Unknown provider: {provider}")
    by_id = {item["batch_job_id"]: item for item in matches}
    return [by_id[key] for key in sorted(by_id)]


def recover_submitting_shard(connection: Any, *, shard_id: str, provider: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT shard_state,provider_job_id,submission_correlation_key FROM shards WHERE shard_id=?",
        (shard_id,),
    ).fetchone()
    if row is None:
        raise ContractError("Cannot recover an unknown shard")
    if row["provider_job_id"]:
        return {"status": "ALREADY_BOUND", "batch_job_id": str(row["provider_job_id"])}
    if row["shard_state"] != "SUBMITTING" or not row["submission_correlation_key"]:
        raise ContractError("Shard is not an unbound correlated SUBMITTING shard")
    correlation = str(row["submission_correlation_key"])
    candidates = _find_remote_jobs(provider, correlation)
    if len(candidates) > 1:
        raise ContractError(f"Multiple remote jobs match shard correlation {correlation}; fail closed")
    if len(candidates) == 1:
        job = candidates[0]
        bind_remote_job(connection, shard_id=shard_id, provider_job_id=job["batch_job_id"])
        return {"status": "RECOVERED_REMOTE_JOB", **job}
    reset_shard_after_confirmed_remote_absence(connection, shard_id=shard_id, correlation_key=correlation)
    return {"status": "CONFIRMED_ABSENT_RESET"}


def reconcile_remote_job(
    project_root: Path,
    release_hash: str,
    *,
    shard_id: str,
    provider_job_id: str,
) -> dict[str, Any]:
    directory, _ = load_release(project_root, release_hash)
    ledger = connect(directory / "generation_ledger.sqlite")
    try:
        bind_remote_job(ledger, shard_id=shard_id, provider_job_id=provider_job_id)
        return {"status": "RECONCILED", "shard_id": shard_id, "provider_job_id": provider_job_id}
    finally:
        ledger.close()


def submit_wave(
    project_root: Path,
    release_hash: str,
    wave: int,
    *,
    approved: bool,
    attempt_index: int = 1,
    max_new_shards_per_provider: int | None = None,
    provider_filter: str | None = None,
) -> dict[str, Any]:
    directory, release = load_release(project_root, release_hash)
    if int(attempt_index) not in {1, 2, 3}:
        raise ContractError("attempt_index must be 1..3")
    if provider_filter is not None and provider_filter not in {"google", "openai"}:
        raise ContractError("provider_filter must be google or openai when provided")
    assert_live_authorized(approved=approved, release=release, attempt_index=attempt_index)
    prepared_name = (
        f"wave{int(wave)}_prepared.json"
        if int(attempt_index) == 1
        else f"wave{int(wave)}_attempt{int(attempt_index)}_retry_prepared.json"
    )
    prepared = load_json(directory / "manifests" / prepared_name)
    if prepared.get("status") != "PREPARED":
        raise ContractError(f"Wave {wave} attempt {attempt_index} is not prepared")
    jobs_path = directory / "manifests" / f"wave{int(wave)}_attempt{int(attempt_index)}_jobs.json"
    jobs = (
        load_json(jobs_path)
        if jobs_path.exists()
        else {
            "schema_version": "llm_v43_live_jobs_v3",
            "protocol_release_hash": release_hash,
            "wave": int(wave),
            "attempt_index": int(attempt_index),
            "jobs": [],
            "submission_complete": False,
        }
    )
    by_shard = {item["shard_id"]: item for item in jobs["jobs"]}
    if max_new_shards_per_provider is not None and int(max_new_shards_per_provider) < 1:
        raise ContractError("max_new_shards_per_provider must be positive when provided")
    newly_submitted_by_provider: dict[str, int] = {}
    ledger = connect(directory / "generation_ledger.sqlite")
    existing_bound_by_provider = {
        str(row["provider"]): int(row["n"])
        for row in ledger.execute(
            "SELECT provider,COUNT(*) AS n FROM shards WHERE wave=? AND attempt_index=? AND provider_job_id IS NOT NULL GROUP BY provider",
            (int(wave), int(attempt_index)),
        ).fetchall()
    }
    try:
        for item in jobs["jobs"]:
            bind_remote_job(ledger, shard_id=item["shard_id"], provider_job_id=item["batch_job_id"])
        for item in prepared["provider_batches"]:
            sid = item["shard_id"]
            provider = str(item["provider"])
            if provider_filter is not None and provider != provider_filter:
                continue
            if sid in by_shard:
                continue
            shard = ledger.execute("SELECT shard_state,provider_job_id,ordinal FROM shards WHERE shard_id=?", (sid,)).fetchone()
            if shard is None:
                raise ContractError("Prepared manifest references an unregistered shard")
            if shard["provider_job_id"]:
                recovered_bound = {
                    "provider": provider,
                    "batch_job_id": str(shard["provider_job_id"]),
                    "input_file_id": "",
                    "state": "RECOVERED_ALREADY_BOUND",
                    "shard_id": sid,
                    "model_key": item["model_key"],
                    "replicate": item.get("replicate"),
                    "ordinal": int(shard["ordinal"]),
                    "input_path": item["path"],
                    "input_sha256": item["sha256"],
                    "requests": int(item["requests"]),
                    "submitted_utc": utc_now(),
                    "recovered": True,
                }
                jobs["jobs"].append(recovered_bound)
                by_shard[sid] = recovered_bound
                write_json(jobs_path, jobs)
                bind_remote_job(ledger, shard_id=sid, provider_job_id=shard["provider_job_id"])
                continue
            if shard["shard_state"] == "SUBMITTING":
                recovered = recover_submitting_shard(ledger, shard_id=sid, provider=str(item["provider"]))
                if recovered["status"] == "RECOVERED_REMOTE_JOB":
                    recovered.update(
                        {
                            "shard_id": sid,
                            "model_key": item["model_key"],
                            "replicate": item.get("replicate"),
                            "ordinal": int(shard["ordinal"]),
                            "input_path": item["path"],
                            "input_sha256": item["sha256"],
                            "requests": int(item["requests"]),
                            "submitted_utc": utc_now(),
                            "recovered": True,
                        }
                    )
                    jobs["jobs"].append(recovered)
                    by_shard[sid] = recovered
                    write_json(jobs_path, jobs)
                    continue
            if (
                max_new_shards_per_provider is not None
                and existing_bound_by_provider.get(provider, 0) + newly_submitted_by_provider.get(provider, 0) >= int(max_new_shards_per_provider)
            ):
                continue
            source = directory / item["path"]
            if file_hash(source) != item["sha256"]:
                raise ContractError("Prepared shard changed before submission")
            begin_shard_submission(ledger, shard_id=sid, correlation_key=sid)
            if provider == "openai":
                job = _openai_submit(source, release_hash, int(wave), sid)
            elif provider == "google":
                model_spec = (release.get("runtime_binding") or {}).get("models", {}).get(str(item["model_key"]), {})
                exact_model_id = str(model_spec.get("exact_model_id") or "")
                if not exact_model_id:
                    raise ContractError("Frozen release has no exact Gemini model identifier for prepared shard")
                job = _google_submit(source, exact_model_id, f"llm-v43-{sid}")
            else:
                raise ContractError(f"Unknown provider: {provider}")
            job.update(
                {
                    "shard_id": sid,
                    "model_key": item["model_key"],
                    "replicate": item.get("replicate"),
                    "ordinal": int(shard["ordinal"]),
                    "input_path": item["path"],
                    "input_sha256": item["sha256"],
                    "requests": int(item["requests"]),
                    "submitted_utc": utc_now(),
                }
            )
            jobs["jobs"].append(job)
            by_shard[sid] = job
            newly_submitted_by_provider[provider] = newly_submitted_by_provider.get(provider, 0) + 1
            write_json(jobs_path, jobs)
            bind_remote_job(ledger, shard_id=sid, provider_job_id=job["batch_job_id"])
        remaining_shards = [item["shard_id"] for item in prepared["provider_batches"] if item["shard_id"] not in by_shard]
        jobs["submission_complete"] = not remaining_shards
        jobs["status"] = "SUBMITTED" if not remaining_shards else "PARTIALLY_SUBMITTED"
        jobs["newly_submitted_by_provider"] = newly_submitted_by_provider
        jobs["existing_bound_by_provider_at_call_start"] = existing_bound_by_provider
        jobs["max_new_shards_per_provider"] = max_new_shards_per_provider
        jobs["provider_filter"] = provider_filter
        jobs["remaining_shard_count"] = len(remaining_shards)
        write_json(jobs_path, jobs)
    finally:
        ledger.close()
    release["production_started"] = True
    release.setdefault("production_started_utc", utc_now())
    write_json(directory / "protocol_release.json", release)
    return jobs


def _openai_artifact_target(target: Path, kind: str) -> Path:
    if kind == "output":
        return target
    suffix = "_output.jsonl"
    if not target.name.endswith(suffix):
        raise ContractError("OpenAI output target must end with _output.jsonl")
    return target.with_name(target.name[: -len(suffix)] + f"_{kind}.jsonl")


def _openai_terminal_ready(state: str) -> bool:
    return normalize_provider_job_state("openai", state) in {"COMPLETED", "FAILED", "EXPIRED", "CANCELLED"}


def _download_openai(job_id: str, target: Path) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    job = client.batches.retrieve(job_id)
    state = str(job.status)
    if not _openai_terminal_ready(state):
        return {"state": state, "artifacts": []}
    artifacts: list[dict[str, Any]] = []
    for kind, file_id in (("output", getattr(job, "output_file_id", None)), ("error", getattr(job, "error_file_id", None))):
        if not file_id:
            continue
        path = _openai_artifact_target(target, kind)
        client.files.content(file_id).write_to_file(path)
        if not path.is_file():
            raise ContractError(f"OpenAI {kind}_file download did not create its target")
        artifacts.append(
            {
                "kind": kind,
                "file_id": str(file_id),
                "path": path,
                "sha256": file_hash(path),
            }
        )
    if normalize_provider_job_state("openai", state) == "COMPLETED" and not artifacts:
        raise ContractError("Completed OpenAI batch has neither output_file_id nor error_file_id")
    return {"state": state, "artifacts": artifacts}


def _google_terminal_ready(state: str) -> bool:
    return normalize_provider_job_state("google", state) in {"SUCCEEDED", "FAILED", "EXPIRED", "CANCELLED"}


def _download_google(job_id: str, target: Path) -> dict[str, Any]:
    from google import genai

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    job = client.batches.get(name=job_id)
    state = str(getattr(getattr(job, "state", None), "name", getattr(job, "state", "")))
    if not _google_terminal_ready(state):
        return {"state": state, "artifacts": []}
    dest = getattr(job, "dest", None)
    name = getattr(dest, "file_name", None)
    artifacts: list[dict[str, Any]] = []
    if name:
        content = client.files.download(file=name)
        target.write_bytes(content if isinstance(content, bytes) else bytes(content))
        artifacts.append({"kind": "output", "file_id": str(name), "path": target, "sha256": file_hash(target)})
    if normalize_provider_job_state("google", state) == "SUCCEEDED" and not artifacts:
        raise ContractError("Completed Gemini batch has no output file")
    return {"state": state, "artifacts": artifacts}


def _terminal_failure(provider: str, state: str) -> bool:
    return normalize_provider_job_state(provider, state) in {"FAILED", "EXPIRED", "CANCELLED"}


def cancel_shard(
    project_root: Path,
    release_hash: str,
    *,
    shard_id: str,
    reason: str,
    approved: bool,
) -> dict[str, Any]:
    directory, release = load_release(project_root, release_hash)
    assert_live_authorized(approved=approved, release=release)
    ledger = connect(directory / "generation_ledger.sqlite")
    try:
        row = ledger.execute(
            "SELECT provider,provider_job_id,shard_state FROM shards WHERE shard_id=?",
            (shard_id,),
        ).fetchone()
        if row is None or not row["provider_job_id"]:
            raise ContractError("Only a bound remote shard may be cancelled")
        if row["shard_state"] in {"COLLECTED", "FAILED_TERMINAL"}:
            raise ContractError("Collected/terminal shard cannot be cancelled")
        record_shard_cancellation_provenance(
            ledger,
            shard_id=shard_id,
            origin=CancellationOrigin.USER_INTENTIONAL,
            reason=reason,
            actor="final_release_cli",
        )
        if row["provider"] == "openai":
            from openai import OpenAI

            job = OpenAI().batches.cancel(str(row["provider_job_id"]))
            state = str(getattr(job, "status", ""))
        elif row["provider"] == "google":
            from google import genai

            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
            job = client.batches.cancel(name=str(row["provider_job_id"]))
            state = str(getattr(getattr(job, "state", None), "name", getattr(job, "state", "")))
        else:
            raise ContractError(f"Unknown provider: {row['provider']}")
        append_event(
            ledger,
            event_type="PROVIDER_SHARD_CANCEL_REQUESTED",
            shard_id=shard_id,
            detail={"provider": row["provider"], "provider_job_id": row["provider_job_id"], "state": state},
        )
        return {"status": "CANCEL_REQUESTED", "shard_id": shard_id, "provider_job_id": row["provider_job_id"], "state": state}
    finally:
        ledger.close()


def poll_and_download(
    project_root: Path,
    release_hash: str,
    wave: int,
    *,
    attempt_index: int = 1,
) -> dict[str, Any]:
    directory, _ = load_release(project_root, release_hash)
    jobs = load_json(directory / "manifests" / f"wave{int(wave)}_attempt{int(attempt_index)}_jobs.json")
    states: list[dict[str, Any]] = []
    ledger = connect(directory / "generation_ledger.sqlite")
    try:
        for item in jobs["jobs"]:
            sid = item["shard_id"]
            bind_remote_job(ledger, shard_id=sid, provider_job_id=item["batch_job_id"])
            row = ledger.execute(
                "SELECT cancellation_origin,cancellation_reason,cancellation_actor FROM shards WHERE shard_id=?",
                (sid,),
            ).fetchone()
            provider = str(item["provider"])
            target = (
                directory
                / "provider_batches"
                / f"wave{int(wave)}"
                / provider
                / f"r{int(item.get('replicate') or 1)}"
                / f"shard_{int(item['ordinal']):05d}_attempt{int(attempt_index)}_output.jsonl"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            control_plane_error = None
            try:
                download = (
                    _download_openai(item["batch_job_id"], target)
                    if provider == "openai"
                    else _download_google(item["batch_job_id"], target)
                    if provider == "google"
                    else (_ for _ in ()).throw(ContractError(f"Unknown provider in frozen two-model panel: {provider}"))
                )
            except Exception as exc:
                if not is_control_plane_transport_exception(exc):
                    raise
                # The asynchronous remote job may be running or already complete.
                # Retry polling/download of this exact provider_job_id only.
                download = {"state": "CONTROL_PLANE_TRANSPORT_FAILURE", "artifacts": []}
                control_plane_error = type(exc).__name__
                append_event(
                    ledger,
                    event_type="PROVIDER_CONTROL_PLANE_TRANSPORT_FAILURE",
                    shard_id=sid,
                    detail={
                        "provider": provider,
                        "provider_job_id": item["batch_job_id"],
                        "exception_type": control_plane_error,
                        "recovery": "RETRY_SAME_PROVIDER_JOB_POLL_OR_DOWNLOAD",
                    },
                )
            state = str(download["state"])
            terminal = _terminal_failure(provider, state)
            ready = terminal or (
                _openai_terminal_ready(state) if provider == "openai" else _google_terminal_ready(state)
            )
            artifacts = [
                {
                    "kind": artifact["kind"],
                    "file_id": artifact["file_id"],
                    "path": Path(artifact["path"]).relative_to(directory).as_posix(),
                    "sha256": artifact["sha256"],
                }
                for artifact in download["artifacts"]
            ]
            if artifacts:
                with ledger:
                    ledger.execute(
                        "UPDATE shards SET shard_state='OUTPUT_DOWNLOADED',resolved_utc=? WHERE shard_id=?",
                        (utc_now(), sid),
                    )
            states.append(
                {
                    "shard_id": sid,
                    "provider": provider,
                    "batch_job_id": item["batch_job_id"],
                    "state": state,
                    "terminal_failure": terminal,
                    "collection_ready": ready,
                    "transport_retry_class": None,
                    "control_plane_retry_pending": control_plane_error is not None,
                    "control_plane_exception_type": control_plane_error,
                    "cancellation_origin": row["cancellation_origin"] if row else None,
                    "cancellation_reason": row["cancellation_reason"] if row else None,
                    "cancellation_actor": row["cancellation_actor"] if row else None,
                    "artifacts": artifacts,
                    "downloaded": bool(artifacts),
                    "output_path": next((artifact["path"] for artifact in artifacts if artifact["kind"] == "output"), None),
                }
            )
        resolved = bool(states) and all(item["collection_ready"] for item in states)
        result = {
            "schema_version": "llm_v43_batch_poll_v4",
            "protocol_release_hash": release_hash,
            "wave": int(wave),
            "attempt_index": int(attempt_index),
            "all_terminal_or_downloaded": resolved,
            "jobs": states,
            "checked_utc": utc_now(),
        }
    finally:
        ledger.close()
    write_json(directory / "manifests" / f"wave{int(wave)}_attempt{int(attempt_index)}_poll.json", result)
    return collect_downloaded(project_root, release_hash, wave, attempt_index=attempt_index) if resolved else result


def collect_downloaded(project_root: Path, release_hash: str, wave: int, *, attempt_index: int = 1) -> dict[str, Any]:
    from .collection import collect_downloaded as collect

    return collect(project_root, release_hash, wave, attempt_index=attempt_index)
