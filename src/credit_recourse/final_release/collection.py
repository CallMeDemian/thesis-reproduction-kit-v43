from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .common import ContractError, iter_jsonl, load_json, utc_now, write_json, write_jsonl
from .ledger import (
    append_event,
    connect,
    lock_selected_attempt_with_parse,
    mark_shard_terminal_failure,
    record_late_duplicate,
    resolve_attempt,
)
from .executor import generation_status, load_release
from .parsing import parse_policy_response, parse_probe_response
from .providers import normalize_google, normalize_openai
from .retry import prepare_retry
from .retry_policy import RETRYABLE_TECHNICAL_CLASSES, RetryClass, is_retryable_class


def _artifact_rows(directory: Path, state: Mapping[str, Any]) -> list[tuple[str, Path, Mapping[str, Any]]]:
    artifacts = list(state.get("artifacts") or [])
    if not artifacts and state.get("output_path"):
        artifacts = [{"kind": "output", "path": state["output_path"]}]
    rows: list[tuple[str, Path, Mapping[str, Any]]] = []
    seen_paths: set[tuple[str, str]] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ContractError("Provider artifact descriptor must be an object")
        kind = str(artifact.get("kind") or "")
        if kind not in {"output", "error"}:
            raise ContractError(f"Unsupported provider artifact kind: {kind}")
        path = directory / str(artifact.get("path") or "")
        key = (kind, str(path.resolve()))
        if key in seen_paths:
            raise ContractError("Provider artifact manifest duplicates a source path")
        seen_paths.add(key)
        if not path.is_file():
            raise ContractError(f"Downloaded provider artifact is missing: {path}")
        for raw in iter_jsonl(path):
            if not isinstance(raw, Mapping):
                raise ContractError("Provider artifact JSONL row must be an object")
            rows.append((kind, path, raw))
    return rows


def _missing_retry_class(state: Mapping[str, Any]) -> str:
    transport = state.get("transport_retry_class")
    if transport:
        return str(transport)
    if state.get("terminal_failure"):
        raise ContractError("Terminal missing rows must be resolved by terminal job classifier")
    return RetryClass.PROVIDER_RESULT_MISSING_NONRETRYABLE.value


def _selected_attempt_for_parse(ledger: Any, request_id: str) -> Any:
    request = ledger.execute(
        "SELECT selected_attempt_index FROM logical_requests WHERE request_id=?", (request_id,)
    ).fetchone()
    if request is None:
        raise ContractError(f"Unknown request: {request_id}")
    if request["selected_attempt_index"] is not None:
        index = int(request["selected_attempt_index"])
        attempt = ledger.execute(
            "SELECT * FROM attempts WHERE request_id=? AND attempt_index=?", (request_id, index)
        ).fetchone()
    else:
        attempt = ledger.execute(
            "SELECT * FROM attempts WHERE request_id=? AND attempt_state='DELIVERED' ORDER BY attempt_index LIMIT 1",
            (request_id,),
        ).fetchone()
    if attempt is None:
        raise ContractError("No delivered completion is available for parsing")
    return attempt


def _ensure_selected_parse(ledger: Any, requests: pd.DataFrame, design: Any, request_id: str) -> None:
    attempt = _selected_attempt_for_parse(ledger, request_id)
    raw_value = attempt["raw_visible_text"]
    if isinstance(raw_value, memoryview):
        raw_value = raw_value.tobytes()
    raw_text = raw_value.decode("utf-8") if isinstance(raw_value, bytes) else raw_value
    factors = requests.loc[request_id]
    if factors["mode"] == "probe":
        parsed = parse_probe_response(raw_text)
    else:
        completion = str(attempt["completion_kind"] or "")
        parsed = parse_policy_response(
            design,
            raw_visible_text=raw_text,
            mode=str(factors["mode"]),
            budget=str(factors["budget"]),
            provider_refusal=completion == "provider_refusal",
            output_limit=completion == "output_limit",
        ).to_dict()
    lock_selected_attempt_with_parse(
        ledger,
        request_id,
        expected_attempt_index=int(attempt["attempt_index"]),
        parse_json=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
    )


def collect_downloaded(project_root: Path, release_hash: str, wave: int, *, attempt_index: int = 1) -> dict[str, Any]:
    directory, release = load_release(project_root, release_hash)
    jobs = load_json(directory / "manifests" / f"wave{int(wave)}_attempt{int(attempt_index)}_jobs.json")
    poll_path = directory / "manifests" / f"wave{int(wave)}_attempt{int(attempt_index)}_poll.json"
    poll = load_json(poll_path) if poll_path.exists() else {"jobs": []}
    poll_by_shard = {item["shard_id"]: item for item in poll["jobs"]}
    requests = pd.read_parquet(directory / "logical_requests.parquet").set_index("request_id", drop=False)
    models = release["runtime_binding"]["models"]
    normalizers = {"openai": normalize_openai, "google": normalize_google}
    from .contract import load_design

    design = load_design(project_root)
    ledger = connect(directory / "generation_ledger.sqlite")
    normalized: list[dict[str, Any]] = []
    missing = 0
    explicit_provider_errors = 0
    late = 0
    try:
        for job in jobs["jobs"]:
            sid = job["shard_id"]
            provider = str(job["provider"])
            state = poll_by_shard.get(sid)
            if not isinstance(state, Mapping) or not state.get("collection_ready"):
                raise ContractError(f"Shard is not ready for collection: {sid}")
            input_path = directory / job["input_path"]
            expected = {str(row.get("custom_id") or row.get("key")) for row in iter_jsonl(input_path)}
            model = models[job["model_key"]]
            raw_rows = _artifact_rows(directory, state)

            by_request: dict[str, tuple[str, Path, Mapping[str, Any]]] = {}
            for artifact_kind, artifact_path, raw in raw_rows:
                raw_request_id = str(raw.get("custom_id") or raw.get("key") or "")
                if not raw_request_id or raw_request_id not in expected or raw_request_id not in requests.index:
                    raise ContractError(f"Provider returned unknown request ID: {raw_request_id}")
                if raw_request_id in by_request:
                    prior_kind, prior_path, _ = by_request[raw_request_id]
                    append_event(
                        ledger,
                        event_type="PROVIDER_RESULT_PROVENANCE_ERROR",
                        shard_id=sid,
                        request_id=raw_request_id,
                        detail={
                            "reason": "DUPLICATE_CUSTOM_ID_ACROSS_PROVIDER_ARTIFACTS",
                            "prior_kind": prior_kind,
                            "prior_path": str(prior_path),
                            "current_kind": artifact_kind,
                            "current_path": str(artifact_path),
                        },
                    )
                    raise ContractError(
                        f"Provider custom_id appears more than once across output/error artifacts: {raw_request_id}"
                    )
                by_request[raw_request_id] = (artifact_kind, artifact_path, raw)

            for rid in sorted(by_request):
                artifact_kind, _, raw = by_request[rid]
                item = (
                    normalize_openai(raw, model["exact_model_id"], cancellation_origin=state.get("cancellation_origin"))
                    if provider == "openai"
                    else normalizers[provider](raw, model["exact_model_id"])
                )
                if str(item.get("request_id") or "") != rid:
                    raise ContractError("Provider normalizer changed the artifact request ID")
                if item.get("delivered") and str(item.get("returned_model") or "") != str(
                    model["expected_returned_model_version"]
                ):
                    raise ContractError(f"Returned model version drift for {job['model_key']}")
                attempt = ledger.execute(
                    "SELECT attempt_state,raw_visible_text_sha256,retry_class FROM attempts "
                    "WHERE request_id=? AND attempt_index=?",
                    (rid, int(attempt_index)),
                ).fetchone()
                logical = ledger.execute(
                    "SELECT selected_attempt_index FROM logical_requests WHERE request_id=?",
                    (rid,),
                ).fetchone()
                if attempt is None:
                    raise ContractError("Downloaded row has no reconciled local attempt")
                raw_text = item.get("raw_visible_text")
                raw_hash = hashlib.sha256(str(raw_text).encode()).hexdigest() if raw_text is not None else None
                if attempt["attempt_state"] != "SUBMITTED":
                    if item.get("delivered") and (
                        attempt["attempt_state"] == "NO_CONFIRMED_COMPLETION"
                        or logical["selected_attempt_index"] not in {None, int(attempt_index)}
                    ):
                        record_late_duplicate(
                            ledger,
                            rid,
                            json.dumps(
                                {
                                    "attempt_index": int(attempt_index),
                                    "provider": provider,
                                    "shard_id": sid,
                                    "artifact_kind": artifact_kind,
                                    "raw_visible_text_sha256": raw_hash,
                                },
                                sort_keys=True,
                            ),
                        )
                        late += 1
                    elif item.get("delivered") and attempt["attempt_state"] == "DELIVERED" and attempt[
                        "raw_visible_text_sha256"
                    ] == raw_hash:
                        pass
                    elif (
                        not item.get("delivered")
                        and attempt["attempt_state"] == "NO_CONFIRMED_COMPLETION"
                        and attempt["retry_class"] == item.get("retry_class")
                    ):
                        pass
                    else:
                        raise ContractError("Non-idempotent attempt re-collection")
                    if item.get("delivered"):
                        _ensure_selected_parse(ledger, requests, design, rid)
                    normalized.append({**item, "artifact_kind": artifact_kind})
                    continue

                if not item["delivered"]:
                    retry_class = str(item.get("retry_class") or "")
                    if not retry_class:
                        raise ContractError("Undelivered provider row lacks an explicit retry class")
                    completion = f"technical_{retry_class.lower()}"
                    explicit_provider_errors += 1
                elif item.get("provider_refusal"):
                    retry_class = None
                    completion = "provider_refusal"
                elif item.get("output_limit"):
                    retry_class = None
                    completion = "output_limit"
                elif not str(raw_text or "").strip():
                    retry_class = None
                    completion = "delivered_empty"
                else:
                    retry_class = None
                    completion = "usable_output"
                resolve_attempt(
                    ledger,
                    request_id=rid,
                    attempt_index=int(attempt_index),
                    delivered=bool(item["delivered"]),
                    raw_visible_text=raw_text,
                    completion_kind=completion,
                    retry_class=retry_class,
                    provider_request_id=item.get("provider_request_id"),
                    raw_provider_json=json.dumps(raw, ensure_ascii=False, sort_keys=True),
                    provider_artifact_kind=artifact_kind,
                )
                if item["delivered"]:
                    _ensure_selected_parse(ledger, requests, design, rid)
                normalized.append({**item, "artifact_kind": artifact_kind})

            unresolved_ids = sorted(expected - set(by_request))
            if state.get("terminal_failure"):
                # Always close terminal shard state. Only unresolved members take
                # the provider terminal failure retry class.
                mark_shard_terminal_failure(
                    ledger,
                    sid=sid,
                    provider_state=str(state.get("state") or ""),
                    unresolved_request_ids=unresolved_ids,
                )
            elif unresolved_ids:
                retry_class = _missing_retry_class(state)
                for rid in unresolved_ids:
                        attempt = ledger.execute(
                            "SELECT attempt_state FROM attempts WHERE request_id=? AND attempt_index=?",
                            (rid, int(attempt_index)),
                        ).fetchone()
                        if attempt and attempt["attempt_state"] == "SUBMITTED":
                            resolve_attempt(
                                ledger,
                                request_id=rid,
                                attempt_index=int(attempt_index),
                                delivered=False,
                                raw_visible_text=None,
                                completion_kind=f"missing_provider_result:{retry_class.lower()}",
                                retry_class=retry_class,
                            )
                            missing += 1
            if state.get("terminal_failure"):
                # mark_shard_terminal_failure owns the terminal shard state, including partial-output cancellation.
                pass
            else:
                with ledger:
                    ledger.execute(
                        "UPDATE shards SET shard_state='COLLECTED',resolved_utc=? WHERE shard_id=?",
                        (utc_now(), sid),
                    )
        if int(attempt_index) == 3:
            ledger.execute(
                "UPDATE logical_requests SET request_state='UNRESOLVED_INFRA' "
                "WHERE request_id IN (SELECT request_id FROM attempts WHERE attempt_index=3 "
                "AND attempt_state='NO_CONFIRMED_COMPLETION') AND selected_attempt_index IS NULL"
            )
        ledger.commit()
        retry_classes = tuple(sorted(RETRYABLE_TECHNICAL_CLASSES))
        placeholders = ",".join("?" for _ in retry_classes)
        no_completion = int(
            ledger.execute(
                "SELECT COUNT(*) FROM attempts a JOIN logical_requests r USING(request_id) "
                "WHERE a.attempt_index=? AND a.attempt_state='NO_CONFIRMED_COMPLETION' "
                f"AND a.retry_class IN ({placeholders}) AND r.selected_attempt_index IS NULL",
                (int(attempt_index), *retry_classes),
            ).fetchone()[0]
        )
        nonretryable = int(
            ledger.execute(
                "SELECT COUNT(*) FROM attempts a JOIN logical_requests r USING(request_id) "
                "WHERE a.attempt_index=? AND a.attempt_state='NO_CONFIRMED_COMPLETION' "
                f"AND (a.retry_class IS NULL OR a.retry_class NOT IN ({placeholders})) "
                "AND r.selected_attempt_index IS NULL",
                (int(attempt_index), *retry_classes),
            ).fetchone()[0]
        )
    finally:
        ledger.close()

    output = directory / "raw_responses" / f"wave{int(wave)}_attempt{int(attempt_index)}_normalized.jsonl"
    count, digest = write_jsonl(output, normalized)
    retry = (
        prepare_retry(project_root, release_hash, wave, int(attempt_index) + 1)
        if no_completion and int(attempt_index) < 3
        else None
    )
    status = (
        "BLOCKED_RUNTIME"
        if nonretryable
        else "INCOMPLETE_INFRA"
        if no_completion and int(attempt_index) == 3
        else "RETRY_REQUIRED"
        if no_completion
        else "COLLECTED"
    )
    result = {
        "status": status,
        "wave": int(wave),
        "attempt_index": int(attempt_index),
        "normalized_rows": count,
        "explicit_provider_error_rows": explicit_provider_errors,
        "truly_missing_result_rows": missing,
        "missing_provider_output_rows": missing,
        "late_duplicate_rows": late,
        "retryable_technical_rows": no_completion,
        "nonretryable_technical_rows": nonretryable,
        "normalized_sha256": digest,
        "retry": retry,
        "generation": generation_status(project_root, release_hash),
    }
    write_json(
        directory / "manifests" / f"wave{int(wave)}_attempt{int(attempt_index)}_collected.json",
        result,
    )
    return result
