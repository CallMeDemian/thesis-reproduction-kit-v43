"""Crash-safe, quota-aware Gemini Batch submission for the frozen Plan-3 run.

This module changes only transport scheduling.  Shards, request IDs, prompts,
models, and the frozen release remain untouched.
"""
from __future__ import annotations

import ast
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .common import ContractError, file_hash, load_json, utc_now, write_json
from .executor import load_release
from .ledger import (
    append_event,
    begin_shard_submission,
    bind_remote_job,
    connect,
)
from .live_batch import _find_remote_jobs, _google_submit, assert_live_authorized, recover_submitting_shard
from .retry_policy import normalize_provider_job_state

SCHEMA_VERSION = "gemini_quota_scheduler_v1"
TERMINAL = {"SUCCEEDED", "FAILED", "EXPIRED", "CANCELLED"}
DEFAULT_WINDOW = 3
MAX_WINDOW = 4


def _analysis(root: Path) -> Path:
    path = root / "analysis" / "llm_live_run"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(root: Path) -> Path:
    return _analysis(root) / "gemini_quota_scheduler_state.json"


def _load_state(root: Path, release_hash: str, wave: int) -> dict[str, Any]:
    path = _state_path(root)
    if path.exists():
        state = load_json(path)
        if state.get("protocol_release_hash") != release_hash or int(state.get("wave", -1)) != int(wave):
            raise ContractError("Quota scheduler state belongs to a different frozen release or wave")
        return state
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_release_hash": release_hash,
        "wave": int(wave),
        "submission_window_target": DEFAULT_WINDOW,
        "maximum_window_target": MAX_WINDOW,
        "quota_backpressure": False,
        "quota_type": "UNKNOWN_RESOURCE_EXHAUSTED",
        "consecutive_429_count": 0,
        "current_backoff_seconds": 0,
        "last_429_timestamp": None,
        "next_submission_eligible_at": None,
        "last_successful_submission": None,
        "history": [],
    }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_error(exc: BaseException) -> dict[str, Any]:
    """Extract only provider-supplied diagnostic fields; never infer a quota metric."""
    raw = str(exc)
    payload: dict[str, Any] = {}
    marker = raw.find("{'error':")
    if marker >= 0:
        try:
            parsed = ast.literal_eval(raw[marker:])
            if isinstance(parsed, dict):
                payload = dict(parsed.get("error") or parsed)
        except (SyntaxError, ValueError):
            pass
    code = payload.get("code", getattr(exc, "code", None))
    status = payload.get("status", getattr(exc, "status", None))
    message = payload.get("message", raw)
    details = payload.get("details") if isinstance(payload.get("details"), list) else []
    metric = quota_id = quota_value = retry_delay = request_id = None
    for item in details:
        if not isinstance(item, Mapping):
            continue
        metric = metric or item.get("quotaMetric") or item.get("quota_metric")
        quota_id = quota_id or item.get("quotaId") or item.get("quota_id")
        quota_value = quota_value or item.get("quotaValue") or item.get("quota_value")
        retry_delay = retry_delay or item.get("retryDelay") or item.get("retry_delay")
        request_id = request_id or item.get("requestId") or item.get("request_id")
    text = " ".join(str(value or "") for value in (metric, quota_id, message)).lower()
    if "enqueued" in text and "token" in text:
        quota_type = "BATCH_ENQUEUED_TOKENS"
    elif "spend" in text or "billing" in text and "rate" in text:
        quota_type = "SPEND_RATE_LIMIT"
    elif "concurrent" in text and "batch" in text:
        quota_type = "CONCURRENT_BATCH_LIMIT"
    elif "request" in text and "rate" in text:
        quota_type = "REQUEST_RATE_LIMIT"
    else:
        quota_type = "UNKNOWN_RESOURCE_EXHAUSTED"
    return {
        "error_code": code,
        "status": status,
        "message": str(message),
        "quota_metric": metric,
        "quota_id": quota_id,
        "quota_value": quota_value,
        "retry_delay": retry_delay,
        "provider_request_id": request_id,
        "provider_details": details,
        "quota_type": quota_type,
    }


def _is_429(detail: Mapping[str, Any]) -> bool:
    return str(detail.get("error_code")) == "429" or str(detail.get("status", "")).upper() == "RESOURCE_EXHAUSTED"


def _backoff_seconds(consecutive: int, retry_delay: Any) -> int:
    if retry_delay:
        text = str(retry_delay).rstrip("s")
        try:
            return min(3600, max(600, int(float(text))))
        except ValueError:
            pass
    return min(3600, 600 * (2 ** max(0, consecutive - 1)))


def _write_diagnostic(root: Path, *, release_hash: str, detail: Mapping[str, Any], shard_id: str | None, timestamp: str) -> None:
    path = _analysis(root) / "gemini_quota_diagnostic.json"
    prior: dict[str, Any] = load_json(path) if path.exists() else {"schema_version": SCHEMA_VERSION, "events": []}
    event = {"timestamp": timestamp, "protocol_release_hash": release_hash, "shard_id": shard_id, **dict(detail)}
    history = list(prior.get("events") or [])
    if not history or history[-1] != event:
        history.append(event)
    prior.update({"latest": event, "events": history, "scientific_interpretation": "control_plane_quota_backpressure_no_inference"})
    write_json(path, prior)


def _auto_pause(root: Path) -> bool:
    """Only clear the old scheduler's exact auto-sentinel; never override user intent."""
    path = root / "PAUSE_LLM_RUN"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    expected = "Wave 2 OpenAI submission process exited. Google quota remains blocked; verify ledger before any further operation."
    if text != expected:
        raise ContractError("LLM_RUN_PAUSED_SAFE: user or unknown PAUSE_LLM_RUN sentinel exists; no Gemini submission permitted")
    path.unlink()
    return True


def _remote_state(job_id: str) -> str:
    from google import genai
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    job = client.batches.get(name=job_id)
    return normalize_provider_job_state("google", str(getattr(getattr(job, "state", None), "name", getattr(job, "state", ""))))


def _counts(ledger: sqlite3.Connection, wave: int) -> dict[str, int]:
    rows = ledger.execute(
        "SELECT shard_state, COUNT(*) n, COALESCE(SUM(request_count),0) requests FROM shards "
        "WHERE wave=? AND attempt_index=1 AND provider='google' GROUP BY shard_state", (int(wave),)
    ).fetchall()
    result = {"prepared_shards": 0, "deferred_quota_shards": 0, "submitted_shards_total": 0, "completed_remote_shards": 0, "failed_remote_shards": 0, "unsent_shards": 0, "submitted_requests": 0, "unsent_requests": 0}
    for row in rows:
        state, number, requests = str(row["shard_state"]), int(row["n"]), int(row["requests"])
        if state == "PREPARED": result["prepared_shards"] += number; result["unsent_shards"] += number; result["unsent_requests"] += requests
        elif state in {"SUBMITTED", "OUTPUT_DOWNLOADED", "COLLECTED"}:
            result["submitted_shards_total"] += number; result["submitted_requests"] += requests
            if state == "COLLECTED": result["completed_remote_shards"] += number
        elif state == "FAILED_TERMINAL": result["failed_remote_shards"] += number
    return result


def _duplicate_audit(ledger: sqlite3.Connection, release_hash: str, wave: int) -> dict[str, Any]:
    logical = ledger.execute("SELECT COUNT(*) n FROM (SELECT request_id FROM shard_requests GROUP BY request_id HAVING COUNT(*)>1)").fetchone()["n"]
    bindings = ledger.execute(
        "SELECT COUNT(*) n FROM (SELECT provider_job_id FROM shards WHERE provider='google' AND provider_job_id IS NOT NULL GROUP BY provider_job_id HAVING COUNT(*)>1)"
    ).fetchone()["n"]
    uncertain = ledger.execute("SELECT COUNT(*) n FROM shards WHERE provider='google' AND shard_state='SUBMITTING' AND provider_job_id IS NULL").fetchone()["n"]
    return {"protocol_release_hash": release_hash, "wave": int(wave), "duplicate_logical_request_ids": int(logical), "duplicate_gemini_job_bindings": int(bindings), "unbound_submitting_gemini_shards": int(uncertain), "exactly_once_pass": int(logical) == 0 and int(bindings) == 0 and int(uncertain) == 0, "checked_utc": utc_now()}


def _append_job_manifest(directory: Path, release_hash: str, wave: int, prepared: Mapping[str, Any], prepared_item: Mapping[str, Any], shard: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    path = directory / "manifests" / f"wave{int(wave)}_attempt1_jobs.json"
    manifest = load_json(path) if path.exists() else {"schema_version": "llm_v43_live_jobs_v3", "protocol_release_hash": release_hash, "wave": int(wave), "attempt_index": 1, "jobs": []}
    jobs = list(manifest.get("jobs") or [])
    matches = [item for item in jobs if item.get("shard_id") == prepared_item["shard_id"]]
    if len(matches) > 1 or (matches and matches[0].get("batch_job_id") != job["batch_job_id"]):
        raise ContractError("Job manifest shard binding ambiguity")
    if not matches:
        jobs.append({**dict(job), "shard_id": prepared_item["shard_id"], "model_key": prepared_item["model_key"], "replicate": prepared_item.get("replicate"), "ordinal": int(shard["ordinal"]), "input_path": prepared_item["path"], "input_sha256": prepared_item["sha256"], "requests": int(prepared_item["requests"]), "submitted_utc": utc_now(), "scheduler_managed": True})
    manifest["jobs"] = jobs
    manifest["status"] = "PARTIALLY_SUBMITTED"
    manifest["remaining_shard_count"] = len([x for x in prepared["provider_batches"] if x["shard_id"] not in {j["shard_id"] for j in jobs}])
    write_json(path, manifest)


def scheduler_tick(project_root: Path, release_hash: str, *, wave: int = 2, execute_api: bool = False, window_target: int = DEFAULT_WINDOW) -> dict[str, Any]:
    """Reconcile first, then create at most one deterministic next Gemini shard."""
    root = Path(project_root).resolve()
    if not 1 <= int(window_target) <= MAX_WINDOW:
        raise ContractError(f"Gemini scheduler window must be 1..{MAX_WINDOW}")
    directory, release = load_release(root, release_hash)
    if execute_api:
        assert_live_authorized(approved=True, release=release, attempt_index=1)
    state = _load_state(root, release_hash, wave)
    state["submission_window_target"] = int(window_target)
    state["auto_pause_sentinel_cleared"] = _auto_pause(root) if execute_api else False
    prepared = load_json(directory / "manifests" / f"wave{int(wave)}_prepared.json")
    if prepared.get("status") != "PREPARED":
        raise ContractError("Frozen Wave 2 prepared manifest is unavailable")
    ledger = connect(directory / "generation_ledger.sqlite")
    try:
        # Resolve every interrupted submit before considering a new create.
        uncertain = ledger.execute("SELECT shard_id FROM shards WHERE wave=? AND provider='google' AND shard_state='SUBMITTING' AND provider_job_id IS NULL", (int(wave),)).fetchall()
        for row in uncertain:
            recovered = recover_submitting_shard(ledger, shard_id=str(row["shard_id"]), provider="google")
            if recovered["status"] == "CONFIRMED_ABSENT_RESET":
                append_event(ledger, event_type="GEMINI_SUBMISSION_DEFERRED_QUOTA", shard_id=str(row["shard_id"]), detail={"recovery": "confirmed_absent_before_scheduler", "quota_type": state.get("quota_type"), "ledger_state": "PREPARED"})
        bound = ledger.execute("SELECT shard_id,provider_job_id,shard_state FROM shards WHERE wave=? AND provider='google' AND provider_job_id IS NOT NULL", (int(wave),)).fetchall()
        active = 0; remote_states: dict[str, str] = {}
        if execute_api:
            for row in bound:
                remote = _remote_state(str(row["provider_job_id"]))
                remote_states[str(row["shard_id"])] = remote
                if remote not in TERMINAL: active += 1
        else:
            active = sum(1 for row in bound if str(row["shard_state"]) not in {"COLLECTED", "FAILED_TERMINAL"})
        now = _now()
        eligible_at = state.get("next_submission_eligible_at")
        backoff_elapsed = not eligible_at or now >= datetime.fromisoformat(str(eligible_at))
        submitted = None
        if execute_api and active < int(window_target) and backoff_elapsed:
            candidates = [item for item in prepared["provider_batches"] if item["provider"] == "google"]
            next_item = None
            for item in candidates:
                row = ledger.execute("SELECT * FROM shards WHERE shard_id=?", (item["shard_id"],)).fetchone()
                if row is not None and row["provider_job_id"] is None and row["shard_state"] == "PREPARED":
                    next_item, next_shard = item, row; break
            if next_item is not None:
                source = directory / str(next_item["path"])
                if file_hash(source) != next_item["sha256"]:
                    raise ContractError("Frozen Gemini shard bytes changed before resumed submission")
                begin_shard_submission(ledger, shard_id=str(next_item["shard_id"]), correlation_key=str(next_item["shard_id"]))
                exact_model = str(((release.get("runtime_binding") or {}).get("models") or {}).get(str(next_item["model_key"]), {}).get("exact_model_id") or "")
                if not exact_model:
                    raise ContractError("Frozen release has no exact Gemini model identifier")
                try:
                    job = _google_submit(source, exact_model, f"llm-v43-{next_item['shard_id']}")
                except Exception as exc:
                    detail = _parse_error(exc)
                    if not _is_429(detail):
                        raise
                    recovery = recover_submitting_shard(ledger, shard_id=str(next_item["shard_id"]), provider="google")
                    if recovery["status"] != "CONFIRMED_ABSENT_RESET":
                        raise ContractError("429 create outcome was remotely recoverable; refusing duplicate create")
                    state["consecutive_429_count"] = int(state.get("consecutive_429_count", 0)) + 1
                    delay = _backoff_seconds(int(state["consecutive_429_count"]), detail.get("retry_delay"))
                    state.update({"quota_backpressure": True, "quota_type": detail["quota_type"], "current_backoff_seconds": delay, "last_429_timestamp": utc_now(), "next_submission_eligible_at": (now + timedelta(seconds=delay)).isoformat()})
                    append_event(ledger, event_type="GEMINI_SUBMISSION_DEFERRED_QUOTA", shard_id=str(next_item["shard_id"]), detail={**detail, "backoff_seconds": delay, "scientific_failure": False, "ledger_state": "PREPARED"})
                    _write_diagnostic(root, release_hash=release_hash, detail=detail, shard_id=str(next_item["shard_id"]), timestamp=str(state["last_429_timestamp"]))
                else:
                    bind_remote_job(ledger, shard_id=str(next_item["shard_id"]), provider_job_id=str(job["batch_job_id"]))
                    _append_job_manifest(directory, release_hash, wave, prepared, next_item, next_shard, job)
                    append_event(ledger, event_type="GEMINI_QUOTA_SCHEDULER_SUBMITTED", shard_id=str(next_item["shard_id"]), detail={"batch_job_id": job["batch_job_id"], "window_target": int(window_target)})
                    state.update({"quota_backpressure": False, "consecutive_429_count": 0, "current_backoff_seconds": 0, "next_submission_eligible_at": None, "last_successful_submission": utc_now()})
                    submitted = {"shard_id": next_item["shard_id"], "batch_job_id": job["batch_job_id"], "requests": next_item["requests"]}
                    # The just-created job is active even though the pre-submit remote
                    # reconciliation naturally did not see it.
                    active += 1
                    remote_states[str(next_item["shard_id"])] = normalize_provider_job_state("google", str(job.get("state", "")))
        counts = _counts(ledger, wave)
        audit = _duplicate_audit(ledger, release_hash, wave)
        if not audit["exactly_once_pass"]:
            raise ContractError("Exactly-once duplicate audit failed; scheduler stopped")
        state.update({"last_tick_utc": utc_now(), "active_remote_shards": active, "remote_states": remote_states, **counts, "resume_safe_checkpoint": {"ledger": str(directory / "generation_ledger.sqlite"), "prepared_manifest": str(directory / "manifests" / f"wave{int(wave)}_prepared.json")}})
        state.setdefault("history", []).append({"timestamp": state["last_tick_utc"], "active": active, "unsent": counts["unsent_shards"], "submitted": submitted, "quota_backpressure": state.get("quota_backpressure", False)})
        state["history"] = state["history"][-100:]
        write_json(_state_path(root), state)
        write_json(_analysis(root) / "duplicate_submission_audit.json", audit)
        heartbeat = {"timestamp": state["last_tick_utc"], "phase": "gemini_quota_scheduler", "gemini": {key: state.get(key) for key in ("submitted_shards_total", "unsent_shards", "active_remote_shards", "completed_remote_shards", "failed_remote_shards", "quota_backpressure", "quota_type", "last_429_timestamp", "consecutive_429_count", "current_backoff_seconds", "next_submission_eligible_at", "submission_window_target")}, "openai": {"status": "independent_existing_jobs_preserved"}, "duplicate_audit": audit, "status": "GEMINI_QUOTA_BACKPRESSURE_RECOVERING" if state.get("quota_backpressure") else "GEMINI_RESUMED" if submitted else "GEMINI_WAITING"}
        write_json(_analysis(root) / "heartbeat_latest.json", heartbeat)
        with (_analysis(root) / "heartbeat_history.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(heartbeat, ensure_ascii=False, sort_keys=True) + "\n")
        return {"status": heartbeat["status"], "submitted": submitted, "state": state, "duplicate_audit": audit}
    finally:
        ledger.close()
