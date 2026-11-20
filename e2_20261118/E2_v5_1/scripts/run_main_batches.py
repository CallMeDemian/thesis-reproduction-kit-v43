from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
E2 = WORKSPACE / "e2_20261118/E2_v5_1"
E1_PROVIDER_SCRIPT = E2.parents[0] / "source_inputs/github_release_ablation_20261102_readonly/contents/ablation_20261102/E1/scripts/e1_providers.py"
JOB_STATE = E2 / "logs_main/provider_jobs_main.json"
RESPONSES = E2 / "responses_main"
EXPECTED_MODELS = {"openai": "gpt-5.4-mini-2026-03-17", "google": "gemini-3.1-flash-lite"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def providers_module():
    spec = importlib.util.spec_from_file_location("e1_providers_main", E1_PROVIDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen E1 provider implementation: {E1_PROVIDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_state() -> dict:
    return json.loads(JOB_STATE.read_text(encoding="utf-8")) if JOB_STATE.exists() else {"jobs": {}, "events": []}


def write_state(state: dict) -> None:
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    JOB_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def job_name(provider: str, condition: str, chunk: int) -> str:
    return f"e2-v51-main-20261119-{provider}-{condition}-c{int(chunk):02d}"


def frozen_files() -> list[dict[str, str]]:
    path = E2 / "logs_main/pre_call_hashes_main.csv"
    if not path.is_file():
        raise RuntimeError("Pre-call hash manifest is missing")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    if len(rows) < 30:
        raise RuntimeError(f"Pre-call hash manifest is incomplete: {len(rows)} files")
    mismatches = []
    for row in rows:
        target = WORKSPACE / row["path"]
        if not target.is_file():
            mismatches.append({"path": row["path"], "error": "missing"})
        elif target.stat().st_size != int(row["bytes"]) or sha256(target) != row["sha256"]:
            mismatches.append({"path": row["path"], "error": "size_or_hash_mismatch"})
    if mismatches:
        raise RuntimeError("Frozen pre-call files changed: " + json.dumps(mismatches, ensure_ascii=False))
    return rows


def request_file_rows() -> list[dict[str, str]]:
    path = E2 / "manifests_main/new_request_files.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    if len(rows) != 24:
        raise RuntimeError(f"Expected 24 provider chunk files; found {len(rows)}")
    return rows


def submit() -> None:
    frozen = frozen_files()
    rows = request_file_rows()
    providers = providers_module()
    providers.load_keys()
    state = load_state()
    state.setdefault("jobs", {})
    state.setdefault("events", [])
    all_names = [job_name(r["provider"], r["condition"], int(r["chunk"])) for r in rows]
    provider_by_name = {job_name(r["provider"], r["condition"], int(r["chunk"])): r["provider"] for r in rows}
    remote = {name: providers.find_jobs(provider_by_name[name], name) for name in all_names}
    for name in all_names:
        matches = remote[name]
        if len(matches) > 1:
            raise RuntimeError(f"Multiple remote batches use reserved name {name}")
        local = state["jobs"].get(name)
        if matches and local and local.get("batch_job_id") and local["batch_job_id"] != matches[0]["batch_job_id"]:
            raise RuntimeError(f"Local/remote job ID mismatch for {name}")
    state["request_freeze"] = {"hashed_files": len(frozen), "unique_requests": 5150,
                               "estimated_cost_usd": 69.84319128125,
                               "approx_one_sided_95pct_upper_usd": 70.4904741045096,
                               "output_cap_scenario_usd_not_hard_cap": 260.88812815625}
    state["submit_started_at_utc"] = state.get("submit_started_at_utc") or datetime.now(timezone.utc).isoformat()
    write_state(state)
    for row in rows:
        provider = row["provider"]
        condition = row["condition"]
        chunk = int(row["chunk"])
        name = job_name(provider, condition, chunk)
        path = WORKSPACE / row["file"]
        record = state["jobs"].get(name, {})
        record.update({"reserved_name": name, "provider": provider, "condition": condition,
                       "chunk": chunk, "model_id": EXPECTED_MODELS[provider],
                       "request_count": int(row["request_count"]), "request_file": row["file"],
                       "request_file_sha256": row["sha256"], "request_file_bytes": int(row["bytes"])})
        if record.get("batch_job_id"):
            continue
        if remote[name]:
            existing = remote[name][0]
            record.update(existing)
            record["adopted_remote_job"] = True
            record["adopted_at_utc"] = datetime.now(timezone.utc).isoformat()
            state["jobs"][name] = record
            state["events"].append({"event": "adopted_existing_job", "reserved_name": name,
                                    "batch_job_id": record.get("batch_job_id")})
            write_state(state)
            continue
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError(f"Request file changed after freeze: {path}")
        record["submitted_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            if provider == "openai":
                result = providers.submit_openai(path, name)
            else:
                result = providers.submit_google(path, EXPECTED_MODELS[provider], name)
            record.update(result)
            state["jobs"][name] = record
            state["events"].append({"event": "batch_submitted", "reserved_name": name,
                                    "provider": provider, "condition": condition, "chunk": chunk,
                                    "request_count": int(row["request_count"]), "batch_job_id": record.get("batch_job_id"),
                                    "state": record.get("state")})
            write_state(state)
        except Exception as exc:
            record["submit_error"] = f"{type(exc).__name__}: {exc}"
            state["jobs"][name] = record
            state["events"].append({"event": "batch_submit_error", "reserved_name": name,
                                    "provider": provider, "error_type": type(exc).__name__, "error": str(exc)})
            write_state(state)
            raise
    print(json.dumps({"submitted_or_adopted_jobs": len(state["jobs"]), "total_jobs": len(rows),
                      "newly_submitted": sum(x.get("event") == "batch_submitted" for x in state["events"]),
                      "status_file": str(JOB_STATE.relative_to(WORKSPACE))}, ensure_ascii=False, indent=2))


def poll_download() -> None:
    providers = providers_module()
    providers.load_keys()
    state = load_state()
    if len(state.get("jobs", {})) != 24:
        raise RuntimeError(f"Expected all 24 batch jobs to be recorded; found {len(state.get('jobs', {}))}")
    RESPONSES.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, job in state["jobs"].items():
        provider = job["provider"]
        job_id = job.get("batch_job_id")
        if not job_id:
            summaries.append({"reserved_name": name, "state": "NO_JOB_ID"})
            continue
        info = providers.poll_openai(job_id) if provider == "openai" else providers.poll_google(job_id)
        job["last_poll"] = {"at_utc": datetime.now(timezone.utc).isoformat(), **info}
        artifacts = []
        if info.get("terminal"):
            success_states = {"completed", "succeeded", "job_state_succeeded"}
            if str(info.get("state", "")).lower() in success_states:
                stem = f"main_{provider}_{job['condition']}_chunk{int(job['chunk']):02d}"
                existing_paths = [a.get("path") for a in job.get("downloaded_artifacts", [])]
                if existing_paths and all((RESPONSES / p).is_file() for p in existing_paths):
                    artifacts = job["downloaded_artifacts"]
                else:
                    downloaded = providers.download_openai(info, RESPONSES, stem) if provider == "openai" else providers.download_google(info, RESPONSES, stem)
                    for artifact in downloaded:
                        artifact["sha256"] = sha256(RESPONSES / artifact["path"])
                        artifact["bytes"] = (RESPONSES / artifact["path"]).stat().st_size
                    artifacts = downloaded
                    job["downloaded_artifacts"] = artifacts
                if not artifacts:
                    job["terminal_failure"] = {"reason": "No output/error artifact returned", **info}
            else:
                job["terminal_failure"] = info
        summaries.append({"reserved_name": name, "batch_job_id": job_id,
                          "state": info.get("state"), "terminal": info.get("terminal"),
                          "counts": info.get("counts"), "artifacts": artifacts,
                          "failure": job.get("terminal_failure")})
        write_state(state)
    write_state(state)
    print(json.dumps({"jobs": summaries, "completed": sum(str(x.get("state", "")).lower() in {"completed", "succeeded", "job_state_succeeded"} for x in summaries),
                      "total": len(summaries)}, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit", "poll-download"))
    args = parser.parse_args()
    if args.action == "submit":
        submit()
    else:
        poll_download()


if __name__ == "__main__":
    main()
