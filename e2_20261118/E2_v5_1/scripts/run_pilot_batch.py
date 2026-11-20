from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
WORKSPACE = E2.parents[1]
E1_PROVIDER_SCRIPT = E2.parents[0] / "source_inputs/github_release_ablation_20261102_readonly/contents/ablation_20261102/E1/scripts/e1_providers.py"
EXPECTED_MODELS = {"openai": "gpt-5.4-mini-2026-03-17", "google": "gemini-3.1-flash-lite"}
JOB_NAMES = {"openai": "e2-v5.1-pilot-20261118-openai-attempt1", "google": "e2-v5.1-pilot-20261118-google-attempt1"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provider_module():
    spec = importlib.util.spec_from_file_location("e1_providers", E1_PROVIDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen E1 provider implementation: {E1_PROVIDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_frozen_inputs() -> dict:
    rows = list(csv.DictReader((E2 / "pre_call_hashes.csv").open(encoding="utf-8-sig", newline="")))
    if len(rows) < 20:
        raise RuntimeError(f"Pre-call freeze manifest is incomplete: {len(rows)} files")
    mismatches = []
    for row in rows:
        path = WORKSPACE / row["path"]
        if not path.is_file():
            mismatches.append({"path": row["path"], "error": "missing"})
        elif path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            mismatches.append({"path": row["path"], "error": "hash_or_size_mismatch"})
    if mismatches:
        raise RuntimeError("Frozen pre-call inputs changed: " + json.dumps(mismatches, ensure_ascii=False))

    report = (E2 / "preflight_report.md").read_text(encoding="utf-8")
    if "# E2 v5.1 Pilot Preflight" not in report or "Logical requests: 1000" not in report:
        raise RuntimeError("The final 1,000-request preflight report is absent or failed")
    model_meta = json.loads((E2 / "model_preflight.json").read_text(encoding="utf-8"))
    if not all(item["available"] for item in model_meta["models"].values()):
        raise RuntimeError("Exact-model metadata check did not pass")
    if any(model_meta["existing_jobs_with_reserved_names"].values()):
        raise RuntimeError("A reserved provider batch name already exists")

    expected_counts = {}
    all_ids: set[str] = set()
    for provider in ("openai", "google"):
        path = E2 / "requests" / f"{provider}_pilot_mixed.jsonl"
        ids: list[str] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                rid = item["custom_id"] if provider == "openai" else item["key"]
                if rid in all_ids:
                    raise RuntimeError(f"Duplicate request ID: {rid}")
                all_ids.add(rid)
                ids.append(rid)
        if len(ids) != 500:
            raise RuntimeError(f"Expected 500 {provider} requests; found {len(ids)}")
        expected_counts[provider] = len(ids)
    if len(all_ids) != 1000:
        raise RuntimeError(f"Expected 1,000 globally unique IDs; found {len(all_ids)}")
    return {"frozen_files": len(rows), "provider_requests": expected_counts, "unique_ids": len(all_ids)}


def read_job_manifest() -> dict:
    path = E2 / "logs/provider_jobs.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"jobs": {}, "events": []}


def write_job_manifest(data: dict) -> None:
    data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = E2 / "logs/provider_jobs.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def submit() -> None:
    frozen = verify_frozen_inputs()
    providers = provider_module()
    providers.load_keys()
    state = read_job_manifest()
    existing = {}
    for provider, name in JOB_NAMES.items():
        matches = providers.find_jobs(provider, name)
        existing[provider] = matches
    if any(existing.values()):
        raise RuntimeError("Reserved provider batch name is already present; refusing duplicate submission")
    if state.get("jobs"):
        raise RuntimeError("Local provider job manifest is not empty; refusing duplicate submission")

    state["request_freeze"] = frozen
    state["submit_started_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_job_manifest(state)
    for provider in ("openai", "google"):
        req_path = E2 / "requests" / f"{provider}_pilot_mixed.jsonl"
        record = {
            "provider": provider,
            "reserved_name": JOB_NAMES[provider],
            "model_id": EXPECTED_MODELS[provider],
            "request_count": 500,
            "request_file": str(req_path.relative_to(WORKSPACE)),
            "request_file_sha256": sha256(req_path),
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            if provider == "openai":
                result = providers.submit_openai(req_path, JOB_NAMES[provider])
            else:
                result = providers.submit_google(req_path, EXPECTED_MODELS[provider], JOB_NAMES[provider])
            record.update(result)
            state.setdefault("jobs", {})[provider] = record
            state.setdefault("events", []).append({"event": "batch_submitted", **record})
            write_job_manifest(state)
        except Exception as exc:
            record["submit_error"] = f"{type(exc).__name__}: {exc}"
            state.setdefault("jobs", {})[provider] = record
            state.setdefault("events", []).append({"event": "batch_submit_error", "provider": provider,
                                                   "reserved_name": JOB_NAMES[provider],
                                                   "error_type": type(exc).__name__, "error": str(exc)})
            write_job_manifest(state)
            raise
    print(json.dumps({"submitted": state["jobs"], "frozen_inputs": frozen}, ensure_ascii=False, indent=2))


def poll_and_download() -> None:
    providers = provider_module()
    providers.load_keys()
    state = read_job_manifest()
    if set(state.get("jobs", {})) != {"openai", "google"}:
        raise RuntimeError("Both provider jobs must exist in local manifest before polling")
    for provider in ("openai", "google"):
        job = state["jobs"][provider]
        job_id = job.get("batch_job_id")
        if not job_id:
            raise RuntimeError(f"Missing provider job id for {provider}")
        info = providers.poll_openai(job_id) if provider == "openai" else providers.poll_google(job_id)
        job["last_poll"] = {"at_utc": datetime.now(timezone.utc).isoformat(), **info}
        if info["terminal"]:
            if info.get("state", "").lower() not in {"completed", "succeeded", "job_state_succeeded"}:
                job["terminal_failure"] = info
            else:
                stem = f"pilot_{provider}_attempt1"
                artifacts = providers.download_openai(info, E2 / "responses", stem) if provider == "openai" else providers.download_google(info, E2 / "responses", stem)
                for artifact in artifacts:
                    artifact["sha256"] = sha256(E2 / "responses" / artifact["path"])
                    artifact["bytes"] = (E2 / "responses" / artifact["path"]).stat().st_size
                job["downloaded_artifacts"] = artifacts
        write_job_manifest(state)
    print(json.dumps({"jobs": {k: {"batch_job_id": v.get("batch_job_id"), "last_poll": v.get("last_poll"),
                                   "downloaded_artifacts": v.get("downloaded_artifacts"),
                                   "terminal_failure": v.get("terminal_failure")}
                             for k, v in state["jobs"].items()}}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit", "poll-download"))
    parser.add_argument("--submit-authorized-pilot", action="store_true",
                        help="required acknowledgement for the explicitly authorized 1,000-request pilot")
    args = parser.parse_args()
    if args.action == "submit":
        if not args.submit_authorized_pilot:
            raise SystemExit("Refusing submission without --submit-authorized-pilot")
        submit()
    else:
        poll_and_download()


if __name__ == "__main__":
    main()
