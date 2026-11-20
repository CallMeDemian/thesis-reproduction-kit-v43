from __future__ import annotations

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def get_provider_module():
    spec = importlib.util.spec_from_file_location("e1_providers_submit_missing", E1_PROVIDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen E1 provider implementation: {E1_PROVIDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch_name(row: dict) -> str:
    return f"e2-v51-main-20261119-{row['provider']}-{row['condition']}-c{int(row['chunk']):02d}"


def main() -> None:
    providers = get_provider_module()
    providers.load_keys()
    rows = list(csv.DictReader((E2 / "manifests_main/new_request_files.csv").open(encoding="utf-8-sig", newline="")))
    state = json.loads(JOB_STATE.read_text(encoding="utf-8"))
    state.setdefault("jobs", {})
    state.setdefault("events", [])
    outcomes = []
    for row in rows:
        name = batch_name(row)
        local = state["jobs"].get(name, {})
        if local.get("batch_job_id"):
            continue
        path = WORKSPACE / row["file"]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError(f"Frozen request file changed: {row['file']}")
        matches = providers.find_jobs(row["provider"], name)
        if len(matches) > 1:
            raise RuntimeError(f"Multiple provider jobs exist under {name}")
        if matches:
            existing = matches[0]
            local.update(existing)
            local["reserved_name"] = name
            local["provider"] = row["provider"]
            local["condition"] = row["condition"]
            local["chunk"] = int(row["chunk"])
            local["request_file"] = row["file"]
            local["request_file_sha256"] = row["sha256"]
            local["request_count"] = int(row["request_count"])
            local["adopted_remote_job"] = True
            state["jobs"][name] = local
            state["events"].append({"event": "adopted_remote_job_after_submit_retry", "reserved_name": name,
                                    "batch_job_id": local.get("batch_job_id")})
            outcomes.append({"name": name, "action": "adopted", "batch_job_id": local.get("batch_job_id")})
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            JOB_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            continue
        local.update({"reserved_name": name, "provider": row["provider"], "condition": row["condition"],
                      "chunk": int(row["chunk"]), "model_id": "gpt-5.4-mini-2026-03-17" if row["provider"] == "openai" else "gemini-3.1-flash-lite",
                      "request_file": row["file"], "request_file_sha256": row["sha256"],
                      "request_file_bytes": int(row["bytes"]), "request_count": int(row["request_count"]),
                      "submitted_at_utc": datetime.now(timezone.utc).isoformat()})
        try:
            if row["provider"] == "openai":
                result = providers.submit_openai(path, name)
            else:
                result = providers.submit_google(path, "gemini-3.1-flash-lite", name)
            local.update(result)
            state["jobs"][name] = local
            state["events"].append({"event": "batch_submitted_after_quota_wait", "reserved_name": name,
                                    "batch_job_id": local.get("batch_job_id"), "state": local.get("state")})
            outcomes.append({"name": name, "action": "submitted", "batch_job_id": local.get("batch_job_id"),
                             "state": local.get("state")})
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            JOB_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            local["submit_error"] = f"{type(exc).__name__}: {exc}"
            local["submit_error_at_utc"] = datetime.now(timezone.utc).isoformat()
            state["jobs"][name] = local
            state["events"].append({"event": "submit_still_blocked", "reserved_name": name,
                                    "provider": row["provider"], "error_type": type(exc).__name__, "error": str(exc)})
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            JOB_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            outcomes.append({"name": name, "action": "blocked", "error": str(exc)})
            # Do not repeatedly upload files against the same provider quota in one pass.
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                break
            raise
    print(json.dumps({"outcomes": outcomes, "recorded_job_count": len(state["jobs"]),
                      "job_names_total": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
