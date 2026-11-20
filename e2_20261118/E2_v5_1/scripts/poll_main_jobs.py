from __future__ import annotations

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def providers_module():
    spec = importlib.util.spec_from_file_location("e1_providers_poll_main", E1_PROVIDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen E1 provider implementation: {E1_PROVIDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    providers = providers_module()
    providers.load_keys()
    state = json.loads(JOB_STATE.read_text(encoding="utf-8"))
    RESPONSES.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, job in list(state.get("jobs", {}).items()):
        provider = job.get("provider")
        job_id = job.get("batch_job_id")
        if not job_id:
            summary.append({"name": name, "state": "NOT_SUBMITTED", "submit_error": job.get("submit_error")})
            continue
        info = providers.poll_openai(job_id) if provider == "openai" else providers.poll_google(job_id)
        job["last_poll"] = {"at_utc": datetime.now(timezone.utc).isoformat(), **info}
        if info.get("terminal"):
            successful = str(info.get("state", "")).lower() in {"completed", "succeeded", "job_state_succeeded"}
            if successful:
                stem = f"main_{provider}_{job['condition']}_chunk{int(job['chunk']):02d}"
                existing = job.get("downloaded_artifacts", [])
                if existing and all((RESPONSES / a["path"]).is_file() for a in existing):
                    artifacts = existing
                else:
                    downloaded = providers.download_openai(info, RESPONSES, stem) if provider == "openai" else providers.download_google(info, RESPONSES, stem)
                    artifacts = []
                    for artifact in downloaded:
                        target = RESPONSES / artifact["path"]
                        artifact["sha256"] = sha256(target)
                        artifact["bytes"] = target.stat().st_size
                        artifacts.append(artifact)
                    job["downloaded_artifacts"] = artifacts
                if not artifacts:
                    job["terminal_failure"] = {"reason": "No provider output/error artifact was returned", **info}
            else:
                job["terminal_failure"] = info
        summary.append({"name": name, "provider": provider, "job_id": job_id,
                        "state": info.get("state"), "terminal": info.get("terminal"),
                        "counts": info.get("counts"), "artifacts": job.get("downloaded_artifacts", []),
                        "terminal_failure": job.get("terminal_failure")})
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        JOB_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state["last_poll_summary"] = {"at_utc": datetime.now(timezone.utc).isoformat(), "jobs": summary}
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    JOB_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"polled": len(summary), "jobs": summary}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
