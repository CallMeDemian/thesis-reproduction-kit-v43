"""Submit the eight E1 shards (idempotent).

For every shard: verify the shard SHA-256 against the build report, list existing provider jobs with
the display name ``e1-<provider>-shard<k>-attempt<n>`` and submit only if none exists. Gemini 429
(RESOURCE_EXHAUSTED) is waited out and the same shard resubmitted. Job ids go to
``E1/runs/submissions.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_providers as prov  # noqa: E402
from e1_process import EXPECTED_MODEL  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-root", type=Path, required=True)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--max-wait-minutes", type=int, default=90, help="total wait budget per shard for 429")
    args = ap.parse_args()
    e1 = args.kit_root.resolve() / "ablation_20261102/E1"
    runs = e1 / "runs"
    runs.mkdir(exist_ok=True)
    state_path = runs / "submissions.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"jobs": []}
    report = json.loads((e1 / "requests/E1_build_report.json").read_text(encoding="utf-8"))
    prov.load_keys()
    errors = 0
    for shard in report["shards"]:
        provider, k = shard["provider"], int(shard["shard"])
        name = prov.job_name(provider, k, args.attempt)
        path = e1 / "requests/batches" / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise SystemExit(f"SHARD HASH MISMATCH: {shard['file']}")
        known = [j for j in state["jobs"] if j["name"] == name]
        if known:
            print(f"{name}: already recorded ({known[0]['batch_job_id']})")
            continue
        remote = prov.find_jobs(provider, name)
        if remote:
            job = remote[0]
            print(f"{name}: found on provider ({job['batch_job_id']}); not resubmitting")
            if len(remote) > 1:
                print(f"  WARNING: {len(remote)} jobs with this name")
        else:
            deadline = time.time() + args.max_wait_minutes * 60
            while True:
                try:
                    job = (prov.submit_openai(path, name) if provider == "openai"
                           else prov.submit_google(path, EXPECTED_MODEL["google"], name))
                    break
                except Exception as exc:  # noqa: BLE001
                    text = f"{type(exc).__name__}: {str(exc)[:300]}"
                    quota = "429" in text or "RESOURCE_EXHAUSTED" in text
                    print(f"{name}: submit error: {text}")
                    if quota and time.time() < deadline:
                        time.sleep(120)
                        again = prov.find_jobs(provider, name)  # never run a shard twice
                        if again:
                            job = again[0]
                            break
                        continue
                    errors += 1
                    job = None
                    break
            if job is None:
                continue
            print(f"{name}: submitted ({job['batch_job_id']}) state {job['state']}")
        state["jobs"].append({
            "name": name, "provider": provider, "shard": k, "attempt": args.attempt,
            "batch_job_id": job["batch_job_id"], "input_file_id": job.get("input_file_id", ""),
            "state_at_submit": job["state"], "input_file": shard["file"], "input_sha256": shard["sha256"],
            "requests": shard["requests"], "submitted_utc": now(),
            "collected": False,
        })
        save(state_path, state)
    print("submissions recorded:", len(state["jobs"]), "errors:", errors)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
