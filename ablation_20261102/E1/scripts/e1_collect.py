"""E1 collector: one-shot, idempotent. Run by Windows Task Scheduler every 30 minutes.

Each run: poll every uncollected job; download finished jobs (output and error files); process each
line with ``e1_process.process_line`` (normalize -> parse -> Strict ITT); resubmit only undelivered
requests with identical bodies (max 3 attempts per request, existing jobs checked first); select the
lowest-numbered delivered attempt per request and log later duplicates; append to
``runs/collector_log.jsonl``; when all 3,450 are delivered write ``runs/e1_selected.jsonl`` and disable
the scheduled task. No scores are computed or read.

A returned model that differs from the expected version stops resubmission and writes
``runs/BLOCKED.txt`` (never substitute a model).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_providers as prov  # noqa: E402
from e1_process import EXPECTED_MODEL, load_processing_context, process_line  # noqa: E402

TASK_NAME = "E1_collector"
MAX_ATTEMPTS = 3


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def usage_of(provider: str, line: dict):
    try:
        if provider == "openai":
            return ((line.get("response") or {}).get("body") or {}).get("usage")
        return (line.get("response") or {}).get("usageMetadata")
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-root", type=Path, required=True)
    ap.add_argument("--no-resubmit", action="store_true")
    args = ap.parse_args()
    root = args.kit_root.resolve()
    e1 = root / "ablation_20261102/E1"
    runs = e1 / "runs"
    raw_dir = runs / "raw"
    retry_dir = runs / "retry"
    raw_dir.mkdir(parents=True, exist_ok=True)
    retry_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs / "collector_log.jsonl"

    def log(event: str, **kw) -> None:
        with log_path.open("a", encoding="utf-8") as h:
            h.write(json.dumps({"utc": now(), "event": event, **kw}, ensure_ascii=False, default=str) + "\n")

    lock = runs / "collector.lock"
    if lock.exists() and time.time() - lock.stat().st_mtime < 3 * 3600:
        log("SKIP_LOCKED")
        return
    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        run(root, e1, runs, raw_dir, retry_dir, log, args)
    except Exception as exc:  # noqa: BLE001
        log("ERROR", error=f"{type(exc).__name__}: {str(exc)[:500]}")
        raise
    finally:
        lock.unlink(missing_ok=True)


def run(root, e1, runs, raw_dir, retry_dir, log, args) -> None:
    prov.load_keys()
    state_path = runs / "submissions.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(e1 / "requests/E1_request_manifest.csv", dtype=str)
    expected_ids = set(manifest.e1_request_id)
    provider_of = dict(zip(manifest.e1_request_id, manifest.provider))
    design = load_processing_context(root)
    log("RUN_START", jobs=len(state["jobs"]), uncollected=sum(not j["collected"] for j in state["jobs"]))

    # 1. poll and download
    for job in state["jobs"]:
        if job["collected"]:
            continue
        poll = prov.poll_openai(job["batch_job_id"]) if job["provider"] == "openai" else prov.poll_google(job["batch_job_id"])
        job["last_state"], job["last_polled_utc"], job["counts"] = poll["state"], now(), poll.get("counts")
        log("POLL", name=job["name"], state=poll["state"], counts=poll.get("counts"))
        if not poll["terminal"]:
            continue
        stem = job["name"]
        arts = (prov.download_openai(poll, raw_dir, stem) if job["provider"] == "openai"
                else prov.download_google(poll, raw_dir, stem))
        for a in arts:
            a["sha256"] = sha256_file(raw_dir / a["path"])
        job["artifacts"], job["terminal_state"], job["downloaded_utc"] = arts, poll["state"], now()
        log("DOWNLOADED", name=job["name"], state=poll["state"], artifacts=[a["path"] for a in arts])
        save_json(state_path, state)

    # 2. process downloaded jobs once
    responses_path = runs / "responses.jsonl"
    for job in state["jobs"]:
        if job["collected"] or "artifacts" not in job:
            continue
        rows, seen = [], set()
        for art in job["artifacts"]:
            with (raw_dir / art["path"]).open(encoding="utf-8") as h:
                for text in h:
                    if not text.strip():
                        continue
                    line = json.loads(text)
                    rid = str(line.get("custom_id") or line.get("key") or "")
                    if rid not in expected_ids or provider_of[rid] != job["provider"]:
                        log("UNKNOWN_REQUEST_ID", name=job["name"], request_id=rid)
                        continue
                    if rid in seen:
                        log("DUPLICATE_ID_IN_JOB", name=job["name"], request_id=rid)
                        continue
                    seen.add(rid)
                    res = process_line(job["provider"], line, design)
                    res.update(job_name=job["name"], attempt=job["attempt"], usage=usage_of(job["provider"], line),
                               processed_utc=now(), artifact_kind=art["kind"])
                    rows.append(res)
        with responses_path.open("a", encoding="utf-8") as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        job["collected"], job["processed_utc"], job["rows"] = True, now(), len(rows)
        job["delivered_rows"] = sum(r["outcome"] == "DELIVERED" for r in rows)
        job["retry_rows"] = sum(r["outcome"] == "RETRY" for r in rows)
        log("PROCESSED", name=job["name"], rows=len(rows), delivered=job["delivered_rows"], retry=job["retry_rows"],
            missing=job["requests"] - len(rows))
        save_json(state_path, state)

    # 3. selection
    rows = [json.loads(t) for t in responses_path.open(encoding="utf-8")] if responses_path.exists() else []
    drift = [r for r in rows if r["outcome"] == "DELIVERED" and r["returned_model_ok"] is not True]
    if drift:
        (runs / "BLOCKED.txt").write_text(
            f"{now()} returned_model differs from expected for {len(drift)} responses; no resubmission, no substitution. "
            f"First: {drift[0]['request_id']} -> {drift[0]['returned_model']}\n", encoding="utf-8")
        log("MODEL_DRIFT", count=len(drift))
    good = [r for r in rows if r["outcome"] == "DELIVERED" and r["returned_model_ok"] is True]
    selected: dict[str, dict] = {}
    for r in sorted(good, key=lambda r: (r["attempt"], r["job_name"])):
        if r["request_id"] in selected:
            log("LATE_DUPLICATE_NOT_USED", request_id=r["request_id"], used_attempt=selected[r["request_id"]]["attempt"],
                late_attempt=r["attempt"], job=r["job_name"])
            continue
        selected[r["request_id"]] = r
    n_sel = len(selected)
    by_provider = {p: sum(1 for i in selected if provider_of[i] == p) for p in ("openai", "google")}
    log("SELECTION", delivered=n_sel, of=len(expected_ids), by_provider=by_provider)

    # 4. complete?
    if n_sel == len(expected_ids) and not drift:
        out = runs / "e1_selected.jsonl"
        with out.open("w", encoding="utf-8") as h:
            for rid in sorted(selected):
                h.write(json.dumps(selected[rid], ensure_ascii=False, default=str) + "\n")
        (runs / "ALL_DELIVERED.txt").write_text(f"{now()} all {n_sel} requests delivered; selected file {out.name} sha256 {sha256_file(out)}\n", encoding="utf-8")
        log("ALL_DELIVERED", selected_sha256=sha256_file(out))
        subprocess.run(["schtasks", "/Change", "/TN", TASK_NAME, "/DISABLE"], capture_output=True)
        log("TASK_DISABLED", task=TASK_NAME)
        return

    # 5. resubmission of undelivered requests (identical bodies), existing jobs checked first
    if args.no_resubmit or drift:
        return
    member: dict[str, list[dict]] = {}
    for job in state["jobs"]:
        p = e1 / "requests/batches" / job["input_file"]
        p = p if p.exists() else retry_dir / job["input_file"]
        ids = []
        with p.open(encoding="utf-8") as h:
            for text in h:
                rec = json.loads(text)
                ids.append(rec.get("custom_id") or rec.get("key"))
        job["_ids"] = ids
        for rid in ids:
            member.setdefault(rid, []).append(job)
    todo: dict[tuple[str, int, int], list[str]] = {}
    exhausted = []
    for rid in sorted(expected_ids - set(selected)):
        jobs = member.get(rid, [])
        if any(not j["collected"] for j in jobs):
            continue  # still pending in some job
        attempts = max(j["attempt"] for j in jobs)
        if attempts >= MAX_ATTEMPTS:
            exhausted.append(rid)
            continue
        shard = min(j["shard"] for j in jobs)
        todo.setdefault((provider_of[rid], shard, attempts + 1), []).append(rid)
    if exhausted:
        log("INCOMPLETE_INFRA", requests=len(exhausted), first=exhausted[:3])
    for (provider, shard, attempt), ids in sorted(todo.items()):
        name = prov.job_name(provider, shard, attempt)
        if any(j["name"] == name for j in state["jobs"]):
            continue
        remote = prov.find_jobs(provider, name)
        orig = e1 / "requests/batches" / f"{provider}_e1_shard_{shard:05d}.jsonl"
        recs = {}
        with orig.open(encoding="utf-8") as h:
            for text in h:
                rec = json.loads(text)
                recs[rec.get("custom_id") or rec.get("key")] = text
        fname = f"{provider}_e1_retry_shard{shard}_attempt{attempt}.jsonl"
        path = retry_dir / fname
        with path.open("w", encoding="utf-8", newline="\n") as h:
            for rid in ids:
                h.write(recs[rid] if recs[rid].endswith("\n") else recs[rid] + "\n")
        try:
            if remote:
                job = remote[0]
                log("RESUBMIT_SKIPPED_EXISTING", name=name, batch_job_id=job["batch_job_id"])
            else:
                job = (prov.submit_openai(path, name) if provider == "openai"
                       else prov.submit_google(path, EXPECTED_MODEL["google"], name))
                log("RESUBMITTED", name=name, requests=len(ids), batch_job_id=job["batch_job_id"])
        except Exception as exc:  # noqa: BLE001
            log("RESUBMIT_FAILED", name=name, error=f"{type(exc).__name__}: {str(exc)[:300]}")
            continue
        state["jobs"].append({"name": name, "provider": provider, "shard": shard, "attempt": attempt,
                              "batch_job_id": job["batch_job_id"], "input_file_id": job.get("input_file_id", ""),
                              "state_at_submit": job["state"], "input_file": fname,
                              "input_sha256": sha256_file(path), "requests": len(ids), "submitted_utc": now(),
                              "collected": False})
        save_json(state_path, state)
    for job in state["jobs"]:
        job.pop("_ids", None)
    save_json(state_path, state)


if __name__ == "__main__":
    main()
