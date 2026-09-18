from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .common import ContractError, find_repo_root
from .doctor import run_doctor
from .executor import freeze_release, generation_status, prepare_wave, run_full_dry_run
from .live_batch import poll_and_download, submit_wave
from .gemini_quota_scheduler import scheduler_tick
from .retry import prepare_retry


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m credit_recourse.final_release")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor"); d.add_argument("--json-out", type=Path)
    d = sub.add_parser("dry-run"); d.add_argument("--output-dir", type=Path, required=True)
    sub.add_parser("freeze-release")
    d = sub.add_parser("prepare-wave"); d.add_argument("--release-hash", required=True); d.add_argument("--wave", type=int, choices=(1,2), required=True)
    d = sub.add_parser("submit"); d.add_argument("--release-hash", required=True); d.add_argument("--wave", type=int, choices=(1,2), required=True); d.add_argument("--attempt-index", type=int, choices=(1,2,3), default=1); d.add_argument("--max-new-shards-per-provider", type=int, default=None); d.add_argument("--provider", choices=("google","openai"), default=None); d.add_argument("--execute-api", action="store_true")
    d = sub.add_parser("poll"); d.add_argument("--release-hash", required=True); d.add_argument("--wave", type=int, choices=(1,2), required=True); d.add_argument("--attempt-index", type=int, choices=(1,2,3), default=1); d.add_argument("--execute-api", action="store_true")
    d = sub.add_parser("gemini-quota-tick"); d.add_argument("--release-hash", required=True); d.add_argument("--wave", type=int, choices=(1,2), default=2); d.add_argument("--window-target", type=int, default=3); d.add_argument("--execute-api", action="store_true")
    d = sub.add_parser("retry"); d.add_argument("--release-hash", required=True); d.add_argument("--wave", type=int, choices=(1,2), required=True); d.add_argument("--attempt-index", type=int, choices=(2,3), required=True)
    d = sub.add_parser("status"); d.add_argument("--release-hash", required=True)
    d = sub.add_parser("materialize-actions"); d.add_argument("--release-hash", required=True)
    d = sub.add_parser("evaluate"); d.add_argument("--release-hash", required=True)
    return p


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.project_root.resolve()
    if args.command == "doctor":
        result = run_doctor(root)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True); args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        return result
    if args.command == "dry-run": return run_full_dry_run(root, args.output_dir)
    if args.command == "freeze-release":
        path, release = freeze_release(root); return {"status": release["status"], "path": str(path), "release": release}
    if args.command == "prepare-wave": return prepare_wave(root, args.release_hash, args.wave)
    if args.command in {"submit", "poll", "gemini-quota-tick"}:
        if args.command == "submit" and (root / "PAUSE_LLM_RUN").is_file():
            raise ContractError("LLM_RUN_PAUSED_SAFE: PAUSE_LLM_RUN exists; no new provider submission is permitted")
        doctor = run_doctor(root)
        if not args.execute_api or os.environ.get("CREDIT_RECOURSE_ENABLE_LLM_V43_FINAL_PLAN3") != "I_APPROVE_LLM_V43_FINAL_PLAN3_BATCH":
            raise ContractError("Live provider operation requires --execute-api and the exact final Plan-3 environment sentinel")
        if not doctor["live_api_ready"]:
            raise ContractError("Live provider operation blocked by doctor: " + "; ".join(doctor["live_api_blockers"]))
        if args.command == "gemini-quota-tick":
            return scheduler_tick(root, args.release_hash, wave=args.wave, execute_api=True, window_target=args.window_target)
        if args.command == "submit": return submit_wave(root, args.release_hash, args.wave, approved=True, attempt_index=args.attempt_index, max_new_shards_per_provider=args.max_new_shards_per_provider, provider_filter=args.provider)
        return poll_and_download(root, args.release_hash, args.wave, attempt_index=args.attempt_index)
    if args.command == "retry": return prepare_retry(root, args.release_hash, args.wave, args.attempt_index)
    if args.command == "status": return generation_status(root, args.release_hash)
    if args.command == "materialize-actions":
        from .evaluation_bridge import materialize_evaluation_inputs
        return materialize_evaluation_inputs(root, args.release_hash)
    if args.command == "evaluate":
        from .evaluation_bridge import materialize_evaluation_inputs
        from credit_recourse.eval.final_stage8_llm_multi_oracle_eval.pipeline import run_stage8
        from credit_recourse.eval.final_stage9_llm_rl_comparison.pipeline import run_stage9
        bridge = materialize_evaluation_inputs(root, args.release_hash)
        return {"status": "PASS", "materialization": bridge, "stage8": run_stage8(project_root=root), "stage9": run_stage9(project_root=root)}
    raise AssertionError(args.command)


def main(argv: list[str] | None = None) -> int:
    try:
        result = execute(parser().parse_args(argv))
    except (ContractError, FileNotFoundError, KeyError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False, indent=2)); return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0


if __name__ == "__main__":
    raise SystemExit(main())



