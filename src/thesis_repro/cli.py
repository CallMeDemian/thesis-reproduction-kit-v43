from __future__ import annotations

import argparse
import json
import platform
import sys
import zipfile
from pathlib import Path

from .compare import compare_run
from .c3e_rebuild import rebuild_original_c3e
from .contracts import EXPECTED_REQUESTS, contract_report
from .data import data_doctor, restore_data
from .frozen import verify_frozen
from .original_release import verify_original_release
from .paths import ROOT, load_json
from .run_engine import STAGES, execute, plan, trace_run


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m thesis_repro")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("verify")
    sub.add_parser("frozen-replay")
    sub.add_parser("verify-original")
    sub.add_parser("rebuild-c3e").add_argument("--release", choices=("original",), required=True)
    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_sub.add_parser("doctor")
    restore = data_sub.add_parser("restore")
    restore.add_argument("--raw-all", type=Path, required=True)
    restore.add_argument("--raw-nonfinancial", type=Path, required=True)
    restore.add_argument("--ratings", type=Path, required=True)
    fresh = sub.add_parser("fresh")
    fresh.add_argument("--mode", choices=tuple(STAGES), required=True)
    fresh.add_argument("--run-id", required=True)
    fresh.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    fresh.add_argument("--plan", action="store_true")
    fresh.add_argument("--resume", action="store_true")
    fresh.add_argument("--from-stage")
    fresh.add_argument("--to-stage")
    fresh.add_argument("--execute-llm", action="store_true")
    fresh.add_argument("--dry-render", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--run-id", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--run-id", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("kind", choices=("claim", "result", "run"))
    inspect.add_argument("identifier")
    trace = sub.add_parser("trace")
    trace.add_argument("--run-id", required=True)
    return parser


def _doctor() -> dict[str, object]:
    required = [ROOT / "PROVENANCE.md", ROOT / "provenance/source_release.json", ROOT / "frozen/release/frozen_manifest.json", ROOT / "contracts/llm/final_as_executed_generation_contract.json"]
    result = {"status": "PASS" if all(path.is_file() for path in required) else "FAIL", "python": sys.version.split()[0], "platform": platform.platform(), "root": str(ROOT), "required_files": {str(path.relative_to(ROOT)): path.is_file() for path in required}, "runtime_old_repository_dependency": 0}
    return result


def _inspect(kind: str, identifier: str) -> object:
    if kind == "run":
        return load_json(ROOT / "runs" / identifier / "run_manifest.json")
    if kind == "result":
        frozen = verify_frozen()
        return {"result_id": identifier, "frozen_registry": "available" if frozen["status"] == "PASS" else "unavailable", "hint": "Extract the registry member from the frozen ZIP for row-level inspection."}
    text = (ROOT / "frozen/evidence/registries/claims/claims.yaml").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if identifier in line]
    return {"claim_id": identifier, "matches": lines[:20], "match_count": len(lines)}


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            _print(_doctor())
        elif args.command == "verify":
            result = verify_frozen()
            result["doctor"] = _doctor()
            _print(result)
            if result["status"] != "PASS":
                raise SystemExit(1)
        elif args.command == "frozen-replay":
            result = verify_frozen()
            _print(result)
            if result["status"] != "PASS":
                raise SystemExit(1)
        elif args.command == "verify-original":
            result = verify_original_release()
            _print(result)
            if result["status"] != "PASS":
                raise SystemExit(1)
        elif args.command == "rebuild-c3e":
            result = rebuild_original_c3e()
            _print(result)
            if result["status"] != "PASS":
                raise SystemExit(1)
        elif args.command == "data":
            _print(data_doctor() if args.data_command == "doctor" else restore_data(args.raw_all, args.raw_nonfinancial, args.ratings))
        elif args.command == "fresh":
            if args.plan:
                _print(plan(args.mode, args.run_id, args.profile))
            else:
                _print(execute(args.mode, args.run_id, args.profile, args.resume, args.from_stage, args.to_stage, args.execute_llm, args.dry_render))
        elif args.command == "status":
            _print(load_json(ROOT / "runs" / args.run_id / "run_manifest.json"))
        elif args.command == "compare":
            _print(compare_run(args.run_id))
        elif args.command == "inspect":
            _print(_inspect(args.kind, args.identifier))
        elif args.command == "trace":
            _print(trace_run(args.run_id))
    except (FileNotFoundError, ValueError, PermissionError) as exc:
        _print({"status": "FAIL", "error": str(exc)})
        raise SystemExit(2)
