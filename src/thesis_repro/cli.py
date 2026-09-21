from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import zipfile
from pathlib import Path

from .c3e_rebuild import rebuild_original_c3e
from .data import data_doctor, restore_data
from .frozen import verify_frozen
from .original_release import verify_original_release
from .paths import ROOT, load_json
from .reproduce import reproduce
from .run_engine import execute, trace_run
from .certify import certify_run
from .status import is_scientific_accepted


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m thesis_repro")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("verify")
    sub.add_parser("frozen-replay")
    replay = sub.add_parser("reproduce")
    replay.add_argument("--full", action="store_true", help="run the optional raw-input fresh reproduction path")
    replay.add_argument("--run-id", default="thesis-reproduction")
    replay.add_argument("--live-llm", action="store_true")
    replay.add_argument("--resume", action="store_true")
    replay.add_argument("--from-stage")
    replay.add_argument("--to-stage")
    acceptance = sub.add_parser("acceptance-e2e")
    acceptance.add_argument("--run-id", default="ci-synthetic-full")
    trace = sub.add_parser("trace")
    trace.add_argument("--run-id", required=True)
    certify = sub.add_parser("certify")
    certify.add_argument("--run-id", required=True)
    search = sub.add_parser("search")
    search.add_argument("--rl", action="store_true", required=True)
    search.add_argument("--catalog", type=Path)
    search.add_argument("--source-run", required=True)
    search.add_argument("--run-id", default="search-reproduction")
    sub.add_parser("verify-original")
    sub.add_parser("rebuild-c3e").add_argument("--release", choices=("original",), required=True)
    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_sub.add_parser("doctor")
    restore = data_sub.add_parser("restore")
    restore.add_argument("--raw-all", type=Path, required=True)
    restore.add_argument("--raw-nonfinancial", type=Path, required=True)
    restore.add_argument("--ratings", type=Path, required=True)
    restore.add_argument("--stage2-source", type=Path)
    restore.add_argument("--c6ex-permutation", type=Path, help="optional exact-byte C6-EX override; absent uses the certified distribution member")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("kind", choices=("claim", "result", "run"))
    inspect.add_argument("identifier")
    return parser


def _doctor() -> dict[str, object]:
    def module_available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    required = [ROOT / "PROVENANCE.md", ROOT / "provenance/source_release.json", ROOT / "frozen/release/frozen_manifest.json", ROOT / "contracts/scientific/v43_action_contract.json"]
    required_modules = {name: module_available(name) for name in ("pandas", "pyarrow", "openpyxl", "yaml")}
    optional_modules = {name: module_available(name) for name in ("scipy", "sklearn", "statsmodels", "torch", "openai", "google.genai")}
    optional_modules["cuda"] = bool(module_available("torch") and __import__("torch").cuda.is_available())
    result = {
        "status": "PASS" if all(path.is_file() for path in required) and all(required_modules.values()) else "FAIL",
        "python": sys.version.split()[0], "platform": platform.platform(), "root": str(ROOT),
        "required_now": {"files": {str(path.relative_to(ROOT)): path.is_file() for path in required}, "modules": required_modules},
        "optional_for_oracle_rl_llm": optional_modules,
        "requested_gates": {"heavy_rl": "reproduce --full", "live_llm": "reproduce --full --live-llm"},
        "runtime_old_repository_dependency": 0,
    }
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
            result = reproduce(ROOT, "frozen-replay")
            _print(result)
            if result["status"] != "PASS":
                raise SystemExit(1)
        elif args.command == "reproduce":
            if args.full:
                result = execute("FullClean", args.run_id, profile="full", resume=args.resume, from_stage=args.from_stage, to_stage=args.to_stage, execute_llm=args.live_llm)
                result["scope"] = "FULL_EXPERIMENTAL_REPRODUCTION"
            else:
                result = reproduce(ROOT, args.run_id)
            _print(result)
            completion_state = result.get("completion_state") if args.full else result.get("status")
            if not is_scientific_accepted(str(completion_state)):
                raise SystemExit(1)
        elif args.command == "acceptance-e2e":
            from .acceptance import run_acceptance
            result = run_acceptance(args.run_id)
            _print(result)
            if result.get("completion_state") != "PASS" or result.get("dag_complete") is not True or result.get("certifiable") is not False:
                raise SystemExit(1)
        elif args.command == "trace":
            result = trace_run(args.run_id)
            _print(result)
            if result.get("lineage_closed") is not True:
                raise SystemExit(1)
        elif args.command == "certify":
            result = certify_run(args.run_id)
            _print(result)
            if result.get("state") not in {"CERTIFIED_FRESH_REPLICATION", "QUALIFIED_FRESH_REPLICATION"}:
                raise SystemExit(1)
        elif args.command == "search":
            from .search import reproduce_search
            result = reproduce_search(ROOT, mode="rl", catalog_path=args.catalog, source_run=args.source_run, run_id=args.run_id)
            _print(result)
            if result.get("status") != "PASS":
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
            _print(data_doctor() if args.data_command == "doctor" else restore_data(args.raw_all, args.raw_nonfinancial, args.ratings, stage2_source=args.stage2_source, c6ex_permutation=args.c6ex_permutation))
        elif args.command == "inspect":
            _print(_inspect(args.kind, args.identifier))
    except (FileNotFoundError, ValueError, PermissionError) as exc:
        _print({"status": "FAIL", "error": str(exc)})
        raise SystemExit(2)
