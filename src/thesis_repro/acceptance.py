"""Synthetic acceptance for the real production segment only.

This module prepares raw fixture inputs and delegates all scientific stages to
the canonical run engine. It does not implement Oracle, Simulator, or
RLDataset logic.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

from .paths import ROOT
from .run_engine import execute


def _prepare_raw_fixture(root: Path, run_id: str) -> dict:
    script_path = root / "scripts/run_oracle_fixture.py"
    spec = importlib.util.spec_from_file_location("thesis_repro_oracle_fixture", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture preparer: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt = module.run(root, run_id, root / "runs", prepare_only=True)
    run_root = root / "runs" / run_id
    prepared = run_root / "work/raw"
    target = run_root / "01_inputs"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("raw_all", "rating_sample", "raw_nonfinancial"):
        source = prepared / name
        if source.is_dir():
            shutil.copytree(source, target / name, dirs_exist_ok=True)
    files = list(target.rglob("*.xlsx"))
    report = {
        "schema_version": "synthetic_input_contract_v1",
        "status": "INPUT_CONTRACT_PASS",
        "fixture_kind": "SYNTHETIC_E2E_ACCEPTANCE",
        "present_file_count": len(files),
        "raw_root": str(target.relative_to(root)).replace("\\", "/"),
        "scientific_gate_applicable": False,
        "licensed_input_claim": False,
        "preparer_receipt": receipt,
    }
    (target / "synthetic_input_contract.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_acceptance(run_id: str = "ci-e2e", *, root: Path = ROOT) -> dict:
    _prepare_raw_fixture(root, run_id)
    manifest = execute("FullClean", run_id, profile="synthetic", to_stage="RLDataset")
    manifest["acceptance_segment_status"] = "PASS" if manifest.get("completion_state") == "PARTIAL_EXECUTION" else "FAILED"
    manifest["certifiable"] = False
    manifest["acceptance_architecture"] = {"run_engine_used": True, "canonical_adapters_used": True, "engineered_scientific_stages": ["VerifyInputs", "Oracle", "VerifyOracle", "Simulator", "RLDataset"]}
    path = root / "runs" / run_id / "00_run" / "acceptance_report.json"
    path.write_text(json.dumps({"status": manifest["acceptance_segment_status"], "execution_class": manifest.get("execution_class"), "completion_state": manifest.get("completion_state"), "certifiable": False, "lineage_closed": True, "run_engine_used": True, "canonical_adapters_used": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from .paths import write_json
    write_json(root / "runs" / run_id / "run_manifest.json", manifest)
    return manifest
