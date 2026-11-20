"""Rebuild and verify the frozen Stage2 raw action-source substrate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

FROZEN_RELATIVE = Path("data/final_freeze/stage2_candidate_projection/action_sources")
PRODUCER_RELATIVE = Path("src/credit_recourse/rl/pipelines/final_stage2_raw_action_source_precompute/pipeline.py")
CONTRACT_RELATIVE = Path("src/credit_recourse/rl/contracts/avs256_acd_v2.py")
EXPECTED_HASHES = {
    "stage2_raw_action_source_panel.parquet": "48ac0b0ac470c3fb4e6f36991479e352f09644a20b48593b8981257bc51f0cb4",
    "stage2_raw_concept_values_panel.parquet": "c67ebbb591ebcc4acadb724a4b90e05241f1330e38a18f49efeeb681b896067a",
    "stage2_raw_action_source_coverage.csv": "9d243e11facce604dc634c117efd1b4f0d101f41adf59960bf648d5f569c4d93",
}
EXPECTED_PRODUCER_HASH = "58d8dad1d01b3323274a5916e5e7b52b7b9cd7ce9098f607b06efbed913cf4ac"
EXPECTED_CONTRACT_HASH = "8e4683520c5b1738b95bf6c062d2da4f49339d6f366ed171f7bd2db7f0e08573"
EXPECTED_ROWS = 74867
EXPECTED_COLUMNS = 403
EXPECTED_CONTRACT_VERSION = "stage2_raw_data_direct_action_source_v3_stage1_actual_state_substrate"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def verify_directory(project_root: Path, directory: Path) -> dict[str, Any]:
    root = project_root.resolve()
    target = directory.resolve()
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    for name, expected in EXPECTED_HASHES.items():
        path = target / name
        observed = _sha256(path) if path.is_file() else None
        status = "PASS" if observed == expected else "FAIL"
        checks.append({"name": name, "expected_sha256": expected, "observed_sha256": observed, "status": status})
        if status == "FAIL":
            errors.append(f"missing or changed deterministic output: {path}")

    producer = root / PRODUCER_RELATIVE
    contract = root / CONTRACT_RELATIVE
    for name, path, expected in (
        ("producer", producer, EXPECTED_PRODUCER_HASH),
        ("avs_contract", contract, EXPECTED_CONTRACT_HASH),
    ):
        observed = _sha256(path) if path.is_file() else None
        status = "PASS" if observed == expected else "FAIL"
        checks.append({"name": name, "path": str(path), "expected_sha256": expected, "observed_sha256": observed, "status": status})
        if status == "FAIL":
            errors.append(f"missing or changed source: {path}")

    panel = target / "stage2_raw_action_source_panel.parquet"
    if panel.is_file():
        metadata = pq.read_metadata(panel)
        shape_ok = metadata.num_rows == EXPECTED_ROWS and metadata.num_columns == EXPECTED_COLUMNS
        checks.append({
            "name": "action_panel_shape",
            "rows": metadata.num_rows,
            "columns": metadata.num_columns,
            "expected_rows": EXPECTED_ROWS,
            "expected_columns": EXPECTED_COLUMNS,
            "status": "PASS" if shape_ok else "FAIL",
        })
        if not shape_ok:
            errors.append("action-source panel shape differs from frozen contract")

    report_path = target / "stage2_raw_action_source_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        contract_ok = report.get("contract_version") == EXPECTED_CONTRACT_VERSION and report.get("status") == "PASS"
        checks.append({
            "name": "report_contract",
            "contract_version": report.get("contract_version"),
            "producer_status": report.get("status"),
            "status": "PASS" if contract_ok else "FAIL",
        })
        if not contract_ok:
            errors.append("action-source report contract/status mismatch")
    else:
        errors.append(f"missing report: {report_path}")

    return {
        "schema_version": "stage2_action_source_reproduction_verification_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "project_root": str(root),
        "directory": str(target),
        "checks": checks,
        "errors": errors,
    }


def rebuild_and_verify(project_root: Path, output_dir: Path) -> dict[str, Any]:
    root = project_root.resolve()
    target = output_dir if output_dir.is_absolute() else root / output_dir
    target = target.resolve()
    frozen = (root / "data/final_freeze").resolve()
    if _inside(target, frozen):
        raise ValueError(f"refusing to rebuild inside immutable final_freeze: {target}")
    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "credit_recourse.rl.pipelines.final_stage2_raw_action_source_precompute.pipeline",
        "--project-root",
        str(root),
        "--out-dir",
        str(target),
    ]
    subprocess.run(command, cwd=root, check=True)
    return verify_directory(root, target)


def _write_report(project_root: Path, report: dict[str, Any], name: str) -> None:
    path = project_root / "data/reproduction/action_sources" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-existing")
    rebuild = sub.add_parser("rebuild-verify")
    rebuild.add_argument("--output-dir", type=Path, default=Path("data/reproduction/action_sources/rebuilt"))
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "verify-existing":
        result = verify_directory(root, root / FROZEN_RELATIVE)
        _write_report(root, result, "verify_existing.json")
    else:
        result = rebuild_and_verify(root, args.output_dir)
        _write_report(root, result, "rebuild_verify.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
