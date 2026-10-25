"""Build fresh Stage9 outputs without touching the frozen thesis registry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fresh_thesis_outputs(root: Path, run_id: str) -> dict[str, Any]:
    root = Path(root).resolve()
    run = root / "runs" / run_id
    stage9 = run / "11_stage9"
    out = run / "13_thesis_outputs"
    required = {
        "stage9_metadata": stage9 / "metadata.json",
        "primary_contrast_summary": stage9 / "llm_stage9_primary_contrast_summary.csv",
        "budget_interaction_summary": stage9 / "llm_stage9_budget_interaction_summary.csv",
        "revision_metrics": stage9 / "llm_stage9_revision_metrics.parquet",
        "identity_contrast": stage9 / "llm_stage9_identity_contrast.parquet",
        "failure_audit": stage9 / "llm_stage9_failure_audit.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fresh Stage9 outputs missing: {missing}")
    out.mkdir(parents=True, exist_ok=True)
    workbook = out / "fresh_stage9_outputs.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for sheet, path in (
            ("primary_contrasts", required["primary_contrast_summary"]),
            ("budget_interaction", required["budget_interaction_summary"]),
            ("failure_audit", required["failure_audit"]),
        ):
            pd.read_csv(path).to_excel(writer, sheet_name=sheet[:31], index=False)
        pd.read_parquet(required["revision_metrics"]).head(10000).to_excel(writer, sheet_name="revision_metrics", index=False)
        pd.read_parquet(required["identity_contrast"]).head(10000).to_excel(writer, sheet_name="identity_contrast", index=False)
    # Read the workbook back and compare each sheet's row count with its source table.
    expected_rows = {
        "primary_contrasts": len(pd.read_csv(required["primary_contrast_summary"])),
        "budget_interaction": len(pd.read_csv(required["budget_interaction_summary"])),
        "failure_audit": len(pd.read_csv(required["failure_audit"])),
        "revision_metrics": min(len(pd.read_parquet(required["revision_metrics"])), 10000),
        "identity_contrast": min(len(pd.read_parquet(required["identity_contrast"])), 10000),
    }
    written_rows = {sheet: len(frame) for sheet, frame in pd.read_excel(workbook, sheet_name=None).items()}
    package_check = {sheet: {"source_rows": rows, "workbook_rows": written_rows.get(sheet)} for sheet, rows in expected_rows.items()}
    metadata = {
        "schema_version": "fresh_thesis_outputs_v1",
        "status": "PASS" if all(v["source_rows"] == v["workbook_rows"] for v in package_check.values()) else "FAIL",
        "status_basis": "workbook sheets read back and row counts compared with their same-run Stage9 sources",
        "package_check": package_check,
        "value_comparison_with_frozen": "NOT_CHECKED (done by the CompareFrozen stage)",
        "run_id": run_id,
        "source": "same_run_stage9",
        "frozen_registry_overwritten": False,
        "fresh_registry_claimed_rows": None,
        "artifacts": {
            name: {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in {**required, "workbook": workbook}.items()
        },
    }
    (out / "OUTPUT_METADATA.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(build_fresh_thesis_outputs(args.project_root, args.run_id), ensure_ascii=False, indent=2))
