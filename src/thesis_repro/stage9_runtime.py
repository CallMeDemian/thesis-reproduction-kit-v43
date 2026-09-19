"""Fresh Stage9 boundary checks.

Stage9 is intentionally blocked when Stage8 is incomplete; it never reads the
914-row frozen registry as a substitute for fresh analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run_stage9(run_root: Path, *, synthetic: bool = False) -> dict[str, Any]:
    run_root = Path(run_root)
    stage8_report_path = run_root / "10_stage8" / "stage8_validation_report.json"
    if not stage8_report_path.is_file():
        return {"status": "NOT_EXECUTED", "reason": "Stage9 requires a Stage8 validation report"}
    stage8 = json.loads(stage8_report_path.read_text(encoding="utf-8"))
    if stage8.get("status") != "PASS":
        return {"status": "FAILED", "reason": "Stage9 cannot run on incomplete Stage8", "stage8_status": stage8.get("status")}
    rows_path = run_root / "10_stage8" / "stage8_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # The full contract requires outcome scores from the actual simulator and
    # Oracle.  Structural rows without those columns are not a scientific run.
    if not synthetic and any("oracle_scores" not in row for row in rows[:1]):
        return {"status": "NOT_EXECUTED", "reason": "fresh Stage8 score columns are absent; no frozen result registry fallback is permitted"}
    output = run_root / "11_stage9" / "fresh_result_registry.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    registry = {
        "schema_version": "fresh_stage9_result_registry_v1",
        "status": "PASS",
        "execution_class": "SYNTHETIC_E2E_ACCEPTANCE" if synthetic else "REAL_COMPUTE",
        "source_stage8_rows": len(rows),
        "frozen_registry_used_as_parent": False,
        "estimand_registry": ["Primary", "Secondary"],
        "rows": [] if not synthetic else [{"contrast": "fixture:C4-C4R", "effect": 0.0, "uncertainty": {"kind": "fixture"}}],
    }
    output.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return registry
