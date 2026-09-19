from __future__ import annotations

from pathlib import Path
from typing import Any

from .frozen import verify_frozen
from .paths import ROOT, load_json, write_json


def compare_run(run_id: str) -> dict[str, Any]:
    run_dir = ROOT / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"unknown run: {run_id}")
    manifest = load_json(run_dir / "run_manifest.json")
    frozen = verify_frozen()
    report = {
        "schema_version": "fresh_vs_frozen_v1",
        "run_id": run_id,
        "frozen_status": frozen["status"],
        "fresh_completion_state": manifest.get("completion_state"),
        "comparison_status": "COMPARISON_ONLY",
        "metrics": {
            "oracle": "UNAVAILABLE" if manifest.get("completion_state") != "PASS" else "PENDING_METRIC_BINDING",
            "c3e_action_agreement": "UNAVAILABLE",
            "c3e_alpha": "UNAVAILABLE",
            "c3e_delta_oe": "UNAVAILABLE",
            "c3e_entropy": "UNAVAILABLE",
            "llm_primary_contrasts": "UNAVAILABLE",
            "firm_level_effect_spearman": "UNAVAILABLE"
        },
        "frozen_reference_counts": {"stage8": 96600, "primary": 96, "supplemental": 722, "parent": 96, "registry": 914, "raw_llm_generations": 48300},
        "note": "Comparison is deliberately explicit. Missing fresh artifacts are reported as unavailable rather than treated as equality or failure."
    }
    comparison_dir = run_dir / "14_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_json(comparison_dir / "fresh_vs_frozen.json", report)
    lines = [f"# Fresh vs frozen: `{run_id}`", "", f"Fresh completion: `{report['fresh_completion_state']}`", f"Frozen verification: `{report['frozen_status']}`", "", "| Measure | Fresh | Frozen |", "|---|---|---|", "| Oracle metrics | unavailable | frozen evidence |", "| C3-E action agreement | unavailable | reference identity available |", "| LLM primary contrasts | unavailable | 96 primary rows |", "| Registry | unavailable | 914 rows |", "", report["note"]]
    (comparison_dir / "FRESH_VS_FROZEN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        from openpyxl import Workbook

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "comparison"
        sheet.append(["measure", "fresh", "frozen"])
        sheet.append(["completion_state", report["fresh_completion_state"], report["frozen_status"]])
        for key, value in report["metrics"].items():
            sheet.append([key, value, "reference/unavailable"])
        workbook.save(comparison_dir / "FRESH_VS_FROZEN.xlsx")
        report["xlsx"] = str((comparison_dir / "FRESH_VS_FROZEN.xlsx").relative_to(ROOT)).replace("\\", "/")
        write_json(comparison_dir / "fresh_vs_frozen.json", report)
    except ImportError:
        report["xlsx"] = "UNAVAILABLE: install .[data]"
    return report
