import pandas as pd

from scripts.build_thesis_outputs_v21 import _reproduction_status
from thesis_repro import compare


def test_reproduction_status_comes_from_the_comparison():
    assert _reproduction_status(pd.Series({"estimate_reconstructed": 0.1, "estimate_frozen": 0.1})) == "PASS"
    assert _reproduction_status(pd.Series({"estimate_reconstructed": 0.1, "estimate_frozen": 0.2})) == "FAIL"
    assert _reproduction_status(pd.Series({"estimate_reconstructed": None, "estimate_frozen": 0.2})) == "NOT_CHECKED"
    assert _reproduction_status(pd.Series({"estimate": 0.2})) == "NOT_CHECKED"


def test_stage9_comparison_without_matches_is_unavailable(tmp_path):
    run_dir = tmp_path / "runs" / "x"
    fresh = run_dir / "11_stage9/llm_stage9_primary_contrast_summary.csv"
    frozen = tmp_path / "frozen/original_release/evaluation/llm_final_evaluation_20260913/stage9/BASELINE_STRICT_ITT/llm_stage9_primary_contrast_summary.csv"
    for path, contrast in ((fresh, "C5-C4"), (frozen, "C4R-C4")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"contrast,backend,mean_effect\n{contrast},alpha,0.1\n", encoding="utf-8")
    result = compare._stage9_comparison(tmp_path, run_dir)
    assert result["status"] == "UNAVAILABLE"
    assert result["matched_estimands"] == 0
