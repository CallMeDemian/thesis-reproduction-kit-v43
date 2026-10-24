from thesis_repro import compare

POINTER = "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 84535\n"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_c3e_comparison_without_lfs_is_unavailable(tmp_path):
    run_dir = tmp_path / "runs" / "synthetic"
    _write(run_dir / "07_c3e/c3e_actions.parquet", "fresh table is not read")
    _write(tmp_path / "frozen/original_release/rl/C3E_E2_7SEED_BALANCED_DFEBAFA6/C3E_firm_actions.parquet", POINTER)
    result = compare._c3e_comparison(tmp_path, run_dir)
    assert result["status"] == "UNAVAILABLE"
    assert "LFS" in result["reason"]


def test_stage9_comparison_without_lfs_is_unavailable(tmp_path):
    run_dir = tmp_path / "runs" / "synthetic"
    _write(run_dir / "11_stage9/llm_stage9_primary_contrast_summary.csv", "contrast,backend,mean_effect\nC5-C4,alpha,0.1\n")
    _write(tmp_path / "frozen/original_release/evaluation/llm_final_evaluation_20260913/stage9/BASELINE_STRICT_ITT/llm_stage9_primary_contrast_summary.csv", POINTER)
    result = compare._stage9_comparison(tmp_path, run_dir)
    assert result["status"] == "UNAVAILABLE"
    assert "LFS" in result["reason"]


def test_real_file_is_not_a_pointer(tmp_path):
    path = tmp_path / "table.csv"
    _write(path, "row_id,action_id\n0,OE\n")
    assert compare._is_lfs_pointer(path) is False
