from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from thesis_repro import llm_runtime
from thesis_repro.llm_runtime import (
    CERTIFIED_DISTRIBUTION_SHA256,
    C6EX_PERMUTATION_SHA256,
    _evaluate_industry_binding_candidate,
    _materialize_fresh_industry_binding,
    _resolve_c6ex_permutation,
)
from credit_recourse.final_release.llm_contract import industry_binding_status


def _industry_paths(tmp_path: Path) -> SimpleNamespace:
    run_root = tmp_path / "runs" / "run"
    stage2_root = run_root / "03_stage2"
    input_source = stage2_root / "input_source"
    (input_source / "input_splits").mkdir(parents=True)
    config_root = run_root / "09_llm" / "config"
    config_root.mkdir(parents=True)
    canonical = pd.DataFrame(
        {
            "canonical_firm_year_id": [f"{index:06d}::2024" for index in range(575)],
            "firm_id": [f"{index:06d}" for index in range(575)],
            "fiscal_year": [2024] * 575,
        }
    )
    canonical.to_parquet(input_source / "input_splits" / "canonical_evaluation_row_ids.parquet", index=False)
    (config_root / "information_contract.json").write_text(
        json.dumps(
            {
                "industry_binding_manifest": "FY2024_INDUSTRY_BINDING_MANIFEST.json",
                "valid_industry_nonmissing": 575,
                "valid_industry_missing": 0,
                "valid_industry_coverage_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(root=tmp_path, run_root=run_root, stage2_root=stage2_root, llm_root=run_root / "09_llm")


def _valid_industry_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "firm_key": [f"{index:06d}::2024" for index in range(575)],
            "induty_code": ["212"] * 575,
            "industry_display_value": ["OpenDART 212"] * 575,
        }
    )


def _write_candidate(paths: SimpleNamespace, relative: str, frame: pd.DataFrame) -> Path:
    path = paths.stage2_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def test_industry_candidate_skips_existing_schema_failure(tmp_path):
    paths = _industry_paths(tmp_path)
    _write_candidate(paths, "input_source/phase_eval_candidate.parquet", pd.DataFrame({"firm_key": ["000000::2024"] * 575}))
    second = _write_candidate(paths, "phase_eval_candidate.parquet", _valid_industry_frame())
    result = _materialize_fresh_industry_binding(paths, paths.llm_root / "config")
    assert result["status"] == "PASS"
    assert result["selected_source"] == str(second.relative_to(tmp_path)).replace("\\", "/")
    assert any(item["reason"] == "INDUSTRY_COLUMNS_INCOMPLETE" for item in result["rejected_candidates"])


def test_industry_candidate_skips_partial_code_coverage(tmp_path):
    paths = _industry_paths(tmp_path)
    partial = _valid_industry_frame()
    partial.loc[:8, "induty_code"] = ""
    _write_candidate(paths, "input_source/phase_eval_candidate.parquet", partial)
    second = _write_candidate(paths, "phase_eval_candidate.parquet", _valid_industry_frame())
    result = _materialize_fresh_industry_binding(paths, paths.llm_root / "config")
    assert result["status"] == "PASS"
    assert result["selected_source"] == str(second.relative_to(tmp_path)).replace("\\", "/")
    assert any(item["reason"] == "INDUSTRY_CODE_INCOMPLETE" for item in result["rejected_candidates"])


def test_complete_same_run_industry_source_is_preferred(tmp_path):
    paths = _industry_paths(tmp_path)
    selected = _write_candidate(paths, "input_source/FY2024_INDUSTRY_BINDING.parquet", _valid_industry_frame())
    result = _materialize_fresh_industry_binding(paths, paths.llm_root / "config")
    assert result["status"] == "PASS"
    assert result["role"] == "FRESH_EXOGENOUS_INFORMATION_INPUT"
    assert result["selected_source"] == str(selected.relative_to(tmp_path)).replace("\\", "/")


def test_incomplete_same_run_source_falls_back_to_authorized_frozen_source(tmp_path, monkeypatch):
    paths = _industry_paths(tmp_path)
    _write_candidate(paths, "input_source/phase_eval_candidate.parquet", pd.DataFrame({"firm_key": ["000000::2024"] * 575}))
    frozen = paths.root / "frozen/original_release/llm/final_plan3/2cf6d6d0e4250e66ce882ab95f9d641f2c73711ffbc6429e9a203dcc2ee680a2/firm_payload_source.parquet"
    frozen.parent.mkdir(parents=True)
    _valid_industry_frame().to_parquet(frozen, index=False)
    result = _materialize_fresh_industry_binding(paths, paths.llm_root / "config")
    assert result["status"] == "PASS"
    assert result["role"] == "FROZEN_EXOGENOUS_INFORMATION_INPUT"
    monkeypatch.setenv("THESIS_REPRO_LLM_CONFIG_ROOT", str(paths.llm_root / "config"))
    status = industry_binding_status(paths.root)
    assert status["ready"] is True
    assert status["runtime_provenance_authorized"] is True
    assert status["api_key_leak_scan"] == "NOT_APPLICABLE_FRESH_RUNTIME"


def test_industry_no_valid_candidate_is_input_required_with_rejections(tmp_path):
    paths = _industry_paths(tmp_path)
    _write_candidate(paths, "input_source/phase_eval_candidate.parquet", pd.DataFrame({"firm_key": ["000000::2024"] * 575}))
    result = _materialize_fresh_industry_binding(paths, paths.llm_root / "config")
    assert result["status"] == "INPUT_REQUIRED"
    assert result["rejected_candidates"]
    assert any(item["reason"] == "INDUSTRY_COLUMNS_INCOMPLETE" for item in result["rejected_candidates"])


def test_computed_source_is_not_an_industry_candidate(tmp_path):
    paths = _industry_paths(tmp_path)
    computed = paths.root / "frozen/stage8/oracle_scores.parquet"
    computed.parent.mkdir(parents=True)
    _valid_industry_frame().to_parquet(computed, index=False)
    result = _materialize_fresh_industry_binding(paths, paths.llm_root / "config")
    assert result["status"] == "INPUT_REQUIRED"
    assert all("oracle_scores" not in item["source"] for item in result["rejected_candidates"])


def _copy_distribution(tmp_path: Path) -> tuple[SimpleNamespace, Path]:
    source_repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "repo"
    distribution = root / "frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip"
    distribution.parent.mkdir(parents=True)
    shutil.copy2(source_repo / "frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip", distribution)
    paths = SimpleNamespace(root=root, llm_root=root / "runs/run/09_llm")
    paths.llm_root.mkdir(parents=True)
    return paths, distribution


def test_bundled_c6ex_permutation_is_exact_and_resolved_automatically(tmp_path):
    paths, distribution = _copy_distribution(tmp_path)
    with zipfile.ZipFile(distribution) as archive:
        member = next(name for name in archive.namelist() if name.endswith("configs/current/llm/C6EX_permutation.parquet"))
        raw = archive.read(member)
    assert hashlib.sha256(distribution.read_bytes()).hexdigest() == CERTIFIED_DISTRIBUTION_SHA256
    assert hashlib.sha256(raw).hexdigest() == C6EX_PERMUTATION_SHA256
    result = _resolve_c6ex_permutation(paths, paths.llm_root / "config")
    assert result["status"] == "PASS"
    assert result["provenance"]["source_role"] == "CERTIFIED_BUNDLED_DESIGN_INPUT"
    assert result["provenance"]["member"] == member
    assert result["path"].read_bytes() == raw


def test_explicit_c6ex_override_is_accepted_when_exact(tmp_path):
    paths, _ = _copy_distribution(tmp_path)
    with zipfile.ZipFile(paths.root / "frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip") as archive:
        member = next(name for name in archive.namelist() if name.endswith("configs/current/llm/C6EX_permutation.parquet"))
        raw = archive.read(member)
    override = paths.root / "data/design/C6EX_permutation.parquet"
    override.parent.mkdir(parents=True)
    override.write_bytes(raw)
    result = _resolve_c6ex_permutation(paths, paths.llm_root / "config")
    assert result["status"] == "PASS"
    assert result["provenance"]["source_role"] == "EXPLICIT_RESTORED_DESIGN_INPUT"


def test_tampered_explicit_c6ex_override_fails_closed(tmp_path):
    paths, _ = _copy_distribution(tmp_path)
    override = paths.root / "data/design/C6EX_permutation.parquet"
    override.parent.mkdir(parents=True)
    override.write_bytes(b"tampered")
    result = _resolve_c6ex_permutation(paths, paths.llm_root / "config")
    assert result["status"] == "FAILED"
    assert result["reason"] == "C6EX_PERMUTATION_HASH_MISMATCH"


def test_missing_bundled_c6ex_is_input_required(tmp_path):
    paths = SimpleNamespace(root=tmp_path, llm_root=tmp_path / "runs/run/09_llm")
    paths.llm_root.mkdir(parents=True)
    result = _resolve_c6ex_permutation(paths, paths.llm_root / "config")
    assert result["status"] == "INPUT_REQUIRED"
    assert result["reason"] == "C6EX_PERMUTATION_REQUIRED"


def test_bundled_c6ex_member_hash_mismatch_fails_closed(tmp_path, monkeypatch):
    paths, distribution = _copy_distribution(tmp_path)
    bad_distribution = tmp_path / "bad.zip"
    member = "THESIS_REPRO_KIT_v2.1.1_FINAL/configs/current/llm/C6EX_permutation.parquet"
    with zipfile.ZipFile(bad_distribution, "w") as archive:
        archive.writestr(member, b"wrong-bytes")
    target = paths.root / "frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip"
    target.unlink()
    shutil.copy2(bad_distribution, target)
    real_sha = llm_runtime._sha256
    monkeypatch.setattr(llm_runtime, "_sha256", lambda path: CERTIFIED_DISTRIBUTION_SHA256 if Path(path).name == target.name else real_sha(path))
    result = _resolve_c6ex_permutation(paths, paths.llm_root / "config")
    assert result["status"] == "FAILED"
    assert result["reason"] == "C6EX_BUNDLED_PERMUTATION_HASH_MISMATCH"


def test_distribution_stage2_files_are_not_authoritative_producer_inputs():
    distribution = Path(__file__).resolve().parents[1] / "frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip"
    with zipfile.ZipFile(distribution) as archive:
        names = set(archive.namelist())
    assert any(name.endswith("stage2_candidate_projection/phase_eval_candidate.parquet") for name in names)
    assert any(name.endswith("stage2_candidate_projection/input_splits/canonical_business_plan_history.parquet") for name in names)
    assert not any(name.endswith("phase1_pretrain.parquet") for name in names)
    assert not any(name.endswith("phase2_bc.parquet") for name in names)
    assert not any(name.endswith("phase3_iql.parquet") for name in names)
    assert not any(name.endswith("actual_context.parquet") for name in names)
    assert not any(name.endswith("actual_financial_states.parquet") for name in names)
