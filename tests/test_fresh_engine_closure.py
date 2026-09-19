from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from credit_recourse.contracts.v43_action_contract import ACTION_DIMENSIONS, ACTION_IDS, load_action_contract
from credit_recourse.eval.v43_oracle_backends import score_alpha, score_beta_ordered_logit_params, score_gamma_model
from credit_recourse.oracle.stage0.build_stage0_foundation_from_raw import build_stage0_foundation
from thesis_repro.contracts import EXPECTED_REQUESTS, render_requests
from thesis_repro.live_llm import gate_status, mock_responses
from thesis_repro.runtime_paths import FreshRuntimePaths
from thesis_repro.stages.base import StageResult
from thesis_repro.stages.adapters import HEAVY_GATE, LIVE_GATE
from thesis_repro.stages.oracle import _legacy_references, _require_stage1_success
from credit_recourse.oracle.fresh_runtime import resolve_fresh_oracle_runtime

ROOT = Path(__file__).resolve().parents[1]


def test_action_contract_is_eight_dimensional():
    assert len(load_action_contract(ROOT)["action_columns"]) == 8


def test_action_contract_has_nine_actions():
    assert len(load_action_contract(ROOT)["candidate_ids"]) == 9


def test_action_ids_are_canonical():
    assert ACTION_IDS == ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")


def test_action_dimensions_are_canonical():
    assert "revenue_growth" not in ACTION_DIMENSIONS


def test_fresh_contract_has_no_policy_revenue_growth_column():
    assert not any("revenue_growth" in c for c in load_action_contract(ROOT)["action_columns"])


def test_fresh_request_cardinality():
    assert len(render_requests(run_id="closure-test")) == EXPECTED_REQUESTS


def test_fresh_request_ids_are_unique():
    rows = render_requests(limit=200, run_id="closure-test")
    assert len({row["request_id"] for row in rows}) == len(rows)


def test_fresh_request_namespace_isolated():
    assert render_requests(limit=1, run_id="a")[0]["request_id"] != render_requests(limit=1, run_id="b")[0]["request_id"]


def test_live_gate_is_closed_by_default():
    assert gate_status()["authorized"] is False


def test_gate_strings_are_explicit():
    assert HEAVY_GATE == "I_APPROVE_28_ACTOR_RETRAIN"
    assert LIVE_GATE == "I_APPROVE_FRESH_REPLICATION"


def test_runtime_paths_are_run_local():
    paths = FreshRuntimePaths.from_run(ROOT, "closure-path-test", create=False)
    assert "runs/closure-path-test" in str(paths.run_root).replace("\\", "/")


def test_runtime_paths_have_stage8_and_stage9():
    paths = FreshRuntimePaths.from_run(ROOT, "closure-path-test", create=False)
    assert paths.stage8_root.name == "10_stage8"
    assert paths.stage9_root.name == "11_stage9"


def test_pass_real_compute_requires_execution():
    with pytest.raises(ValueError):
        StageResult("Stage8", "PASS", "REAL_COMPUTE")


def test_audit_missing_module_count_is_zero():
    payload = json.loads((ROOT / "runs/_runtime_audit/missing_internal_modules.json").read_text(encoding="utf-8"))
    assert payload["missing"] == []


def test_audit_records_required_raw_inputs():
    assert (ROOT / "runs/_runtime_audit/required_raw_inputs.json").is_file()


def test_audit_records_rebinding_plan():
    assert (ROOT / "runs/_runtime_audit/runtime_rebinding_plan.json").is_file()


def test_full_no_input_acceptance_receipt(tmp_path, monkeypatch):
    import thesis_repro.data as data
    import thesis_repro.run_engine as engine
    report = {"status": "INPUT_REQUIRED", "present_file_count": 0}
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "verify_input_contract", lambda write=True: report)
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/input_contract_report.json").write_text(json.dumps(report), encoding="utf-8")
    manifest = engine.execute("FullClean", "test-full-no-input", "full")
    assert manifest["completion_state"] == "INPUT_REQUIRED"
    assert manifest["stage_status"]["VerifyInputs"] == "INPUT_REQUIRED"


def test_full_no_input_does_not_claim_scientific_execution(tmp_path, monkeypatch):
    import thesis_repro.data as data
    import thesis_repro.run_engine as engine
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "verify_input_contract", lambda write=True: {"status": "INPUT_REQUIRED", "present_file_count": 0})
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/input_contract_report.json").write_text("{}", encoding="utf-8")
    manifest = engine.execute("FullClean", "test-full-no-input", "full")
    assert manifest["execution_class"] == "fresh_replication"
    assert list(manifest["stage_status"]) == ["VerifyInputs"]


def test_smoke_artifacts_are_marked_contract_smoke_only(tmp_path, monkeypatch):
    import thesis_repro.data as data
    import thesis_repro.run_engine as engine
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "verify_input_contract", lambda write=True: {"status": "INPUT_REQUIRED", "present_file_count": 0})
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/input_contract_report.json").write_text("{}", encoding="utf-8")
    manifest = engine.execute("OracleClean", "test-smoke", "smoke")
    assert manifest["execution_class"] == "contract_smoke"


def test_stage1_rc2_cannot_pass_with_existing_backend_params(tmp_path):
    backend_root = tmp_path / "runs" / "fake" / "02_oracle" / "work" / "stage1_oracle_backends"
    for backend, filename in {
        "alpha": "oracle_alpha_params.json",
        "beta": "benchmark_beta_params.json",
        "gamma": "benchmark_gamma_params.json",
    }.items():
        path = backend_root / backend / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    ledger = tmp_path / "stage1_oracle_backends_full_development.json"
    ledger.write_text(json.dumps({"status": "FAIL", "final_result_allowed": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact rc=0"):
        _require_stage1_success(2, ledger)


def test_fresh_runtime_resolves_environment_at_call_time(tmp_path, monkeypatch):
    first = tmp_path / "runs" / "first" / "02_oracle" / "work"
    second = tmp_path / "runs" / "second" / "02_oracle" / "work"
    monkeypatch.setenv("THESIS_REPRO_ORACLE_WORK_ROOT", str(first))
    assert resolve_fresh_oracle_runtime(tmp_path).work_root == first.resolve()
    monkeypatch.setenv("THESIS_REPRO_ORACLE_WORK_ROOT", str(second))
    assert resolve_fresh_oracle_runtime(tmp_path).work_root == second.resolve()


def test_fresh_oracle_verifier_call_graph_has_no_legacy_parent_literals():
    verifier_paths = [
        ROOT / "src/credit_recourse/oracle/verification/verify_stage00_04_growth_eligibility_contract.py",
        ROOT / "src/credit_recourse/oracle/verification/verify_alpha_contract.py",
        ROOT / "src/credit_recourse/oracle/verification/verify_oracle_semantic_closure.py",
        ROOT / "src/credit_recourse/oracle/verification/verify_stage1_substrate_validation.py",
        ROOT / "src/credit_recourse/oracle/verification/verify_stage1_outputs.py",
    ]
    forbidden = ("data/final_freeze", "configs/current", "archive/DEPLOYED_RELEASE", "frozen/")
    for path in verifier_paths:
        source = path.read_text(encoding="utf-8").replace("\\", "/")
        assert not any(token in source for token in forbidden), path


def test_fresh_registry_lineage_detector_rejects_legacy_parent():
    assert _legacy_references({"output": "runs/demo/02_oracle/work/x.parquet"}) == []
    assert _legacy_references({"output": "data/final_freeze/stage1/x.parquet"})


def test_mock_materialization_does_not_require_network(tmp_path):
    rows = render_requests(limit=2, run_id="mock-closure")
    assert mock_responses(rows, tmp_path / "mock-closure-responses.jsonl")["execution_class"] == "mock_only"


def test_capabilities_do_not_claim_full_execution():
    payload = json.loads((ROOT / "capabilities.json").read_text(encoding="utf-8"))
    assert "UNEXECUTED" in payload["FullClean"]


class _TinyGamma:
    def predict(self, frame):
        return [4.0] * len(frame)


def test_production_alpha_beta_gamma_paths_are_distinct(tmp_path):
    frame = pd.DataFrame({"x": [0.25, 1.75]})
    alpha = {
        "selected_variables": ["x"], "fin_ids": ["x"], "nonfin_ids": [],
        "directions": {"x": "higher_good"}, "bin_edges": {"x": {"edges": [0.0, 1.0, 2.0]}},
        "bin_score_table_isotonic": {"x": {"0": 10.0, "1": 90.0}}, "item_weights": {"x": 1.0},
        "block_norm": {"financial": {"p01": 0.0, "p99": 100.0}, "nonfinancial": {"p01": 0.0, "p99": 100.0}},
        "boundaries": {"AAA": 90, "AA": 80, "A": 70, "BBB": 60, "BB": 50, "B": 40, "CCC": 30, "CC": 20, "C": 10},
        "imputation_map": {"x": 50.0}, "pd_map": {}, "combined_weights": {"financial": 1.0, "nonfinancial": 0.0}
    }
    alpha_path = tmp_path / "alpha.json"; alpha_path.write_text(json.dumps(alpha), encoding="utf-8")
    beta = {"model_name": "ordered-logit-beta", "selected_variables": ["x"], "standardization_params": {"x": {"mean": 0, "std": 1}}, "coefficients": [{"variable": "x", "coefficient": 1.0}], "modeled_grade_nums": list(range(1, 11)), "finite_cutpoints": list(range(1, 10))}
    beta_path = tmp_path / "beta.json"; beta_path.write_text(json.dumps(beta), encoding="utf-8")
    gamma_path = tmp_path / "gamma.joblib"; joblib.dump(_TinyGamma(), gamma_path)
    gamma_params = {"selected_variables": ["x"]}; gamma_param_path = tmp_path / "gamma.json"; gamma_param_path.write_text(json.dumps(gamma_params), encoding="utf-8")
    a = score_alpha(frame, alpha_path); b = score_beta_ordered_logit_params(frame, beta_path); g = score_gamma_model(frame, gamma_param_path, gamma_path)
    assert not a.equals(b) and not b.equals(g) and not a.equals(g)


def test_real_stage0_fixture_produces_numerical_panel(tmp_path):
    raw_all = tmp_path / "raw_all"; raw_rating = tmp_path / "rating_sample"; out = tmp_path / "stage0"
    raw_all.mkdir(); raw_rating.mkdir()
    pd.DataFrame({"거래소코드": [1], "회계년도": [2020], "회사명": ["Fixture Co"], "시장": ["KOSPI"], "U01A110000000": [1000.0]}).to_excel(raw_all / "재무상태표.xlsx", index=False)
    pd.DataFrame({"거래소코드": [1], "회계년도": [2020], "회사명": ["Fixture Co"], "시장": ["KOSPI"], "신용등급": ["BBB"], "증권구분": [40], "평가사구분": [10], "평가사명 및 등급": ["NICE BBB"], "평가일": ["2020/06/30"]}).to_excel(raw_rating / "rating.xlsx", index=False)
    meta = build_stage0_foundation(tmp_path, raw_all, raw_rating, out, clean=True)
    panel = pd.read_parquet(out / "canonical_panel/stage0_canonical_panel.parquet")
    validation = json.loads((out / "stage0_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "PASS"
    assert len(panel) == 1 and panel["rating_num_10"].notna().all()
