from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
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
from thesis_repro.status import aggregate_completion
from thesis_repro.run_engine import STAGES
from thesis_repro.execution_context import ExecutionContext, FRESH_REPLICATION, SYNTHETIC_E2E_ACCEPTANCE, scoped_oracle_compatibility, receipt_context
from thesis_repro.c3e import aggregate_hierarchical, load_fresh_rl_contract
from thesis_repro.fresh_rl import verify_actor_graph
from thesis_repro.fresh_llm import prepare_requests, generate_mock, materialize_responses
from thesis_repro.stage8_runtime import verify_stage8
from thesis_repro.stage9_runtime import run_stage9
from thesis_repro.stages.oracle import _legacy_references, _require_stage1_success
from credit_recourse.oracle.fresh_runtime import materialize_fresh_oracle_registry, resolve_fresh_oracle_runtime
from credit_recourse.oracle.verification.verify_stage1_substrate_validation import _verdict
from thesis_repro.fresh_simulator import _validate as validate_simulator_panel
from thesis_repro.fresh_rl_dataset import verify_fresh_rl_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_action_contract_is_eight_dimensional():
    assert len(load_action_contract(ROOT)["action_columns"]) == 8


def test_action_contract_has_nine_actions():
    assert len(load_action_contract(ROOT)["candidate_ids"]) == 9


def test_action_ids_are_canonical():
    assert ACTION_IDS == ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")


def test_action_dimensions_are_canonical():
    assert "revenue_growth" not in ACTION_DIMENSIONS


def _minimal_simulator_panel(contract, *, total_debt_delta_by_action=None, accounting_check="ok"):
    deltas = total_debt_delta_by_action or {}
    rows = []
    for candidate_id in ACTION_IDS:
        row = {
            "firm_id": "000001",
            "base_year": 2022,
            "candidate_id": candidate_id,
            "state__total_debt": 100.0,
            "sim__total_debt": 100.0 + float(deltas.get(candidate_id, 0.0)),
            "accounting_check_json": json.dumps({"check": accounting_check}),
        }
        row.update({column: 0.0 for column in contract["action_columns"]})
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_minimal_simulator_panel(tmp_path, frame):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"fixture")
    return validate_simulator_panel(
        frame,
        load_action_contract(ROOT),
        parent_hashes=["same-run-verify-oracle"],
        synthetic=True,
        source_panel=source,
    )


def test_simulator_duplicate_firm_year_action_is_rejected(tmp_path):
    frame = _minimal_simulator_panel(load_action_contract(ROOT))
    with pytest.raises(ValueError, match="duplicate firm/year/action"):
        _validate_minimal_simulator_panel(tmp_path, pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


def test_simulator_broken_accounting_identity_is_rejected(tmp_path):
    frame = _minimal_simulator_panel(load_action_contract(ROOT), accounting_check="mismatch")
    with pytest.raises(ValueError, match="accounting identity"):
        _validate_minimal_simulator_panel(tmp_path, frame)


def test_simulator_rf_principal_change_is_rejected(tmp_path):
    frame = _minimal_simulator_panel(load_action_contract(ROOT), total_debt_delta_by_action={"RF": 1.0})
    with pytest.raises(ValueError, match="RF candidate changed total principal"):
        _validate_minimal_simulator_panel(tmp_path, frame)


def test_simulator_dl_principal_increase_is_rejected(tmp_path):
    frame = _minimal_simulator_panel(load_action_contract(ROOT), total_debt_delta_by_action={"DL": 1.0})
    with pytest.raises(ValueError, match="DL candidate increased principal"):
        _validate_minimal_simulator_panel(tmp_path, frame)


def test_rl_dataset_training_temporal_cutoff_is_rejected(tmp_path):
    rows = []
    for candidate_id in ACTION_IDS:
        rows.append({
            "firm_id": "000001", "fiscal_year": 2024, "outcome_year": 2025,
            "candidate_id": candidate_id, "reward_train": 0.0,
            "rl_fit_allowed": True, "outcome_available": True,
            "action_contract_sha256": "contract",
        })
    path = tmp_path / "rl_dataset.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="cutoff|leaked"):
        verify_fresh_rl_dataset(path, parent_hashes=["same-run-simulator"])


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
    monkeypatch.setattr(engine, "verify_input_contract", lambda write=True: {"status": "INPUT_CONTRACT_PASS", "present_file_count": 1})
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
    assert manifest["execution_class"] == "FRESH_REPLICATION"
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
    assert manifest["execution_class"] == "CONTRACT_SMOKE"


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


def test_fresh_registry_materializes_run_local_bindings(tmp_path, monkeypatch):
    source = tmp_path / "contracts/scientific/final_freeze"
    source.mkdir(parents=True)
    shutil.copy2(ROOT / "contracts/scientific/final_freeze/oracle_backend_registry.yaml", source / "oracle_backend_registry.yaml")
    work = tmp_path / "runs/registry-test/02_oracle/work"
    config = work / "contracts/oracle_components"
    monkeypatch.setenv("THESIS_REPRO_ORACLE_WORK_ROOT", str(work))
    monkeypatch.setenv("THESIS_REPRO_ORACLE_CONFIG_ROOT", str(config))
    registry_path = materialize_fresh_oracle_registry(tmp_path)
    import yaml
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert registry["provenance"]["materialized_run_id"] == "registry-test"
    assert registry["provenance"]["source_contract_sha256"]
    for backend in registry["backends"].values():
        for key in ("path", "params", "output", "metrics", "model"):
            if key in backend:
                assert str(backend[key]).replace("\\", "/").startswith(str(work / "stage1_oracle_backends").replace("\\", "/"))
                assert not any(token in str(backend[key]).replace("\\", "/") for token in ("data/final_freeze", "configs/current", "archive/DEPLOYED_RELEASE", "frozen/"))


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
    assert "PARTIAL_EXECUTION" in payload["FullClean"]
    assert "INPUT_GATED" in payload["FullClean"]


@pytest.mark.parametrize("status", ["FAILED", "INPUT_REQUIRED", "APPROVAL_REQUIRED", "NOT_IMPLEMENTED", "NOT_EXECUTED", "EXECUTED_UNVERIFIED"])
def test_full_terminal_failure_never_becomes_pass_with_skips(status):
    assert aggregate_completion([status], profile="full") == status


def test_smoke_status_is_never_scientific_pass():
    assert aggregate_completion(["SMOKE_PASS"], profile="smoke") == "SMOKE_PASS"
    assert aggregate_completion(["SMOKE_PASS_WITH_SKIPS"], profile="smoke") == "SMOKE_PASS_WITH_SKIPS"
    assert aggregate_completion(["FAILED"], profile="smoke") == "SMOKE_FAILED"


def test_fullclean_prefix_is_explicitly_partial(tmp_path, monkeypatch):
    import thesis_repro.data as data
    import thesis_repro.run_engine as engine
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "verify_input_contract", lambda write=True: {"status": "INPUT_CONTRACT_PASS", "present_file_count": 1})
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/input_contract_report.json").write_text(json.dumps({"status": "INPUT_CONTRACT_PASS"}), encoding="utf-8")
    manifest = engine.execute("FullClean", "partial-prefix", "smoke", to_stage="VerifyOracle")
    assert manifest["completion_state"] == "PARTIAL_EXECUTION"
    assert manifest["dag_complete"] is False
    assert manifest["certifiable"] is False
    assert manifest["missing_required_stages"]


def test_new_run_cannot_begin_from_downstream_stage(tmp_path, monkeypatch):
    import thesis_repro.data as data
    import thesis_repro.run_engine as engine
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "verify_input_contract", lambda write=True: {"status": "INPUT_REQUIRED", "present_file_count": 0})
    with pytest.raises(ValueError, match="may not begin from a downstream stage"):
        engine.execute("FullClean", "new-downstream", "full", from_stage="Simulator")


def test_ambient_synthetic_profile_cannot_downgrade_full_run(tmp_path, monkeypatch):
    import thesis_repro.data as data
    import thesis_repro.run_engine as engine
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setenv("THESIS_REPRO_ORACLE_PROFILE", "synthetic")
    monkeypatch.setattr(engine, "verify_input_contract", lambda write=True: {"status": "INPUT_REQUIRED", "present_file_count": 0})
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/input_contract_report.json").write_text(json.dumps({"status": "INPUT_REQUIRED"}), encoding="utf-8")
    manifest = engine.execute("FullClean", "ambient-profile", "full")
    assert manifest["execution_class"] == FRESH_REPLICATION
    assert manifest["scientific_gate_applicable"] is True
    assert manifest["stage_status"]["VerifyInputs"] == "INPUT_REQUIRED"
    assert manifest["completion_state"] == "INPUT_REQUIRED"
    assert os.environ["THESIS_REPRO_ORACLE_PROFILE"] == "synthetic"


def test_full_context_scoped_compatibility_restores_ambient_environment(monkeypatch):
    context = ExecutionContext.from_profile("full", "restore-env", "FullClean")
    monkeypatch.setenv("THESIS_REPRO_ORACLE_PROFILE", "synthetic")
    with scoped_oracle_compatibility(context):
        assert os.environ["THESIS_REPRO_ORACLE_PROFILE"] == "production"
    assert os.environ["THESIS_REPRO_ORACLE_PROFILE"] == "synthetic"


def test_stage_receipts_carry_authoritative_execution_context():
    synthetic = ExecutionContext.from_profile("synthetic", "receipt-synthetic", "FullClean")
    full = ExecutionContext.from_profile("full", "receipt-full", "FullClean")
    assert receipt_context(synthetic) == {"execution_class": SYNTHETIC_E2E_ACCEPTANCE, "execution_profile": "synthetic", "scientific_gate_applicable": False}
    assert receipt_context(full) == {"execution_class": FRESH_REPLICATION, "execution_profile": "full", "scientific_gate_applicable": True}


def test_full_run_rejects_non_scientific_stage_result(tmp_path, monkeypatch):
    import thesis_repro.data as data
    import thesis_repro.run_engine as engine
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "verify_input_contract", lambda write=True: {"status": "INPUT_CONTRACT_PASS", "present_file_count": 1})
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/input_contract_report.json").write_text(json.dumps({"status": "INPUT_CONTRACT_PASS"}), encoding="utf-8")
    monkeypatch.setattr(engine, "run_real_stage", lambda *args, **kwargs: StageResult("Oracle", "PASS", "REAL_COMPUTE", executed=True, details={"scientific_gate_applicable": False}))
    manifest = engine.execute("OracleClean", "scientific-gate", "full")
    assert manifest["stage_status"]["Oracle"] == "FAILED"
    assert manifest["completion_state"] == "FAILED"


def test_acceptance_uses_canonical_dag_and_real_dispatch():
    import thesis_repro.acceptance as acceptance
    from thesis_repro import dag
    assert not hasattr(acceptance, "STAGES")
    assert tuple(dag.stages_for("FullClean")[:5]) == ("VerifyInputs", "Oracle", "VerifyOracle", "Simulator", "RLDataset")


def test_verify_oracle_precedes_every_oracle_consumer():
    for mode, stages in STAGES.items():
        if "Oracle" in stages and len(stages) > stages.index("Oracle") + 1:
            assert stages[stages.index("Oracle") + 1] == "VerifyOracle", mode


def test_heavy_gate_is_before_first_rl_training_stage():
    for mode in ("OracleRLClean", "OracleRLLLMClean", "FullClean"):
        stages = STAGES[mode]
        assert stages.index("RLExecutionGate") < stages.index("RLEncoder") < stages.index("RLIQL")


def test_rq1_verdict_has_three_distinct_branches():
    assert _verdict({"status": "insufficient_movers"}) == "fail"
    assert _verdict({"status": "ok", "level_validity_spearman_oriented": 0.80, "lead_direction_agreement": 0.60, "lead_direction_agreement_ci95": [0.51, 0.70]}) == "partial_pass"
    assert _verdict({"status": "ok", "level_validity_spearman_oriented": 0.80, "lead_direction_agreement": 0.80, "lead_direction_agreement_ci95": [0.70, 0.88]}) == "strong_pass"


def test_resume_invalidates_tampered_stage_artifact(tmp_path, monkeypatch):
    import thesis_repro.data as data
    import thesis_repro.run_engine as engine
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "verify_input_contract", lambda write=True: {"status": "INPUT_CONTRACT_PASS", "present_file_count": 1})
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    report = {"status": "INPUT_CONTRACT_PASS", "present_file_count": 1}
    (raw / "input_contract_report.json").write_text(json.dumps(report), encoding="utf-8")
    first = engine.execute("OracleClean", "tamper-test", "smoke")
    oracle_manifest = tmp_path / "runs/tamper-test/02_oracle/stage_manifest_Oracle.json"
    oracle_artifact = tmp_path / json.loads(oracle_manifest.read_text(encoding="utf-8"))["artifacts"][0]["path"]
    oracle_artifact.write_text(oracle_artifact.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    second = engine.execute("OracleClean", "tamper-test", "smoke", resume=True)
    assert second["invalidations"]
    assert second["invalidations"][-1]["from_stage"] == "Oracle"
    assert second["stage_status"]["Oracle"] == "SMOKE_PASS"


def test_trace_detects_nested_forbidden_parent_and_unresolved_hash(tmp_path, monkeypatch):
    import thesis_repro.data as data
    import thesis_repro.run_engine as engine
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "verify_input_contract", lambda write=True: {"status": "INPUT_CONTRACT_PASS", "present_file_count": 1})
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/input_contract_report.json").write_text("{}", encoding="utf-8")
    engine.execute("OracleClean", "trace-nested", "smoke")
    artifact = tmp_path / "runs/trace-nested/02_oracle/oracle_contract.json"
    artifact.write_text(json.dumps({"nested": {"parent": "frozen/hidden"}}), encoding="utf-8")
    stage_manifest = tmp_path / "runs/trace-nested/02_oracle/stage_manifest_Oracle.json"
    payload = json.loads(stage_manifest.read_text(encoding="utf-8"))
    payload["parent_hashes"] = ["not-a-real-parent"]
    stage_manifest.write_text(json.dumps(payload), encoding="utf-8")
    trace = engine.trace_run("trace-nested")
    assert trace["nested_forbidden_parent_references"]
    assert trace["lineage_errors"]
    assert trace["lineage_closed"] is False


def test_fresh_rl_contract_uses_canonical_seed_labels_and_actor_count():
    contract = load_fresh_rl_contract(ROOT)
    assert contract["seeds"] == [2, 11, 12, 13, 14, 15, 16]
    assert contract["expected_actor_count"] == 28


def test_c3e_hierarchical_weights_are_not_flat_actor_average():
    contract = load_fresh_rl_contract(ROOT)
    contract["aggregation"]["outer_configuration_weights"] = {
        "B27": 0.1, "M2_S0935": 0.2, "DT06": 0.3, "T15_REWARD_S088": 0.4,
    }
    actors = {}
    for index, config in enumerate(contract["selected_configurations"]):
        actors[config] = {}
        for seed in contract["seeds"]:
            first = float(index + 1)
            values = np.array([[first, 1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
            values /= values.sum(axis=1, keepdims=True)
            actors[config][seed] = values
    final, detail = aggregate_hierarchical(actors, contract)
    expected = sum(
        actors[config][contract["seeds"][0]] * weight
        for config, weight in contract["aggregation"]["outer_configuration_weights"].items()
    )
    flat = np.mean(np.concatenate([actors[c][s] for c in actors for s in contract["seeds"]], axis=0), axis=0, keepdims=True)
    assert np.allclose(final, expected)
    assert not np.allclose(final, flat)
    assert detail["actor_count"] == 28


def test_rl_actor_graph_rejects_missing_and_duplicate_actor(tmp_path):
    contract = load_fresh_rl_contract(ROOT)
    records = []
    for config in contract["selected_configurations"]:
        for seed in contract["seeds"]:
            path = tmp_path / config / f"{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{config}:{seed}".encode())
            import hashlib
            records.append({"configuration": config, "seed": seed, "checkpoint_path": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "evaluation_base_year": 2024, "evaluation_firm_count": 575, "trained_on_evaluation_cohort": False, "oracle_output_in_reward": False})
    records.pop()
    records.append(dict(records[0]))
    report = verify_actor_graph(records, contract, root=tmp_path)
    assert report["status"] == "FAILED"
    assert any("missing_actor" in error for error in report["errors"])
    assert any("duplicate_actor" in error for error in report["errors"])


def test_llm_materialization_blocks_incomplete_response_set(tmp_path):
    report = prepare_requests(tmp_path / "llm-run", limit=3)
    assert report["status"] == "PASS"
    generate_mock(tmp_path / "llm-run")
    response_path = tmp_path / "llm-run/09_llm/raw_mock_responses.jsonl"
    response_path.write_text(response_path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    result = materialize_responses(tmp_path / "llm-run", expected_count=3)
    assert result["status"] == "FAILED"
    assert result["missing_request_ids"]


def test_stage8_requires_exact_strict_repaired_pairing(tmp_path):
    stage8 = tmp_path / "10_stage8"
    stage8.mkdir(parents=True)
    (stage8 / "stage8_rows.jsonl").write_text(json.dumps({"semantic_key": "r:Strict", "request_id": "r", "policy": "Strict", "lineage": {"same_run": True}}) + "\n", encoding="utf-8")
    report = verify_stage8(tmp_path, expected_rows=2)
    assert report["status"] == "FAILED"
    assert "strict_repaired_pairing_incomplete" in report["errors"]


def test_stage9_cannot_run_on_incomplete_stage8(tmp_path):
    (tmp_path / "10_stage8").mkdir(parents=True)
    (tmp_path / "10_stage8/stage8_validation_report.json").write_text(json.dumps({"status": "FAILED"}), encoding="utf-8")
    result = run_stage9(tmp_path)
    assert result["status"] == "FAILED"


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
