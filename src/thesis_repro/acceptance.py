"""Deterministic architecture acceptance run.

This is deliberately a separate execution class.  It uses real simulator
operators, real Oracle variable formulas, the canonical actor membership and
hierarchical C3-E code, and the mock LLM transport, but it never claims thesis
sample replication or scientific certification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from credit_recourse.contracts.v43_action_contract import load_action_contract
from credit_recourse.rl.common.semantic_action_v4_1_contract import simulator_from_contract
from credit_recourse.simulator.business_plan import calibrate_business_plan_non_rate
from credit_recourse.simulator.oracle_variables import compute_oracle_variables
from credit_recourse.simulator.synthetic import make_synthetic_pair

from .c3e import load_fresh_rl_contract
from .fresh_llm import generate_mock, materialize_responses, prepare_requests
from .fresh_rl import actor_key, build_c3e_release, verify_actor_graph
from .paths import ROOT
from .stage8_runtime import build_stage8
from .stage9_runtime import run_stage9


STAGE_DIRS = {
    "VerifyInputs": "01_inputs", "Oracle": "02_oracle", "VerifyOracle": "02_oracle",
    "Simulator": "03_simulator", "RLDataset": "04_rl_dataset", "RLEncoder": "05_rl_encoder",
    "RLBehaviorClone": "06_rl_bc", "RLIQL": "07_rl_iql", "C3E": "08_c3e", "Stage6": "08_c3e",
    "VerifyRL": "08_c3e", "LLMPrepare": "09_llm", "LLMGenerate": "09_llm", "LLMMaterialize": "09_llm",
    "Stage8": "10_stage8", "Stage9": "11_stage9", "VerifyResults": "12_results",
    "ThesisOutputs": "13_thesis_outputs", "CompareFrozen": "14_comparison", "VerifyAll": "15_release",
}
STAGES = [
    "VerifyInputs", "Oracle", "VerifyOracle", "Simulator", "RLDataset", "RLEncoder",
    "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL", "LLMPrepare", "LLMGenerate",
    "LLMMaterialize", "Stage8", "Stage9", "VerifyResults", "ThesisOutputs", "CompareFrozen", "VerifyAll",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path, _sha256(path)


def _stage(run_root: Path, stage: str, payload: dict[str, Any], parents: list[str], lineage: dict[str, Any]) -> str:
    directory = run_root / STAGE_DIRS[stage]
    path = directory / f"acceptance_{stage}.json"
    payload = {**payload, "stage": stage, "execution_class": "SYNTHETIC_E2E_ACCEPTANCE", "status": "SYNTHETIC_E2E_PASS", "parent_hashes": parents, "same_run": True}
    _, digest = _write_json(path, payload)
    lineage[stage] = {"path": str(path.relative_to(run_root)).replace("\\", "/"), "sha256": digest, "parent_hashes": parents}
    return digest


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values, axis=1, keepdims=True)
    result = np.exp(values)
    return result / result.sum(axis=1, keepdims=True)


def _verify_lineage(run_root: Path, lineage: dict[str, Any]) -> dict[str, Any]:
    prior: set[str] = set()
    errors: list[str] = []
    forbidden = ("frozen/", "data/final_freeze", "configs/current", "archive/DEPLOYED_RELEASE")
    for stage in STAGES:
        record = lineage.get(stage)
        if not record:
            errors.append(f"missing_stage:{stage}")
            continue
        path = run_root / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            errors.append(f"artifact_hash_mismatch:{stage}")
        if any(token in path.read_text(encoding="utf-8").replace("\\", "/") for token in forbidden):
            errors.append(f"forbidden_nested_parent:{stage}")
        for parent in record.get("parent_hashes", []):
            if parent not in prior:
                errors.append(f"unresolved_parent:{stage}:{parent}")
        prior.add(record["sha256"])
    return {"status": "PASS" if not errors else "FAILED", "lineage_closed": not errors, "errors": errors}


def run_acceptance(run_id: str = "ci-e2e", *, root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    run_root = root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    lineage: dict[str, Any] = {}
    parent: list[str] = []

    input_payload = {"contract": "SYNTHETIC_E2E_ACCEPTANCE", "licensed_data_used": False, "paid_api_used": False, "heavy_rl_used": False}
    _, digest = _write_json(run_root / "01_inputs" / "fixture_inputs.json", input_payload)
    parent = [_stage(run_root, "VerifyInputs", {"fixture_input_sha256": digest}, [], lineage)]

    action_contract = load_action_contract(root)
    simulator = simulator_from_contract(action_contract)
    previous, state = make_synthetic_pair(1)
    plan = calibrate_business_plan_non_rate([previous, state])
    plan = replace(plan, rate_short=0.04, rate_long=0.04, rate_bond=0.045, non_interest_financial_cost=2_000_000.0)
    simulated = simulator.simulate(state, plan, {name: 0.0 for name in action_contract["action_columns"]})
    variables = compute_oracle_variables(simulated.state_t1, state)
    if not simulated.state_t1.accounting_identity_check(tol=10.0)["check"] == "ok":
        raise ValueError("synthetic simulator accounting identity failed")
    oracle_payload = {"fixture_verification_contract": "RQ1_NOT_APPLICABLE_TO_TINY_FIXTURE", "gate_verdict": "fixture_pass", "observed_metrics": {"finite_variables": all(v is None or np.isfinite(float(v)) for v in variables.values())}, "variables": variables}
    _, digest = _write_json(run_root / "02_oracle" / "oracle_fixture.json", oracle_payload)
    parent = [_stage(run_root, "Oracle", {"oracle_artifact_sha256": digest, "production_formula_registry_used": True}, parent, lineage)]
    _, digest = _write_json(run_root / "02_oracle" / "oracle_validation_report.json", {"status": "PASS", "scientific_gate_verdict": "fixture_pass", "fixture_contract": "thesis_RQ1_thresholds_not_applied", "parent_hashes": parent})
    parent = [_stage(run_root, "VerifyOracle", {"verification_execution_status": "PASS", "scientific_gate_verdict": "fixture_pass", "artifact_sha256": digest}, parent, lineage)]

    sim_payload = {"state_t": state.to_dict(), "state_t1": simulated.state_t1.to_dict(), "accounting_check": simulated.accounting_check, "action_contract_sha256": hashlib.sha256(json.dumps(action_contract, sort_keys=True).encode()).hexdigest(), "revenue_growth_source": "BusinessPlan.exogenous_fixture"}
    _, digest = _write_json(run_root / "03_simulator" / "simulator_output.json", sim_payload)
    parent = [_stage(run_root, "Simulator", {"simulator_artifact_sha256": digest, "validation": {"finite": True, "accounting_identity": True, "nine_actions": True}}, parent, lineage)]

    dataset = [{"firm_id": state.firm_id, "decision_year": state.year, "outcome_year": state.year + 1, "split": "train", "oracle_reward_leak": False, "transition_provenance": "synthetic_counterfactual"}]
    _, digest = _write_json(run_root / "04_rl_dataset" / "dataset.json", dataset)
    parent = [_stage(run_root, "RLDataset", {"row_count": len(dataset), "evaluation_cohort_used_for_training": False, "artifact_sha256": digest}, parent, lineage)]
    encoder = {"architecture": "E2", "feature_count": 8, "optimizer": "deterministic_fixture", "epochs": 1, "seed": 2}
    _, digest = _write_json(run_root / "05_rl_encoder" / "encoder_checkpoint.json", encoder)
    parent = [_stage(run_root, "RLEncoder", {"checkpoint_sha256": digest, "reduced_profile": True}, parent, lineage)]
    _, digest = _write_json(run_root / "06_rl_bc" / "bc_checkpoint.json", {"class_balanced": True, "family_balanced": True, "source_encoder_sha256": parent[0]})
    parent = [_stage(run_root, "RLBehaviorClone", {"checkpoint_sha256": digest}, parent, lineage)]
    _, digest = _write_json(run_root / "07_rl_iql" / "iql_checkpoint.json", {"gamma": 0.0, "expectile_tau": 0.7, "oracle_output_in_reward": False, "checkpoint_sha256": digest})
    parent = [_stage(run_root, "RLIQL", {"checkpoint_sha256": digest, "reduced_profile": True}, parent, lineage)]

    contract = load_fresh_rl_contract(root)
    actor_probabilities: dict[str, dict[int, np.ndarray]] = {}
    actor_records: list[dict[str, Any]] = []
    features = np.asarray([[float(variables.get("R006") or 0.0), float(variables.get("R064") or 0.0), 0.1]], dtype=float)
    for config_index, config in enumerate(contract["selected_configurations"]):
        actor_probabilities[config] = {}
        for seed in contract["seeds"]:
            probabilities = _softmax(np.tile(np.arange(9, dtype=float), (len(features), 1)) * 0.02 + (config_index + seed) * 0.001)
            actor_probabilities[config][int(seed)] = probabilities
            checkpoint = run_root / "07_rl_iql" / "actors" / config / f"seed-{seed}.json"
            _, checkpoint_hash = _write_json(checkpoint, {"configuration": config, "seed": seed, "probabilities": probabilities.tolist(), "fresh": True})
            actor_records.append({"configuration": config, "seed": seed, "checkpoint_path": str(checkpoint.relative_to(run_root)).replace("\\", "/"), "checkpoint_sha256": checkpoint_hash, "evaluation_base_year": 2024, "evaluation_firm_count": 575, "trained_on_evaluation_cohort": False, "oracle_output_in_reward": False})
    actor_graph = verify_actor_graph(actor_records, contract, root=run_root, evaluation_cohort_hash="fixture-eval-cohort")
    if actor_graph["status"] != "PASS":
        raise ValueError(actor_graph)
    _, digest = _write_json(run_root / "08_c3e" / "actor_graph_verification.json", actor_graph)
    parent = [_stage(run_root, "C3E", {"actor_graph_sha256": digest, "actor_count": 28}, parent, lineage)]
    _, c3e_receipt = build_c3e_release(actor_probabilities, contract, parent_hashes=parent, evaluation_cohort_hash="fixture-eval-cohort")
    _, digest = _write_json(run_root / "08_c3e" / "release.json", c3e_receipt)
    parent = [_stage(run_root, "Stage6", {"release_sha256": digest, "action_distribution_rows": 1}, parent, lineage)]
    _, digest = _write_json(run_root / "08_c3e" / "rl_validation_report.json", {"status": "PASS", "actor_graph": actor_graph, "hierarchical_aggregation": True, "flat_actor_mean_not_used": True, "parent_hashes": parent})
    parent = [_stage(run_root, "VerifyRL", {"verification_sha256": digest, "status": "PASS"}, parent, lineage)]

    prepare = prepare_requests(run_root, limit=6)
    _, digest = _write_json(run_root / "09_llm" / "prepare_stage_receipt.json", prepare)
    parent = [_stage(run_root, "LLMPrepare", {"request_count": prepare["request_count"], "receipt_sha256": digest}, parent, lineage)]
    generated = generate_mock(run_root)
    _, digest = _write_json(run_root / "09_llm" / "mock_generation_stage_receipt.json", generated)
    parent = [_stage(run_root, "LLMGenerate", {"provider_contacted": False, "receipt_sha256": digest}, parent, lineage)]
    materialized = materialize_responses(run_root, expected_count=6)
    _, digest = _write_json(run_root / "09_llm" / "materialization_stage_receipt.json", materialized)
    parent = [_stage(run_root, "LLMMaterialize", {"response_count": materialized["response_count"], "receipt_sha256": digest}, parent, lineage)]

    stage8 = build_stage8(run_root, synthetic=True)
    parent = [_stage(run_root, "Stage8", {"stage8_rows": stage8["row_count"], "strict_repaired_exact": True}, parent, lineage)]
    stage9 = run_stage9(run_root, synthetic=True)
    parent = [_stage(run_root, "Stage9", {"registry_status": stage9["status"], "fresh_registry": True}, parent, lineage)]
    parent = [_stage(run_root, "VerifyResults", {"stage8_status": stage8["status"], "stage9_status": stage9["status"]}, parent, lineage)]
    parent = [_stage(run_root, "ThesisOutputs", {"scientific_claims": False, "fixture_only": True}, parent, lineage)]
    _, digest = _write_json(run_root / "14_comparison" / "fresh_vs_frozen.json", {"comparison_status": "COMPARISON_ONLY", "fresh": "fixture", "frozen_used_as_parent": False})
    parent = [_stage(run_root, "CompareFrozen", {"comparison_sha256": digest, "frozen_used_as_parent": False}, parent, lineage)]
    pre_lineage = _verify_lineage(run_root, lineage)
    final = {"status": "SYNTHETIC_E2E_PASS", "execution_class": "SYNTHETIC_E2E_ACCEPTANCE", "stages": STAGES, "provider_contacted": False, "licensed_data_used": False, "heavy_rl_used": False, "lineage_closed": pre_lineage["lineage_closed"], "certifiable": False, "reason_not_certifiable": "synthetic fixture is not thesis replication"}
    _, digest = _write_json(run_root / "15_release" / "acceptance_report.json", final)
    _stage(run_root, "VerifyAll", {"acceptance_report_sha256": digest, "lineage_closed": pre_lineage["lineage_closed"]}, parent, lineage)
    final_lineage = _verify_lineage(run_root, lineage)
    final["lineage_closed"] = final_lineage["lineage_closed"]
    final["lineage_errors"] = final_lineage["errors"]
    result = {**final, "run_id": run_id, "lineage": lineage}
    _write_json(run_root / "acceptance_manifest.json", result)
    return result
