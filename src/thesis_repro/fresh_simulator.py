"""Run-local adapter for the canonical V4.3 semantic simulator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from credit_recourse.contracts.v43_action_contract import load_action_contract
from credit_recourse.rl.common.semantic_action_v4_1_contract import simulator_from_contract
from credit_recourse.simulator.business_plan import BusinessPlan
from credit_recourse.simulator.firm_state import FirmState
from credit_recourse.simulator.oracle_variables import compute_oracle_variables

from .stages.base import StageResult, write_stage_artifact


def _state(payload: dict[str, Any]) -> FirmState:
    return FirmState(**payload)


def run_fresh_simulator(paths, parent_hashes: list[str]) -> StageResult:
    validation_parent = paths.oracle_root / "oracle_validation_report.json"
    input_path = paths.stage2_root / "fresh_simulator_input.json"
    if not validation_parent.is_file():
        return StageResult("Simulator", "INPUT_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "same-run VerifyOracle artifact is required"})
    if not input_path.is_file():
        return StageResult("Simulator", "INPUT_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "authorized run-local simulator input is missing; frozen historical state is not accepted", "expected": str(input_path.relative_to(paths.root)).replace("\\", "/")})
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if payload.get("parent_oracle_sha256") not in parent_hashes:
            raise ValueError("simulator input is not bound to the same-run VerifyOracle manifest")
        contract = load_action_contract(paths.root)
        simulator = simulator_from_contract(contract)
        state_t = _state(payload["state_t"])
        plan = BusinessPlan(**payload["business_plan"])
        action = payload["action"]
        result = simulator.simulate(state_t, plan, action)
        state_t1 = result.state_t1
        accounting = state_t1.accounting_identity_check(tol=10.0)
        variables = compute_oracle_variables(state_t1, state_t)
        finite = all(value is None or math.isfinite(float(value)) for value in variables.values())
        if accounting.get("check") != "ok" or not finite:
            raise ValueError(f"simulator validation failed: accounting={accounting}, finite={finite}")
        output = {"state_t1": state_t1.to_dict(), "accounting_check": accounting, "oracle_variables": variables, "action_contract_hash": contract["candidate_action_contract_hash"], "parent_hashes": parent_hashes}
        receipt = {"schema_version": "fresh_simulator_execution_receipt_v1", "execution_class": "REAL_COMPUTE", "executed": True, "canonical_factory": "simulator_from_contract", "revenue_growth_source": "BusinessPlan/scenario", "parent_hashes": parent_hashes}
        report = {"schema_version": "fresh_simulator_validation_report_v1", "status": "PASS", "row_count": 1, "row_key_unique": True, "accounting_identity": accounting, "finite_oracle_variables": finite, "action_contract_identity": contract["candidate_action_contract_hash"], "forbidden_parent_path": False, "input_cardinality": 1, "output_cardinality": 1, "deterministic_fixed_fixture": payload.get("deterministic_fixture", False)}
        artifacts = [write_stage_artifact(paths, "03_simulator/simulator_output.json", output, "fresh:simulator:output", ({"sha256": h} for h in parent_hashes)), write_stage_artifact(paths, "03_simulator/simulator_execution_receipt.json", receipt, "fresh:simulator:receipt", ({"sha256": h} for h in parent_hashes)), write_stage_artifact(paths, "03_simulator/simulator_validation_report.json", report, "fresh:simulator:validation", ({"sha256": h} for h in parent_hashes))]
        return StageResult("Simulator", "PASS", "REAL_COMPUTE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"production_factory": "simulator_from_contract", "validation": report})
    except Exception as exc:
        return StageResult("Simulator", "FAILED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"error": repr(exc)})
