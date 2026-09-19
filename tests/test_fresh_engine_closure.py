from __future__ import annotations

import json
from pathlib import Path

from credit_recourse.contracts.v43_action_contract import ACTION_DIMENSIONS, ACTION_IDS, load_action_contract
from thesis_repro.contracts import EXPECTED_REQUESTS, render_requests
from thesis_repro.live_llm import gate_status, mock_responses
from thesis_repro.runtime_paths import FreshRuntimePaths
from thesis_repro.stages.base import StageResult
from thesis_repro.stages.adapters import HEAVY_GATE, LIVE_GATE

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


def test_stage_result_defaults_to_implemented_not_executed():
    result = StageResult("Stage8", "PASS_WITH_SKIPS", "REAL_COMPUTE")
    assert result.implemented is True and result.executed is False


def test_audit_missing_module_count_is_zero():
    payload = json.loads((ROOT / "runs/_runtime_audit/missing_internal_modules.json").read_text(encoding="utf-8"))
    assert payload["missing"] == []


def test_audit_records_required_raw_inputs():
    assert (ROOT / "runs/_runtime_audit/required_raw_inputs.json").is_file()


def test_audit_records_rebinding_plan():
    assert (ROOT / "runs/_runtime_audit/runtime_rebinding_plan.json").is_file()


def test_full_no_input_acceptance_receipt():
    manifest = json.loads((ROOT / "runs/mission-full-check/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["completion_state"] == "INPUT_REQUIRED"
    assert manifest["stage_status"]["VerifyInputs"] == "INPUT_REQUIRED"


def test_full_no_input_does_not_claim_scientific_execution():
    manifest = json.loads((ROOT / "runs/mission-full-check/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_class"] == "fresh_replication"
    assert list(manifest["stage_status"]) == ["VerifyInputs"]


def test_smoke_artifacts_are_marked_contract_smoke_only():
    manifest = json.loads((ROOT / "runs/mission-smoke-check/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_class"] == "contract_smoke"


def test_mock_materialization_does_not_require_network():
    rows = render_requests(limit=2, run_id="mock-closure")
    assert mock_responses(rows, ROOT / "runs/mock-closure-responses.jsonl")["execution_class"] == "mock_only"


def test_capabilities_do_not_claim_full_execution():
    payload = json.loads((ROOT / "capabilities.json").read_text(encoding="utf-8"))
    assert "UNEXECUTED" in payload["FullClean"]
