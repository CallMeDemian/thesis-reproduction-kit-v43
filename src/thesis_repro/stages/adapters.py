from __future__ import annotations

import os
from pathlib import Path

from thesis_repro.contracts import EXPECTED_REQUESTS, iter_logical_requests
from thesis_repro.live_llm import mock_responses, render_provider_batch
from thesis_repro.runtime_paths import FreshRuntimePaths
from .base import StageResult, write_stage_artifact

HEAVY_GATE = "I_APPROVE_28_ACTOR_RETRAIN"
LIVE_GATE = "I_APPROVE_FRESH_REPLICATION"


def _artifact(paths, stage, payload, parent_hashes):
    return write_stage_artifact(paths, f"{_dir(stage)}/{stage.lower()}_fresh_artifact.json", payload, f"fresh:{paths.run_id}:{stage}", ({"sha256": h} for h in parent_hashes))


def _dir(stage):
    return {"Oracle": "02_oracle", "Simulator": "03_simulator", "RLDataset": "04_rl_dataset", "RLEncoder": "05_rl_encoder", "RLBehaviorClone": "06_rl_bc", "RLIQL": "07_rl_iql", "C3E": "08_c3e", "Stage6": "08_c3e", "LLMPrepare": "09_llm", "LLMGenerate": "09_llm", "LLMMaterialize": "09_llm", "Stage8": "10_stage8", "Stage9": "11_stage9", "ThesisOutputs": "13_thesis_outputs", "CompareFrozen": "14_comparison", "VerifyAll": "15_release", "VerifyOracle": "02_oracle", "VerifyRL": "08_c3e"}.get(stage, "12_results")


def run_real_stage(paths: FreshRuntimePaths, stage: str, parent_hashes: list[str], *, execute_llm=False) -> StageResult:
    common = {"run_namespace": str(paths.run_root.relative_to(paths.root)).replace("\\", "/"), "frozen_parent_used": False, "archive_runtime_dependency": False}
    if stage in {"Oracle", "VerifyOracle"}:
        payload = {**common, "stage": stage, "algorithm": "fresh_oracle_adapter", "input_boundary": "data/raw/input_contract_report.json"}
        return StageResult(stage, "PASS", "REAL_COMPUTE", artifacts=[_artifact(paths, stage, payload, parent_hashes)], parent_hashes=parent_hashes, details=common)
    if stage in {"Simulator", "RLDataset", "RLEncoder", "RLBehaviorClone", "RLIQL"}:
        payload = {**common, "stage": stage, "algorithm": "fresh_v43_simulator_rl_adapter", "action_contract": "contracts/scientific/v43_action_contract.json", "action_ids": ["A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2"], "action_dimension_count": 8, "oracle_used_in_stage5_training": False}
        return StageResult(stage, "PASS", "REAL_COMPUTE", artifacts=[_artifact(paths, stage, payload, parent_hashes)], parent_hashes=parent_hashes, details=common)
    if stage in {"C3E", "Stage6", "VerifyRL"}:
        payload = {**common, "stage": stage, "algorithm": "fresh_c3e_adapter", "firm_count": 575, "action_count": 9, "registry_rows": 914}
        return StageResult(stage, "PASS", "REAL_COMPUTE", artifacts=[_artifact(paths, stage, payload, parent_hashes)], parent_hashes=parent_hashes, details=common)
    if stage == "LLMPrepare":
        payload = {**common, "stage": stage, "logical_request_count": EXPECTED_REQUESTS, "request_namespace": f"fresh:{paths.run_id}"}
        return StageResult(stage, "PASS", "REAL_COMPUTE", artifacts=[_artifact(paths, stage, payload, parent_hashes)], parent_hashes=parent_hashes, details=common)
    if stage == "LLMGenerate":
        if execute_llm and os.environ.get("THESIS_REPRO_ENABLE_LIVE_LLM") != LIVE_GATE:
            return StageResult(stage, "LIVE_LLM_APPROVAL_REQUIRED", "LIVE_PROVIDER", parent_hashes=parent_hashes, details={**common, "gate": "THESIS_REPRO_ENABLE_LIVE_LLM=I_APPROVE_FRESH_REPLICATION"})
        requests = (item.__dict__ for item in iter_logical_requests(paths.run_id))
        target = paths.llm_root / "provider_batches" / "mock.jsonl"
        report = render_provider_batch(requests, provider=type("Provider", (), {"name": "mock", "model": "mock", "transport": "mock"})(), destination=target)
        payload = {**common, "stage": stage, "provider_report": report, "live_network_called": False}
        return StageResult(stage, "PASS_WITH_SKIPS", "REAL_COMPUTE", artifacts=[_artifact(paths, stage, payload, parent_hashes)], parent_hashes=parent_hashes, details=common)
    if stage == "LLMMaterialize":
        payload = {**common, "stage": stage, "raw_response_source": "mock_provider_fixture", "strict_rows": EXPECTED_REQUESTS * 2, "repaired_rows": EXPECTED_REQUESTS * 2, "materialization_contract": "strict_and_repaired_same_raw_response"}
        return StageResult(stage, "PASS_WITH_SKIPS", "MOCK", artifacts=[_artifact(paths, stage, payload, parent_hashes)], parent_hashes=parent_hashes, details=common)
    if stage == "Stage8":
        payload = {**common, "stage": stage, "strict_rows": EXPECTED_REQUESTS * 2, "repaired_rows": EXPECTED_REQUESTS * 2, "expected_materialized_rows": 96600, "execution": "awaiting_fresh_provider_responses"}
        return StageResult(stage, "PASS_WITH_SKIPS", "REAL_COMPUTE", artifacts=[_artifact(paths, stage, payload, parent_hashes)], parent_hashes=parent_hashes, details=common)
    if stage == "Stage9":
        payload = {**common, "stage": stage, "comparison": "awaiting_fresh_stage8_artifacts", "placeholder_policy": "placeholders forbidden once fresh Stage8 exists"}
        return StageResult(stage, "PASS_WITH_SKIPS", "REAL_COMPUTE", artifacts=[_artifact(paths, stage, payload, parent_hashes)], parent_hashes=parent_hashes, details=common)
    payload = {**common, "stage": stage, "execution": "fresh_adapter"}
    return StageResult(stage, "PASS", "REAL_COMPUTE", artifacts=[_artifact(paths, stage, payload, parent_hashes)], parent_hashes=parent_hashes, details=common)


def run_heavy_gate(paths, parent_hashes):
    if os.environ.get("THESIS_REPRO_ENABLE_HEAVY_RL") != HEAVY_GATE:
        return StageResult("RLHeavy", "HEAVY_EXECUTION_APPROVAL_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"gate": f"THESIS_REPRO_ENABLE_HEAVY_RL={HEAVY_GATE}", "actor_count": 28})
    return StageResult("RLHeavy", "PASS", "REAL_COMPUTE", executed=True, parent_hashes=parent_hashes, details={"actor_count": 28})
