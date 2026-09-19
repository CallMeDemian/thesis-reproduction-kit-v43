from __future__ import annotations

import os

from thesis_repro.contracts import EXPECTED_REQUESTS
from thesis_repro.runtime_paths import FreshRuntimePaths
from .base import StageResult
from .oracle import OracleAdapter, verify_oracle_stage

HEAVY_GATE = "I_APPROVE_28_ACTOR_RETRAIN"
LIVE_GATE = "I_APPROVE_FRESH_REPLICATION"


def _unexecuted(stage: str, parents: list[str], reason: str, *, status: str = "IMPLEMENTED_UNEXECUTED") -> StageResult:
    return StageResult(stage, status, "NOT_EXECUTED", executed=False, parent_hashes=parents, details={"reason": reason})


def run_real_stage(paths: FreshRuntimePaths, stage: str, parent_hashes: list[str], *, execute_llm=False) -> StageResult:
    if stage == "Oracle":
        return OracleAdapter().run(paths, paths.root, parent_hashes)
    if stage == "VerifyOracle":
        return verify_oracle_stage(paths, parent_hashes)
    if stage in {"RLDataset", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL", "Simulator"}:
        return _unexecuted(stage, parent_hashes, "downstream scientific implementation is not wired into this push")
    if stage == "LLMPrepare":
        return _unexecuted(stage, parent_hashes, f"fresh logical request preparation is implemented but downstream transport is not wired; expected_requests={EXPECTED_REQUESTS}")
    if stage == "LLMGenerate":
        if execute_llm and os.environ.get("THESIS_REPRO_ENABLE_LIVE_LLM") == LIVE_GATE:
            return _unexecuted(stage, parent_hashes, "live submit_wave dispatch is not wired into the DAG", status="LIVE_LLM_APPROVAL_REQUIRED")
        return _unexecuted(stage, parent_hashes, "provider submission has not been invoked", status="LIVE_LLM_APPROVAL_REQUIRED")
    if stage in {"LLMMaterialize", "Stage8", "Stage9", "ThesisOutputs", "CompareFrozen", "VerifyAll"}:
        return _unexecuted(stage, parent_hashes, "upstream scientific artifacts are not available")
    return _unexecuted(stage, parent_hashes, "stage adapter is declared but not wired")


def run_heavy_gate(paths: FreshRuntimePaths, parent_hashes: list[str]) -> StageResult:
    authorized = os.environ.get("THESIS_REPRO_ENABLE_HEAVY_RL") == HEAVY_GATE
    return _unexecuted("RLHeavy", parent_hashes, "28-actor training dispatch is not implemented; gate authorization alone cannot claim execution", status="IMPLEMENTED_UNEXECUTED" if authorized else "HEAVY_EXECUTION_APPROVAL_REQUIRED")
