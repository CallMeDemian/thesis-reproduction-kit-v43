from __future__ import annotations

import os

from thesis_repro.contracts import EXPECTED_REQUESTS
from thesis_repro.runtime_paths import FreshRuntimePaths
from thesis_repro.status import APPROVAL_REQUIRED, NOT_EXECUTED, NOT_IMPLEMENTED
from .base import StageResult
from .oracle import OracleAdapter, verify_oracle_stage
from thesis_repro.fresh_simulator import run_fresh_simulator

HEAVY_GATE = "I_APPROVE_28_ACTOR_RETRAIN"
LIVE_GATE = "I_APPROVE_FRESH_REPLICATION"


def _unexecuted(stage: str, parents: list[str], reason: str, *, status: str = NOT_EXECUTED) -> StageResult:
    return StageResult(stage, status, "NOT_EXECUTED", executed=False, parent_hashes=parents, details={"reason": reason})


def _unimplemented(stage: str, parents: list[str], reason: str) -> StageResult:
    return StageResult(stage, NOT_IMPLEMENTED, "NOT_IMPLEMENTED", implemented=False, executed=False, parent_hashes=parents, details={"reason": reason})


def run_real_stage(paths: FreshRuntimePaths, stage: str, parent_hashes: list[str], *, execute_llm=False) -> StageResult:
    if stage == "Oracle":
        return OracleAdapter().run(paths, paths.root, parent_hashes)
    if stage == "VerifyOracle":
        return verify_oracle_stage(paths, parent_hashes)
    if stage == "Simulator":
        return run_fresh_simulator(paths, parent_hashes)
    if stage in {"RLDataset", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL"}:
        return _unimplemented(stage, parent_hashes, "production downstream scientific implementation is not wired into the fresh DAG")
    if stage == "LLMPrepare":
        return _unimplemented(stage, parent_hashes, f"fresh logical request preparation is implemented separately but not wired into full DAG dispatch; expected_requests={EXPECTED_REQUESTS}")
    if stage == "LLMGenerate":
        if execute_llm and os.environ.get("THESIS_REPRO_ENABLE_LIVE_LLM") == LIVE_GATE:
            return _unexecuted(stage, parent_hashes, "live submit_wave dispatch is not wired into the DAG", status=APPROVAL_REQUIRED)
        return _unexecuted(stage, parent_hashes, "provider submission has not been invoked", status=APPROVAL_REQUIRED)
    if stage in {"LLMMaterialize", "Stage8", "Stage9", "ThesisOutputs", "CompareFrozen", "VerifyAll"}:
        return _unimplemented(stage, parent_hashes, "upstream scientific artifacts are not available")
    return _unimplemented(stage, parent_hashes, "stage adapter is declared but not wired")


def run_heavy_gate(paths: FreshRuntimePaths, parent_hashes: list[str]) -> StageResult:
    authorized = os.environ.get("THESIS_REPRO_ENABLE_HEAVY_RL") == HEAVY_GATE
    if not authorized:
        return _unexecuted("RLExecutionGate", parent_hashes, "28-actor fresh RL training requires explicit approval before any heavy compute", status=APPROVAL_REQUIRED)
    return _unimplemented("RLExecutionGate", parent_hashes, "approval is present, but the 28-actor training dispatch is not implemented")
