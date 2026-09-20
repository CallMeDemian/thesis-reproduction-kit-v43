from __future__ import annotations

import os

from thesis_repro.contracts import EXPECTED_REQUESTS
from thesis_repro.runtime_paths import FreshRuntimePaths
from thesis_repro.status import APPROVAL_REQUIRED, NOT_EXECUTED, NOT_IMPLEMENTED
from thesis_repro.status import CREDENTIALS_REQUIRED, FAILED, INPUT_REQUIRED, PASS
from .base import StageResult, sha256_file, write_stage_artifact
from .oracle import OracleAdapter, verify_oracle_stage
from thesis_repro.fresh_simulator import run_fresh_simulator

HEAVY_GATE = "I_APPROVE_28_ACTOR_RETRAIN"
LIVE_GATE = "I_APPROVE_FRESH_REPLICATION"


def _unexecuted(stage: str, parents: list[str], reason: str, *, status: str = NOT_EXECUTED) -> StageResult:
    return StageResult(stage, status, "NOT_EXECUTED", executed=False, parent_hashes=parents, details={"reason": reason})


def _unimplemented(stage: str, parents: list[str], reason: str) -> StageResult:
    return StageResult(stage, NOT_IMPLEMENTED, "NOT_IMPLEMENTED", implemented=False, executed=False, parent_hashes=parents, details={"reason": reason})


def run_real_stage(paths: FreshRuntimePaths, stage: str, parent_hashes: list[str], *, execute_llm=False, context=None) -> StageResult:
    if stage == "Oracle":
        return OracleAdapter().run(paths, paths.root, parent_hashes, context=context)
    if stage == "VerifyOracle":
        return verify_oracle_stage(paths, parent_hashes, context=context)
    if stage == "Simulator":
        return run_fresh_simulator(paths, parent_hashes, context=context)
    if stage == "RLDataset":
        from thesis_repro.fresh_rl_dataset import run_fresh_rl_dataset
        return run_fresh_rl_dataset(paths, parent_hashes, context=context)
    if stage in {"RLEncoder", "RLBehaviorClone", "RLIQL"}:
        from thesis_repro.fresh_rl_runtime import run_training_stage
        return run_training_stage(paths, stage, parent_hashes, context=context)
    if stage == "C3E":
        from thesis_repro.fresh_rl_runtime import run_c3e
        return run_c3e(paths, parent_hashes, context=context)
    if stage == "Stage6":
        return StageResult(stage, INPUT_REQUIRED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "the preserved repository contains Stage6 artifacts and consumers but no authoritative fresh Stage6 production producer; frozen Stage6 output is forbidden as a compute parent", "blocker_class": "ENGINEERING", "frozen_compute_fallback": False})
    if stage == "VerifyRL":
        from thesis_repro.fresh_rl_runtime import run_verify_rl
        return run_verify_rl(paths, parent_hashes, context=context)
    if stage == "LLMPrepare":
        from thesis_repro.fresh_llm import prepare_requests
        release = paths.c3e_root / "release.json"
        if not release.is_file():
            return StageResult(stage, INPUT_REQUIRED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "same-run C3-E release is required before request preparation"})
        try:
            report = prepare_requests(paths.run_root)
            if report.get("status") != "PASS":
                return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"report": report})
            artifacts = [
                {"logical_id": "fresh:llm:logical_requests", "path": str((paths.llm_root / "logical_requests.jsonl").relative_to(paths.root)).replace("\\", "/"), "sha256": sha256_file(paths.llm_root / "logical_requests.jsonl"), "size_bytes": (paths.llm_root / "logical_requests.jsonl").stat().st_size, "producer": "thesis_repro.fresh_llm.prepare_requests", "parents": [{"sha256": value} for value in parent_hashes]},
                write_stage_artifact(paths, "09_llm/prepare_report.json", report, "fresh:llm:prepare", ({"sha256": value} for value in parent_hashes)),
            ]
            return StageResult(stage, PASS, "REAL_COMPUTE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"report": report, "provider_identities": {"contract": "contracts/llm/fresh_replication_contract.json"}})
        except Exception as exc:
            return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "LLM_PREPARATION_FAILED"})
    if stage == "LLMGenerate":
        if context is not None and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE":
            from thesis_repro.fresh_llm import generate_mock
            try:
                receipt = generate_mock(paths.run_root)
                artifact = write_stage_artifact(paths, "09_llm/generate_receipt.json", receipt, "fresh:llm:mock_receipt", ({"sha256": value} for value in parent_hashes))
                return StageResult(stage, PASS, "SYNTHETIC_E2E_ACCEPTANCE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details={"provider_contacted": False, "mock_only": True, "receipt": receipt})
            except Exception as exc:
                return StageResult(stage, FAILED, "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "LLM_MOCK_EXECUTION_FAILED"})
        if execute_llm and os.environ.get("THESIS_REPRO_ENABLE_LIVE_LLM") == LIVE_GATE:
            credentials = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
            if not credentials:
                return _unexecuted(stage, parent_hashes, "live provider credentials are required before submission", status=CREDENTIALS_REQUIRED)
            return _unexecuted(stage, parent_hashes, "provider-native live submission requires an authorized final release and is intentionally not invoked by ordinary execution", status=APPROVAL_REQUIRED)
        return _unexecuted(stage, parent_hashes, "provider submission has not been invoked", status=APPROVAL_REQUIRED)
    if stage == "LLMMaterialize":
        from thesis_repro.fresh_llm import materialize_responses
        try:
            report = materialize_responses(paths.run_root, expected_count=EXPECTED_REQUESTS)
            if report.get("status") != "PASS":
                return StageResult(stage, INPUT_REQUIRED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"report": report, "reason": "complete provider response set is required"})
            if report.get("execution_class") == "MOCK_ONLY" and not (context and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE"):
                return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "mock responses cannot satisfy fresh scientific materialization"})
            artifact = write_stage_artifact(paths, "09_llm/materialization_report.json", report, "fresh:llm:materialization", ({"sha256": value} for value in parent_hashes))
            return StageResult(stage, PASS, "SYNTHETIC_E2E_ACCEPTANCE" if context and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE" else "REAL_COMPUTE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details={"report": report})
        except Exception as exc:
            return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "LLM_MATERIALIZATION_FAILED"})
    if stage in {"Stage8", "Stage9"}:
        try:
            import os as _os
            previous = _os.environ.get("THESIS_REPRO_RUN_ROOT")
            _os.environ["THESIS_REPRO_RUN_ROOT"] = str(paths.run_root)
            try:
                if stage == "Stage8":
                    from credit_recourse.eval.final_stage8_llm_multi_oracle_eval.pipeline import run_stage8
                    payload = run_stage8(project_root=paths.root)
                    destination = paths.stage8_root / "metadata.json"
                else:
                    from credit_recourse.eval.final_stage9_llm_rl_comparison.pipeline import run_stage9
                    payload = run_stage9(project_root=paths.root)
                    destination = paths.stage9_root / "metadata.json"
            finally:
                if previous is None:
                    _os.environ.pop("THESIS_REPRO_RUN_ROOT", None)
                else:
                    _os.environ["THESIS_REPRO_RUN_ROOT"] = previous
            if payload.get("status") != "PASS":
                return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"payload": payload})
            artifact = {"logical_id": f"fresh:{stage}:metadata", "path": str(destination.relative_to(paths.root)).replace("\\", "/"), "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size, "producer": f"credit_recourse.eval.final_{stage.lower()}", "parents": [{"sha256": value} for value in parent_hashes]}
            return StageResult(stage, PASS, "REAL_COMPUTE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details={"payload": payload})
        except FileNotFoundError as exc:
            return StageResult(stage, INPUT_REQUIRED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "frozen_compute_fallback": False})
        except Exception as exc:
            return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": f"{stage.upper()}_EXECUTION_FAILED"})
    if stage in {"ThesisOutputs", "CompareFrozen", "VerifyAll"}:
        return _unimplemented(stage, parent_hashes, "no authoritative final producer is present in the migrated source; historical outputs remain comparison-only")
    return _unimplemented(stage, parent_hashes, "stage adapter is declared but not wired")


def run_heavy_gate(paths: FreshRuntimePaths, parent_hashes: list[str], *, context=None) -> StageResult:
    if context is not None and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE":
        return StageResult("RLExecutionGate", PASS, "SYNTHETIC_E2E_ACCEPTANCE", executed=True, parent_hashes=parent_hashes, details={"approval_required_for": "28_actor_fresh_replication", "synthetic_reduced_training_allowed": False, "gate_checked_before_training": True})
    authorized = os.environ.get("THESIS_REPRO_ENABLE_HEAVY_RL") == HEAVY_GATE
    if not authorized:
        return StageResult("RLExecutionGate", APPROVAL_REQUIRED, "CONTROL", executed=False, parent_hashes=parent_hashes, details={"reason": "28-actor fresh RL training requires explicit approval before any heavy compute", "gate_checked_before_training": True})
    return StageResult("RLExecutionGate", PASS, "CONTROL", executed=True, parent_hashes=parent_hashes, details={"approval_required_for": "28_actor_fresh_replication", "gate_checked_before_training": True})
