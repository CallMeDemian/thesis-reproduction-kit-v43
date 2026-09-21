from __future__ import annotations

import os
import json

import pandas as pd

from thesis_repro.llm_runtime import EXPECTED_REQUESTS, execute_full_llm, materialize_full_stage7, prepare_full_llm, prepare_real_llm, execute_real_llm, materialize_real_stage7
from thesis_repro.runtime_paths import FreshRuntimePaths
from thesis_repro.status import APPROVAL_REQUIRED
from thesis_repro.status import CREDENTIALS_REQUIRED, FAILED, INPUT_REQUIRED, PASS
from .base import StageResult, sha256_file, write_stage_artifact
from .oracle import OracleAdapter, verify_oracle_stage
from thesis_repro.stage2 import run_stage2

def _blocked(stage: str, parents: list[str], reason: str, *, status: str = INPUT_REQUIRED) -> StageResult:
    return StageResult(stage, status, "REAL_COMPUTE", executed=False, parent_hashes=parents, details={"reason": reason})


def run_real_stage(paths: FreshRuntimePaths, stage: str, parent_hashes: list[str], *, execute_llm=False, context=None) -> StageResult:
    if stage == "Oracle":
        return OracleAdapter().run(paths, paths.root, parent_hashes, context=context)
    if stage == "VerifyOracle":
        return verify_oracle_stage(paths, parent_hashes, context=context)
    if stage == "Stage2":
        return run_stage2(paths, parent_hashes, context=context)
    if stage in {"RLEncoder", "RLBehaviorClone", "RLIQL"}:
        from thesis_repro.fresh_rl_runtime import run_training_stage
        return run_training_stage(paths, stage, parent_hashes, context=context)
    if stage == "C3E":
        from thesis_repro.fresh_rl_runtime import run_c3e
        return run_c3e(paths, parent_hashes, context=context)
    if stage == "Stage6":
        from thesis_repro.stage6 import run_stage6
        return run_stage6(paths, parent_hashes, context=context)
    if stage == "VerifyRL":
        from thesis_repro.fresh_rl_runtime import run_verify_rl
        return run_verify_rl(paths, parent_hashes, context=context)
    if stage == "LLMPrepare":
        release = paths.c3e_root / "release.json"
        if not release.is_file():
            return StageResult(stage, INPUT_REQUIRED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "same-run C3-E release is required before request preparation"})
        try:
            if context and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE":
                firm_count = 8
                report = prepare_full_llm(paths, firm_count=firm_count)
            else:
                report = prepare_real_llm(paths)
            expected_status = "PASS" if context and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE" else "PREPARED"
            if report.get("status") != expected_status:
                status = INPUT_REQUIRED if report.get("status") == "INPUT_REQUIRED" else FAILED
                return StageResult(stage, status, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"report": report})
            artifacts = [write_stage_artifact(paths, "09_llm/prepare_report.json", report, "fresh:llm:prepare", ({"sha256": value} for value in parent_hashes))]
            if expected_status == "PASS":
                request_path = paths.llm_root / "logical_requests.jsonl"
                artifacts.append({"logical_id": "fresh:llm:logical_requests", "path": str(request_path.relative_to(paths.root)).replace("\\", "/"), "sha256": sha256_file(request_path), "size_bytes": request_path.stat().st_size, "producer": "thesis_repro.llm_runtime.prepare_full_llm", "parents": [{"sha256": value} for value in parent_hashes]})
            else:
                from credit_recourse.final_release.executor import output_root
                for label in ("baseline_hash", "high_hash"):
                    request_path = output_root(paths.root) / str(report[label]) / "logical_requests.parquet"
                    if not request_path.is_file():
                        raise FileNotFoundError(request_path)
                    artifacts.append({"logical_id": f"fresh:llm:{label}:logical_requests", "path": str(request_path.relative_to(paths.root)).replace("\\", "/"), "sha256": sha256_file(request_path), "size_bytes": request_path.stat().st_size, "producer": "credit_recourse.final_release.executor", "parents": [{"sha256": value} for value in parent_hashes]})
            return StageResult(stage, PASS, "REAL_COMPUTE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"report": report, "provider_identities": {"contract": "thesis_repro.llm_runtime"}})
        except Exception as exc:
            return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "LLM_PREPARATION_FAILED"})
    if stage == "LLMGenerate":
        if context is not None and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE":
            try:
                receipt = execute_full_llm(paths, live=False, firm_count=8)
                artifact = write_stage_artifact(paths, "09_llm/generate_receipt.json", receipt, "fresh:llm:mock_receipt", ({"sha256": value} for value in parent_hashes))
                return StageResult(stage, PASS, "SYNTHETIC_E2E_ACCEPTANCE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details={"provider_contacted": False, "mock_only": True, "receipt": receipt})
            except Exception as exc:
                return StageResult(stage, FAILED, "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "LLM_MOCK_EXECUTION_FAILED"})
        if not execute_llm:
            return _blocked(stage, parent_hashes, "live provider transport was not requested; credentials/provider approval are required", status=CREDENTIALS_REQUIRED)
        credentials = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        if not credentials:
            return _blocked(stage, parent_hashes, "live provider credentials are required before submission", status=CREDENTIALS_REQUIRED)
        try:
            receipt = execute_real_llm(paths, resume=True)
            status = receipt.get("status")
            if status == "PASS":
                artifact = write_stage_artifact(paths, "09_llm/generate_receipt.json", receipt, "fresh:llm:live_receipt", ({"sha256": value} for value in parent_hashes))
                return StageResult(stage, PASS, "REAL_COMPUTE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details=receipt)
            if status in {"APPROVAL_REQUIRED", "CREDENTIALS_REQUIRED", "EXTERNAL_WAIT"}:
                return StageResult(stage, status, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details=receipt)
            return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details=receipt)
        except Exception as exc:
            return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "LIVE_LLM_EXECUTION_FAILED"})
    if stage == "LLMMaterialize":
        try:
            if context and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE":
                expected_count = 42 * 2 * 8
                report = materialize_full_stage7(paths, expected_count=expected_count)
            else:
                report = materialize_real_stage7(paths)
            if report.get("status") != "PASS":
                return StageResult(stage, INPUT_REQUIRED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"report": report, "reason": "complete provider response set is required"})
            if report.get("execution_class") == "SYNTHETIC_E2E_ACCEPTANCE" and not (context and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE"):
                return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "mock responses cannot satisfy fresh scientific materialization"})
            artifact = write_stage_artifact(paths, "09_llm/materialization_report.json", report, "fresh:llm:materialization", ({"sha256": value} for value in parent_hashes))
            return StageResult(stage, PASS, "SYNTHETIC_E2E_ACCEPTANCE" if context and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE" else "REAL_COMPUTE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details={"report": report})
        except Exception as exc:
            return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "LLM_MATERIALIZATION_FAILED"})
    if stage == "VerifyResults":
        stage8_meta = paths.stage8_root / "metadata.json"
        stage9_meta = paths.stage9_root / "metadata.json"
        required = [stage8_meta, stage9_meta]
        missing = [str(path.relative_to(paths.root)).replace("\\", "/") for path in required if not path.is_file()]
        if missing:
            return StageResult(stage, INPUT_REQUIRED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "Stage8/Stage9 metadata is required", "missing": missing})
        stage8 = json.loads(stage8_meta.read_text(encoding="utf-8"))
        stage9 = json.loads(stage9_meta.read_text(encoding="utf-8"))
        if stage8.get("status") != "PASS" or stage9.get("status") != "PASS":
            return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "final scientific metadata did not PASS", "stage8": stage8.get("status"), "stage9": stage9.get("status")})
        fixture = bool(context and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE")
        expected_firms = 8 if fixture else 575
        expected_requests = 672 if fixture else 48300
        expected_payoffs = expected_firms * 9
        try:
            stage6_pointer = json.loads((paths.stage6_root / "CURRENT_RELEASE.json").read_text(encoding="utf-8"))
            payoff_path = paths.root / str(stage6_pointer["payoff_surface_path"])
            payoff = pd.read_parquet(payoff_path)
            c3e = json.loads((paths.c3e_root / "release.json").read_text(encoding="utf-8"))
            c3e_actions = pd.read_parquet(paths.c3e_root / "c3e_actions.parquet")
            llm_meta = json.loads((paths.llm_root / "metadata.json").read_text(encoding="utf-8"))
            primary = pd.read_parquet(paths.stage9_root / "llm_stage9_primary_contrast_firm_level.parquet")
            if len(payoff) != expected_payoffs or payoff[["row_id", "action"]].duplicated().any():
                raise ValueError("Stage6 payoff surface is not a complete cohort x 9 grid")
            c3e_identity = next((column for column in ("row_id", "firm_id", "evaluation_ordinal") if column in c3e_actions.columns), None)
            if len(c3e_actions) != expected_firms or c3e_identity is None or c3e_actions[c3e_identity].nunique() != expected_firms:
                raise ValueError("fresh C3-E action table is not a complete cohort")
            if int(llm_meta.get("total", -1)) != expected_requests:
                raise ValueError("LLM logical inventory does not match the execution profile")
            if set(primary["contrast"].astype(str)) != {"C5-C4", "C4R-C4", "C6-E-C4R", "C6-E-C6-EX"}:
                raise ValueError("Stage9 primary contrast registry is incomplete")
            if set(primary["budget"].astype(str)) != {"B1", "BINF"} or primary["model_key"].nunique() < 1:
                raise ValueError("Stage9 primary results lack budget/model coverage")
        except Exception as exc:
            return StageResult(stage, FAILED, "SYNTHETIC_E2E_ACCEPTANCE" if fixture else "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "FINAL_RESULTS_VERIFICATION_FAILED"})
        report = {"status": "PASS", "stage": "VerifyResults", "stage8_metadata_sha256": sha256_file(stage8_meta), "stage9_metadata_sha256": sha256_file(stage9_meta), "execution_class": context.execution_class if context else "REAL_COMPUTE", "expected_firms": expected_firms, "expected_llm_requests": expected_requests, "stage6_payoff_rows": expected_payoffs, "c3e_rows": int(len(c3e_actions)), "stage9_primary_rows": int(len(primary)), "stage6_release_id": stage6_pointer.get("release_id"), "c3e_release_status": c3e.get("status"), "parents": parent_hashes}
        artifact = write_stage_artifact(paths, "12_results/results_verification.json", report, "fresh:results:verification", ({"sha256": value} for value in parent_hashes))
        return StageResult(stage, PASS, "SYNTHETIC_E2E_ACCEPTANCE" if context and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE" else "REAL_COMPUTE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details=report)
    if stage in {"Stage8", "Stage9"}:
        if context is not None and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE":
            previous_run_root = os.environ.get("THESIS_REPRO_RUN_ROOT")
            previous_stage_path = os.environ.get("CREDIT_REPRO_RUN_PATH")
            os.environ["THESIS_REPRO_RUN_ROOT"] = str(paths.run_root)
            os.environ["CREDIT_REPRO_RUN_PATH"] = str(paths.run_root)
            try:
                if stage == "Stage8":
                    from credit_recourse.eval.final_stage8_llm_multi_oracle_eval.pipeline import run_stage8
                    payload = run_stage8(project_root=paths.root, fixture_mode=True)
                    destination = paths.stage8_root / "metadata.json"
                else:
                    from credit_recourse.eval.final_stage9_llm_rl_comparison.pipeline import run_stage9
                    payload = run_stage9(project_root=paths.root, fixture_mode=True)
                    destination = paths.stage9_root / "metadata.json"
            except FileNotFoundError as exc:
                return StageResult(stage, INPUT_REQUIRED, "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "fixture_producer": "real_final_stage_producer"})
            except Exception as exc:
                return StageResult(stage, FAILED, "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": f"{stage.upper()}_FIXTURE_EXECUTION_FAILED", "fixture_producer": "real_final_stage_producer"})
            finally:
                if previous_run_root is None:
                    os.environ.pop("THESIS_REPRO_RUN_ROOT", None)
                else:
                    os.environ["THESIS_REPRO_RUN_ROOT"] = previous_run_root
                if previous_stage_path is None:
                    os.environ.pop("CREDIT_REPRO_RUN_PATH", None)
                else:
                    os.environ["CREDIT_REPRO_RUN_PATH"] = previous_stage_path
            if payload.get("status") != "PASS" or not destination.is_file():
                return StageResult(stage, FAILED, "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"payload": payload, "fixture_producer": "real_final_stage_producer"})
            artifact = {"logical_id": f"fresh:{stage}:metadata", "path": str(destination.relative_to(paths.root)).replace("\\", "/"), "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size, "producer": f"credit_recourse.eval.final_{stage.lower()}", "parents": [{"sha256": value} for value in parent_hashes]}
            return StageResult(stage, PASS, "SYNTHETIC_E2E_ACCEPTANCE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details={"payload": payload, "fixture_producer": "real_final_stage_producer"})
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
        return _final_stage(paths, stage, parent_hashes, context=context)
    raise ValueError(f"unknown or unwired scientific stage: {stage}")


def _final_stage(paths, stage: str, parent_hashes: list[str], *, context=None) -> StageResult:
    import json
    execution_class = context.execution_class if context else "REAL_COMPUTE"
    try:
        if stage == "ThesisOutputs":
            from scripts.build_fresh_thesis_outputs import build_fresh_thesis_outputs
            payload = build_fresh_thesis_outputs(paths.root, paths.run_id)
            destination = paths.thesis_outputs_root
            path = destination / "OUTPUT_METADATA.json"
        elif stage == "CompareFrozen":
            from thesis_repro.compare import compare_run
            payload = compare_run(paths.run_id, root=paths.root)
            allowed_synthetic = execution_class == "SYNTHETIC_E2E_ACCEPTANCE" and payload.get("comparison_status") == "SYNTHETIC_COMPARISON_NOT_APPLICABLE"
            if payload.get("comparison_status") != "PASS" and not allowed_synthetic:
                raise ValueError(f"fresh-vs-frozen comparison did not complete: {payload.get('comparison_status')}")
            destination = paths.comparison_root
            path = destination / "fresh_vs_frozen.json"
        else:
            destination = paths.release_root
            destination.mkdir(parents=True, exist_ok=True)
            from thesis_repro.certify import certify_run
            payload = certify_run(paths.run_id)
            expected_certified = execution_class == "REAL_COMPUTE"
            if expected_certified and payload.get("certified") is not True:
                raise ValueError(f"fresh certification failed: {payload.get('errors')}")
            if not expected_certified and not (payload.get("certified") is False and payload.get("state") == "NOT_CERTIFIABLE"):
                raise ValueError("synthetic certification did not fail closed as NOT_CERTIFIABLE")
            payload = {**payload, "status": "PASS", "stage": stage, "execution_class": execution_class, "parent_hashes": parent_hashes, "same_run_only": True}
            path = destination / "verification.json"
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        return StageResult(stage, FAILED, execution_class, executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": f"{stage.upper()}_FAILED"})
    destination.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload = {**payload, "status": "PASS", "stage": stage, "execution_class": execution_class, "parent_hashes": parent_hashes, "same_run_only": True}
    if stage != "ThesisOutputs" and stage != "CompareFrozen":
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    artifact = {"logical_id": f"fresh:{stage}:output", "path": str(path.relative_to(paths.root)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "producer": "thesis_repro.stages.adapters", "parents": [{"sha256": value} for value in parent_hashes]}
    return StageResult(stage, PASS, execution_class, executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details=payload)
