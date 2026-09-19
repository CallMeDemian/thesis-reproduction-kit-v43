from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from contextlib import contextmanager
from typing import Any

from .base import StageResult, sha256_file, write_stage_artifact
from credit_recourse.oracle.fresh_runtime import resolve_fresh_oracle_runtime


def _file_artifact(root: Path, path: Path, logical_id: str, parents=()):
    return {"logical_id": logical_id, "path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "producer": "credit_recourse.oracle.production", "parents": list(parents)}


def _copy_contract_templates(project_root: Path, work_root: Path) -> Path:
    source = project_root / "contracts" / "oracle_components"
    target = work_root / "contracts" / "oracle_components"
    if not source.is_dir():
        raise FileNotFoundError(f"fresh Oracle contract templates are missing: {source}")
    shutil.copytree(source, target, dirs_exist_ok=True)
    rl_contract = project_root / "contracts" / "scientific" / "final_freeze" / "final_oracle_rl_contract.json"
    if rl_contract.is_file():
        shutil.copy2(rl_contract, target / rl_contract.name)
    return target


@contextmanager
def _oracle_environment(
    root: Path,
    runtime=None,
    *,
    work_root: Path | None = None,
    config_root: Path | None = None,
    raw_root: Path | None = None,
):
    """Bind and resolve the one call-time runtime used by fresh Oracle.

    The adapter cannot resolve the default ``unbound`` runtime before it has
    established the current run's work directory.  Bind the run-local roots
    first, then ask the canonical resolver to materialize the runtime object.
    """
    if runtime is not None:
        work_root = runtime.work_root
        config_root = runtime.config_root
        raw_root = runtime.raw_root
    work_root = Path(work_root or root / "runs" / "unbound" / "oracle_work").resolve()
    config_root = Path(config_root or root / "contracts" / "oracle_components").resolve()
    raw_root = Path(raw_root or root / "data" / "raw").resolve()
    names = {
        "THESIS_REPRO_PROJECT_ROOT": str(root),
        "THESIS_REPRO_RUN_ID": work_root.parent.parent.name,
        "THESIS_REPRO_ORACLE_WORK_ROOT": str(work_root),
        "THESIS_REPRO_ORACLE_CONFIG_ROOT": str(config_root),
        "THESIS_REPRO_ORACLE_RAW_ROOT": str(raw_root),
    }
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    try:
        yield resolve_fresh_oracle_runtime(root)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_stage1_success(return_code: int, ledger_path: Path) -> dict[str, Any]:
    """Enforce the executable Stage1 acceptance boundary."""
    if int(return_code) != 0:
        raise RuntimeError(f"production Stage1 returned unexpected code {return_code}; exact rc=0 is required")
    report = _read_json(ledger_path)
    if report.get("status") != "PASS" or report.get("final_result_allowed") is not True:
        raise RuntimeError("Stage1 ledger is not PASS with final_result_allowed=true")
    return report


def _legacy_references(value: Any, *, key: str = "") -> list[str]:
    """Find scientific parent references in generated fresh metadata."""
    forbidden = ("data/final_freeze", "configs/current", "archive/DEPLOYED_RELEASE", "frozen/")
    if isinstance(value, dict):
        found: list[str] = []
        for name, item in value.items():
            if name in {"source_contract_path", "source_contract_sha256"}:
                continue
            found.extend(_legacy_references(item, key=str(name)))
        return found
    if isinstance(value, list):
        found: list[str] = []
        for item in value:
            found.extend(_legacy_references(item, key=key))
        return found
    text = str(value).replace("\\", "/")
    return [text] if any(token in text for token in forbidden) else []


def _verify_production_oracle(root: Path, runtime, *, write_validation_report: bool = False) -> dict[str, Any]:
    """Run and check the real Stage1 verifiers against this run's artifacts."""
    import numpy as np
    import pandas as pd

    errors: list[str] = []
    ledger_path = runtime.ledgers_root / "stage1_oracle_backends_full_development.json"
    stage1_report = _read_json(ledger_path)
    if stage1_report.get("status") != "PASS" or stage1_report.get("final_result_allowed") is not True:
        errors.append("Stage1 final ledger is not PASS with final_result_allowed=true")

    stage0_validation = _read_json(runtime.stage0_root / "stage0_validation.json")
    if stage0_validation.get("status") != "PASS":
        errors.append("Stage0 validation is not PASS")

    from credit_recourse.oracle.verification.verify_stage00_04_growth_eligibility_contract import verify_contract
    from credit_recourse.oracle.verification.verify_alpha_contract import verify_and_write
    from credit_recourse.oracle.verification import verify_stage1_outputs, verify_stage1_substrate_validation
    from credit_recourse.oracle.verification.verify_oracle_semantic_closure import verify as verify_semantic

    growth = verify_contract(root, require_selection_output=True)
    if growth.get("status") != "PASS":
        errors.append("Stage00_04 growth eligibility verifier failed")
    alpha, alpha_manifest = verify_and_write(root, check_output=True)
    if alpha.get("status") != "PASS" or alpha_manifest is None:
        errors.append("Alpha strict contract verifier failed")
    outputs_rc = verify_stage1_outputs.main(["--project-root", str(root), "--strict"])
    outputs = _read_json(runtime.ledgers_root / "stage1_contract_verification.json")
    if outputs_rc != 0 or outputs.get("status") != "PASS":
        errors.append("Stage1 output contract verifier failed")
    semantic = verify_semantic(root)
    if semantic.get("status") != "PASS":
        errors.append("Oracle semantic closure verifier failed")
    substrate_rc = verify_stage1_substrate_validation.main(["--project-root", str(root)])
    substrate = _read_json(runtime.ledgers_root / "stage1_substrate_validation_loopB1.json")
    if substrate_rc != 0 or substrate.get("status") != "PASS":
        errors.append("Stage1 substrate validation infrastructure did not PASS")

    import yaml
    registry_path = runtime.registry_path
    registry_obj = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    provenance = registry_obj.get("provenance") or {}
    if not provenance.get("source_contract_sha256") or not provenance.get("materialized_run_id"):
        errors.append("run-local Oracle registry lacks provenance")
    if str(provenance.get("materialized_run_id")) != runtime.work_root.parent.parent.name:
        errors.append("run-local Oracle registry run id does not match the active run")

    required = {
        "alpha_params": runtime.backends_root / "alpha/oracle_alpha_params.json",
        "alpha_output": runtime.backends_root / "alpha/oracle_firm_year_output_alpha.parquet",
        "beta_params": runtime.backends_root / "beta/benchmark_beta_params.json",
        "beta_output": runtime.backends_root / "beta/benchmark_firm_year_output_beta.parquet",
        "gamma_params": runtime.backends_root / "gamma/benchmark_gamma_params.json",
        "gamma_model": runtime.backends_root / "gamma/benchmark_gamma_model.joblib",
        "gamma_output": runtime.backends_root / "gamma/benchmark_firm_year_output_gamma.parquet",
    }
    numerical: list[dict[str, Any]] = []
    for label, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing required production artifact: {label}: {path}")
    for path in [required["alpha_output"], required["beta_output"], required["gamma_output"]]:
        if path.is_file():
            frame = pd.read_parquet(path)
            values = frame.select_dtypes(include=["number"])
            finite = bool(np.isfinite(values.to_numpy(dtype=float)).all()) if not values.empty else False
            numerical.append({"path": path.as_posix(), "rows": int(len(frame)), "finite": finite})
            if len(frame) <= 0 or not finite:
                errors.append(f"backend output is empty or non-finite: {path}")

    lineage = []
    for path in [runtime.registry_path, ledger_path, runtime.ledgers_root / "stage1_contract_verification.json", runtime.ledgers_root / "oracle_semantic_closure.json"]:
        if path.suffix.lower() in {".json", ".yaml", ".yml"} and path.is_file():
            try:
                obj = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else yaml.safe_load(path.read_text(encoding="utf-8"))
                lineage.extend(_legacy_references(obj))
            except Exception as exc:
                errors.append(f"cannot inspect lineage artifact {path}: {exc!r}")
    if lineage:
        errors.append("fresh Oracle verification artifacts contain forbidden legacy/frozen parent references")

    evidence = {
        "schema_version": "fresh_oracle_production_verification_v2",
        "status": "PASS" if not errors else "FAIL",
        "final_result_allowed": not errors,
        "stage1_report": {"status": stage1_report.get("status"), "final_result_allowed": stage1_report.get("final_result_allowed")},
        "verifiers": {
            "stage00_04_growth": growth.get("status"),
            "alpha_strict": alpha.get("status"),
            "stage1_outputs": outputs.get("status"),
            "semantic_closure": semantic.get("status"),
            "substrate_validation": substrate.get("status"),
        },
        "numerical_outputs": numerical,
        "registry": {"path": registry_path.as_posix(), "provenance": provenance},
        "forbidden_legacy_parent_references": lineage,
        "errors": errors,
    }
    if write_validation_report:
        path = runtime.work_root.parent / "oracle_validation_report.json"
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return evidence


def run_oracle_stage(paths, project_root: Path, parent_hashes: list[str]) -> StageResult:
    """Execute the production Stage0 and Stage1 Oracle code in a run namespace."""
    root = Path(project_root).resolve()
    raw_all = root / "data" / "raw" / "raw_all"
    raw_rating = root / "data" / "raw" / "rating_sample"
    if not raw_all.is_dir() or not raw_rating.is_dir() or not any(raw_all.rglob("*.xlsx")) or not any(raw_rating.rglob("*.xlsx")):
        return StageResult("Oracle", "INPUT_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "production Stage0 requires data/raw/raw_all and data/raw/rating_sample Excel inputs", "raw_all": str(raw_all), "raw_rating": str(raw_rating)})

    work = paths.oracle_root / "work"
    work.mkdir(parents=True, exist_ok=True)
    config_root = _copy_contract_templates(root, work)
    stage0_dir = work / "stage0_oracle_foundation"
    try:
        with _oracle_environment(
            root,
            work_root=work,
            config_root=config_root,
            raw_root=root / "data" / "raw",
        ) as runtime:
            from credit_recourse.oracle.stage0.build_stage0_foundation_from_raw import build_stage0_foundation
            from credit_recourse.oracle.stage1.run_stage1_oracle_development import main as stage1_main

            stage0_meta = build_stage0_foundation(root, raw_all, raw_rating, stage0_dir, clean=True)
            rc = stage1_main([
                "--project-root", str(root), "--raw-rating-dir", str(raw_rating), "--clean",
            ])
            _require_stage1_success(rc, runtime.ledgers_root / "stage1_oracle_backends_full_development.json")
            evidence = _verify_production_oracle(root, runtime, write_validation_report=True)
            if evidence.get("status") != "PASS" or evidence.get("final_result_allowed") is not True:
                raise RuntimeError("production Oracle verification failed: " + json.dumps(evidence.get("errors", []), ensure_ascii=False))
    except Exception as exc:
        return StageResult("Oracle", "ORACLE_EXECUTION_FAILED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"error": repr(exc), "work_root": str(work)})

    generated = sorted(p for p in work.rglob("*") if p.is_file() and p.suffix.lower() in {".parquet", ".json", ".joblib", ".pkl", ".yaml"})
    required_names = {
        "stage0_canonical_panel.parquet", "stage0_validation.json",
        "oracle_alpha_params.json", "oracle_firm_year_output_alpha.parquet",
        "benchmark_beta_params.json", "benchmark_firm_year_output_beta.parquet",
        "benchmark_gamma_params.json", "benchmark_gamma_model.joblib",
        "benchmark_firm_year_output_gamma.parquet",
    }
    found_names = {p.name for p in generated}
    missing = sorted(required_names - found_names)
    if missing:
        return StageResult("Oracle", "ORACLE_ARTIFACTS_INCOMPLETE", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"missing_artifacts": missing, "generated_count": len(generated), "verification": evidence})

    receipt = {"schema_version": "fresh_oracle_execution_receipt_v2", "stage": "Oracle", "execution_class": "REAL_COMPUTE", "executed": True, "production_functions": ["credit_recourse.oracle.stage0.build_stage0_foundation", "credit_recourse.oracle.stage1.run_stage1_oracle_development"], "stage0": stage0_meta, "verification": evidence, "generated_artifact_count": len(generated), "generated_artifacts": [str(p.relative_to(paths.root)).replace("\\", "/") for p in generated], "parent_hashes": parent_hashes}
    receipt_path = paths.oracle_root / "oracle_execution_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    artifacts = [write_stage_artifact(paths, "02_oracle/oracle_execution_receipt.json", receipt, "fresh:oracle:execution_receipt", ({"sha256": h} for h in parent_hashes))]
    artifacts.extend(_file_artifact(paths.root, p, f"fresh:oracle:{p.name}", ({"sha256": h} for h in parent_hashes)) for p in generated)
    return StageResult("Oracle", "PASS", "REAL_COMPUTE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"production_execution": True, "generated_artifact_count": len(generated), "work_root": str(work.relative_to(paths.root)).replace("\\", "/"), "stage1_report_status": evidence["stage1_report"]["status"], "verifier_statuses": evidence["verifiers"]})


class OracleAdapter:
    name = "Oracle"

    def run(self, paths, project_root, parents):
        return run_oracle_stage(paths, project_root, parents)


def verify_oracle_stage(paths, parent_hashes: list[str]) -> StageResult:
    """Re-run the production verifiers; finite Parquet values alone are insufficient."""
    root = Path(paths.root).resolve()
    try:
        work = paths.oracle_root / "work"
        config_root = work / "contracts" / "oracle_components"
        with _oracle_environment(
            root,
            work_root=work,
            config_root=config_root,
            raw_root=root / "data" / "raw",
        ) as runtime:
            evidence = _verify_production_oracle(root, runtime, write_validation_report=True)
    except Exception as exc:
        evidence = {"schema_version": "fresh_oracle_validation_v2", "status": "FAIL", "final_result_allowed": False, "errors": [repr(exc)]}
    report = {**evidence, "parent_hashes": parent_hashes, "executed": True}
    artifact = write_stage_artifact(paths, "02_oracle/oracle_validation_report.json", report, "fresh:oracle:validation", ({"sha256": h} for h in parent_hashes))
    passed = report.get("status") == "PASS" and report.get("final_result_allowed") is True
    return StageResult("VerifyOracle", "PASS" if passed else "ORACLE_VERIFICATION_FAILED", "REAL_COMPUTE", executed=passed, artifacts=[artifact], parent_hashes=parent_hashes, details={"production_verifiers": report.get("verifiers", {}), "stage1_report": report.get("stage1_report"), "errors": report.get("errors", [])})
