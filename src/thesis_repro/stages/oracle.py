from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .base import StageResult, sha256_file, write_stage_artifact


def _file_artifact(root: Path, path: Path, logical_id: str, parents=()):
    return {"logical_id": logical_id, "path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "producer": "credit_recourse.oracle.production", "parents": list(parents)}


def _copy_contract_templates(project_root: Path, work_root: Path) -> Path:
    source = project_root / "contracts" / "oracle_components"
    target = work_root / "contracts" / "oracle_components"
    if not source.is_dir():
        raise FileNotFoundError(f"fresh Oracle contract templates are missing: {source}")
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


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
    old_work = os.environ.get("THESIS_REPRO_ORACLE_WORK_ROOT")
    old_config = os.environ.get("THESIS_REPRO_ORACLE_CONFIG_ROOT")
    try:
        os.environ["THESIS_REPRO_ORACLE_WORK_ROOT"] = str(work)
        os.environ["THESIS_REPRO_ORACLE_CONFIG_ROOT"] = str(config_root)
        from credit_recourse.oracle.stage0.build_stage0_foundation_from_raw import build_stage0_foundation
        from credit_recourse.oracle.stage1.run_stage1_oracle_development import main as stage1_main

        stage0_meta = build_stage0_foundation(root, raw_all, raw_rating, stage0_dir, clean=True)
        rc = stage1_main([
            "--project-root", str(root), "--raw-rating-dir", str(raw_rating),
            "--clean", "--end-step", "backend_gamma", "--pending-ok",
        ])
        if rc not in (0, 2):
            raise RuntimeError(f"production Stage1 returned unexpected code {rc}")
    except Exception as exc:
        return StageResult("Oracle", "ORACLE_EXECUTION_FAILED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"error": repr(exc), "work_root": str(work)})
    finally:
        if old_work is None:
            os.environ.pop("THESIS_REPRO_ORACLE_WORK_ROOT", None)
        else:
            os.environ["THESIS_REPRO_ORACLE_WORK_ROOT"] = old_work
        if old_config is None:
            os.environ.pop("THESIS_REPRO_ORACLE_CONFIG_ROOT", None)
        else:
            os.environ["THESIS_REPRO_ORACLE_CONFIG_ROOT"] = old_config

    generated = sorted(p for p in work.rglob("*") if p.is_file() and p.suffix.lower() in {".parquet", ".json", ".joblib", ".pkl"})
    required_names = {"stage0_canonical_panel.parquet", "oracle_alpha_params.json", "benchmark_beta_params.json", "benchmark_gamma_params.json"}
    found_names = {p.name for p in generated}
    missing = sorted(required_names - found_names)
    if missing:
        return StageResult("Oracle", "ORACLE_ARTIFACTS_INCOMPLETE", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"missing_artifacts": missing, "generated_count": len(generated)})

    receipt = {"schema_version": "fresh_oracle_execution_receipt_v1", "stage": "Oracle", "execution_class": "REAL_COMPUTE", "executed": True, "production_functions": ["credit_recourse.oracle.stage0.build_stage0_foundation", "credit_recourse.oracle.stage1.run_stage1_oracle_development"], "stage0": stage0_meta, "generated_artifact_count": len(generated), "generated_artifacts": [str(p.relative_to(paths.root)).replace("\\", "/") for p in generated], "parent_hashes": parent_hashes}
    receipt_path = paths.oracle_root / "oracle_execution_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    artifacts = [write_stage_artifact(paths, "02_oracle/oracle_execution_receipt.json", receipt, "fresh:oracle:execution_receipt", ({"sha256": h} for h in parent_hashes))]
    artifacts.extend(_file_artifact(paths.root, p, f"fresh:oracle:{p.name}", ({"sha256": h} for h in parent_hashes)) for p in generated)
    return StageResult("Oracle", "PASS", "REAL_COMPUTE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"production_execution": True, "generated_artifact_count": len(generated), "work_root": str(work.relative_to(paths.root)).replace("\\", "/")})


class OracleAdapter:
    name = "Oracle"

    def run(self, paths, project_root, parents):
        return run_oracle_stage(paths, project_root, parents)


def verify_oracle_stage(paths, parent_hashes: list[str]) -> StageResult:
    """Verify numerical artifacts emitted by the production Oracle branch."""
    import pandas as pd

    work = paths.oracle_root / "work"
    params = sorted(work.rglob("*params.json"))
    scored = sorted(p for p in work.rglob("*.parquet") if "output" in p.name or "panel" in p.name)
    if not params or not scored:
        return StageResult("VerifyOracle", "ORACLE_VERIFICATION_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "production Oracle artifacts are incomplete"})
    checks = []
    for path in scored:
        frame = pd.read_parquet(path)
        numeric = frame.select_dtypes(include=["number"])
        finite = bool(numeric.empty or numeric.replace([float("inf"), float("-inf")], float("nan")).notna().all().all())
        checks.append({"path": str(path.relative_to(paths.root)).replace("\\", "/"), "rows": int(len(frame)), "columns": int(len(frame.columns)), "numeric_finite": finite})
    passed = all(item["rows"] > 0 and item["numeric_finite"] for item in checks)
    report = {"schema_version": "fresh_oracle_validation_v1", "status": "PASS" if passed else "FAIL", "executed": True, "parameter_artifacts": [str(p.relative_to(paths.root)).replace("\\", "/") for p in params], "scored_artifacts": checks, "parent_hashes": parent_hashes}
    artifact = write_stage_artifact(paths, "02_oracle/oracle_validation_report.json", report, "fresh:oracle:validation", ({"sha256": h} for h in parent_hashes))
    return StageResult("VerifyOracle", "PASS" if passed else "ORACLE_VERIFICATION_FAILED", "REAL_COMPUTE", executed=passed, artifacts=[artifact], parent_hashes=parent_hashes, details={"numerical_artifact_count": len(checks), "parameter_artifact_count": len(params)})
