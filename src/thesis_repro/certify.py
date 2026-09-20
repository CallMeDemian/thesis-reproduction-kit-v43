from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import ROOT, load_json, sha256_file, write_json
from .run_engine import STAGES, _stage_manifest, trace_run
from .status import PASS, PASS_WITH_QUALIFICATION, SCIENTIFIC_ACCEPTED, PARTIAL_EXECUTION


FORBIDDEN = ("data/final_freeze", "configs/current", "archive/DEPLOYED_RELEASE", "frozen/")


def certify_run(run_id: str) -> dict[str, Any]:
    run_dir = ROOT / "runs" / run_id
    errors: list[str] = []
    manifest_path = run_dir / "run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    if not manifest:
        errors.append("run_manifest_missing")
    if manifest.get("execution_class") != "FRESH_REPLICATION":
        errors.append("smoke_or_non_fresh_execution")
    if manifest.get("scientific_gate_applicable") is False or (manifest.get("execution_context") or {}).get("scientific_gate_applicable") is not True:
        errors.append("scientific_gate_not_applicable")
    if manifest.get("completion_state") == PARTIAL_EXECUTION or manifest.get("dag_complete") is not True:
        errors.append("partial_execution_not_certifiable")
    if manifest.get("completion_state") not in SCIENTIFIC_ACCEPTED:
        errors.append(f"run_completion_not_accepted:{manifest.get('completion_state')}")
    if (manifest.get("source_state") or {}).get("git_tree_clean") is not True:
        errors.append("source_tree_not_clean")
    mode = manifest.get("mode")
    required = STAGES.get("FullClean", []) if mode == "FullClean" else STAGES.get(mode, [])
    if mode != "FullClean":
        errors.append("certification_requires_FullClean_mode")
    stage2_validation = run_dir / "03_stage2/stage2_validation_report.json"
    if not stage2_validation.is_file():
        errors.append("stage2_validation_missing")
    else:
        try:
            if load_json(stage2_validation).get("status") != "PASS":
                errors.append("stage2_validation_not_pass")
        except Exception:
            errors.append("stage2_validation_unreadable")
    if run_dir.is_dir() and manifest:
        trace = trace_run(run_id)
        if trace.get("lineage_closed") is not True:
            errors.append("fresh_lineage_graph_not_closed")
        if trace.get("forbidden_legacy_parent_count", 0) or trace.get("nested_forbidden_parent_references"):
            errors.append("forbidden_historical_parent_reference")
    stage_records: list[dict[str, Any]] = []
    for stage in required:
        path = _stage_manifest(run_dir, stage)
        if not path.is_file():
            errors.append(f"stage_manifest_missing:{stage}")
            continue
        stage_manifest = load_json(path)
        status = stage_manifest.get("status")
        if status not in SCIENTIFIC_ACCEPTED:
            errors.append(f"stage_not_scientifically_accepted:{stage}:{status}")
        if stage_manifest.get("details", {}).get("execution_class") == "contract_smoke":
            errors.append(f"smoke_stage_in_certification:{stage}")
        if stage_manifest.get("details", {}).get("executed") is not True:
            errors.append(f"stage_not_executed:{stage}")
        for artifact in stage_manifest.get("artifacts", []):
            artifact_path = ROOT / artifact.get("path", "")
            if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.get("sha256") or artifact_path.stat().st_size != artifact.get("size_bytes"):
                errors.append(f"artifact_hash_mismatch:{stage}:{artifact.get('path')}")
            if any(token in str(artifact.get("path", "")).replace("\\", "/") for token in FORBIDDEN):
                errors.append(f"forbidden_fresh_parent_path:{stage}:{artifact.get('path')}")
        stage_records.append({"stage": stage, "status": status, "manifest_sha256": sha256_file(path), "artifacts": len(stage_manifest.get("artifacts", []))})
    state = "NOT_CERTIFIABLE"
    if not errors:
        state = "QUALIFIED_FRESH_REPLICATION" if manifest.get("completion_state") == PASS_WITH_QUALIFICATION else "CERTIFIED_FRESH_REPLICATION"
    result = {
        "schema_version": "fresh_certification_v1",
        "run_id": run_id,
        "state": state,
        "certified": state != "NOT_CERTIFIABLE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "source_state": manifest.get("source_state"),
        "stages": stage_records,
    }
    if run_dir.is_dir():
        release = run_dir / "15_release"
        release.mkdir(parents=True, exist_ok=True)
        write_json(release / "CERTIFICATION.json", result)
        (release / "CERTIFICATION.md").write_text(
            "# Fresh V4.3 Certification\n\n"
            f"State: `{state}`\n\n"
            + ("\n".join(f"- {item}" for item in errors) if errors else "All required checks passed.\n"),
            encoding="utf-8",
        )
    return result
