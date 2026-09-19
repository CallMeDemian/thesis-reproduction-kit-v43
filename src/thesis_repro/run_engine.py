from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    EXPECTED_REQUESTS,
    contract_report,
    fresh_contract_ready,
    iter_logical_requests,
    live_execution_authorized,
    live_runner_ready,
)
from .data import verify_input_contract
from .paths import ROOT, load_json, sha256_file, write_json
from .runtime_paths import FreshRuntimePaths
from .stages.adapters import run_heavy_gate, run_real_stage
from .status import (
    APPROVAL_REQUIRED,
    FAILED,
    INPUT_REQUIRED,
    PASS,
    PASS_WITH_QUALIFICATION,
    SCIENTIFIC_ACCEPTED,
    SMOKE_PASS,
    SMOKE_PASS_WITH_SKIPS,
    aggregate_completion,
    is_scientific_accepted,
    normalize_legacy_status,
)

STAGES = {
    "OracleClean": ["VerifyInputs", "Oracle", "VerifyOracle"],
    "OracleRLClean": ["VerifyInputs", "Oracle", "VerifyOracle", "Simulator", "RLDataset", "RLExecutionGate", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL"],
    "OracleRLLLMClean": ["VerifyInputs", "Oracle", "VerifyOracle", "Simulator", "RLDataset", "RLExecutionGate", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL", "LLMPrepare", "LLMGenerate", "LLMMaterialize", "Stage8", "Stage9", "VerifyResults"],
    "FullClean": ["VerifyInputs", "Oracle", "VerifyOracle", "Simulator", "RLDataset", "RLExecutionGate", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL", "LLMPrepare", "LLMGenerate", "LLMMaterialize", "Stage8", "Stage9", "VerifyResults", "ThesisOutputs", "CompareFrozen", "VerifyAll"],
}
STAGE_DIRS = {
    "VerifyInputs": "01_inputs", "Oracle": "02_oracle", "VerifyOracle": "02_oracle",
    "Simulator": "03_simulator", "RLDataset": "04_rl_dataset", "RLEncoder": "05_rl_encoder",
    "RLExecutionGate": "05_rl_encoder",
    "RLBehaviorClone": "06_rl_bc", "RLIQL": "07_rl_iql", "C3E": "08_c3e", "Stage6": "08_c3e",
    "VerifyRL": "08_c3e", "LLMPrepare": "09_llm", "LLMGenerate": "09_llm",
    "LLMMaterialize": "09_llm", "Stage8": "10_stage8", "Stage9": "11_stage9",
    "VerifyResults": "12_results", "ThesisOutputs": "13_thesis_outputs",
    "CompareFrozen": "14_comparison", "VerifyAll": "15_release",
}
RUN_DIRS = ["00_run", *[f"{i:02d}_{name}" for i, name in enumerate(("inputs", "oracle", "simulator", "rl_dataset", "rl_encoder", "rl_bc", "rl_iql", "c3e", "llm", "stage8", "stage9", "results", "thesis_outputs", "comparison", "release"), start=1)], "logs"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNCOMMITTED"


def _git_source_state() -> dict[str, Any]:
    """Capture the exact source state used by a fresh execution."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return {
            "git_commit": _git_commit(),
            "git_tree_clean": not bool(status.strip()),
            "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except Exception:
        return {"git_commit": _git_commit(), "git_tree_clean": None, "git_diff_sha256": "UNAVAILABLE"}


def _input_contract_hash() -> str:
    path = ROOT / "data/raw/input_contract_report.json"
    return sha256_file(path) if path.is_file() else "MISSING"


def _scientific_fingerprint(input_contract_hash: str | None = None) -> str:
    """Hash every source that can change fresh scientific execution semantics."""
    paths = [
        ROOT / "contracts/scientific/input_inventory.json",
        ROOT / "contracts/scientific/v43_action_contract.json",
        ROOT / "contracts/scientific/v43_alpha_contract.json",
        ROOT / "contracts/scientific/final_freeze/final_oracle_rl_contract.json",
        ROOT / "contracts/scientific/final_freeze/oracle_backend_registry.yaml",
        ROOT / "contracts/llm/fresh_replication_contract.json",
    ]
    paths.extend(sorted((ROOT / "contracts/oracle_components").rglob("*")))
    records = [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(path)} for path in paths if path.is_file()]
    payload = {
        "schema_version": "scientific_execution_fingerprint_v2",
        "git_commit": _git_commit(),
        "input_contract_hash": input_contract_hash if input_contract_hash is not None else _input_contract_hash(),
        "sources": records,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _contract_hash() -> str:
    return _scientific_fingerprint()


def _stage_manifest(run_dir: Path, stage: str) -> Path:
    return run_dir / STAGE_DIRS[stage] / f"stage_manifest_{stage}.json"


def _artifact(path: Path, logical_id: str, producer: str, parents: Iterable[dict[str, str]] = ()) -> dict[str, Any]:
    return {"logical_id": logical_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "producer": producer, "parents": list(parents)}


def _write_artifact(run_dir: Path, relative: str, payload: Any, logical_id: str, producer: str, parents: Iterable[dict[str, str]] = ()) -> dict[str, Any]:
    path = run_dir / relative
    write_json(path, payload)
    return _artifact(path, logical_id, producer, parents)


def _write_stage(run_dir: Path, stage: str, status: str, parent_hashes: list[str], details: dict[str, Any] | None = None, artifacts: list[dict[str, Any]] | None = None) -> str:
    if status == "PASS" and (details or {}).get("execution_class") == "REAL_COMPUTE" and (details or {}).get("executed") is not True:
        raise ValueError(f"scientific invariant violated: {stage} PASS/REAL_COMPUTE requires executed=True")
    payload = {"schema_version": "stage_manifest_v2", "stage": stage, "status": status, "run_id": run_dir.name, "created_at": datetime.now(timezone.utc).isoformat(), "contract_hash": _contract_hash(), "parent_hashes": parent_hashes, "artifacts": artifacts or [], "details": details or {}}
    path = _stage_manifest(run_dir, stage)
    write_json(path, payload)
    return sha256_file(path)


def _prepare_run_dirs(run_dir: Path) -> None:
    for name in RUN_DIRS:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def plan(mode: str, run_id: str, profile: str) -> dict[str, Any]:
    if mode not in STAGES:
        raise ValueError(f"unknown mode: {mode}")
    return {"run_id": run_id, "mode": mode, "profile": profile, "stages": STAGES[mode], "run_root": f"runs/{run_id}", "required_directories": RUN_DIRS, "resources": {"profile": profile, "gpu_required": profile == "full" and mode in {"OracleRLClean", "OracleRLLLMClean", "FullClean"}, "live_api": False}, "expected_outputs": [str(_stage_manifest(ROOT / "runs" / run_id, stage).relative_to(ROOT)) for stage in STAGES[mode]], "contract_hash": _contract_hash(), "input_policy": "full execution requires INPUT_CONTRACT_PASS; smoke is contract-only and never substitutes frozen scientific parents"}


def render_dry_run(run_dir: Path, full: bool = True) -> dict[str, Any]:
    report = contract_report(run_id=run_dir.name)
    if report["generated"] != EXPECTED_REQUESTS or report["unique_ids"] != EXPECTED_REQUESTS:
        raise ValueError(f"LLM dry-render cardinality failure: {report}")
    request_path = run_dir / "09_llm" / "logical_requests.jsonl"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = 0
    limit = None if full else 25
    with request_path.open("w", encoding="utf-8") as handle:
        for item in iter_logical_requests(run_id=run_dir.name):
            if limit is not None and rendered >= limit:
                break
            handle.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
            rendered += 1
    report.update({"rendered_to_disk": rendered, "dry_run_mode": "full" if full else "smoke", "status": SMOKE_PASS if full and rendered == EXPECTED_REQUESTS else SMOKE_PASS_WITH_SKIPS})
    write_json(run_dir / "09_llm/dry_run_manifest.json", report)
    return report


def _path_from_artifact(artifact: dict[str, Any]) -> Path:
    return (ROOT / str(artifact["path"])).resolve()


def _artifact_is_intact(artifact: dict[str, Any]) -> bool:
    path = _path_from_artifact(artifact)
    if not path.is_file() or path.stat().st_size != int(artifact.get("size_bytes", -1)):
        return False
    return sha256_file(path) == artifact.get("sha256")


def _manifest_is_reusable(run_dir: Path, stage: str, old: dict[str, Any], manifest: dict[str, Any], *, profile: str, fingerprint: str) -> tuple[bool, str]:
    status = normalize_legacy_status(str(old.get("status", "")))
    allowed = {SMOKE_PASS, SMOKE_PASS_WITH_SKIPS} if profile == "smoke" else set(SCIENTIFIC_ACCEPTED)
    if status not in allowed:
        return False, f"status_not_reusable:{status}"
    if old.get("contract_hash") != fingerprint:
        return False, "scientific_fingerprint_changed"
    recorded_manifest_hash = (manifest.get("stage_manifest_hashes") or {}).get(stage)
    existing_path = _stage_manifest(run_dir, stage)
    if recorded_manifest_hash and sha256_file(existing_path) != recorded_manifest_hash:
        return False, "stage_manifest_tampered"
    if not all(_artifact_is_intact(item) for item in old.get("artifacts", [])):
        return False, "stage_artifact_tampered_or_missing"
    known_manifest_hashes = {
        sha256_file(path) for path in run_dir.rglob("stage_manifest_*.json") if path.is_file()
    }
    if any(parent not in known_manifest_hashes for parent in old.get("parent_hashes", [])):
        return False, "parent_manifest_missing_or_tampered"
    for artifact in old.get("artifacts", []):
        for parent in artifact.get("parents", []):
            if parent.get("sha256") not in known_manifest_hashes:
                return False, "artifact_parent_manifest_missing_or_tampered"
    return True, "ok"


def _invalidate_from(run_dir: Path, manifest: dict[str, Any], stages: list[str], start_index: int, reason: str) -> None:
    """Remove stale manifests/artifacts for a stage and every downstream stage."""
    invalidated = manifest.setdefault("invalidations", [])
    invalidated.append({"from_stage": stages[start_index], "reason": reason, "at": datetime.now(timezone.utc).isoformat()})
    removed_paths: set[str] = set()
    for stage in stages[start_index:]:
        old_path = _stage_manifest(run_dir, stage)
        if old_path.is_file():
            try:
                old = load_json(old_path)
                for artifact in old.get("artifacts", []):
                    path = _path_from_artifact(artifact)
                    removed_paths.add(str(artifact.get("path", "")))
                    if path.is_relative_to(run_dir.resolve()) and path.is_file():
                        path.unlink()
            except Exception:
                pass
            old_path.unlink(missing_ok=True)
        manifest.get("stage_status", {}).pop(stage, None)
        manifest.get("stage_manifests", {}).pop(stage, None)
        manifest.get("stage_manifest_hashes", {}).pop(stage, None)
    manifest["artifacts"] = [item for item in manifest.get("artifacts", []) if item.get("path") not in removed_paths]
    manifest["completion_state"] = "RUNNING"


def _smoke_stage_artifact(run_dir: Path, stage: str, parent_hashes: list[str], profile: str) -> dict[str, Any]:
    directory = STAGE_DIRS[stage]
    payload = {"schema_version": "fresh_stage_contract_v1", "run_id": run_dir.name, "stage": stage, "execution_class": "contract_smoke", "profile": profile, "same_run_parent_policy": "all fresh smoke parents are namespaced under this run; frozen artifacts are references only", "parent_hashes": parent_hashes}
    parents = [{"logical_id": f"stage-parent:{index}", "sha256": value} for index, value in enumerate(parent_hashes)]
    return _write_artifact(run_dir, f"{directory}/{stage.lower()}_contract.json", payload, f"fresh:{run_dir.name}:{stage}", "thesis_repro.run_engine", parents)


def execute(mode: str, run_id: str, profile: str = "smoke", resume: bool = False, from_stage: str | None = None, to_stage: str | None = None, execute_llm: bool = False, dry_render: bool = False) -> dict[str, Any]:
    if mode not in STAGES:
        raise ValueError(f"unknown mode: {mode}")
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single safe namespace component")
    if execute_llm and (not live_execution_authorized() or not live_runner_ready()):
        raise PermissionError("live LLM requires the explicit gate and an installed provider runner")
    if execute_llm and not fresh_contract_ready():
        raise ValueError("live LLM is blocked: fresh replication contract is not ready")

    run_dir = ROOT / "runs" / run_id
    paths = FreshRuntimePaths.from_run(ROOT, run_id, create=True)
    _prepare_run_dirs(run_dir)
    contract_report_now = verify_input_contract(write=True)
    input_report_path = ROOT / "data/raw/input_contract_report.json"
    input_hash = sha256_file(input_report_path) if input_report_path.is_file() else "MISSING"
    fingerprint_now = _scientific_fingerprint(input_hash)
    source_state_now = _git_source_state()
    manifest_path = run_dir / "run_manifest.json"
    if resume and manifest_path.is_file():
        manifest = load_json(manifest_path)
        global_reasons = []
        if manifest.get("scientific_contract_hash") != fingerprint_now:
            global_reasons.append("scientific_fingerprint_changed")
        if manifest.get("scientific_execution_fingerprint") != fingerprint_now:
            global_reasons.append("execution_fingerprint_changed")
        if manifest.get("input_contract_hash") != input_hash:
            global_reasons.append("input_contract_changed")
        if manifest.get("source_state") != source_state_now:
            global_reasons.append("source_state_changed")
        if manifest.get("mode") != mode or manifest.get("profile") != profile:
            global_reasons.append("run_mode_or_profile_changed")
        if global_reasons:
            _invalidate_from(run_dir, manifest, STAGES.get(manifest.get("mode"), STAGES[mode]), 0, ";".join(global_reasons))
            manifest.update({
                "git_commit": source_state_now["git_commit"],
                "source_state": source_state_now,
                "mode": mode,
                "profile": profile,
                "input_contract_hash": input_hash,
                "scientific_contract_hash": fingerprint_now,
                "scientific_execution_fingerprint": fingerprint_now,
            })
    else:
        manifest = {
            "schema_version": "run_manifest_v3", "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "git_commit": source_state_now["git_commit"], "source_state": source_state_now, "mode": mode, "profile": profile, "execution_class": "contract_smoke" if profile == "smoke" else "fresh_replication", "environment": {"python": sys.version, "platform": platform.platform(), "cwd": str(ROOT)}, "gpu_info": {"status": "not_probed"}, "input_contract_status": contract_report_now["status"], "input_contract_hash": input_hash, "scientific_contract_hash": fingerprint_now, "scientific_execution_fingerprint": fingerprint_now, "input_receipt": "data/raw/input_receipt.json" if (ROOT / "data/raw/input_receipt.json").is_file() else None, "random_seeds": {"oracle": 73019, "simulator": 73020, "rl": {"contract": "contracts/scientific/fresh_rl_execution_contract.json", "selected_configurations": ["B27", "M2_S0935", "DT06", "T15_REWARD_S088"], "seeds": [2, 11, 12, 13, 14, 15, 16], "actor_count": 28}, "llm": None}, "provider_identities": {}, "stage_status": {}, "stage_manifests": {}, "stage_manifest_hashes": {}, "artifacts": [], "invalidations": [], "completion_state": "RUNNING"
        }
    write_json(run_dir / "00_run/input_contract_snapshot.json", contract_report_now)

    stages = STAGES[mode]
    selected = stages[:]
    if from_stage:
        selected = selected[selected.index(from_stage):]
    if to_stage:
        selected = selected[: selected.index(to_stage) + 1]
    parent_hashes: list[str] = []
    for stage_index, stage in enumerate(selected):
        existing = _stage_manifest(run_dir, stage)
        if resume and existing.is_file():
            old = load_json(existing)
            reusable, reason = _manifest_is_reusable(run_dir, stage, old, manifest, profile=profile, fingerprint=fingerprint_now)
            if reusable:
                digest = sha256_file(existing)
                parent_hashes = [digest]
                manifest["stage_status"][stage] = normalize_legacy_status(old["status"])
                manifest["stage_manifests"][stage] = str(existing.relative_to(ROOT)).replace("\\", "/")
                manifest.setdefault("stage_manifest_hashes", {})[stage] = digest
                manifest["artifacts"].extend(old.get("artifacts", []))
                continue
            _invalidate_from(run_dir, manifest, selected, stage_index, reason)

        status = PASS
        details: dict[str, Any] = {"profile": profile, "execution_class": manifest["execution_class"], "same_run_compute_parent": True}
        artifacts: list[dict[str, Any]] = []
        if stage == "VerifyInputs":
            details.update({"input_contract_status": contract_report_now["status"], "raw_data_present": contract_report_now["present_file_count"] > 0})
            status = (SMOKE_PASS if contract_report_now["status"] == "INPUT_CONTRACT_PASS" else SMOKE_PASS_WITH_SKIPS) if profile == "smoke" else (PASS if contract_report_now["status"] == "INPUT_CONTRACT_PASS" else INPUT_REQUIRED)
            artifacts.append(_artifact(input_report_path, "input.contract.report", "thesis_repro.data") if input_report_path.is_file() else _write_artifact(run_dir, "01_inputs/input_contract_report.json", contract_report_now, "input.contract.report", "thesis_repro.data"))
        elif profile == "full" and contract_report_now["status"] != "INPUT_CONTRACT_PASS":
            status = INPUT_REQUIRED
            details["reason"] = "full fresh stages are blocked until data/raw/input_contract_report.json is INPUT_CONTRACT_PASS"
        else:
            if profile == "smoke":
                status = SMOKE_PASS
                artifacts.append(_smoke_stage_artifact(run_dir, stage, parent_hashes, profile))
                if stage == "LLMGenerate" and dry_render:
                    details.update(render_dry_run(run_dir, full=True))
                    artifacts.append(_artifact(run_dir / "09_llm/logical_requests.jsonl", f"fresh:{run_id}:llm.logical_requests", "thesis_repro.run_engine"))
            else:
                if stage == "RLExecutionGate":
                    result = run_heavy_gate(paths, parent_hashes)
                    status = normalize_legacy_status(result.status)
                    details.update(result.details)
                    details.update({"execution_class": result.execution_class, "implemented": result.implemented, "executed": result.executed, "gate_checked_before_heavy_stage": True})
                    artifacts.extend(result.artifacts)
                else:
                    result = run_real_stage(paths, stage, parent_hashes, execute_llm=execute_llm)
                    status = normalize_legacy_status(result.status)
                    details.update(result.details)
                    details.update({"execution_class": result.execution_class, "implemented": result.implemented, "executed": result.executed})
                    artifacts.extend(result.artifacts)
                    if stage in {"Oracle", "VerifyOracle"} and details.get("rq1"):
                        manifest["oracle"] = {"rq1": details["rq1"]}
                if stage == "LLMPrepare" and mode in {"OracleRLLLMClean", "FullClean"} and not execute_llm:
                    details.update(render_dry_run(run_dir, full=True))
                    artifacts.append(_artifact(run_dir / "09_llm/logical_requests.jsonl", f"fresh:{run_id}:llm.logical_requests", "thesis_repro.run_engine"))
                    status = APPROVAL_REQUIRED
                    details.update({"executed": False, "reason": "48,300 fresh logical requests rendered and validated; provider transport requires explicit live gate"})
                    digest = _write_stage(run_dir, stage, status, parent_hashes, details, artifacts)
                    parent_hashes = [digest]
                    manifest["stage_status"][stage] = status
                    manifest["stage_manifests"][stage] = str(_stage_manifest(run_dir, stage).relative_to(ROOT)).replace("\\", "/")
                    manifest.setdefault("stage_manifest_hashes", {})[stage] = digest
                    manifest["artifacts"].extend(artifacts)
                    manifest["completion_state"] = status
                    break
        digest = _write_stage(run_dir, stage, status, parent_hashes, details, artifacts)
        parent_hashes = [digest]
        manifest["stage_status"][stage] = status
        manifest["stage_manifests"][stage] = str(_stage_manifest(run_dir, stage).relative_to(ROOT)).replace("\\", "/")
        manifest.setdefault("stage_manifest_hashes", {})[stage] = digest
        manifest["artifacts"].extend(artifacts)
        if profile != "smoke" and not is_scientific_accepted(status):
            manifest["completion_state"] = status
            break

    if manifest.get("completion_state") == "RUNNING":
        manifest["completion_state"] = aggregate_completion(list(manifest["stage_status"].values()), profile=profile)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    return manifest


def trace_run(run_id: str) -> dict[str, Any]:
    run_dir = ROOT / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"unknown run: {run_id}")
    manifest = load_json(run_dir / "run_manifest.json")
    artifacts: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("stage_manifest_*.json")):
        artifacts.extend(load_json(path).get("artifacts", []))
    forbidden = ("data/final_freeze", "configs/current", "archive/DEPLOYED_RELEASE", "frozen/")
    forbidden_legacy = [artifact for artifact in artifacts if any(token in str(artifact.get("path", "")).replace("\\", "/") for token in forbidden)]
    stage_order = STAGES.get(manifest.get("mode"), list(manifest.get("stage_status", {})))
    stage_paths = {path.stem.removeprefix("stage_manifest_"): path for path in run_dir.rglob("stage_manifest_*.json")}
    prior_hashes: set[str] = set()
    lineage_errors: list[dict[str, Any]] = []
    nested_forbidden: list[dict[str, Any]] = []
    for stage in stage_order:
        path = stage_paths.get(stage)
        if path is None:
            continue
        stage_manifest = load_json(path)
        stage_hash = sha256_file(path)
        for parent in stage_manifest.get("parent_hashes", []):
            if parent not in prior_hashes:
                lineage_errors.append({"stage": stage, "parent_sha256": parent, "reason": "parent_not_resolvable_to_earlier_same_run_artifact"})
        for artifact in stage_manifest.get("artifacts", []):
            for parent in artifact.get("parents", []):
                parent_hash = parent.get("sha256") if isinstance(parent, dict) else parent
                if parent_hash not in prior_hashes:
                    lineage_errors.append({"stage": stage, "artifact": artifact.get("logical_id"), "parent_sha256": parent_hash, "reason": "artifact_parent_not_resolvable_to_earlier_same_run_artifact"})
            artifact_path = ROOT / str(artifact.get("path", ""))
            if artifact_path.is_file() and artifact_path.suffix.lower() in {".json", ".yaml", ".yml"}:
                try:
                    raw = artifact_path.read_text(encoding="utf-8")
                    if any(token in raw.replace("\\", "/") for token in forbidden):
                        nested_forbidden.append({"stage": stage, "artifact": artifact.get("logical_id"), "path": artifact.get("path")})
                except (OSError, UnicodeDecodeError):
                    pass
            prior_hashes.add(str(artifact.get("sha256")))
        prior_hashes.add(stage_hash)
    return {
        "schema_version": "recursive_trace_v2",
        "run_id": run_id,
        "completion_state": manifest.get("completion_state"),
        "stages": manifest.get("stage_status", {}),
        "artifacts": artifacts,
        "external_parents": forbidden_legacy,
        "forbidden_legacy_parents": forbidden_legacy,
        "forbidden_legacy_parent_count": len(forbidden_legacy),
        "lineage_errors": lineage_errors,
        "nested_forbidden_parent_references": nested_forbidden,
        "lineage_closed": not lineage_errors and not nested_forbidden,
    }
