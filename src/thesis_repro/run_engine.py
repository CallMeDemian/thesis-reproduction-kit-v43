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

STAGES = {
    "OracleClean": ["VerifyInputs", "Oracle", "VerifyOracle"],
    "OracleRLClean": ["VerifyInputs", "Oracle", "Simulator", "RLDataset", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL"],
    "OracleRLLLMClean": ["VerifyInputs", "Oracle", "Simulator", "RLDataset", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL", "LLMPrepare", "LLMGenerate", "LLMMaterialize", "Stage8", "Stage9", "VerifyResults"],
    "FullClean": ["VerifyInputs", "Oracle", "Simulator", "RLDataset", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL", "LLMPrepare", "LLMGenerate", "LLMMaterialize", "Stage8", "Stage9", "VerifyResults", "ThesisOutputs", "CompareFrozen", "VerifyAll"],
}
STAGE_DIRS = {
    "VerifyInputs": "01_inputs", "Oracle": "02_oracle", "VerifyOracle": "02_oracle",
    "Simulator": "03_simulator", "RLDataset": "04_rl_dataset", "RLEncoder": "05_rl_encoder",
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


def _contract_hash() -> str:
    paths = [ROOT / "contracts/llm/final_as_executed_generation_contract.json", ROOT / "contracts/llm/fresh_replication_contract.json"]
    canonical = "\n".join(json.dumps(load_json(path), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for path in paths if path.is_file())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    report.update({"rendered_to_disk": rendered, "dry_run_mode": "full" if full else "smoke", "status": "PASS" if full and rendered == EXPECTED_REQUESTS else "PASS_WITH_SKIPS"})
    write_json(run_dir / "09_llm/dry_run_manifest.json", report)
    return report


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
    manifest_path = run_dir / "run_manifest.json"
    if resume and manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("scientific_contract_hash") != _contract_hash() or manifest.get("input_contract_hash") != input_hash:
            manifest["completion_state"] = "INVALIDATED"
            write_json(manifest_path, manifest)
            raise ValueError("contract or input hash changed; downstream stages invalidated")
    else:
        manifest = {
            "schema_version": "run_manifest_v2", "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(), "mode": mode, "profile": profile, "execution_class": "contract_smoke" if profile == "smoke" else "fresh_replication", "environment": {"python": sys.version, "platform": platform.platform(), "cwd": str(ROOT)}, "gpu_info": {"status": "not_probed"}, "input_contract_status": contract_report_now["status"], "input_contract_hash": input_hash, "scientific_contract_hash": _contract_hash(), "input_receipt": "data/raw/input_receipt.json" if (ROOT / "data/raw/input_receipt.json").is_file() else None, "random_seeds": {"oracle": 73019, "simulator": 73020, "rl": {"stage3": [1, 2, 3, 4, 5, 6, 7], "stage5": [1, 2, 3, 4, 5, 6, 7]}, "llm": None}, "provider_identities": {}, "stage_status": {}, "stage_manifests": {}, "artifacts": [], "completion_state": "RUNNING"
        }
    write_json(run_dir / "00_run/input_contract_snapshot.json", contract_report_now)

    stages = STAGES[mode]
    selected = stages[:]
    if from_stage:
        selected = selected[selected.index(from_stage):]
    if to_stage:
        selected = selected[: selected.index(to_stage) + 1]
    parent_hashes: list[str] = []
    for stage in selected:
        existing = _stage_manifest(run_dir, stage)
        if resume and existing.is_file():
            old = load_json(existing)
            if old.get("status") in {"PASS", "PASS_WITH_SKIPS", "INPUT_REQUIRED"} and old.get("contract_hash") == _contract_hash() and all((ROOT / a["path"]).is_file() for a in old.get("artifacts", [])):
                digest = sha256_file(existing)
                parent_hashes = [digest]
                manifest["stage_status"][stage] = old["status"]
                manifest["stage_manifests"][stage] = str(existing.relative_to(ROOT)).replace("\\", "/")
                manifest["artifacts"].extend(old.get("artifacts", []))
                continue

        status = "PASS"
        details: dict[str, Any] = {"profile": profile, "execution_class": manifest["execution_class"], "same_run_compute_parent": True}
        artifacts: list[dict[str, Any]] = []
        if stage == "VerifyInputs":
            details.update({"input_contract_status": contract_report_now["status"], "raw_data_present": contract_report_now["present_file_count"] > 0})
            status = "PASS" if contract_report_now["status"] == "INPUT_CONTRACT_PASS" else ("PASS_WITH_SKIPS" if profile == "smoke" else "INPUT_REQUIRED")
            artifacts.append(_artifact(input_report_path, "input.contract.report", "thesis_repro.data") if input_report_path.is_file() else _write_artifact(run_dir, "01_inputs/input_contract_report.json", contract_report_now, "input.contract.report", "thesis_repro.data"))
        elif profile == "full" and contract_report_now["status"] != "INPUT_CONTRACT_PASS":
            status = "INPUT_REQUIRED"
            details["reason"] = "full fresh stages are blocked until data/raw/input_contract_report.json is INPUT_CONTRACT_PASS"
        else:
            if profile == "smoke":
                artifacts.append(_smoke_stage_artifact(run_dir, stage, parent_hashes, profile))
                if stage == "LLMGenerate" and dry_render:
                    details.update(render_dry_run(run_dir, full=True))
                    artifacts.append(_artifact(run_dir / "09_llm/logical_requests.jsonl", f"fresh:{run_id}:llm.logical_requests", "thesis_repro.run_engine"))
            else:
                result = run_real_stage(paths, stage, parent_hashes, execute_llm=execute_llm)
                status = result.status
                details.update(result.details)
                details.update({"execution_class": result.execution_class, "implemented": result.implemented, "executed": result.executed})
                artifacts.extend(result.artifacts)
                if stage == "LLMPrepare" and mode in {"OracleRLLLMClean", "FullClean"} and not execute_llm:
                    details.update(render_dry_run(run_dir, full=True))
                    artifacts.append(_artifact(run_dir / "09_llm/logical_requests.jsonl", f"fresh:{run_id}:llm.logical_requests", "thesis_repro.run_engine"))
                    status = "LIVE_LLM_APPROVAL_REQUIRED"
                    details.update({"executed": False, "reason": "48,300 fresh logical requests rendered and validated; provider transport requires explicit live gate"})
                    digest = _write_stage(run_dir, stage, status, parent_hashes, details, artifacts)
                    parent_hashes = [digest]
                    manifest["stage_status"][stage] = status
                    manifest["stage_manifests"][stage] = str(_stage_manifest(run_dir, stage).relative_to(ROOT)).replace("\\", "/")
                    manifest["artifacts"].extend(artifacts)
                    manifest["completion_state"] = status
                    break
                if stage == "RLIQL" and mode in {"OracleRLClean", "OracleRLLLMClean", "FullClean"}:
                    gate = run_heavy_gate(paths, parent_hashes)
                    details["heavy_gate"] = gate.details
                    if gate.status != "PASS":
                        status = gate.status
                        details["executed"] = False
                        digest = _write_stage(run_dir, stage, status, parent_hashes, details, artifacts)
                        parent_hashes = [digest]
                        manifest["stage_status"][stage] = status
                        manifest["stage_manifests"][stage] = str(_stage_manifest(run_dir, stage).relative_to(ROOT)).replace("\\", "/")
                        manifest["artifacts"].extend(artifacts)
                        manifest["completion_state"] = status
                        break

        digest = _write_stage(run_dir, stage, status, parent_hashes, details, artifacts)
        parent_hashes = [digest]
        manifest["stage_status"][stage] = status
        manifest["stage_manifests"][stage] = str(_stage_manifest(run_dir, stage).relative_to(ROOT)).replace("\\", "/")
        manifest["artifacts"].extend(artifacts)
        if status == "INPUT_REQUIRED" and profile == "full":
            manifest["completion_state"] = "INPUT_REQUIRED"
            break

    if manifest.get("completion_state") == "RUNNING":
        statuses = set(manifest["stage_status"].values())
        manifest["completion_state"] = "PASS" if statuses <= {"PASS"} else "PASS_WITH_SKIPS"
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
    return {"schema_version": "recursive_trace_v1", "run_id": run_id, "completion_state": manifest.get("completion_state"), "stages": manifest.get("stage_status", {}), "artifacts": artifacts, "external_parents": [a for a in artifacts if a.get("path", "").startswith("frozen/")]}
