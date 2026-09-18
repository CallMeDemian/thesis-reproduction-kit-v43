from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import EXPECTED_REQUESTS, contract_report, render_requests
from .paths import ROOT, load_json, sha256_file, write_json

STAGES = {
    "OracleClean": ["VerifyInputs", "Oracle", "VerifyOracle"],
    "OracleRLClean": ["VerifyInputs", "Oracle", "Simulator", "RLDataset", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL"],
    "OracleRLLLMClean": ["VerifyInputs", "Oracle", "Simulator", "RLDataset", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL", "LLMPrepare", "LLMGenerate", "LLMMaterialize", "Stage8", "Stage9", "VerifyResults"],
    "FullClean": ["VerifyInputs", "Oracle", "Simulator", "RLDataset", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL", "LLMPrepare", "LLMGenerate", "LLMMaterialize", "Stage8", "Stage9", "VerifyResults", "ThesisOutputs", "CompareFrozen", "VerifyAll"],
}
STAGE_DIRS = {
    "VerifyInputs": "01_inputs", "Oracle": "02_oracle", "VerifyOracle": "02_oracle", "Simulator": "03_simulator", "RLDataset": "04_rl", "RLEncoder": "04_rl", "RLBehaviorClone": "04_rl", "RLIQL": "04_rl", "C3E": "05_c3e", "Stage6": "05_c3e", "VerifyRL": "05_c3e", "LLMPrepare": "06_llm", "LLMGenerate": "06_llm", "LLMMaterialize": "06_llm", "Stage8": "07_stage8", "Stage9": "08_stage9", "VerifyResults": "09_results", "ThesisOutputs": "10_thesis_outputs", "CompareFrozen": "11_comparison", "VerifyAll": "11_comparison",
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNCOMMITTED"


def _contract_hash() -> str:
    payload = load_json(ROOT / "contracts/llm/final_as_executed_generation_contract.json")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _hash_object(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def plan(mode: str, run_id: str, profile: str) -> dict[str, Any]:
    if mode not in STAGES:
        raise ValueError(f"unknown mode: {mode}")
    return {"run_id": run_id, "mode": mode, "profile": profile, "stages": STAGES[mode], "resources": {"profile": profile, "gpu_required": profile == "full" and mode in {"OracleRLClean", "OracleRLLLMClean", "FullClean"}, "live_api": False}, "expected_outputs": [f"{STAGE_DIRS[stage]}/stage_manifest.json" for stage in STAGES[mode]], "contract_hash": _contract_hash()}


def _stage_manifest(run_dir: Path, stage: str) -> Path:
    return run_dir / STAGE_DIRS[stage] / f"stage_manifest_{stage}.json"


def _write_stage(run_dir: Path, stage: str, status: str, parent_hashes: list[str], details: dict[str, Any] | None = None) -> str:
    payload = {"schema_version": "stage_manifest_v1", "stage": stage, "status": status, "run_id": run_dir.name, "created_at": datetime.now(timezone.utc).isoformat(), "contract_hash": _contract_hash(), "parent_hashes": parent_hashes, "details": details or {}}
    path = _stage_manifest(run_dir, stage)
    write_json(path, payload)
    return sha256_file(path)


def render_dry_run(run_dir: Path, full: bool = True) -> dict[str, Any]:
    report = contract_report()
    if report["generated"] != EXPECTED_REQUESTS or report["unique_ids"] != EXPECTED_REQUESTS:
        raise ValueError(f"LLM dry-render cardinality failure: {report}")
    request_path = run_dir / "06_llm" / "logical_requests.jsonl"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = 0
    with request_path.open("w", encoding="utf-8") as handle:
        values = render_requests() if full else render_requests(limit=25)
        for value in values:
            handle.write(__import__("json").dumps(value, ensure_ascii=False) + "\n")
            rendered += 1
    report["rendered_to_disk"] = rendered
    report["dry_run_mode"] = "full" if full else "smoke"
    report["status"] = "PASS" if full and rendered == EXPECTED_REQUESTS else "PASS_WITH_SKIPS"
    write_json(run_dir / "06_llm/dry_run_manifest.json", report)
    return report


def execute(mode: str, run_id: str, profile: str = "smoke", resume: bool = False, from_stage: str | None = None, to_stage: str | None = None, execute_llm: bool = False, dry_render: bool = False) -> dict[str, Any]:
    if mode not in STAGES:
        raise ValueError(f"unknown mode: {mode}")
    if execute_llm and os.environ.get("THESIS_REPRO_ENABLE_LIVE_LLM") != "I_APPROVE_FRESH_REPLICATION":
        raise PermissionError("live LLM requires THESIS_REPRO_ENABLE_LIVE_LLM=I_APPROVE_FRESH_REPLICATION")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = STAGES[mode]
    selected = stages[:]
    if from_stage:
        selected = selected[selected.index(from_stage):]
    if to_stage:
        selected = selected[: selected.index(to_stage) + 1]
    manifest_path = run_dir / "run_manifest.json"
    manifest = load_json(manifest_path) if resume and manifest_path.is_file() else {"schema_version": "run_manifest_v1", "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(), "mode": mode, "profile": profile, "input_hashes": {}, "scientific_contract_hash": _contract_hash(), "random_seeds": {"inference": 73019}, "stage_status": {}, "stage_outputs": {}, "completion_state": "RUNNING"}
    if manifest.get("scientific_contract_hash") != _contract_hash():
        manifest["completion_state"] = "INCOMPLETE"
        write_json(manifest_path, manifest)
        raise ValueError("scientific contract hash changed; downstream stages invalidated")
    parent_hashes: list[str] = []
    for stage in selected:
        existing = _stage_manifest(run_dir, stage)
        if resume and existing.is_file():
            old = load_json(existing)
            if old.get("status") in {"PASS", "PASS_WITH_SKIPS"} and old.get("contract_hash") == _contract_hash():
                digest = sha256_file(existing)
                parent_hashes = [digest]
                manifest["stage_status"][stage] = old["status"]
                continue
        status = "PASS"
        details: dict[str, Any] = {"profile": profile, "implementation": "synthetic smoke harness" if profile == "smoke" else "foundation placeholder"}
        if stage == "VerifyInputs":
            receipt = ROOT / "data/raw/input_receipt.json"
            details["raw_data_present"] = receipt.is_file()
            if profile == "full" and not receipt.is_file():
                status = "INCOMPLETE"
        if stage == "Oracle" and profile == "full":
            status = "INCOMPLETE"
            details["reason"] = "Oracle full clean runner is not yet wired to the restored raw input adapters."
        if stage == "LLMGenerate":
            dry = render_dry_run(run_dir, full=dry_render or profile == "full")
            details.update(dry)
            status = "PASS_WITH_SKIPS" if not execute_llm else "INCOMPLETE"
            if execute_llm:
                details["reason"] = "live provider execution is gated until the historical source snapshot gap is closed"
        if stage == "LLMMaterialize" and not execute_llm:
            status = "SKIPPED"
        if stage in {"Stage8", "Stage9", "ThesisOutputs", "CompareFrozen", "VerifyAll"} and not execute_llm:
            status = "SKIPPED"
        digest = _write_stage(run_dir, stage, status, parent_hashes, details)
        parent_hashes = [digest]
        manifest["stage_status"][stage] = status
        manifest["stage_outputs"][stage] = str(_stage_manifest(run_dir, stage).relative_to(ROOT))
        if status == "INCOMPLETE":
            manifest["completion_state"] = "INCOMPLETE"
            break
    if manifest.get("completion_state") == "RUNNING":
        statuses = set(manifest["stage_status"].values())
        manifest["completion_state"] = "PASS" if statuses <= {"PASS"} else "PASS_WITH_SKIPS"
    write_json(manifest_path, manifest)
    return manifest
