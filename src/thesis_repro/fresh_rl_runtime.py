"""Thin adapters around the preserved V4.3 RL implementation.

This module is deliberately a dispatcher, not a second RL implementation.  It
binds the production Stage3--6 entry points to the run namespace and fails
closed when the authorized Stage2 input pack or the actor graph is absent.
Frozen checkpoints and frozen Stage2 tables are never copied or used here.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.rl.v43_runtime import stage_main

from .c3e import aggregate_hierarchical
from .fresh_rl import build_c3e_release, load_contract, verify_actor_graph
from .stages.base import StageResult, sha256_file, write_stage_artifact
from .status import FAILED, INPUT_REQUIRED


_RL_INPUTS = (
    "01_contract/encoder_contract.json",
    "01_contract/training_preprocessing.json",
    "01_contract/resolved_run_config.json",
    "02_data/statistics_fit_rows.parquet",
    "02_data/target_only_financial_accounts.parquet",
    "02_data/feature_manifest.json",
    "02_data/training_and_target_features.parquet",
    "02_data/stage3_rows.parquet",
    "02_data/stage4_rows.parquet",
    "02_data/stage5_rows.parquet",
    "04_validation/pretraining_gate.json",
)


@contextmanager
def _runtime_environment(paths):
    """Bind legacy producer path lookups for one call and restore the process."""
    names = ("CREDIT_REPRO_RUN_PATH", "THESIS_REPRO_RUN_ROOT")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["CREDIT_REPRO_RUN_PATH"] = str(paths.run_root)
    os.environ["THESIS_REPRO_RUN_ROOT"] = str(paths.run_root)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _required_input_report(paths) -> tuple[bool, list[str]]:
    missing = [name for name in _RL_INPUTS if not (paths.stage2_root / name).is_file()]
    return not missing, missing


def _artifact(paths, relative: str, logical_id: str, parents: list[str]) -> dict[str, Any]:
    path = paths.run_root / relative
    return {
        "logical_id": logical_id,
        "path": str(path.relative_to(paths.root)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "producer": "credit_recourse.rl.v43_runtime",
        "parents": [{"sha256": value} for value in parents],
    }


def _blocked(stage: str, parents: list[str], missing: list[str], reason: str) -> StageResult:
    return StageResult(
        stage,
        INPUT_REQUIRED,
        "REAL_COMPUTE",
        executed=False,
        parent_hashes=parents,
        details={"reason": reason, "missing_same_run_inputs": missing, "frozen_compute_fallback": False},
    )


def run_training_stage(paths, stage: str, parent_hashes: list[str], *, context=None) -> StageResult:
    ready, missing = _required_input_report(paths)
    if not ready:
        return _blocked(stage, parent_hashes, missing, "authorized fresh V4.3 Stage2 input pack is required")
    outputs = {
        "RLEncoder": (3, paths.run_root / "05_rl_encoder/final_epoch.pt", None, None),
        "RLBehaviorClone": (4, paths.run_root / "06_rl_bc/final_epoch.pt", paths.run_root / "05_rl_encoder/final_epoch.pt", None),
        "RLIQL": (5, paths.run_root / "07_rl_iql/final_epoch.pt", paths.run_root / "06_rl_bc/final_epoch.pt", (paths.stage2_root / "02_data/stage5_counterfactual.parquet", paths.stage2_root / "02_data/stage5_counterfactual_metadata.json")),
    }
    if stage not in outputs:
        raise ValueError(f"unsupported training stage: {stage}")
    production_stage, destination, predecessor, transition = outputs[stage]
    if predecessor is not None and not predecessor.is_file():
        return _blocked(stage, parent_hashes, [str(predecessor.relative_to(paths.root)).replace("\\", "/")], "same-run predecessor checkpoint is missing")
    if transition is not None and any(not path.is_file() for path in transition):
        return _blocked(stage, parent_hashes, [str(path.relative_to(paths.root)).replace("\\", "/") for path in transition], "same-run Stage5 counterfactual transition pack is missing")
    if destination.exists():
        return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "refusing to overwrite an existing fresh checkpoint"})
    argv = ["--project-root", str(paths.root), "--train", "--output", str(destination)]
    if predecessor is not None:
        argv.extend(["--encoder-checkpoint", str(predecessor)])
    if transition is not None:
        argv.extend(["--transition-input", str(transition[0]), "--transition-metadata", str(transition[1])])
    try:
        with _runtime_environment(paths):
            rc = int(stage_main(production_stage, argv))
        execution = destination.parent / "execution.json"
        if rc != 0 or not destination.is_file() or not execution.is_file():
            return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "V4.3 training entry point did not produce a completed checkpoint", "return_code": rc})
        execution_payload = json.loads(execution.read_text(encoding="utf-8"))
        if execution_payload.get("status") != "PASS":
            return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "V4.3 execution receipt is not PASS", "execution": execution_payload})
        artifacts = [_artifact(paths, str(destination.relative_to(paths.run_root)).replace("\\", "/"), f"fresh:{stage}:checkpoint", parent_hashes), _artifact(paths, str(execution.relative_to(paths.run_root)).replace("\\", "/"), f"fresh:{stage}:execution", parent_hashes)]
        return StageResult(stage, "PASS", "REAL_COMPUTE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"production_entrypoint": "credit_recourse.rl.v43_runtime.stage_main", "production_stage": production_stage, "execution": execution_payload})
    except FileNotFoundError as exc:
        return _blocked(stage, parent_hashes, [str(exc)], "V4.3 runtime input is unavailable")
    except Exception as exc:
        return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "V43_TRAINING_EXECUTION_FAILED"})


def _load_actor_probabilities(path: Path, contract: dict[str, Any]) -> dict[str, dict[int, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[int, np.ndarray]] = {}
    for configuration, members in payload.items():
        result[str(configuration)] = {int(seed): np.asarray(values, dtype=float) for seed, values in members.items()}
    return result


def _validate_trained_actor_checkpoints(records: list[dict[str, Any]]) -> list[str]:
    """Reject byte-addressable placeholders before C3-E can consume them."""
    import torch

    errors: list[str] = []
    for record in records:
        key = f"{record.get('configuration')}:{record.get('seed')}"
        path = Path(str(record.get("checkpoint_path", "")))
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            errors.append(f"checkpoint_unreadable:{key}:{type(exc).__name__}")
            continue
        if not isinstance(payload, dict) or payload.get("trained") is not True:
            errors.append(f"checkpoint_not_trained:{key}")
        if payload.get("producer_stage") != "Stage5":
            errors.append(f"checkpoint_wrong_stage:{key}")
        if not isinstance(payload.get("policy_state_dict"), dict):
            errors.append(f"checkpoint_missing_policy_state:{key}")
    return errors


def run_c3e(paths, parent_hashes: list[str], *, context=None) -> StageResult:
    graph_path = paths.run_root / "07_rl_iql/actor_graph.json"
    probabilities_path = paths.run_root / "07_rl_iql/actor_probabilities.json"
    evaluation_manifest = paths.stage3_root / "evaluation_manifest.json"
    missing = [str(path.relative_to(paths.root)).replace("\\", "/") for path in (graph_path, probabilities_path, evaluation_manifest) if not path.is_file()]
    if missing:
        return _blocked("C3E", parent_hashes, missing, "fresh 28-actor graph and probabilities are required")
    try:
        contract = load_contract(paths.root)
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        records = graph if isinstance(graph, list) else graph.get("actors", [])
        cohort = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
        cohort_hash = hashlib.sha256(json.dumps(cohort, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        graph_report = verify_actor_graph(records, contract, root=paths.root, evaluation_cohort_hash=cohort_hash)
        checkpoint_errors = _validate_trained_actor_checkpoints(records)
        if checkpoint_errors:
            graph_report["errors"].extend(checkpoint_errors)
            graph_report["status"] = "FAILED"
        if graph_report["status"] != "PASS":
            return StageResult("C3E", FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "fresh actor graph verification failed", "verification": graph_report})
        probabilities = _load_actor_probabilities(probabilities_path, contract)
        final, receipt = build_c3e_release(probabilities, contract, parent_hashes=parent_hashes, evaluation_cohort_hash=cohort_hash)
        paths.c3e_root.mkdir(parents=True, exist_ok=True)
        np.save(paths.c3e_root / "ensemble_probabilities.npy", final)
        actions = np.argmax(final, axis=1).astype(int)
        pd.DataFrame({"evaluation_ordinal": np.arange(len(actions)), "action_index": actions}).to_parquet(paths.c3e_root / "c3e_actions.parquet", index=False)
        receipt.update({"actor_graph_sha256": sha256_file(graph_path), "actor_probabilities_sha256": sha256_file(probabilities_path), "evaluation_manifest_sha256": sha256_file(evaluation_manifest), "output_sha256": sha256_file(paths.c3e_root / "ensemble_probabilities.npy")})
        release = paths.c3e_root / "release.json"
        release.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        current = paths.c3e_root / "CURRENT_RELEASE.json"
        current.write_text(json.dumps({"release_id": paths.run_id, "release_hash": receipt["release_hash"], "payoff_surface_sha256": receipt["output_sha256"]}, indent=2) + "\n", encoding="utf-8")
        artifacts = [_artifact(paths, "08_c3e/ensemble_probabilities.npy", "fresh:c3e:ensemble", parent_hashes), _artifact(paths, "08_c3e/c3e_actions.parquet", "fresh:c3e:actions", parent_hashes), _artifact(paths, "08_c3e/release.json", "fresh:c3e:release", parent_hashes), _artifact(paths, "08_c3e/CURRENT_RELEASE.json", "fresh:c3e:pointer", parent_hashes)]
        return StageResult("C3E", "PASS", "REAL_COMPUTE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"verification": graph_report, "release": receipt, "aggregation": contract["aggregation"]})
    except Exception as exc:
        return StageResult("C3E", FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "C3E_EXECUTION_FAILED"})


def run_verify_rl(paths, parent_hashes: list[str], *, context=None) -> StageResult:
    graph_path = paths.run_root / "07_rl_iql/actor_graph.json"
    release_path = paths.c3e_root / "release.json"
    if not graph_path.is_file() or not release_path.is_file():
        return _blocked("VerifyRL", parent_hashes, [str(path.relative_to(paths.root)).replace("\\", "/") for path in (graph_path, release_path) if not path.is_file()], "same-run actor graph and C3-E release are required")
    try:
        contract = load_contract(paths.root)
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        records = graph if isinstance(graph, list) else graph.get("actors", [])
        report = verify_actor_graph(records, contract, root=paths.root)
        release = json.loads(release_path.read_text(encoding="utf-8"))
        if report["status"] != "PASS" or release.get("status") != "PASS":
            return StageResult("VerifyRL", FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"verification": report, "release_status": release.get("status")})
        verification = {"schema_version": "fresh_rl_verification_v1", "status": "PASS", "actor_graph": report, "c3e_release_hash": release.get("release_hash"), "parent_hashes": parent_hashes}
        artifact = write_stage_artifact(paths, "08_c3e/rl_validation_report.json", verification, "fresh:rl:validation", ({"sha256": value} for value in parent_hashes))
        return StageResult("VerifyRL", "PASS", "REAL_COMPUTE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details={"verification": verification})
    except Exception as exc:
        return StageResult("VerifyRL", FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "RL_VERIFICATION_FAILED"})
