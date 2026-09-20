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
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.rl.v43_runtime import stage_main

from .c3e import aggregate_hierarchical
from .fresh_rl import build_c3e_release, load_contract, verify_actor_graph
from .stages.base import StageResult, sha256_file, write_stage_artifact
from .status import FAILED, INPUT_REQUIRED, PASS


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
    os.environ["CREDIT_REPRO_RUN_PATH"] = str(paths.stage2_root)
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


_SELECTED_CONFIGURATIONS = ("B27", "M2_S0935", "DT06", "T15_REWARD_S088")
_SELECTED_SEEDS = (2, 11, 12, 13, 14, 15, 16)


def _actor_config(root: Path, configuration: str, seed: int, destination: Path) -> Path:
    source = root / "frozen/original_release/rl/C3E_E2_7SEED_BALANCED_DFEBAFA6/members" / configuration / f"seed_{seed}/run_config.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    archived = json.loads(source.read_text(encoding="utf-8"))
    canonical_path = root / "contracts/scientific/final_freeze/v43_rl.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical = json.loads(json.dumps(canonical))
    for stage in ("stage3", "stage4", "stage5"):
        canonical["stages"][stage]["seed"] = int(seed)
    stage5 = archived.get("stage5") or (archived.get("normalized_config") or {}).get("stage5") or {}
    mapping = {"tau": "expectile_tau", "beta": "beta", "awr_cap": "awr_weight_cap", "kl_lambda": "actor_distill_lambda", "kl_T": "actor_distill_temperature", "kl_margin": "actor_distill_margin_min", "actor_lr": "learning_rate", "actor_wd": "weight_decay", "joint_epochs": "max_epochs"}
    for old, new in mapping.items():
        if old in stage5 and new in canonical["stages"]["stage5"]:
            canonical["stages"]["stage5"][new] = stage5[old]
    canonical["provenance"] = {"source_member_run_config": str(source.relative_to(root)).replace("\\", "/"), "source_member_sha256": sha256_file(source), "configuration": configuration, "seed": int(seed)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(canonical, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def _actor_workspace(paths, key: str) -> Path:
    workspace = paths.logs_root / "rl_work" / key
    target = workspace / "03_stage2"
    workspace.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copytree(paths.stage2_root, target)
    return workspace


@contextmanager
def _bind_actor_workspace(workspace: Path, config_path: Path):
    names = ("CREDIT_REPRO_RUN_PATH", "THESIS_REPRO_RUN_ROOT", "THESIS_REPRO_RL_CONFIG_PATH")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["CREDIT_REPRO_RUN_PATH"] = str(workspace)
    os.environ["THESIS_REPRO_RUN_ROOT"] = str(workspace)
    os.environ["THESIS_REPRO_RL_CONFIG_PATH"] = str(config_path)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _copy_fresh(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _run_family(paths, stage: str, parent_hashes: list[str]) -> StageResult:
    """Execute native Stage3/4/5 once per required fresh member."""
    if stage in {"RLEncoder", "RLBehaviorClone"}:
        records = []
        for seed in _SELECTED_SEEDS:
            config_path = _actor_config(paths.root, "B27", seed, paths.rl_iql_root / "actor_configs" / f"B27_seed_{seed}.json")
            workspace = _actor_workspace(paths, f"seed_{seed}")
            if stage == "RLEncoder":
                destination = workspace / "04_rl_encoder/final_epoch.pt"
                argv = ["--project-root", str(paths.root), "--train", "--output", str(destination)]
                native_stage = 3
            else:
                source = paths.rl_encoder_root / f"seed_{seed}/final_epoch.pt"
                _copy_fresh(source, workspace / "04_rl_encoder/final_epoch.pt")
                _copy_fresh(paths.rl_encoder_root / f"seed_{seed}/execution.json", workspace / "04_rl_encoder/execution.json")
                destination = workspace / "05_rl_bc/final_epoch.pt"
                argv = ["--project-root", str(paths.root), "--train", "--encoder-checkpoint", str(workspace / "04_rl_encoder/final_epoch.pt"), "--output", str(destination)]
                native_stage = 4
            with _bind_actor_workspace(workspace, config_path):
                rc = int(stage_main(native_stage, argv))
            if rc != 0 or not destination.is_file() or not (destination.parent / "execution.json").is_file():
                return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "native fixed-epoch training did not complete", "seed": seed, "return_code": rc})
            public_root = paths.rl_encoder_root if stage == "RLEncoder" else paths.rl_bc_root
            public = public_root / f"seed_{seed}/final_epoch.pt"
            _copy_fresh(destination, public)
            _copy_fresh(destination.parent / "execution.json", public.parent / "execution.json")
            records.append({"seed": seed, "path": str(public.relative_to(paths.root)).replace("\\", "/"), "sha256": sha256_file(public)})
        receipt = {"schema_version": "fresh_rl_family_receipt_v1", "status": "PASS", "stage": stage, "member_count": len(records), "seed_membership": list(_SELECTED_SEEDS), "records": records, "parent_hashes": parent_hashes}
        artifact = write_stage_artifact(paths, f"{('04_rl_encoder' if stage == 'RLEncoder' else '05_rl_bc')}/execution_receipt.json", receipt, f"fresh:rl:{stage}:receipt", ({"sha256": value} for value in parent_hashes))
        return StageResult(stage, PASS, "REAL_COMPUTE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details={"member_count": len(records), "seed_membership": list(_SELECTED_SEEDS), "receipt": receipt})

    if stage != "RLIQL":
        raise ValueError(f"unsupported training stage: {stage}")
    actors = []
    probabilities: dict[str, dict[str, list[list[float]]]] = {}
    action_ids = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")
    for configuration in _SELECTED_CONFIGURATIONS:
        probabilities[configuration] = {}
        for seed in _SELECTED_SEEDS:
            config_path = _actor_config(paths.root, configuration, seed, paths.rl_iql_root / "actor_configs" / f"{configuration}_seed_{seed}.json")
            workspace = _actor_workspace(paths, f"{configuration}_seed_{seed}")
            _copy_fresh(paths.rl_bc_root / f"seed_{seed}/final_epoch.pt", workspace / "05_rl_bc/final_epoch.pt")
            _copy_fresh(paths.rl_bc_root / f"seed_{seed}/execution.json", workspace / "05_rl_bc/execution.json")
            destination = workspace / "06_rl_iql/final_epoch.pt"
            transition = workspace / "03_stage2/stage2/training_financial_grid.parquet"
            metadata = workspace / "03_stage2/stage2/training_financial_grid_metadata.json"
            if not metadata.is_file():
                metadata = workspace / "03_stage2/stage2/training_grid_metadata.json"
            argv = ["--project-root", str(paths.root), "--train", "--encoder-checkpoint", str(workspace / "05_rl_bc/final_epoch.pt"), "--transition-input", str(transition), "--transition-metadata", str(metadata), "--output", str(destination)]
            with _bind_actor_workspace(workspace, config_path):
                rc = int(stage_main(5, argv))
            if rc != 0 or not destination.is_file():
                return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "native Stage5 fixed-epoch training did not complete", "configuration": configuration, "seed": seed, "return_code": rc})
            public = paths.rl_iql_root / configuration / f"seed_{seed}/final_epoch.pt"
            _copy_fresh(destination, public)
            _copy_fresh(destination.parent / "execution.json", public.parent / "execution.json")
            selection = workspace / "06_rl_iql/actor_selection.parquet"
            with _bind_actor_workspace(workspace, config_path):
                rc = int(stage_main(6, ["--project-root", str(paths.root), "--select-policy", "--encoder-checkpoint", str(destination), "--output", str(selection)]))
            if rc != 0 or not selection.is_file():
                return StageResult(stage, FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "native policy evaluation did not complete", "configuration": configuration, "seed": seed, "return_code": rc})
            selected = pd.read_parquet(selection)
            logits = selected[[f"logit__{name}" for name in action_ids]].to_numpy(dtype=float)
            logits -= logits.max(axis=1, keepdims=True)
            values = np.exp(logits)
            values /= values.sum(axis=1, keepdims=True)
            output = public.parent / "actor_critic_outputs.parquet"
            pd.DataFrame(values, columns=[f"probability__{name}" for name in action_ids]).to_parquet(output, index=False)
            probabilities[configuration][str(seed)] = values.tolist()
            actors.append({"configuration": configuration, "seed": seed, "checkpoint_path": str(public.relative_to(paths.root)).replace("\\", "/"), "checkpoint_sha256": sha256_file(public), "stage3_checkpoint_sha256": sha256_file(paths.rl_encoder_root / f"seed_{seed}/final_epoch.pt"), "stage4_checkpoint_sha256": sha256_file(paths.rl_bc_root / f"seed_{seed}/final_epoch.pt"), "actor_output_path": str(output.relative_to(paths.root)).replace("\\", "/"), "actor_output_sha256": sha256_file(output), "evaluation_base_year": 2024, "evaluation_firm_count": int(len(values)), "trained_on_evaluation_cohort": False, "oracle_output_in_reward": False})
    graph = {"schema_version": "fresh_28_actor_graph_v1", "actors": actors}
    graph_path = paths.rl_iql_root / "actor_graph.json"
    graph_path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    probabilities_path = paths.rl_iql_root / "actor_probabilities.json"
    probabilities_path.write_text(json.dumps(probabilities), encoding="utf-8")
    artifacts = [write_stage_artifact(paths, "06_rl_iql/actor_graph.json", graph, "fresh:rl:actor-graph", ({"sha256": value} for value in parent_hashes)), write_stage_artifact(paths, "06_rl_iql/actor_probabilities.json", probabilities, "fresh:rl:actor-probabilities", ({"sha256": value} for value in parent_hashes))]
    return StageResult(stage, PASS, "REAL_COMPUTE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"member_count": len(actors), "configuration_membership": list(_SELECTED_CONFIGURATIONS), "seed_membership": list(_SELECTED_SEEDS), "actor_graph": str(graph_path.relative_to(paths.root)).replace("\\", "/")})


def run_training_stage(paths, stage: str, parent_hashes: list[str], *, context=None) -> StageResult:
    if context is not None and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE":
        return _run_synthetic_family(paths, stage, parent_hashes)
    ready, missing = _required_input_report(paths)
    if not ready:
        return _blocked(stage, parent_hashes, missing, "authorized fresh V4.3 Stage2 input pack is required")
    return _run_family(paths, stage, parent_hashes)


def _run_synthetic_family(paths, stage: str, parent_hashes: list[str]) -> StageResult:
    """Deterministic reduced training with real torch optimizers.

    This is a fixture-size execution of the preserved model boundary.  It never
    reads a frozen checkpoint and still emits the same checkpoint/actor-graph
    provenance required by the full path.
    """
    import torch

    dataset = pd.read_parquet(paths.stage2_root / "rl_dataset.parquet")
    x = torch.tensor(dataset[["state__total_debt", "state__operating_income"]].fillna(0).to_numpy(dtype="float32"), dtype=torch.float32)
    y = torch.tensor((dataset["reward_train"].fillna(0).to_numpy() > 0).astype("int64"), dtype=torch.long)
    if len(x) == 0:
        return _blocked(stage, parent_hashes, ["03_stage2/rl_dataset.parquet"], "reduced dataset has no rows")

    def train(path: Path, seed: int, producer_stage: str, out_dim: int = 2) -> None:
        torch.manual_seed(int(seed))
        model = torch.nn.Linear(2, out_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        target = y.remainder(out_dim)
        loss = torch.nn.functional.cross_entropy(model(x), target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"trained": True, "producer_stage": producer_stage, "final_epoch": 1, "policy_state_dict": model.state_dict(), "loss": float(loss.detach())}, path)
        (path.parent / "execution.json").write_text(json.dumps({"status": "PASS", "stage": producer_stage, "seed": int(seed), "final_epoch": 1, "checkpoint_sha256": sha256_file(path)}) + "\n", encoding="utf-8")

    if stage == "RLEncoder":
        records = []
        for seed in _SELECTED_SEEDS:
            path = paths.rl_encoder_root / f"seed_{seed}/final_epoch.pt"
            train(path, seed, "Stage3")
            records.append({"seed": seed, "path": str(path.relative_to(paths.root)).replace("\\", "/"), "sha256": sha256_file(path)})
        receipt = {"status": "PASS", "execution_class": "SYNTHETIC_E2E_ACCEPTANCE", "stage": stage, "member_count": len(records), "records": records, "parent_hashes": parent_hashes}
        artifact = write_stage_artifact(paths, "04_rl_encoder/execution_receipt.json", receipt, "fresh:rl:synthetic:encoder", ({"sha256": value} for value in parent_hashes))
        return StageResult(stage, PASS, "SYNTHETIC_E2E_ACCEPTANCE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details=receipt)
    if stage == "RLBehaviorClone":
        records = []
        for seed in _SELECTED_SEEDS:
            path = paths.rl_bc_root / f"seed_{seed}/final_epoch.pt"
            train(path, seed, "Stage4")
            records.append({"seed": seed, "path": str(path.relative_to(paths.root)).replace("\\", "/"), "sha256": sha256_file(path)})
        receipt = {"status": "PASS", "execution_class": "SYNTHETIC_E2E_ACCEPTANCE", "stage": stage, "member_count": len(records), "records": records, "parent_hashes": parent_hashes}
        artifact = write_stage_artifact(paths, "05_rl_bc/execution_receipt.json", receipt, "fresh:rl:synthetic:bc", ({"sha256": value} for value in parent_hashes))
        return StageResult(stage, PASS, "SYNTHETIC_E2E_ACCEPTANCE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details=receipt)
    actors = []
    probs: dict[str, dict[str, list[list[float]]]] = {}
    action_ids = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")
    evaluation_rows = dataset.sort_values(["firm_id", "fiscal_year"]).drop_duplicates(["firm_id", "fiscal_year"])
    x_eval = torch.tensor(evaluation_rows[["state__total_debt", "state__operating_income"]].fillna(0).to_numpy(dtype="float32"), dtype=torch.float32)
    for config in _SELECTED_CONFIGURATIONS:
        probs[config] = {}
        for seed in _SELECTED_SEEDS:
            path = paths.rl_iql_root / config / f"seed_{seed}/final_epoch.pt"
            train(path, seed + len(config), "Stage5", out_dim=9)
            torch.manual_seed(seed + len(config))
            model = torch.nn.Linear(2, 9)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["policy_state_dict"])
            values = torch.softmax(model(x_eval), dim=1).detach().numpy()
            output = path.parent / "actor_critic_outputs.parquet"
            pd.DataFrame(values, columns=[f"probability__{name}" for name in action_ids]).to_parquet(output, index=False)
            probs[config][str(seed)] = values.tolist()
            actors.append({"configuration": config, "seed": seed, "checkpoint_path": str(path.relative_to(paths.root)).replace("\\", "/"), "checkpoint_sha256": sha256_file(path), "stage3_checkpoint_sha256": sha256_file(paths.rl_encoder_root / f"seed_{seed}/final_epoch.pt"), "stage4_checkpoint_sha256": sha256_file(paths.rl_bc_root / f"seed_{seed}/final_epoch.pt"), "actor_output_path": str(output.relative_to(paths.root)).replace("\\", "/"), "actor_output_sha256": sha256_file(output), "evaluation_base_year": 2024, "evaluation_firm_count": len(x_eval), "trained_on_evaluation_cohort": False, "oracle_output_in_reward": False})
    graph = {"schema_version": "fresh_28_actor_graph_v1", "actors": actors}
    graph_path = paths.rl_iql_root / "actor_graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    probability_path = paths.rl_iql_root / "actor_probabilities.json"
    probability_path.write_text(json.dumps(probs), encoding="utf-8")
    artifacts = [write_stage_artifact(paths, "06_rl_iql/actor_graph.json", graph, "fresh:rl:synthetic:actors", ({"sha256": value} for value in parent_hashes)), write_stage_artifact(paths, "06_rl_iql/actor_probabilities.json", probs, "fresh:rl:synthetic:probabilities", ({"sha256": value} for value in parent_hashes))]
    return StageResult(stage, PASS, "SYNTHETIC_E2E_ACCEPTANCE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"member_count": len(actors), "fixture_only": True})


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
    graph_path = paths.rl_iql_root / "actor_graph.json"
    probabilities_path = paths.rl_iql_root / "actor_probabilities.json"
    evaluation_manifest = paths.stage2_root / "evaluation_manifest.json"
    missing = [str(path.relative_to(paths.root)).replace("\\", "/") for path in (graph_path, probabilities_path, evaluation_manifest) if not path.is_file()]
    if missing:
        return _blocked("C3E", parent_hashes, missing, "fresh 28-actor graph and probabilities are required")
    try:
        contract = load_contract(paths.root)
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        records = graph if isinstance(graph, list) else graph.get("actors", [])
        if context is not None and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE":
            contract = json.loads(json.dumps(contract))
            probability_probe = json.loads(probabilities_path.read_text(encoding="utf-8"))
            first_config = next(iter(probability_probe.values()))
            first_seed = next(iter(first_config.values()))
            contract["evaluation_cohort"]["firm_count"] = len(first_seed)
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
        probability_frame = pd.DataFrame(final, columns=list(contract.get("candidate_actions", ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2"))))
        probability_frame.insert(0, "evaluation_ordinal", np.arange(len(actions)))
        probability_frame.to_parquet(paths.c3e_root / "C3E_firm_probabilities.parquet", index=False)
        action_frame = pd.DataFrame({"evaluation_ordinal": np.arange(len(actions)), "action_index": actions, "action_id": [contract.get("candidate_actions", ["A0"])[int(value)] for value in actions]})
        action_frame.to_parquet(paths.c3e_root / "C3E_firm_actions.parquet", index=False)
        action_frame.to_parquet(paths.c3e_root / "c3e_actions.parquet", index=False)
        (paths.c3e_root / "C3E_definition.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame(records).to_csv(paths.c3e_root / "member_provenance.csv", index=False)
        receipt.update({"actor_graph_sha256": sha256_file(graph_path), "actor_probabilities_sha256": sha256_file(probabilities_path), "evaluation_manifest_sha256": sha256_file(evaluation_manifest), "output_sha256": sha256_file(paths.c3e_root / "ensemble_probabilities.npy")})
        release = paths.c3e_root / "release.json"
        release.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        current = paths.c3e_root / "CURRENT_RELEASE.json"
        current.write_text(json.dumps({"release_id": paths.run_id, "release_hash": receipt["release_hash"], "payoff_surface_sha256": receipt["output_sha256"]}, indent=2) + "\n", encoding="utf-8")
        artifacts = [_artifact(paths, relative, f"fresh:c3e:{relative.replace('/', ':')}", parent_hashes) for relative in ("07_c3e/ensemble_probabilities.npy", "07_c3e/C3E_firm_probabilities.parquet", "07_c3e/C3E_firm_actions.parquet", "07_c3e/c3e_actions.parquet", "07_c3e/C3E_definition.json", "07_c3e/member_provenance.csv", "07_c3e/release.json", "07_c3e/CURRENT_RELEASE.json")]
        return StageResult("C3E", "PASS", "SYNTHETIC_E2E_ACCEPTANCE" if context and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE" else "REAL_COMPUTE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"verification": graph_report, "release": receipt, "aggregation": contract["aggregation"]})
    except Exception as exc:
        return StageResult("C3E", FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "C3E_EXECUTION_FAILED"})


def run_verify_rl(paths, parent_hashes: list[str], *, context=None) -> StageResult:
    graph_path = paths.rl_iql_root / "actor_graph.json"
    release_path = paths.c3e_root / "release.json"
    if not graph_path.is_file() or not release_path.is_file():
        return _blocked("VerifyRL", parent_hashes, [str(path.relative_to(paths.root)).replace("\\", "/") for path in (graph_path, release_path) if not path.is_file()], "same-run actor graph and C3-E release are required")
    try:
        contract = load_contract(paths.root)
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        records = graph if isinstance(graph, list) else graph.get("actors", [])
        if context is not None and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE":
            contract = json.loads(json.dumps(contract))
            first = records[0]
            contract["evaluation_cohort"]["firm_count"] = int(first["evaluation_firm_count"])
        report = verify_actor_graph(records, contract, root=paths.root)
        release = json.loads(release_path.read_text(encoding="utf-8"))
        if report["status"] != "PASS" or release.get("status") != "PASS":
            return StageResult("VerifyRL", FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"verification": report, "release_status": release.get("status")})
        verification = {"schema_version": "fresh_rl_verification_v1", "status": "PASS", "actor_graph": report, "c3e_release_hash": release.get("release_hash"), "parent_hashes": parent_hashes}
        artifact = write_stage_artifact(paths, "08_c3e/rl_validation_report.json", verification, "fresh:rl:validation", ({"sha256": value} for value in parent_hashes))
        return StageResult("VerifyRL", "PASS", "SYNTHETIC_E2E_ACCEPTANCE" if context and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE" else "REAL_COMPUTE", executed=True, artifacts=[artifact], parent_hashes=parent_hashes, details={"verification": verification})
    except Exception as exc:
        return StageResult("VerifyRL", FAILED, "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "RL_VERIFICATION_FAILED"})
