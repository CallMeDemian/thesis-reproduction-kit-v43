"""Executable runtime for the recovered historical V4.3 Stage5 contract."""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from credit_recourse.eval.v43_stage6_reporting import join_choices, summarize_policies, validate_actor
from credit_recourse.rl.contracts.v43_encoder import ACTION_IDS, KEYS, file_sha256
from credit_recourse.rl.v43_actions import V43ActionCodec
from credit_recourse.rl.v43_checkpoints import policy_payload
from credit_recourse.rl.v43_historical_stage5 import (
    CONTRACT_BUNDLE, HistoricalStage5Config, HistoricalStage5ContractError,
    candidate_iql_loss_parts, experiment_identity, make_grouped_adamw,
    recompose_historical_reward, update_schedule, validate_contract_bundle,
)
from credit_recourse.rl.v43_model import V43Policy
from credit_recourse.rl.v43_runtime import V43StageConsumer
from credit_recourse.rl.v43_optimizer_audit import finite_gradients, parameter_digest, update_evidence
from credit_recourse.rl.v43_stage5_search import load_baseline
from credit_recourse.rl.v43_training import load_counterfactual
from credit_recourse.contracts.stage_paths import V43_STAGE5_SEARCH_PATH


ACTIONS = list(ACTION_IDS)
TOL = 1e-10


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False,
                  default=lambda item: item.item() if hasattr(item, "item") else str(item))
        handle.write("\n")


class Heartbeat:
    def __init__(self, run_folder: Path, search_root: Path, phase: str, max_epochs: int = 0):
        self.run_folder, self.search_root, self.phase = Path(run_folder), Path(search_root), phase
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.state = {"run": self.run_folder.name, "phase": phase, "status": "RUNNING",
                      "epoch": 0, "max_epochs": max_epochs}

    def update(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values)

    def emit(self, event: str) -> None:
        with self.lock:
            item = {**self.state, "timestamp": utcnow(), "elapsed_seconds": time.monotonic() - self.started,
                    "event": event}
        line = "HEARTBEAT " + json.dumps(item, ensure_ascii=True, allow_nan=False)
        for path in (self.run_folder / "logs/heartbeat.log", self.search_root / "SEARCH_HEARTBEAT.log"):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        write_json(self.run_folder / "logs/current_status.json", item)
        print(line, flush=True)

    def _worker(self) -> None:
        while not self.stop.wait(1200):
            self.emit("PERIODIC")

    def __enter__(self) -> "Heartbeat":
        self.emit("PHASE_START")
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, typ, value, tb) -> None:
        self.stop.set(); self.thread.join(timeout=5)
        self.update(status="FAIL" if typ else "COMPLETE", error=str(value) if typ else None)
        self.emit("FAIL" if typ else "PHASE_COMPLETE")


class HistoricalSearchConsumer(V43StageConsumer):
    """Delegate frozen Stage0-4 inputs while owning an independent Stage5 namespace."""
    def __init__(self, root: Path, folder: Path, config: HistoricalStage5Config,
                 baseline: dict[str, Any], identity: dict[str, Any]):
        base = V43StageConsumer(root)
        self.baseline_consumer = base
        self.__dict__.update(base.__dict__)
        self.baseline_folder = self.folder
        self.folder = Path(folder)
        self.search_config = config
        self.search_baseline = baseline
        self.identity = identity
        self.config = json.loads(json.dumps(base.config))
        s = self.config["stages"]["stage5"]
        s.update({
            "seed": config.seed, "max_epochs": config.joint_epochs, "batch_size": config.batch_size,
            "gamma": 0.0, "cql_alpha": 0.0, "expectile_tau": config.tau, "beta": config.beta,
            "awr_weight_cap": config.awr_cap, "actor_distill_mode": "kl",
            "actor_distill_lambda": config.kl_lambda, "actor_distill_temperature": config.kl_temperature,
            "actor_distill_margin_min": config.kl_margin, "q_lr": config.q_lr, "q_wd": config.q_wd,
            "value_lr": config.value_lr, "value_wd": config.value_wd, "actor_lr": config.actor_lr,
            "actor_wd": config.actor_wd, "smooth_l1_beta": config.smooth_l1_beta,
            "warmup_epochs": config.warmup_epochs, "actor_only_epochs": config.actor_only_epochs,
            "critic_update_ratio": config.critic_update_ratio, "encoder_finetune": False,
        })
        self.config["stages"]["stage2"].update(
            rho=0.0, merton_lambda=config.reward_merton, fcff_lambda=config.reward_fcff,
            liquidity_lambda=config.reward_liquidity, profitability_lambda=config.reward_profitability,
        )
        self.config["seed"] = config.seed
        self.config["config_hash"] = identity["full_experiment_hash"]

    def rows(self, stage):
        return self.baseline_consumer.rows(stage)

    def training_features(self, keys):
        return self.baseline_consumer.training_features(keys)

    def prepare(self, stage, *, limit=None):
        return self.baseline_consumer.prepare(stage, limit=limit)

    def _verify_final_checkpoint(self, path, payload, stage):
        path = Path(path).resolve()
        if stage in (3, 4):
            return self.baseline_consumer._verify_final_checkpoint(path, payload, stage)
        if stage != 5 or not path.is_relative_to(self.folder.resolve()) or path.name != "final_epoch.pt":
            raise HistoricalStage5ContractError("Stage5 checkpoint outside recovered run namespace")
        execution = json.loads((path.parent / "execution.json").read_text(encoding="utf-8"))
        if execution.get("status") != "PASS" or execution.get("checkpoint_sha256") != file_sha256(path):
            raise HistoricalStage5ContractError("Stage5 checkpoint execution/hash mismatch")


def _encode_states(model: V43Policy, batch, device: torch.device) -> torch.Tensor:
    arrays = [torch.from_numpy(v).to(device) for v in (batch.continuous, batch.missing, batch.categorical)]
    model.encoder.eval()
    with torch.no_grad():
        return torch.cat([model.encoder(*(value[start:start+128] for value in arrays))
                          for start in range(0, len(batch.keys), 128)])


def _output_diagnostics(result: dict[str, torch.Tensor]) -> dict[str, Any]:
    q = torch.minimum(result["q1"], result["q2"])
    logits = result["actor_logits"]
    probability = logits.softmax(1)
    actor = logits.argmax(1)
    critic = q.argmax(1)
    top = q.topk(2, dim=1).values
    distribution = torch.bincount(actor, minlength=len(ACTIONS)).float() / len(actor)
    positive = distribution[distribution > 0]
    output = {
        "q": float(q.gather(1, actor[:, None]).mean()),
        "entropy": float(-(positive * positive.log()).sum()),
        "actor_entropy": float(-(probability * logits.log_softmax(1)).sum(1).mean()),
        "mean_Q": float(q.mean()), "top1_top2_Q_margin": float((top[:, 0] - top[:, 1]).mean()),
        "actor_q_agreement": float((actor == critic).float().mean()), "actions": {},
    }
    for i, action in enumerate(ACTIONS):
        output["actions"][action] = {
            "mean_Q": float(q[:, i].mean()), "Q_argmax_count": int((critic == i).sum()),
            "Q_argmax_share": float((critic == i).float().mean()),
            "actor_probability_mean": float(probability[:, i].mean()),
            "actor_argmax_count": int((actor == i).sum()), "actor_argmax_share": float((actor == i).float().mean()),
        }
    return output


def _train_phase(consumer: HistoricalSearchConsumer, output_folder: Path, upstream: Path,
                 config: HistoricalStage5Config, search_root: Path, *, actor_only: bool = False,
                 actor_epochs: int = 0, payload_upstream: Path | None = None) -> Path:
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    model = consumer.load_policy(upstream, expected_stage=5 if actor_only else 4)
    grid = consumer.folder / "stage2/training_grid.parquet"
    metadata = consumer.folder / "stage2/training_grid_metadata.json"
    batch, state_index, action_ids, rewards, fit, lineage = load_counterfactual(consumer, grid, metadata)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("actor.") if actor_only else not name.startswith("encoder."))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    states = _encode_states(model, batch, device)
    action_ids = torch.from_numpy(action_ids).to(device)
    rewards = torch.from_numpy(rewards).to(device)
    indices = np.flatnonzero(fit)
    if not len(indices):
        raise HistoricalStage5ContractError("no eligible Stage5 training rows")
    before = parameter_digest(model, trainable_only=True)
    encoder_before = parameter_digest(model.encoder)
    frozen_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                     if actor_only and not name.startswith("actor.")}
    warmup = 0 if actor_only else config.warmup_epochs
    epochs = actor_epochs if actor_only else config.joint_epochs + warmup
    ratio = 1 if actor_only else config.critic_update_ratio
    output_folder.mkdir(parents=True, exist_ok=False)
    write_json(output_folder / "execution.json", {
        "status": "RUNNING", "stage": 5, "pid": os.getpid(), "started_utc": utcnow(),
        "config_hash": consumer.config["config_hash"], "input_rows": len(indices), "fixed_epochs": epochs,
        "seed": config.seed, "evaluation_rows_used": 0, "joint_epochs": 0 if actor_only else config.joint_epochs,
        "warmup_epochs": warmup, "actor_only_epochs": actor_epochs, "critic_update_ratio": ratio,
    }, exclusive=True)
    optimizer, group_evidence = make_grouped_adamw(model, config, actor_only=actor_only)
    write_json(output_folder / "optimizer_groups.json", group_evidence, exclusive=True)
    steps, log = 0, []
    phase_name = "actor_only_extraction" if actor_only else "stage5_training"
    with Heartbeat(consumer.folder, search_root, phase_name, epochs) as heartbeat:
        for epoch in range(1, epochs + 1):
            started = time.monotonic(); model.train(); model.encoder.eval(); np.random.shuffle(indices)
            sums: dict[str, torch.Tensor] = {}; count = 0; loss_values = []
            is_warmup = not actor_only and epoch <= warmup
            for start in range(0, len(indices), config.batch_size):
                ix = indices[start:start+config.batch_size]
                schedule = update_schedule(ratio, warmup=is_warmup, actor_only=actor_only)
                for kind in schedule:
                    result = model.forward_from_state(states[state_index[ix]])
                    losses, metrics = candidate_iql_loss_parts(result, action_ids[ix], rewards[ix], config)
                    if kind == "critic":
                        loss = losses["critic"]
                    elif kind == "actor":
                        loss = losses["actor"]
                    elif kind == "critic+value":
                        loss = losses["critic"] + losses["value"]
                    else:
                        loss = sum(losses.values())
                    if not torch.isfinite(loss):
                        raise HistoricalStage5ContractError("non-finite Stage5 loss")
                    optimizer.zero_grad(); loss.backward(); finite_gradients(model)
                    optimizer.step(); steps += 1
                    loss_values.append(float(loss.detach().cpu()))
                for key, value in metrics.items():
                    sums[key] = sums.get(key, torch.zeros_like(value)) + value.detach() * len(ix)
                count += len(ix)
                heartbeat.update(epoch=epoch, optimizer_steps=steps)
            model.eval(); parts = []
            with torch.no_grad():
                for start in range(0, len(batch.keys), config.batch_size):
                    parts.append(model.forward_from_state(states[start:start+config.batch_size]))
            joined = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
            diagnostics = _output_diagnostics(joined)
            entry = {
                "epoch": epoch, "training_phase": "actor_only" if actor_only else "warmup" if is_warmup else "joint",
                "loss": float(np.mean(loss_values)), "optimizer_steps": steps, "reward_rows": count,
                **{key: float(value / count) for key, value in sums.items()}, **diagnostics,
                "last_epoch_seconds": time.monotonic() - started,
            }
            log.append(entry); write_json(output_folder / "training_log.json", log)
            heartbeat.update(epoch=epoch, optimizer_steps=steps, last_epoch_seconds=entry["last_epoch_seconds"],
                             q_loss=entry["q1_loss"] + entry["q2_loss"], value_loss=entry["value_loss"],
                             actor_loss=entry["actor_loss"], actor_entropy=entry["actor_entropy"],
                             actor_q_agreement=entry["actor_q_agreement"],
                             estimated_remaining_seconds=entry["last_epoch_seconds"] * (epochs - epoch),
                             gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                             gpu_memory_allocated=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
    if encoder_before != parameter_digest(model.encoder):
        raise HistoricalStage5ContractError("frozen Stage5 encoder changed")
    for name, tensor in frozen_before.items():
        if not torch.equal(tensor, model.state_dict()[name].cpu()):
            raise HistoricalStage5ContractError("actor-only phase changed frozen Q/value/encoder")
    expected_steps = math.ceil(len(indices) / config.batch_size) * epochs * ratio
    if steps != expected_steps:
        raise HistoricalStage5ContractError(f"optimizer step mismatch: {steps} != {expected_steps}")
    audit = update_evidence(model, before, steps)
    audit.update(encoder_unchanged=True, actor_only_frozen_parameters_unchanged=True if actor_only else None)
    write_json(output_folder / "optimizer_audit.json", audit)
    model.cpu()
    payload = policy_payload(model, stage=5, upstream_path=payload_upstream or upstream, trained=True)
    payload.update(training_config=consumer.config["stages"]["stage5"],
                   historical_stage5_config=config.stage5_dict(), training_lineage=lineage,
                   final_epoch=epochs, optimizer_steps=steps, training_log=log,
                   config_hash=consumer.config["config_hash"], actor_only_epochs=actor_epochs,
                   actor_only_parent_checkpoint=str(upstream) if actor_only else None,
                   actor_only_parent_sha256=file_sha256(upstream) if actor_only else None)
    checkpoint = output_folder / "final_epoch.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    torch.save(payload, checkpoint)
    execution = json.loads((output_folder / "execution.json").read_text(encoding="utf-8"))
    execution.update(status="PASS", finished_utc=utcnow(), final_epoch=epochs, optimizer_steps=steps,
                     expected_optimizer_steps=expected_steps, checkpoint_sha256=file_sha256(checkpoint))
    write_json(output_folder / "execution.json", execution)
    return checkpoint


def _classify(surrogate: float, critic: float, actor: float, fixed: float, c2: float) -> str:
    if actor > max(fixed, c2) + TOL: return "SUCCESS"
    if critic > fixed + TOL: return "ACTOR_LIMITED"
    if surrogate > fixed + TOL: return "CRITIC_LIMITED"
    return "REWARD_LIMITED"


def _evaluate(consumer: HistoricalSearchConsumer, config: HistoricalStage5Config,
              baseline: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    folder = consumer.folder / "stage6"; folder.mkdir(exist_ok=False)
    checkpoint = consumer.folder / "stage5/final_epoch.pt"
    model = consumer.load_policy(checkpoint, expected_stage=5).eval()
    batch, rows = consumer.prepare(6); parts = []
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            parts.append(model(*(torch.from_numpy(value[start:start+128]) for value in
                                 (batch.continuous, batch.missing, batch.categorical))))
    result = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise HistoricalStage5ContractError("non-finite Stage6 model output")
    logits = result["actor_logits"]; probability = logits.softmax(1)
    q = torch.minimum(result["q1"], result["q2"]); actor = logits.argmax(1); critic = q.argmax(1)
    decisions = V43ActionCodec(consumer.contract).decision_frame(rows[list(KEYS)], actor.numpy())
    decisions["encoder_schema_hash"] = consumer.contract.schema_hash
    decisions["config_hash"] = consumer.config["config_hash"]
    for i, action in enumerate(ACTIONS): decisions[f"logit__{action}"] = logits[:, i].numpy()
    validate_actor(decisions)
    decisions.to_parquet(folder / "actor_decisions.parquet", index=False)
    decision_hash = file_sha256(folder / "actor_decisions.parquet")
    write_json(folder / "actor_provenance.json", {
        "actor_logit_argmax_only": True, "checkpoint_sha256": file_sha256(checkpoint),
        "decisions_sha256": decision_hash, "oracle_scores_read_after_decisions_saved": True,
        "policy_selection": "final Stage5 actor logits argmax; diagnostics never rerank",
    })
    critic_decisions = rows[list(KEYS)].copy(); critic_decisions["candidate_id"] = np.asarray(ACTIONS)[critic.numpy()]
    critic_decisions.to_parquet(folder / "critic_diagnostic_decisions.parquet", index=False)
    components = pd.read_parquet(consumer.root / CONTRACT_BUNDLE / "evaluation_components_MFLP.parquet")
    components["weighted_reward"] = sum(config.reward_weights()[axis] * components[f"reward_{axis}_norm"] for axis in config.reward_weights())
    wide = components.pivot(index=["firm_id", "base_year"], columns="candidate_id", values="weighted_reward").reindex(columns=ACTIONS)
    surrogate = wide.idxmax(axis=1).rename("candidate_id").reset_index().rename(columns={"base_year": "fiscal_year"})
    surrogate.to_parquet(folder / "reward_diagnostic_decisions.parquet", index=False)
    scores = pd.read_parquet(consumer.root / baseline["artifacts"]["oracle_surface"])
    c2 = pd.read_parquet(consumer.root / baseline["artifacts"]["c2_decisions"])
    reports = summarize_policies(scores, decisions, c2)
    joined = {"reward_surrogate": join_choices(scores, surrogate), "critic": join_choices(scores, critic_decisions), "C3": reports["C3"]}
    oracle_metrics = {}
    for oracle in ("alpha", "beta", "gamma"):
        comparison = reports["comparison"].query("oracle == @oracle").set_index("policy")
        fixed_means = scores.groupby("candidate_id")[oracle].mean().reindex(ACTIONS)
        fixed = str(fixed_means.idxmax()); actor_score = float(comparison.loc["C3", "mean_score"])
        c2_score = float(comparison.loc["C2", "mean_score"]); fixed_score = float(fixed_means[fixed])
        metric = {
            "C3": actor_score, "C2": c2_score, "OE": float(comparison.loc["OE", "mean_score"]),
            "best_fixed_action": fixed, "best_fixed": fixed_score,
            "ceiling": float(comparison.loc["ceiling", "mean_score"]),
            "C3_minus_C2": actor_score - c2_score, "C3_minus_best_fixed": actor_score - fixed_score,
            "ceiling_minus_C3": float(comparison.loc["ceiling", "mean_score"]) - actor_score,
        }
        for label, frame in joined.items(): metric[f"{label}_score"] = float(frame[oracle].mean())
        oracle_metrics[oracle] = metric
    choices = reports["C3"].set_index("firm_id")
    oe_scores = scores.query("candidate_id == 'OE'").set_index("firm_id").alpha
    delta = choices.alpha - oe_scores.reindex(choices.index); switched = choices.candidate_id.ne("OE")
    switching = {
        "beneficial_switch_count": int((switched & (delta > TOL)).sum()),
        "harmful_switch_count": int((switched & (delta < -TOL)).sum()),
        "tie_switch_count": int((switched & (delta.abs() <= TOL)).sum()),
        "total_beneficial_gain": float(delta[delta > TOL].sum()),
        "total_harmful_loss": float(-delta[delta < -TOL].sum()), "net_switch_gain": float(delta.sum()),
    }
    distribution = pd.Series(decisions.candidate_id).value_counts().reindex(ACTIONS, fill_value=0)
    shares = distribution / len(decisions); positive = shares[shares > 0]
    deterministic_h = float(-(positive * np.log(positive)).sum())
    alpha = oracle_metrics["alpha"]
    diagnostic = {
        "bottleneck_classification": _classify(alpha["reward_surrogate_score"], alpha["critic_score"], alpha["C3"], alpha["best_fixed"], alpha["C2"]),
        "oracles": oracle_metrics, "evaluation": _output_diagnostics(result), "OE_switching": switching,
        "deterministic_action_entropy": deterministic_h, "effective_actions": math.exp(deterministic_h),
        "action_distribution": {a: {"count": int(distribution[a]), "share": float(shares[a])} for a in ACTIONS},
        "actor_decisions_equal_final_logits_argmax": bool(np.array_equal(decisions.candidate_id.to_numpy(), np.asarray(ACTIONS)[logits.argmax(1).numpy()])),
        "Q_rerank": False, "Oracle_rerank": False, "action_forcing": False, **switching, **identity,
    }
    write_json(folder / "diagnostics.json", diagnostic)
    for key in ("comparison", "contrasts", "firm_contrasts", "ceiling"):
        reports[key].to_csv(folder / f"{key}.csv", index=False)
    detail = decisions.copy()
    for i, action in enumerate(ACTIONS):
        detail[f"Q__{action}"] = q[:, i].numpy(); detail[f"probability__{action}"] = probability[:, i].numpy()
    detail.to_parquet(folder / "actor_critic_outputs.parquet", index=False)
    if file_sha256(folder / "actor_decisions.parquet") != decision_hash:
        raise HistoricalStage5ContractError("actor decisions changed after Oracle scoring")
    return diagnostic


def prepare_or_run(root: Path, config: HistoricalStage5Config, *, train: bool = False,
                   campaign_name: str | None = None) -> dict[str, Any]:
    root = Path(root).resolve(); config.validate()
    baseline = load_baseline(root, training=train)
    bundle = validate_contract_bundle(root)
    search_root = root / V43_STAGE5_SEARCH_PATH
    if campaign_name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", campaign_name):
        raise HistoricalStage5ContractError("invalid campaign_name")
    heartbeat_root = search_root / campaign_name if campaign_name else search_root
    destination = heartbeat_root / "runs" / config.run_id if campaign_name else search_root / config.run_id
    if destination.exists():
        raise FileExistsError(f"run namespace exists: {destination}")
    source = pd.read_parquet(root / CONTRACT_BUNDLE / "reward_components_MFLP.parquet")
    reward, statistics = recompose_historical_reward(source, config)
    canonical_component_path = root / baseline["artifacts"]["reward_components"]
    canonical_source = pd.read_parquet(canonical_component_path)
    shared_non_profitability = [name for name in canonical_source.columns if "profitability" not in name]
    differing = [name for name in shared_non_profitability if name not in source or not source[name].equals(canonical_source[name])]
    if len(source) != len(canonical_source) or differing:
        raise HistoricalStage5ContractError(f"historical MFLP/current Stage2 equivalence failed: {differing}")
    identity = experiment_identity(root, baseline, config, statistics)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "run_config.json", {**config.serialize(), **identity,
                                                  "mode": "training" if train else "schema_dry_run"}, exclusive=True)
    saved_config = HistoricalStage5Config.from_mapping(json.loads((destination / "run_config.json").read_text(encoding="utf-8")))
    if saved_config.run_id != config.run_id or saved_config.normalized_scientific() != config.normalized_scientific():
        raise HistoricalStage5ContractError("written run_config is not losslessly reloadable")
    stage2 = destination / "stage2"; stage2.mkdir()
    grid_path = stage2 / "training_grid.parquet"; reward.to_parquet(grid_path, index=False)
    historical_metadata_path = root / CONTRACT_BUNDLE / "reward_components_metadata.json"
    canonical_metadata_path = canonical_component_path.parent / "reward_components_metadata.json"
    historical_metadata = json.loads(historical_metadata_path.read_text(encoding="utf-8-sig"))
    metadata = json.loads(canonical_metadata_path.read_text(encoding="utf-8-sig"))
    for name in ("profitability_contract_hash", "profitability_scale_sha256", "profitability_fit_rows", "profitability_evaluation_fit_rows"):
        metadata[name] = historical_metadata[name]
    metadata.update(
        dataset_sha256=file_sha256(grid_path), reward_statistics=statistics,
        config_hash=identity["full_experiment_hash"], run_id=config.run_id,
        source_component_sha256=bundle["files"]["reward_components_MFLP.parquet"],
        artifact_kind="weight_independent_MFLP_components", axes=list(config.reward_weights()),
        role="historical MFLP reward composition on current canonical Stage2/3/4 lineage",
        canonical_mflp_rebase={
            "status": "PASS",
            "historical_metadata_sha256": file_sha256(historical_metadata_path),
            "canonical_metadata_sha256": file_sha256(canonical_metadata_path),
            "historical_MFLP_component_sha256": bundle["files"]["reward_components_MFLP.parquet"],
            "canonical_MFL_component_sha256": file_sha256(canonical_component_path),
            "shared_non_profitability_column_count": len(shared_non_profitability),
            "shared_non_profitability_columns_exact": True,
            "profitability_columns_source": "frozen historical P95 MFLP contract",
            "encoder_schema_hash_before_path_rebase": historical_metadata["encoder_schema_hash"],
            "encoder_schema_hash_after_path_rebase": metadata["encoder_schema_hash"],
            "source_paths": "current canonical Stage2 paths; scientific row values unchanged",
        },
    )
    write_json(stage2 / "training_grid_metadata.json", metadata)
    write_json(destination / "reward_statistics.json", statistics)
    write_json(destination / "reward_composition.json", {"rating_lambda": 1.0, "lambdas": config.reward_weights(),
                                                          "auxiliary_total": config.auxiliary_total,
                                                          "source_component_sha256": bundle["files"]["reward_components_MFLP.parquet"]})
    consumer = HistoricalSearchConsumer(root, destination, config, baseline, identity)
    *_, lineage = load_counterfactual(consumer, grid_path, stage2 / "training_grid_metadata.json")
    preflight = {"status": "PASS", "mode": "training_preflight" if train else "schema_dry_run",
                 "optimizer_steps": 0, "simulator_calls": 0, "evaluation_statistics_fit_rows": 0,
                 "lineage": lineage, "contract_bundle": bundle, **identity}
    write_json(destination / "preflight.json", preflight)
    write_json(destination / "resolved_config.json", consumer.config)
    if not train:
        write_json(destination / "dry_run.json", preflight)
        return preflight
    upstream = root / baseline["artifacts"]["stage4_checkpoint"]
    if config.actor_only_epochs:
        joint = _train_phase(consumer, destination / "joint_stage5", upstream, config, heartbeat_root)
        _train_phase(consumer, destination / "stage5", joint, config, heartbeat_root,
                     actor_only=True, actor_epochs=config.actor_only_epochs, payload_upstream=upstream)
    else:
        _train_phase(consumer, destination / "stage5", upstream, config, heartbeat_root)
    with Heartbeat(destination, heartbeat_root, "stage6_evaluation"):
        diagnostics = _evaluate(consumer, config, baseline, identity)
    fresh_baseline = load_baseline(root, training=True)
    if fresh_baseline["upstream_baseline_hash"] != identity["provenance"]["upstream_baseline_hash"]:
        raise HistoricalStage5ContractError("frozen upstream changed during Stage5 run")
    execution = json.loads((destination / "stage5/execution.json").read_text(encoding="utf-8"))
    validation = {
        "status": "PASS", "final_epoch": execution["final_epoch"], "optimizer_steps": execution["optimizer_steps"],
        "evaluation_optimizer_rows": 0, "evaluation_statistics_fit_rows": 0, "encoder_frozen": True,
        "actor_decisions_equal_final_logits_argmax": diagnostics["actor_decisions_equal_final_logits_argmax"],
        "Q_rerank": False, "Oracle_rerank": False, "action_forcing": False,
        "checkpoint_sha256": execution["checkpoint_sha256"], **identity,
    }
    write_json(destination / "validation.json", validation)
    result = {"status": "PASS", "run_id": config.run_id, "config": config.serialize(),
              "diagnostics": diagnostics, "validation": validation, "completed_utc": utcnow()}
    write_json(destination / "run_result.json", result)
    print("V43_HISTORICAL_STAGE5_RUN_PASS", config.run_id, diagnostics["oracles"]["alpha"]["C3"], flush=True)
    return result
