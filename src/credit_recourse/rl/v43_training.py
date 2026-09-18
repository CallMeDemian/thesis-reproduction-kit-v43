"""Native V43 BC/IQL training. Never invoked by preparation/dry runs.

The IQL input is a newly materialized nine-candidate counterfactual grid from
the V43 production simulation bundle. The old eleven-candidate grid is not a
valid substitute. State features always come from the shared producer.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet  # Initialize Arrow before Torch on Windows.
import torch
from torch import nn
from credit_recourse.rl.contracts.v43_encoder import ACTION_IDS, KEYS, file_sha256
from credit_recourse.rl.v43_features import canonical_keys, key_hash
from credit_recourse.simulator.historical_source import firm_key
from credit_recourse.rl.v43_model import V43Policy
from credit_recourse.rl.v43_checkpoints import policy_payload
from credit_recourse.rl.v43_training_math import (
    _class_balance_audit, _family_balance_audit, _combine_stage4_loss_weights,
    _validate_projected_rating_supervision, expectile_loss,
)


def soft_targets(rows):
    """Retain soft nearest-archetype labels; missing support is never A0."""
    ids = {name: i for i, name in enumerate(ACTION_IDS)}
    values = np.zeros((len(rows), 9), dtype=np.float32)
    available = rows.action_class_id.notna().to_numpy(copy=True)
    for k in range(1, 4):
        for i, (name, probability) in enumerate(zip(rows[f"soft_cand_id_{k}"], rows[f"soft_cand_prob_{k}"])):
            if not available[i]:
                continue
            if pd.isna(name) or str(name) == "":
                if pd.notna(probability) and float(probability) != 0:
                    raise ValueError("Soft target probability without a candidate")
                continue
            if name not in ids or not np.isfinite(probability) or probability < 0:
                raise ValueError("Invalid V43 soft candidate target")
            values[i, ids[name]] += float(probability)
    mass = values.sum(axis=1)
    if (mass[available] <= 0).any() or (mass[~available] != 0).any():
        raise ValueError("Soft target support and action observation disagree")
    values[available] /= mass[available, None]
    return torch.from_numpy(values), available


def prepare_bc(consumer, *, limit=None):
    batch, rows = consumer.prepare(4, limit=limit)
    targets, available = soft_targets(rows)
    fit = rows.rl_fit_allowed.to_numpy(dtype=bool) & available
    from credit_recourse.rl.v43_one_pass_contract import assert_training_rows
    assert_training_rows(rows.loc[fit])
    return batch, targets, fit, {"original_rows": len(rows), "fit_rows": int(fit.sum()),
        "unavailable_action_rows": int((~available).sum()), "all_original_rows_preserved": True,
        "source_key_hash": key_hash(rows), "state_feature_hash": batch.feature_hash}


def load_counterfactual(consumer, path, metadata_path):
    """Validate complete production transitions before any IQL optimization."""
    path, metadata_path = Path(path), Path(metadata_path)
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {"encoder_schema_hash": consumer.contract.schema_hash,
        "action_contract_sha256": consumer.contract.action_contract_sha256,
        "simulation_bundle": "V43ProductionSimulationBundle", "transition_source": "counterfactual",
        "gamma": 0.0, "oracle_used_in_stage5_training": False,
        "non_rate_calibration_sha256": consumer.contract.non_rate_calibration_sha256,
        "dataset_sha256": file_sha256(path), "schema_fixture": False}
    for name, value in expected.items():
        if name not in meta or meta[name] != value:
            raise ValueError("Invalid V43 counterfactual provenance: " + name)
    sources = meta.get("source_hashes", {})
    if not sources:
        raise ValueError("Counterfactual inputs require verified source lineage")
    for relative, digest in sources.items():
        source = (consumer.root / relative).resolve()
        if not source.is_relative_to(consumer.root) or file_sha256(source) != digest:
            raise ValueError("Counterfactual source hash changed")
    rows = pd.read_parquet(path)
    if rows[list(KEYS)].isna().any().any():
        raise ValueError("Missing counterfactual firm/year key")
    rows["firm_id"] = rows.firm_id.map(firm_key)
    years = pd.to_numeric(rows.fiscal_year, errors="raise")
    if not np.isfinite(years).all() or (years != years.astype("int64")).any():
        raise ValueError("Invalid counterfactual year")
    rows["fiscal_year"] = years.astype("int64")
    required = {*KEYS, "candidate_id", "action_contract_sha256", "done", "reward_train"}
    if not required.issubset(rows):
        raise ValueError("Incomplete V43 counterfactual transition inputs")
    if set(rows.action_contract_sha256) != {consumer.contract.action_contract_sha256}:
        raise ValueError("Counterfactual action contract mismatch")
    if rows.duplicated([*KEYS, "candidate_id"]).any():
        raise ValueError("Duplicate firm/year/candidate counterfactual transition")
    groups = rows.groupby(list(KEYS), sort=False).candidate_id.agg(list)
    if not groups.map(lambda x: len(x) == 9 and set(x) == set(ACTION_IDS)).all():
        raise ValueError("IQL requires exactly the frozen nine actions for every factual key")
    keys = canonical_keys(rows[list(KEYS)].drop_duplicates())
    authoritative = consumer.rows(5)
    if key_hash(keys) != key_hash(authoritative) or len(keys) != len(authoritative):
        raise ValueError("Counterfactual factual cohort changed")
    if not rows.done.eq(1).all() or not np.isfinite(rows.reward_train).all():
        raise ValueError("One-step IQL requires finite rewards and terminal transitions")
    # This is preserved reward-composition/standardization validation, not a fit.
    reward_audit = _validate_projected_rating_supervision(rows, list(ACTION_IDS))
    batch = consumer.training_features(keys)
    index = pd.MultiIndex.from_frame(keys)
    state_index = index.get_indexer(pd.MultiIndex.from_frame(rows[list(KEYS)]))
    if (state_index < 0).any():
        raise ValueError("Missing counterfactual decision-state key")
    action_ids = rows.candidate_id.map({a: i for i, a in enumerate(ACTION_IDS)}).to_numpy(dtype=np.int64)
    fit_keys = authoritative.loc[authoritative.rl_fit_allowed, list(KEYS)]
    fit_index = pd.MultiIndex.from_frame(canonical_keys(fit_keys))
    fit = pd.MultiIndex.from_frame(rows[list(KEYS)]).isin(fit_index)
    from credit_recourse.rl.v43_one_pass_contract import assert_training_rows
    assert_training_rows(rows.loc[fit])
    return batch, state_index, action_ids, rows.reward_train.to_numpy(dtype=np.float32), fit, {
        "original_factual_rows": len(keys), "original_grid_rows": len(rows), "fit_rows": int(fit.sum()),
        "dataset_sha256": file_sha256(path), "metadata_sha256": file_sha256(metadata_path),
        "source_key_hash": key_hash(keys), "state_feature_hash": batch.feature_hash,
        "reward_audit": reward_audit, "target_formula": "target = immediate_reward_train",
        "next_state_used_in_q_target": False, "transition_source": "counterfactual"}


def bc_loss(logits, targets, weights):
    return -(weights[None] * targets * nn.functional.log_softmax(logits, dim=1)).sum(dim=1).mean()


def iql_losses(result, action_ids, rewards, config):
    """Existing terminal IQL, AWR and detached critic-to-actor KL objectives."""
    if config["gamma"] != 0 or config["cql_alpha"] != 0 or config["actor_distill_mode"] != "kl":
        raise ValueError("V43 requires the frozen terminal IQL profile")
    q1 = result["q1"].gather(1, action_ids[:, None]).squeeze(1)
    q2 = result["q2"].gather(1, action_ids[:, None]).squeeze(1)
    q = torch.minimum(q1, q2)
    value, logits = result["value"], result["actor_logits"]
    critic = nn.functional.smooth_l1_loss(q1, rewards) + nn.functional.smooth_l1_loss(q2, rewards)
    vloss = expectile_loss(q.detach() - value, config["expectile_tau"])
    weights = torch.exp(config["beta"] * (q.detach() - value.detach())).clamp(max=config["awr_weight_cap"])
    awr = (weights * nn.functional.cross_entropy(logits, action_ids, reduction="none")).mean()
    q_all = torch.minimum(result["q1"], result["q2"]).detach()
    top = q_all.topk(2, dim=1).values
    eligible = top[:, 0] - top[:, 1] >= config["actor_distill_margin_min"]
    kl = logits.sum() * 0
    if eligible.any():
        target = nn.functional.softmax(q_all[eligible] / config["actor_distill_temperature"], dim=1)
        kl = nn.functional.kl_div(nn.functional.log_softmax(logits[eligible], dim=1), target, reduction="batchmean")
    return {"critic": critic, "value": vloss, "actor": awr + config["actor_distill_lambda"] * kl}


def _tensors(batch, device):
    return tuple(torch.from_numpy(v).to(device) for v in (batch.continuous, batch.missing, batch.categorical))


def policy_diagnostics(model, arrays, indices, batch_size):
    """Training-time diagnostic only; never called during preparation."""
    model.eval()
    chosen, selected_q = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start+batch_size]
            result = model.forward_from_state(arrays[idx])
            action = result["actor_logits"].argmax(dim=1)
            q = torch.minimum(result["q1"], result["q2"])
            chosen.extend(action.cpu().tolist())
            selected_q.extend(q.gather(1, action[:, None]).squeeze(1).cpu().tolist())
    probabilities = np.bincount(chosen, minlength=9) / len(chosen)
    positive = probabilities[probabilities > 0]
    return {"q": float(np.mean(selected_q)), "entropy": float(-(positive * np.log(positive)).sum())}


def train_policy(consumer, stage, checkpoint, output, *, transition_input=None, transition_metadata=None):
    """Single full-population fixed-epoch training using the canonical fixed profile."""
    from credit_recourse.rl.v43_one_pass_contract import begin_training, finish_training, write_json
    if stage not in (4, 5):
        raise ValueError("Expected Stage4 or Stage5")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing checkpoint")
    config = consumer.config["stages"][f"stage{stage}"]
    profile_meta = {"config_hash":consumer.config["config_hash"]}
    if config["encoder_finetune"]:
        raise ValueError("The frozen V43 training profile requires a frozen encoder")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if stage == 4:
        model = V43Policy(consumer.load_encoder(checkpoint))
        batch, targets, fit, lineage = prepare_bc(consumer)
        state_index = np.arange(len(batch.keys))
        family = consumer.contract.action_contract["family_by_candidate"]
        train_targets = targets[fit]
        cw, _, _ = _class_balance_audit(train_targets, list(ACTION_IDS), family,
            enabled=config["class_balanced"], beta=config["class_balance_beta"], weight_cap=config["class_weight_cap"])
        fw, _, _ = _family_balance_audit(train_targets, list(ACTION_IDS), family,
            enabled=config["family_balanced"], power=config["family_balance_power"], weight_cap=config["family_weight_cap"])
        loss_weights = _combine_stage4_loss_weights(cw, fw, train_targets, combined_weight_cap=config["combined_weight_cap"])
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("actor."))
    else:
        model = consumer.load_policy(checkpoint, expected_stage=4)
        if not transition_input or not transition_metadata:
            raise ValueError("Stage5 requires explicit V43 counterfactual input and metadata")
        batch, state_index, action_ids, rewards, fit, lineage = load_counterfactual(consumer, transition_input, transition_metadata)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not name.startswith("encoder."))
    train_index = np.flatnonzero(fit)
    if not len(train_index):
        raise ValueError("No eligible training rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    arrays = _tensors(batch, device)
    model.encoder.eval()
    with torch.no_grad():
        states = torch.cat([model.encoder(*(v[start:start+128] for v in arrays))
                            for start in range(0,len(batch.keys),128)])
    del arrays
    if stage == 4:
        targets, loss_weights = targets.to(device), loss_weights.to(device)
    else:
        action_ids, rewards = torch.from_numpy(action_ids).to(device), torch.from_numpy(rewards).to(device)
    from credit_recourse.rl.v43_optimizer_audit import parameter_digest, finite_gradients, update_evidence
    before_parameters=parameter_digest(model,trainable_only=True)
    encoder_before=parameter_digest(model.encoder)
    if hasattr(consumer,'begin_run_training'):
        consumer.begin_run_training(stage,destination,len(train_index))
    else:
        begin_training(consumer, stage, destination, len(train_index))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"], weight_decay=config["weight_decay"])
    log, optimizer_steps = [], 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        model.encoder.eval()
        np.random.shuffle(train_index)
        values = []
        for start in range(0, len(train_index), config["batch_size"]):
            idx = train_index[start:start+config["batch_size"]]
            result = model.forward_from_state(states[state_index[idx]])
            loss = bc_loss(result["actor_logits"], targets[idx], loss_weights) if stage == 4 else sum(iql_losses(result, action_ids[idx], rewards[idx], config).values())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite V43 policy loss")
            optimizer.zero_grad()
            loss.backward()
            finite_gradients(model)
            optimizer.step()
            optimizer_steps += 1
            values.append(float(loss.detach().cpu()))
        entry = {"epoch": epoch, "loss": float(np.mean(values)), "optimizer_steps": optimizer_steps}
        if stage == 5:
            entry.update(policy_diagnostics(model, states, state_index[train_index], config["batch_size"]))
        log.append(entry)
        write_json(destination.parent / "training_log.json", log)
        print("ONE_PASS_EPOCH", stage, json.dumps(entry), flush=True)
    audit=update_evidence(model,before_parameters,optimizer_steps)
    if encoder_before!=parameter_digest(model.encoder):raise ValueError("Frozen encoder changed")
    audit["encoder_unchanged"]=True
    write_json(destination.parent/"optimizer_audit.json",audit)
    if stage==4:
        from credit_recourse.rl.v43_bc_diagnostics import save_bc_diagnostics
        save_bc_diagnostics(consumer,model,states,targets,fit,cw,fw,loss_weights,destination.parent)
    model.cpu()
    payload = policy_payload(model, stage=stage, upstream_path=checkpoint, trained=True)
    payload.update(training_lineage=lineage, training_config=config, profile_metadata=profile_meta,
        optimizer_steps=optimizer_steps, training_log=log, final_epoch=config["max_epochs"],
        config_hash=consumer.config["config_hash"],
        resolved_architecture=consumer.contract.to_dict()["architecture"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing checkpoint")
    torch.save(payload, destination)
    finish_training(consumer, stage, destination, optimizer_steps, len(train_index))
    return lineage
