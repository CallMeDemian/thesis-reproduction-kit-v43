"""Production Stage3--6 state/encoder seam for the frozen V43 contract."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet  # Initialize Arrow before Torch on Windows.
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from credit_recourse.rl.contracts.v43_encoder import ACTION_IDS, KEYS, V43EncoderContract, content_hash
from credit_recourse.rl.v43_features import V43FeatureProducer, canonical_keys, key_hash
from credit_recourse.rl.v43_model import V43Encoder, V43Policy, encoder_payload, load_encoder_payload

from credit_recourse.rl.v43_one_pass_contract import RUN_PATH, load_run_config, assert_training_rows, write_json
PREPARATION = RUN_PATH
STAGE_DATA = {3: "stage3_input", 4: "stage4_input", 5: "stage5_observed"}

def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

class V43StageConsumer:
    """Every production stage calls this same producer, stats and loader."""
    def __init__(self, project_root):
        self.root = Path(project_root).resolve()
        self.contract = V43EncoderContract.from_project_root(self.root)
        configured_run = os.environ.get("CREDIT_REPRO_RUN_PATH") or os.environ.get("THESIS_REPRO_RUN_ROOT")
        self.run_root = Path(configured_run).resolve() if configured_run else self.root / PREPARATION
        self.folder = self.run_root if self.run_root.name == "03_stage2" else self.run_root / PREPARATION
        frozen = _read_json(self.folder / "01_contract/encoder_contract.json")
        if frozen["schema_hash"] != self.contract.schema_hash:
            raise ValueError("V43 frozen contract identity changed")
        self.config = load_run_config(self.root, output_root=self.folder)
        self.statistics = _read_json(self.folder / "01_contract/training_preprocessing.json")
        fit = pd.read_parquet(self.folder / "02_data/statistics_fit_rows.parquet")
        assert_training_rows(fit)
        if self.statistics["fit_key_hash"] != key_hash(fit):
            raise ValueError("Checkpoint preprocessing population differs from eligible RL training rows")
        targets = pd.read_parquet(self.folder / "02_data/target_only_financial_accounts.parquet")
        from credit_recourse.rl.v43_rate_extension import read_extended_rates
        rates,_ = read_extended_rates(self.root)
        self.producer = V43FeatureProducer.from_project_root(self.root, source_year_max=2023, target_accounts=targets, rate_rows_override=rates)
        self._cached_features = None

    def training_features(self, keys):
        from credit_recourse.rl.contracts.v43_encoder import file_sha256
        if self._cached_features is None:
            manifest = _read_json(self.folder / "02_data/feature_manifest.json")
            path = self.folder / "02_data/training_and_target_features.parquet"
            if file_sha256(path) != manifest["raw_sha256"] or self.statistics["statistics_hash"] != manifest["statistics_hash"]:
                raise ValueError("Shared training feature artifact changed")
            for name, digest in manifest["source_hashes"].items():
                if file_sha256(self.root / name) != digest:
                    raise ValueError("Training feature source changed: "+name)
            self._cached_features = pd.read_parquet(path).set_index(list(KEYS), drop=False)
        index = pd.MultiIndex.from_frame(canonical_keys(keys))
        frame = self._cached_features.loc[index].reset_index(drop=True)
        return self.producer.transform_frame(frame, self.statistics)

    def rows(self, stage):
        if stage == 6:
            path = self.folder / "02_data/stage6_rows.parquet"
            rows = pd.read_parquet(path, columns=list(KEYS))
            if rows.empty:
                raise ValueError("Fresh Stage6 cohort is empty")
            return canonical_keys(rows)
        if stage not in STAGE_DATA:
            raise ValueError("Expected Stage3,4,5 or 6")
        rows = pd.read_parquet(self.folder / "02_data" / f"stage{stage}_rows.parquet")
        canonical_keys(rows)
        if set(rows.action_contract_sha256) != {self.contract.action_contract_sha256}:
            raise ValueError("Transition supervision uses another action contract")
        if not set(rows.candidate_id.dropna()).issubset(ACTION_IDS):
            raise ValueError("Legacy label in V43 transition supervision")
        from credit_recourse.rl.v43_support import read_action_support
        from credit_recourse.rl.v43_one_pass_contract import rl_fit_allowed
        support,_=read_action_support(self.root)
        evidence=support.set_index(list(KEYS)).action_support_valid.loc[pd.MultiIndex.from_frame(rows[list(KEYS)])].to_numpy()
        if not np.array_equal(rows.candidate_action_support_valid,evidence) or not np.array_equal(rows.action_support_valid,evidence & rows.action_class_id.notna()):
            raise ValueError('Training row support differs from the frozen common nine-action evidence')
        expected=rl_fit_allowed(rows,action_support=rows.action_support_valid.to_numpy(),reward_support=rows.reward_support_valid.to_numpy(),outcome_available=rows.outcome_available.to_numpy())
        if not np.array_equal(expected,rows.rl_fit_allowed):raise ValueError('Training row admission differs from common rl_fit_allowed')
        return rows.loc[rows.rl_fit_allowed].reset_index(drop=True)

    def prepare(self, stage, *, limit=None):
        rows = self.rows(stage)
        if limit is not None:
            rows = rows.iloc[:limit].copy()
        # Direct production feature call. No stage-specific column assembly.
        if stage == 6:
            producer = V43FeatureProducer.from_project_root(self.root, source_year_max=2024)
            batch = producer.transform(rows[list(KEYS)], self.statistics)
        else:
            assert_training_rows(rows)
            batch = self.training_features(rows[list(KEYS)])
        return batch, rows

    def initialize_encoder(self):
        return V43Encoder(self.contract, self.statistics)

    def _verify_final_checkpoint(self, path, payload, stage):
        from credit_recourse.rl.contracts.v43_encoder import file_sha256
        expected = Path(path).resolve()
        if expected.name != "final_epoch.pt" or not expected.is_relative_to(self.run_root):
            raise ValueError('Only a fresh final_epoch.pt inside the active run namespace is accepted')
        run=_read_json(expected.parent/'execution.json')
        epochs=self.config['stages'][f'stage{stage}']['max_epochs']
        if (run['status']!='PASS' or run['final_epoch']!=epochs or payload.get('final_epoch')!=epochs
            or payload.get('config_hash')!=self.config['config_hash'] or run.get('config_hash')!=self.config['config_hash'] or run['checkpoint_sha256']!=file_sha256(expected)):
            raise ValueError('Checkpoint does not match the completed fixed run')

    def load_encoder(self, path, *, schema_dry_run=False):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not schema_dry_run:
            self._verify_final_checkpoint(path,payload,3)
        if payload["statistics"].get("statistics_hash") != self.statistics["statistics_hash"]:
            raise ValueError("Checkpoint uses different frozen training preprocessing")
        return load_encoder_payload(payload, self.contract, allow_untrained_schema_dry_run=schema_dry_run)

    def load_policy(self, path, *, expected_stage, schema_dry_run=False):
        from credit_recourse.rl.v43_checkpoints import load_policy_payload
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not schema_dry_run:
            self._verify_final_checkpoint(path,payload,expected_stage)
        return load_policy_payload(payload, self.contract, self.statistics, expected_stage=expected_stage,
                                   allow_untrained_schema_dry_run=schema_dry_run)

    def dry_run(self, stage, *, checkpoint=None, limit=16, fixture_output=None):
        from credit_recourse.rl.v43_checkpoints import policy_payload
        batch, rows = self.prepare(stage, limit=limit)
        if stage > 3 and checkpoint is None:
            checkpoint = self.folder / "04_validation" / ("untrained_stage"+str(stage-1)+"_schema_fixture.pt")
        if stage in (3, 4):
            encoder = self.load_encoder(checkpoint, schema_dry_run=True) if checkpoint else self.initialize_encoder()
            encoder.eval()
            policy = V43Policy(encoder).eval() if stage == 4 else None
        else:
            policy = self.load_policy(checkpoint, expected_stage=stage-1, schema_dry_run=True).eval()
            encoder = policy.encoder
        x, missing, categorical = (torch.from_numpy(a) for a in (batch.continuous, batch.missing, batch.categorical))
        with torch.no_grad():
            if stage == 3:
                observed = rows.action_class_id.notna().to_numpy()
                if not observed.any():
                    raise ValueError("Schema dry-run batch has no observed ACD action")
                keep = np.flatnonzero(observed)
                ids = torch.tensor(rows.iloc[keep].action_class_id.to_numpy(dtype=np.int64))
                result = encoder.forward_pretrain(x[keep], missing[keep], categorical[keep], ids)
            else:
                result = policy(x, missing, categorical)
        if any(not torch.isfinite(v).all() for v in result.values()):
            raise ValueError("Non-finite production schema forward")
        output = {"stage": stage, "input_shape": list(batch.continuous.shape),
                  "missing_dtype": str(batch.missing.dtype), "categorical_dtype": str(batch.categorical.dtype),
                  "outputs": {k: list(v.shape) for k, v in result.items()},
                  "schema_hash": batch.schema_hash, "statistics_hash": batch.statistics_hash,
                  "feature_hash": batch.feature_hash, "optimizer_steps": 0, "policy_evaluation_executed": False}
        if checkpoint:
            from credit_recourse.rl.contracts.v43_encoder import file_sha256
            output["input_checkpoint_sha256"] = file_sha256(checkpoint)
        if fixture_output is not None:
            if stage not in (4, 5):
                raise ValueError("Only Stage4/5 produce intermediate policy fixtures")
            destination = Path(fixture_output).resolve()
            permitted = (self.folder / "04_validation").resolve()
            if destination.parent != permitted or not destination.name.startswith("untrained_"):
                raise ValueError("Schema fixtures must use an explicit untrained name in the validation folder")
            torch.save(policy_payload(policy, stage=stage, upstream_path=checkpoint, trained=False), destination)
            output["output_fixture_sha256"] = file_sha256(destination)
        return output

    def select_policy(self, checkpoint):
        """Normal Stage6 selection requires a trained Stage5 policy."""
        policy = self.load_policy(checkpoint, expected_stage=5, schema_dry_run=False).eval()
        batch, rows = self.prepare(6)
        selected, all_logits = [], []
        with torch.no_grad():
            for start in range(0, len(rows), 128):
                x = torch.from_numpy(batch.continuous[start:start+128])
                missing = torch.from_numpy(batch.missing[start:start+128])
                categorical = torch.from_numpy(batch.categorical[start:start+128])
                logits = policy(x, missing, categorical)["actor_logits"]
                if logits.shape[1] != 9 or not torch.isfinite(logits).all():
                    raise ValueError("Invalid V43 policy logits")
                selected.extend(logits.argmax(dim=1).tolist())
                all_logits.append(logits.cpu().numpy())
        from credit_recourse.rl.v43_actions import V43ActionCodec
        result = V43ActionCodec(self.contract).decision_frame(rows[list(KEYS)], np.asarray(selected, dtype=np.int64))
        result["encoder_schema_hash"] = self.contract.schema_hash
        scores = np.concatenate(all_logits)
        for j,name in enumerate(ACTION_IDS): result["logit__"+name] = scores[:,j]
        result["config_hash"] = self.config["config_hash"]
        return result

    def stage3_training_data(self):
        rows = self.rows(3)
        current_keys = canonical_keys(rows)
        next_keys = current_keys.copy()
        next_keys["fiscal_year"] += 1
        present = np.array([key in self.producer._positions for key in next_keys.itertuples(index=False, name=None)])
        # Current/next requests overlap heavily. Build each authoritative state
        # once through the same producer, then use explicit keyed target joins.
        requested = pd.concat([current_keys, next_keys.loc[present]], ignore_index=True).drop_duplicates(list(KEYS))
        all_batch = self.training_features(requested)
        index = pd.MultiIndex.from_frame(all_batch.keys)
        current_index = index.get_indexer(pd.MultiIndex.from_frame(current_keys))
        from credit_recourse.rl.v43_features import FeatureBatch
        batch = FeatureBatch(current_keys, all_batch.continuous[current_index],
                             all_batch.missing[current_index], all_batch.categorical[current_index],
                             all_batch.schema_hash, all_batch.statistics_hash)
        next_values = np.zeros_like(batch.continuous)
        next_missing = np.ones_like(batch.missing)
        if present.any():
            following_index = index.get_indexer(pd.MultiIndex.from_frame(next_keys.loc[present]))
            next_values[present] = all_batch.continuous[following_index]
            next_missing[present] = all_batch.missing[following_index]
        ids = rows.action_class_id.to_numpy(dtype=np.int64, na_value=-1)
        acd_eligible = present & (ids >= 0)
        # Preserve every original row; OOT/POST never receive an optimizer update.
        assert_training_rows(rows)
        fit = rows.rl_fit_allowed.to_numpy(dtype=bool)
        firms = pd.factorize(rows.firm_id.astype(str), sort=True)[0]
        arrays = (batch.continuous, batch.missing, batch.categorical, ids, next_values,
                  next_missing, acd_eligible, firms.astype(np.int64))
        return TensorDataset(*(torch.from_numpy(a[fit]) for a in arrays)), {
            "original_rows": len(rows), "training_rows": int(fit.sum()), "transform_only_rows": int((~fit).sum()),
            "ACD_missing_action_rows": int((ids < 0).sum()), "next_state_missing_rows": int((~present).sum()),
            "all_original_rows_preserved": True, "schema_hash": self.contract.schema_hash}

def _contrastive(z, firms, temperature=.1):
    z = nn.functional.normalize(z, dim=1)
    diagonal = torch.eye(len(z), device=z.device, dtype=torch.bool)
    same = (firms[:, None] == firms[None, :]) & ~diagonal
    if not same.any():
        return z.sum()*0
    similarity = (z @ z.T / temperature).masked_fill(diagonal, -1e9)
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    return -log_probability[same].mean()

def train_fresh_stage3(consumer, output, *, epochs=30, seed=2, batch_size=512,
                       learning_rate=3e-4, weight_decay=1e-5, masking_ratio=.15):
    """Explicit future training entry point. Preparation never calls this."""
    if epochs < 1 or batch_size < 1 or not 0 < masking_ratio < 1:
        raise ValueError("Invalid Stage3 training configuration")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing trained checkpoint")
    from credit_recourse.rl.v43_one_pass_contract import begin_training, finish_training
    fixed = consumer.config["stages"]["stage3"]
    actual = dict(max_epochs=epochs,seed=seed,batch_size=batch_size,learning_rate=learning_rate,weight_decay=weight_decay,masking_ratio=masking_ratio)
    if any(actual[k] != fixed[k] for k in actual):
        raise ValueError("Stage3 arguments differ from the physically frozen run configuration")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset, lineage = consumer.stage3_training_data()
    model = consumer.initialize_encoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    from credit_recourse.rl.v43_optimizer_audit import parameter_digest, finite_gradients, update_evidence
    before_parameters=parameter_digest(model,trainable_only=True)
    begin_training(consumer,3,destination,len(dataset))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    log, optimizer_steps = [], 0
    for _epoch in range(epochs):
        values=[]
        for raw in loader:
            x, missing, cats, action_ids, next_x, next_missing, acd_eligible, firms = [v.to(device) for v in raw]
            mask = (torch.rand_like(x) < masking_ratio) & ~missing
            state, tokens = model.encode(x, missing, cats, mask)
            reconstruction = model.reconstruction(tokens).squeeze(-1)
            mcm = nn.functional.mse_loss(reconstruction[mask], x[mask]) if mask.any() else (
                nn.functional.mse_loss(reconstruction[~missing], x[~missing]) if (~missing).any() else state.sum()*0)
            if acd_eligible.any():
                state_acd = state[acd_eligible]
                action = model.action_embedding(model.action_table[action_ids[acd_eligible]])
                prediction = model.forward_head(torch.cat([state_acd, action, state_acd*action], dim=-1))
                observed_targets = ~next_missing[acd_eligible]
                acd = nn.functional.mse_loss(prediction[observed_targets], next_x[acd_eligible][observed_targets]) if observed_targets.any() else state.sum()*0
            else:
                acd = state.sum()*0
            contrastive = _contrastive(model.contrastive(state), firms)
            loss = mcm + .5*acd + .3*contrastive
            if not torch.isfinite(loss):
                raise ValueError("Non-finite Stage3 loss")
            optimizer.zero_grad()
            loss.backward()
            finite_gradients(model)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
            values.append(float(loss.detach().cpu()))
        entry = {"epoch":_epoch+1,"loss":float(np.mean(values)),"optimizer_steps":optimizer_steps}
        log.append(entry)
        write_json(destination.parent/"training_log.json",log)
        print("ONE_PASS_EPOCH",3,json.dumps(entry),flush=True)
    audit=update_evidence(model,before_parameters,optimizer_steps)
    write_json(destination.parent/"optimizer_audit.json",audit)
    model.cpu()
    payload = encoder_payload(model, trained=True, producer_stage="Stage3")
    payload["training_lineage"] = lineage
    payload.update(config_hash=consumer.config["config_hash"],final_epoch=epochs,optimizer_steps=optimizer_steps,training_log=log)
    payload["training_config"] = {"epochs":epochs,"seed":seed,"batch_size":batch_size,
                                  "learning_rate":learning_rate,"weight_decay":weight_decay,"masking_ratio":masking_ratio,
                                  "loss_weights":{"MCM":1.0,"ACD":.5,"same_firm_contrastive":.3}}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing trained checkpoint")
    torch.save(payload, destination)
    finish_training(consumer,3,destination,optimizer_steps,len(dataset))
    return lineage

def stage_main(stage, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--select-policy", action="store_true")
    parser.add_argument("--encoder-checkpoint", "--policy-checkpoint")
    parser.add_argument("--schema-fixture-output")
    parser.add_argument("--transition-input")
    parser.add_argument("--transition-metadata")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    consumer = V43StageConsumer(args.project_root)
    if args.select_policy:
        if stage != 6 or args.train or args.dry_run or args.prepare_only or not args.encoder_checkpoint or not args.output:
            raise ValueError("Stage6 selection requires an explicit trained checkpoint and output")
        output = Path(args.output)
        if output.exists():
            raise FileExistsError("Refusing to overwrite an existing policy selection")
        result = consumer.select_policy(args.encoder_checkpoint)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output, index=False)
        return 0
    if args.train:
        if stage == 6 or args.dry_run or args.prepare_only or not args.output:
            raise ValueError("Training requires Stage3/4/5, an explicit output, and no preparation flags")
        if stage in (4, 5):
            if not args.encoder_checkpoint:
                raise ValueError("Policy training requires its trained predecessor checkpoint")
            from credit_recourse.rl.v43_training import train_policy
            train_policy(consumer, stage, args.encoder_checkpoint, args.output,
                         transition_input=args.transition_input, transition_metadata=args.transition_metadata)
            return 0
        config = consumer.config["stages"]["stage3"]
        train_fresh_stage3(consumer, args.output, epochs=config["max_epochs"], seed=config["seed"],
                          batch_size=config["batch_size"], learning_rate=config["learning_rate"],
                          weight_decay=config["weight_decay"], masking_ratio=config["masking_ratio"])
        return 0
    if args.prepare_only:
        if args.dry_run or args.schema_fixture_output:
            raise ValueError("Full preparation cannot combine with schema-fixture generation")
        if stage == 3:
            dataset, result = consumer.stage3_training_data()
            result["prepared_training_rows"] = len(dataset)
        elif stage == 4:
            from credit_recourse.rl.v43_training import prepare_bc
            _, _, _, result = prepare_bc(consumer)
        elif stage == 5:
            if not args.transition_input or not args.transition_metadata:
                raise ValueError("Full Stage5 preparation requires its V43 counterfactual grid and metadata")
            from credit_recourse.rl.v43_training import load_counterfactual
            *_, result = load_counterfactual(consumer, args.transition_input, args.transition_metadata)
        else:
            batch, rows = consumer.prepare(6)
            result = {"rows": len(rows), "feature_hash": batch.feature_hash}
        result.update(stage=stage, mode="full_input_preparation", optimizer_steps=0, training_executed=False)
        print(json.dumps(result), flush=True)
        return 0
    # A dry-run is a schema forward only: it never creates an optimizer.
    result = consumer.dry_run(stage, checkpoint=args.encoder_checkpoint, fixture_output=args.schema_fixture_output)
    print(json.dumps(result), flush=True)
    return 0

