"""V43 encoder, nine-logit actor and explicit state-action critics.

Fresh weights only: legacy feature/action checkpoints are rejected. Loading an
untrained schema fixture requires an explicit dry-run flag.
"""
from __future__ import annotations
import numpy as np
import pyarrow.parquet  # Initialize Arrow before Torch on Windows.
import torch
from torch import nn
from credit_recourse.rl.contracts.v43_encoder import ACTION_IDS, CATEGORICAL_COLUMNS, SCHEMA_VERSION
from credit_recourse.rl.v43_actions import V43ActionCodec
from credit_recourse.rl.v43_features import validate_statistics
from credit_recourse.rl.contracts.v43_encoder import content_hash


class V43Encoder(nn.Module):
    def __init__(self, contract, statistics):
        super().__init__()
        validate_statistics(contract, statistics)
        self.contract = contract
        self.statistics = statistics
        self.n_features = len(contract.specs)
        width = 256
        self.value_projection = nn.Linear(1, width)
        self.feature_embedding = nn.Embedding(self.n_features, width)
        blocks = {name: i for i, name in enumerate(dict.fromkeys(s.block for s in contract.specs))}
        self.block_embedding = nn.Embedding(len(blocks), width)
        self.register_buffer("block_ids", torch.tensor([blocks[s.block] for s in contract.specs], dtype=torch.long))
        self.status_embedding = nn.Embedding(3, width)
        self.categories = nn.ModuleList([nn.Embedding(len(statistics["vocabulary"][name])+2, width) for name in CATEGORICAL_COLUMNS])
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=width, nhead=8, dim_feedforward=width*4, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True), num_layers=4, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.reconstruction = nn.Linear(width, 1)
        self.contrastive = nn.Linear(width, 64)
        self.action_embedding = nn.Sequential(nn.Linear(25, width), nn.GELU(), nn.Linear(width, width))
        self.forward_head = nn.Sequential(nn.Linear(width*3, width), nn.GELU(), nn.Linear(width, self.n_features))
        self.register_buffer("action_table", torch.from_numpy(V43ActionCodec(contract).representations))

    def tokens(self, x, missing, categorical, mcm_mask=None):
        if x.dtype != torch.float32 or missing.dtype != torch.bool or categorical.dtype != torch.int64:
            raise TypeError("V43 requires float32/bool/int64 inputs")
        if x.ndim != 2 or x.shape[1] != self.n_features or x.shape != missing.shape or categorical.shape != (len(x), 2):
            raise ValueError("V43 feature tensor shape mismatch")
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite V43 model input")
        masked = torch.zeros_like(missing) if mcm_mask is None else mcm_mask
        if masked.dtype != torch.bool or masked.shape != missing.shape:
            raise ValueError("MCM mask shape/dtype mismatch")
        # Hide the actual value, not merely its status embedding.
        hidden_value = x.masked_fill(missing | masked, 0.0)
        status = missing.long().masked_fill(masked & ~missing, 2)
        ids = torch.arange(self.n_features, device=x.device)
        numeric = (self.value_projection(hidden_value.unsqueeze(-1)) + self.feature_embedding(ids)[None]
                   + self.block_embedding(self.block_ids)[None] + self.status_embedding(status))
        cats = torch.stack([embedding(categorical[:, j]) for j, embedding in enumerate(self.categories)], dim=1)
        return torch.cat([self.cls.expand(len(x), -1, -1), numeric, cats], dim=1)

    def encode(self, x, missing, categorical, mcm_mask=None):
        tokens = self.tokens(x, missing, categorical, mcm_mask)
        if self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            encoded = tokens
            for layer in self.transformer.layers:
                encoded = checkpoint(layer, encoded, use_reentrant=False, preserve_rng_state=True)
            if self.transformer.norm is not None:
                encoded = self.transformer.norm(encoded)
        else:
            encoded = self.transformer(tokens)
        encoded = self.norm(encoded)
        return encoded[:, 0], encoded[:, 1:1+self.n_features]

    def forward(self, x, missing, categorical):
        return self.encode(x, missing, categorical)[0]

    def forward_pretrain(self, x, missing, categorical, action_ids, mcm_mask=None):
        if action_ids.dtype != torch.int64 or action_ids.shape != (len(x),) or (action_ids < 0).any() or (action_ids >= 9).any():
            raise ValueError("ACD requires observed V43 class IDs; unavailable actions must be excluded from ACD")
        state, tokens = self.encode(x, missing, categorical, mcm_mask)
        action = self.action_embedding(self.action_table[action_ids])
        return {"state": state, "mcm_reconstruction": self.reconstruction(tokens).squeeze(-1),
                "next_prediction": self.forward_head(torch.cat([state, action, state*action], dim=-1)),
                "projection": nn.functional.normalize(self.contrastive(state), dim=-1)}


class StateActionCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Sequential(nn.Linear(25, 256), nn.GELU(), nn.Linear(256, 256))
        self.output = nn.Sequential(nn.Linear(256*3, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, state, action_table):
        action = self.action(action_table).unsqueeze(0).expand(len(state), -1, -1)
        state = state.unsqueeze(1).expand(-1, len(action_table), -1)
        return self.output(torch.cat([state, action, state*action], dim=-1)).squeeze(-1)


class V43Policy(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.actor = nn.Linear(256, len(ACTION_IDS))
        self.q1, self.q2 = StateActionCritic(), StateActionCritic()
        self.value = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, x, missing, categorical):
        return self.forward_from_state(self.encoder(x, missing, categorical))

    def forward_from_state(self, state):
        return {"actor_logits": self.actor(state), "q1": self.q1(state, self.encoder.action_table),
                "q2": self.q2(state, self.encoder.action_table), "value": self.value(state).squeeze(-1)}


def encoder_payload(encoder, *, trained, producer_stage):
    return {"schema_version": SCHEMA_VERSION, "schema_hash": encoder.contract.schema_hash,
            "contract": encoder.contract.to_dict(), "statistics": encoder.statistics,
            "encoder_state_dict": encoder.state_dict(), "trained": bool(trained),
            "producer_stage": producer_stage, "action_ids": list(ACTION_IDS)}


def load_encoder_payload(payload, contract, *, allow_untrained_schema_dry_run=False):
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("schema_hash") != contract.schema_hash:
        raise ValueError("Legacy/incompatible encoder checkpoint; a fresh V43 Stage3 checkpoint is required")
    if content_hash(payload.get("contract", {})) != contract.schema_hash:
        raise ValueError("Checkpoint embedded contract is inconsistent")
    if payload.get("action_ids") != list(ACTION_IDS):
        raise ValueError("Checkpoint action order differs from frozen V43")
    if not payload.get("trained") and not allow_untrained_schema_dry_run:
        raise ValueError("Untrained fixture cannot be used for training transfer or policy evaluation")
    model = V43Encoder(contract, payload["statistics"])
    expected_actions = model.action_table.clone()
    expected_blocks = model.block_ids.clone()
    model.load_state_dict(payload["encoder_state_dict"], strict=True)
    if not torch.equal(model.action_table, expected_actions) or not torch.equal(model.block_ids, expected_blocks):
        raise ValueError("Checkpoint semantic action/block buffers differ from contract")
    return model
