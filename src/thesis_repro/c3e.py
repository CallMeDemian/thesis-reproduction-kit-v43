from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def load_fresh_rl_contract(root: Path) -> dict:
    path = Path(root) / "contracts/scientific/fresh_rl_execution_contract.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("expected_actor_count") != len(payload["selected_configurations"]) * len(payload["seeds"]):
        raise ValueError("fresh RL contract actor cardinality is inconsistent")
    if payload["aggregation"]["outer"] != "equal_arithmetic_mean_of_four_configuration_consensuses":
        raise ValueError("unsupported C3-E outer aggregation contract")
    return payload


def aggregate_hierarchical(
    actor_probabilities: Mapping[str, Mapping[int, np.ndarray]],
    contract: Mapping[str, object],
) -> tuple[np.ndarray, dict]:
    """Rebuild C3-E as seed-within-config, then config-within-ensemble."""
    configurations = list(contract["selected_configurations"])
    seeds = [int(value) for value in contract["seeds"]]
    expected_shape: tuple[int, int] | None = None
    config_consensus: dict[str, np.ndarray] = {}
    actor_hashes: dict[str, dict[str, str]] = {}
    for config in configurations:
        members = actor_probabilities.get(config)
        if members is None or set(int(seed) for seed in members) != set(seeds):
            raise ValueError(f"C3-E actor membership mismatch for {config}")
        stacked = []
        actor_hashes[config] = {}
        for seed in seeds:
            values = np.asarray(members[seed], dtype=float)
            if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"invalid actor probabilities for {config}/{seed}")
            if expected_shape is None:
                expected_shape = values.shape
            if values.shape != expected_shape:
                raise ValueError("C3-E actor output shapes do not match")
            row_sums = values.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-8):
                raise ValueError(f"actor probabilities are not normalized for {config}/{seed}")
            stacked.append(values)
            actor_hashes[config][str(seed)] = hashlib.sha256(values.tobytes()).hexdigest()
        config_consensus[config] = np.mean(np.stack(stacked, axis=0), axis=0)
    assert expected_shape is not None
    outer_weights = contract["aggregation"]["outer_configuration_weights"]
    weights = np.asarray([float(outer_weights[name]) for name in configurations], dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("C3-E configuration weights must sum to one")
    final = np.sum(np.stack([config_consensus[name] for name in configurations], axis=0) * weights[:, None, None], axis=0)
    actions = np.argmax(final, axis=1).astype(int)
    return final, {
        "configurations": configurations,
        "seeds": seeds,
        "actor_count": len(configurations) * len(seeds),
        "actor_hashes": actor_hashes,
        "configuration_membership": {name: seeds for name in configurations},
        "configuration_consensus_sha256": {name: hashlib.sha256(config_consensus[name].tobytes()).hexdigest() for name in configurations},
        "final_ensemble_sha256": hashlib.sha256(final.tobytes()).hexdigest(),
        "actions": actions.tolist(),
        "aggregation": contract["aggregation"],
    }
