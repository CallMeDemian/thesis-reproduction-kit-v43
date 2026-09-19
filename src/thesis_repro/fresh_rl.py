"""Fresh RL graph contracts and independent verification.

This module deliberately does not train a model.  It verifies the exact actor
graph before a training result can be consumed by C3-E, and it refuses to
reconstruct an actor from a frozen checkpoint or from an incomplete graph.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .c3e import aggregate_hierarchical, load_fresh_rl_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def actor_key(configuration: str, seed: int) -> str:
    return f"{configuration}:seed-{int(seed)}"


def verify_actor_graph(
    records: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    root: Path | None = None,
    evaluation_cohort_hash: str | None = None,
) -> dict[str, Any]:
    """Verify membership, identity, and provenance of all selected actors."""
    configs = [str(value) for value in contract["selected_configurations"]]
    seeds = [int(value) for value in contract["seeds"]]
    expected = {actor_key(config, seed) for config in configs for seed in seeds}
    seen: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for record in records:
        config = str(record.get("configuration", ""))
        seed = int(record.get("seed", -1))
        key = actor_key(config, seed)
        if key in seen:
            errors.append(f"duplicate_actor:{key}")
        seen[key] = record
        if key not in expected:
            errors.append(f"unexpected_actor:{key}")
        if record.get("evaluation_base_year") != contract["evaluation_cohort"]["evaluation_base_year"]:
            errors.append(f"evaluation_year_mismatch:{key}")
        if int(record.get("evaluation_firm_count", -1)) != int(contract["evaluation_cohort"]["firm_count"]):
            errors.append(f"evaluation_cohort_count_mismatch:{key}")
        if record.get("trained_on_evaluation_cohort") is True:
            errors.append(f"evaluation_leakage:{key}")
        if record.get("oracle_output_in_reward") is True:
            errors.append(f"oracle_reward_leakage:{key}")
        checkpoint = Path(str(record.get("checkpoint_path", "")))
        if root is not None and not checkpoint.is_absolute():
            checkpoint = Path(root) / checkpoint
        if not checkpoint.is_file():
            errors.append(f"checkpoint_missing:{key}")
        elif record.get("checkpoint_sha256") != _sha256(checkpoint):
            errors.append(f"checkpoint_hash_mismatch:{key}")
    missing = sorted(expected - set(seen))
    errors.extend(f"missing_actor:{key}" for key in missing)
    return {
        "schema_version": "fresh_rl_actor_graph_verification_v1",
        "status": "PASS" if not errors else "FAILED",
        "expected_actor_count": len(expected),
        "observed_actor_count": len(seen),
        "expected_membership": sorted(expected),
        "observed_membership": sorted(seen),
        "evaluation_cohort_hash": evaluation_cohort_hash,
        "errors": errors,
    }


def build_c3e_release(
    actor_probabilities: Mapping[str, Mapping[int, Any]],
    contract: Mapping[str, Any],
    *,
    parent_hashes: list[str],
    evaluation_cohort_hash: str,
) -> tuple[Any, dict[str, Any]]:
    """Aggregate verified actor probabilities and return a release receipt."""
    final, detail = aggregate_hierarchical(actor_probabilities, contract)
    receipt = {
        "schema_version": "fresh_c3e_release_v1",
        "status": "PASS",
        "execution_class": "REAL_COMPUTE",
        "encoder": contract["encoder"],
        "configuration_membership": detail["configuration_membership"],
        "seed_membership": detail["seeds"],
        "actor_hashes": detail["actor_hashes"],
        "configuration_consensus_sha256": detail["configuration_consensus_sha256"],
        "final_ensemble_sha256": detail["final_ensemble_sha256"],
        "evaluation_cohort_hash": evaluation_cohort_hash,
        "aggregation": detail["aggregation"],
        "parent_hashes": parent_hashes,
        "release_hash": hashlib.sha256(json.dumps(detail, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    return final, receipt


def load_contract(root: Path) -> dict[str, Any]:
    return load_fresh_rl_contract(Path(root))
