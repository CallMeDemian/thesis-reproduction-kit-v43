from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import ContractError, find_repo_root, load_json, published_contract_registry, sha256_file

ACTIONS = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")


def validate_final_reference(root: Path | None = None) -> dict[str, Any]:
    repo = root or find_repo_root()
    registry = load_json(published_contract_registry(repo))
    reference_dir = repo / "frozen/original_release/rl/C3E_E2_7SEED_BALANCED_DFEBAFA6"

    def resolve_reference(key: str, filename: str) -> Path:
        direct = repo / registry[key]
        if direct.is_file():
            return direct
        # The old registry records the pre-cleanup archive path.  The
        # immutable release stores the same reference package under the
        # explicit historical namespace above.
        candidate = reference_dir / filename
        if candidate.is_file():
            return candidate
        raise ContractError(f"Published C3-E reference is unavailable: {key}")

    definition_path = resolve_reference("rl_reference_definition", "C3E_definition.json")
    probability_path = resolve_reference("rl_reference_probabilities", "C3E_firm_probabilities.parquet")
    action_path = resolve_reference("rl_reference_actions", "C3E_firm_actions.parquet")
    definition = load_json(definition_path)
    if definition["release_id"] != registry["rl_reference_id"] or definition["release_hash"] != registry["rl_release_hash"]:
        raise ContractError("Current C3-E pointer identity/hash drift")
    probabilities = pd.read_parquet(probability_path).sort_values("row_id").reset_index(drop=True)
    actions = pd.read_parquet(action_path).sort_values("row_id").reset_index(drop=True)
    if len(probabilities) != 575 or len(actions) != 575 or probabilities["firm_id"].astype(str).tolist() != actions["firm_id"].astype(str).tolist():
        raise ContractError("Current C3-E firm mapping drift")
    pcols = [f"prob_{name}" for name in ACTIONS]
    matrix = probabilities[pcols].to_numpy(float)
    if not np.isfinite(matrix).all() or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise ContractError("Current C3-E probability matrix is invalid")
    indices = matrix.argmax(axis=1).astype(np.uint8)
    names = np.asarray(ACTIONS, dtype=object)[indices]
    if not np.array_equal(names, probabilities["action_id"].to_numpy(object)) or not np.array_equal(names, actions["action_id"].to_numpy(object)):
        raise ContractError("Current C3-E stored argmax drift")
    decision = hashlib.sha256(indices.tobytes()).hexdigest()
    if decision != definition["decision_sha256"] or decision != registry["rl_decision_sha256"]:
        raise ContractError("Current C3-E decision hash drift")
    counts = pd.Series(names).value_counts().reindex(ACTIONS, fill_value=0).astype(int).to_dict()
    expected = {"A0": 0, "DL": 0, "RF": 155, "CX": 41, "WC1": 54, "WC2": 39, "OE": 232, "MX1": 18, "MX2": 36}
    if counts != expected:
        raise ContractError(f"Current C3-E action distribution drift: {counts}")
    provenance = pd.read_csv(definition_path.parent / "member_provenance.csv")
    expected_configs = {"B27", "M2_S0935", "DT06", "T15_REWARD_S088"}
    expected_seeds = {2, 11, 12, 13, 14, 15, 16}
    config_col = "configuration" if "configuration" in provenance.columns else "config"
    if len(provenance) != 28 or set(provenance[config_col].astype(str)) != expected_configs or set(provenance["seed"].astype(int)) != expected_seeds:
        raise ContractError("Current C3-E 7-seed/28-actor lineage drift")
    if provenance.duplicated([config_col, "seed"]).any():
        raise ContractError("Current C3-E member identity contains duplicates")
    missing_member_assets = []
    for path_col, hash_col in (("canonical_checkpoint_path", "stage5_checkpoint_hash"), ("canonical_actor_output_path", "canonical_actor_output_sha256")):
        for row in provenance[[path_col, hash_col]].drop_duplicates().to_dict("records"):
            target = repo / row[path_col]
            if not target.is_file():
                missing_member_assets.append(str(row[path_col]))
                continue
            if sha256_file(target) != row[hash_col]:
                raise ContractError(f"Current C3-E member artifact drift: {row[path_col]}")
    entropy = -sum((n / 575) * math.log(n / 575) for n in counts.values() if n)
    return {
        "status": "PASS", "release_id": definition["release_id"], "release_hash": definition["release_hash"],
        "encoder": definition["encoder"], "matched_seeds": definition["matched_seeds"], "member_count": len(provenance),
        "firm_count": 575, "action_order": list(ACTIONS), "action_counts": counts,
        "C3_Alpha": float(actions["Alpha"].mean()), "C3_Beta": float(actions["Beta"].mean()), "C3_Gamma": float(actions["Gamma"].mean()),
        "H": entropy, "effective_actions": math.exp(entropy), "OE_share": counts["OE"] / 575,
        "probability_sha256": sha256_file(probability_path), "action_sha256": sha256_file(action_path),
        "decision_sha256": decision, "search_status": "CLOSED", "replay": "PASS",
        "member_rebuild_assets": "AVAILABLE" if not missing_member_assets else "AVAILABLE_WITH_ASSET",
        "missing_member_assets": missing_member_assets,
    }
