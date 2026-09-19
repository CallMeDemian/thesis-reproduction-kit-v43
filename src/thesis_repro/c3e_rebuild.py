from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .paths import ROOT, write_json


ACTIONS = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")
CONFIGS = ("B27", "M2_S0935", "DT06", "T15_REWARD_S088")
SEEDS = (2, 11, 12, 13, 14, 15, 16)


def rebuild_original_c3e() -> dict[str, Any]:
    import pyarrow.parquet as pq

    source = ROOT / "frozen/original_release/rl/C3E_E2_7SEED_BALANCED_DFEBAFA6"
    output = ROOT / "runs/original_c3e_rebuild/05_c3e"
    output.mkdir(parents=True, exist_ok=True)
    per_config = []
    for config in CONFIGS:
        members = []
        for seed in SEEDS:
            path = source / "members" / config / f"seed_{seed}" / "actor_critic_outputs.parquet"
            frame = pq.read_table(path).to_pandas().set_index(["row_id", "firm_id", "fiscal_year"])
            frame = frame[[f"probability__{action}" for action in ACTIONS]].rename(columns={f"probability__{action}": f"prob_{action}" for action in ACTIONS})
            members.append(frame)
        per_config.append(pd.concat(members).groupby(level=[0, 1, 2]).mean())
    probabilities = (sum(per_config) / len(per_config)).reset_index()
    pcols = [f"prob_{action}" for action in ACTIONS]
    matrix = probabilities[pcols].to_numpy(float)
    indices = matrix.argmax(axis=1).astype(np.uint8)
    probabilities["action_index"] = indices
    probabilities["action_id"] = np.asarray(ACTIONS, dtype=object)[indices]
    probabilities = probabilities[["row_id", "firm_id", "fiscal_year", *pcols, "action_index", "action_id"]]
    actions = probabilities[["row_id", "firm_id", "fiscal_year", "action_index", "action_id"]].copy()
    probabilities.to_parquet(output / "C3E_firm_probabilities.parquet", index=False)
    actions.to_parquet(output / "C3E_firm_actions.parquet", index=False)
    original_probabilities = pd.read_parquet(source / "C3E_firm_probabilities.parquet").sort_values("row_id").reset_index(drop=True)
    original_actions = pd.read_parquet(source / "C3E_firm_actions.parquet").sort_values("row_id").reset_index(drop=True)
    rebuilt_probabilities = probabilities.sort_values("row_id").reset_index(drop=True)
    rebuilt_actions = actions.sort_values("row_id").reset_index(drop=True)
    max_abs = max(float(np.max(np.abs(rebuilt_probabilities[col] - original_probabilities[col]))) for col in pcols)
    action_match = bool(np.array_equal(rebuilt_actions["action_id"].to_numpy(), original_actions["action_id"].to_numpy()))
    decision_sha256 = hashlib.sha256(indices.tobytes()).hexdigest()
    report = {
        "status": "PASS" if len(probabilities) == 575 and max_abs <= 1e-6 and action_match and decision_sha256 == "dfebafa6b50e55c2f5019766e0e540af9964b2a9ab7d4e49805d03e8011b792e" else "FAIL",
        "release": "original",
        "run_id": "original_c3e_rebuild",
        "config_count": len(CONFIGS),
        "seed_count_per_config": len(SEEDS),
        "member_count": len(CONFIGS) * len(SEEDS),
        "firm_count": len(probabilities),
        "max_probability_abs_diff": max_abs,
        "probability_match": max_abs <= 1e-6,
        "action_match": action_match,
        "decision_sha256": decision_sha256,
        "expected_decision_sha256": "dfebafa6b50e55c2f5019766e0e540af9964b2a9ab7d4e49805d03e8011b792e",
        "outputs": {
            "probabilities": str((output / "C3E_firm_probabilities.parquet").relative_to(ROOT)),
            "actions": str((output / "C3E_firm_actions.parquet").relative_to(ROOT)),
        },
    }
    write_json(output / "rebuild_report.json", report)
    return report
