from __future__ import annotations

import pandas as pd

from .pairing import paired
from .selection import primary_filter


CONTRASTS = (
    ("C5-C4", "C5", "C4"),
    ("C4R-C4", "C4R", "C4"),
    ("C6-E-C4R", "C6-E", "C4R"),
    ("C6-E-C6-EX", "C6-E", "C6-EX"),
)
MODELS = ("openai_gpt54mini", "google_gemini31flashlite")
ORACLES = ("alpha", "beta", "gamma")


def reconstruct(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for regime in ("BASELINE", "HIGH"):
        for model in MODELS:
            for budget in ("B1", "BINF"):
                selected = primary_filter(frame, reasoning_regime=regime, model_key=model, budget=budget)
                for contrast, treatment, control in CONTRASTS:
                    for oracle in ORACLES:
                        pair = paired(selected, treatment, control, oracle=oracle)
                        records.append({
                            "model_key": model, "reasoning_regime": regime,
                            "execution_policy": "STRICT", "action_space": "FREE8",
                            "budget": budget, "information_condition": "IC-b", "replicate": 1,
                            "population": "ITT", "contrast": contrast, "oracle": oracle,
                            "N": int(len(pair)), "estimate_reconstructed": float(pair.difference.mean()),
                            "firm_ids": ",".join(sorted(pair.firm_key.astype(str))),
                            "firm_id_sha256": __import__("hashlib").sha256(",".join(sorted(pair.firm_key.astype(str))).encode("utf-8")).hexdigest(),
                        })
    return pd.DataFrame.from_records(records)
