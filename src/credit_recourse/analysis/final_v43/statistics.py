from __future__ import annotations

import numpy as np
import hashlib
import json


BOOTSTRAP_REPLICATES = 10_000


def canonical_key_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def pcg64_seed(key_json: str) -> int:
    digest = hashlib.sha256(key_json.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def paired_bootstrap(values, *, seed: int, replicates: int = BOOTSTRAP_REPLICATES):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires finite non-empty values")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 128):
        count = min(128, replicates - start)
        draws[start:start + count] = values[rng.integers(0, len(values), size=(count, len(values)))].mean(axis=1)
    neg = int((draws < 0).sum())
    zero = int((draws == 0).sum())
    pos = int((draws > 0).sum())
    return {
        "N": int(values.size), "estimate": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025, method="linear")),
        "ci_high": float(np.quantile(draws, 0.975, method="linear")),
        "n_lt_zero": neg, "n_eq_zero": zero, "n_gt_zero": pos,
        "raw_p": min(1.0, 2.0 * (min(neg + zero, pos + zero) + 1) / (replicates + 1)),
        "bootstrap_draw_sha256": hashlib.sha256(np.asarray(draws, dtype="<f8").tobytes()).hexdigest(),
    }


def summarize(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("empty paired values")
    return {"N": int(values.size), "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0, "min": float(values.min()), "max": float(values.max())}
