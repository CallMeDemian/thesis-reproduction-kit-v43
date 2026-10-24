"""Shared helpers for the ablation scripts (paths, loaders, bootstrap rows, Holm)."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parents[2]
ABL = KIT / "ablation_20261102"
RESULTS = ABL / "results"
FIGURES = ABL / "figures"
if str(KIT / "src") not in sys.path:
    sys.path.insert(0, str(KIT / "src"))

from credit_recourse.analysis.final_v43.statistics import canonical_key_json, paired_bootstrap, pcg64_seed  # noqa: E402

V13 = "frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN"
S9 = "frozen/original_release/evaluation/llm_final_evaluation_20260913/stage9/HIGH_STRICT_ITT"
OBS_PATH = KIT / V13 / "frozen_inputs/stage8/canonical_itt_observations.parquet"
SUPP_PATH = KIT / V13 / "canonical_v13/V13_SUPPLEMENTAL_RESULTS.csv"
PARENT_PATH = KIT / V13 / "canonical_v13/parent_gate_sensitivity_v13.csv"
PARENT_FLAGS_PATH = KIT / V13 / "canonical_v13/parent_gate_firm_flags_v13.csv"
PRIM_BASE = KIT / V13 / "frozen_inputs/stage8/V13_BASELINE_STRICT_PRIMARY_FULLSTREAM.csv"
PRIM_HIGH = KIT / V13 / "frozen_inputs/stage8/V13_HIGH_STRICT_PRIMARY_FULLSTREAM.csv"
C3E_PATH = KIT / S9 / "llm_stage9_llm_rl_comparison.csv"
BASE_COMMIT = "071848881349c8675bb4e15193a747fa02679986"

MODELS = ("openai_gpt54mini", "google_gemini31flashlite")
MODEL_CODE = {"openai_gpt54mini": "GPT", "google_gemini31flashlite": "GEMINI"}
ORACLES = ("alpha", "beta", "gamma")

RESULT_COLS = [
    "result_id", "status", "contrast_id", "lhs", "rhs", "model", "reasoning_regime", "oracle", "budget",
    "action_space", "execution_policy", "replicate", "information_condition", "n_firms", "estimate", "ci_low",
    "ci_high", "p_raw", "holm_p", "holm_family", "holm_p_60", "source_file", "source_key", "source_sha256",
    "analysis_commit", "notes",
]


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with Path(path).open("rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def git_commit() -> str:
    out = subprocess.run(["git", "-C", str(KIT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return out[:12]


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values (same function as e1_analyze.holm)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(len(p))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adj[index] = running
    return adj


def load_obs() -> pd.DataFrame:
    return pd.read_parquet(OBS_PATH)


LFS_POINTER_HEAD = b"version https://git-lfs.github.com/spec/v1"


def require_file(path: Path) -> Path:
    """Stop with a plain message when an input is missing or is still a Git LFS pointer."""
    path = Path(path)
    rel = path.relative_to(KIT).as_posix()
    if not path.is_file():
        raise SystemExit(f"missing input: {rel}")
    with path.open("rb") as handle:
        if handle.read(len(LFS_POINTER_HEAD)) == LFS_POINTER_HEAD:
            raise SystemExit(f"{rel} is a Git LFS pointer. Run: git lfs pull -I \"{rel}\"")
    return path


def seed_key(*, analysis_id, lhs, rhs, model, oracle, reasoning_regime, budget, action_space,
             execution_policy="STRICT", information_condition="IC-b", population="ITT", replicate="RUN1") -> dict:
    return {"action_space": action_space, "analysis_id": analysis_id, "budget": budget, "execution_policy": execution_policy,
            "information_condition": information_condition, "lhs": lhs, "model": model, "oracle": oracle,
            "population": population, "reasoning_regime": reasoning_regime, "replicate": replicate, "rhs": rhs}


def boot(values, key: dict) -> dict:
    kj = canonical_key_json(key)
    res = paired_bootstrap(np.asarray(values, dtype=float), seed=pcg64_seed(kj))
    res["key_json"] = kj
    return res


def paired_by(frame: pd.DataFrame, axis: str, left, right, score: str, keys: list) -> pd.DataFrame:
    """1:1 pairing on `keys` between two levels of `axis`; fails on duplicates and one-sided rows."""
    sub = frame.loc[frame[axis].isin([left, right]), keys + [axis, score]]
    if sub.duplicated(keys + [axis]).any():
        raise ValueError(f"duplicate cells before pairing on {axis}")
    wide = sub.pivot(index=keys, columns=axis, values=score).reset_index()
    if left not in wide or right not in wide or wide[[left, right]].isna().any().any():
        raise ValueError(f"unpaired rows for {left} vs {right}")
    wide = wide.sort_values("firm_key").reset_index(drop=True)
    wide["difference"] = wide[left] - wide[right]
    return wide


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", na_rep="N/A", encoding="utf-8")
