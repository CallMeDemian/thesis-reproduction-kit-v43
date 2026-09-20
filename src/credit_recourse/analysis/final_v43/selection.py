from __future__ import annotations

from pathlib import Path

from .paths import frozen_v13_root


def source_path(root: Path, run_id: str, profile: str) -> Path:
    root = Path(root)
    if profile == "MANUSCRIPT_FROZEN":
        return frozen_v13_root(root) / "frozen_inputs/stage8/canonical_itt_observations.parquet"
    raise ValueError(f"published-result analysis accepts only MANUSCRIPT_FROZEN, got {profile!r}")


def primary_filter(frame, *, reasoning_regime, model_key, budget, execution_policy="STRICT"):
    return frame.loc[
        (frame.reasoning_regime == reasoning_regime)
        & (frame.model_key == model_key)
        & (frame.budget == budget)
        & (frame.execution_policy == execution_policy)
        & (frame.action_space == "FREE8")
        & (frame.replicate == 1)
        & (frame.information_condition == "IC-b")
        & (frame.analysis_population.astype(str).str.lower() == "itt")
    ].copy()
