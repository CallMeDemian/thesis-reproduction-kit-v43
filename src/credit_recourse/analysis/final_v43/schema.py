from __future__ import annotations

REQUIRED = {
    "firm_key", "model_key", "reasoning_regime", "execution_policy",
    "action_space", "budget", "information_condition", "replicate",
    "policy_condition", "analysis_population", "strict_usable",
    "R_score_alpha", "R_score_beta", "R_score_gamma",
    "delta_R_score_alpha", "delta_R_score_beta", "delta_R_score_gamma",
}


def validate(frame) -> None:
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"analysis source missing columns: {sorted(missing)}")
    if frame["firm_key"].nunique() != 575:
        raise ValueError("analysis source must cover 575 firms")
    identity = [
        "firm_key", "model_key", "reasoning_regime", "execution_policy",
        "action_space", "budget", "information_condition", "replicate",
        "policy_condition",
    ]
    duplicates = frame.loc[frame.duplicated(identity, keep=False), identity]
    if not duplicates.empty:
        raise ValueError(f"duplicate analysis identity keys: {len(duplicates)} rows")
    if not frame["replicate"].isin([1, 2]).all():
        raise ValueError("unexpected replicate value")
