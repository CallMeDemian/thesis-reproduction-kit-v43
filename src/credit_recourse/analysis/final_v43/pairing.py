from __future__ import annotations

import pandas as pd


def paired(frame: pd.DataFrame, left: str, right: str, *, oracle: str = "alpha") -> pd.DataFrame:
    # Final V1.3 contrasts are defined on policy-value deltas relative to the
    # no-op baseline, not on the raw R score columns.
    score = f"delta_R_score_{oracle}"
    # Stage8 distinguishes execution policy (STRICT_ITT/REPAIRED_ITT)
    # from the experimental condition (C4/C5/C4R/C6-E/C6-EX).  Pairing
    # must therefore use policy_condition for the left/right contrast.
    keys = ["firm_key", "model_key", "reasoning_regime", "execution_policy", "action_space", "budget", "information_condition", "replicate"]
    subset = frame.loc[frame["policy_condition"].isin([left, right]), keys + ["policy_condition", score]].copy()
    duplicates = subset.duplicated(keys + ["policy_condition"], keep=False)
    if bool(duplicates.any()):
        raise ValueError("duplicate treatment/control cells before pairing")
    wide = subset.pivot(index=keys, columns="policy_condition", values=score).reset_index()
    if left not in wide or right not in wide:
        raise ValueError(f"cannot pair {left} and {right}")
    # A frozen Stage8 file contains cells that are not present in every
    # contrast.  Paired statistics are defined on the complete intersection,
    # so one-sided rows must be removed before calculating differences.
    wide = wide.dropna(subset=[left, right]).copy()
    # The caller supplies treatment as ``left`` and control as ``right``;
    # every canonical contrast is treatment minus control.
    wide["difference"] = wide[left] - wide[right]
    return wide
