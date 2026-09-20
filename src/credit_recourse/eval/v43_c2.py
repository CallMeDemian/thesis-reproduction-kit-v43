"""Frozen C2 financial comparator, independent of encoder/policy checkpoints."""

from __future__ import annotations

import numpy as np

import pandas as pd

from credit_recourse.rl.v43_reward_math import PHI_COMPONENTS, LOWER_GOOD, materialize_phi_aliases, sector_col

def _percentile(values: pd.Series, ref: np.ndarray) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    ref = np.asarray(ref, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    pct = np.searchsorted(np.sort(ref), arr, side="right") / max(len(ref), 1)
    pct[~np.isfinite(arr)] = np.nan
    return pd.Series(pct, index=values.index)

def _build_sector_refs(train_df: pd.DataFrame) -> tuple[dict[str, dict[str, np.ndarray]], str | None]:
    train_df = materialize_phi_aliases(train_df, next_state=False)
    missing = [c for c in PHI_COMPONENTS if c not in train_df.columns]
    if missing:
        raise ValueError(f"C2 sector-phi weakest rule requires frozen reference components; missing {missing}")
    sec = sector_col(train_df)
    refs: dict[str, dict[str, np.ndarray]] = {}
    for comp in PHI_COMPONENTS:
        refs[comp] = {}
        global_vals = pd.to_numeric(train_df[comp], errors="coerce").dropna().to_numpy(dtype=float)
        refs[comp]["__GLOBAL__"] = np.sort(global_vals)
        if sec and sec in train_df.columns:
            for key, g in train_df.groupby(sec, dropna=False):
                vals = pd.to_numeric(g[comp], errors="coerce").dropna().to_numpy(dtype=float)
                refs[comp][str(key)] = np.sort(vals) if len(vals) >= 20 else refs[comp]["__GLOBAL__"]
    return refs, sec

def c2_row_conditional(df: pd.DataFrame, space, reference_df: pd.DataFrame, fixed_actions: pd.DataFrame | None = None, *, candidate_ids_only: bool = False) -> pd.DataFrame:
    """Row-conditional C2 baseline: frozen sector-relative weakest component top-1 rule.

    fixed_actions carries the active Stage2-recalibrated candidate magnitudes
    embedded in the selected Stage5 checkpoint. C2 is a row-conditional
    comparator, not a train label, but it must use the same active candidate
    vectors as the other Stage6 policies.
    """
    df = materialize_phi_aliases(df.copy(), next_state=False)
    reference_df = materialize_phi_aliases(reference_df.copy(), next_state=False)
    if not candidate_ids_only and (fixed_actions is None or "candidate_id" not in fixed_actions.columns):
        raise ValueError("C2 requires the validated Stage5 checkpoint candidate action frame")
    active_fixed = None if candidate_ids_only else fixed_actions.set_index("candidate_id")
    refs, sec = _build_sector_refs(reference_df)
    missing = [c for c in PHI_COMPONENTS if c not in df.columns]
    if missing:
        raise ValueError(f"C2 sector-phi weakest rule requires phase_eval components; missing {missing}")
    mapping = {
        "derived__debt_to_assets": "DL1_deleverage_mild",
        "derived__financial_cost_to_revenue": "DL1_deleverage_mild",
        "derived__cogs_to_revenue": "OE1_cost_efficiency_mild",
        "derived__sga_to_revenue": "OE1_cost_efficiency_mild",
        "derived__operating_margin": "OE1_cost_efficiency_mild",
        "derived__roa_proxy": "OE1_cost_efficiency_mild",
    }
    quality = pd.DataFrame(index=df.index)
    for comp in PHI_COMPONENTS:
        pct = pd.Series(index=df.index, dtype=float)
        if sec and sec in df.columns:
            for key, idx in df.groupby(sec, dropna=False).groups.items():
                ref = refs[comp].get(str(key), refs[comp]["__GLOBAL__"])
                pct.loc[idx] = _percentile(df.loc[idx, comp], ref)
        else:
            pct = _percentile(df[comp], refs[comp]["__GLOBAL__"])
        quality[comp] = (1.0 - pct if comp in LOWER_GOOD else pct).fillna(0.5)
    weakness = 1.0 - quality
    rows = []
    for idx, r in weakness.iterrows():
        component = str(r.astype(float).idxmax())
        name = mapping[component]
        if candidate_ids_only:
            name = {"DL1_deleverage_mild": "DL", "OE1_cost_efficiency_mild": "OE"}[name]
            vec = {"candidate_id": name}
        else:
            if name not in active_fixed.index:
                raise ValueError(f"C2 mapped candidate missing from validated checkpoint action payload: {name}")
            vec = {col: float(active_fixed.loc[name, col]) for col in space.columns}
            vec["candidate_id"] = "C2_weakest_component_rule"
        vec["c2_mapped_candidate"] = name
        vec["c2_rule_id"] = "frozen_sector_phi_percentile_weakest_component_top1_rule"
        vec["c2_weakest_component"] = component
        vec["c2_weakest_component_quality_pct"] = float(quality.loc[idx, component])
        vec["c2_weakest_component_weakness"] = float(weakness.loc[idx, component])
        rows.append(vec)
    return pd.DataFrame(rows)


