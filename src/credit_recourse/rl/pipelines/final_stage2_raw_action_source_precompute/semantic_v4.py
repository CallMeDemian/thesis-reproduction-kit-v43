"""Training-side historical producer for CX/DL/RF semantic action v4.

No Oracle package, score, bin, weight, rating prediction, or evaluation cohort
is imported or accepted here.  All calculations use consecutive factual
financial-statement amounts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SEMANTIC_ACTION_V4_PRODUCER_VERSION = "stage2_factual_semantic_action_v4_v2_positive_events"


def _num(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce")


def materialize_semantic_v4_history(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create historical v4 primitive observations from a t/t+1 raw merge."""
    out = pd.DataFrame(index=merged.index)

    capex_t = _num(merged, "capex").clip(lower=0.0)
    capex_n = _num(merged, "capex__next").clip(lower=0.0)
    dep_t = _num(merged, "depreciation").clip(lower=0.0)
    dep_n = _num(merged, "depreciation__next").clip(lower=0.0)
    growth_t = (capex_t - dep_t).clip(lower=0.0)
    growth_n = (capex_n - dep_n).clip(lower=0.0)
    cx_complete = capex_t.notna() & capex_n.notna() & dep_t.notna() & dep_n.notna()
    cx_eligible = cx_complete & (growth_t > 1.0e-9)
    cx_event = cx_eligible & (growth_n < growth_t - 1.0e-9)
    cx = ((growth_t - growth_n) / growth_t.where(growth_t > 1.0e-9)).clip(0.0, 1.0)
    out["v4_action__growth_capex_reduction_pct"] = cx.where(cx_event, np.nan)
    out["v4_observed__growth_capex_reduction_pct"] = cx_event
    out["v4_actual_action__growth_capex_reduction_pct"] = cx.where(cx_event, 0.0).where(cx_complete, np.nan)
    out["v4_actual_observed__growth_capex_reduction_pct"] = cx_complete
    out["v4_eligible__growth_capex_present"] = cx_eligible
    out["v4_raw__baseline_growth_capex"] = growth_t
    out["v4_raw__next_growth_capex"] = growth_n

    short_t, short_n = _num(merged, "short_debt"), _num(merged, "short_debt__next")
    long_t, long_n = _num(merged, "long_debt"), _num(merged, "long_debt__next")
    cpld_t, cpld_n = _num(merged, "current_portion_long_debt"), _num(merged, "current_portion_long_debt__next")
    bond_t, bond_n = _num(merged, "bond"), _num(merged, "bond__next")
    gross_t, gross_n = long_t + cpld_t, long_n + cpld_n
    total_t, total_n = short_t + gross_t + bond_t, short_n + gross_n + bond_n
    debt_complete = pd.concat([short_t, short_n, gross_t, gross_n, bond_t, bond_n], axis=1).notna().all(axis=1)
    debt_obs = debt_complete & (total_t > 1.0e-9)
    repay_short = (short_t - short_n).clip(lower=0.0)
    repay_gross = (gross_t - gross_n).clip(lower=0.0)
    repay_bond = (bond_t - bond_n).clip(lower=0.0)
    repay_total = repay_short + repay_gross + repay_bond
    no_stack_increase = (short_n <= short_t + 1.0e-9) & (gross_n <= gross_t + 1.0e-9) & (bond_n <= bond_t + 1.0e-9)
    dl_event = debt_obs & no_stack_increase & (repay_total > 1.0e-9)
    out["v4_action__deleveraging_total_debt_pct"] = (repay_total / total_t).clip(0.0, 1.0).where(dl_event, np.nan)
    out["v4_observed__deleveraging_event"] = dl_event
    out["v4_actual_action__deleveraging_total_debt_pct"] = (repay_total / total_t.where(total_t > 1.0e-9)).clip(0.0, 1.0).where(dl_event, 0.0).where(debt_complete, np.nan)
    out["v4_actual_observed__deleveraging_total_debt_pct"] = debt_complete
    out["v4_dl__short_debt_weight"] = (repay_short / repay_total).where(dl_event, np.nan)
    out["v4_dl__gross_ltd_weight"] = (repay_gross / repay_total).where(dl_event, np.nan)
    out["v4_dl__bond_weight"] = (repay_bond / repay_total).where(dl_event, np.nan)

    x = (short_t - short_n).clip(lower=0.0)
    ltd_change = gross_n - gross_t
    bond_change = bond_n - bond_t
    ltd_increase = ltd_change.clip(lower=0.0)
    bond_increase = bond_change.clip(lower=0.0)
    replacement = ltd_increase + bond_increase
    neutrality_error = (total_n - total_t).abs() / x.where(x > 1.0e-9)
    stack_tolerance = 0.10 * x
    rf_event = (
        debt_obs
        & (x > 1.0e-9)
        & (replacement > 1.0e-9)
        & (gross_n >= gross_t - stack_tolerance)
        & (bond_n >= bond_t - stack_tolerance)
        & (neutrality_error <= 0.10)
    )
    out["v4_action__refinancing_short_debt_pct"] = (x / short_t.where(short_t > 1.0e-9)).clip(0.0, 1.0).where(rf_event, np.nan)
    out["v4_observed__refinancing_event"] = rf_event
    out["v4_actual_action__refinancing_short_debt_pct"] = (x / short_t.where(short_t > 1.0e-9)).clip(0.0, 1.0).where(rf_event, 0.0).where(debt_complete, np.nan)
    out["v4_actual_observed__refinancing_short_debt_pct"] = debt_complete
    out["v4_rf__long_weight"] = (ltd_increase / replacement).where(rf_event, np.nan)
    out["v4_rf__bond_weight"] = (bond_increase / replacement).where(rf_event, np.nan)
    out["v4_rf__principal_neutrality_error"] = neutrality_error.where(rf_event, np.nan)
    out["v4_raw__gross_ltd"] = gross_t
    out["v4_raw__next_gross_ltd"] = gross_n
    out["v4_raw__total_interest_bearing_debt"] = total_t
    out["v4_raw__next_total_interest_bearing_debt"] = total_n

    metadata = {
        "producer_version": SEMANTIC_ACTION_V4_PRODUCER_VERSION,
        "source": "consecutive factual financial statements only",
        "active_projection_fields": "v4_actual_action__* with separate completeness masks; event-only v4_action__* retained for P50 calibration",
        "gross_ltd_formula": "noncurrent_long_term_debt + current_portion_long_debt",
        "cx_formula": "positive growth-CAPEX reduction events only: growth_t>0 and growth_t1<growth_t",
        "dl_event_rule": "all debt stacks non-increasing and aggregate repayment positive",
        "rf_event_rule": "short reduction positive; gross LTD and bond non-decreasing within 10% of X; actual total-principal relative change <= 0.10",
        "oracle_used": False,
        "rating_prediction_used": False,
        "ppe_pct_deprecated_for_v4": True,
        "counts": {
            "cx_eligible_growth_capex_positive": int(cx_eligible.sum()),
            "cx_positive_reduction_events": int(cx_event.sum()),
            "cx_non_event_or_increase": int((cx_eligible & ~cx_event).sum()),
            "dl_events": int(dl_event.sum()),
            "rf_events": int(rf_event.sum()),
        },
    }
    return out, metadata


def joint_medoid(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Deterministic L1 medoid after robust per-column median scaling."""
    clean = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if clean.empty:
        raise ValueError(f"No complete historical rows for medoid columns={columns}")
    values = clean.to_numpy(dtype=float)
    center = np.nanmedian(values, axis=0)
    scale = np.nanmedian(np.abs(values - center), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    distance = np.abs((values - center) / scale).sum(axis=1)
    position = int(np.argmin(distance))
    return clean.iloc[position]

