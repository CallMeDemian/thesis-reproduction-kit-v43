"""Frozen corrected Alpha evaluation helpers; no RL control dependencies."""

from __future__ import annotations

from dataclasses import fields

from typing import Any

import numpy as np

import pandas as pd

from credit_recourse.simulator.firm_state import FirmState

from credit_recourse.simulator.historical_source import firm_key

from credit_recourse.simulator.oracle_variables import compute_oracle_variables

def _target_operating_loss_freq_3y(hist: list[FirmState], state_t1: FirmState) -> float | None:
    states = list(hist)[-2:] + [state_t1]
    observed = [float(s.operating_income) for s in states if s.operating_income is not None and np.isfinite(float(s.operating_income))]
    if not observed:
        return None
    return float(sum(value < 0.0 for value in observed))

def _prepare_stage6_context_lookup(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Index canonical nonfinancial history without changing FirmState history.

    ``cap_change_count_3y`` is a target-year rolling event variable in Stage1.
    Its counterfactual t+1 value therefore needs the canonical cumulative event
    history in addition to the financial FirmState history used by the
    BusinessPlan calibrator.
    """
    if 'firm_id' not in base.columns and 'corp_code' not in base.columns:
        return {}
    fid_col = 'firm_id' if 'firm_id' in base.columns else 'corp_code'
    y_col = 'fiscal_year' if 'fiscal_year' in base.columns else ('year' if 'year' in base.columns else None)
    if y_col is None:
        return {}
    ordered = base.copy()
    ordered['_history_year'] = pd.to_numeric(ordered[y_col], errors='coerce')
    ordered = ordered.loc[ordered['_history_year'].notna()].sort_values([fid_col, '_history_year'])
    return {
        str(firm_id): group.reset_index(drop=True)
        for firm_id, group in ordered.groupby(fid_col, dropna=False, sort=False)
    }

def _target_cap_change_count_3y(
    context_lookup: dict[str, pd.DataFrame], firm_id: str, decision_year: int
) -> float:
    """Return Stage1-equivalent t+1 rolling capital-event count.

    For a decision at t, Stage1's target-year window is (t-1, t, t+1).
    Candidate actions contain no capital-issuance event primitive, so the
    pre-specified counterfactual t+1 event increment is zero.  The historical
    portion is recovered exactly as cumulative(t) - cumulative(t-2), avoiding
    the invalid shortcut of carrying the already-aggregated t-2..t value into
    t+1.
    """
    group = context_lookup.get(str(firm_id))
    if group is None or group.empty or 'cap_change_cumulative' not in group.columns:
        raise ValueError(
            f'Canonical history lacks cap_change_cumulative for firm={firm_id}; '
            'cannot construct target-year cap_change_count_3y without freezing the decision-year value.'
        )
    years = pd.to_numeric(group['_history_year'], errors='coerce')
    cumulative = pd.to_numeric(group['cap_change_cumulative'], errors='coerce')
    through_t = cumulative.loc[(years <= int(decision_year)) & cumulative.notna()]
    through_tminus2 = cumulative.loc[(years <= int(decision_year) - 2) & cumulative.notna()]
    if through_t.empty:
        raise ValueError(f'Canonical capital-event history is empty through decision year for firm={firm_id}')
    cumulative_t = float(through_t.iloc[-1])
    cumulative_tminus2 = float(through_tminus2.iloc[-1]) if not through_tminus2.empty else 0.0
    return float(max(0.0, cumulative_t - cumulative_tminus2))

def _state_to_frame_dict(sim_vars: dict[str,Any], base_row: pd.Series) -> dict[str,Any]:
    out=dict(sim_vars)
    # RL-S6-007: preserve Stage1-selected nonfinancial/context variables
    # required by Alpha/Beta/Gamma backend params. These are not simulator
    # formulas and must be supplied from the original phase_eval row.
    passthrough_cols = [
        'industry_median_rating_lag1_self_excl', 'industry_avg_rating_lag1_self_excl',
        'industry_bad_grade_share_lag1_self_excl',
        'cap_change_count_3y', 'financial_data_completeness',
        'ratio_missing_rate', 'nf_ratio_missing_rate',
        'nf_retained_earnings_negative_flag',
        'firm_id', 'year', 'fiscal_year', 'sector_7', 'industry_class'
    ]
    for c in passthrough_cols:
        if c in base_row and c not in out:
            out[c]=base_row[c]
    if 'industry_bad_grade_share_lag1_self_excl' not in out and 'alpha__industry_bad_grade_share_lag1_self_excl' in base_row:
        out['industry_bad_grade_share_lag1_self_excl'] = base_row['alpha__industry_bad_grade_share_lag1_self_excl']
    if 'ratio_missing_rate' not in out and 'nf_ratio_missing_rate' in out:
        out['ratio_missing_rate'] = out['nf_ratio_missing_rate']
    return out

def state_from_financial(raw, prefix):
    values = {}
    for f in fields(FirmState):
        value = raw.get(prefix + f.name)
        values[f.name] = None if value is None or pd.isna(value) else value
    return FirmState(**values)

def candidate_variables(raw, base_raw, history, context_lookup):
    state = state_from_financial(raw, "state__")
    target = state_from_financial(raw, "sim__")
    hist = sorted((s for s in history[firm_key(state.firm_id)] if s.year <= state.year), key=lambda s: s.year)[-3:]
    exog = {k: base_raw.get(k) for k in (
        "industry_median_rating_lag1_self_excl", "industry_avg_rating_lag1_self_excl",
        "cap_change_count_3y", "financial_data_completeness", "nf_retained_earnings_negative_flag",
        "nf_ratio_missing_rate", "ratio_missing_rate")}
    exog["cap_change_count_3y_target"] = _target_cap_change_count_3y(context_lookup, str(state.firm_id), state.year)
    exog["operating_loss_freq_3y_target"] = _target_operating_loss_freq_3y(hist, target)
    variables = _state_to_frame_dict(compute_oracle_variables(target, prev_state=state, exogenous=exog), pd.Series(base_raw))
    return variables


def candidate_alpha(raw, base_raw, history, context_lookup, scorer, selected):
    variables = candidate_variables(raw, base_raw, history, context_lookup)
    return alpha_from_variables(variables, scorer, selected)


def alpha_from_variables(variables, scorer, selected):
    scored = scorer({name: variables.get(name) for name in selected})
    if not np.isfinite(scored["R_score"]):
        raise ValueError("Nonfinite corrected candidate Alpha")
    return {"alpha": float(scored["R_score"]),
            **{"alpha_input__" + k: variables.get(k) for k in selected},
            **{"alpha_item__" + k: scored["item_scores"][k] for k in selected},
            **{"alpha_imputed__" + k: scored["imputed"][k] for k in selected}}
