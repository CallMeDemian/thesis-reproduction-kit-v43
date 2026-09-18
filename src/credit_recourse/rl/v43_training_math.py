"""Outcome-independent loss and validation math preserved from the original pipelines.
No feature assembly, legacy Action object, checkpoint loading, or training entrypoint.
"""

from __future__ import annotations

import numpy as np

import pandas as pd

import torch

from credit_recourse.rl.common.actions import PROJECTION_NEAR_TIE_MARGIN

from credit_recourse.rl.common.reward_contract import EXPLICIT_COMPONENT_COLUMNS, compose_reward

RATING_SUPERVISION_CONTRACT_VERSION = "projected_observed_rating_event_nearest_p50_coverage_non_neartie_v5"

def _weighted_mean_one(weights: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    counts = counts.detach().float().clamp_min(0.0)
    denom = counts.sum().clamp_min(1e-12)
    return (weights.float() * counts).sum() / denom

def _normalize_by_soft_mass(weights: torch.Tensor, counts: torch.Tensor, *, cap: float | None = None) -> torch.Tensor:
    out = weights.float() / _weighted_mean_one(weights.float(), counts).clamp_min(1e-12)
    if cap is not None and float(cap) > 0:
        # Preserve the user-visible cap as a hard maximum.  We deliberately do
        # not renormalize after capping because that could push rare-candidate
        # weights above the cap again.
        out = out.clamp(max=float(cap))
    return out

def _class_balance_audit(y_soft: torch.Tensor, labels: list[str], family_map: dict[str, str], *, enabled: bool, beta: float, weight_cap: float) -> tuple[torch.Tensor, dict, pd.DataFrame]:
    """Return per-class BC loss weights and an auditable class-balance table.

    The default path returns all-ones weights so legacy Stage4 behaviour is
    unchanged unless --class-balanced is explicitly enabled.  When enabled,
    weights are computed from the effective number of soft-target samples
    (Cui et al., 2019), normalized to soft-mass weighted mean 1, and capped to
    prevent rare-class noise from dominating the BC anchor.
    """
    if not (0.0 <= float(beta) < 1.0):
        raise ValueError(f"class_balance_beta must be in [0, 1); got {beta}")
    if float(weight_cap) <= 0.0:
        raise ValueError(f"class_weight_cap must be positive; got {weight_cap}")

    counts = y_soft.detach().float().sum(dim=0).clamp_min(1.0)
    if enabled:
        beta_t = torch.full_like(counts, float(beta))
        eff_num = (1.0 - torch.pow(beta_t, counts)) / max(1e-12, (1.0 - float(beta)))
        weights = (1.0 / eff_num.clamp_min(1e-12))
        weights = (weights / weights.mean().clamp_min(1e-12)).clamp(max=float(weight_cap))
    else:
        weights = torch.ones_like(counts)

    rows = []
    total = float(counts.sum().item())
    for label, count, weight in zip(labels, counts.cpu().tolist(), weights.cpu().tolist()):
        rows.append({
            'candidate_id': str(label),
            'action_family': family_map[str(label)],
            'soft_target_count': float(count),
            'soft_target_share': float(count / total) if total > 0 else 0.0,
            'class_weight': float(weight),
        })
    audit_df = pd.DataFrame(rows)
    audit = {
        'class_balanced_loss': bool(enabled),
        'class_balance_beta': float(beta),
        'class_weight_cap': float(weight_cap),
        'class_weight_formula': 'effective_number_inverse_soft_mass_mean1_capped',
        'class_weights_by_candidate': {str(r['candidate_id']): float(r['class_weight']) for _, r in audit_df.iterrows()},
        'soft_target_counts_by_candidate': {str(r['candidate_id']): float(r['soft_target_count']) for _, r in audit_df.iterrows()},
    }
    return weights, audit, audit_df

def _family_balance_audit(y_soft: torch.Tensor, labels: list[str], family_map: dict[str, str], *, enabled: bool, power: float, weight_cap: float) -> tuple[torch.Tensor, dict, pd.DataFrame]:
    """Return per-candidate family-balance weights.

    Family balancing operates at the economic action-family level rather than
    at the individual candidate level.  This is intentionally oracle-free: the
    family map comes from the action contract semantics.  The goal is to keep
    the BC warm-start from being dominated by broad historical pseudo-actions
    such as capex/WC and cost-efficiency families.
    """
    if float(power) < 0.0:
        raise ValueError(f"family_balance_power must be non-negative; got {power}")
    if float(weight_cap) <= 0.0:
        raise ValueError(f"family_weight_cap must be positive; got {weight_cap}")
    unknown = [x for x in labels if x not in family_map]
    if unknown:
        raise ValueError(f"Stage4 family-balanced BC has no family mapping for candidates: {unknown}")

    counts = y_soft.detach().float().sum(dim=0).clamp_min(1.0)
    family_order = list(dict.fromkeys(family_map[str(label)] for label in labels))
    family_counts: dict[str, float] = {fam: 0.0 for fam in family_order}
    for label, count in zip(labels, counts.cpu().tolist()):
        fam = family_map[str(label)]
        family_counts[fam] = float(family_counts.get(fam, 0.0) + float(count))

    raw = []
    for label in labels:
        fam = family_map[str(label)]
        fc = max(float(family_counts.get(fam, 0.0)), 1.0)
        raw.append(fc ** (-float(power)) if enabled else 1.0)
    weights = torch.tensor(raw, dtype=torch.float32)
    if enabled:
        weights = _normalize_by_soft_mass(weights, counts, cap=float(weight_cap))
    else:
        weights = torch.ones_like(counts)

    rows = []
    total = float(counts.sum().item())
    family_weight_by_family = {}
    for fam in family_order:
        idxs = [i for i, lab in enumerate(labels) if family_map.get(str(lab)) == fam]
        if idxs:
            family_weight_by_family[fam] = float(np.mean([float(weights[i].item()) for i in idxs]))
        else:
            family_weight_by_family[fam] = 0.0
    for label, count, weight in zip(labels, counts.cpu().tolist(), weights.cpu().tolist()):
        fam = family_map[str(label)]
        rows.append({
            'candidate_id': str(label),
            'action_family': fam,
            'soft_target_count': float(count),
            'soft_target_share': float(count / total) if total > 0 else 0.0,
            'family_soft_target_count': float(family_counts.get(fam, 0.0)),
            'family_soft_target_share': float(family_counts.get(fam, 0.0) / total) if total > 0 else 0.0,
            'family_weight': float(weight),
        })
    audit_df = pd.DataFrame(rows)
    audit = {
        'family_balanced_loss': bool(enabled),
        'family_balance_power': float(power),
        'family_weight_cap': float(weight_cap),
        'family_weight_formula': 'inverse_family_soft_count_power_soft_mass_mean1_capped' if enabled else 'disabled_all_ones',
        'action_family_by_candidate': {str(x): family_map[str(x)] for x in labels},
        'family_soft_counts': {str(k): float(v) for k, v in family_counts.items()},
        'family_weights_by_family': {str(k): float(v) for k, v in family_weight_by_family.items()},
        'family_weights_by_candidate': {str(r['candidate_id']): float(r['family_weight']) for _, r in audit_df.iterrows()},
    }
    return weights, audit, audit_df

def _combine_stage4_loss_weights(
    class_weights: torch.Tensor,
    family_weights: torch.Tensor,
    y_soft: torch.Tensor,
    *,
    combined_weight_cap: float,
) -> torch.Tensor:
    if float(combined_weight_cap) <= 0.0:
        raise ValueError(f"combined_weight_cap must be positive; got {combined_weight_cap}")
    counts = y_soft.detach().float().sum(dim=0).clamp_min(1.0)
    weights = class_weights.detach().float() * family_weights.detach().float()
    weights = _normalize_by_soft_mass(weights, counts, cap=float(combined_weight_cap))
    return weights

def _supervision_reason(event: bool, coverage: bool, near_tie: bool) -> str:
    if not event: return 'no_new_rating_event'
    if not coverage: return 'insufficient_observed_action_coverage'
    if near_tie: return 'near_tie_projection'
    return ''

def _validate_projected_rating_supervision(df: pd.DataFrame, labels: list[str]) -> dict:
    required = [
        'factual_transition_id', 'candidate_id', 'nearest_P50_candidate_id',
        'rating_reward_value', 'rating_reward_observed',
        'candidate_rating_outcome_observed', 'rating_supervision_value',
        'rating_supervision_available', 'rating_supervision_source',
        'factual_rating_event_observed', 'rating_supervision_hard_eligible',
        'rating_supervision_exclusion_reason',
        'rating_supervision_projected_candidate_id',
        'rating_supervision_target_candidate_id',
        'rating_supervision_numeric_contribution',
        'near_tie_flag', 'projection_margin', 'projection_near_tie_margin',
        'projection_distance', 'out_of_library_flag',
        'projection_observed_variable_dimension_count',
        'projection_observed_variable_fraction',
        'projection_min_observed_variable_dimension_count',
        'projection_observed_coverage_sufficient',
        'projection_available', 'projection_unavailable_reason',
        'reward_total_raw', 'reward_mean_train', 'reward_std_train', 'reward_train',
        'reward_total', *EXPLICIT_COMPONENT_COLUMNS,
        'lambda_merton', 'lambda_fcff', 'lambda_profitability', 'lambda_liquidity',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Stage5 projected factual rating supervision contract is incomplete: {missing}')
    if df['rating_reward_observed'].fillna(False).astype(bool).any():
        raise ValueError('Stage5 grid must not claim an observed exact-candidate rating reward')
    if df['candidate_rating_outcome_observed'].fillna(False).astype(bool).any():
        raise ValueError('Stage5 projected supervision must not be represented as an observed P50 outcome')
    if pd.to_numeric(df['rating_reward_value'], errors='coerce').notna().any():
        raise ValueError('Stage5 exact-candidate rating_reward_value must remain NaN')

    available = df['rating_supervision_available'].fillna(False).astype(bool)
    event_observed = df['factual_rating_event_observed'].fillna(False).astype(bool)
    hard_eligible = df['rating_supervision_hard_eligible'].fillna(False).astype(bool)
    near_tie = df['near_tie_flag'].fillna(False).astype(bool)
    coverage_sufficient = df['projection_observed_coverage_sufficient'].fillna(False).astype(bool)
    projection_available = df['projection_available'].fillna(False).astype(bool)
    observed_count = pd.to_numeric(df['projection_observed_variable_dimension_count'], errors='coerce')
    if observed_count.isna().any() or not (projection_available == (observed_count > 0)).all():
        raise ValueError('Stage5 projection_available disagrees with observed controllable-action support')
    out_of_library = df['out_of_library_flag'].fillna(False).astype(bool)
    margin = pd.to_numeric(df['projection_margin'], errors='coerce')
    threshold = pd.to_numeric(df['projection_near_tie_margin'], errors='coerce')
    if threshold.isna().any() or not np.allclose(
        threshold.to_numpy(dtype=float), PROJECTION_NEAR_TIE_MARGIN, atol=0.0, rtol=0.0
    ):
        raise ValueError('Stage5 near-tie threshold differs from the fixed Stage2 projection contract')
    if margin[projection_available].isna().any() or margin[~projection_available].notna().any() or not (
        near_tie == (projection_available & (margin < PROJECTION_NEAR_TIE_MARGIN))
    ).all():
        raise ValueError('Stage5 near_tie_flag disagrees with projection_margin')
    if not (hard_eligible == (event_observed & coverage_sufficient & ~near_tie)).all():
        raise ValueError('Stage5 hard rating supervision must equal new event AND sufficient observed-action coverage AND non-near-tie; out_of_library is diagnostic only')
    group_sizes = df.groupby('factual_transition_id', sort=False).size()
    if not (group_sizes == len(labels)).all():
        raise ValueError('Stage5 requires the complete canonical nine-candidate grid for every factual transition')
    candidate_sets = df.groupby('factual_transition_id', sort=False)['candidate_id'].agg(
        lambda x: set(x.astype(str))
    )
    expected_labels = set(labels)
    if not candidate_sets.map(lambda x: x == expected_labels).all():
        raise ValueError('Stage5 factual groups do not all contain the exact P50 action vocabulary')
    event_by_group = event_observed.groupby(df['factual_transition_id'], sort=False).agg(
        lambda values: values.iloc[0] if values.nunique(dropna=False) == 1 else np.nan
    )
    hard_by_group = hard_eligible.groupby(df['factual_transition_id'], sort=False).agg(
        lambda values: values.iloc[0] if values.nunique(dropna=False) == 1 else np.nan
    )
    near_tie_by_group = near_tie.groupby(df['factual_transition_id'], sort=False).agg(
        lambda values: values.iloc[0] if values.nunique(dropna=False) == 1 else np.nan
    )
    if event_by_group.isna().any() or hard_by_group.isna().any() or near_tie_by_group.isna().any():
        raise ValueError('Stage5 event, near-tie, and hard-eligibility provenance must be constant within each factual group')
    supervision_counts = available.groupby(df['factual_transition_id'], sort=False).sum()
    if not (supervision_counts.astype(int) == hard_by_group.astype(bool).astype(int)).all():
        raise ValueError('Stage5 requires one projected row only for new-rating-event non-near-tie factual transitions')

    reasons = df['rating_supervision_exclusion_reason'].fillna('').astype(str)
    expected_reasons = np.array([
        _supervision_reason(bool(e), bool(c), bool(n))
        for e, c, n in zip(event_observed, coverage_sufficient, near_tie)
    ], dtype=object)
    if not (reasons.to_numpy() == expected_reasons).all():
        raise ValueError('Stage5 rating-supervision exclusion reason is inconsistent')
    projected_ids = df['rating_supervision_projected_candidate_id'].astype(str)
    if not (projected_ids == df['nearest_P50_candidate_id'].astype(str)).all():
        raise ValueError('Stage5 projected candidate diagnostic differs from nearest P50')

    supervised = df.loc[available]
    if not (supervised['candidate_id'].astype(str) == supervised['nearest_P50_candidate_id'].astype(str)).all():
        raise ValueError('Stage5 rating supervision is attached to a non-nearest P50 candidate')
    if not (supervised['candidate_id'].astype(str) == supervised['rating_supervision_target_candidate_id'].astype(str)).all():
        raise ValueError('Stage5 rating-supervision target provenance mismatch')
    if not (
        supervised['rating_supervision_source'].astype(str)
        == 'projected_observed_rating_event_to_nearest_P50_candidate'
    ).all():
        raise ValueError('Stage5 rating-supervision source is not projected factual supervision')
    targets = df['rating_supervision_target_candidate_id'].fillna('').astype(str)
    if (targets[~available] != '').any():
        raise ValueError('Stage5 rating-supervision target must be empty off the supervised row')
    for reason, source in {
        'insufficient_observed_action_coverage': 'excluded_insufficient_observed_action_coverage',
        'near_tie_projection': 'excluded_near_tie_projection',
    }.items():
        selected = reasons == reason
        if not (df.loc[selected, 'rating_supervision_source'].astype(str) == source).all():
            raise ValueError(f'Stage5 {reason} rows must be marked {source}')
    if not (
        df.loc[~event_observed, 'rating_supervision_source'].astype(str)
        == 'no_new_rating_event_tplus1_no_projected_supervision'
    ).all():
        raise ValueError('Stage5 no-event groups have invalid rating-supervision provenance')
    values = pd.to_numeric(df['rating_supervision_value'], errors='coerce')
    contributions = pd.to_numeric(df['rating_supervision_numeric_contribution'], errors='coerce')
    if values[available].isna().any() or values[~available].notna().any():
        raise ValueError('Stage5 rating_supervision_value availability/NaN contract failed')
    if not np.allclose(values[available], contributions[available], atol=0.0, rtol=0.0):
        raise ValueError('Stage5 projected rating contribution differs from the factual rating value')
    if not np.allclose(contributions[~available], 0.0, atol=0.0, rtol=0.0):
        raise ValueError('Stage5 non-nearest rating contribution must be arithmetic zero')
    def constant_weight(column: str) -> float:
        values = pd.to_numeric(df[column], errors='coerce')
        if values.isna().any() or values.nunique(dropna=False) != 1:
            raise ValueError(f'Stage5 reward weight column must be finite and constant: {column}')
        return float(values.iloc[0])

    component_total = compose_reward(
        base_component=contributions.fillna(0.0),
        sector_phi_component=pd.to_numeric(df['delta_phi_clipped'], errors='coerce').fillna(0.0),
        sector_phi_lambda=constant_weight('lambda_phi'),
        merton_component=pd.to_numeric(df['reward_merton_norm'], errors='coerce').fillna(0.0),
        fcff_component=pd.to_numeric(df['reward_fcff_norm'], errors='coerce').fillna(0.0),
        profitability_component=pd.to_numeric(df['reward_profitability_norm'], errors='coerce').fillna(0.0),
        liquidity_component=pd.to_numeric(df['reward_liquidity_norm'], errors='coerce').fillna(0.0),
        merton_lambda=constant_weight('lambda_merton'),
        fcff_lambda=constant_weight('lambda_fcff'),
        profitability_lambda=constant_weight('lambda_profitability'),
        liquidity_lambda=constant_weight('lambda_liquidity'),
    )
    reward_total = pd.to_numeric(df['reward_total_raw'], errors='coerce')
    if not np.allclose(
        reward_total.to_numpy(dtype=float), component_total.to_numpy(dtype=float),
        atol=1e-12, rtol=0.0, equal_nan=False,
    ):
        raise ValueError('Stage5 reward_total_raw does not include projected rating supervision exactly once')
    reward_mean = pd.to_numeric(df['reward_mean_train'], errors='coerce')
    reward_std = pd.to_numeric(df['reward_std_train'], errors='coerce')
    if reward_mean.isna().any() or reward_std.isna().any() or (reward_std <= 0.0).any():
        raise ValueError('Stage5 reward standardization statistics are missing or invalid')
    expected_reward_train = (reward_total - reward_mean) / reward_std
    observed_reward_train = pd.to_numeric(df['reward_train'], errors='coerce')
    if not np.allclose(
        observed_reward_train.to_numpy(dtype=float), expected_reward_train.to_numpy(dtype=float),
        atol=1e-10, rtol=1e-10, equal_nan=False,
    ):
        raise ValueError('Stage5 reward_train does not preserve the projected rating supervision in reward_total_raw')
    group_diag = df.groupby('factual_transition_id', sort=False).first()
    event_groups = group_diag['factual_rating_event_observed'].fillna(False).astype(bool)
    hard_groups = group_diag['rating_supervision_hard_eligible'].fillna(False).astype(bool)
    near_tie_groups = group_diag['near_tie_flag'].fillna(False).astype(bool)
    coverage_groups = group_diag['projection_observed_coverage_sufficient'].fillna(False).astype(bool)
    out_of_library_groups = group_diag['out_of_library_flag'].fillna(False).astype(bool)
    rating_event_transition_count = int(event_groups.sum())
    hard_eligible_count = int(hard_groups.sum())
    near_tie_excluded_count = int((event_groups & coverage_groups & near_tie_groups).sum())
    before_distribution = {
        str(k): int(v) for k, v in group_diag.loc[
            event_groups, 'rating_supervision_projected_candidate_id'
        ].astype(str).value_counts().sort_index().items()
    }
    after_distribution = {
        str(k): int(v) for k, v in group_diag.loc[
            hard_groups, 'rating_supervision_projected_candidate_id'
        ].astype(str).value_counts().sort_index().items()
    }
    if int(available.sum()) != hard_eligible_count:
        raise ValueError('Stage5 projected supervision row count differs from hard-eligible transition count')
    return {
        'rating_supervision_contract_version': RATING_SUPERVISION_CONTRACT_VERSION,
        'rating_supervision_contract': 'one hard nearest-P50 factual rating contribution only for new-event, sufficient-coverage, non-near-tie transitions; out_of_library is diagnostic only; all candidate grids and auxiliary rewards are retained',
        'factual_transition_count': int(len(group_sizes)),
        'rating_event_transition_count': rating_event_transition_count,
        'rating_supervision_near_tie_excluded_count': near_tie_excluded_count,
        'rating_supervision_insufficient_coverage_excluded_count': int((event_groups & ~coverage_groups).sum()),
        'rating_supervision_out_of_library_excluded_count': 0,
        'out_of_library_used_as_rating_supervision_exclusion': False,
        'rating_supervision_hard_eligible_count': hard_eligible_count,
        'projected_rating_supervision_rows': int(available.sum()),
        'rating_supervision_retention_rate': (
            float(hard_eligible_count / rating_event_transition_count)
            if rating_event_transition_count else 0.0
        ),
        'near_tie_threshold': PROJECTION_NEAR_TIE_MARGIN,
        'near_tie_transition_count': int(near_tie_groups.sum()),
        'near_tie_rate': float(near_tie_groups.mean()) if len(near_tie_groups) else 0.0,
        'rating_event_near_tie_rate': (
            float(near_tie_excluded_count / rating_event_transition_count)
            if rating_event_transition_count else 0.0
        ),
        'rating_supervision_candidate_distribution_before_quality_gates': before_distribution,
        'rating_supervision_candidate_distribution_after_quality_gates': after_distribution,
        'out_of_library_transition_count': int(group_diag['out_of_library_flag'].fillna(False).astype(bool).sum()),
        'rating_event_out_of_library_transition_count': int((event_groups & group_diag['out_of_library_flag'].fillna(False).astype(bool)).sum()),
        'candidate_rating_outcome_observed_rows': 0,
        'candidate_rating_outcome_claimed': False,
        'projected_rating_supervision_in_reward_train': True,
    }

def expectile_loss(diff, tau):
    return torch.mean(torch.where(diff>0, tau*diff.pow(2), (1-tau)*diff.pow(2)))



def _normalise_objective(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value):
        raise ValueError(f"Cannot normalise non-finite objective value: {value!r}")
    if hi <= lo:
        return 1.0
    return float((value - lo) / (hi - lo))

