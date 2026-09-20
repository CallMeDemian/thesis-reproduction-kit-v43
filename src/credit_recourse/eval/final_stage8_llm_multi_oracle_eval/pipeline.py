from __future__ import annotations

"""Stage 8 — LLM Multi-Oracle Evaluation.

Consumes the Stage 7 LLM action table and evaluates it through the **exact
same** simulator + Alpha/Beta/Gamma scoring substrate Stage 6 uses for the
RL/candidate ladder.  No re-implementation: the simulator function
(``simulate_policy_states``) and scorer functions (``score_alpha``,
``score_beta_ordered_logit_params``, ``score_gamma_model``) are imported
directly from Stage 6.

Per the Stage 7-9 LLM contract §6 (Evaluation rule):

::

    (s_t, a_t) -> financial simulator -> ŝ_{t+1} -> Alpha/Beta/Gamma R_score
    delta_R_score_backend(policy) = R_score_backend(policy) - R_score_backend(A0)

The no-op baseline (``A0``) is **not** re-simulated.  It is read from
Stage 6's ``oracle_scores_{backend}.parquet`` so the delta_R_score baseline
is identical between RL and LLM stacks.  This guarantees Stage 9 comparisons
are paired on the same no-op-adjusted scale.

Fail-fast rules enforced (per Stage 7-9 LLM contract §9):

* Stage 8 reads the **same** backend params Stage 6 used (no re-train, no
  new artifacts).
* Stage 8 scores simulator output, not real ``s_{t+1}``.
* LLM rows are not inserted into Stage 6 policy ladder; Stage 8 writes to a
  separate directory.
* The Stage 6 substrate hashes are recorded and Stage 8 fails if Stage 6
  metadata is unavailable.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


from credit_recourse.contracts.stage_paths import stage_dir, final_root
from credit_recourse.contracts.runtime_assets import active_action_contract, active_oracle_registry, active_stage2_history, active_stage2_run_config
from typing import Any
from credit_recourse.oracle.artifact_io import load_registry, resolve_backend_artifact
from credit_recourse.eval.v43_oracle_backends import score_alpha, score_beta_ordered_logit_params, score_gamma_model
from credit_recourse.eval.v43_financial_inputs import _row_to_firm_state
from credit_recourse.eval.v43_alpha_evaluation import (
    _prepare_stage6_context_lookup, _target_operating_loss_freq_3y, _target_cap_change_count_3y, _state_to_frame_dict,
)
from credit_recourse.simulator.firm_state import FirmState
from credit_recourse.simulator.semantic_action_v4 import SemanticActionV4
from credit_recourse.simulator.business_plan import BusinessPlan, calibrate_business_plan
from credit_recourse.simulator.business_plan_interest_rate_v3 import BusinessPlanRateResolverV3, RATE_CONTRACT_VERSION
from credit_recourse.simulator.business_plan_factory import CALIBRATED_BP_RATE_V4_MODE, build_business_plan
from credit_recourse.simulator.v43_production_bundle import V43ProductionSimulationBundle, financial_record
from credit_recourse.simulator.financial_simulator import FinancialSimulator
from credit_recourse.simulator.oracle_variables import compute_oracle_variables, ORACLE_COUNTERFACTUAL_VARIABLE_CONTRACT_VERSION
from credit_recourse.rl.common.io import read_parquet_required, write_json
from credit_recourse.final_release.action_validation import CANDIDATES, DIMENSIONS
from .failure_enrichment import enrich_failure_audit


@dataclass(frozen=True)
class CurrentActionSpace:
    columns: tuple[str, ...]
    bounds: dict[str, tuple[float, float]]
    train_labels: tuple[str, ...]
    candidate_library_hash: str


def load_current_action_space(project_root: Path) -> CurrentActionSpace:
    path = active_action_contract(Path(project_root).resolve())
    contract = json.loads(path.read_text(encoding="utf-8"))
    columns = tuple(contract["action_columns"])
    if tuple(name.removeprefix("action__") for name in columns) != tuple(DIMENSIONS):
        raise ValueError("Current action contract is not the canonical eight-dimensional order")
    if tuple(contract["candidate_ids"]) != tuple(CANDIDATES):
        raise ValueError("Current action contract is not the canonical nine-action order")
    bounds = {
        name: tuple(map(float, contract["action_bounds"][name])) for name in columns
    }
    return CurrentActionSpace(
        columns=columns,
        bounds=bounds,
        train_labels=tuple(CANDIDATES),
        candidate_library_hash=_sha256_file(path),
    )

def load_canonical_business_plan_history(project_root: Path) -> tuple[pd.DataFrame, Path]:
    path = active_stage2_history(Path(project_root).resolve())
    if not path.exists():
        raise FileNotFoundError(
            f'Missing canonical BusinessPlan history substrate: {path}. '
            'Evaluation-phase-only history fallback is forbidden.'
        )
    frame = read_parquet_required(path)
    required = {'firm_id', 'fiscal_year', 'canonical_business_plan_history_row_id'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'Canonical BusinessPlan history is missing columns: {missing}')
    if frame.duplicated(['firm_id', 'fiscal_year']).any():
        raise ValueError('Canonical BusinessPlan history must be unique by (firm_id,fiscal_year)')
    return frame, path

def prepare_llm_history_lookup(base: pd.DataFrame) -> dict[str, list[tuple[int, FirmState]]]:
    """Precompute firm histories for repeated evaluator-only simulation draws.

    The canonical implementation formerly filtered the full base DataFrame once
    per action row. Repeated permutation analysis therefore performed an
    avoidable O(n^2) history lookup on every draw. This lookup preserves the
    exact same <=year, sort, and tail(3) semantics while materializing each base
    FirmState once per worker process.
    """
    if 'firm_id' not in base.columns and 'corp_code' not in base.columns:
        return {}
    fid_col = 'firm_id' if 'firm_id' in base.columns else 'corp_code'
    y_col = 'fiscal_year' if 'fiscal_year' in base.columns else ('year' if 'year' in base.columns else None)
    if y_col is None:
        return {}
    lookup: dict[str, list[tuple[int, FirmState]]] = {}
    ordered = base.copy()
    ordered['_history_year'] = pd.to_numeric(ordered[y_col], errors='coerce')
    ordered = ordered.loc[ordered['_history_year'].notna()].sort_values([fid_col, '_history_year'])
    for firm_id, group in ordered.groupby(fid_col, dropna=False, sort=False):
        records: list[tuple[int, FirmState]] = []
        for _, row in group.drop(columns=['_history_year']).iterrows():
            try:
                fs = _row_to_firm_state(row)
                records.append((int(float(row[y_col])), fs))
            except Exception:
                continue
        lookup[str(firm_id)] = records
    return lookup

def _build_llm_history(
    base: pd.DataFrame,
    firm_id: str,
    year: int,
    *,
    history_lookup: dict[str, list[tuple[int, FirmState]]] | None = None,
) -> list[FirmState]:
    lookup = history_lookup if history_lookup is not None else prepare_llm_history_lookup(base)
    records = lookup.get(str(firm_id), [])
    return [state for hist_year, state in records if int(hist_year) <= int(year)][-3:]

def _action_from_row(row: pd.Series, space) -> SemanticActionV4:
    vals = {
        c.replace("action__", ""): float(row.get(c, 0.0) or 0.0)
        for c in space.columns
    }
    return SemanticActionV4.from_mapping(vals)

def _select_llm_business_plan(mode: str, hist: list[FirmState], rating_grade=None, rate_override: float | None = None,
                                rate_resolver=None, firm_id=None, base_year: int | None = None) -> tuple[BusinessPlan, object | None]:
    if mode == CALIBRATED_BP_RATE_V4_MODE:
        return build_business_plan(mode, hist, rating_grade=rating_grade, firm_id=firm_id,
                                   base_year=base_year, rate_resolver=rate_resolver)
    if mode == 'default':
        return BusinessPlan(), None
    if mode == 'calibrated':
        return (calibrate_business_plan(hist, grade=rating_grade) if hist else BusinessPlan()), None
    if mode == 'calibrated_bp_rate_v3':
        if rate_override is None:
            raise ValueError('calibrated_bp_rate_v3 requires BusinessPlanRateResolverV3')
        return (calibrate_business_plan(hist, grade=rating_grade, interest_rate_override=rate_override) if hist else BusinessPlan(rate_short=rate_override, rate_long=rate_override, rate_bond=rate_override)), None
    raise ValueError(f'Unsupported sim_business_plan_mode: {mode}')

def _business_plan_signature(bp: BusinessPlan) -> str:
    import hashlib
    payload = json.dumps(bp.to_dict(), sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def _r136_from_state(state: FirmState) -> float:
    denom = float(getattr(state, 'current_liabilities', 0.0) or 0.0)
    if abs(denom) <= 1e-12:
        return float('nan')
    return float(getattr(state, 'payables', 0.0) or 0.0) / denom

def _selected_oracle_audit_from_state(state: FirmState, prev_state: FirmState | None = None) -> dict[str, Any]:
    """Small audit payload for current and legacy selected financial variables.

    These columns are diagnostic only.  Scoring uses ``compute_oracle_variables``
    and backend-selected variables, while formula coverage is guarded by
    ``audit_formula_registry`` before scoring.
    """
    return compute_oracle_variables(state, prev_state=prev_state)

def simulate_llm_policy_states(base: pd.DataFrame, policy_actions: pd.DataFrame, space, out: Path, *, predicted_fiscal_year: int | None = None, preserve_current_non_current_residual: bool = False, sim_business_plan_mode: str = 'default', write_outputs: bool = True, collect_audit: bool = True, history_lookup: dict[str, list[tuple[int, FirmState]]] | None = None, project_root: Path | None = None, financial_only: bool = False, fixture_mode: bool = False) -> tuple[pd.DataFrame,pd.DataFrame]:
    bundle = None
    sim = None if sim_business_plan_mode == CALIBRATED_BP_RATE_V4_MODE else FinancialSimulator(preserve_current_non_current_residual=preserve_current_non_current_residual)
    rows=[]; audits=[]
    canonical_history = None
    if history_lookup is None:
        if project_root is None:
            raise ValueError('simulate_llm_policy_states requires project_root or a canonical history_lookup; phase-eval-only history fallback is forbidden')
        canonical_history, _canonical_history_path = load_canonical_business_plan_history(project_root)
        history_lookup = prepare_llm_history_lookup(canonical_history)
    else:
        _canonical_history_path = None
    if canonical_history is None:
        if project_root is None:
            raise ValueError(
                'simulate_llm_policy_states requires project_root to load canonical nonfinancial history '
                'for target-year Oracle variables'
            )
        canonical_history, _loaded_history_path = load_canonical_business_plan_history(project_root)
        _canonical_history_path = _loaded_history_path
    context_lookup = _prepare_stage6_context_lookup(canonical_history)
    rate_resolver = None
    if sim_business_plan_mode == 'calibrated_bp_rate_v3':
        raise ValueError('calibrated_bp_rate_v3 is retired; the current production contract requires calibrated_bp_rate_v4')
    elif sim_business_plan_mode == CALIBRATED_BP_RATE_V4_MODE:
        if project_root is None:
            raise ValueError('calibrated_bp_rate_v4 requires project_root')
        bundle = V43ProductionSimulationBundle.from_project_root(project_root)
    elif fixture_mode:
        from credit_recourse.rl.common.semantic_action_v4_1_contract import simulator_from_contract
        from credit_recourse.simulator.business_plan_interest_rate_v4 import BorrowingRateV4Lineage, BorrowingRateV4Result
        action_path = active_action_contract(project_root)
        action_contract = json.loads(action_path.read_text(encoding="utf-8"))

        class FixtureRates:
            def resolve(self, firm_id, base_year):
                result = BorrowingRateV4Result(0.04, "synthetic_fixture", int(base_year), None, None, 0.04)
                return BorrowingRateV4Lineage(str(firm_id), int(base_year), result, "synthetic_fixture", "synthetic_fixture", "synthetic_fixture", "synthetic_fixture_v1")

        class FixtureCosts:
            def calibrate(self, state, history, a0_revenue):
                amount = max(0.0, float(a0_revenue) * 0.01)
                return amount, {"financial_cost_contract_version": "synthetic_fixture", "non_interest_financial_cost": amount, "decision_non_interest_financial_cost": amount, "decision_pure_interest_expense": state.pure_interest_expense, "non_interest_source_years": [int(item.year) for item in history], "non_interest_source_year_max": max((int(item.year) for item in history), default=None)}

        bundle = V43ProductionSimulationBundle(project_root, action_contract, _sha256_file(action_path), FixtureRates(), simulator_from_contract(action_contract), FixtureCosts())

    # Stage2 fixtures may persist their own row identity for lineage.  The
    # final evaluator owns the zero-based join identity, so discard only that
    # duplicate transport column before constructing the canonical index.
    base_idx=base.drop(columns=['row_id'], errors='ignore').reset_index(drop=True).reset_index().rename(columns={'index':'row_id'})
    pa=policy_actions.merge(base_idx, on='row_id', how='left', suffixes=('','__base'))
    if pa.isna().all(axis=1).any(): raise ValueError('Policy actions contain row_id not found in phase_eval base')
    for _,r in pa.iterrows():
        fs=_row_to_firm_state(r)
        hist=_build_llm_history(base, str(fs.firm_id), int(fs.year), history_lookup=history_lookup)
        if not hist or int(hist[-1].year) != int(fs.year):
            raise ValueError(f'Canonical BusinessPlan history does not end at decision year for firm={fs.firm_id} year={fs.year}')
        if bundle is not None:
            action = _action_from_row(r, space)
            result, bp, rate_lineage, action, primitives, production_lineage = bundle.simulate_action(
                fs,
                hist,
                action.to_dict(),
                action_label=str(r["candidate_id"]),
            )
        else:
            action = _action_from_row(r, space)
            rate_lineage = rate_resolver.resolve(fs.firm_id, int(fs.year)) if rate_resolver else None
            bp, _ = _select_llm_business_plan(sim_business_plan_mode, hist, rating_grade=getattr(fs, 'rating_grade', None), rate_override=None if rate_lineage is None else rate_lineage.result.blended_rate)
            primitives = None
            production_lineage = {}
            result=sim.simulate(fs, bp, action)
        if financial_only:
            if bundle is None:
                raise ValueError('Financial-only Stage6 requires V4.3')
            rows.append(financial_record(fs, str(r['candidate_id']), (result, bp, rate_lineage, action, primitives, production_lineage)))
            continue
        bp_signature = production_lineage.get('business_plan_signature', _business_plan_signature(bp))
        history_years = [int(s.year) for s in hist]
        target_year = int(fs.year) + 1
        if predicted_fiscal_year is not None and int(predicted_fiscal_year) != target_year:
            raise ValueError(f'Stage6 target-year mismatch: decision_year={fs.year} simulated={target_year} configured={predicted_fiscal_year}')
        residual_audit = result.diagnostics.get('residual_audit', {}) or {}
        r136_before = _r136_from_state(fs)
        r136_after = _r136_from_state(result.state_t1)
        selected_audit_before = _selected_oracle_audit_from_state(fs)
        selected_audit_after = _selected_oracle_audit_from_state(result.state_t1, prev_state=fs)
        sim_vars=compute_oracle_variables(result.state_t1, prev_state=fs, exogenous={
            'industry_median_rating_lag1_self_excl': r.get('industry_median_rating_lag1_self_excl'),
            'industry_avg_rating_lag1_self_excl': r.get('industry_avg_rating_lag1_self_excl'),
            'cap_change_count_3y': r.get('cap_change_count_3y'),
            'cap_change_count_3y_target': _target_cap_change_count_3y(
                context_lookup, str(fs.firm_id), int(fs.year)
            ),
            'operating_loss_freq_3y_target': _target_operating_loss_freq_3y(hist, result.state_t1),
            'financial_data_completeness': r.get('financial_data_completeness'),
            'nf_retained_earnings_negative_flag': r.get('nf_retained_earnings_negative_flag'),
            'nf_ratio_missing_rate': r.get('nf_ratio_missing_rate'),
            'ratio_missing_rate': r.get('ratio_missing_rate'),
        })
        row=_state_to_frame_dict(sim_vars, r)
        # Preserve same-run decision-year context columns required by the
        # promoted Oracle backends.  The simulator-derived variables above are
        # the scored values; these fields are immutable contextual covariates,
        # not frozen score outputs.
        for source_column, source_value in r.items():
            if source_column not in row:
                row[source_column] = source_value
        row.update({'row_id':r['row_id'],'policy':r['policy'],'candidate_id':r['candidate_id'],'sustainability':result.sustainability,'plug_used':result.plug_used,'plug_amount':result.plug_amount,
                    'mode': r.get('mode') if 'mode' in r.index else None,
                    'analysis_population': r.get('analysis_population') if 'analysis_population' in r.index else None,
                    'sim_business_plan_mode': sim_business_plan_mode, 'business_plan_rate_contract_version': (RATE_CONTRACT_VERSION if sim_business_plan_mode == 'calibrated_bp_rate_v3' and rate_lineage is not None else (rate_lineage.bp_rate_contract_id if rate_lineage is not None else None)), **({} if rate_lineage is None else rate_lineage.as_dict()), 'preserve_current_non_current_residual': bool(preserve_current_non_current_residual),
                    'business_plan_history_source': str(_canonical_history_path) if _canonical_history_path is not None else 'canonical_history_lookup_supplied_by_caller',
                    'business_plan_history_years': '|'.join(str(x) for x in history_years),
                    'business_plan_history_row_count': int(len(hist)),
                    'business_plan_signature': bp_signature, **production_lineage,
                    'oracle_variable_contract_version': ORACLE_COUNTERFACTUAL_VARIABLE_CONTRACT_VERSION,
                    'oracle_financial_target_year': target_year,
                    'oracle_frozen_context_year': int(fs.year),
                    'oracle_time_rule__log_assets': 'target_year_log1p_simulated_total_assets',
                    'oracle_time_rule__nf_log_assets': 'target_year_log1p_simulated_total_assets',
                    'oracle_time_rule__operating_loss_freq_3y': 'target_year_rolling_count_from_canonical_tminus1_t_plus_simulated_tplus1',
                    'oracle_time_rule__cap_change_count_3y': 'target_year_rolling_count_from_canonical_cumulative_t_minus_tminus2_plus_zero_counterfactual_tplus1_events',
                    'oracle_time_rule__industry_avg_rating_lag1_self_excl': 'frozen_decision_year_t_lagged_self_excluded_context_no_counterfactual_peer_rating_process',
                    'oracle_time_rule__ratio_missing_rate': 'frozen_decision_year_t_observation_quality_context_no_simulated_134_ratio_missingness_mask',
                    'simulated_total_assets': float(getattr(result.state_t1, 'total_assets', np.nan)),
                    'total_assets_before': float(getattr(fs, 'total_assets', np.nan) or 0.0),
                    'current_liabilities_before': float(getattr(fs, 'current_liabilities', np.nan) or 0.0),
                    'current_liabilities_after': float(getattr(result.state_t1, 'current_liabilities', np.nan) or 0.0),
                    'current_assets_before': float(getattr(fs, 'current_assets', np.nan) or 0.0),
                    'current_assets_after': float(getattr(result.state_t1, 'current_assets', np.nan) or 0.0),
                    'R136_before': r136_before, 'R136_after': r136_after,
                    'R133_before': selected_audit_before.get('R133'), 'R133_after': selected_audit_after.get('R133'),
                    'R182_before': selected_audit_before.get('R182'), 'R182_after': selected_audit_after.get('R182'),
                    'oracle_formula_values_before_json': json.dumps(selected_audit_before, ensure_ascii=False, default=str, sort_keys=True),
                    'oracle_formula_values_after_json': json.dumps(selected_audit_after, ensure_ascii=False, default=str, sort_keys=True),
                    'residual_negative_flag': bool(any(bool(residual_audit.get(k, False)) for k in ['other_current_assets_clipped','other_non_current_assets_clipped','other_current_liabilities_clipped','other_non_current_liabilities_clipped'])),
                    'other_current_assets_t': float(residual_audit.get('other_current_assets_raw', np.nan)),
                    'other_non_current_assets_t': float(residual_audit.get('other_non_current_assets_raw', np.nan)),
                    'other_current_liab_t': float(residual_audit.get('other_current_liabilities_raw', np.nan)),
                    'other_non_current_liab_t': float(residual_audit.get('other_non_current_liabilities_raw', np.nan))})
        for identity_key in (
            "request_id",
            "cell_id",
            "firm_key",
            "model_key",
            "generation_regime",
            "phase",
            "info",
            "information_condition",
            "budget",
            "replicate",
            "wave",
            "parent_cell_id",
            "parent_request_id",
            "reference_source",
            "reference_row_id",
            "reference_candidate_id",
            "action_layer",
            "action_application_status",
        ):
            if identity_key in r.index:
                row[identity_key] = r.get(identity_key)
        if predicted_fiscal_year is not None:
            row['predicted_fiscal_year'] = int(predicted_fiscal_year)
        if bundle is not None:
            row.update(financial_record(fs, str(r['candidate_id']), (result, bp, rate_lineage, action, primitives, production_lineage)))
        rows.append(row)
        if collect_audit:
            # Per-action effect audit: record the financial account/ratio affected by each nonzero action.
            # This is not a proxy evaluator; it is an audit trail around the full simulator result.
            audit_targets = {
                "action__growth_capex_reduction_pct": [
                    ("ppe", getattr(fs, "ppe", np.nan), getattr(result.state_t1, "ppe", np.nan))
                ],
                "action__deleveraging_total_debt_pct": [
                    ("short_term_debt", getattr(fs, "short_term_debt", np.nan), getattr(result.state_t1, "short_term_debt", np.nan)),
                    ("long_term_debt", getattr(fs, "long_term_debt", np.nan), getattr(result.state_t1, "long_term_debt", np.nan)),
                ],
                "action__refinancing_short_debt_pct": [
                    ("short_term_debt", getattr(fs, "short_term_debt", np.nan), getattr(result.state_t1, "short_term_debt", np.nan)),
                    ("long_term_debt", getattr(fs, "long_term_debt", np.nan), getattr(result.state_t1, "long_term_debt", np.nan)),
                ],
                "action__inv_turnover_chg": [
                    ("inventory", getattr(fs, "inventory", np.nan), getattr(result.state_t1, "inventory", np.nan))
                ],
                "action__ar_turnover_chg": [
                    ("receivables", getattr(fs, "receivables", np.nan), getattr(result.state_t1, "receivables", np.nan))
                ],
                "action__ap_turnover_chg": [
                    ("payables", getattr(fs, "payables", np.nan), getattr(result.state_t1, "payables", np.nan))
                ],
                "action__cogs_ratio_chg": [
                    ("cogs", getattr(fs, "cogs", np.nan), getattr(result.state_t1, "cogs", np.nan))
                ],
                "action__sga_ratio_chg": [
                    ("sga", getattr(fs, "sga", np.nan), getattr(result.state_t1, "sga", np.nan))
                ],
            }
            audit_columns = list(space.columns)
            for ac in audit_columns:
                val = float(r.get(ac, 0.0) or 0.0)
                if val==0.0: continue
                raw_name=ac.replace('action__','')
                for target,before,after in audit_targets.get(ac, [('financial_statement_simulator', np.nan, np.nan)]):
                    try:
                        delta=float(after)-float(before)
                    except Exception:
                        delta=np.nan
                    audits.append({'firm_id':fs.firm_id,'year':fs.year,'row_id':r['row_id'],'policy':r['policy'],'candidate_id':r['candidate_id'],'mode':r.get('mode') if 'mode' in r.index else None,'action_dim':ac,'action_value':val,'target_variable':target,'affected_account_or_ratio':target,'before_value':before,'after_value':after,'delta_value':delta,'clipped':False,'clip_reason':'strict_no_clipping_contract','adapter_rule_id':'full_financial_statement_simulator','oracle_backend':'pending','score_before':np.nan,'score_after':np.nan,'score_delta':np.nan,'simulator_preflight_status':'ok','sustainability':result.sustainability,'fallback_metrics':json.dumps(result.diagnostics.get('fallback_metrics',{}),ensure_ascii=False),'plug_used':result.plug_used,'accounting_check':json.dumps(result.accounting_check,ensure_ascii=False),'sim_business_plan_mode':sim_business_plan_mode,'preserve_current_non_current_residual':bool(preserve_current_non_current_residual),'total_assets_before':float(getattr(fs,'total_assets',np.nan) or 0.0),'current_liabilities_before':float(getattr(fs,'current_liabilities',np.nan) or 0.0),'current_liabilities_after':float(getattr(result.state_t1,'current_liabilities',np.nan) or 0.0),'current_assets_before':float(getattr(fs,'current_assets',np.nan) or 0.0),'current_assets_after':float(getattr(result.state_t1,'current_assets',np.nan) or 0.0),'R136_before':r136_before,'R136_after':r136_after,'R133_before':selected_audit_before.get('R133'),'R133_after':selected_audit_after.get('R133'),'R182_before':selected_audit_before.get('R182'),'R182_after':selected_audit_after.get('R182'),'residual_negative_flag':bool(any(bool(residual_audit.get(k, False)) for k in ['other_current_assets_clipped','other_non_current_assets_clipped','other_current_liabilities_clipped','other_non_current_liabilities_clipped']))})
    state=pd.DataFrame(rows); audit=pd.DataFrame(audits)
    if write_outputs:
        state.to_parquet(out/'simulated_oracle_input_frame.parquet', index=False)
        audit.to_parquet(out/'action_effect_audit.parquet', index=False)
    return state,audit

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_stage6_payoff_surface(project_root: Path, *, fixture_mode: bool = False) -> tuple[pd.DataFrame, Path]:
    stage6 = stage_dir(project_root, "stage6")
    pointer_path = stage6 / "CURRENT_RELEASE.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(f"Missing current Stage6 release pointer: {pointer_path}")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    relative = pointer.get("payoff_surface_path")
    if not relative:
        relative = str(Path(pointer["artifact_path"]) / "firm_action_oracle_payoffs.parquet")
    path = Path(project_root).resolve() / relative
    if not path.is_file():
        raise FileNotFoundError(
            "Current Stage6 release lacks the canonical 575x9 Oracle payoff surface: "
            f"{path}"
        )
    frame = pd.read_parquet(path)
    required = {"row_id", "firm_id", "action", "Alpha", "Beta", "Gamma"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Stage6 payoff surface missing columns: {sorted(missing)}")
    expected_rows = 5_175 if not fixture_mode else len(frame)
    if len(frame) != expected_rows or len(frame) % 9 != 0 or frame[["row_id", "action"]].duplicated().any():
        raise ValueError("Stage6 payoff surface must be a unique cohort x 9 grid")
    if set(frame["action"].astype(str)) != set(CANDIDATES):
        raise ValueError("Stage6 payoff surface action universe drift")
    return frame, path


def _load_stage6_noop_scores(
    project_root: Path,
    *,
    fixture_mode: bool = False,
) -> dict[str, pd.DataFrame]:
    surface, _ = _load_stage6_payoff_surface(project_root, fixture_mode=fixture_mode)
    noop = surface.loc[surface["action"].astype(str).eq("A0")].copy()
    expected_firms = 575 if not fixture_mode else len(surface) // 9
    if len(noop) != expected_firms or noop["row_id"].nunique() != expected_firms:
        raise ValueError("Stage6 payoff surface must contain one A0 row per firm")
    out: dict[str, pd.DataFrame] = {}
    for backend, source in (("alpha", "Alpha"), ("beta", "Beta"), ("gamma", "Gamma")):
        out[backend] = noop[["row_id", source]].rename(
            columns={source: f"noop_R_score_{backend}"}
        )
    return out

def _stage6_substrate_hashes(project_root: Path) -> dict:
    """Record the Stage 6 substrate identity (backend registry + each backend's
    params artifact hash) so Stage 8 metadata can prove the same substrate
    was used."""
    final = final_root(project_root)
    reg_path = active_oracle_registry(project_root)
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing oracle backend registry: {reg_path}")
    reg = load_registry(reg_path)
    backends = reg.get("backends", {})
    hashes: dict[str, str] = {
        "oracle_backend_registry_sha256": _sha256_file(reg_path),
    }
    for bk in ["alpha", "beta", "gamma"]:
        bp = resolve_backend_artifact(project_root, final, backends.get(bk, {}).get("params", ""))
        if not bp.exists():
            raise FileNotFoundError(f"Missing {bk} params: {bp}")
        hashes[f"{bk}_params_sha256"] = _sha256_file(bp)
        model_val = backends.get(bk, {}).get("model", "")
        if model_val:
            mp = resolve_backend_artifact(project_root, final, model_val)
            if mp.exists():
                hashes[f"{bk}_model_sha256"] = _sha256_file(mp)
    return hashes




def _load_stage6_simulator_identity(project_root: Path, *, fixture_mode: bool = False) -> dict[str, object]:
    root = Path(project_root).resolve()
    action_path = active_action_contract(root)
    run_config_path = active_stage2_run_config(root)
    action = json.loads(action_path.read_text(encoding="utf-8"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    target_year = int(run_config["temporal"]["evaluation_rollout_year"])
    if not fixture_mode and target_year != 2025:
        raise ValueError("Current Stage8 contract requires 2025 simulated target year")
    return {
        "stage2_run_config_path": str(run_config_path),
        "stage2_run_config_sha256": _sha256_file(run_config_path),
        "sim_business_plan_mode": "default" if fixture_mode else CALIBRATED_BP_RATE_V4_MODE,
        "preserve_current_non_current_residual": False if fixture_mode else True,
        "predicted_fiscal_year": None if fixture_mode else target_year,
        "semantic_contract_version": "V4.3_FINAL_20260912",
        "candidate_runtime_hash": _sha256_file(action_path),
        "action_semantic_contract_version": action[
            "simulator_action_semantic_contract_version"
        ],
    }

def _validate_llm_action_table(
    df: pd.DataFrame,
    space: CurrentActionSpace,
    allowed_conditions: set[str],
) -> None:
    """Fail closed on any post-parse action mutation or Plan-3 identity drift."""
    required = {
        "request_id",
        "cell_id",
        "row_id",
        "firm_key",
        "model_key",
        "policy",
        "mode",
        "information_condition",
        "budget",
        "replicate",
        "candidate_id",
        "reference_source",
        "reference_row_id",
        "reference_candidate_id",
        "analysis_population",
        "action_layer",
        "action_application_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LLM action table missing required columns: {sorted(missing)}")

    action_cols = [c for c in df.columns if c.startswith("action__")]
    if action_cols != list(space.columns):
        raise ValueError(
            "LLM action column order differs from the current eight-dimensional "
            f"contract: got {action_cols}, expected {list(space.columns)}"
        )
    if df["request_id"].duplicated().any():
        raise ValueError("Each analysis population must contain at most one row per request")
    if not set(df["policy"].astype(str)).issubset(allowed_conditions):
        bad = sorted(set(df["policy"].astype(str)) - allowed_conditions)
        raise ValueError(f"Forbidden Plan-3 condition codes: {bad}")
    if not set(df["mode"].astype(str)).issubset({"free8", "candidate9"}):
        raise ValueError("Unknown action mode in Stage8 input")
    if not set(df["budget"].astype(str)).issubset({"B1", "BINF"}):
        raise ValueError("Unknown action budget in Stage8 input")
    if not set(df["candidate_id"].astype(str)).issubset(set(CANDIDATES) | {"FREE8"}):
        raise ValueError("Stage8 input contains an unknown current-contract candidate label")

    c6e = df[df["policy"].astype(str).eq("C6-E")]
    if not c6e.empty:
        if not c6e["reference_source"].astype(str).eq("C3-E").all():
            raise ValueError("C6-E rows must carry reference_source='C3-E'")
        if not (
            pd.to_numeric(c6e["reference_row_id"], errors="coerce")
            .astype("Int64")
            .eq(pd.to_numeric(c6e["row_id"], errors="coerce").astype("Int64"))
            .all()
        ):
            raise ValueError("C6-E must reference the same firm's frozen C3-E action")
    c6ex = df[df["policy"].astype(str).eq("C6-EX")]
    if not c6ex.empty:
        if not c6ex["reference_source"].astype(str).eq("C3-EX").all():
            raise ValueError("C6-EX rows must carry reference_source='C3-EX'")
        if (
            pd.to_numeric(c6ex["reference_row_id"], errors="coerce")
            .astype("Int64")
            .eq(pd.to_numeric(c6ex["row_id"], errors="coerce").astype("Int64"))
            .any()
        ):
            raise ValueError("C6-EX must use the frozen non-self donor mapping")
    other = df[~df["policy"].astype(str).isin({"C6-E", "C6-EX"})]
    if not other.empty and not other["reference_source"].astype(str).eq("none").all():
        raise ValueError("Only C6-E/C6-EX may carry an action reference")

    numeric = df.loc[:, list(space.columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Stage8 refuses non-finite action values")
    for col in space.columns:
        lo, hi = space.bounds[col]
        values = numeric[col]
        bad = (values < lo - 1e-9) | (values > hi + 1e-9)
        if bad.any():
            sample = df.loc[bad, ["request_id", "row_id", "policy", col]].head(5)
            raise ValueError(
                f"Stage8 refuses out-of-bounds action {col} in lieu of clipping:\n{sample}"
            )
    dl = numeric["action__deleveraging_total_debt_pct"]
    rf = numeric["action__refinancing_short_debt_pct"]
    if ((dl > 1e-9) & (rf > 1e-9)).any():
        raise ValueError("Stage8 refuses DL/RF mutual-exclusion violations")

    scales = {
        f"action__{key}": float(value)
        for key, value in json.loads(
            active_action_contract(Path(__file__).resolve().parents[4]).read_text(encoding="utf-8")
        )["normalization_scale_by_dimension"].items()
    }
    b1 = df["budget"].astype(str).eq("B1")
    intensity = sum(numeric[col].abs() / scales[col] for col in space.columns)
    if (intensity.loc[b1] > 1.0 + 1e-9).any():
        raise ValueError("Stage8 refuses B1 intensity violations; no rescaling is allowed")

def _per_policy_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Compact policy-level summary across backends."""
    rows = []
    group_cols = ["analysis_population", "policy"] if "analysis_population" in merged.columns else ["policy"]
    for keys, g in merged.groupby(group_cols):
        if len(group_cols) == 2:
            population, pol = keys
        else:
            population, pol = "legacy_unspecified", keys
        rec: dict = {"analysis_population": str(population), "policy": str(pol), "n_rows": int(len(g))}
        for bk in ["alpha", "beta", "gamma"]:
            col = f"delta_R_score_{bk}"
            if col not in g.columns:
                continue
            s = pd.to_numeric(g[col], errors="coerce")
            rec[f"mean_delta_R_score_{bk}"] = float(s.mean()) if s.notna().any() else float("nan")
            rec[f"median_delta_R_score_{bk}"] = float(s.median()) if s.notna().any() else float("nan")
            rec[f"std_delta_R_score_{bk}"] = float(s.std()) if s.notna().any() else float("nan")
            rec[f"valid_fraction_{bk}"] = float(s.notna().mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def run_stage8(
    *,
    project_root: Path,
    fixture_mode: bool = False,
) -> dict:
    """Execute Stage 8 end-to-end and write all outputs."""
    project_root = Path(project_root).resolve()
    final = final_root(project_root)
    s7 = stage_dir(project_root, "stage7")
    out = stage_dir(project_root, "stage8")
    out.mkdir(parents=True, exist_ok=True)

    # --- current V4.3 action contract and Stage7 lineage ---
    action_contract_path = active_action_contract(project_root)
    action_contract_hash = _sha256_file(action_contract_path)
    space = load_current_action_space(project_root)
    base_hashes = {
        "candidate_template_hash": action_contract_hash,
        "candidate_library_hash": action_contract_hash,
        "candidate_library_path": str(action_contract_path),
        "final_action_contract_hash": action_contract_hash,
    }
    hashes = dict(base_hashes)
    selected_candidate_library_path = action_contract_path

    action_table_path = s7 / "llm_stage7_action_table_per_protocol.parquet"
    action_table_itt_path = s7 / "llm_stage7_action_table_itt.parquet"
    if not action_table_path.exists() or not action_table_itt_path.exists():
        raise FileNotFoundError(
            "Stage8 requires the canonical accepted and ITT Stage7 tables"
        )
    stage7_meta_path = s7 / "metadata.json"
    if not stage7_meta_path.exists():
        raise FileNotFoundError(f"Missing Stage7 metadata: {stage7_meta_path}")
    stage7_meta = json.loads(stage7_meta_path.read_text(encoding="utf-8"))
    if stage7_meta.get("scientific_contract_version") != "V4.3_FINAL_20260912":
        raise ValueError("Stage8 refuses a Stage7 table outside the final V4.3 contract")
    if stage7_meta.get("final_action_contract_hash") != action_contract_hash:
        raise ValueError("Stage8/Stage7 action-contract hash mismatch")
    q_raw = None
    action_table_pp = read_parquet_required(action_table_path)
    action_table_itt = read_parquet_required(action_table_itt_path)
    _validate_llm_action_table(
        action_table_pp, space, allowed_conditions={"C4", "C5", "C4R", "C6-E", "C6-EX"}
    )
    _validate_llm_action_table(
        action_table_itt, space, allowed_conditions={"C4", "C5", "C4R", "C6-E", "C6-EX"}
    )
    if not action_table_pp.get("model_response_usable", pd.Series(False, index=action_table_pp.index)).fillna(False).astype(bool).all():
        raise ValueError("Per-protocol Stage7 action table contains unusable responses")
    if len(action_table_pp) and set(action_table_pp.get("analysis_population", pd.Series(dtype=str)).astype(str).unique()) != {"per_protocol"}:
        raise ValueError("Per-protocol Stage7 action table has invalid analysis_population")
    if set(action_table_itt.get("analysis_population", pd.Series(dtype=str)).astype(str).unique()) != {"itt"}:
        raise ValueError("ITT Stage7 action table has invalid analysis_population")
    key_cols = ["request_id"]
    pp_keys = set(map(tuple, action_table_pp[key_cols].astype(str).to_numpy().tolist()))
    itt_keys = set(map(tuple, action_table_itt[key_cols].astype(str).to_numpy().tolist()))
    if not pp_keys.issubset(itt_keys):
        raise ValueError("Per-protocol request keys are not a subset of the ITT request universe")
    fallback = action_table_itt["itt_noop_fallback_applied"].fillna(False).astype(bool)
    if not (action_table_itt.loc[fallback, "candidate_id"].astype(str) == "A0").all():
        raise ValueError("ITT unusable responses must use the pre-specified A0 fallback")
    if not np.allclose(action_table_itt.loc[fallback, space.columns].astype(float).to_numpy(), 0.0, atol=0.0, rtol=0.0):
        raise ValueError("ITT A0 fallback must be the exact zero action vector")
    action_table = pd.concat([action_table_itt, action_table_pp], ignore_index=True, sort=False)

    # --- shared state base from Stage 2 (same as Stage 6) ---
    base_path = stage_dir(project_root, "stage2") / "phase_eval_candidate.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing Stage 2 phase_eval_candidate.parquet at {base_path}.")
    base = read_parquet_required(base_path)

    # --- backend registry (identical to Stage 6) ---
    reg_path = active_oracle_registry(project_root)
    reg = load_registry(reg_path)
    backends = reg.get("backends", {})
    if reg.get("final_result_allowed") is not True or (not fixture_mode and reg.get("status") != "generated_by_stage1_oracle_development_verified"):
        raise ValueError(
            "Stage 8 refuses to score: Oracle backend registry is not "
            "final/generated_by_stage1_oracle_development_verified."
        )

    # --- run the Stage 6 simulator on the LLM actions ---
    # Contract v4: substrate identity is non-negotiable.  The settings below
    # are read from Stage 6 metadata and passed explicitly; using the Stage 6
    # function defaults would silently score LLM actions on a different
    # simulator substrate than the current A0 baseline consumed below.
    simulator_identity = _load_stage6_simulator_identity(project_root, fixture_mode=fixture_mode)
    sim_state, audit = simulate_llm_policy_states(
        base,
        action_table,
        space,
        out,
        predicted_fiscal_year=(None if simulator_identity["predicted_fiscal_year"] is None else int(simulator_identity["predicted_fiscal_year"])),
        preserve_current_non_current_residual=bool(
            simulator_identity["preserve_current_non_current_residual"]
        ),
        sim_business_plan_mode=str(simulator_identity["sim_business_plan_mode"]),
        project_root=project_root,
        fixture_mode=fixture_mode,
    )

    if "sim_business_plan_mode" in audit.columns:
        got_modes = set(audit["sim_business_plan_mode"].astype(str).dropna().unique())
        expected_mode = str(simulator_identity["sim_business_plan_mode"])
        if got_modes and got_modes != {expected_mode}:
            raise ValueError(
                f"Stage 8 simulator audit mode mismatch: got {sorted(got_modes)}, "
                f"expected {expected_mode}"
            )
    if "preserve_current_non_current_residual" in audit.columns:
        got_preserve = set(audit["preserve_current_non_current_residual"].dropna().astype(bool).unique())
        expected_preserve = bool(simulator_identity["preserve_current_non_current_residual"])
        if got_preserve and got_preserve != {expected_preserve}:
            raise ValueError(
                f"Stage 8 simulator audit preserve flag mismatch: got {sorted(got_preserve)}, "
                f"expected {expected_preserve}"
            )

    # --- post-simulation feasibility failure coding ---
    stage7_failure_path = s7 / "llm_stage7_failure_audit.csv"
    if not stage7_failure_path.exists():
        raise FileNotFoundError(
            f"Stage 8 requires Stage 7 failure audit for taxonomy enrichment: {stage7_failure_path}"
        )
    stage7_failure_audit = pd.read_csv(stage7_failure_path)
    itt_sim_state = sim_state.loc[sim_state["analysis_population"].astype(str) == "itt"].copy()
    # The ITT state frame contains exactly one simulated row for every request,
    # including A0 fallbacks. Sim-state diagnostics are sufficient here and
    # avoid duplicating action-audit rows shared with per-protocol simulations.
    enriched_failure_audit, failure_coder_manifest = enrich_failure_audit(
        stage7_failure_audit=stage7_failure_audit,
        simulated_state=itt_sim_state,
        action_effect_audit=pd.DataFrame(),
        out_dir=out,
    )

    # --- score with the Stage 6 scorers (Alpha/Beta/Gamma) ---
    # The LLM action table can contain two rows for the same (row_id, policy)
    # when both modes (candidate9, free8) are exercised in
    # one Stage 7 run.  Stage 8 keeps ``mode`` in the score-frame primary
    # key so the per-mode rows do not silently fan out under the three-way
    # outer merge.  Stage 9's revision-metric and comparison logic already
    # carries ``mode`` so this preserves the row-level pairing end to end.
    score_key_cols = [
        "request_id",
        "cell_id",
        "row_id",
        "firm_key",
        "model_key",
        "generation_regime",
        "phase",
        "policy",
        "mode",
        "info",
        "information_condition",
        "budget",
        "replicate",
        "wave",
        "parent_cell_id",
        "parent_request_id",
        "analysis_population",
        "candidate_id",
        "action_layer",
        "action_application_status",
        "reference_source",
        "reference_row_id",
        "reference_candidate_id",
    ]
    missing_score_keys = [key for key in score_key_cols if key not in action_table]
    if missing_score_keys:
        raise ValueError(f"Stage8 action table lacks identity keys: {missing_score_keys}")
    all_scores = []
    for backend in ["alpha", "beta", "gamma"]:
        b = backends[backend]
        params = resolve_backend_artifact(project_root, final, b.get("params", ""))
        if not params.exists():
            raise FileNotFoundError(f"Missing {backend} params: {params}")
        if backend == "alpha":
            score = score_alpha(sim_state, params)
            col = "R_score_alpha"
        elif backend == "beta":
            score = score_beta_ordered_logit_params(sim_state, params)
            col = "R_score_beta"
        else:
            model = resolve_backend_artifact(project_root, final, b.get("model", ""))
            if not model.exists():
                raise FileNotFoundError(f"Missing gamma model artifact: {model}")
            score = score_gamma_model(sim_state, params, model)
            col = "R_score_gamma"
        scored = action_table[score_key_cols + list(space.columns)].copy()
        scored[col] = score.to_numpy()
        scored.to_parquet(out / f"llm_oracle_scores_{backend}_all_populations.parquet", index=False)
        scored.loc[scored["analysis_population"].astype(str) == "per_protocol"].to_parquet(
            out / f"llm_oracle_scores_{backend}.parquet", index=False
        )
        scored.loc[scored["analysis_population"].astype(str) == "itt"].to_parquet(
            out / f"llm_oracle_scores_{backend}_itt.parquet", index=False
        )
        all_scores.append(scored[score_key_cols + [col]])

    merged = all_scores[0]
    for s in all_scores[1:]:
        merged = merged.merge(s, on=score_key_cols, how="outer")

    # --- pull current A0 scores from Stage 6 and compute delta_R_score ---
    noop_by_backend = _load_stage6_noop_scores(project_root, fixture_mode=fixture_mode)
    for backend in ["alpha", "beta", "gamma"]:
        merged = merged.merge(noop_by_backend[backend], on="row_id", how="left")
        merged[f"delta_R_score_{backend}"] = (
            pd.to_numeric(merged[f"R_score_{backend}"], errors="coerce")
            - pd.to_numeric(merged[f"noop_R_score_{backend}"], errors="coerce")
        )

    scores_itt = merged.loc[merged["analysis_population"].astype(str) == "itt"].copy()
    scores_pp = merged.loc[merged["analysis_population"].astype(str) == "per_protocol"].copy()
    merged.to_parquet(out / "llm_stage8_multi_oracle_scores_all_populations.parquet", index=False)
    scores_itt.to_parquet(out / "llm_stage8_multi_oracle_scores_itt.parquet", index=False)
    scores_pp.to_parquet(out / "llm_stage8_multi_oracle_scores_per_protocol.parquet", index=False)
    # Backward-compatible Stage9 input remains the historical per-protocol
    # estimand. ITT is an additional, explicit output and never silently
    # doubles the legacy score table's request keys.
    scores_pp.to_parquet(out / "llm_stage8_multi_oracle_scores.parquet", index=False)

    summary = _per_policy_summary(merged)
    summary_itt = summary.loc[summary["analysis_population"].astype(str) == "itt"].copy()
    summary_pp = summary.loc[summary["analysis_population"].astype(str) == "per_protocol"].copy()
    summary.to_csv(out / "llm_stage8_policy_summary_all_populations.csv", index=False, encoding="utf-8-sig")
    summary_itt.to_csv(out / "llm_stage8_policy_summary_itt.csv", index=False, encoding="utf-8-sig")
    summary_pp.to_csv(out / "llm_stage8_policy_summary_per_protocol.csv", index=False, encoding="utf-8-sig")
    summary_pp.to_csv(out / "llm_stage8_policy_summary.csv", index=False, encoding="utf-8-sig")

    # --- metadata ---
    substrate_hashes = _stage6_substrate_hashes(project_root)
    if simulator_identity["semantic_contract_version"] != "V4.3_FINAL_20260912":
        raise ValueError("Stage8 refuses a non-final V4.3 simulator identity")
    if simulator_identity["candidate_runtime_hash"] != space.candidate_library_hash:
        raise ValueError("Stage8 current action-contract hash mismatch")
    expected_action_semantics = json.loads(
        selected_candidate_library_path.read_text(encoding="utf-8")
    )["simulator_action_semantic_contract_version"]
    if simulator_identity["action_semantic_contract_version"] != expected_action_semantics:
        raise ValueError("Stage8 Simulator/action semantic contract mismatch")
    meta = {
        "stage": "final_stage8_llm_multi_oracle_eval",
        "status": "PASS",
        "created_utc": _now(),
        "semantic_contract_version": "V4.3_FINAL_20260912",
        "candidate_template_hash": base_hashes["candidate_template_hash"],
        "candidate_runtime_hash": space.candidate_library_hash,
        "candidate_library_hash": space.candidate_library_hash,
        "candidate_library_path": str(selected_candidate_library_path),
        "candidate_action_values_source": "strict accepted/ITT action vectors from final Plan-3 parser",
        "candidate_library_quantile": None,
        "selected_action_contract_hash": space.candidate_library_hash,
        "selected_action_contract_path": str(selected_candidate_library_path),
        "base_candidate_library_hash": base_hashes["candidate_library_hash"],
        "base_candidate_library_path": base_hashes["candidate_library_path"],
        "final_action_contract_hash": base_hashes["final_action_contract_hash"],
        "stage7_metadata_consumed": str(stage7_meta_path),
        "stage7_backend_is_live": bool(stage7_meta.get("backend_is_live")),
        "final_paper_run_allowed": bool(stage7_meta.get("final_paper_run_allowed")),
        "stage6_substrate_hashes": substrate_hashes,
        "stage6_simulator_identity": simulator_identity,
        "sim_business_plan_mode": simulator_identity["sim_business_plan_mode"],
        "preserve_current_non_current_residual": simulator_identity["preserve_current_non_current_residual"],
        "predicted_fiscal_year": simulator_identity["predicted_fiscal_year"],
        "no_separate_oracle_used": True,
        "scored_via_simulator_only": True,
        "delta_R_score_baseline_source": "current Stage6 575x9 payoff surface (A0 rows)",
        "row_count_evaluated": int(merged["row_id"].nunique()),
        "analysis_population_contract": {
            "itt_rows": int((merged["analysis_population"].astype(str) == "itt").sum()),
            "per_protocol_rows": int((merged["analysis_population"].astype(str) == "per_protocol").sum()),
            "itt_request_keys": int(len(itt_keys)),
            "per_protocol_request_keys": int(len(pp_keys)),
            "itt_noop_fallback_rows": int(fallback.sum()),
            "itt_fallback_rule": "pre_specified_A0_for_confirmed_terminal_unusable_response",
        },
        "policies_evaluated": sorted(set(merged["policy"].astype(str).unique())),
        "no_clipping": True,
        "no_projection": True,
        "no_rescale": True,
        "outputs": {
            "multi_oracle_scores": "llm_stage8_multi_oracle_scores.parquet (per-protocol explicit alias)",
            "multi_oracle_scores_all_populations": "llm_stage8_multi_oracle_scores_all_populations.parquet",
            "multi_oracle_scores_itt": "llm_stage8_multi_oracle_scores_itt.parquet",
            "multi_oracle_scores_per_protocol": "llm_stage8_multi_oracle_scores_per_protocol.parquet",
            "policy_summary": "llm_stage8_policy_summary.csv (per-protocol explicit alias)",
            "policy_summary_all_populations": "llm_stage8_policy_summary_all_populations.csv",
            "policy_summary_itt": "llm_stage8_policy_summary_itt.csv",
            "policy_summary_per_protocol": "llm_stage8_policy_summary_per_protocol.csv",
            "simulated_oracle_input_frame": "simulated_oracle_input_frame.parquet",
            "action_effect_audit": "action_effect_audit.parquet",
            "failure_audit_enriched": "llm_stage8_failure_audit_enriched.csv",
            "failure_coder_manifest": "failure_coder_manifest.json",
        },
        "failure_coder_manifest": failure_coder_manifest,
        "failure_audit_enriched_rows": int(len(enriched_failure_audit)),
    }
    write_json(out / "metadata.json", meta)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 8 — LLM Multi-Oracle Evaluation")
    ap.add_argument("--project-root", required=True)
    args = ap.parse_args(argv)
    meta = run_stage8(project_root=Path(args.project_root))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

