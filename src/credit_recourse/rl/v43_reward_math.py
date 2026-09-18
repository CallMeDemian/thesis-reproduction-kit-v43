"""Preserved financial reward math, isolated from legacy stage/control imports."""

from __future__ import annotations

from pathlib import Path

from datetime import datetime, timezone

import json, re

import numpy as np

import pandas as pd

from credit_recourse.rl.common.reward_contract import (EXPLICIT_COMPONENT_COLUMNS, FCFF_CLIP_BOUNDS, LIQUIDITY_CLIP_BOUNDS, MERTON_CLIP_BOUNDS, PROFITABILITY_CLIP_BOUNDS, compose_reward, fit_robust_p95_abs, normalize_reward_component)

PHI_COMPONENTS=["derived__roa_proxy","derived__operating_margin","derived__cogs_to_revenue","derived__sga_to_revenue","derived__financial_cost_to_revenue","derived__debt_to_assets"]

LOWER_GOOD={"derived__cogs_to_revenue","derived__sga_to_revenue","derived__financial_cost_to_revenue","derived__debt_to_assets"}

REQ_REWARD=["reward_raw_notch","reward_raw","phi_t","phi_tplusH","delta_phi","delta_phi_clipped","lambda_phi","reward_aux_phi","reward_total_raw","reward_mean_train","reward_std_train","reward_train","reward_original","reward"]

CPLD_STATE_COLUMN = "balance_sheet__[U01A811027400]      유동성장기부채(*)(IFRS)(천원)"

MERTON_AUX_COMPONENTS = [
    "sim__total_assets", "sim__short_term_debt", CPLD_STATE_COLUMN,
    "sim__long_term_debt", "sim__bonds",
]

FCFF_AUX_COMPONENTS = ["sim__total_assets", "sim__operating_cf", "sim__capex"]

LIQUIDITY_AUX_COMPONENTS = ["sim__cash", "sim__short_term_investments", "sim__total_assets"]

AUX_REWARD_COLUMNS = [
    "merton_default_point_t", "merton_default_point_tplusH", "merton_badness_t", "merton_badness_tplusH",
    "delta_merton_badness", "delta_merton_badness_scaled", "lambda_merton", "reward_aux_merton",
    "fcff_capacity_t", "fcff_capacity_tplusH", "delta_fcff_capacity", "delta_fcff_capacity_scaled",
    "lambda_fcff", "reward_aux_fcff",
    "liquid_capacity_t", "liquid_capacity_tplusH", "delta_liquid_capacity",
    "delta_liquid_capacity_scaled", "lambda_liquidity", "reward_aux_liquidity",
    "profitability_raw", "profitability_normalized", "lambda_profitability", "reward_aux_profitability",
    *EXPLICIT_COMPONENT_COLUMNS,
]

KOREAN_PHI_ALIASES={
 "derived__roa_proxy":["총자본순이익률(IFRS)","ROA","roa_proxy"],
 "derived__operating_margin":["매출액정상영업이익률(IFRS)","영업이익률","operating_margin"],
 "derived__cogs_to_revenue":["매출원가 대 매출액비율(IFRS)","cogs_to_revenue"],
 "derived__sga_to_revenue":["영업비용 대 영업수익비율(IFRS)","sga_to_revenue"],
 "derived__financial_cost_to_revenue":["금융비용부담률(IFRS)","financial_cost_to_revenue"],
 "derived__debt_to_assets":["타인자본구성비율(IFRS)","부채비율(IFRS)","debt_to_assets"],
}

PHI_ALIAS_CONTRACT_VERSION='sector_phi_semantic_aliases_no_r_code_v1'

def now(): return datetime.now(timezone.utc).isoformat()

def materialize_phi_aliases(df:pd.DataFrame, *, next_state:bool=False)->pd.DataFrame:
    forbidden={canonical:[alias for alias in aliases if re.fullmatch(r'R\d{3}',str(alias))] for canonical,aliases in KOREAN_PHI_ALIASES.items()}
    forbidden={k:v for k,v in forbidden.items() if v}
    if forbidden: raise ValueError(f'Sector-Phi aliases must not contain Oracle R-code synonyms: {forbidden}')
    out=df.copy(); prefix='next__' if next_state else ''; suffix='__next' if next_state else ''
    for canonical, aliases in KOREAN_PHI_ALIASES.items():
        target=prefix+canonical if next_state else canonical
        if target in out.columns: continue
        found=None
        for a in aliases:
            for cand in (prefix+a, a+suffix):
                if cand in out.columns: found=cand; break
            if found: break
        if found is not None: out[target]=pd.to_numeric(out[found], errors='coerce')
    return out

def require_canonical_phi_columns(df:pd.DataFrame, *, require_next:bool, context:str)->None:
    required=list(PHI_COMPONENTS)
    if require_next: required.extend([f'next__{c}' for c in PHI_COMPONENTS])
    missing=[c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f'{context}: semantic-fresh Stage2 requires canonical formula-derived sector-Phi columns; '
            f'alias/R-code recovery is forbidden. Missing: {missing}'
        )

def sector_col(df):
    for c in ['sector_7','industry_class','sector','industry']:
        if c in df.columns: return c
    return None

def _vals(s): return pd.to_numeric(s,errors='coerce').dropna().to_numpy(dtype=float)

def build_frozen_phi_cdf(train_df:pd.DataFrame, out_dir:Path)->dict:
    require_canonical_phi_columns(train_df,require_next=False,context='sector-Phi CDF reference')
    missing=[c for c in PHI_COMPONENTS if c not in train_df.columns]
    if missing: raise ValueError(f'Cannot build frozen sector-phi CDF; missing components: {missing}')
    sec=sector_col(train_df); rows=[]; ref={}
    for comp in PHI_COMPONENTS:
        ref[comp]={}; global_vals=np.sort(_vals(train_df[comp]))
        if len(global_vals)==0: raise ValueError(f'No non-null values for sector-phi component {comp}')
        ref[comp]['__GLOBAL__']=global_vals.tolist()
        group_iter=train_df.groupby(sec, dropna=False) if sec else [('__GLOBAL__', train_df)]
        for key,g in group_iter:
            vals=np.sort(_vals(g[comp])); used_fallback=False
            if len(vals)<20: vals=global_vals; used_fallback=True
            ref[comp][str(key)]=vals.tolist(); q=np.quantile(vals,[.01,.05,.10,.25,.50,.75,.90,.95,.99])
            rows.append({'component':comp,'sector':str(key),'n':int(len(vals)),'used_global_fallback':bool(used_fallback),'p01':q[0],'p05':q[1],'p10':q[2],'p25':q[3],'p50':q[4],'p75':q[5],'p90':q[6],'p95':q[7],'p99':q[8],'lower_good':comp in LOWER_GOOD})
    out_dir.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_parquet(out_dir/'sector_phi_breakpoints.parquet', index=False)
    meta={'schema_version':'sector_phi_frozen_cdf_v28','created_utc':now(),'phi_cdf_source':'eligible_phase3_iql_t_le_2022_outcome_le_2023','eval_distribution_used_for_cdf':False,'cdf_frozen':True,'components':PHI_COMPONENTS,'lower_good_components':sorted(LOWER_GOOD),'sector_column':sec,'n_reference_rows':int(len(train_df)),'reference_phases':['phase3_iql'],'excluded_from_cdf':['phase_eval']}
    (out_dir/'sector_phi_cdf_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'reference':ref,'sector_column':sec,'metadata':meta}

def _pct(values, ref_vals):
    arr=pd.to_numeric(values,errors='coerce').to_numpy(dtype=float); ref_vals=np.asarray(ref_vals,dtype=float); ref_vals=ref_vals[np.isfinite(ref_vals)]
    if len(ref_vals)==0: return pd.Series(np.full(len(values),.5),index=values.index)
    pct=np.searchsorted(ref_vals, arr, side='right')/max(len(ref_vals),1); pct[~np.isfinite(arr)]=np.nan
    return pd.Series(pct,index=values.index)

def compute_phi_with_frozen_cdf(df:pd.DataFrame, frozen:dict)->pd.Series:
    miss=[c for c in PHI_COMPONENTS if c not in df.columns]
    if miss: raise ValueError(f'Cannot compute sector-phi with frozen CDF; missing components: {miss}')
    sec=frozen.get('sector_column'); qcols=[]
    for comp in PHI_COMPONENTS:
        comp_ref=frozen['reference'][comp]; pct=pd.Series(index=df.index,dtype=float)
        if sec and sec in df.columns:
            for key,idx in df.groupby(sec,dropna=False).groups.items(): pct.loc[idx]=_pct(df.loc[idx,comp], np.asarray(comp_ref.get(str(key)) or comp_ref['__GLOBAL__'],dtype=float))
        else: pct=_pct(df[comp], np.asarray(comp_ref['__GLOBAL__'],dtype=float))
        qcols.append((1-pct if comp in LOWER_GOOD else pct).fillna(pct.median()).fillna(.5))
    q=pd.concat(qcols,axis=1); return .4*q.iloc[:,[0,1]].mean(axis=1)+.3*q.iloc[:,[2,3,4]].mean(axis=1)+.3*q.iloc[:,5]

def _required_next_columns(cols: list[str]) -> list[str]:
    return [f"next__{c}" for c in cols]

def _validate_no_r_code_aux_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    bad = [c for c in columns if str(c).startswith("R") and len(str(c)) == 4 and str(c)[1:].isdigit()]
    bad += [c for c in columns if str(c).startswith("next__R") and str(c)[6:].isdigit()]
    if bad:
        raise ValueError(f"{context}: R-code/oracle scorecard columns are forbidden in Merton/FCFF/liquidity auxiliary reward path: {bad}")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required non-oracle simulator columns for Merton/FCFF/liquidity auxiliary reward: {missing}")

def _numeric_checked(df: pd.DataFrame, col: str, context: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"{context}: missing required column {col}")
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        raise ValueError(f"{context}: duplicate column label {col!r} would make auxiliary reward ambiguous")
    x = pd.to_numeric(obj, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.isna().any():
        raise ValueError(f"{context}: non-finite values in required auxiliary reward column {col}: n={int(x.isna().sum())}")
    return x.astype(float)

def _numeric_checked_any(df: pd.DataFrame, cols: list[str], context: str) -> pd.Series:
    """Return the first finite numeric Series among equivalent source columns.

    Used for default-vs-reward_only simulator substrate aliases, e.g.
    reward_only__sim__operating_cf should be preferred when present while
    retaining the legacy sim__operating_cf fallback. This helper is intentionally
    strict: it only falls back when a candidate column is absent; if a present
    column is duplicated or contains non-finite values, fail fast rather than
    silently changing reward semantics.
    """
    missing: list[str] = []
    for col in cols:
        if col not in df.columns:
            missing.append(col)
            continue
        return _numeric_checked(df, col, context)
    raise ValueError(f"{context}: missing all equivalent required columns {cols}; absent={missing}")

def _compute_aux_reward_raw_deltas(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Compute no-oracle Merton/KMV and OCF-Capex auxiliary raw deltas.

    This helper intentionally consumes only simulator/accounting primitive columns
    already present in the Stage2 AVS256 state contract. It does not resolve R###
    scorecard aliases and does not import or call any reference oracle backend.
    """
    required = (
        list(MERTON_AUX_COMPONENTS) + _required_next_columns(MERTON_AUX_COMPONENTS)
        + list(FCFF_AUX_COMPONENTS) + _required_next_columns(FCFF_AUX_COMPONENTS)
        + list(LIQUIDITY_AUX_COMPONENTS) + _required_next_columns(LIQUIDITY_AUX_COMPONENTS)
    )
    _validate_no_r_code_aux_columns(df, required, context)
    out = df.copy()
    assets_t = _numeric_checked(out, "sim__total_assets", context)
    assets_n = _numeric_checked(out, "next__sim__total_assets", context)
    bad_assets = (assets_t <= 0) | (assets_n <= 0)
    if bad_assets.any():
        raise ValueError(f"{context}: total assets must be positive for Merton/FCFF/liquidity auxiliary reward: n={int(bad_assets.sum())}")
    short_t = _numeric_checked(out, "sim__short_term_debt", context)
    cpld_t = _numeric_checked(out, CPLD_STATE_COLUMN, context)
    long_t = _numeric_checked(out, "sim__long_term_debt", context)
    bonds_t = _numeric_checked(out, "sim__bonds", context)
    short_n = _numeric_checked(out, "next__sim__short_term_debt", context)
    cpld_n = _numeric_checked(out, f"next__{CPLD_STATE_COLUMN}", context)
    long_n = _numeric_checked(out, "next__sim__long_term_debt", context)
    bonds_n = _numeric_checked(out, "next__sim__bonds", context)
    debt_components = [short_t, cpld_t, long_t, bonds_t, short_n, cpld_n, long_n, bonds_n]
    neg_debt = np.zeros(len(out), dtype=bool)
    for x in debt_components:
        neg_debt |= (x < -1e-9).to_numpy(dtype=bool)
    if bool(neg_debt.any()):
        raise ValueError(f"{context}: debt components must be non-negative for Merton default-point proxy: n={int(neg_debt.sum())}")
    default_t = short_t + cpld_t + 0.5 * long_t + bonds_t
    default_n = short_n + cpld_n + 0.5 * long_n + bonds_n
    out["merton_default_point_t"] = default_t
    out["merton_default_point_tplusH"] = default_n
    out["merton_badness_t"] = default_t / assets_t
    out["merton_badness_tplusH"] = default_n / assets_n
    out["delta_merton_badness"] = out["merton_badness_t"] - out["merton_badness_tplusH"]

    ocf_t = _numeric_checked_any(out, ["reward_only__sim__operating_cf", "sim__operating_cf"], context)
    capex_t = _numeric_checked_any(out, ["reward_only__sim__capex", "sim__capex"], context)
    ocf_n = _numeric_checked_any(out, ["next__reward_only__sim__operating_cf", "next__sim__operating_cf"], context)
    capex_n = _numeric_checked_any(out, ["next__reward_only__sim__capex", "next__sim__capex"], context)
    out["fcff_capacity_t"] = (ocf_t - capex_t) / assets_t
    out["fcff_capacity_tplusH"] = (ocf_n - capex_n) / assets_n
    out["delta_fcff_capacity"] = out["fcff_capacity_tplusH"] - out["fcff_capacity_t"]

    cash_t = _numeric_checked(out, "sim__cash", context)
    sti_t = _numeric_checked(out, "sim__short_term_investments", context)
    cash_n = _numeric_checked(out, "next__sim__cash", context)
    sti_n = _numeric_checked(out, "next__sim__short_term_investments", context)
    liquid_components = [cash_t, sti_t, cash_n, sti_n]
    neg_liquid = np.zeros(len(out), dtype=bool)
    for x in liquid_components:
        neg_liquid |= (x < -1e-9).to_numpy(dtype=bool)
    if bool(neg_liquid.any()):
        raise ValueError(f"{context}: liquid asset components must be non-negative for liquidity-capacity auxiliary reward: n={int(neg_liquid.sum())}")
    out["liquid_capacity_t"] = (cash_t + sti_t) / assets_t
    out["liquid_capacity_tplusH"] = (cash_n + sti_n) / assets_n
    out["delta_liquid_capacity"] = out["liquid_capacity_tplusH"] - out["liquid_capacity_t"]
    return out

def apply_merton_fcff_aux_reward(df: pd.DataFrame, stats: dict, *, phase_name: str) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    lam_m = float(stats.get('lambda_merton', 0.0) or 0.0)
    lam_f = float(stats.get('lambda_fcff', 0.0) or 0.0)
    lam_l = float(stats.get('lambda_liquidity', 0.0) or 0.0)
    lam_p = float(stats.get('lambda_profitability', 0.0) or 0.0)
    # Preserve a stable schema even when disabled; do not require simulator next-state columns unless used.
    if lam_m == 0.0 and lam_f == 0.0 and lam_l == 0.0 and lam_p == 0.0:
        # Overwrite any stale pre-existing aux columns as well as filling absent ones;
        # this preserves lambda=0 no-op semantics and prevents reward-mode gates from
        # leaking previously computed auxiliary rewards into counterfactual transitions.
        for c in AUX_REWARD_COLUMNS:
            out[c] = 0.0
        return out, {'phase': phase_name, 'merton_aux_enabled': False, 'fcff_aux_enabled': False, 'profitability_aux_enabled': False, 'liquidity_aux_enabled': False}
    out = _compute_aux_reward_raw_deltas(out, context=f'{phase_name}_merton_fcff_liquidity_aux')
    m_scale = float(stats['merton_aux_scale_p95_abs'])
    f_scale = float(stats['fcff_aux_scale_p95_abs'])
    l_scale = float(stats.get('liquidity_aux_scale_p95_abs', 1.0) or 1.0)
    out['delta_merton_badness_scaled'] = normalize_reward_component(out['delta_merton_badness'], scale_parameter=m_scale, clip_bounds=MERTON_CLIP_BOUNDS, label='delta_merton_badness')
    out['delta_fcff_capacity_scaled'] = normalize_reward_component(out['delta_fcff_capacity'], scale_parameter=f_scale, clip_bounds=FCFF_CLIP_BOUNDS, label='delta_fcff_capacity')
    out['delta_liquid_capacity_scaled'] = normalize_reward_component(out['delta_liquid_capacity'], scale_parameter=l_scale, clip_bounds=LIQUIDITY_CLIP_BOUNDS, label='delta_liquid_capacity')
    if 'profitability_raw' in out.columns:
        p_scale = float(stats.get('profitability_aux_scale_p95_abs', 1.0) or 1.0)
        out['profitability_normalized'] = normalize_reward_component(out['profitability_raw'], scale_parameter=p_scale, clip_bounds=PROFITABILITY_CLIP_BOUNDS, label='profitability_raw')
    elif lam_p != 0.0:
        raise ValueError(f'{phase_name}: profitability_lambda is nonzero but profitability_raw was not produced from counterfactual financial statements')
    else:
        out['profitability_raw'] = 0.0
        out['profitability_normalized'] = 0.0
    out['lambda_merton'] = lam_m
    out['lambda_fcff'] = lam_f
    out['lambda_liquidity'] = lam_l
    out['lambda_profitability'] = lam_p
    out['reward_aux_merton'] = lam_m * out['delta_merton_badness_scaled'].fillna(0.0)
    out['reward_aux_fcff'] = lam_f * out['delta_fcff_capacity_scaled'].fillna(0.0)
    out['reward_aux_liquidity'] = lam_l * out['delta_liquid_capacity_scaled'].fillna(0.0)
    out['reward_aux_profitability'] = lam_p * out['profitability_normalized'].fillna(0.0)
    out['reward_merton_raw'] = out['delta_merton_badness']
    out['reward_merton_norm'] = out['delta_merton_badness_scaled']
    out['reward_fcff_raw'] = out['delta_fcff_capacity']
    out['reward_fcff_norm'] = out['delta_fcff_capacity_scaled']
    out['reward_profitability_raw'] = out['profitability_raw']
    out['reward_profitability_norm'] = out['profitability_normalized']
    out['reward_liquidity_raw'] = out['delta_liquid_capacity']
    out['reward_liquidity_norm'] = out['delta_liquid_capacity_scaled']
    meta = {
        'phase': phase_name,
        'merton_aux_enabled': bool(lam_m != 0.0),
        'fcff_aux_enabled': bool(lam_f != 0.0),
        'liquidity_aux_enabled': bool(lam_l != 0.0),
        'profitability_aux_enabled': bool(lam_p != 0.0),
        'merton_aux_mean': float(out['reward_aux_merton'].mean()),
        'fcff_aux_mean': float(out['reward_aux_fcff'].mean()),
        'liquidity_aux_mean': float(out['reward_aux_liquidity'].mean()),
        'profitability_aux_mean': float(out['reward_aux_profitability'].mean()),
        'merton_aux_nonzero_rate': float((out['reward_aux_merton'].abs() > 1e-12).mean()),
        'fcff_aux_nonzero_rate': float((out['reward_aux_fcff'].abs() > 1e-12).mean()),
        'liquidity_aux_nonzero_rate': float((out['reward_aux_liquidity'].abs() > 1e-12).mean()),
        'profitability_aux_nonzero_rate': float((out['reward_aux_profitability'].abs() > 1e-12).mean()),
        'reference_oracle_scores_used': False,
        'reference_oracle_variables_used': False,
        'r_code_fallback_allowed': False,
    }
    return out, meta

def _derived_from_state_dict(d: dict) -> dict[str, float]:
    def val(k):
        x = d.get(k, 0.0)
        try:
            x = float(x)
        except Exception:
            return 0.0
        return x if np.isfinite(x) else 0.0

    revenue = val("revenue")
    assets = val("total_assets")
    return {
        "derived__roa_proxy": val("net_income") / assets if assets else 0.0,
        "derived__operating_margin": val("operating_income") / revenue if revenue else 0.0,
        "derived__cogs_to_revenue": val("cogs") / revenue if revenue else 0.0,
        "derived__sga_to_revenue": val("sga") / revenue if revenue else 0.0,
        "derived__financial_cost_to_revenue": val("financial_cost") / revenue if revenue else 0.0,
        "derived__debt_to_assets": val("total_liabilities") / assets if assets else 0.0,
    }
