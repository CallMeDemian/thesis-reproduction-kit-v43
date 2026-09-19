"""Frozen Beta/Gamma scoring math without legacy Stage6 control imports."""
from pathlib import Path
from functools import lru_cache
from typing import Any
import json
import numpy as np
import pandas as pd

def _logistic_cdf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))

def _transform_ordered_logit_thresholds(raw_threshold_params: list[float]) -> np.ndarray:
    """Statsmodels OrderedModel threshold transform.

    The first threshold is direct; later threshold parameters are exponentiated
    increments and cumulatively summed to enforce monotonic cutpoints.
    """
    raw = np.asarray(raw_threshold_params, dtype=float)
    if raw.size == 0:
        return raw
    increments = np.concatenate([raw[:1], np.exp(raw[1:])])
    return np.cumsum(increments)

@lru_cache(maxsize=16)
def _load_json_artifact_cached(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    return json.loads(path.read_text(encoding='utf-8'))

@lru_cache(maxsize=8)
def _load_gamma_model_cached(path_text: str):
    import joblib
    return joblib.load(Path(path_text))

def score_beta_ordered_logit_params(df: pd.DataFrame, params_path: Path) -> pd.Series:
    params=_load_json_artifact_cached(str(Path(params_path).resolve()))
    if 'ordered' not in str(params.get('model_name','')).lower() and 'ordered' not in str(params.get('version','')).lower():
        raise ValueError(f'Beta params are not ordered-logit backend params: {params_path}')
    vars=params.get('selected_variables') or []
    std=params.get('standardization_params') or {}; coef_records=params.get('coefficients') or []
    coef={str(r.get('variable')): float(r.get('coefficient')) for r in coef_records if str(r.get('variable')) in vars and r.get('coefficient') is not None}
    missing=[v for v in vars if v not in df.columns or v not in std or v not in coef]
    if missing: raise KeyError(f'Beta ordered-logit scoring missing exported feature inputs: {missing[:20]}')

    grade_nums = [int(x) for x in (params.get('modeled_grade_nums') or params.get('probability_output_grade_nums') or [])]
    if len(grade_nums) < 2:
        raise ValueError('Beta ordered-logit params missing modeled_grade_nums/probability_output_grade_nums')

    finite_cutpoints = params.get('ordered_logit_finite_cutpoints') or params.get('finite_cutpoints')
    if finite_cutpoints is None:
        raw_thresholds = params.get('ordered_logit_threshold_raw_params') or []
        if isinstance(raw_thresholds, list) and raw_thresholds and isinstance(raw_thresholds[0], dict):
            raw_thresholds = [r.get('coefficient') for r in raw_thresholds]
        if not raw_thresholds:
            # Backward-compatible fallback for older beta params: threshold rows are
            # the coefficient records whose variable names are not selected features.
            raw_thresholds = [r.get('coefficient') for r in coef_records if str(r.get('variable')) not in set(vars) and r.get('coefficient') is not None]
        finite_cutpoints = _transform_ordered_logit_thresholds([float(x) for x in raw_thresholds]).tolist()

    finite_cutpoints = np.asarray(finite_cutpoints, dtype=float)
    if len(finite_cutpoints) != len(grade_nums) - 1:
        raise ValueError(f'Beta cutpoint/class mismatch: {len(finite_cutpoints)} cutpoints for {len(grade_nums)} classes')

    xb=np.zeros(len(df), dtype=float)
    for v in vars:
        mean=float(std[v].get('mean',0.0)); scale=float(std[v].get('std',1.0)) or 1.0
        x=pd.to_numeric(df[v], errors='coerce').fillna(mean)
        xb += ((x-mean)/scale).to_numpy(dtype=float) * coef[v]

    thresholds = np.concatenate([[-np.inf], finite_cutpoints, [np.inf]])
    upper = _logistic_cdf(thresholds[1:][None, :] - xb[:, None])
    lower = _logistic_cdf(thresholds[:-1][None, :] - xb[:, None])
    upper[:, -1] = 1.0
    lower[:, 0] = 0.0
    probs = np.clip(upper - lower, 0.0, 1.0)
    denom = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, denom, out=np.full_like(probs, 1.0 / probs.shape[1]), where=denom > 1e-12)
    expected_rating = probs @ np.asarray(grade_nums, dtype=float)
    score = np.clip(100.0 * (10.0 - expected_rating) / 9.0, 0.0, 100.0)
    return pd.Series(score, index=df.index, name='R_score_beta')

def score_gamma_model(df: pd.DataFrame, params_path: Path, model_path: Path) -> pd.Series:
    params=_load_json_artifact_cached(str(Path(params_path).resolve()))
    vars=params.get('selected_variables') or []
    if not vars: raise ValueError('Gamma params missing selected_variables')
    missing=[v for v in vars if v not in df.columns]
    if missing: raise KeyError(f'Gamma model scoring missing features: {missing[:20]}')
    model=_load_gamma_model_cached(str(Path(model_path).resolve()))
    pred=np.asarray(model.predict(df[vars].apply(pd.to_numeric, errors='coerce')), dtype=float)
    return pd.Series((100.0*(1.0-(pred-1.0)/9.0)).clip(0,100), index=df.index, name='R_score_gamma')

@lru_cache(maxsize=8)
def _load_alpha_scorer_cached(path_text: str):
    from credit_recourse.oracle.backends.alpha.modules.oracle_alpha_scorer import build_alpha_scorer
    params = _load_json_artifact_cached(path_text)
    return build_alpha_scorer(params)

def score_alpha(df: pd.DataFrame, params_path: Path) -> pd.Series:
    params=_load_json_artifact_cached(str(Path(params_path).resolve()))
    vars=params.get('selected_variables') or [v.get('variable_id') for v in params.get('variables',[]) if isinstance(v,dict)]
    vars=[v for v in vars if v]
    if not vars: raise ValueError('Alpha params missing selected_variables')
    missing=[v for v in vars if v not in df.columns]
    if missing: raise KeyError(f'Alpha scoring missing variables: {missing[:20]}')
    scorer=_load_alpha_scorer_cached(str(Path(params_path).resolve()))
    vals=[]
    for _, row in df.iterrows():
        vals.append(float(scorer({v: row.get(v) for v in vars})['R_score']))
    return pd.Series(vals, index=df.index, name='R_score_alpha')
