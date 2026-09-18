"""Immutable raw M/F/L components and per-run reward recomposition."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from credit_recourse.rl.contracts.v43_encoder import ACTION_IDS, KEYS, file_sha256, content_hash
from credit_recourse.rl.v43_one_pass_contract import assert_training_rows, write_json
AXES=('merton','fcff','liquidity')
CLIPS={'merton':[-1.,1.],'fcff':[-.5,.5],'liquidity':[-.5,.5]}
WEIGHTED={'reward_total_raw','reward_total','reward_train','reward_mean_train','reward_std_train','reward','reward_original'}

def freeze_components(frame,statistics,metadata,folder):
    folder=Path(folder);path=folder/'reward_components.parquet'
    if path.exists():raise FileExistsError('Weight-independent components already frozen')
    omit=[c for c in frame if c in WEIGHTED or c.startswith(('lambda_','reward_aux_'))]
    component=frame.drop(columns=omit).copy()
    for axis in AXES:
        component['component_scale__'+axis]=statistics[axis+'_aux_scale_p95_abs']
        component['component_clip_low__'+axis],component['component_clip_high__'+axis]=CLIPS[axis]
    assert_training_rows(component)
    if any(c in component for c in WEIGHTED):raise ValueError('Weighted reward leaked into independent artifact')
    component.to_parquet(path,index=False)
    sources=dict(metadata['source_hashes'])
    report={k:v for k,v in metadata.items() if k not in ('reward_statistics','dataset_sha256','config_hash')}
    report.update(status='PASS',artifact_kind='weight_independent_MFL_components',rows=len(component),
        factual_rows=component.groupby(list(KEYS)).ngroups,dataset_sha256=file_sha256(path),
        source_hashes=sources,axes=list(AXES),clip_contract=CLIPS,
        scales={a:statistics[a+'_aux_scale_p95_abs'] for a in AXES},
        statistics_fit='eligible factual t<=2022 outcome<=2023 only',weighted_columns=[],
        scale_fit_key_hash=statistics['fit_key_hash'],evaluation_rows_used=0)
    report['component_contract_hash']=content_hash(report)
    write_json(folder/'reward_components_metadata.json',report)
    return report

def recompose(frame,weights):
    if set(weights)!=set(AXES) or any(not np.isfinite(v) or v<0 for v in weights.values()) or sum(weights.values())<=0:
        raise ValueError('M/F/L weights must be finite nonnegative with positive sum')
    assert_training_rows(frame)
    if frame.duplicated([*KEYS,'candidate_id']).any() or not frame.groupby(list(KEYS)).candidate_id.agg(lambda a:set(a)==set(ACTION_IDS) and len(a)==9).all():
        raise ValueError('Incomplete nine-action component grid')
    out=frame.copy();total=out.rating_supervision_numeric_contribution.copy()
    for axis in AXES:
        low,high=CLIPS[axis];scale=out['component_scale__'+axis]
        if scale.nunique()!=1 or (scale<=0).any():raise ValueError('Invalid frozen component scale')
        expected=np.clip(out['reward_'+axis+'_raw']/scale,low,high)
        np.testing.assert_allclose(out['reward_'+axis+'_norm'],expected,rtol=0,atol=1e-12)
        if not np.isfinite(expected).all():raise ValueError('Nonfinite component')
        out['lambda_'+axis]=weights[axis]
        out['reward_aux_'+axis]=weights[axis]*out['reward_'+axis+'_norm']
        total=total+out['reward_aux_'+axis]
    out['lambda_phi']=0.;out['lambda_profitability']=0.
    out['reward_aux_phi']=0.;out['reward_aux_profitability']=0.
    if not np.isfinite(total).all():raise ValueError('Nonfinite recomposed reward')
    mean=float(total.mean());std=float(total.std(ddof=1))
    fallback=not np.isfinite(std) or std<=1e-12
    if fallback:std=1.
    out['reward_total_raw']=total;out['reward_total']=total
    out['reward_mean_train']=mean;out['reward_std_train']=std;out['reward_train']=(total-mean)/std
    from credit_recourse.rl.v43_training_math import _validate_projected_rating_supervision
    supervision=_validate_projected_rating_supervision(out,list(ACTION_IDS))
    stats={'fit_rows':len(out),'factual_rows':out.groupby(list(KEYS)).ngroups,'mean':mean,'std':std,
        'std_ddof':1,'degenerate_std_fallback':fallback,**assert_training_rows(out),'evaluation_rows_used':0,
        'weights':weights,'component_contract':'frozen raw/normalized components; no simulator invocation'}
    return out,stats,supervision
