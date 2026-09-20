"""The single production time boundary and physically frozen run settings."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import importlib.abc
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from credit_recourse.rl.contracts.v43_encoder import content_hash, file_sha256
from credit_recourse.contracts.stage_paths import V43_STAGE2_RUNTIME_PATH

RUN_PATH = V43_STAGE2_RUNTIME_PATH.as_posix()


def canonical_rl_config_path(root: Path) -> Path:
    """Resolve the immutable V4.3 config without reviving the old archive root."""
    root = Path(root)
    configured = os.environ.get('THESIS_REPRO_RL_CONFIG_PATH')
    if configured:
        selected = Path(configured)
        selected = selected if selected.is_absolute() else root / selected
        if not selected.is_file():
            raise FileNotFoundError(f'configured V4.3 RL config is unavailable: {selected}')
        return selected.resolve()
    candidates = (
        root / "contracts/scientific/final_freeze/v43_rl.json",
        root / "configs/current/final_freeze/v43_rl.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "V4.3 RL config is unavailable; restore the authorized scientific contract "
        "under contracts/scientific/final_freeze/v43_rl.json"
    )


def run_path(project_root=None, run_id=None) -> str:
    """Resolve the producer namespace without breaking legacy callers.

    Existing production code still imports ``RUN_PATH``.  A coordinator may
    set ``CREDIT_RECOURSE_RUN_PATH`` before importing the producers, which
    redirects only outputs while leaving scientific inputs and formulas
    unchanged.
    """
    configured = os.environ.get("CREDIT_RECOURSE_RUN_PATH")
    if configured:
        return Path(configured).as_posix()
    if run_id:
        return (Path("runs") / run_id / "stage2_runtime").as_posix()
    return RUN_PATH

@dataclass(frozen=True)
class V43TemporalContract:
    eval_base_year: int = 2024
    train_transition_year_max: int = 2022
    train_outcome_year_max: int = 2023
    evaluation_rollout_year: int = 2025
    version: str = 'V43OnePassTemporal/1'

    def to_dict(self):
        return asdict(self)

TEMPORAL = V43TemporalContract()

def rl_fit_allowed(rows, *, action_support, reward_support, outcome_available):
    """One eligibility formula for all stages and all fitted statistics."""
    year = pd.to_numeric(rows.fiscal_year, errors='raise').to_numpy(dtype=float)
    target = pd.to_numeric(rows.outcome_year, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(year).all() or not np.isfinite(target).all() or (year != year.astype('int64')).any() or (target != target.astype('int64')).any():
        raise ValueError('Missing or invalid transition years')
    if (target != year + 1).any():
        raise ValueError('Transitions must be decision t -> outcome t+1')
    supports = []
    for support in (action_support, reward_support, outcome_available):
        values = np.asarray(support)
        if values.dtype != bool or values.shape != year.shape:
            raise ValueError('Explicit boolean support is required for every row')
        supports.append(values)
    return ((year <= TEMPORAL.train_transition_year_max) & (target <= TEMPORAL.train_outcome_year_max)
            & supports[0] & supports[1] & supports[2])

def assert_training_rows(rows):
    eligible = rl_fit_allowed(rows, action_support=rows.action_support_valid.to_numpy(dtype=bool),
        reward_support=rows.reward_support_valid.to_numpy(dtype=bool),
        outcome_available=rows.outcome_available.to_numpy(dtype=bool))
    if not len(rows) or not eligible.all() or not rows.rl_fit_allowed.eq(True).all():
        raise ValueError('Production training/statistics received an ineligible transition')
    return {'rows':len(rows), 'decision_year_max':int(rows.fiscal_year.max()),
            'outcome_year_max':int(rows.outcome_year.max()), 'evaluation_rows_used':0}

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')

def freeze_run_config(root, encoder_contract, *, config_path: Path | None = None, output_root: Path | None = None):
    root = Path(root)
    source = Path(config_path).resolve() if config_path is not None else canonical_rl_config_path(root)
    canonical = json.loads(source.read_text(encoding='utf-8'))
    stages = canonical['stages']
    from credit_recourse.simulator.v43_production_bundle import CANDIDATE_DIVIDEND_POLICY, BUNDLE_VERSION
    if canonical['candidate_dividend_policy'] != CANDIDATE_DIVIDEND_POLICY:
        raise ValueError('Canonical candidate dividend policy changed')
    if canonical['action_support_rule'] != 'all_nine_candidates_pass_frozen_simulator_invariants':
        raise ValueError('Canonical action-support rule changed')
    if canonical['temporal'] != TEMPORAL.to_dict() or canonical['architecture'] != encoder_contract.to_dict()['architecture']:
        raise ValueError('Canonical V43 architecture or temporal contract differs')
    stage_seeds = {name: int(value['seed']) for name, value in stages.items() if isinstance(value, dict) and 'seed' in value}
    if not stage_seeds or len(set(stage_seeds.values())) != 1:
        raise ValueError(f'Canonical V43 stage seeds are inconsistent: {stage_seeds}')
    selected_seed = next(iter(stage_seeds.values()))
    payload = {'version':'V43OnePassRun/4','temporal':TEMPORAL.to_dict(),
        'candidate_dividend_policy':CANDIDATE_DIVIDEND_POLICY,'production_bundle_version':BUNDLE_VERSION,
        'action_support_rule':canonical['action_support_rule'],
        'reproducibility_boundary':canonical['reproducibility_boundary'],
        'seed':selected_seed,'stages':stages,'architecture':encoder_contract.to_dict()['architecture'],
        'architecture_authority':'canonical V43 RL config and V43 encoder contract',
        'epoch_policy':'complete fixed epochs; save final epoch only',
        'execution':{'precision':'float32','activation_checkpointing':True,'preserve_dropout_rng':True,'frozen_encoder_embedding_cache':True},
        'statistics_population':'unique union of eligible Stage3/4/5 decision keys; loss-specific support must pass the common rl_fit_allowed rule',
        'action_scale_policy':'unchanged frozen nine-action geometry fitted on t<=2022; no evaluation projection or fit',
        'auxiliary_scale_population':'eligible factual Stage5 transitions, current t<=2022 and observed outcome<=2023',
        'reward_standardization_population':'eligible Stage5 nine-candidate training grid only',
        'canonical_config_sha256':file_sha256(source),
        'schema_hash':encoder_contract.schema_hash,
        'action_contract_sha256':encoder_contract.action_contract_sha256,
        'non_rate_calibration_sha256':encoder_contract.non_rate_calibration_sha256,
        'stage6_policy':'Stage5 final actor argmax; fixed-action comparators and ceiling are reporting only',
        'evaluation_role':'final_OOT; any subsequent settings informed by these results are exploratory/development',
        'numeric_profile_changes':False}
    assert all(int(stages[s]['seed'])==payload['seed'] for s in ('stage3','stage4','stage5'))
    payload['config_hash'] = content_hash(payload)
    destination = (Path(output_root) if output_root is not None else root/Path(run_path(root)))/'01_contract/resolved_run_config.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != payload:
            raise ValueError('The resolved run configuration is already frozen with different contents')
    else:
        write_json(destination,payload)
        write_json(destination.with_suffix('.sha256.json'), {'sha256':file_sha256(destination)})
    return payload

def load_run_config(root, *, output_root: Path | None = None):
    path = (Path(output_root) if output_root is not None else Path(root)/Path(run_path(root)))/'01_contract/resolved_run_config.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    digest = payload.pop('config_hash')
    if content_hash(payload) != digest or payload['temporal'] != TEMPORAL.to_dict():
        raise ValueError('Frozen one-pass configuration changed')
    if file_sha256(path) != json.loads(path.with_suffix('.sha256.json').read_text(encoding='utf-8'))['sha256']:
        raise ValueError('Physical run-config hash changed')
    if file_sha256(canonical_rl_config_path(root)) != payload['canonical_config_sha256']:
        raise ValueError('Canonical V43 config changed after freeze')
    return {**payload,'config_hash':digest}

class _LegacyImportBlocker(importlib.abc.MetaPathFinder):
    def __init__(self, counts):
        self.counts = counts

    def find_spec(self, fullname, path=None, target=None):
        if 'legacy_pipeline' in fullname or fullname.startswith('credit_recourse.rl.legacy_v43_') or fullname == 'credit_recourse.rl.common.temporal':
            self.counts['legacy_imports'] += 1
            raise RuntimeError('Legacy production dependency is forbidden: '+fullname)

@contextmanager
def production_guard():
    """Fail immediately on a legacy import or retired training-control call."""
    counts = {'legacy_imports':0,'retired_training_control_calls':0}
    finder = _LegacyImportBlocker(counts)
    previous = sys.getprofile()
    def trace(frame,event,arg):
        if event == 'call' and 'credit_recourse' in frame.f_code.co_filename:
            name = frame.f_code.co_name.lower()
            if (any(token in name for token in ('pareto','refit','frontier','best_epoch','best_checkpoint','best_seed','early_stopping','q_rerank','q_argmax','topk_rerank','top_k_rerank')) or name == '_selection_diagnostic' or frame.f_globals.get('__name__') == 'credit_recourse.rl.common.temporal'):
                counts['retired_training_control_calls'] += 1
                raise RuntimeError('Retired training-control function called: '+name)
            if 'legacy_pipeline' in frame.f_code.co_filename:
                counts['legacy_imports'] += 1
                raise RuntimeError('Legacy pipeline called from V43 production')
    sys.meta_path.insert(0,finder)
    sys.setprofile(trace)
    try:
        yield counts
    finally:
        sys.setprofile(previous)
        sys.meta_path.remove(finder)


def begin_training(consumer, stage, destination, rows):
    import os
    from datetime import datetime, timezone
    destination=Path(destination).resolve()
    from credit_recourse.contracts.stage_paths import stage_dir
    active_root = Path(consumer.root).resolve()
    if destination.name != "final_epoch.pt" or not destination.is_relative_to(active_root):
        raise ValueError("Production training output must be final_epoch.pt inside the active fresh run namespace")
    gate=json.loads((consumer.folder/'04_validation/pretraining_gate.json').read_text(encoding='utf-8'))
    if gate['status']!='PASS' or gate['config_hash']!=consumer.config['config_hash']:
        raise ValueError('Required pretraining gate is not PASS for this run')
    for name,digest in gate['source_hashes'].items():
        if file_sha256(consumer.root/name)!=digest:
            raise ValueError('Production source changed after passing the training gate: '+name)
    if stage>3:
        previous_dir = stage_dir(consumer.root, f"stage{stage-1}")
        previous=json.loads((previous_dir/'execution.json').read_text(encoding='utf-8'))
        if previous['status']!='PASS' or previous['final_epoch']!=consumer.config['stages'][f'stage{stage-1}']['max_epochs']:
            raise ValueError('Predecessor did not complete its fixed final epoch')
        if previous['checkpoint_sha256']!=file_sha256(previous_dir/'final_epoch.pt'):
            raise ValueError('Predecessor final checkpoint changed')
    destination.parent.mkdir(parents=True,exist_ok=True)
    execution={'status':'RUNNING','stage':stage,'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
        'config_hash':consumer.config['config_hash'],'input_rows':rows,
        'fixed_epochs':consumer.config['stages'][f'stage{stage}']['max_epochs'],'seed':consumer.config['seed']}
    with (destination.parent/'execution.json').open('x',encoding='utf-8') as stream:
        json.dump(execution,stream,indent=2)


def finish_training(consumer, stage, destination, optimizer_steps, rows):
    import math
    config=consumer.config['stages'][f'stage{stage}']
    expected_steps=math.ceil(rows/config['batch_size'])*config['max_epochs']
    if optimizer_steps!=expected_steps:
        raise ValueError('Optimizer step count differs from the fixed full-population epochs')
    path=Path(destination).parent/'execution.json'
    execution=json.loads(path.read_text(encoding='utf-8'))
    execution.update(status='PASS',final_epoch=config['max_epochs'],optimizer_steps=optimizer_steps,
        checkpoint_sha256=file_sha256(destination),evaluation_rows_used=0)
    write_json(path,execution)
