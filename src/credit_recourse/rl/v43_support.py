"""Candidate-independent support admission under the unchanged V43 simulator."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from credit_recourse.rl.v43_one_pass_contract import RUN_PATH, TEMPORAL, write_json, run_path
from credit_recourse.rl.v43_one_pass_data import input_root
from credit_recourse.rl.contracts.v43_encoder import ACTION_IDS, KEYS, file_sha256, content_hash
from credit_recourse.rl.v43_features import canonical_keys, key_hash
from credit_recourse.simulator.historical_source import read_financial_panel, state_from_record
from credit_recourse.simulator.v43_production_bundle import V43ProductionSimulationBundle, financial_record

PHASES = {3: 'phase1_pretrain.parquet', 4: 'phase2_bc.parquet', 5: 'phase3_iql.parquet'}
SUPPORT_RULE = 'all_nine_candidates_pass_frozen_simulator_invariants'
INFEASIBILITY_PREFIXES = (
    'R085_MISSING_ASOF_COMPONENT:', 'NONPHYSICAL_STATE', 'DL feasibility invariant failed:', 'Executed principal exceeds opening stock;',
    'v4 incremental accounting residual exceeds contract tolerance:',
    'v4 liquidity financing failed to close negative cash:',
)


def temporal_source_rows(root):
    frames = []
    for name in PHASES.values():
        path = input_root(root) / 'input_splits' / name
        columns = list(KEYS) + (['fiscal_year_next'] if 'fiscal_year_next' in pq.read_schema(path).names else [])
        frame = pd.read_parquet(path, columns=columns)
        frame[list(KEYS)] = canonical_keys(frame)
        frame['outcome_year'] = frame.pop('fiscal_year_next') if 'fiscal_year_next' in frame else frame.fiscal_year + 1
        if not frame.outcome_year.eq(frame.fiscal_year + 1).all():
            raise ValueError('Factual transitions must be t -> t+1')
        frames.append(frame.loc[frame.fiscal_year.le(TEMPORAL.train_transition_year_max) & frame.outcome_year.le(TEMPORAL.train_outcome_year_max)])
    return pd.concat(frames, ignore_index=True).drop_duplicates(list(KEYS)).sort_values(list(KEYS)).reset_index(drop=True)


def collapse_candidate_support(candidates):
    if candidates.duplicated([*KEYS, 'candidate_id']).any():
        raise ValueError('Duplicate candidate support evidence')
    groups = candidates.groupby(list(KEYS), sort=True)
    if not groups.candidate_id.agg(lambda x: len(x) == 9 and set(x) == set(ACTION_IDS)).all():
        raise ValueError('Support requires an attempted complete nine-action grid')
    if candidates.candidate_supported.dtype != bool:
        raise ValueError('Explicit boolean candidate support is required')
    return groups.agg(action_support_valid=('candidate_supported', 'all'),
        feasible_candidates=('candidate_supported', 'sum'),
        failed_candidates=('candidate_id', lambda x: '|'.join(x.loc[~candidates.loc[x.index, 'candidate_supported']]))).reset_index()


def read_action_support(root):
    folder = Path(root) / run_path(root) / '02_data'
    metadata = json.loads((folder / 'action_support_metadata.json').read_text(encoding='utf-8'))
    path = folder / 'action_support.parquet'
    if metadata['status'] != 'PASS' or metadata['support_rule'] != SUPPORT_RULE or file_sha256(path) != metadata['support_sha256']:
        raise ValueError('Unverified action-support evidence')
    for name, digest in metadata['source_hashes'].items():
        if file_sha256(Path(root) / name) != digest:
            raise ValueError('Action-support source changed: ' + name)
    for name,digest in metadata['part_hashes'].items():
        if file_sha256(folder/metadata['parts_relative_path']/name)!=digest:raise ValueError('Support financial/attempt evidence changed')
    support = pd.read_parquet(path)
    if key_hash(support) != metadata['source_key_hash']:
        raise ValueError('Action-support population changed')
    return support, metadata


def prepare_action_support(root):
    root = Path(root)
    folder = root / run_path(root) / '02_data'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'action_support_metadata.json').exists():
        raise FileExistsError('Action support already frozen')
    rows = temporal_source_rows(root)
    history_path = input_root(root) / 'input_splits/canonical_business_plan_history.parquet'
    accounts, _ = read_financial_panel(history_path)
    accounts = accounts.loc[accounts.fiscal_year.le(2022)]
    states, histories = {}, {}
    for raw in accounts.to_dict('records'):
        state = state_from_record(raw)
        key = (str(state.firm_id), int(state.year))
        states[key] = state
        histories.setdefault(key[0], []).append(state)
    from credit_recourse.rl.v43_rate_extension import training_rate_resolver, read_extended_rates
    bundle = replace(V43ProductionSimulationBundle.from_project_root(root), _rate_resolver=training_rate_resolver(root))
    _, rate_report = read_extended_rates(root)
    unavailable_rates = {(r['firm_id'],r['fiscal_year']):r['reason'] for r in rate_report.get('unavailable_rate_keys',[])}
    stage5 = canonical_keys(pd.read_parquet(input_root(root) / 'input_splits' / PHASES[5], columns=list(KEYS)))
    retain = set(stage5.itertuples(index=False, name=None))
    sources = [history_path, root / RUN_PATH / '01_contract/extended_bp_rate_ledger.parquet', root / RUN_PATH / '01_contract/extended_bp_rate_preflight.json',
        root / 'src/credit_recourse/rl/v43_support.py', root / 'src/credit_recourse/rl/v43_one_pass_contract.py']
    sources += [input_root(root) / 'input_splits' / name for name in PHASES.values()]
    sources += list((root / 'src/credit_recourse/simulator').glob('*.py'))
    sources += [input_root(root) / 'candidate_action_contract_v4_3.json']
    from credit_recourse.simulator.financial_cost_v43 import SOURCE_PATH
    sources += [root/SOURCE_PATH, root/RUN_PATH/'01_contract/r085_financial_cost_contract.json', root/'src/credit_recourse/contracts/account_registry.py']
    source_hashes = {p.relative_to(root).as_posix(): file_sha256(p) for p in sources}
    stamp = {'support_rule': SUPPORT_RULE, 'source_key_hash': key_hash(rows), 'source_hashes': source_hashes}
    parts = folder / 'support_parts' / content_hash(stamp)[:16]
    parts.mkdir(parents=True,exist_ok=True)
    manifest = parts / 'inputs.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != stamp:
        raise ValueError('Cannot resume support audit with changed inputs')
    if not manifest.exists(): write_json(manifest, stamp)
    for start in range(0, len(rows), 250):
        part = parts / f'part_{start:06d}.parquet'
        financial_path = parts / f'financial_{start:06d}.parquet'
        done = part.with_suffix('.json')
        if done.exists():
            saved = json.loads(done.read_text(encoding='utf-8'))
            for name, digest in saved['files'].items():
                if file_sha256(parts / name) != digest: raise ValueError('Support audit part changed')
            continue
        attempts, financial = [], []
        for row in rows.iloc[start:start+250].itertuples(index=False):
            key = (row.firm_id, int(row.fiscal_year))
            state = states[key]
            for candidate in ACTION_IDS:
                entry = {'firm_id': key[0], 'fiscal_year': key[1], 'outcome_year': key[1]+1,
                    'candidate_id': candidate, 'candidate_supported': True, 'simulation_executed': key not in unavailable_rates, 'error': ''}
                if key in unavailable_rates:
                    entry.update(candidate_supported=False,error='missing_asof_BP_rate: '+unavailable_rates[key])
                    attempts.append(entry)
                    continue
                try:
                    payload = bundle.simulate_candidate(state, histories[key[0]], candidate)
                except ValueError as error:
                    if not str(error).startswith(INFEASIBILITY_PREFIXES):
                        write_json(parts / 'unexpected_error.json', {**entry, 'error': str(error)})
                        raise
                    entry.update(candidate_supported=False, error=str(error))
                else:
                    if key in retain:
                        financial.append(financial_record(state, candidate, payload))
                attempts.append(entry)
        pd.DataFrame(attempts).to_parquet(part, index=False)
        files = {part.name: file_sha256(part)}
        if financial:
            pd.DataFrame(financial).to_parquet(financial_path, index=False)
            files[financial_path.name] = file_sha256(financial_path)
        write_json(done, {'files': files})
        print('V43_SUPPORT', min(start+250, len(rows)), len(rows), sum(not a['candidate_supported'] for a in attempts), flush=True)
    candidates = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob('part_*.parquet'))], ignore_index=True)
    support = collapse_candidate_support(candidates)
    if key_hash(support) != key_hash(rows): raise ValueError('Support audit omitted factual keys')
    support.to_parquet(folder / 'action_support.parquet', index=False)
    candidates.to_parquet(folder / 'candidate_support.parquet', index=False)
    rejected = support.loc[~support.action_support_valid]
    rejected.to_csv(folder / 'unsupported_states.csv', index=False)
    metadata = {'status': 'PASS', **stamp, 'source_rows': len(rows), 'candidate_rows': len(candidates),
        'unavailable_rate_states':len(unavailable_rates),'simulator_candidate_calls':int(candidates.simulation_executed.sum()),'unsupported_states': len(rejected), 'failed_candidates': int((~candidates.candidate_supported).sum()),
        'support_sha256': file_sha256(folder / 'action_support.parquet'),
        'candidate_support_sha256': file_sha256(folder / 'candidate_support.parquet'),
        'parts_relative_path':parts.relative_to(folder).as_posix(),
        'part_hashes':{p.name:file_sha256(p) for p in parts.glob('*.parquet')},
        'decision_year_max': int(rows.fiscal_year.max()), 'outcome_year_max': int(rows.outcome_year.max()),
        'evaluation_rows_used': 0, 'firm_specific_exceptions': False, 'simulator_modified': True, 'simulator_change': 'R085 broad cost repair authorized by frozen user request'}
    write_json(folder / 'action_support_metadata.json', metadata)
    print('V43_SUPPORT_PASS', len(rows), len(rejected), metadata['failed_candidates'], flush=True)
    return metadata
