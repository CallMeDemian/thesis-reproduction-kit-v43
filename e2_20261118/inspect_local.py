import pandas as pd
base = r'analysis/rl_reference_gate_runtime/data/final_freeze'
files = [
    'stage7_llm_action_generation/llm_stage7_action_table_itt.parquet',
    'stage7_llm_action_generation/llm_stage7_action_table_per_protocol.parquet',
    'stage8_llm_multi_oracle_eval/llm_stage8_multi_oracle_scores_itt.parquet',
    'stage8_llm_multi_oracle_eval/llm_stage8_multi_oracle_scores_per_protocol.parquet',
    'stage8_llm_multi_oracle_eval/simulated_oracle_input_frame.parquet',
    'stage8_llm_multi_oracle_eval/action_effect_audit.parquet',
]
for name in files:
    d = pd.read_parquet(base + '/' + name)
    print('\nFILE', name, 'shape=', d.shape)
    print('COLUMNS', d.columns.tolist())
    for c in ['row_id','model_key','policy_condition','condition','replicate','mode','info','budget','analysis_arm','strict_usable']:
        if c in d.columns:
            print('UNIQUE', c, d[c].nunique(dropna=False))
    for c in ['policy_condition','condition','analysis_arm','mode','info','budget','replicate','model_key']:
        if c in d.columns:
            print('COUNTS', c, d[c].value_counts(dropna=False).to_dict())
