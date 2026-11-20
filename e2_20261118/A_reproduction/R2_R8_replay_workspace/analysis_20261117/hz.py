"""Lean re-execution harness around the kit's own simulator + Oracle scorers (no re-implementation)."""
import sys, os, json, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT = Path(os.environ.get('KIT_ROOT', '/home/claude/kitzip/THESIS_REPRO_KIT_v2.1.1_FINAL'))  # extracted frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip
SRC = os.environ.get('HZ_SRC', str(ROOT / 'src'))
sys.path.insert(0, SRC)
from credit_recourse.eval.final_stage8_llm_multi_oracle_eval.pipeline import (
    load_current_action_space, load_canonical_business_plan_history, prepare_llm_history_lookup, _build_llm_history)
from credit_recourse.eval.v43_financial_inputs import _row_to_firm_state
from credit_recourse.eval.v43_alpha_evaluation import (
    _prepare_stage6_context_lookup, _target_operating_loss_freq_3y, _target_cap_change_count_3y, _state_to_frame_dict)
from credit_recourse.simulator.v43_production_bundle import V43ProductionSimulationBundle
from credit_recourse.simulator.oracle_variables import compute_oracle_variables
from credit_recourse.eval.v43_oracle_backends import score_alpha, score_beta_ordered_logit_params, score_gamma_model

DR = ROOT / 'archive/DEPLOYED_RELEASE'
PA = DR / 'stage1_oracle_backends/alpha/oracle_alpha_params.json'
PB = DR / 'stage1_oracle_backends/beta/benchmark_beta_params.json'
PG = DR / 'stage1_oracle_backends/gamma/benchmark_gamma_params.json'
MG = DR / 'stage1_oracle_backends/gamma/benchmark_gamma_model.joblib'
AX = ['growth_capex_reduction_pct', 'deleveraging_total_debt_pct', 'refinancing_short_debt_pct', 'inv_turnover_chg',
      'ar_turnover_chg', 'ap_turnover_chg', 'cogs_ratio_chg', 'sga_ratio_chg']

def _sel(path):
    p = json.loads(Path(path).read_text(encoding='utf-8'))
    v = p.get('selected_variables') or [x.get('variable_id') for x in p.get('variables', []) if isinstance(x, dict)]
    return [x for x in v if x]

class Harness:
    def __init__(self):
        self.space = load_current_action_space(ROOT)
        self.base = pd.read_parquet(DR / 'stage2_candidate_projection/phase_eval_candidate.parquet').reset_index(drop=True)
        hist, _ = load_canonical_business_plan_history(ROOT)
        keep = set(self.base['firm_id'].astype(str))
        hist = hist[hist['firm_id'].astype(str).isin(keep)].reset_index(drop=True)
        self.hist_lookup = prepare_llm_history_lookup(hist)
        self.ctx = _prepare_stage6_context_lookup(hist)
        self.bundle = V43ProductionSimulationBundle.from_project_root(ROOT)
        self.contract = json.loads((DR / 'stage2_candidate_projection/candidate_action_contract_v4_3.json').read_text(encoding='utf-8'))
        self.scale = self.contract['normalization_scale_by_dimension']
        self.cands = {k: {a: float(v['action__' + a]) for a in AX} for k, v in self.contract['fixed_candidates'].items()}
        self.sel = {'alpha': _sel(PA), 'beta': _sel(PB), 'gamma': _sel(PG)}
        self.need = sorted(set(sum(self.sel.values(), [])))
        self._firm = {}
    def firm(self, i):
        if i not in self._firm:
            r = self.base.iloc[i]
            fs = _row_to_firm_state(r)
            hist = _build_llm_history(self.base, str(fs.firm_id), int(fs.year), history_lookup=self.hist_lookup)
            assert hist and int(hist[-1].year) == int(fs.year)
            exo = {'industry_median_rating_lag1_self_excl': r.get('industry_median_rating_lag1_self_excl'),
                   'industry_avg_rating_lag1_self_excl': r.get('industry_avg_rating_lag1_self_excl'),
                   'cap_change_count_3y': r.get('cap_change_count_3y'),
                   'cap_change_count_3y_target': _target_cap_change_count_3y(self.ctx, str(fs.firm_id), int(fs.year)),
                   'financial_data_completeness': r.get('financial_data_completeness'),
                   'nf_retained_earnings_negative_flag': r.get('nf_retained_earnings_negative_flag'),
                   'nf_ratio_missing_rate': r.get('nf_ratio_missing_rate'), 'ratio_missing_rate': r.get('ratio_missing_rate')}
            self._firm[i] = (r, fs, hist, exo)
        return self._firm[i]
    def sim_row(self, i, action, label='X', extra=False):
        r, fs, hist, exo = self.firm(i)
        res, bp, rate, act, prim, lin = self.bundle.simulate_action(fs, hist, dict(action), action_label=label)
        ex = dict(exo); ex['operating_loss_freq_3y_target'] = _target_operating_loss_freq_3y(hist, res.state_t1)
        sv = compute_oracle_variables(res.state_t1, prev_state=fs, exogenous=ex)
        row = _state_to_frame_dict(sv, r)
        out = {v: (row[v] if v in row else r.get(v)) for v in self.need}
        out['row_id'] = i
        if extra:
            out['_state'] = res.state_t1; out['_res'] = res; out['_bp'] = bp
        return out
    def score(self, rows):
        df = pd.DataFrame([{k: v for k, v in x.items() if not k.startswith('_')} for x in rows])
        return pd.DataFrame({'row_id': df['row_id'].values,
                             'alpha': score_alpha(df, PA).values,
                             'beta': score_beta_ordered_logit_params(df, PB).values,
                             'gamma': score_gamma_model(df, PG, MG).values})
    def evaluate(self, items):
        """items: iterable of (row_id, action_dict). Returns DataFrame alpha/beta/gamma (absolute scores) + ok flag."""
        rows = []; ok = []
        for i, a in items:
            try:
                rows.append(self.sim_row(i, a)); ok.append(True)
            except Exception as e:
                rows.append(None); ok.append(False)
        good = [x for x in rows if x is not None]
        sc = self.score(good) if good else pd.DataFrame(columns=['row_id', 'alpha', 'beta', 'gamma'])
        out = pd.DataFrame({'ok': ok}); 
        for c in ['alpha', 'beta', 'gamma']:
            out[c] = np.nan; out.loc[out.ok.values, c] = sc[c].values
        return out
