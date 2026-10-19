"""T4: robustness R1-R12 (summaries of frozen rows), R9 (candidate9 A2/A3, 24 rows) and R10 (reference assignment).

R9: CANDIDATE9, Strict ITT, B1, IC-b, Run1. Baseline rows are recomputed from the observations (bootstrap seed from
the registry key rule, analysis_id ABL_CAND9_BASELINE); High rows take estimate, interval and raw p from
HIGH_REASONING_SUPPLEMENTAL. Holm per model x Oracle x regime over the two contrasts (A2, A3) for both regimes; the
60-family p is carried next to the High rows. Differences from the thesis are logged, not forced.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import common as c

NOTE_R9_BASE = "recomputed from observations (not in the frozen registry; thesis Sec.3)"
NOTE_R9_HIGH = "estimate, CI and raw p from HIGH_REASONING_SUPPLEMENTAL; Holm re-applied over the two contrasts; holm_p_60 is the frozen 60-family value"


def r9() -> pd.DataFrame:
    obs = c.load_obs()
    supp = pd.read_csv(c.SUPP_PATH)
    base = obs[(obs.execution_policy == "STRICT") & (obs.analysis_population.astype(str).str.lower() == "itt") & (obs.action_space == "CANDIDATE9")
               & (obs.information_condition == "IC-b") & (obs.replicate == 1) & (obs.budget == "B1")]
    rows = []
    for cid, lhs, rhs in (("A2", "C4R", "C4"), ("A3", "C6-E", "C4R")):
        for regime in ("BASELINE", "HIGH"):
            for model in c.MODELS:
                for oracle in c.ORACLES:
                    base_row = dict(contrast_id=cid, lhs=lhs, rhs=rhs, model=c.MODEL_CODE[model], reasoning_regime=regime,
                                    oracle=oracle, budget="B1", action_space="CANDIDATE9", execution_policy="STRICT",
                                    replicate="RUN1", information_condition="IC-b", status="RECOMPUTED",
                                    holm_family=f"R9_{c.MODEL_CODE[model]}_{oracle}_{regime}_2")
                    if regime == "BASELINE":
                        sub = base[(base.reasoning_regime == "BASELINE") & (base.model_key == model)]
                        pair = c.paired_by(sub, "policy_condition", lhs, rhs, f"delta_R_score_{oracle}", ["firm_key"])
                        if len(pair) != 575:
                            raise SystemExit(f"R9: {cid}/{model}/{oracle} has {len(pair)} firms")
                        key = c.seed_key(analysis_id="ABL_CAND9_BASELINE", lhs=lhs, rhs=rhs, model=c.MODEL_CODE[model], oracle=oracle,
                                         reasoning_regime="BASELINE", budget="B1", action_space="CANDIDATE9")
                        res = c.boot(pair["difference"].to_numpy(), key)
                        base_row.update(n_firms=res["N"], estimate=res["estimate"], ci_low=res["ci_low"], ci_high=res["ci_high"],
                                        p_raw=res["raw_p"], holm_p_60=float("nan"), source_file=c.OBS_PATH.name,
                                        source_key=res["key_json"], source_sha256=c.sha256_file(c.OBS_PATH),
                                        notes=NOTE_R9_BASE + f"; draws_sha256={res['bootstrap_draw_sha256']}")
                    else:
                        m = supp[(supp.holm_family_id == "HIGH_REASONING_SUPPLEMENTAL") & (supp.action_space == "CANDIDATE9")
                                 & (supp.model == c.MODEL_CODE[model]) & (supp.oracle == oracle.upper())
                                 & (supp.lhs == lhs) & (supp.rhs == rhs)]
                        if len(m) != 1:
                            raise SystemExit(f"R9: no unique HIGH row for {cid}/{model}/{oracle}")
                        m = m.iloc[0]
                        base_row.update(n_firms=int(m["N"]), estimate=m["estimate"], ci_low=m["ci_low"], ci_high=m["ci_high"],
                                        p_raw=m["raw_p"], holm_p_60=m["holm_p"], source_file=c.SUPP_PATH.name,
                                        source_key=m["key_json"], source_sha256=c.sha256_file(c.SUPP_PATH), notes=NOTE_R9_HIGH)
                    base_row["result_id"] = f"R9_{cid}_{c.MODEL_CODE[model]}_{regime}_{oracle}"
                    rows.append(base_row)
    t = pd.DataFrame(rows)
    t["holm_p"] = np.nan
    for (_, _, _), idx in t.groupby(["model", "oracle", "reasoning_regime"]).groups.items():
        assert len(idx) == 2
        t.loc[idx, "holm_p"] = c.holm(t.loc[idx, "p_raw"].to_numpy())
    t["analysis_commit"] = c.git_commit()
    t = t[c.RESULT_COLS]
    assert len(t) == 24
    c.write_csv(t, c.RESULTS / "R9_candidate_selection.csv")
    return t


def thesis_targets():
    """Table 6-5 (A3 means and stars) and the Sec.6.3 A2 values, read from the thesis in the repo (not copied)."""
    import docx
    d = docx.Document(str(c.KIT / "human_baseline_final_20261019/02_학위논문_및_발표/학위논문_최종본_개정.docx"))
    table = None
    for t in d.tables:
        if t.rows[0].cells[0].text.strip() == "모델 · 생성 설정" and t.rows[0].cells[1].text.strip() == "α":
            table = t
    if table is None:
        raise SystemExit("R9: thesis Table 6-5 not found")
    out = {}
    for r in table.rows[1:]:
        cells = [x.text.strip() for x in r.cells]
        model = "GPT" if cells[0].startswith("GPT") else "GEMINI"
        regime = "BASELINE" if "낮음" in cells[0] else "HIGH"
        for oracle, v in zip(c.ORACLES, cells[1:4]):
            out[("A3", model, regime, oracle)] = (float(v.replace("−", "-").replace("+", "").replace("*", "")), v.endswith("*"))
    a2 = {("A2", "GPT", "BASELINE", "alpha"): (0.025, True), ("A2", "GPT", "HIGH", "alpha"): (0.040, True),
          ("A2", "GEMINI", "BASELINE", "beta"): (-0.020, True), ("A2", "GEMINI", "HIGH", "beta"): (-0.004, True)}
    out.update(a2)
    return out


def r9_checks(t: pd.DataFrame) -> dict:
    targets = thesis_targets()
    detail, mean_ok, star_ok = [], 0, 0
    a2_sig = set(k for k, v in targets.items() if k[0] == "A2")
    for _, r in t.iterrows():
        key = (r.contrast_id, r.model, r.reasoning_regime, r.oracle)
        sig = bool(r.holm_p < 0.05)
        row = {"key": "|".join(key), "estimate": round(float(r.estimate), 6), "holm_p": round(float(r.holm_p), 6),
               "holm_p_60": None if pd.isna(r.holm_p_60) else round(float(r.holm_p_60), 6), "significant": sig}
        if key in targets:
            tv, ts = targets[key]
            row.update(thesis_value=tv, thesis_star=ts, mean_match_3dp=bool(round(float(r.estimate), 3) == round(tv, 3)),
                       star_match=bool(sig == ts))
            mean_ok += row["mean_match_3dp"]
            star_ok += row["star_match"]
        elif r.contrast_id == "A3":
            row["thesis_star"] = False
            row["star_match"] = bool(not sig)
        else:
            row["star_match"] = bool(not sig) if key not in a2_sig else None
        detail.append(row)
    a3 = t[t.contrast_id == "A3"]
    a2 = t[t.contrast_id == "A2"]
    summary = {
        "a3_positive_of_12": int((a3.estimate > 0).sum()), "a3_holm_sig_of_12": int((a3.holm_p < 0.05).sum()),
        "a2_positive_of_12": int((a2.estimate > 0).sum()), "a2_holm_sig_of_12": int((a2.holm_p < 0.05).sum()),
        "a2_sig_with_60family_of_12": int((((a2.reasoning_regime == "BASELINE") & (a2.holm_p < 0.05)) | ((a2.reasoning_regime == "HIGH") & (a2.holm_p_60 < 0.05))).sum()),
        "thesis_expected": {"a3_positive": 11, "a3_sig": 2, "a2_positive": 6, "a2_sig": 4, "a2_sig_60": 2},
        "a3_means_matched_3dp": f"{sum(1 for d in detail if d.get('mean_match_3dp') and d['key'].startswith('A3'))}/12",
        "a2_means_matched_3dp": f"{sum(1 for d in detail if d.get('mean_match_3dp') and d['key'].startswith('A2'))}/4 (the four significant values quoted in Sec.6.3)",
        "all_star_checks_match": all(d["star_match"] for d in detail if d.get("star_match") is not None),
        "rows": detail,
    }
    (c.RESULTS / "_R9_checks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    return summary


def r10() -> pd.DataFrame:
    obs = c.load_obs()
    ex = obs[(obs.policy_condition == "C6-EX") & (obs.execution_policy == "STRICT")]
    ce = obs[(obs.policy_condition == "C6-E") & (obs.execution_policy == "STRICT") & (obs.action_space == "FREE8")]
    own = ce.groupby("firm_key")["reference_candidate_id"].agg(lambda s: sorted(set(s)))
    if (own.map(len) != 1).any():
        raise SystemExit("R10: C6-E reference is not constant per firm")
    own = own.map(lambda v: v[0])
    per = ex.groupby("firm_key").agg(ref_row=("reference_row_id", lambda s: sorted(set(s))), ref_cand=("reference_candidate_id", lambda s: sorted(set(s))),
                                     row_id=("row_id", "first"))
    if (per.ref_row.map(len) != 1).any() or (per.ref_cand.map(len) != 1).any():
        raise SystemExit("R10: C6-EX reference is not constant per firm across cells")
    per["ref_row"] = per.ref_row.map(lambda v: int(v[0]))
    per["ref_cand"] = per.ref_cand.map(lambda v: v[0])
    per["own_action"] = own.reindex(per.index)
    per["self_reference"] = per.ref_row == per.row_id
    per["same_action"] = per.ref_cand == per.own_action
    assert len(per) == 575
    out = per.reset_index()[["firm_key", "row_id", "ref_row", "ref_cand", "own_action", "self_reference", "same_action"]]
    out = out.rename(columns={"ref_row": "c6ex_reference_row_id", "ref_cand": "c6ex_reference_action", "own_action": "own_c3e_action"})
    c.write_csv(out, c.RESULTS / "R10_reference_assignment.csv")
    return out


def summaries(r9t: pd.DataFrame, r10t: pd.DataFrame) -> pd.DataFrame:
    prim = pd.concat([pd.read_csv(c.RESULTS / "T1_primary_A1_A4.csv")])
    supp = pd.read_csv(c.SUPP_PATH)
    parent = pd.read_csv(c.PARENT_PATH)
    rows = []

    def add(rid, name, element, source, n, pos, sig, rule, status, note):
        rows.append(dict(check_id=rid, check=name, changed_element=element, source=source, n_rows=n, n_positive=pos,
                         n_significant=sig, significance_rule=rule, status=status, note=note))

    # R1: Oracle beta, gamma versus alpha
    a = prim[prim.oracle == "alpha"].set_index(["model", "reasoning_regime", "budget", "contrast_id"])
    bg = prim[prim.oracle != "alpha"]
    same = sum(np.sign(r.estimate) == np.sign(a.loc[(r.model, r.reasoning_regime, r.budget, r.contrast_id)].estimate) for r in bg.itertuples())
    add("R1", "Evaluation model", "Oracle alpha -> beta, gamma", "T1 (primary beta/gamma rows)", len(bg), int((bg.estimate > 0).sum()),
        int((bg.holm_p < 0.05).sum()), "frozen primary Holm p < .05", "FROZEN",
        f"same sign as alpha in {same} of {len(bg)}; scales differ, magnitudes are not compared across Oracles")
    h = prim[prim.reasoning_regime == "HIGH"]
    add("R2", "Correction scope", "High rows re-corrected in the 60-comparison family", "HIGH_REASONING_SUPPLEMENTAL", 60,
        int((supp[supp.holm_family_id == "HIGH_REASONING_SUPPLEMENTAL"].estimate > 0).sum()),
        int((supp[supp.holm_family_id == "HIGH_REASONING_SUPPLEMENTAL"].holm_p < 0.05).sum()), "60-family Holm p < .05", "FROZEN",
        f"main-Holm significant among the 48 High free-text rows: {int((h.holm_p < 0.05).sum())}; 60-family: {int((h.holm_p_60 < 0.05).sum())}")

    def fam(rid, name, element, aid, rule, ci_rule=True, note=""):
        s = supp[supp.analysis_id == aid]
        sig = int(((s.ci_low > 0) | (s.ci_high < 0)).sum()) if ci_rule else int((s.holm_p < 0.05).sum())
        add(rid, name, element, f"{aid} (V13_SUPPLEMENTAL_RESULTS)", len(s), int((s.estimate > 0).sum()), sig, rule, "FROZEN", note)

    fam("R3", "Generation-setting interaction", "High minus Baseline of the same contrast (B1)", "REASONING_REGIME_INTERACTION",
        "95% interval excludes 0", True, "a difference between conditions is claimed only when the interval excludes 0")
    fam("R4", "Budget interaction", "B1 minus BINF of the same contrast", "BUDGET_INTERACTION", "95% interval excludes 0", True,
        "Baseline/alpha rows; High is in the stage9 budget interaction summary")
    fam("R5", "Execution rule", "Repaired minus Strict of the same contrast (B1)", "EXECUTION_SENSITIVITY", "95% interval excludes 0", True,
        "Repaired is not read as restored intent")
    fam("R6", "Accepted firms only", "ITT -> per-protocol", "PP_CONTRAST", "frozen Holm p < .05", False, "selection into acceptance may bias; N differs by row")
    st = supp[supp.holm_family_id == "STABILITY_SUPPLEMENTAL"]
    add("R7", "Repetition", "Run1 -> Run2 (C4, C4R, C6-E)", "STABILITY_SUPPLEMENTAL; thesis Table 6-8", len(st), int((st.estimate > 0).sum()),
        int((st.holm_p < 0.05).sum()), "frozen Holm p < .05", "FROZEN",
        "rank correlation of per-firm gains across runs is 0.06-0.29 on Oracle-alpha (thesis Sec. 6, repeated runs; not recomputed here)")
    add("R8", "Draft-link rule", "990 of 4,600 firm-conditions with a non-passing C4 draft set to A0", "parent_gate_sensitivity_v13.csv",
        len(parent), int((parent.parent_gate_mean > 0).sum()), "N/A", "none (descriptive)", "DESCRIPTIVE",
        "post-processing rule comparison, not a rerun")
    a3, a2 = r9t[r9t.contrast_id == "A3"], r9t[r9t.contrast_id == "A2"]
    add("R9", "Candidate selection", "free8 -> candidate9 (A2, A3)", "R9_candidate_selection.csv (24 rows)", 24,
        int((r9t.estimate > 0).sum()), int((r9t.holm_p < 0.05).sum()), "Holm over A2/A3 per model x Oracle x regime", "RECOMPUTED",
        f"A3 positive {int((a3.estimate > 0).sum())}/12, significant {int((a3.holm_p < 0.05).sum())}/12 (all Baseline); "
        f"A2 positive {int((a2.estimate > 0).sum())}/12, significant {int((a2.holm_p < 0.05).sum())}/12; A1 and A4 do not exist for candidate selection")
    add("R10", "Reference assignment", "does the C6-EX reference equal the firm's own?", "R10_reference_assignment.csv", len(r10t),
        "N/A", int(r10t.same_action.sum()), "count of firms whose C6-EX reference action equals their own C3-E action", "RECOMPUTED",
        f"self-reference {int(r10t.self_reference.sum())} of 575; same action {int(r10t.same_action.sum())} of 575 ({r10t.same_action.mean():.1%}); one fixed permutation only")
    add("R11", "C3-E selection", "C3-E was chosen on 2024 data", "thesis Sec.5; stage9 metadata", "N/A", "N/A", "N/A", "N/A", "NOTE",
        "2024 is a repeatedly used development surface, not a hold-out; footnote for REF-1 and A3")
    add("R12", "Human baseline", "separate external comparison", "thesis Sec.7", "N/A", "N/A", "N/A", "N/A", "NOTE",
        "56 firms for actions, 53 for scores; not part of the 575-firm ablation tables")
    t4 = pd.DataFrame(rows)
    c.write_csv(t4, c.RESULTS / "T4_robustness.csv")
    return t4


def main():
    t = r9()
    chk = r9_checks(t)
    r10t = r10()
    t4 = summaries(t, r10t)
    return t, chk, r10t, t4


if __name__ == "__main__":
    t, chk, r10t, t4 = main()
    print(json.dumps({k: v for k, v in chk.items() if k != "rows"}, ensure_ascii=False, indent=1))
    bad = [d for d in chk["rows"] if d.get("mean_match_3dp") is False or d.get("star_match") is False]
    print("mismatching rows:", json.dumps(bad, ensure_ascii=False, indent=1))
    print(t4[["check_id", "n_rows", "n_positive", "n_significant"]].to_string(index=False))
    print("R10: self", int(r10t.self_reference.sum()), "same", int(r10t.same_action.sum()))
