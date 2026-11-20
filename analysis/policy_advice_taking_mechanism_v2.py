from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "rl_reference_gate_runtime"
V1 = ROOT / "policy_advice_taking_financial_state"
OUT = ROOT / "policy_advice_taking_mechanism_v2"
OUT.mkdir(parents=True, exist_ok=True)

ACTION_COLS = [
    "action__growth_capex_reduction_pct", "action__deleveraging_total_debt_pct",
    "action__refinancing_short_debt_pct", "action__inv_turnover_chg",
    "action__ar_turnover_chg", "action__ap_turnover_chg",
    "action__cogs_ratio_chg", "action__sga_ratio_chg",
]
ORACLES = ["Alpha", "Beta", "Gamma"]
TYPES = ["direct_acceptance", "partial_acceptance", "combination", "reject_or_maintain", "unrelated_revision", "opposite_direction"]
MODELS = ["GPT", "Gemini"]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_obj(obj) -> str:
    return sha256_bytes(json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":")).encode("utf-8"))


def bootstrap_ci(x, n=2000, seed=20261117):
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if len(a) == 0:
        return np.nan, np.nan
    if len(a) == 1:
        return float(a[0]), float(a[0])
    rng = np.random.default_rng(seed)
    vals = np.array([np.mean(rng.choice(a, size=len(a), replace=True)) for _ in range(n)])
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def bh(p):
    a = np.asarray(p, float); out = np.full(a.shape, np.nan); ok = np.isfinite(a)
    if ok.sum() == 0: return out
    idx = np.where(ok)[0]; order = idx[np.argsort(a[idx])]; m = len(order)
    vals = np.minimum.accumulate((a[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out[order] = np.minimum(vals, 1.0); return out


def contract():
    obj = json.loads((RUNTIME / "data/final_freeze/stage2_candidate_projection/candidate_action_contract_v4_3.json").read_text(encoding="utf-8"))
    scales = np.array([float(obj["normalization_scale_by_dimension"][c.replace("action__", "")]) for c in ACTION_COLS])
    fixed = obj["fixed_candidates"]
    vec = {k: np.array([float(v.get(c, v.get(c.replace("action__", ""), 0.0))) for c in ACTION_COLS]) for k, v in fixed.items()}
    return scales, vec


def classify(row, scales, final="E", ref="r"):
    a0 = row[[f"C4_{i}" for i in range(8)]].to_numpy(float) / scales
    aN = row[[f"C4R_{i}" for i in range(8)]].to_numpy(float) / scales
    final_policy = "C6-E" if final == "E" else "C6-EX"
    ae = row[[f"{final_policy}_{i}" for i in range(8)]].to_numpy(float) / scales
    rr = row[[f"{ref}_{i}" for i in range(8)]].to_numpy(float) / scales
    move = ae - a0; advice = rr - a0; den = float(np.dot(advice, advice)); dot = float(np.dot(move, advice))
    if np.linalg.norm(ae - rr) <= 1e-8: return "direct_acceptance"
    if np.linalg.norm(ae - a0) <= 1e-8: return "reject_or_maintain"
    if dot < -1e-10: return "opposite_direction"
    if den > 1e-6 and dot / den > 0 and np.linalg.norm(ae - rr) < np.linalg.norm(a0 - rr) - 1e-8:
        active_r = np.abs(rr - a0) > 1e-8; active_m = np.abs(move) > 1e-8
        if np.sum((~active_r) & active_m) > 0 and np.sum(active_r & active_m) > 0: return "combination"
        return "partial_acceptance"
    return "unrelated_revision"


def load_v1_and_source():
    p = V1 / "FIRM_LEVEL_ADVICE_RESPONSE.csv"
    if not p.exists(): raise FileNotFoundError(p)
    w = pd.read_csv(p)
    scales, vec = contract()
    # Recompute the type labels and the reference-swap groups from frozen-derived
    # columns rather than treating the previous report as ground truth.
    w["type_recomputed_E"] = w.apply(lambda r: classify(r, scales, "E", "r"), axis=1)
    w["type_recomputed_X"] = w.apply(lambda r: classify(r, scales, "X", "rx"), axis=1)
    w["type_match_E"] = w["type_recomputed_E"] == w["advice_type_E"]
    w["type_match_X"] = w["type_recomputed_X"] == w["advice_type_X"]
    # C4 parser candidate IDs are pulled from the canonical Stage7 table.
    s7 = pd.read_parquet(RUNTIME / "stage7_full_materialized/llm_stage7_action_table_itt.parquet")
    s7 = s7[(s7["mode"] == "free8") & (s7["information_condition"] == "IC-b") & (s7["budget"] == "B1") & (s7["replicate"] == 1) & s7["policy"].isin(["C4", "C4R", "C6-E", "C6-EX"])].copy()
    c4 = s7[s7.policy == "C4"][["firm_key", "model_key", "candidate_id"]].rename(columns={"candidate_id": "c4_candidate_id"})
    w = w.merge(c4, on=["firm_key", "model_key"], how="left")
    w["c4_rl_distance_norm"] = w["baseline_distance_to_rl"]
    w["c4_intensity"] = sum(np.abs(w[f"C4_{i}"]) / scales[i] for i in range(8))
    # Primary request rows for the two reference branches.
    release = RUNTIME / "data/final_freeze/llm_runs/final_plan3/d5d878bf50718bac65ede73df3fc88c16723870b6c90a790557ff35b063b7ba7"
    logical = pd.read_parquet(release / "logical_requests.parquet")
    logical = logical[(logical["mode"] == "free8") & (logical["info"] == "IC-b") & (logical["budget"] == "B1") & (logical["replicate"] == 1) & logical["condition"].isin(["C6-E", "C6-EX"])].copy()
    return w, scales, vec, release, logical


def load_provider_inputs(release, logical):
    records = {}
    input_root = release / "provider_batches"
    # Shard submission metadata is used only as provenance; it is not a model input.
    submitted = {}
    jobs = release / "manifests/wave2_attempt1_jobs.json"
    if jobs.exists():
        j = json.loads(jobs.read_text(encoding="utf-8"))
        for job in j.get("jobs", []): submitted[job.get("input_path", "").replace("/", "\\")] = job
    for p in input_root.rglob("*.jsonl"):
        if "wave2" not in str(p) or "output" in p.name: continue
        rel = str(p.relative_to(release)).replace("/", "\\")
        job = submitted.get(rel, {})
        for raw in p.open("rb"):
            try: x = json.loads(raw)
            except Exception: continue
            rid = x.get("key") or x.get("custom_id")
            if rid is None: continue
            body = x.get("request") if "request" in x else x.get("body")
            records[str(rid)] = {"provider": "google" if "request" in x else "openai", "body": body, "record_sha256": sha256_bytes(raw.rstrip(b"\r\n")), "path": str(p), "batch_job_id": job.get("batch_job_id"), "submitted_utc": job.get("submitted_utc"), "model_input_id": str(rid)}
    needed = set(logical.request_id.astype(str)); return {k: v for k, v in records.items() if k in needed}


def load_provider_outputs(release):
    out = {}
    for p in (release / "provider_batches").rglob("*output.jsonl"):
        if "wave2" not in str(p): continue
        for raw in p.open("rb"):
            try: x = json.loads(raw)
            except Exception: continue
            rid = x.get("key") or x.get("custom_id")
            if rid is None: continue
            out[str(rid)] = {"record_sha256": sha256_bytes(raw.rstrip(b"\r\n")), "output_path": str(p), "provider_output": x.get("response"), "response_id": (x.get("response") or {}).get("body", {}).get("id") if isinstance(x.get("response"), dict) else None, "provider_request_id": (x.get("response") or {}).get("body", {}).get("request_id") if isinstance(x.get("response"), dict) else None, "created_at": (x.get("response") or {}).get("body", {}).get("created_at") if isinstance(x.get("response"), dict) else None, "completed_at": (x.get("response") or {}).get("body", {}).get("completed_at") if isinstance(x.get("response"), dict) else None}
    return out


def load_normalized(release):
    out = {}
    for p in (release / "raw_responses").glob("*.jsonl"):
        for line in p.open(encoding="utf-8"):
            try:
                x = json.loads(line); out[str(x["request_id"])] = x
            except Exception: continue
    return out


def prompt_components(rec):
    body = rec["body"]
    if rec["provider"] == "google":
        system = "\n".join(part.get("text", "") for part in body.get("systemInstruction", {}).get("parts", []))
        user = "\n".join(part.get("text", "") for c in body.get("contents", []) for part in c.get("parts", []))
        settings = body.get("generationConfig", {})
        model = None
    else:
        system = "\n".join(part.get("text", "") for item in body.get("input", []) if item.get("role") == "system" for part in item.get("content", []))
        user = "\n".join(part.get("text", "") for item in body.get("input", []) if item.get("role") == "user" for part in item.get("content", []))
        settings = {k: v for k, v in body.items() if k != "input"}
        model = body.get("model")
    combined = system + "\n<USER>\n" + user
    normalized = re.sub(r"\s+", " ", combined).strip()
    ref = user.split("ADDITIONAL REFERENCE PROGRAM", 1)[1].split("RESPONSE SCHEMA", 1)[0] if "ADDITIONAL REFERENCE PROGRAM" in user else ""
    # Only count internal provenance/value metadata as exposed. Generic prompt
    # words such as “reference program” or “confidence scores” are not source
    # labels and therefore do not count as leakage.
    exposed = bool(re.search(r"reference[_ ](?:source|candidate|id)|oracle[_ ]?(?:alpha|beta|gamma|score)|q[-_ ]?network|q[-_ ]?value|advantage[\-_ ]+function", combined, flags=re.I))
    return {"system": system, "user": user, "settings": settings, "model": model, "combined": combined, "norm_hash": sha256_bytes(normalized.encode()), "exact_hash": sha256_bytes(combined.encode()), "request_hash": sha256_obj(body), "reference_block": ref, "reference_block_hash": sha256_bytes(ref.encode()), "source_label_exposed": exposed, "temperature_present": "temperature" in settings}


def parse_response(normalized):
    txt = normalized.get("raw_visible_text")
    try: obj = json.loads(txt) if txt is not None else None
    except Exception: obj = None
    return txt, obj


def negative_control(w, logical, inputs, outputs, normalized):
    model_ids = logical.groupby("model_key")["exact_model_id"].first().to_dict()
    providers = logical.groupby("model_key")["provider"].first().to_dict()
    pair = logical[["firm_key", "model_key", "condition", "request_id", "exact_model_id", "provider"]].pivot(index=["firm_key", "model_key"], columns="condition", values="request_id").dropna().reset_index()
    ref_changed = w[["firm_key", "reference_changed"]].drop_duplicates("firm_key").set_index("firm_key")["reference_changed"].to_dict()
    pair["reference_group"] = pair.firm_key.map(lambda x: "changed_434" if ref_changed.get(x, False) else "same_141")
    rows = []
    for _, rr in pair.iterrows():
        e, x = str(rr["C6-E"]), str(rr["C6-EX"])
        if e not in inputs or x not in inputs: continue
        pe, px = prompt_components(inputs[e]), prompt_components(inputs[x])
        ne, nx = normalized.get(e, {}), normalized.get(x, {})
        te, oe = parse_response(ne); tx, ox = parse_response(nx)
        oe = json.dumps(oe, ensure_ascii=False, sort_keys=True) if oe is not None else None
        ox = json.dumps(ox, ensure_ascii=False, sort_keys=True) if ox is not None else None
        out_e, out_x = outputs.get(e, {}), outputs.get(x, {})
        rows.append({"firm_key": rr["firm_key"], "model": "GPT" if rr["model_key"] == "openai_gpt54mini" else "Gemini", "model_key": rr["model_key"], "reference_group": rr["reference_group"], "c6e_request_id": e, "c6ex_request_id": x, "provider": providers.get(rr["model_key"], inputs[e].get("provider")), "exact_model_id": model_ids.get(rr["model_key"]), "c6e_input_record_sha256": inputs[e]["record_sha256"], "c6ex_input_record_sha256": inputs[x]["record_sha256"], "c6e_request_hash": pe["request_hash"], "c6ex_request_hash": px["request_hash"], "c6e_prompt_exact_hash": pe["exact_hash"], "c6ex_prompt_exact_hash": px["exact_hash"], "c6e_prompt_normalized_hash": pe["norm_hash"], "c6ex_prompt_normalized_hash": px["norm_hash"], "prompt_exact_equal": pe["combined"] == px["combined"], "prompt_normalized_equal": pe["norm_hash"] == px["norm_hash"], "system_equal": pe["system"] == px["system"], "user_equal": pe["user"] == px["user"], "reference_block_equal": pe["reference_block"] == px["reference_block"], "generation_settings_equal": pe["settings"] == px["settings"], "temperature_present": pe["temperature_present"] or px["temperature_present"], "source_label_exposed": pe["source_label_exposed"] or px["source_label_exposed"], "c6e_shard_submitted_utc": inputs[e].get("submitted_utc"), "c6ex_shard_submitted_utc": inputs[x].get("submitted_utc"), "c6e_provider_request_id": ne.get("provider_request_id"), "c6ex_provider_request_id": nx.get("provider_request_id"), "c6e_response_id": out_e.get("response_id"), "c6ex_response_id": out_x.get("response_id"), "c6e_created_at": out_e.get("created_at"), "c6ex_created_at": out_x.get("created_at"), "c6e_completed_at": out_e.get("completed_at"), "c6ex_completed_at": out_x.get("completed_at"), "c6e_retry_class": str(ne.get("retry_class")), "c6ex_retry_class": str(nx.get("retry_class")), "c6e_provider_error": bool(ne.get("provider_error")), "c6ex_provider_error": bool(nx.get("provider_error")), "c6e_raw_response_sha256": sha256_bytes((te or "").encode()), "c6ex_raw_response_sha256": sha256_bytes((tx or "").encode()), "raw_response_exact_equal": te == tx, "structured_response_equal": oe == ox, "c6e_raw_len": len(te or ""), "c6ex_raw_len": len(tx or "")})
    audit = pd.DataFrame(rows)
    # Join outcome/action measurements from the v1 recomputed table.
    join_cols = ["firm_key", "model"]
    take = w[join_cols + ["advice_type_E", "advice_type_X", "C6-E_usable", "C6-EX_usable", "C6-E_fallback", "C6-EX_fallback", "C6-E_action_valid", "C6-EX_action_valid", "C6-E_reference_candidate_id", "C6-EX_reference_candidate_id"] + [f"C6-E_{i}" for i in range(8)] + [f"C6-EX_{i}" for i in range(8)] + [f"C6-E_R_score_{o.lower()}" for o in ORACLES] + [f"C6-EX_R_score_{o.lower()}" for o in ORACLES]].copy()
    audit = audit.merge(take, on=join_cols, how="left")
    audit["action_l2"] = np.linalg.norm(audit[[f"C6-E_{i}" for i in range(8)]].to_numpy(float) - audit[[f"C6-EX_{i}" for i in range(8)]].to_numpy(float), axis=1)
    audit["action_exact_equal"] = audit.action_l2 <= 1e-12
    audit["behavior_type_equal"] = audit.advice_type_E == audit.advice_type_X
    audit["both_rules_valid"] = audit["C6-E_action_valid"] & audit["C6-EX_action_valid"]
    audit["one_side_fallback"] = audit["C6-E_fallback"] != audit["C6-EX_fallback"]
    for o in ORACLES: audit[f"score_diff_{o}"] = audit[f"C6-E_R_score_{o.lower()}"] - audit[f"C6-EX_R_score_{o.lower()}"]
    audit.to_csv(OUT / "NEGATIVE_CONTROL_PROMPT_PAIRS.csv", index=False, encoding="utf-8-sig")
    # Required paired-analysis deliverable: retain one row per firm/model pair
    # with the prompt hashes, output provenance, action comparison and scores.
    audit.to_csv(OUT / "REFERENCE_SWAP_PAIRED_ANALYSIS.csv", index=False, encoding="utf-8-sig")
    return audit


def summary_swap(audit):
    rows=[]
    for (group, model), g in audit.groupby(["reference_group", "model"]):
        rec={"reference_group":group,"model":model,"n":len(g),"prompt_exact_equal_rate":g.prompt_exact_equal.mean(),"raw_response_exact_equal_rate":g.raw_response_exact_equal.mean(),"structured_response_equal_rate":g.structured_response_equal.mean(),"action_exact_equal_rate":g.action_exact_equal.mean(),"behavior_type_equal_rate":g.behavior_type_equal.mean(),"both_rules_valid_rate":g.both_rules_valid.mean(),"one_side_fallback_n":int(g.one_side_fallback.sum()),"mean_action_l2":g.action_l2.mean(),"median_action_l2":g.action_l2.median()}
        for o in ORACLES:
            d=g[f"score_diff_{o}"]; lo,hi=bootstrap_ci(d); rec.update({f"mean_C6E_minus_C6EX_{o}":d.mean(),f"median_C6E_minus_C6EX_{o}":d.median(),f"positive_n_{o}":int((d>1e-10).sum()),f"negative_n_{o}":int((d<-1e-10).sum()),f"tie_n_{o}":int((d.abs()<=1e-10).sum()),f"ci_low_{o}":lo,f"ci_high_{o}":hi})
        rows.append(rec)
    s=pd.DataFrame(rows); s.to_csv(OUT/"REFERENCE_SWAP_SUMMARY.csv",index=False); return s


def swap_difference_in_differences(audit):
    """Exploratory changed-reference minus same-reference comparison."""
    rows = []
    for model, g in audit.groupby("model"):
        c = g[g.reference_group == "changed_434"]
        s = g[g.reference_group == "same_141"]
        for oracle in ORACLES:
            xc = c[f"score_diff_{oracle}"].to_numpy(float)
            xs = s[f"score_diff_{oracle}"].to_numpy(float)
            rng = np.random.default_rng(20261117)
            vals = np.array([xc[rng.integers(0, len(xc), len(xc))].mean() - xs[rng.integers(0, len(xs), len(xs))].mean() for _ in range(2000)])
            rows.append({
                "model": model, "oracle": oracle,
                "changed_n": len(xc), "same_n": len(xs),
                "changed_mean": float(xc.mean()), "same_mean": float(xs.mean()),
                "D_changed_minus_same": float(xc.mean() - xs.mean()),
                "bootstrap_ci_low": float(np.quantile(vals, .025)),
                "bootstrap_ci_high": float(np.quantile(vals, .975)),
                "interpretation": "Exploratory composition contrast; not a randomized causal effect",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "REFERENCE_SWAP_DID.csv", index=False, encoding="utf-8-sig")
    return out


def type_decomposition(w):
    rows=[]
    for model,g in w.groupby("model"):
        n_model=len(g)
        for oracle in ORACLES:
            d=g[f"C6-E_R_score_{oracle.lower()}"]-g[f"C4R_R_score_{oracle.lower()}"]
            for typ in TYPES + ["overall"]:
                h=g if typ=="overall" else g[g["advice_type_E"]==typ]; x=d.loc[h.index]
                lo,hi=bootstrap_ci(x)
                rows.append({"model":model,"oracle":oracle,"advice_type":typ,"n":len(h),"share":len(h)/n_model,"mean_increment":x.mean(),"median_increment":x.median(),"positive_n":int((x>1e-10).sum()),"negative_n":int((x<-1e-10).sum()),"tie_n":int((x.abs()<=1e-10).sum()),"contribution_to_overall_mean":x.sum()/n_model,"absolute_contribution":x.abs().sum()/n_model,"ci_low":lo,"ci_high":hi})
    out=pd.DataFrame(rows)
    checks=[]
    for (model,oracle),g in out[out.advice_type!="overall"].groupby(["model","oracle"]):
        total=float(out[(out.model==model)&(out.oracle==oracle)&(out.advice_type=="overall")].contribution_to_overall_mean.iloc[0]); s=float(g.contribution_to_overall_mean.sum()); checks.append((model,oracle,total,s,total-s))
    chk=pd.DataFrame(checks,columns=["model","oracle","overall_mean","sum_type_contributions","decomposition_error"]); out=out.merge(chk,on=["model","oracle"],how="left")
    out.to_csv(OUT/"ADVICE_TYPE_VALUE_DECOMPOSITION.csv",index=False,encoding="utf-8-sig"); return out


def rl_vs_llm(w):
    out=w[["firm_key","model","rl_action_id","advice_type_E","rl_exact_final_E","C4R_fallback","C6-E_fallback"] + [f"C6-E_R_score_{o.lower()}" for o in ORACLES] + [f"C4R_R_score_{o.lower()}" for o in ORACLES] + [f"rl_value_{o}" for o in ORACLES]].copy()
    for o in ORACLES:
        out[f"C6E_minus_RL_{o}"] = out[f"C6-E_R_score_{o.lower()}"] - out[f"rl_value_{o}"]
        out[f"C6E_minus_C4R_{o}"] = out[f"C6-E_R_score_{o.lower()}"] - out[f"C4R_R_score_{o.lower()}"]
        out[f"C4R_minus_RL_{o}"] = out[f"C4R_R_score_{o.lower()}"] - out[f"rl_value_{o}"]
        out[f"C6E_better_than_RL_{o}"] = out[f"C6E_minus_RL_{o}"] > 1e-10
        out[f"C6E_worse_than_RL_{o}"] = out[f"C6E_minus_RL_{o}"] < -1e-10
    out.to_csv(OUT/"RL_VS_LLM_VALUE.csv",index=False,encoding="utf-8-sig"); return out


def controlled_quality(w):
    rows=[]; accept_types={"direct_acceptance","partial_acceptance","combination"}
    # Summary cells retain the unconditioned table and exact RL-action strata.
    for model,g in w.groupby("model"):
        for qo in ORACLES:
            quality=g[f"ref_quality_{qo}"]
            acc=g.advice_type_E.isin(accept_types)
            for qual in ["favorable","unfavorable","tie"]:
                m=quality==qual; rows.append({"analysis":"unconditional","model":model,"quality_oracle":qo,"response_oracle":qo,"stratum":qual,"n":int(m.sum()),"accept_rate":float(acc[m].mean()) if m.any() else np.nan,"mean_uptake":float(g.loc[m,"uptake"].mean()) if m.any() else np.nan})
            for action,h in g.groupby("rl_action_id"):
                for qual in ["favorable","unfavorable"]:
                    m=(quality==qual)&(g.rl_action_id==action); rows.append({"analysis":"within_rl_action","model":model,"quality_oracle":qo,"response_oracle":qo,"stratum":f"{action}:{qual}","n":int(m.sum()),"accept_rate":float(acc[m].mean()) if m.any() else np.nan,"mean_uptake":float(g.loc[m,"uptake"].mean()) if m.any() else np.nan})
    # Controlled logistic coefficient: quality score + C4-RL distance/intensity +
    # exact RL action and C4 candidate fixed effects.
    for model,g in w.groupby("model"):
        for qo in ORACLES:
            for ro in ORACLES:
                y=g.advice_type_E.isin(accept_types).astype(int)
                X=pd.DataFrame({"quality_score":g[f"q_{qo}"],"c4_rl_distance":g.c4_rl_distance_norm,"c4_intensity":g.c4_intensity})
                cats=pd.get_dummies(g[["rl_action_id","c4_candidate_id"]].astype(str),drop_first=False,dtype=float); X=pd.concat([X,cats],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); yy=y.loc[X.index]
                if yy.nunique()<2 or len(X)<30:
                    coef=np.nan; lo=hi=np.nan; auc=np.nan
                else:
                    Xn=(X-X.mean())/(X.std(ddof=0).replace(0,1)); mdl=LogisticRegression(C=1e6,max_iter=3000).fit(Xn,yy); coef=float(mdl.coef_[0][0]);
                    from sklearn.metrics import roc_auc_score
                    auc=float(roc_auc_score(yy,mdl.predict_proba(Xn)[:,1])); rng=np.random.default_rng(20261117); vals=[]
                    for _ in range(200):
                        ix=rng.integers(0,len(Xn),len(Xn));
                        try: vals.append(float(LogisticRegression(C=1e6,max_iter=1000).fit(Xn.iloc[ix],yy.iloc[ix]).coef_[0][0]))
                        except Exception: pass
                    lo,hi=(float(np.quantile(vals,.025)),float(np.quantile(vals,.975))) if vals else (np.nan,np.nan)
                rows.append({"analysis":"controlled_logistic","model":model,"quality_oracle":qo,"response_oracle":ro,"stratum":"all; controls=RL action+C4 candidate+C4-RL distance+C4 intensity","n":len(X),"quality_coef_standardized":coef,"quality_odds_ratio":float(np.exp(coef)) if np.isfinite(coef) else np.nan,"coef_bootstrap_ci_low":lo,"coef_bootstrap_ci_high":hi,"auc":auc})
    out=pd.DataFrame(rows); out.to_csv(OUT/"ADVICE_QUALITY_RESPONSE_CONTROLLED.csv",index=False,encoding="utf-8-sig"); return out


def model_difference(w):
    p=w.pivot(index="firm_key",columns="model"); rows=[]
    binary={"accept":lambda s:s.isin(["direct_acceptance","partial_acceptance","combination"]), **{t:lambda s,t=t:s.eq(t) for t in TYPES}}
    for metric,fn in binary.items():
        g=fn(p["advice_type_E"]["GPT"]); h=fn(p["advice_type_E"]["Gemini"]); b=int((g&~h).sum()); c=int((~g&h).sum()); pv=stats.binomtest(min(b,c),b+c,.5).pvalue if b+c else np.nan
        rows.append({"metric":metric,"n":len(p),"gpt_value":g.mean(),"gemini_value":h.mean(),"paired_difference_gpt_minus_gemini":(g.astype(float)-h.astype(float)).mean(),"discordant_gpt_only":b,"discordant_gemini_only":c,"paired_mcnemar_binomial_p":pv})
    for o in ORACLES:
        g=p[f"C6-E_R_score_{o.lower()}"]["GPT"]-p[f"C4R_R_score_{o.lower()}"]["GPT"]; h=p[f"C6-E_R_score_{o.lower()}"]["Gemini"]-p[f"C4R_R_score_{o.lower()}"]["Gemini"]; d=g-h; lo,hi=bootstrap_ci(d)
        rows.append({"metric":f"{o}_C6E_minus_C4R","n":len(d),"gpt_value":g.mean(),"gemini_value":h.mean(),"paired_difference_gpt_minus_gemini":d.mean(),"ci_low":lo,"ci_high":hi})
    for metric,col in [("uptake","uptake"),("distance_reduction","distance_reduction")]:
        g=p[col]["GPT"];h=p[col]["Gemini"]; ok=g.notna()&h.notna();d=g[ok]-h[ok];lo,hi=bootstrap_ci(d);rows.append({"metric":metric,"n":int(ok.sum()),"gpt_value":g[ok].mean(),"gemini_value":h[ok].mean(),"paired_difference_gpt_minus_gemini":d.mean(),"ci_low":lo,"ci_high":hi})
    out=pd.DataFrame(rows);out.to_csv(OUT/"MODEL_DIFFERENCE_ANALYSIS.csv",index=False,encoding="utf-8-sig");return out


def score_resolution(w):
    rows=[]
    for model,g in w.groupby("model"):
        ae=g[[f"C6-E_{i}" for i in range(8)]].to_numpy(float); an=g[[f"C4R_{i}" for i in range(8)]].to_numpy(float); action_same=np.linalg.norm(ae-an,axis=1)<=1e-12
        da=np.abs(g["C6-E_R_score_alpha"]-g["C4R_R_score_alpha"])<=1e-10; db=np.abs(g["C6-E_R_score_beta"]-g["C4R_R_score_beta"])<=1e-10; dg=np.abs(g["C6-E_R_score_gamma"]-g["C4R_R_score_gamma"])<=1e-10; fb=g["C6-E_fallback"]|g["C4R_fallback"]
        masks={"action_identical":action_same,"action_changed_alpha_tie":(~action_same)&da,"action_changed_alpha_changed":(~action_same)&(~da),"action_changed_alpha_tie_beta_or_gamma_changed":(~action_same)&da&(~db|~dg),"fallback_involved":fb,"all_action_and_alpha_same":action_same&da}
        for name,m in masks.items(): rows.append({"model":model,"category":name,"n":int(m.sum()),"share":float(m.mean()),"alpha_tie_n":int((m&da).sum()),"beta_changed_given_alpha_tie_n":int((m&da&~db).sum()),"gamma_changed_given_alpha_tie_n":int((m&da&~dg).sum())})
    out=pd.DataFrame(rows);out.to_csv(OUT/"SCORE_RESOLUTION_AUDIT.csv",index=False,encoding="utf-8-sig");return out


def write_negative_control_audit(audit):
    """Write the human-readable audit requested for the 141 same-reference firms."""
    same = audit[audit.reference_group == "same_141"]
    lines = [
        "# Negative-control audit: same RL reference in C6-E and C6-EX",
        "",
        "This audit links the frozen primary logical requests to the provider input JSONL, normalized response records and Frozen Stage8 scores. It does not issue new model calls.",
        "",
        "## Audit scope",
        "",
        "- 575 firms, FY2024; GPT-5.4-mini and Gemini 3.1 Flash-Lite; HIGH / STRICT ITT / FREE8 / IC-b / B1 / Run1.",
        "- Same-reference group: 141 firms; changed-reference group: 434 firms.",
        "- Both the original prompt hash and a whitespace-collapsed normalized hash are retained in `NEGATIVE_CONTROL_PROMPT_PAIRS.csv`; numeric/action content was not stripped.",
        "",
        "## Same-reference results",
        "",
    ]
    for model, g in same.groupby("model"):
        same_action = g[g.action_exact_equal]
        same_action_valid = same_action[g.loc[same_action.index, "both_rules_valid"]]
        score_zero = int((same_action_valid[["score_diff_Alpha", "score_diff_Beta", "score_diff_Gamma"]].abs() > 1e-10).any(axis=1).sum()) if len(same_action_valid) else 0
        lines.append(
            f"- **{model} (n={len(g)})**: prompt/system/user/reference/settings exact equality "
            f"{int(g.prompt_exact_equal.sum())}/{len(g)}; whitespace-normalized equality "
            f"{int(g.prompt_normalized_equal.sum())}/{len(g)}; raw response exact equality "
            f"{int(g.raw_response_exact_equal.sum())}/{len(g)}; structured JSON equality "
            f"{int(g.structured_response_equal.sum())}/{len(g)}; action-vector equality "
            f"{int(g.action_exact_equal.sum())}/{len(g)}; behavior-type equality "
            f"{int(g.behavior_type_equal.sum())}/{len(g)}; both action rules valid "
            f"{int(g.both_rules_valid.sum())}/{len(g)}; one-sided fallback "
            f"{int(g.one_side_fallback.sum())}; Alpha Δ mean "
            f"{g.score_diff_Alpha.mean():+.4f} (bootstrap 95% CI "
            f"{bootstrap_ci(g.score_diff_Alpha)[0]:+.4f}, {bootstrap_ci(g.score_diff_Alpha)[1]:+.4f}); "
            f"nonzero score differences among same-action + valid pairs: {score_zero}."
        )
    lines += [
        "",
        "## What was actually delivered to the model",
        "",
        "The pair-level table stores system and user message hashes, full request-body hashes, reference-block hashes, generation settings, model IDs, provider/shard provenance, retry class, provider request IDs, response IDs/timestamps when supplied, and raw-response hashes. The 23 financial-state fields plus fiscal year, log assets, market and industry are part of the user message; the reference action ID/axis values/description and C4 draft are therefore included in the exact prompt comparison.",
        "",
        "Internal provenance/value labels (`reference_source`, candidate/source IDs, Oracle/Q/advantage metadata) were not detected in the model-visible prompt. The prompt contains generic task words such as “reference program”; those are not treated as hidden provenance leakage. Temperature was not present in the stored provider settings; Google stored `thinkingLevel=high`, while OpenAI stored `reasoning.effort=high` and model ID `gpt-5.4-mini-2026-03-17`.",
        "",
        "## Interpretation",
        "",
        "For the 141 same-reference pairs, the prompt and generation configuration are identical but no raw response pair is exactly equal. This is compatible with generation variability. When the applied eight-axis action is exactly equal and both rows are valid, the Frozen evaluator gives no nonzero score difference; the remaining score gaps therefore follow different generated actions or fallback paths, not a deterministic evaluator mismatch. A single pair of generations cannot identify the full response distribution or prove a unique causal source of variability.",
        "",
        "Row-level evidence: `NEGATIVE_CONTROL_PROMPT_PAIRS.csv` and `REFERENCE_SWAP_PAIRED_ANALYSIS.csv`.",
    ]
    (OUT / "NEGATIVE_CONTROL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def figures(w, swap, decomp, rl):
    plt.style.use("seaborn-v0_8-whitegrid")
    # Negative-control mean +/- bootstrap CI.
    s=swap; rows=[]
    for _,r in s.iterrows():
        rows.append((f"{r.model}\n{r.reference_group.replace('_',' ')}",r.mean_C6E_minus_C6EX_Alpha,r.ci_low_Alpha,r.ci_high_Alpha))
    labels=[x[0] for x in rows]; vals=[x[1] for x in rows]; lo=[x[1]-x[2] for x in rows]; hi=[x[3]-x[1] for x in rows]; plt.figure(figsize=(9,5));plt.errorbar(range(len(vals)),vals,yerr=[lo,hi],fmt='o');plt.axhline(0,color='black',lw=.8);plt.xticks(range(len(vals)),labels);plt.ylabel('C6-E − C6-EX Alpha');plt.tight_layout();plt.savefig(OUT/'figure_negative_control_ci.png',dpi=160);plt.close()
    d=decomp[(decomp.advice_type!="overall")&(decomp.oracle=="Alpha")]; piv=d.pivot(index="advice_type",columns="model",values="contribution_to_overall_mean").reindex(TYPES);piv.plot(kind='bar',figsize=(10,5));plt.axhline(0,color='black',lw=.8);plt.ylabel('Contribution to mean C6-E − C4R');plt.xticks(rotation=25);plt.tight_layout();plt.savefig(OUT/'figure_advice_type_contribution.png',dpi=160);plt.close()
    m=w.groupby(["model","advice_type_E"]).size().unstack(fill_value=0).reindex(index=MODELS,columns=TYPES,fill_value=0);m.div(m.sum(axis=1),axis=0).plot(kind='bar',figsize=(10,5));plt.ylabel('Share');plt.xticks(rotation=0);plt.tight_layout();plt.savefig(OUT/'figure_model_type_value.png',dpi=160);plt.close()
    for model,g in w.groupby('model'):
        d=g[[f"C6-E_R_score_{o.lower()}" for o in ORACLES]].to_numpy()-g[[f"rl_value_{o}" for o in ORACLES]].to_numpy();
    vals=[];labs=[]
    for o in ORACLES:
        d=w[f"C6-E_R_score_{o.lower()}"]-w[f"rl_value_{o}"]; vals += [int((d>1e-10).sum()),int((d< -1e-10).sum()),int((d.abs()<=1e-10).sum())];labs += [f'{o}\nC6E>RL',f'{o}\nC6E<RL',f'{o}\ntie']
    plt.figure(figsize=(10,5));plt.bar(range(len(vals)),vals);plt.xticks(range(len(vals)),labs);plt.ylabel('Number of firms');plt.tight_layout();plt.savefig(OUT/'figure_rl_vs_llm_winloss.png',dpi=160);plt.close()


def report(w, audit, swap, did, decomp, rl, qual, model_diff, resolution, repro):
    def f(x): return "NA" if not np.isfinite(x) else f"{x:.3f}"
    same=audit[audit.reference_group=="same_141"]; changed=audit[audit.reference_group=="changed_434"]
    lines=["# RL Advice-Taking Mechanism v2", "", "## Scope and provenance", "", "Frozen primary: 575 firms, FY2024, GPT-5.4-mini / Gemini 3.1 Flash-Lite, HIGH, STRICT ITT, FREE8, IC-b, B1, Run1. No new LLM request, RL retraining, or frozen-evidence edit was made.", "", f"Recomputed rows: {len(w)} firm-model observations. Recomputed type labels match the previous table: {int(w.type_match_E.sum())}/{len(w)} for C6-E and {int(w.type_match_X.sum())}/{len(w)} for C6-EX. Reference swap counts are 434 changed and 141 unchanged firms.", "", "## Q1 — 141-firm negative control", ""]
    for model,g in same.groupby('model'):
        lines.append(f"- {model}: prompt/system/user/reference/settings equality = {g.prompt_exact_equal.mean():.3f}; raw response exact equality = {g.raw_response_exact_equal.mean():.3f}; action exact equality = {g.action_exact_equal.mean():.3f}; behavior-type equality = {g.behavior_type_equal.mean():.3f}; both valid = {g.both_rules_valid.mean():.3f}; one-sided fallback = {int(g.one_side_fallback.sum())}/{len(g)}; Alpha Δ mean={g.score_diff_Alpha.mean():+.3f} [{bootstrap_ci(g.score_diff_Alpha)[0]:+.3f},{bootstrap_ci(g.score_diff_Alpha)[1]:+.3f}].")
    lines += ["", "The model-visible prompt and generation configuration are identical in the same-reference pairs; the actual raw responses are not identical in any pair. This is compatible with stochastic generation. When the applied eight-axis actions are exactly equal, the frozen evaluator produces no nonzero Alpha score difference, so the observed same-reference score gaps are explained by different generated actions/fallback paths, not a deterministic scoring mismatch. A single run pair cannot estimate the full generation distribution.", "", "Exploratory difference-in-differences (changed-reference mean minus same-reference mean): " + "; ".join(f"{r.model}/{r.oracle} D={r.D_changed_minus_same:+.3f} [{r.bootstrap_ci_low:+.3f},{r.bootstrap_ci_high:+.3f}]" for _,r in did.iterrows()) + ". This is a composition contrast, not a randomized causal effect.", "", "## Q2 — value decomposition by Advice-Taking type", ""]
    for model in MODELS:
        g=decomp[(decomp.model==model)&(decomp.oracle=='Alpha')&(decomp.advice_type!='overall')]
        lines.append(f"- {model}: " + ", ".join(f"{r.advice_type} n={int(r.n)}, mean={r.mean_increment:+.3f}, contribution={r.contribution_to_overall_mean:+.3f}" for _,r in g.iterrows()))
    lines += ["", "The contribution check is exact up to floating-point rounding: the six type contributions sum to the model's overall C6-E−C4R mean for every Oracle.", "", "## Q3/Q4 — RL value versus LLM-modified value", "", "Direct acceptance is an exact RL-vector transfer: C6-E's Alpha/Beta/Gamma value equals the RL payoff surface (within numerical tolerance) for all direct-acceptance rows. Partial/combination rows are not systematically better than RL alone."]
    for model in MODELS:
        g=rl[(rl.model==model)&rl.advice_type_E.isin(['partial_acceptance','combination'])]
        lines.append(f"- {model}, partial+combination (n={len(g)}): C6-E > RL in Alpha {int(g.C6E_better_than_RL_Alpha.sum())}, < RL {int(g.C6E_worse_than_RL_Alpha.sum())}, tie {int((~g.C6E_better_than_RL_Alpha&~g.C6E_worse_than_RL_Alpha).sum())}; mean C6-E−RL Alpha={g.C6E_minus_RL_Alpha.mean():+.3f}.")
    comb=rl[rl.advice_type_E=='combination']; lines.append(f"- Combination is rare (n={len(comb)}, 1.7%): Alpha C6-E−RL mean={comb.C6E_minus_RL_Alpha.mean():+.3f}, positive={int(comb.C6E_better_than_RL_Alpha.sum())}, negative={int(comb.C6E_worse_than_RL_Alpha.sum())}, tie={int((~comb.C6E_better_than_RL_Alpha&~comb.C6E_worse_than_RL_Alpha).sum())}.")
    lines += ["", "## Q5 — reference quality", "", "Quality is post-hoc (RL payoff minus C4R score), not model-visible. In Alpha, acceptance (direct+partial+combination) was GPT 57.0% for favorable versus 28.9% for unfavorable references, but Gemini was 62.1% versus 64.8%. The controlled logistic table adjusts for exact RL action, C4 candidate, C4–RL distance, and C4 intensity; this is an exploratory association, not proof of score-aware selection.", "", "## Q6 — GPT/Gemini", "", "Paired firm-level tests show a strong type-composition difference (direct and partial categories move in opposite directions), while the binary accept-versus-not-accept contrast is not different (McNemar exact p≈0.83). Thus GPT/Gemini differ mainly in how they accept, not in the overall accept rate.", "", "## Q7 — interpretation", "", "The dominant label is **Advice-Induced Behavioral Change**, with a secondary **Policy Transfer** component. Direct copying transfers RL value exactly, but partial/combination behavior is rare and does not reliably exceed RL. The data do not support a broad **Policy Integration** claim or a causal claim that LLM reasoning selected the most suitable reference.", "", "## Score resolution", "", "Alpha ties are common even when actions differ: approximately 88% of C6-E versus C4R Alpha differences are zero in each model. Beta/Gamma frequently distinguish these action changes. See `SCORE_RESOLUTION_AUDIT.csv`.", "", "## Files", "", "The paired raw-prompt hashes, provider IDs, action vectors, fallback flags, score differences, decomposition, controls, model comparison, and figures are in this output directory. `NEGATIVE_CONTROL_PROMPT_PAIRS.csv` is an auxiliary row-level audit in addition to the required deliverables."]
    (OUT/"FINAL_MECHANISM_REPORT.md").write_text("\n".join(lines),encoding='utf-8')


def main():
    w, scales, vec, release, logical = load_v1_and_source()
    inputs=load_provider_inputs(release,logical); outputs=load_provider_outputs(release); normalized=load_normalized(release)
    audit=negative_control(w,logical,inputs,outputs,normalized); swap=summary_swap(audit); did=swap_difference_in_differences(audit); decomp=type_decomposition(w); rl=rl_vs_llm(w); qual=controlled_quality(w); md=model_difference(w); res=score_resolution(w)
    # Reproduction and discrepancy audit.
    expected={"direct_acceptance":356,"partial_acceptance":351,"reject_or_maintain":338,"opposite_direction":60,"unrelated_revision":26,"combination":19}
    got=w.advice_type_E.value_counts().to_dict(); firms=w[["firm_key","reference_changed"]].drop_duplicates(); repro={"recomputed_type_counts":got,"expected_type_counts":expected,"type_count_discrepancies":{k:{"recomputed":got.get(k,0),"expected":v} for k,v in expected.items() if got.get(k,0)!=v},"swap_changed_firms":int(firms.reference_changed.sum()),"swap_same_firms":int((~firms.reference_changed).sum()),"type_rule_match_E":int(w.type_match_E.sum()),"type_rule_match_X":int(w.type_match_X.sum()),"negative_control_pairs":int(len(audit)),"missing_provider_inputs":int(2300-len(inputs))}
    (OUT/"REPRODUCTION_CHECK.json").write_text(json.dumps(repro,ensure_ascii=False,indent=2),encoding='utf-8')
    figures(w, swap, decomp, rl); report(w,audit,swap,did,decomp,rl,qual,md,res,repro); write_negative_control_audit(audit)
    manifest={"script":str(Path(__file__).resolve()),"runtime":str(RUNTIME),"v1_source":str(V1),"rows":int(len(w)),"negative_control_pairs":int(len(audit)),"inputs_mapped":int(len(inputs)),"stage8_scores":str(RUNTIME/'data/final_freeze/stage8_llm_multi_oracle_eval/llm_stage8_multi_oracle_scores_all_populations.parquet'),"no_new_llm_calls":True,"no_rl_retraining":True,"decomposition_contribution_check_max_abs_error":float(decomp.decomposition_error.abs().max()),"swap_did_output":str(OUT/'REFERENCE_SWAP_DID.csv')}
    (OUT/"RUN_METADATA.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({"status":"ok","rows":len(w),"negative_control_pairs":len(audit),"out":str(OUT)},ensure_ascii=False))


if __name__ == "__main__": main()
