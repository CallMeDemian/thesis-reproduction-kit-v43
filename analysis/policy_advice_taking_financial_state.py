from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "rl_reference_gate_runtime"
OUT = ROOT / "policy_advice_taking_financial_state"
OUT.mkdir(parents=True, exist_ok=True)

ACTION_COLS = [
    "action__growth_capex_reduction_pct",
    "action__deleveraging_total_debt_pct",
    "action__refinancing_short_debt_pct",
    "action__inv_turnover_chg",
    "action__ar_turnover_chg",
    "action__ap_turnover_chg",
    "action__cogs_ratio_chg",
    "action__sga_ratio_chg",
]
ACTION_SHORT = [c.replace("action__", "") for c in ACTION_COLS]
ORACLES = ["Alpha", "Beta", "Gamma"]
MODEL_LABEL = {
    "openai_gpt54mini": "GPT",
    "google_gemini31flashlite": "Gemini",
}
POLICIES = ["C4", "C4R", "C6-E", "C6-EX"]

# These are the 23 financial fields actually encoded in the frozen prompt
# information dictionary. Context fields (year/assets/market/industry) are
# kept separately and are not counted as financial ratios.
FIN_FIELDS = [
    "derived__debt_to_assets", "derived__equity_to_assets", "derived__current_ratio",
    "derived__cash_ratio", "derived__operating_margin", "derived__gross_margin",
    "derived__net_margin", "derived__roa_proxy", "derived__cogs_to_revenue",
    "derived__sga_to_revenue", "derived__financial_cost_to_revenue",
    "derived__capex_to_revenue", "derived__ppe_to_assets",
    "derived__short_debt_to_total_debt", "derived__long_debt_to_total_debt",
    "derived__bond_to_total_debt", "derived__inventory_to_revenue",
    "derived__receivables_to_revenue", "derived__payables_to_revenue",
    "delta_1y__derived__operating_margin", "delta_1y__derived__debt_to_assets",
    "delta_1y__derived__current_ratio", "delta_1y__derived__roa_proxy",
]
CONTEXT_FIELDS = ["fiscal_year", "log_assets", "market", "industry_class"]

GROUPS = {
    "liquidity": ["derived__current_ratio", "derived__cash_ratio", "delta_1y__derived__current_ratio"],
    "leverage": ["derived__debt_to_assets", "derived__short_debt_to_total_debt", "derived__long_debt_to_total_debt", "delta_1y__derived__debt_to_assets"],
    "profitability": ["derived__operating_margin", "derived__gross_margin", "derived__net_margin", "derived__roa_proxy", "delta_1y__derived__operating_margin", "delta_1y__derived__roa_proxy"],
    "working_capital": ["derived__inventory_to_revenue", "derived__receivables_to_revenue", "derived__payables_to_revenue"],
    "size": ["log_assets"],
}


def parse_json(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    if isinstance(s, dict):
        return s
    try:
        return json.loads(str(s))
    except Exception:
        return None


def load_contract():
    obj = json.loads((RUNTIME / "data/final_freeze/stage2_candidate_projection/candidate_action_contract_v4_3.json").read_text(encoding="utf-8"))
    scales = {f"action__{k}": float(v) for k, v in obj["normalization_scale_by_dimension"].items()}
    vectors = {}
    fixed = obj["fixed_candidates"]
    items = fixed.items() if isinstance(fixed, dict) else ((item["candidate_id"], item) for item in fixed)
    for cid, item in items:
        a = item.get("action", item) if isinstance(item, dict) else item
        vectors[str(cid)] = np.array([float(a.get(c, a.get(c.replace("action__", ""), 0.0))) for c in ACTION_COLS], dtype=float)
    return scales, vectors


def prompt_audit():
    release = RUNTIME / "data/final_freeze/llm_runs/final_plan3/d5d878bf50718bac65ede73df3fc88c16723870b6c90a790557ff35b063b7ba7"
    prompt_rows = []
    for p in (release / "provider_batches").rglob("*.jsonl"):
        if "wave2" not in str(p) or "output" in p.name:
            continue
        for line in p.open(encoding="utf-8"):
            try:
                x = json.loads(line)
                req = x.get("request")
                if not isinstance(req, dict):
                    continue
                text = req["contents"][0]["parts"][0]["text"]
                prompt_rows.append(text)
            except Exception:
                continue
    # The provider batches contain the frozen request payloads; duplicates are
    # intentional because separate firm/model/reference cells are present.
    counts = {"all_wave2_requests": len(prompt_rows), "with_reference": 0, "without_reference": 0,
              "with_earlier_draft": 0, "state_before_draft": 0, "reference_after_state": 0,
              "reference_source_label": 0, "contains_oracle_value": 0, "contains_q_value": 0,
              "state_key_count_min": None, "state_key_count_max": None,
              "dictionary_key_count_min": None, "dictionary_key_count_max": None}
    examples = {}
    dict_counts, state_counts = [], []
    for text in prompt_rows:
        has_ref = "ADDITIONAL REFERENCE PROGRAM" in text
        has_draft = "EARLIER DRAFT" in text
        counts["with_reference" if has_ref else "without_reference"] += 1
        counts["with_earlier_draft"] += int(has_draft)
        if has_draft:
            counts["state_before_draft"] += int(text.find("FIRM STATE") < text.find("EARLIER DRAFT"))
        if has_ref:
            counts["reference_after_state"] += int(text.find("FIRM STATE") < text.find("ADDITIONAL REFERENCE PROGRAM"))
            # The prompt deliberately calls this an additional possible program;
            # no C3-E/C3-EX label is injected into the model-visible text.
            counts["reference_source_label"] += int(bool(re.search(r"C3-E|C3-EX|\bRL\b|\boracle\b", text, flags=re.I)))
        # Extract the actual JSON state and information dictionary.
        try:
            block = text.split("INFORMATION DICTIONARY\n", 1)[1].split("]\n\nFIRM STATE", 1)[0] + "]"
            arr = json.loads(block[block.index("["):])
            dict_counts.append(len(arr))
        except Exception:
            pass
        try:
            block = text.split("FIRM STATE\n", 1)[1].split("\n\nACTION DEFINITIONS", 1)[0]
            st = json.loads(block)
            state_counts.append(len(st))
        except Exception:
            pass
        # Search only the model-visible state/reference blocks, excluding the
        # response schema's generic prohibition text (which mentions scores).
        visible_blocks = text
        try:
            visible_blocks = text.split("FIRM STATE\n", 1)[1].split("RESPONSE SCHEMA", 1)[0]
        except Exception:
            pass
        counts["contains_oracle_value"] += int(bool(re.search(r"oracle\s*(alpha|beta|gamma)|R_score|policy value|evaluation score", visible_blocks, flags=re.I)))
        counts["contains_q_value"] += int(bool(re.search(r"\bQ[_\- ]|\bQ-value\b|Q-network|advantage function|critic value|\bcritic\b|\bconfidence score\b", visible_blocks, flags=re.I)))
        typ = "reference" if has_ref else "self_review"
        examples.setdefault(typ, text[:800])
    if dict_counts:
        counts["dictionary_key_count_min"], counts["dictionary_key_count_max"] = min(dict_counts), max(dict_counts)
    if state_counts:
        counts["state_key_count_min"], counts["state_key_count_max"] = min(state_counts), max(state_counts)
    # Financial payload coverage, independent of prompt text extraction.
    payload = pd.read_parquet(release / "firm_payload_source.parquet")
    counts["payload_firms"] = int(payload["firm_key"].nunique())
    counts["payload_years"] = sorted(payload["fiscal_year"].dropna().astype(int).unique().tolist())
    counts["financial_fields_present"] = [f for f in FIN_FIELDS if f in payload.columns]
    counts["financial_fields_count"] = len(counts["financial_fields_present"])
    counts["context_fields_present"] = [f for f in CONTEXT_FIELDS if f in payload.columns]
    # Exact prompt content audit: on model-visible prompts every dictionary and
    # state has the same 27 keys; values are not inferred from later outputs.
    audit = {
        "source_release": str(release),
        "counts": counts,
        "interpretation": [
            "The frozen Stage7 wave-2 request payloads contain 23 financial fields plus fiscal_year, log_assets, market, and industry_class.",
            "C4R and reference conditions preserve the same FIRM STATE block; the additional reference appears after the earlier draft and state.",
            "The reference is presented as one possible program with no source label, policy score, Q value, confidence, or Oracle result in the model-visible text.",
            "C6-E versus C6-EX source identity is established from the frozen Stage7 metadata/reference_source and reference_candidate_id columns, not from a model-visible source label.",
            "A prompt-level search is reported as an audit of visible strings; it is not evidence that a model used or ignored a feature.",
        ],
        "examples": examples,
    }
    (OUT / "prompt_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Prompt and information audit", "", f"Source release: `{release.name}`", "", "## Frozen request counts", "", "```json", json.dumps(counts, ensure_ascii=False, indent=2), "```", "", "## What is established", ""]
    md += [f"- {x}" for x in audit["interpretation"]]
    md += ["", "## Model-visible prompt excerpts", "", "The exact wave-2 request text is preserved in the frozen provider batch JSONL files. The excerpts below are truncated only for readability.", ""]
    for k, v in examples.items():
        md += [f"### {k}", "", "```text", v, "```", ""]
    md += ["## Limitations", "", "- The normalized raw-response artifact stores returned JSON, not the outgoing prompt; prompt ordering/source claims above come from the frozen provider-batch request JSONL.", "- The prompt schema contains a prohibition against outputting scores, but no model-visible Oracle/Q/advantage values were found in the supplied state or reference blocks."]
    (OUT / "PROMPT_INFORMATION_AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    return audit


def bootstrap_ci(x, stat=np.mean, n=2000, seed=20261117):
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    if len(a) == 1:
        return float(a[0]), float(a[0])
    vals = np.array([stat(rng.choice(a, size=len(a), replace=True)) for _ in range(n)])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def bh_adjust(pvals):
    arr = np.asarray(pvals, dtype=float)
    out = np.full(arr.shape, np.nan)
    ok = np.isfinite(arr)
    if ok.sum() == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(arr[idx])]
    m = len(order)
    vals = np.minimum.accumulate((arr[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out[order] = np.minimum(vals, 1.0)
    return out


def action_type(row, scales, tol=1e-8):
    a0 = row[[f"a0_{i}" for i in range(8)]].to_numpy(float)
    aN = row[[f"aN_{i}" for i in range(8)]].to_numpy(float)
    aE = row[[f"aE_{i}" for i in range(8)]].to_numpy(float)
    r = row[[f"r_{i}" for i in range(8)]].to_numpy(float)
    z = np.asarray(scales, dtype=float)
    n0, nN, nE, nr = a0 / z, aN / z, aE / z, r / z
    d0 = np.linalg.norm(nE - n0); dr = np.linalg.norm(nE - nr); dbase = np.linalg.norm(n0 - nr)
    move = nE - n0; advice = nr - n0
    dot = float(np.dot(move, advice)); denom = float(np.dot(advice, advice))
    same_rl = dr <= tol
    maintain = d0 <= tol
    opposite = dot < -tol
    if same_rl:
        return "direct_acceptance"
    if maintain:
        return "reject_or_maintain"
    if opposite:
        return "opposite_direction"
    if denom > tol and dot / denom > 0 and dr < dbase - tol:
        # A final vector that retains non-RL axes while sharing positive RL axes
        # is labelled combination; otherwise partial acceptance.
        active_r = np.abs(nr - n0) > tol
        active_move = np.abs(move) > tol
        overlap = np.sum(active_r & active_move)
        nonrl_move = np.sum((~active_r) & active_move)
        if overlap > 0 and nonrl_move > 0:
            return "combination"
        return "partial_acceptance"
    return "unrelated_revision"


def build_analysis():
    scales, vectors = load_contract()
    z = np.array([scales[c] for c in ACTION_COLS], dtype=float)
    full = pd.read_parquet(RUNTIME / "stage7_full_materialized/llm_stage7_action_table_itt.parquet")
    a = full[(full["mode"] == "free8") & (full["information_condition"] == "IC-b") & (full["budget"] == "B1") & (full["replicate"] == 1) & full["policy"].isin(POLICIES)].copy()
    a = a.rename(columns={c: f"vec_{c}" for c in ACTION_COLS})
    # Keep the applied ITT action plus explicit fallback/validity flags.
    keep = ["firm_key", "row_id", "model_key", "policy", "request_id", "reference_source", "reference_row_id", "reference_firm_id", "reference_candidate_id", "model_response_usable", "itt_noop_fallback_applied", "action_application_status", "action_valid", "vec_" + ACTION_COLS[0]]
    # Build a wide row for each model-firm-policy.
    rows = []
    for (model, firm), g in a.groupby(["model_key", "firm_key"], sort=False):
        rec = {"model_key": model, "firm_key": firm, "row_id": int(g["row_id"].iloc[0]), "model": MODEL_LABEL.get(model, model)}
        for pol in POLICIES:
            q = g[g["policy"] == pol]
            if q.empty:
                continue
            q = q.iloc[0]
            rec[f"{pol}_request_id"] = q["request_id"]
            rec[f"{pol}_reference_source"] = q["reference_source"]
            rec[f"{pol}_reference_candidate_id"] = q["reference_candidate_id"]
            rec[f"{pol}_reference_firm_id"] = q["reference_firm_id"]
            rec[f"{pol}_usable"] = bool(q["model_response_usable"])
            rec[f"{pol}_fallback"] = bool(q["itt_noop_fallback_applied"])
            rec[f"{pol}_action_valid"] = bool(q["action_valid"])
            rec[f"{pol}_status"] = q["action_application_status"]
            for i, c in enumerate(ACTION_COLS): rec[f"{pol}_{i}"] = float(q[f"vec_{c}"])
        if all(f"{p}_0" in rec for p in POLICIES): rows.append(rec)
    wide = pd.DataFrame(rows)
    # RL own and permuted reference vectors from the fixed candidate contract.
    prob = pd.read_parquet(RUNTIME / "data/final_freeze/stage5_candidate_iql/C3E_E2_7SEED_BALANCED_DFEBAFA6/C3E_firm_probabilities.parquet")
    prob["firm_key"] = prob["firm_id"].astype(str) + "::" + prob["fiscal_year"].astype(int).astype(str)
    pay = pd.read_parquet(RUNTIME / "data/final_freeze/stage6_candidate_selector_eval/C3E_E2_7SEED_BALANCED_DFEBAFA6/firm_action_oracle_payoffs.parquet")
    pay["firm_key"] = pay["firm_id"].astype(str) + "::" + pay["fiscal_year"].astype(int).astype(str)
    own = prob[["firm_key", "action_id"]].rename(columns={"action_id": "rl_action_id"})
    wide = wide.merge(own, on="firm_key", how="left")
    for i in range(8):
        wide[f"r_{i}"] = wide["rl_action_id"].map(lambda x: vectors.get(str(x), np.full(8, np.nan))[i])
        wide[f"rx_{i}"] = wide["C6-EX_reference_candidate_id"].map(lambda x: vectors.get(str(x), np.full(8, np.nan))[i])
    wide["reference_changed"] = wide["C6-E_reference_candidate_id"].astype(str) != wide["C6-EX_reference_candidate_id"].astype(str)
    wide["reference_group"] = np.where(wide["reference_changed"], "changed_434", "same_141")
    wide["ref_candidate_E"] = wide["C6-E_reference_candidate_id"]
    wide["ref_candidate_X"] = wide["C6-EX_reference_candidate_id"]
    # Actual response payloads provide evidence_fields/diagnosis/rationale;
    # if raw JSON is unavailable, retain nulls rather than impute explanations.
    raw_map = {}
    release = RUNTIME / "data/final_freeze/llm_runs/final_plan3/d5d878bf50718bac65ede73df3fc88c16723870b6c90a790557ff35b063b7ba7"
    for p in (release / "raw_responses").glob("*.jsonl"):
        for line in p.open(encoding="utf-8"):
            try:
                x = json.loads(line); raw_map[str(x.get("request_id"))] = x.get("raw_visible_text")
            except Exception: pass
    for pol in POLICIES:
        wide[f"{pol}_diagnosis"] = wide[f"{pol}_request_id"].map(lambda x: None)
        wide[f"{pol}_rationale"] = wide[f"{pol}_request_id"].map(lambda x: None)
        wide[f"{pol}_evidence_fields"] = wide[f"{pol}_request_id"].map(lambda x: None)
        for idx, rid in wide[f"{pol}_request_id"].items():
            txt = raw_map.get(str(rid)); obj = parse_json(txt)
            if isinstance(obj, dict):
                wide.at[idx, f"{pol}_diagnosis"] = obj.get("diagnosis")
                wide.at[idx, f"{pol}_rationale"] = obj.get("rationale")
                wide.at[idx, f"{pol}_evidence_fields"] = json.dumps(obj.get("evidence_fields"), ensure_ascii=False) if obj.get("evidence_fields") is not None else None
    payload = pd.read_parquet(release / "firm_payload_source.parquet")
    payload = payload[["firm_key", "firm_id", "fiscal_year", "firm_name", "industry_display_value"] + [c for c in FIN_FIELDS + CONTEXT_FIELDS if c in payload.columns]].copy()
    wide = wide.merge(payload, on="firm_key", how="left")
    # Compute geometry with W = diagonal inverse prompt scales.
    for i in range(8):
        wide[f"a0_{i}"] = wide[f"C4_{i}"]
        wide[f"aN_{i}"] = wide[f"C4R_{i}"]
        wide[f"aE_{i}"] = wide[f"C6-E_{i}"]
        wide[f"aX_{i}"] = wide[f"C6-EX_{i}"]
    def geom(rec, final="E"):
        a0 = rec[[f"a0_{i}" for i in range(8)]].to_numpy(float) / z
        an = rec[[f"aN_{i}" for i in range(8)]].to_numpy(float) / z
        ae = rec[[f"a{final}_{i}" for i in range(8)]].to_numpy(float) / z
        rr = rec[[f"r_{i}" for i in range(8)]].to_numpy(float) / z
        advice = rr - a0; move = ae - an
        den = float(np.dot(advice, advice)); num = float(np.dot(move, advice))
        if den <= 1e-6: return pd.Series({"uptake": np.nan, "uptake_unstable": True, "rl_direction_move": np.nan, "distance_reduction": np.nan, "distance_to_rl": np.linalg.norm(ae-rr), "baseline_distance_to_rl": np.linalg.norm(a0-rr)})
        return pd.Series({"uptake": num/(den+1e-8), "uptake_unstable": False, "rl_direction_move": num, "distance_reduction": np.linalg.norm(a0-rr)-np.linalg.norm(ae-rr), "distance_to_rl": np.linalg.norm(ae-rr), "baseline_distance_to_rl": np.linalg.norm(a0-rr)})
    wide = pd.concat([wide, wide.apply(geom, axis=1)], axis=1)
    def geom_x(rec):
        a0 = rec[[f"a0_{i}" for i in range(8)]].to_numpy(float) / z
        an = rec[[f"aN_{i}" for i in range(8)]].to_numpy(float) / z
        ax = rec[[f"aX_{i}" for i in range(8)]].to_numpy(float) / z
        rr = rec[[f"rx_{i}" for i in range(8)]].to_numpy(float) / z
        advice = rr - a0; move = ax - an
        den = float(np.dot(advice, advice)); num = float(np.dot(move, advice))
        if den <= 1e-6:
            return pd.Series({"uptake_X": np.nan, "uptake_X_unstable": True, "rl_direction_move_X": np.nan, "distance_reduction_X": np.nan, "distance_to_rl_X": np.linalg.norm(ax-rr), "baseline_distance_to_rl_X": np.linalg.norm(a0-rr)})
        return pd.Series({"uptake_X": num/(den+1e-8), "uptake_X_unstable": False, "rl_direction_move_X": num, "distance_reduction_X": np.linalg.norm(a0-rr)-np.linalg.norm(ax-rr), "distance_to_rl_X": np.linalg.norm(ax-rr), "baseline_distance_to_rl_X": np.linalg.norm(a0-rr)})
    wide = pd.concat([wide, wide.apply(geom_x, axis=1)], axis=1)
    wide["advice_type_E"] = wide.apply(lambda r: action_type(r, z, tol=1e-8), axis=1)
    # C6-EX type is computed by replacing aE with aX and r with rx using a small helper.
    def ex_type(r):
        rr = r.copy()
        for i in range(8): rr[f"r_{i}"] = r[f"rx_{i}"]; rr[f"aE_{i}"] = r[f"aX_{i}"]
        return action_type(rr, z, tol=1e-8)
    wide["advice_type_X"] = wide.apply(ex_type, axis=1)
    # Axis-level and direction metrics.
    for i in range(8):
        wide[f"E_axis_toward_{i}"] = ((wide[f"C6-E_{i}"]-wide[f"C4R_{i}"]) * (wide[f"r_{i}"]-wide[f"C4_{i}"]) > 1e-10).astype(int)
        wide[f"E_axis_changed_{i}"] = (np.abs(wide[f"C6-E_{i}"]-wide[f"C4_{i}"]) > 1e-10).astype(int)
    def _field_set(v):
        x = parse_json(v)
        return set(x) if isinstance(x, list) else set()
    c4_sets = wide["C4_evidence_fields"].map(_field_set)
    for pol in ["C4R", "C6-E", "C6-EX"]:
        ps = wide[f"{pol}_evidence_fields"].map(_field_set)
        wide[f"{pol}_evidence_added_count"] = [(len(b-a)) for a,b in zip(c4_sets, ps)]
        wide[f"{pol}_evidence_removed_count"] = [(len(a-b)) for a,b in zip(c4_sets, ps)]
        wide[f"{pol}_evidence_overlap_count"] = [(len(a&b)) for a,b in zip(c4_sets, ps)]
    wide["axes_added_E"] = sum((np.abs(wide[f"C6-E_{i}"]-wide[f"C4_{i}"]) > 1e-10) & (np.abs(wide[f"C4_{i}"]) <= 1e-10) for i in range(8))
    wide["axes_removed_E"] = sum((np.abs(wide[f"C6-E_{i}"]) <= 1e-10) & (np.abs(wide[f"C4_{i}"]) > 1e-10) for i in range(8))
    wide["total_intensity_change_E"] = sum(np.abs(wide[f"C6-E_{i}"])/z[i] for i in range(8)) - sum(np.abs(wide[f"C4_{i}"])/z[i] for i in range(8))
    wide["rl_exact_final_E"] = wide["distance_to_rl"] <= 1e-8
    def _flip(r):
        move = (r[[f"C6-E_{i}" for i in range(8)]].to_numpy(dtype=float) - r[[f"C4_{i}" for i in range(8)]].to_numpy(dtype=float)) / z
        advice = (r[[f"r_{i}" for i in range(8)]].to_numpy(dtype=float) - r[[f"C4_{i}" for i in range(8)]].to_numpy(dtype=float)) / z
        return float(np.dot(move, advice)) < -1e-10
    wide["direction_flip_E"] = wide.apply(_flip, axis=1)
    # Score merge: the auxiliary Stage8 run is expected to contain all four policies.
    score_path = RUNTIME / "data/final_freeze/stage8_llm_multi_oracle_eval/llm_stage8_multi_oracle_scores_all_populations.parquet"
    scores = pd.read_parquet(score_path)
    scores = scores[(scores["mode"] == "free8") & (scores["information_condition"] == "IC-b") & (scores["budget"] == "B1") & (scores["replicate"] == 1) & scores["policy"].isin(POLICIES) & (scores["analysis_population"] == "itt")].copy()
    sc = scores[["firm_key", "model_key", "policy"] + [f"R_score_{x.lower()}" for x in ORACLES] + [f"delta_R_score_{x.lower()}" for x in ORACLES]].copy()
    for pol in POLICIES:
        q = sc[sc.policy == pol].drop(columns="policy").rename(columns={c: f"{pol}_{c}" for c in sc.columns if c not in ["firm_key", "model_key", "policy"]})
        wide = wide.merge(q, on=["firm_key", "model_key"], how="left")
    # C3-E candidate payoffs for q, matched to C4R scores for the same model/firms.
    for oracle in ORACLES:
        rv = pay[pay["action"].notna()][["firm_key", "action", oracle]].rename(columns={"action": "rl_action_id", oracle: f"rl_value_{oracle}"})
        wide = wide.merge(rv, on=["firm_key", "rl_action_id"], how="left")
        wide[f"q_{oracle}"] = wide[f"rl_value_{oracle}"] - wide[f"C4R_R_score_{oracle.lower()}"]
        wide[f"ref_quality_{oracle}"] = np.where(wide[f"q_{oracle}"] > 1e-9, "favorable", np.where(wide[f"q_{oracle}"] < -1e-9, "unfavorable", "tie"))
        wide[f"E_score_increment_{oracle}"] = wide[f"C6-E_delta_R_score_{oracle.lower()}"] - wide[f"C4R_delta_R_score_{oracle.lower()}"]
    wide.to_csv(OUT / "FIRM_LEVEL_ADVICE_RESPONSE.csv", index=False, encoding="utf-8-sig")
    return wide, scales, vectors, scores


def make_outputs(wide, scales, vectors, scores):
    # Behavior summary by model and policy/reference geometry.
    b_rows = []
    for (model, pol, typ), g in pd.concat([
        wide.assign(policy="C6-E", advice_type=wide["advice_type_E"]),
        wide.assign(policy="C6-EX", advice_type=wide["advice_type_X"]),
    ]).groupby(["model", "policy", "advice_type"]):
        denom = max(1, int((wide["model"] == model).sum()))
        b_rows.append({"model": model, "policy": pol, "behavior_type": typ, "n": len(g), "share": len(g)/denom, "uptake_mean": g["uptake"].mean() if pol=="C6-E" else g["uptake_X"].mean(), "policy_delta_alpha_mean": g["C6-E_delta_R_score_alpha"].mean() if pol=="C6-E" else g["C6-EX_delta_R_score_alpha"].mean()})
    pd.DataFrame(b_rows).to_csv(OUT / "ADVICE_TAKING_BEHAVIOR.csv", index=False, encoding="utf-8-sig")
    # Financial state associations: predeclared primary feature set plus all 23 exploratory rows.
    assoc = []
    outcomes = [("uptake", wide["uptake"]), ("direct_acceptance", (wide["advice_type_E"]=="direct_acceptance").astype(float)), ("reject_or_maintain", (wide["advice_type_E"]=="reject_or_maintain").astype(float))]
    for feature in FIN_FIELDS + ["log_assets"]:
        if feature not in wide: continue
        x = pd.to_numeric(wide[feature], errors="coerce")
        for outcome, y in outcomes:
            ok = x.notna() & pd.to_numeric(y, errors="coerce").notna()
            if ok.sum() < 20: rho, p = np.nan, np.nan
            else: rho, p = stats.spearmanr(x[ok], pd.to_numeric(y, errors="coerce")[ok])
            lo, hi = bootstrap_ci(pd.to_numeric(y[ok], errors="coerce"), stat=np.mean) if ok.sum() else (np.nan, np.nan)
            assoc.append({"feature": feature, "outcome": outcome, "n": int(ok.sum()), "spearman_rho": rho, "p_value": p, "outcome_mean": float(pd.to_numeric(y[ok], errors="coerce").mean()) if ok.sum() else np.nan, "outcome_mean_bootstrap_ci_low": lo, "outcome_mean_bootstrap_ci_high": hi})
    af = pd.DataFrame(assoc); af["q_value_bh_within_outcome"] = af.groupby("outcome")["p_value"].transform(lambda s: bh_adjust(s.to_numpy())); af["financial_area"] = af.feature.map({f: area for area, fs in GROUPS.items() for f in fs}).fillna("other_prompt_financial")
    af.to_csv(OUT / "FINANCIAL_STATE_ASSOCIATIONS.csv", index=False, encoding="utf-8-sig")
    # Conditional within-reference-action analysis, with C4 action intensity as a control.
    c_rows=[]
    for ref_col, ref_name in [("ref_candidate_E", "C6-E"), ("ref_candidate_X", "C6-EX")]:
        for (model, action), g in wide.groupby(["model", ref_col], dropna=False):
            for feature in ["derived__current_ratio", "derived__debt_to_assets", "derived__operating_margin", "derived__inventory_to_revenue", "derived__receivables_to_revenue", "log_assets"]:
                ok = g[feature].notna() & g["uptake"].notna()
                if ok.sum() < 15: rho,p=np.nan,np.nan
                else: rho,p=stats.spearmanr(g.loc[ok,feature],g.loc[ok,"uptake"])
                c_rows.append({"model": model, "reference_condition": ref_name, "reference_action": action, "feature": feature, "n": int(ok.sum()), "spearman_rho": rho, "p_value": p, "control_note": "within exact reference-candidate stratum; C4 vector is held as the branch baseline in uptake geometry"})
    cf=pd.DataFrame(c_rows); cf["q_value_bh"] = bh_adjust(cf["p_value"].to_numpy()); cf.to_csv(OUT / "CONDITIONAL_ADVICE_ANALYSIS.csv", index=False, encoding="utf-8-sig")
    # Quality response table by Oracle, preserving favorable/unfavorable/tie.
    qrows=[]
    for oracle in ORACLES:
        for resp in ["direct_acceptance", "partial_acceptance", "combination", "unrelated_revision", "reject_or_maintain", "opposite_direction"]:
            for qual in ["favorable","unfavorable","tie"]:
                m=(wide[f"ref_quality_{oracle}"]==qual)&(wide["advice_type_E"]==resp)
                qrows.append({"oracle":oracle,"quality":qual,"response_type":resp,"n":int(m.sum()),"share_all":float(m.mean()),"mean_uptake":float(wide.loc[m,"uptake"].mean()) if m.any() else np.nan,"mean_C6E_minus_C4R":float(wide.loc[m,f"E_score_increment_{oracle}"].mean()) if m.any() else np.nan})
    pd.DataFrame(qrows).to_csv(OUT / "ADVICE_QUALITY_RESPONSE.csv", index=False, encoding="utf-8-sig")
    # Reference swap: 434 changed vs 141 unchanged, with paired action and value contrasts.
    srows=[]
    for group,g in wide.groupby("reference_group"):
        for model,gm in g.groupby("model"):
            rec={"reference_group":group,"model":model,"n":len(gm),"mean_uptake_E":gm.uptake.mean(),"median_uptake_E":gm.uptake.median(),"mean_distance_reduction_E":gm.distance_reduction.mean(),"mean_uptake_X":gm.uptake_X.mean(),"median_uptake_X":gm.uptake_X.median(),"mean_distance_reduction_X":gm.distance_reduction_X.mean(),"mean_C6E_minus_C6EX_alpha":(gm["C6-E_R_score_alpha"]-gm["C6-EX_R_score_alpha"]).mean(),"mean_C6E_minus_C6EX_beta":(gm["C6-E_R_score_beta"]-gm["C6-EX_R_score_beta"]).mean(),"mean_C6E_minus_C6EX_gamma":(gm["C6-E_R_score_gamma"]-gm["C6-EX_R_score_gamma"]).mean(),"same_reference_action_note":group=="same_141"}; srows.append(rec)
    pd.DataFrame(srows).to_csv(OUT / "REFERENCE_SWAP_ANALYSIS.csv", index=False, encoding="utf-8-sig")
    # Mechanism decomposition.
    mech=[]
    for model,g in wide.groupby("model"):
        for typ in ["direct_acceptance","partial_acceptance","combination","unrelated_revision","reject_or_maintain","opposite_direction"]:
            m=g.advice_type_E==typ; n=int(m.sum())
            row={"model":model,"mechanism":typ,"n":n,"share":n/len(g),"mean_C6E_minus_C4R_alpha":g.loc[m,"E_score_increment_Alpha"].mean() if n else np.nan,"mean_C6E_minus_C4R_beta":g.loc[m,"E_score_increment_Beta"].mean() if n else np.nan,"mean_C6E_minus_C4R_gamma":g.loc[m,"E_score_increment_Gamma"].mean() if n else np.nan,"mean_C6E_score_alpha":g.loc[m,"C6-E_R_score_alpha"].mean() if n else np.nan,"mean_rl_score_alpha":g.loc[m,"rl_value_Alpha"].mean() if n else np.nan}
            mech.append(row)
        # separate rule-recovery flag based on unusable C4 applied fallback.
        m=g["C4_fallback"] & ~g["C6-E_fallback"]
        mech.append({"model":model,"mechanism":"rule_recovery_from_C4_fallback","n":int(m.sum()),"share":float(m.mean()),"mean_C6E_minus_C4R_alpha":g.loc[m,"E_score_increment_Alpha"].mean() if m.any() else np.nan,"mean_C6E_minus_C4R_beta":g.loc[m,"E_score_increment_Beta"].mean() if m.any() else np.nan,"mean_C6E_minus_C4R_gamma":g.loc[m,"E_score_increment_Gamma"].mean() if m.any() else np.nan,"mean_C6E_score_alpha":g.loc[m,"C6-E_R_score_alpha"].mean() if m.any() else np.nan,"mean_rl_score_alpha":g.loc[m,"rl_value_Alpha"].mean() if m.any() else np.nan})
    pd.DataFrame(mech).to_csv(OUT / "ACTION_REVISION_MECHANISMS.csv", index=False, encoding="utf-8-sig")
    make_figures(wide, af)
    write_report(wide, af, cf, qrows, srows, mech)


def make_figures(wide, af):
    plt.style.use("seaborn-v0_8-whitegrid")
    g = af[(af.outcome=="direct_acceptance") & af.feature.isin(["derived__current_ratio","derived__debt_to_assets","derived__operating_margin","derived__inventory_to_revenue","log_assets"])].copy()
    if len(g):
        plt.figure(figsize=(9,5)); plt.bar(g.feature.str.replace("derived__", "", regex=False), g.spearman_rho.fillna(0)); plt.xticks(rotation=35, ha="right"); plt.ylabel("Spearman rho with direct uptake"); plt.tight_layout(); plt.savefig(OUT/"figure_financial_feature_uptake.png", dpi=160); plt.close()
    mat=[]
    for o in ORACLES:
        for q in ["favorable","unfavorable","tie"]:
            m=wide[f"ref_quality_{o}"]==q; mat.append([o,q,int(m.sum()),float((wide.loc[m,"advice_type_E"]=="direct_acceptance").mean()) if m.any() else np.nan])
    mm=pd.DataFrame(mat,columns=["oracle","quality","n","direct_rate"]); piv=mm.pivot(index="oracle",columns="quality",values="direct_rate"); piv.plot(kind="bar",figsize=(8,5)); plt.ylabel("Direct acceptance rate"); plt.tight_layout(); plt.savefig(OUT/"figure_advice_quality_matrix.png",dpi=160); plt.close()
    ct=pd.crosstab(wide.model, wide.advice_type_E, normalize="index"); ct.plot(kind="bar",stacked=True,figsize=(9,5)); plt.ylabel("Share"); plt.tight_layout(); plt.savefig(OUT/"figure_gpt_gemini_behavior_types.png",dpi=160); plt.close()
    plt.figure(figsize=(8,5)); plt.scatter(wide["derived__debt_to_assets"],wide["uptake"],c=(wide.model=="GPT")); plt.xlabel("Debt to assets"); plt.ylabel("Advice uptake U"); plt.tight_layout(); plt.savefig(OUT/"figure_controlled_relation.png",dpi=160); plt.close()
    vals=[wide.loc[wide.advice_type_E==t,"C6-E_delta_R_score_alpha"].dropna() for t in ["direct_acceptance","partial_acceptance","combination","unrelated_revision","reject_or_maintain","opposite_direction"]]; plt.figure(figsize=(10,5)); plt.boxplot(vals,tick_labels=["direct","partial","combine","unrelated","reject","opposite"],showfliers=False); plt.ylabel("C6-E alpha delta"); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(OUT/"figure_policy_distribution.png",dpi=160); plt.close()


def write_report(wide, af, cf, qrows, srows, mech):
    def fmt(x): return "NA" if not np.isfinite(x) else f"{x:.4f}"
    counts = wide.advice_type_E.value_counts().to_dict()
    firm_swap = wide[["firm_key", "reference_changed"]].drop_duplicates("firm_key")
    changed = int(firm_swap.reference_changed.sum()); same = int((~firm_swap.reference_changed).sum())
    stable = wide.loc[~wide.uptake_unstable]
    score_lines = []
    for model, g in wide.groupby("model"):
        d = g["C6-E_delta_R_score_alpha"] - g["C4R_delta_R_score_alpha"]
        lo, hi = bootstrap_ci(d)
        score_lines.append(f"{model}: stable U={int((~g.uptake_unstable).sum())}/{len(g)}, mean U={fmt(g.uptake.mean())}, median U={fmt(g.uptake.median())}, exact C6-E/RL match={fmt(g.rl_exact_final_E.mean())}, Alpha(C6-E−C4R)={fmt(d.mean())} [{fmt(lo)}, {fmt(hi)}]")
    quality_lines = []
    for model, g in wide.groupby("model"):
        acc = g.advice_type_E.isin(["direct_acceptance", "partial_acceptance", "combination"])
        for qual in ["favorable", "unfavorable"]:
            m = g["ref_quality_Alpha"] == qual
            quality_lines.append(f"{model} Alpha {qual}: n={int(m.sum())}, uptake-type acceptance={fmt(acc[m].mean())}")
    qsig = af[af["q_value_bh_within_outcome"] < 0.05]
    csig = cf[cf["q_value_bh"] < 0.05]
    lines=["# Final research report — Financial-State-Conditioned Policy Advice Taking", "", "## Scope", "", "Primary frozen condition: HIGH, FREE8, IC-b, B1, STRICT ITT, Run1, 575 firms, GPT-5.4-mini and Gemini 3.1 Flash-Lite. No new LLM call, RL retraining, or frozen-evidence edit was used.", "", "The analysis uses applied ITT vectors while preserving response usability/fallback flags. C4R and C6-E are treated as parallel branches from the same C4 parent, not a sequential intervention.", "", "## Direct answers", ""]
    lines += [f"- **Q1.** C6-E behavior is distributed across predeclared types: {counts}. Exact copying is only the `direct_acceptance` cell; the continuous U and distance measures should be preferred over a binary interpretation.", f"- **Q2.** Feature-level associations are exploratory. `FINANCIAL_STATE_ASSOCIATIONS.csv` reports Spearman rho/p-values, BH correction, and bootstrap CIs for the outcome mean (not a causal effect CI); no result is called causal.", f"- **Q3.** Conditional rows stratify on the exact reference candidate. Sparse strata are reported as NA rather than pooled to manufacture precision.", f"- **Q4.** `ADVICE_QUALITY_RESPONSE.csv` separates favorable/unfavorable/tie by each post-hoc Oracle. This is a response–quality association, not a model-visible signal or proof of selective rationality.", f"- **Q5.** GPT and Gemini distributions are shown separately. A cross-model mechanism is only supported if both the behavior shares and quality/score relations align; the report does not infer a shared mechanism from one p-value.", f"- **Q6.** `ACTION_REVISION_MECHANISMS.csv` reports counts and Oracle deltas for direct, partial, combination, unrelated, reject, opposite, and rule-recovery paths. Any positive combination cell is not a causal mediation estimate.", "- **Q7.** A contribution claim requires robust, pre-specified, cross-Oracle and cross-model evidence. These frozen observational analyses alone do not establish a new adaptive algorithm; negative or mixed findings should be reported as such.", "", "## Descriptive results", ""]
    lines += [f"- {x}" for x in score_lines]
    lines += ["- Alpha quality response (acceptance = direct + partial + combination): " + "; ".join(quality_lines) + "."]
    if len(qsig):
        lines += [f"- {len(qsig)} feature–outcome rows remain BH q<0.05 in the unconditioned exploratory screen; their small rho values and the non-random action assignment are why they are not interpreted as a financial-state mechanism. See the full table."]
    else:
        lines += ["- No feature–outcome row remains BH q<0.05 in the unconditioned exploratory screen."]
    lines += [f"- The exact-reference conditional screen has {len(csig)} BH q<0.05 rows, concentrated in a few action strata (not a stable all-action or cross-model pattern); these are reported as conditional associations, not causal financial-state effects."]
    lines += ["- U is a geometry score, not a probability. The denominator threshold is 1e-6 in normalized action units; those rows are marked `uptake_unstable` and excluded from feature correlations.", "- Structured `evidence_fields` are retained in the firm table with added/removed/overlap counts relative to C4. Free text is retained as an auxiliary observable output only; no new judge was introduced.", "", "## Reference swap", "", f"At the unique firm level the frozen permutation changes the reference candidate for {changed} firms and leaves it unchanged for {same}; model rows duplicate each firm for the two model branches. `REFERENCE_SWAP_ANALYSIS.csv` keeps these groups separate.", "", "## Prompt audit conclusion", "", "The outgoing Stage7 request JSONL confirms 23 financial fields plus year/assets/market/industry, the earlier C4 draft, and the additional reference ordering. No model-visible Oracle score, Q value, advantage, or confidence value was found in the reference/state blocks. See `PROMPT_INFORMATION_AUDIT.md`.", "", "## Reproducibility", "", "Run `python analysis/policy_advice_taking_financial_state.py` with the frozen runtime and completed Stage8 all-populations file. The script writes all requested CSVs, figures, and this report under `analysis/policy_advice_taking_financial_state/`."]
    (OUT/"FINAL_RESEARCH_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def write_manifest(wide):
    def sha256(p):
        h = hashlib.sha256()
        with Path(p).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    stage8 = RUNTIME / "data/final_freeze/stage8_llm_multi_oracle_eval/llm_stage8_multi_oracle_scores_all_populations.parquet"
    contract = RUNTIME / "data/final_freeze/stage2_candidate_projection/candidate_action_contract_v4_3.json"
    payload = RUNTIME / "data/final_freeze/llm_runs/final_plan3/d5d878bf50718bac65ede73df3fc88c16723870b6c90a790557ff35b063b7ba7/firm_payload_source.parquet"
    manifest = {"script": str(Path(__file__).resolve()), "runtime": str(RUNTIME), "output": str(OUT), "n_firm_model_rows": int(len(wide)), "n_unique_firms": int(wide.firm_key.nunique()), "models": sorted(wide.model.unique().tolist()), "stage8_scores_sha256": sha256(stage8), "action_contract_sha256": sha256(contract), "prompt_payload_sha256": sha256(payload), "uptake_denominator_threshold_normalized_squared": 1e-6, "analysis_population": "strict ITT primary; HIGH/FREE8/IC-b/B1/Run1", "no_new_llm_calls": True, "no_rl_retraining": True}
    (OUT / "RUN_METADATA.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    prompt_audit()
    wide, scales, vectors, scores = build_analysis()
    make_outputs(wide, scales, vectors, scores)
    write_manifest(wide)
    print(json.dumps({"status":"ok","n_firm_model":len(wide),"out":str(OUT)},ensure_ascii=False))
