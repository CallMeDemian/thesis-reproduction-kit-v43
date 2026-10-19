"""E1 analysis exactly as fixed in the plan (Sec. 4, "E1 통계 계약"). Commit before the first provider call.

Inputs : an E1 observation table with one row per model x firm x condition (3,450 rows) and the columns
         model_key, row_id, firm_key, e1_condition, delta_R_score_alpha, delta_R_score_beta,
         delta_R_score_gamma (Strict ITT), plus the request manifest (own_C3E_action per firm).
Outputs: T5_E1.csv (30 rows) and E1_wording.json.

    python e1_analyze.py --kit-root . --observations <E1 observations> --manifest <E1_request_manifest.csv> --out <dir>
    python e1_analyze.py --kit-root . --selftest      # code test on frozen September data, before any E1 data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_CODE = {"openai_gpt54mini": "GPT", "google_gemini31flashlite": "GEMINI"}
ORACLES = ("alpha", "beta", "gamma")
CONDITIONS = ("E1-C4R", "E1-RL", "E1-OE")
# contrast_id, lhs, rhs, firm subset, analysis_id, Holm family (None = descriptive, no test), status
CONTRASTS = (
    ("E1a", "E1-RL", "E1-OE", "S343", "ABL_E1_343", "E1_PRIMARY", "EXPLORATORY"),
    ("E1b", "E1-OE", "E1-C4R", "ALL575", "ABL_E1", "E1_SECONDARY", "EXPLORATORY"),
    ("E1c", "E1-RL", "E1-C4R", "ALL575", "ABL_E1", "E1_SECONDARY", "EXPLORATORY"),
    ("E1a_575", "E1-RL", "E1-OE", "ALL575", "ABL_E1", None, "DESCRIPTIVE"),
    ("E1a_placebo", "E1-RL", "E1-OE", "S232", "ABL_E1_232", "E1_PLACEBO_GATE", "EXPLORATORY"),
)
SENTENCES = {
    "ALL_POSITIVE": "학습된 행동 내용이 고정 OE보다 세 평가모형 모두에서 높은 정책가치를 만들었다",
    "ALL_NEGATIVE": "고정 OE가 학습된 행동 내용보다 세 평가모형 모두에서 높은 정책가치를 만들었다",
    "ALPHA_ONLY_SAME": "학습된 행동 내용의 이득은 α에서 확인됐고 β·γ는 방향만 같았다. α는 C3-E 선택에 쓴 평가모형이다",
    "ALPHA_ONLY_OPPOSITE": "α에서만 확인되어 C3-E 선택의 영향과 구별할 수 없다",
    "NONE": "학습된 행동 내용의 추가 이득을 확인하지 못했다",
    "MIXED": "평가모형에 따라 엇갈렸다",  # report the per-Oracle values next to it (per_oracle)
}


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values, monotone and capped at 1."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def seed_key(analysis_id: str, lhs: str, rhs: str, model: str, oracle: str) -> dict:
    return {"action_space": "FREE8", "analysis_id": analysis_id, "budget": "B1", "execution_policy": "STRICT",
            "information_condition": "IC-b", "lhs": lhs, "model": MODEL_CODE[model], "oracle": oracle,
            "population": "ITT", "reasoning_regime": "HIGH", "replicate": "RUN_E1", "rhs": rhs}


def analyze(obs: pd.DataFrame, own: pd.Series, stats, source_file: str, source_sha256: str, commit: str) -> pd.DataFrame:
    paired_bootstrap, pcg64_seed, canonical_key_json = stats
    need = {"model_key", "firm_key", "e1_condition", *(f"delta_R_score_{o}" for o in ORACLES)}
    missing = need - set(obs.columns)
    if missing:
        raise SystemExit(f"E1 ANALYSIS BLOCKED: missing columns {sorted(missing)}")
    if obs.duplicated(["model_key", "firm_key", "e1_condition"]).any():
        raise SystemExit("E1 ANALYSIS BLOCKED: duplicate model/firm/condition rows")
    for model in MODEL_CODE:
        for cond in CONDITIONS:
            n = int(((obs["model_key"] == model) & (obs["e1_condition"] == cond)).sum())
            if n != 575:
                raise SystemExit(f"E1 ANALYSIS BLOCKED: {model}/{cond} has {n} rows, expected 575")
    if len(own) != 575 or int((own == "OE").sum()) != 232:
        raise SystemExit("E1 ANALYSIS BLOCKED: own C3-E action map is not 575 firms with 232 OE")
    for (model, cond), part in obs.groupby(["model_key", "e1_condition"]):
        if set(part["firm_key"]) != set(own.index):
            raise SystemExit(f"E1 ANALYSIS BLOCKED: firm set of {model}/{cond} differs from the manifest")
    subsets = {"ALL575": set(own.index), "S343": set(own.index[own != "OE"]), "S232": set(own.index[own == "OE"])}
    rows = []
    for contrast, lhs, rhs, subset, analysis_id, family, status in CONTRASTS:
        for model in MODEL_CODE:
            for oracle in ORACLES:
                col = f"delta_R_score_{oracle}"
                part = obs[(obs["model_key"] == model) & obs["e1_condition"].isin([lhs, rhs])]
                wide = part.pivot(index="firm_key", columns="e1_condition", values=col)
                wide = wide.loc[sorted(set(wide.index) & subsets[subset])]  # firm_key ascending
                if wide[[lhs, rhs]].isna().any().any():
                    raise SystemExit(f"E1 ANALYSIS BLOCKED: incomplete pairs in {contrast}/{model}/{oracle}")
                diff = (wide[lhs] - wide[rhs]).to_numpy(dtype=float)
                if len(diff) != {"ALL575": 575, "S343": 343, "S232": 232}[subset]:
                    raise SystemExit(f"E1 ANALYSIS BLOCKED: {contrast}/{model}/{oracle} has n={len(diff)}")
                key = seed_key(analysis_id, lhs, rhs, model, oracle)
                key_json = canonical_key_json(key)
                result = paired_bootstrap(diff, seed=pcg64_seed(key_json))
                rows.append({
                    "result_id": f"{contrast}__{MODEL_CODE[model]}__{oracle}", "status": status,
                    "contrast_id": contrast, "lhs": lhs, "rhs": rhs, "model": MODEL_CODE[model],
                    "reasoning_regime": "HIGH", "oracle": oracle, "budget": "B1", "action_space": "FREE8",
                    "execution_policy": "STRICT", "replicate": "RUN_E1", "information_condition": "IC-b",
                    "n_firms": result["N"], "estimate": result["estimate"], "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"], "p_raw": result["raw_p"] if family else np.nan,
                    "holm_p": np.nan, "holm_family": family or "N/A", "holm_p_60": np.nan,
                    "source_file": source_file, "source_key": key_json, "source_sha256": source_sha256,
                    "analysis_commit": commit, "notes": f"subset={subset}; draws_sha256={result['bootstrap_draw_sha256']}",
                })
    table = pd.DataFrame(rows)
    for family in ("E1_PRIMARY", "E1_SECONDARY", "E1_PLACEBO_GATE"):
        mask = table["holm_family"] == family
        table.loc[mask, "holm_p"] = holm(table.loc[mask, "p_raw"].to_numpy())
    expected = {"E1_PRIMARY": 6, "E1_SECONDARY": 12, "E1_PLACEBO_GATE": 6, "N/A": 6}
    if table["holm_family"].value_counts().to_dict() != expected or len(table) != 30:
        raise SystemExit("E1 ANALYSIS BLOCKED: family sizes drift")
    return table


def wording(table: pd.DataFrame) -> dict:
    placebo = table[table["contrast_id"] == "E1a_placebo"]
    gate_pass = bool((placebo["holm_p"] >= 0.05).all())
    out = {"placebo_gate": "PASS" if gate_pass else "FAIL: HOLD the E1 interpretation", "by_model": {}}
    for model in ("GPT", "GEMINI"):
        prim = table[(table["contrast_id"] == "E1a") & (table["model"] == model)].set_index("oracle")
        sig = prim["holm_p"] < 0.05
        pos = prim["estimate"] > 0
        if sig.all() and pos.all():
            case = "ALL_POSITIVE"
        elif sig.all() and (~pos).all():
            case = "ALL_NEGATIVE"
        elif sig["alpha"] and pos["alpha"] and not sig["beta"] and not sig["gamma"]:
            case = "ALPHA_ONLY_SAME" if pos["beta"] and pos["gamma"] else "ALPHA_ONLY_OPPOSITE"
        elif not sig.any():
            case = "NONE"
        else:
            case = "MIXED"
        out["by_model"][model] = {"case": case, "sentence": SENTENCES[case] if gate_pass else "HOLD",
                                  "per_oracle": prim[["estimate", "ci_low", "ci_high", "holm_p"]].round(6).to_dict("index")}
    return out


def selftest(root: Path, stats) -> None:
    v13 = root / "frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN/frozen_inputs/stage8"
    # 1) Holm reproduces every frozen primary family (model x oracle, 8 contrasts each)
    for name in ("V13_BASELINE_STRICT_PRIMARY_FULLSTREAM.csv", "V13_HIGH_STRICT_PRIMARY_FULLSTREAM.csv"):
        frozen = pd.read_csv(v13 / name)
        for _, fam in frozen.groupby(["model", "oracle"]):
            if not np.allclose(holm(fam["raw_p"].to_numpy()), fam["holm_p"].to_numpy(), atol=1e-12, rtol=0):
                raise SystemExit(f"SELFTEST FAIL: Holm does not reproduce {name}")
    # 2) the whole pipeline on September data standing in for E1
    #    E1-C4R := C4R, E1-RL := C6-E, E1-OE := C6-EX  ->  E1c must equal frozen A3 and E1a_575 frozen A4
    obs = pd.read_parquet(v13 / "canonical_itt_observations.parquet")
    cell = obs[(obs["reasoning_regime"] == "HIGH") & (obs["execution_policy"] == "STRICT") & (obs["action_space"] == "FREE8")
               & (obs["budget"] == "B1") & (obs["information_condition"] == "IC-b") & (obs["replicate"].astype(int) == 1)
               & obs["policy_condition"].isin(["C4R", "C6-E", "C6-EX"])].copy()
    cell["e1_condition"] = cell["policy_condition"].map({"C4R": "E1-C4R", "C6-E": "E1-RL", "C6-EX": "E1-OE"})
    own = (cell[cell["policy_condition"] == "C6-E"].groupby("firm_key")["reference_candidate_id"].first())
    table = analyze(cell, own, stats, "SELFTEST", "SELFTEST", "SELFTEST")
    frozen = pd.read_csv(v13 / "V13_HIGH_STRICT_PRIMARY_FULLSTREAM.csv")
    frozen = frozen[frozen["budget"] == "B1"]
    code = {v: k for k, v in MODEL_CODE.items()}
    for contrast_id, frozen_contrast in (("E1c", "C6-E-C4R"), ("E1a_575", "C6-E-C6-EX")):
        for _, r in table[table["contrast_id"] == contrast_id].iterrows():
            f = frozen[(frozen["model"] == code[r["model"]]) & (frozen["oracle"] == r["oracle"]) & (frozen["contrast"] == frozen_contrast)]
            if len(f) != 1 or not np.isclose(r["estimate"], f["estimate"].iloc[0], atol=1e-12, rtol=1e-12):
                raise SystemExit(f"SELFTEST FAIL: {contrast_id} {r['model']} {r['oracle']} != frozen {frozen_contrast}")
    if sorted(table["n_firms"].unique().tolist()) != [232, 343, 575]:
        raise SystemExit("SELFTEST FAIL: subset sizes")
    print(json.dumps({"selftest": "PASS", "holm": "reproduces the 12 frozen primary families (96 rows)",
                      "pipeline": "E1c == frozen A3 and E1a_575 == frozen A4 for High/free8/B1 (both models, 3 Oracles)"},
                     ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-root", type=Path, required=True)
    ap.add_argument("--observations", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--commit", default="UNCOMMITTED")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    root = args.kit_root.resolve()
    sys.path.insert(0, str(root / "src"))
    from credit_recourse.analysis.final_v43.statistics import canonical_key_json, paired_bootstrap, pcg64_seed
    stats = (paired_bootstrap, pcg64_seed, canonical_key_json)
    if args.selftest:
        selftest(root, stats)
        return
    obs_path, manifest_path = args.observations.resolve(), args.manifest.resolve()
    obs = pd.read_parquet(obs_path) if obs_path.suffix == ".parquet" else pd.read_csv(obs_path)
    manifest = pd.read_csv(manifest_path)
    own = manifest.drop_duplicates("firm_key").set_index("firm_key")["own_C3E_action"]
    digest = hashlib.sha256(obs_path.read_bytes()).hexdigest()
    table = analyze(obs, own, stats, obs_path.name, digest, args.commit)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "T5_E1.csv", index=False, lineterminator="\n", na_rep="N/A")
    (args.out / "E1_wording.json").write_text(json.dumps(wording(table), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(table), "placebo_gate": wording(table)["placebo_gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
