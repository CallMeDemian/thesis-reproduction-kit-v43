"""Resume a Stage-8 run after Alpha/Beta files were written but Gamma loading failed."""
from pathlib import Path
import json
import pandas as pd

from credit_recourse.contracts.stage_paths import final_root, stage_dir
from credit_recourse.eval.final_stage8_llm_multi_oracle_eval.pipeline import (
    _load_stage6_noop_scores, _per_policy_summary, _stage6_substrate_hashes, _now,
)
from credit_recourse.eval.v43_oracle_backends import score_gamma_model
from credit_recourse.oracle.artifact_io import load_registry, resolve_backend_artifact
from credit_recourse.rl.common.io import write_json

ROOT = Path(r"C:\Users\Demian\Desktop\REPRO_KIT_semantic_repair\analysis\rl_reference_gate_runtime")
FINAL = final_root(ROOT)
OUT = stage_dir(ROOT, "stage8")
SIM = pd.read_parquet(OUT / "simulated_oracle_input_frame.parquet")
ALPHA = pd.read_parquet(OUT / "llm_oracle_scores_alpha_all_populations.parquet")
BETA = pd.read_parquet(OUT / "llm_oracle_scores_beta_all_populations.parquet")
REG = load_registry(FINAL / "configs" / "oracle_backend_registry.yaml")
GAMMA = REG["backends"]["gamma"]
PARAMS = resolve_backend_artifact(ROOT, FINAL, GAMMA["params"])
MODEL = resolve_backend_artifact(ROOT, FINAL, GAMMA["model"])
gamma = ALPHA[[c for c in ALPHA.columns if c != "R_score_alpha"]].copy()
gamma["R_score_gamma"] = score_gamma_model(SIM, PARAMS, MODEL).to_numpy()
gamma.to_parquet(OUT / "llm_oracle_scores_gamma_all_populations.parquet", index=False)
gamma.loc[gamma.analysis_population.astype(str).eq("itt")].to_parquet(OUT / "llm_oracle_scores_gamma_itt.parquet", index=False)
gamma.loc[gamma.analysis_population.astype(str).eq("per_protocol")].to_parquet(OUT / "llm_oracle_scores_gamma.parquet", index=False)
keys = [c for c in ALPHA.columns if c not in {"R_score_alpha"}]
merged = ALPHA.merge(BETA[keys + ["R_score_beta"]], on=keys, how="outer").merge(gamma[keys + ["R_score_gamma"]], on=keys, how="outer")
for backend in ("alpha", "beta", "gamma"):
    merged = merged.merge(_load_stage6_noop_scores(ROOT)[backend], on="row_id", how="left")
    merged[f"delta_R_score_{backend}"] = pd.to_numeric(merged[f"R_score_{backend}"], errors="coerce") - pd.to_numeric(merged[f"noop_R_score_{backend}"], errors="coerce")
merged.to_parquet(OUT / "llm_stage8_multi_oracle_scores_all_populations.parquet", index=False)
merged.loc[merged.analysis_population.astype(str).eq("itt")].to_parquet(OUT / "llm_stage8_multi_oracle_scores_itt.parquet", index=False)
merged.loc[merged.analysis_population.astype(str).eq("per_protocol")].to_parquet(OUT / "llm_stage8_multi_oracle_scores_per_protocol.parquet", index=False)
merged.loc[merged.analysis_population.astype(str).eq("per_protocol")].to_parquet(OUT / "llm_stage8_multi_oracle_scores.parquet", index=False)
summary = _per_policy_summary(merged)
summary.to_csv(OUT / "llm_stage8_policy_summary_all_populations.csv", index=False, encoding="utf-8-sig")
summary.loc[summary.analysis_population.astype(str).eq("itt")].to_csv(OUT / "llm_stage8_policy_summary_itt.csv", index=False, encoding="utf-8-sig")
summary.loc[summary.analysis_population.astype(str).eq("per_protocol")].to_csv(OUT / "llm_stage8_policy_summary_per_protocol.csv", index=False, encoding="utf-8-sig")
metadata = {"status": "PASS", "resumed_gamma_only": True, "created_utc": _now(), "stage6_substrate_hashes": _stage6_substrate_hashes(ROOT), "row_count_evaluated": int(merged.row_id.nunique()), "outputs": {"multi_oracle_scores_all_populations": "llm_stage8_multi_oracle_scores_all_populations.parquet"}}
write_json(OUT / "metadata.json", metadata)
print(json.dumps(metadata, ensure_ascii=False, indent=2))
