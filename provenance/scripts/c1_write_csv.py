"""Write SCIENTIFIC_CODE_CHANGES.csv from c1_collect.py output.

Category is assigned by reading each diff (see SCIENTIFIC_CODE_CHANGES.md for the rule).
reason_recorded quotes a commit message only if the message states a reason; none of the
14 kit commits involved has a body, and no subject line states a reason.

Usage: python c1_write_csv.py <c1_data.json> <out_csv>
"""
import csv, json, sys

DATA, OUT = sys.argv[1:3]
PFX = "src/credit_recourse/"
SEM = "semantic_change"; PATH = "path_rebinding"; OTH = "other"
CAT = {
    "analysis/final_v43/parent_gate.py": (PATH, ""),
    "analysis/final_v43/runner.py": (SEM, "CLEAN_REEXECUTION profile removed; only MANUSCRIPT_FROZEN accepted"),
    "analysis/final_v43/selection.py": (SEM, "CLEAN_REEXECUTION source branch removed"),
    "analysis/final_v43/supplemental.py": (PATH, ""),
    "contracts/__init__.py": (OTH, "module docstring added to an empty file"),
    "contracts/stage_paths.py": (PATH, "archive/DEPLOYED_RELEASE stage dirs replaced by run-local numbered dirs; as_manifest() and CLI removed"),
    "contracts/v43_action_contract.py": (SEM, "file-hash, content-hash, semantic-version and timing checks removed; loader renamed; 8D column check added"),
    "eval/final_stage8_llm_multi_oracle_eval/pipeline.py": (SEM, "fixture_mode added (synthetic rates/costs, cohort size from data); fiscal_year required; context columns carried; generation_regime key"),
    "eval/final_stage9_llm_rl_comparison/pipeline.py": (SEM, "fixture_mode cohort size; added row_id identity checks against C3-E table"),
    "eval/final_stage9_llm_rl_comparison/revision_metrics.py": (SEM, "cell_id/generation_regime/phase added to grouping keys"),
    "eval/v43_c2.py": (OTH, "two trailing blank lines"),
    "eval/v43_financial_inputs.py": (SEM, "FirmState construction replaced (alias logic and rating fields dropped)"),
    "eval/v43_one_pass_evaluation.py": (PATH, ""),
    "eval/v43_stage6_reporting.py": (PATH, "beta/gamma params taken from fixed run-local paths instead of registry entries; recorded artifact paths become absolute strings"),
    "eval/v43_stage6_scoring.py": (PATH, ""),
    "final_release/action_validation.py": (PATH, ""),
    "final_release/common.py": (PATH, "repo-root marker changed; published_contract_registry() with frozen fallback added"),
    "final_release/contract.py": (SEM, "config root moved; industry-binding acceptance rule changed (status PASS required, fresh role accepted)"),
    "final_release/doctor.py": (SEM, "golden regression and registry hash checks removed; LLM checks run only with fresh config root"),
    "final_release/evaluation_bridge.py": (PATH, "optional output_dir argument"),
    "final_release/executor.py": (PATH, "OUTPUT_ROOT constant replaced by output_root()"),
    "final_release/llm_contract.py": (SEM, "C6-EX materialization function and stricter C6-EX checks added; industry-binding readiness rule changed"),
    "final_release/rl_reference.py": (PATH, "frozen/original_release fallback for C3-E reference"),
    "high_reasoning/runner.py": (PATH, "OUTPUT_ROOT replaced by output_root(); remaining lines are reformatting"),
    "oracle/backends/alpha/_pipeline_impl_runtime.py": (PATH, ""),
    "oracle/backends/alpha/pipeline.py": (PATH, "also calls configure_utf8_stdio()"),
    "oracle/backends/alpha/production_finalize.py": (SEM, "synthetic profile skips Alpha contract hash checks and rewrites fit_source/contract hash"),
    "oracle/backends/beta/_pipeline_impl_runtime.py": (PATH, ""),
    "oracle/backends/beta/pipeline.py": (OTH, "configure_utf8_stdio() call only"),
    "oracle/backends/gamma/_pipeline_impl_runtime.py": (PATH, ""),
    "oracle/backends/gamma/pipeline.py": (OTH, "configure_utf8_stdio() call only"),
    "oracle/selected_variable_current_contract.py": (PATH, ""),
    "oracle/stage1/stage00_01_rating_statement/final_stage0_adapter.py": (PATH, ""),
    "oracle/stage1/stage00_02_ratio_engineering/modules/_task6_compute_ratios.py": (SEM, "_x/_y merge-suffix fallback and item-code fallback in ratio computation"),
    "oracle/stage1/stage00_04_variable_selection/pipeline.py": (SEM, "Spearman screening input coercion and constant-input guard; report table writer changed"),
    "oracle/verification/diagnose_oracle_backends.py": (OTH, "argparse help text only"),
    "oracle/verification/verify_alpha_contract.py": (SEM, "non-production profile replaces fixed-hash checks with self-consistency checks"),
    "oracle/verification/verify_beta_10grade_orderedlogit_contract.py": (PATH, ""),
    "oracle/verification/verify_gamma_10grade_ml_contract.py": (PATH, ""),
    "oracle/verification/verify_oracle_semantic_closure.py": (PATH, ""),
    "oracle/verification/verify_stage00_04_growth_eligibility_contract.py": (PATH, "also replaces the undefined name stage00_04_dir (original) with stage4"),
    "oracle/verification/verify_stage1_outputs.py": (PATH, ""),
    "oracle/verification/verify_stage1_substrate_validation.py": (SEM, "output fields added; failure status value FAIL -> FAILED"),
    "repro/__init__.py": (OTH, "docstring only"),
    "repro/runtime_helpers/__init__.py": (OTH, "docstring only"),
    "repro/runtime_helpers/build_bp_rate_v4_production_lookup.py": (SEM, "production lookup producer removed; two stub helpers remain"),
    "rl/common/actions.py": (SEM, "no-op id A0_noop -> A0; action/candidate source YAML -> one JSON contract"),
    "rl/common/candidate_intensity_contract.py": (SEM, "no-op id A0_noop -> A0"),
    "rl/pipelines/final_stage2_raw_action_source_precompute/pipeline.py": (PATH, "env-bound cleaned-state dir; fails if unbound under FRESH_REPLICATION"),
    "rl/v43_features.py": (PATH, ""),
    "rl/v43_one_pass_contract.py": (SEM, "seed 2 -> seed taken from stage config; checkpoint path check relaxed"),
    "rl/v43_one_pass_data.py": (PATH, "input root now env-only (no default); manifest keys normalised"),
    "rl/v43_one_pass_grid.py": (PATH, ""),
    "rl/v43_rate_extension.py": (PATH, ""),
    "rl/v43_runtime.py": (SEM, "Stage6 rows source changed and 575-row check relaxed to non-empty; checkpoint path check relaxed; run config_hash check added"),
    "rl/v43_stage5_search.py": (PATH, ""),
    "rl/v43_support.py": (PATH, ""),
}
NO_REASON = "사유 기록 없음"

data = json.load(open(DATA, encoding="utf-8"))
assert len(data) == 57 and set(o["path"][len(PFX):] for o in data) == set(CAT)
cols = ["path", "original_sha256", "current_sha256", "lines_changed", "lines_added", "lines_removed",
        "category", "changed_in_commits", "commit_messages", "reason_recorded", "notes"]
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(cols)
    for o in data:
        cat, note = CAT[o["path"][len(PFX):]]
        # every commit has an empty body; a subject line is a description, not a reason
        assert all(b.count("\n") == 0 for b in o["bodies"].values())
        w.writerow([o["path"], o["original_sha256"], o["current_sha256"], o["lines_changed"],
                    o["added"], o["removed"], cat,
                    " ".join(c[0] for c in o["commits"]),
                    " | ".join(f"{c[0]} {c[2]}" for c in o["commits"]),
                    NO_REASON, note])
from collections import Counter
print(Counter(v[0] for v in CAT.values()))
