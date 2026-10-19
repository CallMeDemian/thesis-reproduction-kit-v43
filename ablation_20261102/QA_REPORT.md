# QA_REPORT (ablation_20261102)

File status: DRAFT

## Setup (Q01)

| Item | Value |
| --- | --- |
| Base commit | 071848881349c8675bb4e15193a747fa02679986 (origin/hb-fy24-final) |
| Branch | ablation-20261102 |
| HEAD at start of work | 071848881349c8675bb4e15193a747fa02679986 |
| Python | Python 3.12.10 |
| Packages | google-genai==2.31.0, joblib==1.5.3, matplotlib==3.11.0, numpy==2.4.4, openai==3.35.1, openpyxl==3.1.5, pandas==3.0.2, pyarrow==24.0.0, pytest==9.1.1, python-docx==1.2.0, python-pptx==1.0.2, PyYAML==6.0.3, scikit-learn==1.8.0, scipy==1.17.1, statsmodels==0.14.6 |
| LFS state | Cloned with GIT_LFS_SKIP_SMUDGE=1; E1 inputs and stage9 HIGH_STRICT_ITT pulled; other LFS objects not yet fetched |
| thesis_repro verify | PASS (ablation_20261102/E1/verify_output.txt) |
| Deck SHA-256 | 2667921758789af260215270c99344e3c955a9f38a8f6843405f6881d0f79e4b (matches) |
| Thesis SHA-256 | f6257868c52534fc6477ac249ed463731402553de2cf0786b9667df76b0073a1 (matches) |
| git status --porcelain -- frozen | empty |

## E1 pre-registration gates (Q16, Q17)

| Check | Result | Evidence |
| --- | --- | --- |
| Request build | PASS | E1/requests_build_stdout.txt, E1/requests/E1_build_report.json |
| Manifest SHA-256 | 1f12775443db3a929cddf7a619a27882ad49fe5b189da23572f908854064357a, equal to expected; byte-identical | fc /b against the expected manifest |
| Build report vs expected | equal in checks, counts, manifest and shard hashes; matched_as reads crlf (expected) vs raw on this machine | E1/requests/E1_build_report.json |
| e1_analyze.py --selftest | PASS | E1/selftest_output.txt |

<!-- QA_AUTO_BEGIN -->

## Automated checks (scripts/90_qa.py)

Run at 2026-10-19T19:52:18.315111+00:00 on HEAD 209cf29d3fbe. Evidence: `results/_qa_results.json`. File status of tables and figures: DRAFT until every check below passes and the user has signed off the human lines.

| ID | Check | Result | Key evidence |
| --- | --- | --- | --- |
| Q01 | Source pin | PASS | `{"base": "071848881349c8675bb4e15193a747fa02679986", "branch": "ablation-20261102", "head": "209cf29d3fbe026cea4a31bd313a83738c63ef8c", "python": "3.12.10", "pandas": "3.0.2", "numpy": "2.4.4", "input_sha256": {"frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN/frozen_inputs/sta ...` |
| Q02 | Scale | PASS | `{"requests_total": {"actual": 48300, "expected": 48300}, "strict_requests_21x575x2x2": {"actual": 48300, "expected": 48300}, "observations_rows": {"actual": 96600, "expected": 96600}, "registry_rows": {"actual": 914, "expected": 914}, "primary_rows": {"actual": 96, "expected": 96}, "supplemental_row ...` |
| Q03 | Pairing and row counts | PASS | `{"T1_rows": 96, "T2_rows": 24, "T3_rows": 24, "T3_tested": 12, "T3_descriptive": 12, "R9_rows": 24, "all_n_firms_575": true, "result_id_duplicates": 0}` |
| Q04 | Cell match (only the intended axis differs) | PASS | `{"A1": {"cells_x_firms": 4600, "model_regime_budget_cells": 8, "every_firm_cell_has_both_policies": true}, "A2": {"cells_x_firms": 4600, "model_regime_budget_cells": 8, "every_firm_cell_has_both_policies": true}, "A3": {"cells_x_firms": 4600, "model_regime_budget_cells": 8, "every_firm_cell_has_both ...` |
| Q05 | Draft links | PASS | `{"C4R": {"rows": 11500, "expected": 11500, "parent_is_same_run_same_firm_C4": 11500}, "C6-E": {"rows": 11500, "expected": 11500, "parent_is_same_run_same_firm_C4": 11500}, "C6-EX": {"rows": 4600, "expected": 4600, "parent_is_same_run_same_firm_C4": 4600}}` |
| Q06 | Reference assignment | PASS | `{"firms": 575, "self_reference": 0, "same_action": 141}` |
| Q07 | Strict rule | PASS | `{"UNRESOLVED_INFRA_rows": 0, "strict_unusable_rows": 8376, "all_unusable_mapped_to_zero_action": true, "strict_statuses": {"ACCEPTED": 39924, "CONFIRMED_UNUSABLE_COMPLETION": 8376}, "repaired_rows_equal_requests": 48300}` |
| Q08 | Mean reproduction (T1) | PASS | `{"rows_matched": 96, "np_isclose_atol_rtol_1e-12_all": true, "max_abs_diff": 1.0061396160665481e-16}` |
| Q09 | Candidate selection vs thesis (R9) | PASS | `{"a3_positive_of_12": 11, "a3_holm_sig_of_12": 2, "a2_positive_of_12": 6, "a2_holm_sig_of_12": 4, "a2_sig_with_60family_of_12": 2, "thesis_expected": {"a3_positive": 11, "a3_sig": 2, "a2_positive": 6, "a2_sig": 4, "a2_sig_60": 2}, "a3_means_matched_3dp": "12/12", "a2_means_matched_3dp": "4/4 (the fo ...` |
| Q10 | C3-E | PASS | `{"c3e_rows": 575, "firm_key_unique": true, "c3e_alpha_mean": 0.923299, "high_vs_stage9_gap_rows_each": [575], "high_vs_stage9_gap_max_abs_diff": 8.326672684688674e-17}` |
| Q11 | New analysis contract (seeds, families, status) | PASS | `{"registry_seed_mismatches_of_722": 0, "T2_S1.csv": {"rows": 24, "status": {"EXPLORATORY": 24}, "families": {"S1_ALL24": 24}, "key_fields_ok": true}, "T3_REF1.csv": {"rows": 24, "status": {"EXPLORATORY": 12, "DESCRIPTIVE": 12}, "families": {"REF1_CANDIDATE9_12": 12}, "key_fields_ok": true}, "R9_cand ...` |
| Q12 | Frozen unchanged | PASS | `{"git_diff_base_frozen_empty": true, "git_status_frozen_empty": true, "files_hashed": 1101, "combined_sha256": "dccfce595a222cdd84729c4905c7ae37fbb093173520c15d29162b2d58f6e491", "note": "no frozen/ hash list existed from the start of work; this is the first list (kept outside the repository, privat ...` |
| Q16 | E1 pre-registration order (before first provider call) | PASS | `{"protocol_commit_before_first_provider_call": true, "protocol_commit_time": "2026-10-20T04:14:32+09:00", "first_smoke_response_mtime_utc": "2026-10-19T19:17:33.895954+00:00", "protocol_commit_on_origin": true, "build_checks_all_pass": true, "manifest_sha256_equals_expected": true, "requests": 3450}` |

Q13-Q15 (number cross-check of report and slides, representative-condition notation, Human baseline notation): not yet run; the human sign-off lines are written by the user only.
Q17-Q19: see the E1 section; Q18 and Q19 stay open until all 3,450 responses are collected and analysed.

<!-- QA_AUTO_END -->

## E1 generation and scoring gates (Q17-Q19, in progress)

| Check | Result | Evidence |
| --- | --- | --- |
| Parse replay: September C4R/C6-E outputs of the cell, both models (2,300 rows) | PASS: strict_usable and strict_itt_action_json equal the frozen observations for 2,300/2,300; 34 September-unusable rows map to A0 as frozen | E1/parse_replay_result.json |
| Synthetic refusal, truncated, empty, malformed, bound-violating and error responses | PASS: completed-but-unusable maps to A0, transport and server errors map to retry (never A0), both providers | E1/parse_replay_result.json |
| Smoke test (2 firms x 3 conditions x 2 models) | PASS: access, parse and returned model equal to expected (6/6 OpenAI, 6/6 Gemini). Responses stored, not scored or analysed. The first Gemini attempts failed in my transport code (client lifecycle, then response key case) before reaching the prompt path; no request body or prompt changed | E1/smoke/ (summary files of every run kept) |
| Scoring path: September C4, C4R, C6-E rows of the cell (3,450 rows, both models) | PASS: R_score, A0 score and delta_R_score for alpha, beta and gamma equal the frozen values (atol = rtol = 1e-9; max abs difference 2.8e-14) | E1/scoring/score_gate_result.json |
| Run root | project root = THESIS_REPRO_RUN_ROOT = ..\runroot built from the v2.1.1 distribution zip by E1/scripts/e1_setup_runroot.ps1. Two adjustments, both path mapping only: the frozen action contract is exposed as contracts/scientific/v43_action_contract.json, and the copy of r085_financial_cost_contract.json gets an alias key (03_stage2/... path) with the same digest | E1/scripts/e1_setup_runroot.ps1 |
| Q18 Generation complete (3,450), returned models, Strict rule applied | PASS: 3,450 delivered on attempt 1 (no resubmission, no duplicates); returned_model equals the expected version for all 3,450 (1,725 per provider); 50 completed-but-unusable responses (Gemini 26, OpenAI 24) ran as A0 under Strict; 575 responses per model x condition; no prompt or code path changed after the smoke test; smoke responses excluded (batch regenerated those firms); observations scored with the validated path | E1/runs/collector_log.jsonl, E1/runs/ALL_DELIVERED.txt, E1/observations/e1_observations_summary.json |
| Q19 placebo gate (232 firms) | PASS: all six Holm-adjusted p = 1.000 (>= .05); the Sec. 9 audit was therefore not triggered | results/T5_E1.csv (E1a_placebo rows), results/E1_wording.json |
| E1 analysis (status DRAFT until the final QA on Oct 25) | e1_analyze.py ran once on the complete observations. Pre-registered wording for E1a (343 firms): both models fall in the row for combinations not listed above: "differs by evaluation model". Per Oracle (Holm p): GPT alpha +0.156 (.006), beta -0.022 (.008), gamma +0.052 (.031); Gemini alpha +0.145 (.008), beta -0.018 (.048), gamma +0.027 (.106). E1b and E1c reported unchanged in T5 | results/T5_E1.csv, results/E1_wording.json, figures/F5_E1 |

## Notes recorded for interpretation (2026-10-20)

- S1 (T2) direction: of 24 rows, 18 are positive and 6 negative; 19 have Holm p < .05, of which 16 are positive and 3 are negative. The three significant negative rows are all Gemini, budget BINF: C4 alpha (-0.208 [-0.304, -0.108], Holm p .005), C4 gamma (-0.058 [-0.096, -0.019], .025) and C6-E alpha (-0.180 [-0.272, -0.088], .005). Any sentence on S1 must name these exceptions and cannot say that High raised policy value in every combination.
- Frozen directory: `git diff` and `git status --porcelain` are empty, and `git status --porcelain --ignored -- frozen` is empty too (checked 2026-10-20), so no ignored or untracked file exists under `frozen/`.
- Run-root adjustment for the scoring path (Q18): the financial-cost contract keys its source digests by the `archive/...` path while the loader looks them up by the `03_stage2/...` path. The copy of the contract in the run root outside the repository carries one extra alias key with the identical digest; no value was changed. The frozen action contract is exposed under the path the simulator reads. After these path mappings the September C4, C4R and C6-E rows (3,450) reproduce every frozen score (max abs difference 2.8e-14).
