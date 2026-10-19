# Ablation protocol (branch `ablation-20261102`)

Base commit: `071848881349c8675bb4e15193a747fa02679986` (`hb-fy24-final`). New files live only under
`ablation_20261102/`. `frozen/`, the Human baseline folder and the result registry are not modified.

## 1. Question

With the same 575 firms, the same models and the same evaluation device (simulator and Oracles), how much
do diagnosis prompting, self-review, a reference program from the RL policy, and the firm-to-reference link
change policy value, and how stable are direction, size, significance and per-firm gains across conditions?
(Thesis RQ3 and RQ4.)

## 2. Contrasts (treatment minus control)

| ID | Contrast | What changes | Status |
| --- | --- | --- | --- |
| A1 | C5 - C4 | diagnosis instruction at first generation (C5 is generated independently) | core, frozen |
| A2 | C4R - C4 | same C4 draft re-reviewed without reference; review and output validation combined | core, frozen |
| A3 | C6-E - C4R | same C4 draft plus the firm's own C3-E action as reference: increment of the added reference information | core, frozen |
| A4 | C6-E - C6-EX | only the firm link of the reference | core, frozen |
| S1 | High - Baseline (C4 and C6-E) | reasoning strength, output limit and run time change together | exploratory, new |
| REF-1 | C6-E - C3-E (candidate9, B1) | LLM revision procedure versus the RL policy alone, same 9 candidates | exploratory, new |
| REF-1 reference | C6-E - C3-E (free8, B1) | different action spaces; mean and interval only | descriptive |
| E1a/E1b/E1c | new batch, see `E1/E1_PROTOCOL.md` | split of A3 into shown reference and learned action content | exploratory, new batch |

C4R, C6-E and C6-EX each receive the same C4 response and revise it separately; C4R is not a step of C6-E.
C3-E alone is an external baseline, not a control with the LLM component removed.

## 3. Estimand and Strict rule

Policy value V(i, p, o) = Oracle_o(Simulator(x_i, a_i,p)) - Oracle_o(Simulator(x_i, A0)), the column
`delta_R_score_<oracle>`. Effect = mean over firms of V(i, treatment, o) - V(i, control, o).
Strict ITT as in the kit's `apply_itt_rule`: a completed response that fails validation runs as A0
(8,376 Strict cases); an infrastructure failure is retried and never becomes A0 (0 in the final observations).
Repaired is a re-processing of the same 48,300 responses and is never mixed with Strict in one row.

## 4. Statistics

- 575 firms paired one-to-one within model, budget, Oracle, execution rule, replicate and information condition
  (the C3-E side of REF-1 is a per-firm fixed value matched by firm only). Differences are ordered by `firm_key`.
- Bootstrap: firms resampled with replacement, 10,000 draws, PCG64; 95% interval from the 2.5 and 97.5 percentiles
  (linear interpolation); two-sided p = min(1, 2 (min(#draws <= 0, #draws >= 0) + 1) / 10,001).
- Seed: SHA-256 of the canonical key JSON (12 registry fields, sorted keys, no spaces, UTF-8), first 8 bytes big-endian.
  This rule reproduces the seeds of all 722 registry rows (checked in Q11).
- Core A1-A4 (96 rows): means recomputed and compared with the frozen means (`np.isclose`, atol = rtol = 1e-12); interval, p
  and Holm are the frozen values. Holm family: model x Oracle x generation setting, 8 contrasts (A1-A4 x B1, BINF).
  The 48 High rows were added after the Baseline results; the 60-comparison Holm p is always shown next to the main one.
- Exploratory families fixed before results: S1 one family of 24; REF-1 candidate9 one family of 12 (free8 rows untested);
  R9 per model x Oracle x generation setting over A2 and A3. `analysis_id`: `ABL_S1`, `ABL_REF1`, `ABL_REF1_FREE8`, `ABL_CAND9_BASELINE`.
- Not applicable values are written `N/A`, never 0.

## 5. Interpretation rules

| Situation | Write | Do not write |
| --- | --- | --- |
| interval contains 0 | "no difference was found" | "no effect", "equal" |
| A2 | "combined effect of review and output validation" | "judgement improved" |
| A3 | "increment of the added reference information" | "effect of the RL policy itself" |
| A3 significance | "positive in all 24 combinations, 9 after the main correction, 7 in the 60-comparison family" | "the RL reference always improves significantly" |
| S1 | "changing the generation-setting package changed ... by ..." | "reasoning ability produced the effect" |
| Strict differs from Repaired | "effect size depends on the execution rule" | calling either the true effect |
| REF-1 | "within the same 9 candidates C6-E was higher/lower than C3-E (exploratory; C3-E selected on 2024)" | "reached C3-E level", "equivalent to RL" |
| REF-1 free8 | state first that the action spaces differ | a superiority conclusion |
| representative condition | "free8, High, B1, Oracle-alpha is shown; the rest is in T1" | generalising from it |
| repetition | "mean direction mostly held in Run2, but which firms gain was hardly reproduced (rank correlation 0.06-0.29)" | "the effect holds for a type of firm" |
| effect size | "policy-value difference +0.100" | claims of practical importance, size comparison across Oracles |
| candidate-selection A3 | "11/12 positive, 2 significant after correction (both Baseline)" | "not significant in candidate selection" |
| Human baseline | "a separate external comparison" | combining it with ablation effects |

Differences between conditions (for example "A3 is larger in High") are claimed only when the interaction row's
interval excludes 0.

## 6. Row status

`FROZEN` (frozen result copied), `RECOMPUTED` (thesis number not in the registry, recomputed), `EXPLORATORY`
(new comparison), `DESCRIPTIVE` (no test), `MISSING` (combination not run), `FAIL` (QA failed). File status
(DRAFT, VERIFIED) is recorded in `QA_REPORT.md`.

## 7. Outputs

`results/T1_primary_A1_A4.csv` (96), `T2_S1.csv` (24), `T3_REF1.csv` (12 tested + 12 descriptive),
`T4_robustness.csv` (R1-R12 summary), `R9_candidate_selection.csv` (24), `R10_reference_assignment.csv` (575),
`figures/F1-F4` (PNG 300 dpi and SVG), `E1/` (protocol, requests, runs, scoring, T5 and F5 once collected),
`QA_REPORT.md`. Run `python ablation_20261102/scripts/run_all.py`.

## 8. Not done

No API calls other than E1, no RL retraining, no model or prompt change, no other new experiment, no change of
families after seeing results, no pooling of E1 with the frozen registry.
