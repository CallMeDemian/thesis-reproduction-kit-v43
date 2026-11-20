# RL Advice-Taking Mechanism v2

## Scope and provenance

Frozen primary: 575 firms, FY2024, GPT-5.4-mini / Gemini 3.1 Flash-Lite, HIGH, STRICT ITT, FREE8, IC-b, B1, Run1. No new LLM request, RL retraining, or frozen-evidence edit was made.

Recomputed rows: 1150 firm-model observations. Recomputed type labels match the previous table: 1150/1150 for C6-E and 1150/1150 for C6-EX. Reference swap counts are 434 changed and 141 unchanged firms.

## Q1 — 141-firm negative control

- GPT: prompt/system/user/reference/settings equality = 1.000; raw response exact equality = 0.000; action exact equality = 0.660; behavior-type equality = 0.716; both valid = 0.965; one-sided fallback = 5/141; Alpha Δ mean=-0.041 [-0.116,+0.038].
- Gemini: prompt/system/user/reference/settings equality = 1.000; raw response exact equality = 0.000; action exact equality = 0.277; behavior-type equality = 0.631; both valid = 0.950; one-sided fallback = 7/141; Alpha Δ mean=-0.065 [-0.151,+0.017].

The model-visible prompt and generation configuration are identical in the same-reference pairs; the actual raw responses are not identical in any pair. This is compatible with stochastic generation. When the applied eight-axis actions are exactly equal, the frozen evaluator produces no nonzero Alpha score difference, so the observed same-reference score gaps are explained by different generated actions/fallback paths, not a deterministic scoring mismatch. A single run pair cannot estimate the full generation distribution.

Exploratory difference-in-differences (changed-reference mean minus same-reference mean): GPT/Alpha D=+0.206 [+0.084,+0.313]; GPT/Beta D=+0.053 [+0.006,+0.118]; GPT/Gamma D=+0.034 [-0.014,+0.084]; Gemini/Alpha D=+0.141 [+0.029,+0.252]; Gemini/Beta D=+0.007 [-0.005,+0.019]; Gemini/Gamma D=+0.015 [-0.024,+0.056]. This is a composition contrast, not a randomized causal effect.

## Q2 — value decomposition by Advice-Taking type

- GPT: direct_acceptance n=268, mean=+0.142, contribution=+0.066, partial_acceptance n=89, mean=+0.378, contribution=+0.058, combination n=8, mean=+0.312, contribution=+0.004, reject_or_maintain n=191, mean=-0.005, contribution=-0.002, unrelated_revision n=10, mean=-1.542, contribution=-0.027, opposite_direction n=9, mean=+0.000, contribution=+0.000
- Gemini: direct_acceptance n=88, mean=+0.097, contribution=+0.015, partial_acceptance n=262, mean=+0.086, contribution=+0.039, combination n=11, mean=+0.929, contribution=+0.018, reject_or_maintain n=147, mean=+0.000, contribution=+0.000, unrelated_revision n=16, mean=+0.267, contribution=+0.007, opposite_direction n=51, mean=+0.117, contribution=+0.010

The contribution check is exact up to floating-point rounding: the six type contributions sum to the model's overall C6-E−C4R mean for every Oracle.

## Q3/Q4 — RL value versus LLM-modified value

Direct acceptance is an exact RL-vector transfer: C6-E's Alpha/Beta/Gamma value equals the RL payoff surface (within numerical tolerance) for all direct-acceptance rows. Partial/combination rows are not systematically better than RL alone.
- GPT, partial+combination (n=97): C6-E > RL in Alpha 14, < RL 20, tie 63; mean C6-E−RL Alpha=-0.043.
- Gemini, partial+combination (n=273): C6-E > RL in Alpha 40, < RL 45, tie 188; mean C6-E−RL Alpha=-0.010.
- Combination is rare (n=19, 1.7%): Alpha C6-E−RL mean=-0.038, positive=3, negative=3, tie=13.

## Q5 — reference quality

Quality is post-hoc (RL payoff minus C4R score), not model-visible. In Alpha, acceptance (direct+partial+combination) was GPT 57.0% for favorable versus 28.9% for unfavorable references, but Gemini was 62.1% versus 64.8%. The controlled logistic table adjusts for exact RL action, C4 candidate, C4–RL distance, and C4 intensity; this is an exploratory association, not proof of score-aware selection.

## Q6 — GPT/Gemini

Paired firm-level tests show a strong type-composition difference (direct and partial categories move in opposite directions), while the binary accept-versus-not-accept contrast is not different (McNemar exact p≈0.83). Thus GPT/Gemini differ mainly in how they accept, not in the overall accept rate.

## Q7 — interpretation

The dominant label is **Advice-Induced Behavioral Change**, with a secondary **Policy Transfer** component. Direct copying transfers RL value exactly, but partial/combination behavior is rare and does not reliably exceed RL. The data do not support a broad **Policy Integration** claim or a causal claim that LLM reasoning selected the most suitable reference.

## Score resolution

Alpha ties are common even when actions differ: approximately 88% of C6-E versus C4R Alpha differences are zero in each model. Beta/Gamma frequently distinguish these action changes. See `SCORE_RESOLUTION_AUDIT.csv`.

## Files

The paired raw-prompt hashes, provider IDs, action vectors, fallback flags, score differences, decomposition, controls, model comparison, and figures are in this output directory. `NEGATIVE_CONTROL_PROMPT_PAIRS.csv` is an auxiliary row-level audit in addition to the required deliverables.