# Final research report — Financial-State-Conditioned Policy Advice Taking

## Scope

Primary frozen condition: HIGH, FREE8, IC-b, B1, STRICT ITT, Run1, 575 firms, GPT-5.4-mini and Gemini 3.1 Flash-Lite. No new LLM call, RL retraining, or frozen-evidence edit was used.

The analysis uses applied ITT vectors while preserving response usability/fallback flags. C4R and C6-E are treated as parallel branches from the same C4 parent, not a sequential intervention.

## Direct answers

- **Q1.** C6-E behavior is distributed across predeclared types: {'direct_acceptance': 356, 'partial_acceptance': 351, 'reject_or_maintain': 338, 'opposite_direction': 60, 'unrelated_revision': 26, 'combination': 19}. Exact copying is only the `direct_acceptance` cell; the continuous U and distance measures should be preferred over a binary interpretation.
- **Q2.** Feature-level associations are exploratory. `FINANCIAL_STATE_ASSOCIATIONS.csv` reports Spearman rho/p-values, BH correction, and bootstrap CIs for the outcome mean (not a causal effect CI); no result is called causal.
- **Q3.** Conditional rows stratify on the exact reference candidate. Sparse strata are reported as NA rather than pooled to manufacture precision.
- **Q4.** `ADVICE_QUALITY_RESPONSE.csv` separates favorable/unfavorable/tie by each post-hoc Oracle. This is a response–quality association, not a model-visible signal or proof of selective rationality.
- **Q5.** GPT and Gemini distributions are shown separately. A cross-model mechanism is only supported if both the behavior shares and quality/score relations align; the report does not infer a shared mechanism from one p-value.
- **Q6.** `ACTION_REVISION_MECHANISMS.csv` reports counts and Oracle deltas for direct, partial, combination, unrelated, reject, opposite, and rule-recovery paths. Any positive combination cell is not a causal mediation estimate.
- **Q7.** A contribution claim requires robust, pre-specified, cross-Oracle and cross-model evidence. These frozen observational analyses alone do not establish a new adaptive algorithm; negative or mixed findings should be reported as such.

## Descriptive results

- GPT: stable U=394/575, mean U=0.2887, median U=0.0000, exact C6-E/RL match=0.4661, Alpha(C6-E−C4R)=0.1003 [0.0384, 0.1638]
- Gemini: stable U=533/575, mean U=0.2499, median U=0.1520, exact C6-E/RL match=0.1530, Alpha(C6-E−C4R)=0.0897 [0.0439, 0.1401]
- Alpha quality response (acceptance = direct + partial + combination): GPT Alpha favorable: n=107, uptake-type acceptance=0.5701; GPT Alpha unfavorable: n=76, uptake-type acceptance=0.2895; Gemini Alpha favorable: n=116, uptake-type acceptance=0.6207; Gemini Alpha unfavorable: n=71, uptake-type acceptance=0.6479.
- 14 feature–outcome rows remain BH q<0.05 in the unconditioned exploratory screen; their small rho values and the non-random action assignment are why they are not interpreted as a financial-state mechanism. See the full table.
- The exact-reference conditional screen has 1 BH q<0.05 rows, concentrated in a few action strata (not a stable all-action or cross-model pattern); these are reported as conditional associations, not causal financial-state effects.
- U is a geometry score, not a probability. The denominator threshold is 1e-6 in normalized action units; those rows are marked `uptake_unstable` and excluded from feature correlations.
- Structured `evidence_fields` are retained in the firm table with added/removed/overlap counts relative to C4. Free text is retained as an auxiliary observable output only; no new judge was introduced.

## Reference swap

At the unique firm level the frozen permutation changes the reference candidate for 434 firms and leaves it unchanged for 141; model rows duplicate each firm for the two model branches. `REFERENCE_SWAP_ANALYSIS.csv` keeps these groups separate.

## Prompt audit conclusion

The outgoing Stage7 request JSONL confirms 23 financial fields plus year/assets/market/industry, the earlier C4 draft, and the additional reference ordering. No model-visible Oracle score, Q value, advantage, or confidence value was found in the reference/state blocks. See `PROMPT_INFORMATION_AUDIT.md`.

## Reproducibility

Run `python analysis/policy_advice_taking_financial_state.py` with the frozen runtime and completed Stage8 all-populations file. The script writes all requested CSVs, figures, and this report under `analysis/policy_advice_taking_financial_state/`.