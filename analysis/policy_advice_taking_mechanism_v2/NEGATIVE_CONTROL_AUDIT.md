# Negative-control audit: same RL reference in C6-E and C6-EX

This audit links the frozen primary logical requests to the provider input JSONL, normalized response records and Frozen Stage8 scores. It does not issue new model calls.

## Audit scope

- 575 firms, FY2024; GPT-5.4-mini and Gemini 3.1 Flash-Lite; HIGH / STRICT ITT / FREE8 / IC-b / B1 / Run1.
- Same-reference group: 141 firms; changed-reference group: 434 firms.
- Both the original prompt hash and a whitespace-collapsed normalized hash are retained in `NEGATIVE_CONTROL_PROMPT_PAIRS.csv`; numeric/action content was not stripped.

## Same-reference results

- **GPT (n=141)**: prompt/system/user/reference/settings exact equality 141/141; whitespace-normalized equality 141/141; raw response exact equality 0/141; structured JSON equality 0/141; action-vector equality 93/141; behavior-type equality 101/141; both action rules valid 136/141; one-sided fallback 5; Alpha Δ mean -0.0410 (bootstrap 95% CI -0.1164, +0.0383); nonzero score differences among same-action + valid pairs: 0.
- **Gemini (n=141)**: prompt/system/user/reference/settings exact equality 141/141; whitespace-normalized equality 141/141; raw response exact equality 0/141; structured JSON equality 0/141; action-vector equality 39/141; behavior-type equality 89/141; both action rules valid 134/141; one-sided fallback 7; Alpha Δ mean -0.0651 (bootstrap 95% CI -0.1512, +0.0169); nonzero score differences among same-action + valid pairs: 0.

## What was actually delivered to the model

The pair-level table stores system and user message hashes, full request-body hashes, reference-block hashes, generation settings, model IDs, provider/shard provenance, retry class, provider request IDs, response IDs/timestamps when supplied, and raw-response hashes. The 23 financial-state fields plus fiscal year, log assets, market and industry are part of the user message; the reference action ID/axis values/description and C4 draft are therefore included in the exact prompt comparison.

Internal provenance/value labels (`reference_source`, candidate/source IDs, Oracle/Q/advantage metadata) were not detected in the model-visible prompt. The prompt contains generic task words such as “reference program”; those are not treated as hidden provenance leakage. Temperature was not present in the stored provider settings; Google stored `thinkingLevel=high`, while OpenAI stored `reasoning.effort=high` and model ID `gpt-5.4-mini-2026-03-17`.

## Interpretation

For the 141 same-reference pairs, the prompt and generation configuration are identical but no raw response pair is exactly equal. This is compatible with generation variability. When the applied eight-axis action is exactly equal and both rows are valid, the Frozen evaluator gives no nonzero score difference; the remaining score gaps therefore follow different generated actions or fallback paths, not a deterministic evaluator mismatch. A single pair of generations cannot identify the full response distribution or prove a unique causal source of variability.

Row-level evidence: `NEGATIVE_CONTROL_PROMPT_PAIRS.csv` and `REFERENCE_SWAP_PAIRED_ANALYSIS.csv`.