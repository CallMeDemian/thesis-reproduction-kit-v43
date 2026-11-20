# E2 v5.1 Pilot Preflight

## Fixed input checks

- Sample: 100 firms, stratified allocation `{'OE': 40, 'RF': 27, 'WC1': 10, 'CX': 7, 'WC2': 7, 'MX2': 6, 'MX1': 3}`.
- Logical requests: 1000 = 2 providers × 5 conditions × 100 firms; unique request IDs: 1000.
- Original C4 drafts strictly parsed: 2300; strict-invalid: 47; selected sample invalid firm/model drafts: 3.
- T/RT shared table input: 200 firm/model pairs × 10 rows = 2000 row appearances; T and RT text tables match byte-for-byte.
- Exact models/settings checked in request JSONL; pre-call metadata check confirms both exact model IDs are available. No generation smoke call was sent.
- New Oracle score calculation/reading is not part of this preflight.

## Simulator table integrity (selected 100 firms × 2 models)

- 200 tables / 2000 rows; every action valid and every Simulator call successful.
- Unit/horizon: all `KRW_thousand`, `decision_year_plus_one`; missing financial items max=0.
- Largest normalized action budget T=1.000000000041; no value exceeds 1 by more than 1e-8.
- Largest accounting identity residual: 1.52587891e-05 KRW thousand (Simulator check tolerance 10); A0 residual max=2.38418579e-07.
- Financial-effect statuses: `{'nonzero_financial_effect': 1525, 'zero_financial_effect': 275, 'A0_baseline_zero_change': 200}`. Zero effects are explicitly separate from `NOT_SIMULATED`; selected tables have no unavailable simulation rows.
- Amounts in prompts use signed KRW-thousand changes, rounded to integer thousands (Decimal half-up); raw source floats remain in the frozen B1 parquet.

## Historical-token cost estimate

| Provider | Input tokens est. | Output tokens est. | Point estimate | Historical p90 estimate | Single-attempt cap scenario |
|---|---:|---:|---:|---:|---:|
| openai | 2,365,482 | 1,738,272 | $4.80 | $9.26 | $37.93 |
| google | 2,945,883 | 1,482,078 | $1.48 | $1.90 | $12.73 |
| Total | 5,311,365 | 3,220,350 | $6.28 | $11.16 | $50.66 |

Rates: GPT-5.4 mini standard input/output $0.75/$4.50 per million; OpenAI Batch is 50% cheaper, so this estimate uses $0.375/$2.25. Gemini 3.1 Flash-Lite Batch paid-tier text input/output is $0.125/$0.75 per million. Retries are excluded from the point estimate; the three-attempt cap scenario is recorded in `preflight_cost_estimate.json`.

Pricing references: [OpenAI model pricing](https://developers.openai.com/api/docs/models/gpt-5.4-mini), [OpenAI Batch discount](https://developers.openai.com/api/docs/guides/batch), [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing?hl=en).

These are token-based paid-tier estimates, not an account invoice. Actual provider usage will be recorded from the batch responses.
