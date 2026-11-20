# Prompt and information audit

Source release: `d5d878bf50718bac65ede73df3fc88c16723870b6c90a790557ff35b063b7ba7`

## Frozen request counts

```json
{
  "all_wave2_requests": 6900,
  "with_reference": 4025,
  "without_reference": 2875,
  "with_earlier_draft": 6900,
  "state_before_draft": 6900,
  "reference_after_state": 4025,
  "reference_source_label": 0,
  "contains_oracle_value": 0,
  "contains_q_value": 0,
  "state_key_count_min": 27,
  "state_key_count_max": 27,
  "dictionary_key_count_min": 27,
  "dictionary_key_count_max": 27,
  "payload_firms": 575,
  "payload_years": [
    2024
  ],
  "financial_fields_present": [
    "derived__debt_to_assets",
    "derived__equity_to_assets",
    "derived__current_ratio",
    "derived__cash_ratio",
    "derived__operating_margin",
    "derived__gross_margin",
    "derived__net_margin",
    "derived__roa_proxy",
    "derived__cogs_to_revenue",
    "derived__sga_to_revenue",
    "derived__financial_cost_to_revenue",
    "derived__capex_to_revenue",
    "derived__ppe_to_assets",
    "derived__short_debt_to_total_debt",
    "derived__long_debt_to_total_debt",
    "derived__bond_to_total_debt",
    "derived__inventory_to_revenue",
    "derived__receivables_to_revenue",
    "derived__payables_to_revenue",
    "delta_1y__derived__operating_margin",
    "delta_1y__derived__debt_to_assets",
    "delta_1y__derived__current_ratio",
    "delta_1y__derived__roa_proxy"
  ],
  "financial_fields_count": 23,
  "context_fields_present": [
    "fiscal_year",
    "log_assets",
    "market",
    "industry_class"
  ]
}
```

## What is established

- The frozen Stage7 wave-2 request payloads contain 23 financial fields plus fiscal_year, log_assets, market, and industry_class.
- C4R and reference conditions preserve the same FIRM STATE block; the additional reference appears after the earlier draft and state.
- The reference is presented as one possible program with no source label, policy score, Q value, confidence, or Oracle result in the model-visible text.
- C6-E versus C6-EX source identity is established from the frozen Stage7 metadata/reference_source and reference_candidate_id columns, not from a model-visible source label.
- A prompt-level search is reported as an audit of visible strings; it is not evidence that a model used or ignored a feature.

## Model-visible prompt excerpts

The exact wave-2 request text is preserved in the frozen provider batch JSONL files. The excerpts below are truncated only for readability.

### self_review

```text
TASK
Review the exact earlier draft supplied below using the same firm state and action constraints. No additional reference program is supplied. You may keep the draft unchanged, modify it, or replace it. Return one complete final recommendation, not a list of edits. Do not infer an outside recommendation.


INFORMATION DICTIONARY
[
  {
    "field": "derived__debt_to_assets",
    "definition": "Total liabilities divided by total assets; not interest-bearing debt divided by assets."
  },
  {
    "field": "derived__equity_to_assets",
    "definition": "Total equity divided by total assets."
  },
  {
    "field": "derived__current_ratio",
    "definition": "Current assets divided by current liabilities."
  },
  {
    "field": "derived__cash_ratio",
    "definition": "Cash and cash equivalent
```

### reference

```text
TASK
Review the exact earlier draft supplied below using the same firm state and action constraints. An additional reference program is now supplied. Consider whether it changes your recommendation. You may adopt it, incorporate part of it where the output action space permits, or reject it. You may also keep the draft unchanged. Return one complete final recommendation, not a list of edits.


INFORMATION DICTIONARY
[
  {
    "field": "derived__debt_to_assets",
    "definition": "Total liabilities divided by total assets; not interest-bearing debt divided by assets."
  },
  {
    "field": "derived__equity_to_assets",
    "definition": "Total equity divided by total assets."
  },
  {
    "field": "derived__current_ratio",
    "definition": "Current assets divided by current liabilities."
  
```

## Limitations

- The normalized raw-response artifact stores returned JSON, not the outgoing prompt; prompt ordering/source claims above come from the frozen provider-batch request JSONL.
- The prompt schema contains a prohibition against outputting scores, but no model-visible Oracle/Q/advantage values were found in the supplied state or reference blocks.