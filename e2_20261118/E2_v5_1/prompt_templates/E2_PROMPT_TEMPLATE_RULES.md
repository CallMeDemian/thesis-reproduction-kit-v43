# E2 v5.1 Prompt Template Rules

All four revision prompts use the E1 system prompt, action schema, output rules, company state, and exact archived C4 draft. The E1 condition-specific task lead-in is replaced by this common block:

> Review the exact earlier draft supplied below using the same firm state and action constraints. You may keep the draft unchanged, modify it, or replace it. Return one complete final recommendation, not a list of edits. If a reference program is supplied, consider whether it changes your recommendation and treat it as one possible program, not a verified best choice. Do not infer an outside recommendation.

- N: common firm/draft body, no reference block, no Simulator table.
- R: same common body plus the exact E1-RL `ADDITIONAL REFERENCE PROGRAM` block.
- T: N plus exactly these two sentences and the same 10-row Simulator table used in RT:

> Below is the change in financial-statement amounts one year after each action, calculated by the accounting simulator relative to no action. The effects may not be additive when multiple actions are applied together.

- RT: R plus the same table bytes as T.
- Q: original firm state and action definitions, without the original draft text, reference block, or Simulator table; the task contains nine fixed action vectors and forecasts for five financial metrics.

The 10 row order is A0, C4 draft (STRICT ITT A0 fallback when unusable), then the eight normalized one-axis candidates. Action vectors are rendered in original action-contract units. Monetary changes are rounded to the nearest KRW thousand using decimal half-up; source floats remain in B1.
