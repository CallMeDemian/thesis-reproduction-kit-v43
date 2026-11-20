# Prompt Diff and Invariants

- E1-C4R and E1-RL task preambles were not identical. They were commonized to the single shared E2 instruction recorded in `prompt_templates/COMMON_TASK.txt`; the original E1 prompts remain unchanged in the release inputs.
- The E1-RL reference block is preserved byte-for-byte and appears only in R/RT.
- T/RT differ from their corresponding N/R body only by the exact two-sentence caption and the shared 10-row Simulator table.
- The table has no Oracle scores, rating predictions, evaluator weights, ranks, P3/P5 labels, or policy-value figures. The separate Oracle crosswalk is analysis-only.
- The 1,000 request bodies use two exact model IDs, `high` reasoning, 32,768 output-token maximum, and independent request IDs.

| Provider | Condition | Requests |
|---|---:|---:|
| google | N | 100 |
| google | Q | 100 |
| google | R | 100 |
| google | RT | 100 |
| google | T | 100 |
| openai | N | 100 |
| openai | Q | 100 |
| openai | R | 100 |
| openai | RT | 100 |
| openai | T | 100 |
