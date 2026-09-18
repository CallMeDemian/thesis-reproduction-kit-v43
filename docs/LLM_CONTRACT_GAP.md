# LLM contract gap

The final structured evidence establishes 48,300 realized generations as two
24,150 request regimes over 575 firms. The available repository also contains
a 42-cell matrix whose 42 × 575 cardinality is 24,150.

The exact historical provider request snapshot and raw parent-branch payloads
are not available locally. `repro/contracts/AS_EXECUTED_PROTOCOL.json` records
`SOURCE_SNAPSHOT_UNRECOVERED`, so the deterministic 48,300 plan in
`contracts/llm/final_as_executed_generation_contract.json` is a product plan
that proves cardinality, not a byte-identical historical replay.

Consequences:

- dry rendering and uniqueness checks are available;
- live submission is blocked until the source snapshot is restored;
- Strict and Repaired remain two evaluations of one raw response, never two
  provider calls;
- historical claims must continue to use the frozen lane.

Evidence searched during migration: `repro/contracts`, final release
registries, `repro/release_inputs/configs/llm`, final-freeze contracts, and the
frozen distribution registry.

