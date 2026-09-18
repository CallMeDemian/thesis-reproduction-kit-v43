# LLM contract gap

The final structured evidence establishes 48,300 realized generations as two
24,150 request regimes over 575 firms. The source deployment tree has now been
forensically audited and contains the exact historical logical request
registries, provider request/output JSONL shards, generation ledgers, payload
sources, and normalized raw responses for both executed regimes. See
[`FORENSIC_ARTIFACT_INVENTORY.md`](../FORENSIC_ARTIFACT_INVENTORY.md) for the
measured paths, counts, sizes, and hashes.

The status is therefore **SOURCE_SNAPSHOT_FOUND_MIGRATION_PENDING**, not
`SOURCE_SNAPSHOT_UNRECOVERED`. The deterministic 48,300 plan in
`contracts/llm/final_as_executed_generation_contract.json` remains useful for a
fresh product plan, but it must not be used as a substitute for the exact
historical request registry once the source pack is migrated.

Consequences:

- dry rendering and uniqueness checks are available;
- live submission remains blocked until the source snapshot is migrated and
  independently reconciled;
- Strict and Repaired remain two evaluations of one raw response, never two
  provider calls;
- historical claims must continue to use the frozen lane.

The exact executed source pair is:

- baseline run `2cf6d6d0e4250e66ce882ab95f9d641f2c73711ffbc6429e9a203dcc2ee680a2`;
- high run `d5d878bf50718bac65ede73df3fc88c16723870b6c90a790557ff35b063b7ba7`.

Each has 24,150 logical requests, 24,150 provider inputs, 24,150 provider
outputs, and 24,150 normalized raw responses. Evidence searched during
migration included `repro/contracts`, final release registries,
`repro/release_inputs/configs/llm`, final-freeze contracts, the frozen
distribution registry, and `archive/DEPLOYED_RELEASE/llm_runs/final_plan3`.
