# Scientific DAG closure

This document is the closure boundary for the clean-room product repository. It records which scientific nodes are replayable from the target repository, which are retained only as historical outputs, and which computation parents still exist only in the frozen source repository.

The machine-readable authority is [SCIENTIFIC_DAG_CLOSURE.json](../provenance/SCIENTIFIC_DAG_CLOSURE.json). The source repository remains read-only. Its three release commits are verified and its release refs are preserved in [REPRO_KIT_semantic_repair.bundle](../frozen/original_release/source/REPRO_KIT_semantic_repair.bundle).

## Closed evidence

- Oracle Stage0, Stage1 inputs, and Stage1 backends are copied and hash-verified.
- The selected C3-E pack contains 28 actor checkpoints and records every Stage3/4/5 hash edge in `member_provenance.csv`.
- The historical LLM final-plan-3 pack contains 48,300 reconciled logical generations across baseline and high regimes.
- The frozen Stage8 canonical source is 96,600 rows with SHA-256 `aedfadbf1728a14d3e4cb45bfa8a022d2c10313c52606e7bd1aee68899de8a0d`.
- The final registry contains 914 statistical rows and is part of the frozen evidence capsule.

## Closed selected parent chain

The product repository now contains the selected compute-parent artifacts for RL Stage2 (25 release-critical files), Stage3 (30 files), Stage4 (44 files), and Stage6 (5 files). The exact 28-member Stage3→Stage4→Stage5→Stage6 relationships are recorded in [RL_PARENT_GRAPH.json](../provenance/RL_PARENT_GRAPH.json), with 28/28 members hash-verified.

The simulator release pack contains the 24 simulator implementation files plus the release-critical action, financial, runtime, and input contracts. The full historical archive remains intentionally excluded; the selected runtime pack is the closure boundary for the preserved Stage8 lineage.

Fresh numerical execution still requires the authorized restricted raw inputs and has not been run automatically.

No full RL training or live LLM call is run by default. Those operations require explicit approval after the remaining parent edges are migrated and verified.
