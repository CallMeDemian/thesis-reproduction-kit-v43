# Forensic Artifact Inventory

Audit date: 2026-09-19 (Asia/Seoul)

This inventory is the first clean-room migration gate for the V4.3 thesis
reproduction repository. The original repository at
`C:\Users\Demian\Desktop\REPRO_KIT_semantic_repair` was read only and was not
modified. The target repository is
`C:\Users\Demian\Desktop\thesis-reproduction-kit-v43`.

## Status meanings

- **FOUND** — the artifact is present in the original deployment/release tree
  and its identity can be checked by path, count, size, row count, or SHA-256.
- **PARTIAL** — a release-derived or frozen representation is present, but the
  complete original artifact family, raw inputs, or environment is not yet
  migrated into this repository.
- **MISSING** — the requested object was not available in the local source
  snapshot during this audit. This does not claim that it cannot exist outside
  the local machine.

## Executive finding

The source release is substantially more complete than the first migration
audit indicated. In particular, the exact historical LLM request and response
artifacts are present in `archive\\DEPLOYED_RELEASE\\llm_runs\\final_plan3`.
The current gap is migration and release packaging, not source recoverability.

The original deployment tree contains 1,436 files totalling 5,072,088,366
bytes. The main LLM final-plan tree contains 287 files totalling 2,995,065,654
bytes. These large trees are intentionally not copied into Git history during
this forensic pass; they must be migrated as immutable release assets or Git
LFS packs with a manifest and per-file hashes.

## Artifact inventory

| Area | Status | Original source / evidence | Measured identity | Migration decision |
|---|---|---|---|---|
| Frozen v2.1.1 distribution | FOUND | `dist\\THESIS_REPRO_KIT_v2.1.1_FINAL.zip` | 47,741,133 bytes; SHA-256 `931631d52541bb6dfdd5ed9fda767e2e48ccb8b7521f70ae771db68df2c1831d` | Already preserved at `frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip`; retain as immutable baseline. |
| Frozen Stage8 canonical evidence | FOUND | `repro_audit\\FINAL_RELEASE\\THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN\\frozen_inputs\\stage8\\canonical_itt_observations.parquet` | 96,600 rows; 10,382,945 bytes; SHA-256 `aedfadbf1728a14d3e4cb45bfa8a022d2c10313c52606e7bd1aee68899de8a0d` | Already preserved inside the frozen evidence capsule and distribution ZIP; this is the authority for published replay. |
| Oracle deployment family | FOUND | `archive\\DEPLOYED_RELEASE\\stage0_oracle_foundation`, `stage1_oracle_inputs`, `stage1_oracle_backends` | Deployment total 1,436 files / 5,072,088,366 bytes. Stage0: 7 files / 124,411,620 bytes. Stage1 inputs: 170 files / 178,749,154 bytes. | Preserve the complete tree as a hashed release pack. Keep compact contracts/manifests in Git; do not replace production outputs with synthetic fixtures. |
| Oracle canonical panel | FOUND | `archive\\DEPLOYED_RELEASE\\stage0_oracle_foundation\\canonical_panel` | `stage0_canonical_panel.parquet`: 1,010,628 bytes, SHA-256 `b59122562876bdf6c2be253942eb8646b660826a68f6f71014364c595f7f16e1`; `statement_items_panel.parquet`: 122,427,505 bytes, SHA-256 `c52c08500034c8340dbcfc42d1fb4d439c8103347334c33a20f3f793598ddfd8` | Include in the Oracle release pack and manifest. |
| Oracle alpha/beta/gamma outputs | FOUND | `archive\\DEPLOYED_RELEASE\\stage1_oracle_backends\\{alpha,beta,gamma}` | Alpha output: 480,563 bytes, SHA-256 `9eb0fcf5bb64de58efa86034e741e87bb067a9eb6c5afcee3039f3d4fb5513e1`; beta: 715,028 bytes, SHA-256 `24fc7d63e75c45eaa92ffb79a20151f6a9367636b9f005b0cb6fe282fa6d3a06`; gamma: 383,316 bytes, SHA-256 `1b7d04e3b12a7faef36cf3408269b6f9c4bcb128cd234ffa7a5b927720933ef4` | Include all backend parameters, diagnostics, and output tables in the immutable Oracle pack. |
| RL C3-E final release | FOUND | `archive\\DEPLOYED_RELEASE\\stage5_candidate_iql\\C3E_E2_7SEED_BALANCED_DFEBAFA6` | 205 files / 475,155,224 bytes; exactly 28 `.pt` actor checkpoints / 470,944,432 bytes. Four configurations × seven seeds are present. | Preserve all 28 checkpoints and their provenance in a release asset/LFS pack. The compact C3-E contracts already migrated into the new repo are not sufficient by themselves. |
| RL C3-E definition and outputs | FOUND | Same C3-E tree | `C3E_definition.json`: 6,686 bytes, SHA-256 `768ddb63544ab581a52836852c03debed408fb57f80c9781f2564fb5537c4e26`; `C3E_firm_actions.parquet`: 84,535 bytes, SHA-256 `ced89c428deaab8eeaebbceece95ccd40c86eab194ce4d5c03d291198b3d975a`; `C3E_firm_probabilities.parquet`: 57,492 bytes, SHA-256 `d72e1386d5376c9f5a2da3c5e4a8e117ae5e38756ecc22fe2778dafd0f4cf6c9`; `C3E_metrics.json`: 2,342 bytes, SHA-256 `9016eb5b4ae1620d1158d6f77360b0334306e2dc60bbd49bd025a52f1a14c2c3` | Keep these as the release-level identity files and cross-check them against the checkpoint pack. |
| LLM logical request registries | FOUND | `archive\\DEPLOYED_RELEASE\\llm_runs\\final_plan3\\2cf6...` and `d5d...` | Baseline `logical_requests.parquet`: 24,150 rows, 2,055,748 bytes, SHA-256 `8bf0d112c26fe294466be755587ce6a7eafe9c8053e0b81cad8fd161a5b51989`; high: 24,150 rows, 3,674,261 bytes, SHA-256 `55febcf3f62f8ef98a2216671fea4a92ec87e7ac570c850c16692f9ca0d8cc36` | Migrate byte-for-byte into an original LLM release pack; do not reconstruct IDs from the 42-cell product formula. |
| LLM provider request records | FOUND | `final_plan3\\{run}\\provider_batches\\wave{1,2}` | Each executed regime has 54 input JSONL files with exactly 24,150 request lines and 54 output JSONL files with exactly 24,150 output lines. The two regimes therefore account for 48,300 provider request records and 48,300 provider outputs. | Preserve all provider input/output JSONL shards, including provider/model/arm path structure and batch metadata, in the LLM release asset pack. |
| LLM normalized raw responses | FOUND | `final_plan3\\{run}\\raw_responses` | Baseline wave1: 10,350 lines, 22,291,992 bytes, SHA-256 `cb94342b949ec1a159e9a5bffbb0cc33fb8cf02d4e14fa53dcb71a10e76af1b1`; baseline wave2: 13,800 lines, 28,728,363 bytes, SHA-256 `3e2b61ca8a6323f4e6934135dbb0f9dc122afc18074adcf6f3548a9d4d1145a6`; high wave1: 10,350 lines, 21,209,461 bytes, SHA-256 `f6bc75d82bb7c233590eaa7977f299f508c15f75c587786e78c26d5b797d13f1`; high wave2: 13,800 lines, 27,765,723 bytes, SHA-256 `080f3470a57f5415cb5445ae1f82d464356d8e4021c947b8cf3be1767790389` | Preserve the normalized files and reconcile every `request_id` to the corresponding logical request and provider record. |
| LLM generation ledgers / payload sources | FOUND | `final_plan3\\{run}\\generation_ledger.sqlite`, `firm_payload_source.parquet` | Baseline ledger 591,667,200 bytes; high ledger 975,618,048 bytes. Per-run logical request and payload source files are present. | Migrate as immutable SQLite/Parquet assets; add integrity checks without rewriting SQLite contents. |
| Stage8 evaluation family | FOUND | `data\\reproduction\\llm_final_evaluation_20260913` | 240 stage8/stage9-named paths, 515,657,017 bytes in the source tree. Noncanonical all-firm result: 184,803 rows, 16,260,714 bytes, SHA-256 `32d5dc319256263a133ad02da502b6e3f07a5b69780fee29f844c2bc1b0a6eae`. Canonical frozen Stage8 is the 96,600-row artifact above. | Preserve canonical Stage8 as authority and retain noncanonical/diagnostic variants in a labeled evaluation pack; never silently substitute the 184,803-row file. |
| Stage9 comparison / revision outputs | FOUND | `data\\reproduction\\llm_final_evaluation_20260913\\stage9` | 76 files / 320,187,946 bytes. Four explicit arms exist: `BASELINE_STRICT_ITT`, `BASELINE_REPAIRED_ITT`, `HIGH_STRICT_ITT`, `HIGH_REPAIRED_ITT`. Primary comparison Parquet row counts are 52,824; 60,931; 60,700; and 60,948 respectively. | Preserve all four arms and their manifests. The new repo must identify which Stage9 files are canonical, diagnostic, or supplemental before release. |
| Result Registry | FOUND | Frozen distribution member and new-repo evidence capsule | 914 registry rows; distribution member SHA-256 `6d43643da844315c3a531cdb07c5dddc8f5fef5fc91b00f117f830b6c95e1b56` | Already migrated as frozen evidence through the exact distribution. Add a standalone migration manifest entry and retain the registry as immutable. |
| Raw licensed / external input data | PARTIAL | Source manifests and prepared deployment outputs are present; raw private/licensed inputs are not committed to the new repo | Completeness depends on the source-side input receipts and licensing terms; no byte-complete raw-data pack was identified in this pass. | Keep restore-by-receipt workflow. Add raw-input manifest, access instructions, and validation hashes; do not publish restricted data accidentally. |
| Original source commit `324b5fbb...` | MISSING | Recorded in `provenance/source_release.json`, object not available in the local source Git object database | Local verification: `UNAVAILABLE` | Keep as a provenance claim requiring external Git verification; do not represent it as locally reproduced. |
| Certification commit `e2ad7fb7...` | MISSING | Recorded in `provenance/source_release.json`, object not available locally | Local verification: `UNAVAILABLE` | Same treatment: preserve the claim and mark the local object gap. |
| Scientific evidence base commit `449c942c...` | FOUND | Source Git object database and migrated evidence metadata | Local verification: `AVAILABLE` | Retain as the evidence-base provenance anchor. |

## LLM reconciliation result

The exact source-level cardinality is now independently observed rather than
inferred from the 42-cell product plan:

| Regime | Logical requests | Provider inputs | Provider outputs | Normalized raw responses |
|---|---:|---:|---:|---:|
| Baseline (`2cf6d6d0...`) | 24,150 | 24,150 | 24,150 | 24,150 |
| High (`d5d878bf...`) | 24,150 | 24,150 | 24,150 | 24,150 |
| **Total** | **48,300** | **48,300** | **48,300** | **48,300** |

The third `final_plan3` directory contains a logical request registry but was
not counted as one of the two executed 24,150-response regimes because the raw
response pair was not found there during this audit. It is retained as a
separate historical/diagnostic artifact until its role is resolved.

## Immediate migration actions

1. Replace the old LLM “source snapshot unrecovered” wording with “source
   snapshot found; migration pending”. The live-call gate remains closed until
   the migrated pack is reconciled.
2. Create `provenance/MIGRATION_MANIFEST.json` with one record per migrated
   file, including original relative path, new relative path, size, old SHA-256,
   new SHA-256, semantic role, and migration action.
3. Package the full Oracle tree, all 28 C3-E checkpoints, and the two exact LLM
   executed regimes as immutable release assets or Git LFS packs. Git history
   should contain manifests and compact contracts, not a silently truncated
   subset of the scientific artifacts.
4. Add the canonical Stage9 selection manifest and reconcile all Stage8/Stage9
   request IDs to the migrated LLM request registry.

This document is an inventory and migration gate, not a claim that the new
repository is already a byte-complete copy of the source release.
