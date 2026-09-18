# Migration audit

## Source examined

- final V4.3 release registries under `repro/releases/THESIS_V43_SUBMISSION_FINAL_20260917`
- frozen distribution `dist/THESIS_REPRO_KIT_v2.1.1_FINAL.zip`
- V1.3 evidence capsule under `repro_audit/FINAL_RELEASE`
- final-freeze scientific contracts and input inventory
- final release, Oracle, simulator, RL, Stage8, Stage9, and analysis modules

## Migrated

- frozen distribution ZIP and V1.3 evidence capsule
- release registries, claim registry, and final-freeze contracts
- curated execution packages under `src/credit_recourse`
- standalone `thesis_repro` CLI and run engine

## Intentionally excluded

- raw empirical inputs and provider credentials
- historical analysis/archive trees not needed by the product surface
- manuscript binaries whose exact bytes were unavailable
- live provider responses and the large historical request snapshot, which is
  now found in the source deployment tree and awaiting immutable asset
  migration

## Identity checks

- frozen distribution SHA-256: `931631d52541bb6dfdd5ed9fda767e2e48ccb8b7521f70ae771db68df2c1831d`
- Stage8 canonical SHA-256: `aedfadbf1728a14d3e4cb45bfa8a022d2c10313c52606e7bd1aee68899de8a0d`
- result registry: 914 rows
- raw LLM generations: 48,300

## Runtime dependency audit

The public CLI imports only `thesis_repro` and resolves paths from the product root. It does not import the old repository or use author-specific absolute paths. The curated legacy compatibility modules are not on the public CLI execution path; their remaining historical relative path strings are tracked for the next adapter pass.

Runtime dependency on the old repository: **0**.
