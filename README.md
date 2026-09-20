# V4.3 Thesis Reproduction Kit

An end-to-end product for frozen replay, fresh Oracle/RL/LLM replication, and explicit fresh-vs-frozen comparison for the V4.3 corporate credit recourse thesis.

## Four reviewer workflows

1. **Verify frozen thesis evidence** — `python -m thesis_repro verify` and `python -m thesis_repro verify-original` check the published artifacts only.
2. **Run synthetic acceptance** — `python -m thesis_repro acceptance-e2e --run-id ci-e2e` executes the real `VerifyInputs → Oracle → VerifyOracle → Simulator → RLDataset` prefix with deterministic fixture data; it intentionally stops as `PARTIAL_EXECUTION` and is never certification.
3. **Restore authorized raw inputs** — `python -m thesis_repro data restore ...` records the private-input receipt and contract hash.
4. **Run or continue fresh replication** — use `fresh --mode ... --profile full`; missing licensed inputs, GPU approval, and live-provider approval remain typed blockers.

## Quick start

```powershell
python -m pip install -e ".[full,dev]"
python -m thesis_repro doctor
python -m thesis_repro verify
python -m thesis_repro acceptance-e2e --run-id ci-e2e
python -m thesis_repro data doctor
python -m thesis_repro fresh --mode OracleClean --run-id oracle_smoke --profile smoke
python -m thesis_repro fresh --mode FullClean --run-id full_001 --plan
python -m thesis_repro compare --run-id oracle_smoke
```

## Fresh modes

`OracleClean`, `OracleRLClean`, `OracleRLLMClean`, and `FullClean` are explicit DAG targets. Use `--plan` to inspect stages and resources. Use `--resume`, `--from-stage`, and `--to-stage` for namespaced execution.

The smoke profile exercises the namespaced DAG, artifact ledger, input gate, resume invalidation, and a complete 48,300-request fresh dry render. Synthetic acceptance uses the same run engine and production adapters as a real run, but its fixture-scale RQ1 gate is explicitly inapplicable. Full scientific execution is permitted only after `data doctor` returns `INPUT_CONTRACT_PASS`; it never silently uses `frozen/` artifacts as compute parents. Live provider transport is separately gated by `THESIS_REPRO_ENABLE_LIVE_LLM=I_APPROVE_FRESH_REPLICATION`, and 28-actor retraining by `THESIS_REPRO_ENABLE_HEAVY_RL=I_APPROVE_28_ACTOR_RETRAIN`.

## Frozen lane

The frozen distribution is stored at `frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip` and is checked against its recorded SHA-256. It contains the canonical 96,600 Stage8 observations and the 914-row registry. The frozen lane never reads the source repository at runtime.

## Data preparation

Private inputs are not committed. Restore them with:

```powershell
python -m thesis_repro data restore `
  --raw-all C:\path\raw_all.zip `
  --raw-nonfinancial C:\path\raw_nonfinancial.zip `
  --ratings C:\path\rating_sample.zip
```

The restore receipt and file-level contract report record counts, sizes, and SHA-256 hashes under `data/raw/input_receipt.json` and `data/raw/input_contract_report.json`. A full fresh run is blocked with `INPUT_REQUIRED` until every required inventory entry is present and size-matched.

## Run directory

Every run is isolated under `runs/<run_id>/` with `00_run` through `15_release`, `logs`, a run manifest, recursive artifact hashes, parent hashes, stage status, input metadata, and comparison outputs. `runs/` and `data/raw/` are ignored by Git.

## Requirements

Python 3.11+. `.[data]`, `.[oracle]`, `.[rl]`, `.[llm]`, `.[full]`, and `.[dev]` separate optional dependencies. Full RL execution targets Torch 2.6.0 with CUDA 12.4 and records GPU/environment metadata in the run manifest. Provider credentials are supplied through environment variables and live execution requires an explicit gate.

## Scientific contract

The active scientific identity is V4.3: 9 candidate actions, 8 managerial dimensions, E2 encoder, four RL configurations, seven seeds, and a 575-firm cohort. Historical 24,150 and 34,500 design traces remain evidence only; the final evidence count is 48,300 raw generations and 96,600 Strict/Repaired Stage8 observations.

## Execution-state truth table

`PASS` means all required stages for the requested mode completed and all required scientific verifiers accepted. `PASS_WITH_QUALIFICATION` records a canonical qualified result. `PARTIAL_EXECUTION` is an intentionally incomplete prefix and is never a full-run pass. `FAILED`, `INPUT_REQUIRED`, `APPROVAL_REQUIRED`, `CREDENTIALS_REQUIRED`, `RESOURCE_REQUIRED`, `NOT_IMPLEMENTED`, `NOT_EXECUTED`, and `EXECUTED_UNVERIFIED` are never collapsed into `PASS_WITH_SKIPS`; that label is smoke-only. Synthetic acceptance is reported as `SYNTHETIC_E2E_ACCEPTANCE` with an acceptance-segment result of `PASS`, and cannot be certified.

## Scope

The repository is a product layer over frozen scientific evidence. Fresh numerical estimates are new replication outputs and are never copied into the frozen registry.

## Provenance

See [PROVENANCE.md](PROVENANCE.md), [frozen/release/frozen_manifest.json](frozen/release/frozen_manifest.json), and [MIGRATION_AUDIT.md](MIGRATION_AUDIT.md).
