# V4.3 Thesis Reproduction Kit

An end-to-end product for frozen replay, fresh Oracle/RL/LLM replication, and explicit fresh-vs-frozen comparison for the V4.3 corporate credit recourse thesis.

## What you can do

- **Frozen Replay** — verify the published distribution, Stage8 identity, and 914-row result registry.
- **Fresh Replication** — run clean namespaced plans and smoke execution from restored inputs.
- **Comparison** — write a comparison report that distinguishes unavailable fresh values from frozen evidence.

## Quick start

```powershell
python -m pip install -e .
python -m thesis_repro doctor
python -m thesis_repro verify
python -m thesis_repro frozen-replay
python -m thesis_repro data doctor
python -m thesis_repro fresh --mode OracleClean --run-id oracle_smoke --profile smoke
python -m thesis_repro fresh --mode FullClean --run-id full_001 --plan
python -m thesis_repro compare --run-id oracle_smoke
```

## Fresh modes

`OracleClean`, `OracleRLClean`, `OracleRLLMClean`, and `FullClean` are explicit DAG targets. Use `--plan` to inspect stages and resources. Use `--resume`, `--from-stage`, and `--to-stage` for namespaced execution.

The smoke profile exercises the run engine with deterministic stage manifests. The full Oracle/RL adapters are being wired from the curated source port. The LLM lane renders a deterministic 48,300-request product plan; the exact historical provider snapshot has been found in the original deployment tree but remains blocked from live use until it is migrated and reconciled, as documented in [docs/LLM_CONTRACT_GAP.md](docs/LLM_CONTRACT_GAP.md).

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

The restore receipt records counts, sizes, and SHA-256 hashes under `data/raw/input_receipt.json`.

## Run directory

Every run is isolated under `runs/<run_id>/` with manifests, parent hashes, stage status, input metadata, and comparison outputs. `runs/` and `data/raw/` are ignored by Git.

## Requirements

Python 3.11+. Full RL execution targets Torch 2.6.0 with CUDA 12.4 and records GPU/environment metadata in the run manifest. Provider credentials are supplied through environment variables and live execution requires an explicit gate.

## Scientific contract

The active scientific identity is V4.3: 9 candidate actions, 8 managerial dimensions, E2 encoder, four RL configurations, seven seeds, and a 575-firm cohort. Historical 24,150 and 34,500 design traces remain evidence only; the final evidence count is 48,300 raw generations and 96,600 Strict/Repaired Stage8 observations.

## Scope

The repository is a product layer over frozen scientific evidence. Fresh numerical estimates are new replication outputs and are never copied into the frozen registry.

## Provenance

See [PROVENANCE.md](PROVENANCE.md), [frozen/release/frozen_manifest.json](frozen/release/frozen_manifest.json), and [MIGRATION_AUDIT.md](MIGRATION_AUDIT.md).
