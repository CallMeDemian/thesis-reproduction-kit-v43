# V4.3 Thesis Reproduction Kit

This repository keeps frozen evidence separate from fresh computation and exposes four explicit workflows.

## 1. Published result reproduction

```powershell
python -m pip install -e ".[full,dev]"
python -m thesis_repro verify
python -m thesis_repro verify-original
python -m thesis_repro reproduce --run-id published
```

This reproduces the certified frozen evidence lane, including the 96 primary rows, 722 supplemental rows, 96 parent rows, and 914-row result registry. It does not claim a fresh numerical rerun.

## 2. Synthetic architecture acceptance

CI runs the real `run_engine` and canonical stage adapters against deterministic fixture inputs:

```powershell
python -m thesis_repro acceptance-e2e --run-id ci-synthetic-full
```

This is architectural evidence only. It is marked `SYNTHETIC_E2E_ACCEPTANCE`, is never certifiable, and is not thesis-scale scientific evidence.

## 3. Restore authorized inputs

Restore the licensed raw inputs first:

```powershell
python -m thesis_repro data restore `
  --raw-all C:\path\raw_all.zip `
  --raw-nonfinancial C:\path\raw_nonfinancial.zip `
  --ratings C:\path\rating_sample.zip `
  --stage2-source C:\path\stage2_producer_input_pack.zip `
  --c6ex-permutation C:\path\C6EX_permutation.parquet
```

The raw archives support the rebuilt Oracle and raw Stage2 action source. Full RL continuation additionally requires the exact original Stage2 producer-input pack because it was not distributed with the certified source release. Live Plan-3 LLM execution additionally requires the exact historical C6-EX permutation design input.

## 4. Run or continue fresh replication

```powershell
python -m thesis_repro reproduce --full --run-id full_001
```

Without private inputs, a full run stops at `INPUT_REQUIRED`. With raw inputs but no Stage2 producer pack it stops at `STAGE2_PRODUCER_INPUT_PACK_REQUIRED` after attempting the fresh raw action-source producer.

Use `--live-llm` only when paid provider calls are authorized:

```powershell
python -m thesis_repro reproduce --full --live-llm --run-id full_001
```

Use `--live-llm` only when paid provider calls are authorized, and `--resume` to continue the exact provider jobs:

## RL search from a completed fresh run

```powershell
python -m thesis_repro search --rl `
  --catalog C:\path\RL_Hyperparameter_Search.xlsx `
  --source-run full_001 `
  --run-id search_001
```

The search path runs the preserved Stage5 runtime against an explicit original catalog and a completed same-run fresh baseline. It does not invent a search result when either external requirement is unavailable.

## Scientific identity and provenance

The active V4.3 contract has eight managerial action dimensions and nine candidate actions: `A0`, `DL`, `RF`, `CX`, `WC1`, `WC2`, `OE`, `MX1`, and `MX2`. Fresh runs record source state, input hashes, stage manifests, artifact hashes, and recursive same-run parent lineage under `runs/<run_id>/`.

The published lane is certified frozen evidence. The full lane rebuilds Oracle, Stage2, RL, C3-E, Stage6, LLM, Stage8, Stage9, and final outputs in a fresh namespace. `trace` and `certify` are evidence commands; synthetic and smoke runs cannot be certified.

Private inputs are not committed. Run `python -m thesis_repro doctor` to inspect installed dependencies and input availability.
