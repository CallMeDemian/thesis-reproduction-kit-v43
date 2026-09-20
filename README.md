# V4.3 Thesis Reproduction Kit

This repository exposes three executable workflows and keeps frozen evidence separate from fresh computation.

## 1. Published result reproduction

```powershell
python -m pip install -e ".[full,dev]"
python -m thesis_repro verify
python -m thesis_repro verify-original
python -m thesis_repro reproduce --run-id published
```

This reproduces the certified frozen evidence lane, including the 96 primary rows, 722 supplemental rows, 96 parent rows, and 914-row result registry. It does not claim a fresh numerical rerun.

## 2. Full experimental reproduction

Restore the licensed raw inputs first:

```powershell
python -m thesis_repro data restore `
  --raw-all C:\path\raw_all.zip `
  --raw-nonfinancial C:\path\raw_nonfinancial.zip `
  --ratings C:\path\rating_sample.zip
python -m thesis_repro reproduce --full --run-id full_001
```

Use `--live-llm` only when paid provider calls are authorized:

```powershell
python -m thesis_repro reproduce --full --live-llm --run-id full_001
```

Full execution is fresh and run-namespaced. Licensed inputs, sufficient compute/GPU, and provider credentials are external requirements; frozen numerical outputs are never substituted as fresh parents.

## 3. Original search / model-selection reproduction

```powershell
python -m thesis_repro search --rl --catalog C:\path\RL_Hyperparameter_Search.xlsx
python -m thesis_repro search --encoder --catalog C:\path\encoder_search.csv
```

The search path runs the preserved search runtime against an explicit original catalog. It does not invent a search result when that external catalog is unavailable.

## Scientific identity and provenance

The active V4.3 contract has eight managerial action dimensions and nine candidate actions: `A0`, `DL`, `RF`, `CX`, `WC1`, `WC2`, `OE`, `MX1`, and `MX2`. Fresh runs record source state, input hashes, stage manifests, artifact hashes, and recursive same-run parent lineage under `runs/<run_id>/`.

The published lane is certified frozen evidence. The full lane rebuilds Oracle, Stage2, RL, C3-E, Stage6, LLM, Stage8, Stage9, and final outputs in a fresh namespace. Synthetic fixture execution is available to CI internally and is never certifiable or presented as thesis-scale evidence.

Private inputs are not committed. Run `python -m thesis_repro doctor` to inspect installed dependencies and input availability.
