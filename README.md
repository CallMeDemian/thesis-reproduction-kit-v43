# V4.3 Thesis Reproduction Kit

For thesis reviewers and professors, start with
[docs/PROFESSOR_REPRODUCTION_GUIDE.md](docs/PROFESSOR_REPRODUCTION_GUIDE.md).

This repository keeps frozen evidence separate from fresh computation and
exposes four explicit workflows. The recommended published-result lane does
not require Git LFS, private raw data, RL retraining, or paid API calls.

## 1. Published result reproduction

```powershell
python -m pip install -e ".[full,dev]"
python -m thesis_repro verify
python -m thesis_repro reproduce --run-id professor_check
```

Expected published counts are 96 primary rows, 722 supplemental rows, 96
parent-gate rows, and a 914-row result registry. Historical Stage8 evidence
contains 96,600 rows. This lane makes no provider calls and does not retrain
RL; it recomputes the published firm-level result layer against frozen
production streams.

### Optional: original historical asset verification

The deeper historical-asset lane requires Git LFS. After a complete LFS clone,
run `python -m thesis_repro verify-original` and, when needed,
`python -m thesis_repro rebuild-c3e --release original`. It verifies the
migrated Oracle, 28 historical RL actors, LLM, and evaluation packs.

## 2. Synthetic full-DAG acceptance

CI runs the real `run_engine` and canonical stage adapters against deterministic
fixture inputs:

```powershell
python -m thesis_repro acceptance-e2e --run-id ci-synthetic-full
```

This is architectural evidence only. It is marked
`SYNTHETIC_E2E_ACCEPTANCE`, is never certifiable, and is not thesis-scale
scientific evidence.

## 3. Restore authorized inputs

Restore the licensed raw inputs first:

```powershell
python -m thesis_repro data restore `
  --raw-all C:\path\raw_all.zip `
  --raw-nonfinancial C:\path\raw_nonfinancial.zip `
  --ratings C:\path\rating_sample.zip `
  --stage2-source C:\path\stage2_producer_input_pack.zip
```

The raw archives support the rebuilt Oracle and raw Stage2 action source. Full
RL continuation additionally requires the exact original Stage2 producer-input
pack because it was not distributed with the certified source release. The
exact frozen C6-EX donor permutation is bundled in the certified distribution
and restored automatically after SHA-256 validation. The
`--c6ex-permutation` option remains an optional exact-byte override; a present
but tampered override fails closed.

## 4. Run or continue full fresh replication

```powershell
python -m thesis_repro reproduce --full --run-id full_001
```

Without private inputs, a full run stops at `INPUT_REQUIRED`. With raw inputs
but no Stage2 producer pack it stops at
`STAGE2_PRODUCER_INPUT_PACK_REQUIRED` after attempting the fresh raw
action-source producer.

Use `--live-llm` only when paid provider calls are authorized:

```powershell
python -m thesis_repro reproduce --full --live-llm --run-id full_001
python -m thesis_repro reproduce --full --live-llm --resume --run-id full_001
```

## RL search from a completed fresh run

```powershell
python -m thesis_repro search --rl `
  --catalog C:\path\RL_Hyperparameter_Search.xlsx `
  --source-run full_001 `
  --run-id search_001
```

The search path runs the preserved Stage5 runtime against an explicit original
catalog and a completed same-run fresh baseline. It does not invent a search
result when either external requirement is unavailable.

## Scientific identity and provenance

The active V4.3 contract has eight managerial action dimensions and nine
candidate actions: `A0`, `DL`, `RF`, `CX`, `WC1`, `WC2`, `OE`, `MX1`, and
`MX2`. Fresh runs record source state, input hashes, stage manifests, artifact
hashes, and recursive same-run parent lineage under `runs/<run_id>/`.

The published lane is certified frozen evidence. The full lane rebuilds
Oracle, Stage2, RL, C3-E, Stage6, LLM, Stage8, Stage9, and final outputs in a
fresh namespace. `trace` and `certify` are evidence commands; synthetic and
smoke runs cannot be certified.

Private inputs are not committed. Run `python -m thesis_repro doctor` to
inspect installed dependencies and input availability.
