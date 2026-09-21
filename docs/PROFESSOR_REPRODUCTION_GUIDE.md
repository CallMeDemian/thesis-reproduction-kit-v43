# Professor reproduction guide

This guide separates three different meanings of “reproduction.” The first
is the recommended reviewer workflow.

## Level A — published result reproduction (recommended)

This verifies and recomputes the published scientific result layer from
certified frozen evidence. It does not retrain RL and does not make 48,300 new
provider calls.

```powershell
python -m thesis_repro verify
python -m thesis_repro reproduce --run-id professor_check
```

Expected published contract:

- Primary results: 96
- Supplemental results: 722
- Parent-gate results: 96
- Result registry: 914
- Historical Stage8 evidence: 96,600 rows

Inspect the receipt at
`runs/professor_check/final/REPRODUCTION_RECEIPT.json`. A successful receipt
reports `estimate_source=RECOMPUTED_FIRM_LEVEL_PRIMARY_ESTIMATE` and
`bootstrap_ci_source=FROZEN_PRODUCTION_STREAM`. Thus this is stronger than
copying thesis table values, but it is not a new raw-data, RL, or LLM
experimental campaign. API calls and training runs are zero.

## Level B — original historical asset verification

This deeper lane verifies the migrated historical Oracle, RL, LLM, and
evaluation asset packs, including 28 historical RL actors. It requires Git
LFS and several gigabytes of objects:

```powershell
git clone https://github.com/CallMeDemian/thesis-reproduction-kit-v43.git
cd thesis-reproduction-kit-v43
git lfs install
git lfs pull
git lfs fsck
python -m thesis_repro verify-original
python -m thesis_repro rebuild-c3e --release original
```

The last command is an optional original-release C3-E rebuild. A GitHub source
archive is not equivalent to a clone with all LFS objects.

## Level C — full fresh experimental replication

The conceptual fresh DAG is:

```text
raw inputs → Oracle → Stage2 / Simulator → RL → C3-E → Stage6
→ LLM → Stage8 → Stage9 → thesis outputs → comparison → certification
```

The supported command is:

```powershell
python -m thesis_repro reproduce --full --run-id full_001
```

For authorized paid live generation:

```powershell
python -m thesis_repro reproduce --full --live-llm --run-id full_001
python -m thesis_repro reproduce --full --live-llm --resume --run-id full_001
```

Full fresh execution may require:

- licensed raw financial and nonfinancial inputs;
- the unrecovered original Stage2 producer-input pack;
- GPU/compute resources for fresh 28-actor training;
- OpenAI/Gemini credentials and paid calls;
- asynchronous provider completion; and
- the historical RL search catalog and exact selection-rule artifact.

The exact frozen C6-EX donor permutation is not an external boundary: it is
bundled in the certified distribution and restored automatically after
SHA-256 validation. Fresh execution still must not use frozen computed outputs
as fresh scientific parents.

## Environment and scope

Use Python 3.11 for the reference environment. The convenient supported
install is:

```powershell
python -m pip install -e ".[full,dev]"
```

The exact reference versions are recorded in
`contracts/scientific/final_freeze/reproduction_environment_lock.json` and
`contracts/scientific/final_freeze/requirements.reproduction.lock.txt`.

The published lane is the appropriate first check for a reviewer. Full fresh
execution is not claimed complete merely because the code path is present;
its external boundaries are reported explicitly by the run manifest.
