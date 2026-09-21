# Quickstart

Use Python 3.11 for the reference environment. The convenient supported
installation is:

```powershell
python -m pip install -e ".[full,dev]"
```

For the exact historical package versions, see:

`contracts/scientific/final_freeze/reproduction_environment_lock.json`

and

`contracts/scientific/final_freeze/requirements.reproduction.lock.txt`.

## Published result lane

```powershell
python -m thesis_repro verify
python -m thesis_repro reproduce --run-id quickstart_published
```

The expected contract is 96 primary rows, 722 supplemental rows, 96
parent-gate rows, and a 914-row result registry. No API calls or RL training
are performed.

For the complete reviewer path, see
[PROFESSOR_REPRODUCTION_GUIDE.md](PROFESSOR_REPRODUCTION_GUIDE.md).

## Optional historical assets

Use Git LFS and run `python -m thesis_repro verify-original` only when you need
to verify the multi-gigabyte original asset tree. The published result lane
does not require that full tree.

## Fresh replication

Full fresh execution is a separate workflow and requires external inputs and
resources. Start with `python -m thesis_repro reproduce --full --run-id
full_001`; see [FRESH_REPLICATION.md](FRESH_REPLICATION.md) before enabling
live provider execution.
