# Clean-clone acceptance

## Path A — published result reproduction

This path does not require the original multi-gigabyte LFS tree:

```powershell
python -m pip install -e ".[full,dev]"
python -m thesis_repro verify
python -m thesis_repro reproduce --run-id clean-clone-published
```

Expected result: PASS with 96 primary rows, 722 supplemental rows, 96
parent-gate rows, and 914 registry rows.

## Path B — synthetic full-DAG architecture

```powershell
python -m thesis_repro acceptance-e2e --run-id clean-clone-synthetic
python -m thesis_repro trace --run-id clean-clone-synthetic
```

Expected result:

- `completion_state=PASS`
- `dag_complete=true`
- `certifiable=false`
- `lineage_closed=true`

This is deterministic architecture evidence, not thesis-scale numerical
replication.

## Path C — original LFS asset integrity

After a complete Git LFS pull, optionally run:

```powershell
git lfs fsck
python -m thesis_repro verify-original
python -m thesis_repro rebuild-c3e --release original
```

The source archive alone is not an LFS-complete historical asset checkout.
