# Troubleshooting

Start with the repository diagnostics:

```powershell
python -m thesis_repro doctor
python -m thesis_repro data doctor
python -m thesis_repro inspect run <run_id>
python -m thesis_repro trace --run-id <run_id>
```

Common explicit states are:

- `INPUT_REQUIRED`: licensed or restored input is missing or invalid.
- `CREDENTIALS_REQUIRED`: a live provider needs credentials.
- `APPROVAL_REQUIRED`: an explicitly gated computation has not been approved.
- `RESOURCE_REQUIRED`: required hardware or another runtime resource is absent.
- `EXTERNAL_WAIT`: an asynchronous provider job is still pending; this is not
  a failed experiment. Resume the same run when the provider has completed.
- `FAILED`: an execution, verification, hash, or lineage invariant failed.

A changed scientific contract or source fingerprint invalidates incompatible
downstream stages. Inspect the run manifest and trace before attempting a
resume. Synthetic and smoke artifacts are not certifiable fresh replication.
