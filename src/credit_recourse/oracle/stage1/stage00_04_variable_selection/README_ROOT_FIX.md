# Stage 00-04 variable_selection root fix

Patched items:

1. Restored valid collinearity loop indentation.
2. Preserved `variable_id` for financial-derived nonfinancial proxy candidates instead of blindly overwriting it with `variable_name`.
3. Excluded nonfinancial candidates that are not present in `nonfinancial_metadata_panel.parquet`.
4. Built the collinearity `dev_all` matrix from the full available financial/nonfinancial candidate universe, not only from initial selected variables.
5. Restricted replacement candidates to variables actually available in `dev_all`.
6. Added guards for temporary and final correlation matrices to prevent stale candidate-column KeyErrors.
7. Included root-level `stage3_config.yaml` and `stage_config.yaml` because the legacy pipeline looks for `stage3_config.yaml` in the working directory.

Usage:

```powershell
cd .\src\credit_recourse\oracle\pipelines\stage00_sample_variable_foundation\04_variable_selection
.\.venv\Scripts\python.exe .\pipeline.py
```

