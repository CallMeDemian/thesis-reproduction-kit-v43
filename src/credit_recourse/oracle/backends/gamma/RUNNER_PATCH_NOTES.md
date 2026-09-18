# Runner patch notes

This bundle runner was patched to support the current `thesis_repo` layout.

## Fixed

- Removed legacy sibling-bundle lookup such as `stage1b_*`, `stage2_v3_*`, `stage4_alpha_bundle`.
- Reads required inputs from current project-level `archive/DEPLOYED_RELEASE/stage00_*` and `stage01_alpha_vanilla` directories.
- Writes outputs to project-level `archive/DEPLOYED_RELEASE/stage01_gamma_ml` instead of bundle-local `outputs`.
- Uses the project `.venv` when available.
- Installs `requirements.txt` and explicitly checks `import yaml` before running the pipeline.

## Usage

```powershell
cd .
.\src\credit_recourse\oracle\pipelines\stage01_oracle_construction\gamma\scripts\run_stage4_gamma.ps1 -ForceRefreshInputs
```

If the script cannot infer the repo root:

```powershell
.\src\credit_recourse\oracle\pipelines\stage01_oracle_construction\gamma\scripts\run_stage4_gamma.ps1 `
  -ProjectRoot . `
  -ForceRefreshInputs
```

