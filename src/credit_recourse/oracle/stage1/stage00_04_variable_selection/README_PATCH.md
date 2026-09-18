# Stage 3 Patch Fixed

## What changed

- Uses `stage3_config.yaml` as the source of truth.
- Reads Stage 1C through stable alias `stage1c`, not `stage1c_v2`.
- Restores Stage 3 v3.2 domain guardrails:
  - liquidity sales/revenue/cost exclusion;
  - industry-risk allowlist limited to 9 lag1 self-excluded proxies;
  - diagnostic-only variables blocked from selection.
- Preserves upstream `selected_eligible` / `diagnostic_only` flags where present.
- Keeps R185 and `financial_data_completeness` selected for compatibility, but pins their v3 prior to 0.02 via score override and writes `stage3_subfix_override_log.csv`.
- Canonical outputs are v3:
  - `selected_variables_v3.json`
  - `direction_encoding_v3.json`
  - `weights_prior_v3.json`
  - `collinearity_matrix_v3.csv`
  - `domain_filter_log_v3.csv`
  - `stage3_subfix_override_log.csv`
- Optional latest/v2 aliases are created according to config for downstream compatibility.

## Usage

Place these files at the Stage 3 bundle root, ensuring the input folders exist:

```powershell
.\.venv\Scripts\python.exe stage3_pipeline.py
```

The `stage1c` folder must point to the current Stage 1C v3.2 output.


## Packaging fix applied by ChatGPT
- Added root `stage3_config.yaml` because the pipeline loads `Path("stage3_config.yaml")`.
- Added `stage3_pipeline_v3.py` as canonical execution file; `stage3_pipeline.py` remains as compatibility alias.
