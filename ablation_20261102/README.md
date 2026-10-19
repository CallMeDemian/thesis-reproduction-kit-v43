# ablation_20261102

Ablation tables, figures and the E1 reference-content control experiment, built on the frozen results of
`hb-fy24-final` (commit `0718488`). Design and rules: `ABLATION_PROTOCOL.md`; E1: `E1/E1_PROTOCOL.md`;
checks: `QA_REPORT.md`.

## Reproduce the tables and figures

Python 3.12 with the kit's pinned packages (`frozen_python_environment.json`: pandas 3.0.2, numpy 2.4.4,
scipy 1.17.1, scikit-learn 1.8.0, pyarrow 24.0.0, matplotlib 3.11.0, statsmodels 0.14.6, ...), `PYTHONUTF8=1`,
the kit installed with `pip install -e . --no-deps`. Git LFS files needed: `frozen/evidence/v1.3/...` and the
stage9 `HIGH_STRICT_ITT` folder (`git lfs pull -I "frozen/original_release/evaluation/llm_final_evaluation_20260913/stage9/HIGH_STRICT_ITT/*"`).

```
python ablation_20261102/scripts/run_all.py
```

Order: `00_inputs.py`, `10_primary_T1.py`, `20_s1_T2.py`, `30_ref1_T3.py`, `40_robust_T4.py`, `50_figures.py`, `90_qa.py`.
Bootstrap seeds follow the registry key rule, so every exploratory row is exactly reproducible.

## E1

`E1/E1_PROTOCOL.md` is the pre-registered contract. `E1/scripts/` holds the request builder, analysis, parse path,
scoring path and collector. `E1/requests/` holds the 3,450-request manifest and the eight shards (Git LFS).
`E1/runs/` holds batch ids, the collector log and raw responses. Nothing in the protocol, scripts, manifest or
shards changed after the first provider call.
