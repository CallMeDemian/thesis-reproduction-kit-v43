# ablation_20261102

Ablation tables, figures and the E1 reference-content control experiment, built on the frozen results of
`hb-fy24-final` (commit `0718488`). Design and rules: `ABLATION_PROTOCOL.md`; E1: `E1/E1_PROTOCOL.md`;
checks: `QA_REPORT.md`.

## Reproduce the tables and figures

Python 3.12 with the kit's pinned packages (`frozen_python_environment.json`: pandas 3.0.2, numpy 2.4.4,
scipy 1.17.1, scikit-learn 1.8.0, pyarrow 24.0.0, matplotlib 3.11.0, statsmodels 0.14.6, ...) plus python-docx 1.2.0
(`40_robust_T4.py` reads the thesis docx), `PYTHONUTF8=1`, the kit installed with `pip install -e . --no-deps`.
`frozen/evidence/v1.3/...` is stored as regular files. Git LFS files needed: the stage9 `HIGH_STRICT_ITT` folder for
the inputs check, T3 and QA Q10 (`git lfs pull -I "frozen/original_release/evaluation/llm_final_evaluation_20260913/stage9/HIGH_STRICT_ITT/*"`),
and `E1/requests/batches/`, `E1/runs/` for QA Q17-Q18. On an LFS pointer a script stops (a QA check fails) and prints
the `git lfs pull` line to run. QA Q01, Q12 and Q18 compare with older commits, so they need a full clone (not `--depth 1`).

```
python ablation_20261102/scripts/run_all.py
```

Order: `00_inputs.py`, `10_primary_T1.py`, `15_primary_stream_check.py`, `20_s1_T2.py`, `30_ref1_T3.py`,
`40_robust_T4.py`, `50_figures.py`, then `55_figure_F5.py` and `60_descriptive.py` when `results/T5_E1.csv` exists,
and `90_qa.py`. Bootstrap seeds follow the registry key rule, so every exploratory row is exactly reproducible.

T1 takes CI, p and Holm of the 96 primary rows from the frozen production streams. `15_primary_stream_check.py`
recomputes them from the firm-level observations through the kit's Stage 9 code (ITT and PP rows of each arm, one
PCG64 stream per arm, seed 20260912) and compares them with the frozen values (`results/_primary_stream_check.json`).

`source_sha256` in the result tables was computed on a Windows checkout (CRLF line endings). On an LF checkout
(Linux, macOS) that column alone comes out different; the files themselves are the same.

## E1

`E1/E1_PROTOCOL.md` is the pre-registered contract. `E1/scripts/` holds the request builder, analysis, parse path,
scoring path and collector. `E1/requests/` holds the 3,450-request manifest and the eight shards (Git LFS).
`E1/runs/` holds batch ids, the collector log and raw responses. Nothing in the protocol, scripts, manifest or
shards changed after the first provider call.
