THESIS REPRO KIT v1.3. Primary stream unchanged. Supplemental V1.3 deterministic contract covers historical-row, PP, RQ4, RQ5, budget, reasoning-regime, and execution-sensitivity rows (722 total). Parent-gate sensitivity is computed offline from frozen Stage8 `strict_usable` and firm-level score rows and is stored in `canonical_v13/parent_gate_sensitivity_v13.csv` (96 rows; 4,600 firm flags). Run: `python reproduction/reproduce_all_v13.py`. Historical supplemental SHA256 recovery is not claimed and is explicitly disclosed as superseded by the V1.3 declared supplemental contract.


## Current release closure

One command validates the primary 96-row production stream, 722 supplemental inferential rows, the 96-row descriptive parent-gate sensitivity, and the row-level thesis table contract. Parent-gate rows are descriptive (N/means) because the manuscript does not use them as inferential claims.

The worked primary anchor is Gemini Baseline Strict ITT, B1, C5-C4, Alpha: estimate 0.11527067698397228, raw p 0.0011998800119988, Holm p 0.008399160083991601. It traces through `contracts/THESIS_TABLE_CONTRACT.csv`, `frozen_inputs/stage8/canonical_itt_observations.parquet`, and the primary stream verifier. For the primary stream, CI/raw-p/Holm are retained from the frozen production fullstream artifact; the row-level Stage8 source independently cross-checks cardinality and means. The package does not claim to regenerate the lost production bootstrap draw order from the parquet alone.


## Render-verified final manuscript

The final manuscript was independently rendered to PDF with LibreOffice 25.2.3.2 and visually QA'd after correcting the final V1.3 appendix layout and one stale explanatory sentence. The statistical CSVs/contracts were not changed.

- Final DOCX SHA-256: `0c66d0e344b69031a8577cfd6a86eff89700d1d0476a642da208568101bd2724`
- Final PDF SHA-256: `7c09ff33f4ca8665289d3cfe9a6c32583767c84de442106845031064fa215d72`
- Rendered pages: 142 (140 portrait + 2 landscape)
