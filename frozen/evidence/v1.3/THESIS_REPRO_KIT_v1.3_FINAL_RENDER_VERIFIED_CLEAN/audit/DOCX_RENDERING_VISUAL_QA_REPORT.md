# DOCX Rendering & Visual QA Report

Status: **PASS**

## Rendered artifact
- DOCX SHA-256: `0c66d0e344b69031a8577cfd6a86eff89700d1d0476a642da208568101bd2724`
- PDF SHA-256: `7c09ff33f4ca8665289d3cfe9a6c32583767c84de442106845031064fa215d72`
- Renderer: LibreOffice 25.2.3.2
- Pages: 142
- 140 portrait A4 + 2 landscape A4

## QA performed
- Full-document page-render sweep.
- Targeted full-resolution inspection of the V1.3 appendix and dense table pages.
- Automated page-bound scan: no text bounding box within 15 pt of any page edge.
- No blank final pages.
- Korean/English glyph rendering observed without black-square substitution in reviewed renders.
- Page numbering verified to continue from printed page 123 to 124–125.

## Defects found and corrected
1. V1.3 appendix 10-column table was unreadable in portrait orientation.
   - Corrected with a dedicated landscape final section, fixed widths, compact labels, repeated header, non-splitting rows.
2. One stale V1.3 narrative statement conflated a High/Repaired supplemental value (`+0.0105196562`) with the Baseline/Strict primary C4R−C4 result.
   - Corrected to retain the frozen primary value `+0.1692620467`, consistent with `V13_BASELINE_STRICT_PRIMARY_FULLSTREAM.csv` and `THESIS_TABLE_CONTRACT.csv`.
3. Redundant blank page before the V1.3 appendix was removed.
4. Printed page numbering in the new landscape section was set to continue at 124.

## Scientific impact
Formatting repair only, except for correction of the stale explanatory sentence above. Canonical result CSVs and statistical contracts were not modified.
