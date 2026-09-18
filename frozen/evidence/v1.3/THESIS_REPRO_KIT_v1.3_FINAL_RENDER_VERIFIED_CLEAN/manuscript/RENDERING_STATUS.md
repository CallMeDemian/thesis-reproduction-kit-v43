# DOCX rendering status

**PASS — independently rendered and visually QA'd.**

- Render engine: LibreOffice 25.2.3.2 (headless)
- Final PDF pages: 142
- Page geometry: 140 portrait A4 pages + 2 landscape A4 pages for the V1.3 supplemental appendix
- Printed pagination continues through pages 124–125 in the added V1.3 appendix.
- No text bounding box was within 15 pt of a page edge in the final PDF.
- No blank rendered pages remained.

During independent visual QA, two release defects in the prior DOCX were corrected:
1. The V1.3 representative-results table was unreadable in portrait layout because long identifiers and numeric columns wrapped into vertical fragments. The final appendix is landscape with fixed column widths and repeating headers.
2. A stale V1.3 narrative sentence incorrectly treated GPT Baseline/Strict B1 C4R−C4 `+0.1692620467` as a non-canonical historical aggregate and substituted `+0.0105196562`. The corrected text reflects the package contract: `+0.1692620467` is the Baseline/Strict primary result; `+0.0105196562` belongs to a High/Repaired supplemental cell.

No canonical CSV, statistical contract, RNG result, or scientific numeric table value was changed by the rendering repair.
