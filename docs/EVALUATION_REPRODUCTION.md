# Evaluation reproduction

Historical Stage8 evidence contains 96,600 rows. The published/frozen thesis
result registry contains 914 rows: 96 primary, 722 supplemental, and 96
parent-gate results.

Fresh Stage8 and Stage9 runs produce same-run evaluation and contrast
artifacts under their run namespace. They do not overwrite the frozen 914-row
registry and must not silently claim byte- or number-level identity with it.
`RESULT_COMPARISON.md` describes comparisons between the two authorities.
