# V1.3 Final Refreeze Report

Primary production stream: PASS and unchanged.
Supplemental deterministic V1.3 stream: PASS.
Supplemental result rows: 722 (historical 276; PP 336; RQ4 18; RQ5 12; budget 8; reasoning 24; execution 48; parent-gate 96).
Historical supplemental comparison rows: 276.
Historical significance: 13 unchanged where historical Holm was available; 3 lost; 260 undetermined because historical Holm was absent.
Historical exact SHA-256 substream recovery: unavailable and explicitly superseded.
PP remains strict usable intersection; RQ4/RQ5 remain ITT.

## Parent gate sensitivity completion
The parent-gate sensitivity is now computed from the frozen Stage8 firm-level observations. For each regime/model/B1-or-BINF cell, the C4 strict usability flag is used as the pre-specified gate; when C4 is unusable, downstream C4R/C6-E/C6-EX values are replaced with the frozen A0 delta (0), while C5 is unchanged. The resulting 96 rows and 4,600 firm gate flags are stored in `canonical_v13/parent_gate_sensitivity_v13.csv` and `canonical_v13/parent_gate_firm_flags_v13.csv`. This is an offline sensitivity reconstruction, not a new API execution.

