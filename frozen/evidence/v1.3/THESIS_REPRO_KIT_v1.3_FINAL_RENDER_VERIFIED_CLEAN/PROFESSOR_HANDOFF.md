# Professor handoff

This Tier C package starts from frozen Stage8 firm-level observations and validates the primary production stream, supplemental statistics, parent-gate descriptive sensitivity, and thesis row-level contract without network or provider APIs.

```powershell
python -m pip install -r requirements.txt
python reproduction/reproduce_all_v13.py
```

Expected markers are `PRIMARY_REPRODUCTION_PASS`, `SUPPLEMENTAL_REPRODUCTION_PASS`, `PARENT_GATE_PASS`, `THESIS_CONTRACT_PASS`, and `PACKAGE_INTEGRITY_PASS`, with exit code 0.

The canonical result inputs are under `frozen_inputs/stage8/` and `canonical_v13/`; the row-level trace contract is `contracts/THESIS_TABLE_CONTRACT.csv`. Raw proprietary data and live API generation are outside Tier C. Historical supplemental RNG provenance is not claimed; the final V1.3 PP/RQ4/RQ5 uncertainty statistics are reproducible under the declared V1.3 SHA-256 contract.


The bundled final manuscript has also been independently rendered and visually QA'd; the rendered PDF is included under `manuscript/`.
