# Data preparation

Raw archives are private. Restore them with `python -m thesis_repro data restore`, then run `python -m thesis_repro data doctor`.

The doctor checks every file in `contracts/scientific/input_inventory.json` and emits explicit `PRESENT`, `MISSING`, `SIZE_MATCH`, `SIZE_MISMATCH`, `HASH_MATCH`, `HASH_MISMATCH`, `OPTIONAL_MISSING`, `SCHEMA_MATCH`, and `SCHEMA_MISMATCH` fields. The gate is `INPUT_CONTRACT_PASS`; otherwise a full fresh run returns `INPUT_REQUIRED` and does not read frozen artifacts as compute inputs.

What is compared (also written to the report as `comparison_basis`):

- Presence and size: against `relative_path` and `size_bytes` in the inventory.
- Hash: against `data/raw/input_receipt.json`, which `data restore` writes from the same archives. This shows that the files did not change after restore; it does not prove that they are the original licensed files, because the inventory records no per-file hash.
- Schema: a header check of `.xlsx` files against the columns the raw loaders require — the firm-code and year columns of the `raw_all` statement files (as found by `build_stage0_foundation_from_raw.find_col`) and the exact columns of the `rating_sample` files required by `rating_contract_repair`. Files without a loader column contract (ratio sheets, nonfinancial tables) stay `NOT_CHECKED`. The schema check was tested on synthetic fixtures; it has not been run on the licensed raw inputs.
- KONEX: the inventory description mentions optional KONEX files, but no inventory entry carries `optional: true`, so the eight files under `raw_nonfinancial/konex_optional/` are treated as required.

Generated reports are private runtime state:

- `data/raw/input_receipt.json`
- `data/raw/input_contract_report.json`
