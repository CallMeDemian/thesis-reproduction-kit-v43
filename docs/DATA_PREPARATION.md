# Data preparation

Raw archives are private. Restore them with `python -m thesis_repro data restore`, then run `python -m thesis_repro data doctor`.

The doctor checks every required file in `contracts/scientific/input_inventory.json` and emits explicit `PRESENT`, `MISSING`, `SIZE_MATCH`, `SIZE_MISMATCH`, `HASH_MATCH`, `HASH_MISMATCH`, `OPTIONAL_MISSING`, `SCHEMA_MATCH`, and `SCHEMA_MISMATCH`-compatible contract fields as validation advances. The gate is `INPUT_CONTRACT_PASS`; otherwise a full fresh run returns `INPUT_REQUIRED` and does not read frozen artifacts as compute inputs.

Generated reports are private runtime state:

- `data/raw/input_receipt.json`
- `data/raw/input_contract_report.json`
