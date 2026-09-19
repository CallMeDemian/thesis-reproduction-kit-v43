# V4.3 Fresh Scientific Engine Closure

Status: implementation closed at the adapter-and-gate boundary. No raw input restoration, 28-actor retraining, or live provider call was performed.

## Acceptance evidence

- `FullClean`, profile `full`, no-input run: `INPUT_REQUIRED` at `VerifyInputs`.
- Full Oracle branch proof: `PASS`; smoke artifact helper was monkeypatched to fail and was not called.
- Full RL branch proof: stops at `RLIQL` with `HEAVY_EXECUTION_APPROVAL_REQUIRED` unless `THESIS_REPRO_ENABLE_HEAVY_RL=I_APPROVE_28_ACTOR_RETRAIN` is set.
- Full LLM branch proof: after the heavy gate, renders and validates 48,300 fresh logical requests, then stops with `LIVE_LLM_APPROVAL_REQUIRED` unless the live execution option and `THESIS_REPRO_ENABLE_LIVE_LLM=I_APPROVE_FRESH_REPLICATION` are explicitly supplied.
- Test suite: `28 passed`.
- Runtime closure audit: `runs/_runtime_audit`; reachable modules `71`, missing internal modules `0`, legacy references recorded `159` for review. Legacy references are retained in historical/compatibility source and are not used by the fresh adapter dispatch.

## Canonical fresh topology

Every fresh artifact is under `runs/<run_id>/`:

`01_inputs -> 02_oracle -> 03_simulator -> 04_rl_dataset -> 05_rl_encoder -> 06_rl_bc -> 07_rl_iql -> 08_c3e -> 09_llm -> 10_stage8 -> 11_stage9 -> 12_results -> 13_thesis_outputs -> 14_comparison -> 15_release`.

The fresh action contract is `contracts/scientific/v43_action_contract.json` and has exactly eight managerial dimensions and nine action IDs: `A0, DL, RF, CX, WC1, WC2, OE, MX1, MX2`. Revenue growth remains an exogenous BusinessPlan/scenario variable. DL is principal reduction; RF is principal-neutral maturity transformation.

## Implemented seams

- Runtime paths and parent lineage: `src/thesis_repro/runtime_paths.py`, `src/thesis_repro/stages/base.py`.
- Oracle, simulator, RL, C3E, LLM, materialization, Stage8, and Stage9 adapter entry points: `src/thesis_repro/stages/`.
- Provider-native batch rendering delegates to the production provider builder; network submission delegates to the production live batch runner only after the live gate.
- Import closure bridges for the canonical V4.3 action, account, Alpha, financial-input, oracle-backend, R085, and rate-helper seams.
- Static forensic audit: `scripts/audit_runtime_closure.py`.
- Twenty fresh-engine closure tests plus the existing eight product tests.

## Not executed

- Restricted raw financial/non-financial input restoration: required before numerical FullClean.
- 28 actor RL retraining: intentionally gated.
- 48,300 live provider calls: intentionally gated and not executed.
- Fresh Stage8 numerical materialization and Stage9 scientific comparison: require those upstream inputs and provider receipts.

## Reproduction commands

```powershell
$env:PYTHONPATH='src'
python -m thesis_repro fresh --mode FullClean --run-id <run_id> --profile full
python -m thesis_repro fresh --mode OracleRLLLMClean --run-id <run_id> --profile full
python scripts/audit_runtime_closure.py
python -m pytest -q
```

The frozen/original release remains a provenance and verification source only; it is not a runtime parent of fresh artifacts.
