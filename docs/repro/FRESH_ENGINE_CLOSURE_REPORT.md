# V4.3 Fresh Scientific Engine Closure

Status: IMPLEMENTED for the fresh Oracle entrypoint and explicit downstream unexecuted states. Numerical FullClean has not been claimed.

## Acceptance evidence

- `FullClean`, profile `full`, no-input run: `INPUT_REQUIRED` at `VerifyInputs`.
- Oracle production path is IMPLEMENTED: it dispatches Stage0 and Stage1 production functions into `runs/<run_id>/02_oracle/work`; numerical EXECUTED/VERIFIED evidence requires raw inputs.
- RL production stages are `NOT_IMPLEMENTED` until their fresh producers and independent verifier are wired. The heavy gate is checked before any training and does not claim execution.
- LLM stages use explicit `APPROVAL_REQUIRED`, `CREDENTIALS_REQUIRED`, `NOT_EXECUTED`, or `EXECUTED_UNVERIFIED` states; no live provider submission is claimed without a provider receipt.
- Local regression suite: `30 passed`; CI status must be checked for the pushed commit before certification.
- Runtime closure audit: `runs/_runtime_audit`; reachable modules `71`, missing internal modules `0`, legacy references recorded `159` for review. Legacy references are retained in historical/compatibility source and are not used by the fresh adapter dispatch.

## Canonical fresh topology

Every fresh artifact is under `runs/<run_id>/`:

`01_inputs -> 02_oracle -> 03_simulator -> 04_rl_dataset -> 05_rl_encoder -> 06_rl_bc -> 07_rl_iql -> 08_c3e -> 09_llm -> 10_stage8 -> 11_stage9 -> 12_results -> 13_thesis_outputs -> 14_comparison -> 15_release`.

The fresh action contract is `contracts/scientific/v43_action_contract.json` and has exactly eight managerial dimensions and nine action IDs: `A0, DL, RF, CX, WC1, WC2, OE, MX1, MX2`. Revenue growth remains an exogenous BusinessPlan/scenario variable. DL is principal reduction; RF is principal-neutral maturity transformation.

## Implemented seams

- Runtime paths and parent lineage: `src/thesis_repro/runtime_paths.py`, `src/thesis_repro/stages/base.py`.
- Oracle production adapter and verification entry points: `src/thesis_repro/stages/oracle.py` and `src/thesis_repro/stages/adapters.py`.
- Downstream declarations remain explicit typed non-PASS states; they do not emit scientific PASS artifacts.
- Provider-native batch rendering delegates to the production provider builder; network submission delegates to the production live batch runner only after the live gate.
- Import closure bridges for the canonical V4.3 action, account, Alpha, financial-input, oracle-backend, R085, and rate-helper seams.
- Static forensic audit: `scripts/audit_runtime_closure.py`.
- Twenty fresh-engine closure tests plus the existing eight product tests.

## Not executed

- Restricted raw financial/non-financial input restoration: required before numerical FullClean.
- 28 actor RL retraining: not wired and not executed.
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

The frozen/original release remains a provenance source only; it is not a runtime parent of fresh artifacts. Restored scientific source provenance is recorded in the implementation commit and the source manifests.
