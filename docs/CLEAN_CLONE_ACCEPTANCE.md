# Clean-clone acceptance

The repository is accepted as a fresh clone when the following sequence completes without access to the source repository:

```powershell
python -m pip install -e ".[data,dev]"
python -m thesis_repro frozen-replay
python -m thesis_repro verify-original
python -m thesis_repro rebuild-c3e --release original
python -m thesis_repro fresh --mode OracleRLLLMClean --run-id clean-clone-smoke --profile smoke --dry-render
python -m thesis_repro trace --run-id clean-clone-smoke
python -m thesis_repro compare --run-id clean-clone-smoke
```

Acceptance requires frozen verification PASS, 1,035 original artifacts verified, 28/28 historical RL parent chains closed, a 48,300-request fresh namespace dry render, and an explicit `INPUT_REQUIRED` result for full fresh execution when private raw inputs are not restored. No frozen artifact may appear as a silent fresh compute parent.
