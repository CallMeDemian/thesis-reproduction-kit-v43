# Original asset packs

The original repository is preserved as the scientific source, while this
product repository keeps compact contracts and immutable evidence in Git. The
large Oracle, RL, LLM, and Stage8/Stage9 trees are inventoried before copying
so that no artifact is silently dropped or replaced by a synthetic fixture.

The committed asset verification is run by:

```powershell
python scripts/verify_original_assets.py
```

The committed manifests record the large source trees and their current state.
The Oracle, RL, LLM, and Stage8/Stage9 packs below have now been copied to the
planned `frozen/original_release/` paths and verified file-by-file. The source
repository remains read-only throughout that operation.

## Pack policy

| Pack | Source size class | Planned destination | Current status |
|---|---:|---|---|
| Oracle stage0/stage1 | ~0.3 GB | `frozen/original_release/oracle/` | Copied, hash-verified, and uploaded through Git LFS |
| RL C3-E final | ~0.48 GB | `frozen/original_release/rl/` | Copied, hash-verified, and uploaded through Git LFS |
| LLM final_plan3 | ~3.0 GB | `frozen/original_release/llm/` | Copied, hash-verified, and request-ID reconciled through Git LFS |
| Stage8/Stage9 evaluation | ~0.5 GB | `frozen/original_release/evaluation/` | Copied and hash-verified through Git LFS |

The frozen distribution and canonical V1.3 evidence remain the current
certified replay lane. Original asset packs must pass hash reconciliation before
they are used to upgrade a fresh-replication capability.
