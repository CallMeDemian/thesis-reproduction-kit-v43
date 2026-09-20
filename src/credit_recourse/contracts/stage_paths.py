from __future__ import annotations
import os
from pathlib import Path

V43_STAGE2_RUNTIME_PATH = Path("03_stage2")
V43_BP_RATE_SOURCE_PATH = Path("contracts/scientific/v43_rate_source")
V43_BP_RATE_PRODUCTION_PATH = Path("contracts/scientific/v43_rate_production")

# The original V4.3 producers use ``stage3``/``stage4`` names while the
# reproduction product exposes numbered run directories.  Keep the scientific
# producer names, but resolve them to the active run namespace at call time.
_RUN_STAGE_DIRS = {
    "stage2": "03_stage2",
    "stage3": "04_rl_encoder",
    "stage4": "05_rl_bc",
    "stage5": "06_rl_iql",
    "stage6": "08_stage6",
    "stage7": "09_llm",
    "stage8": "10_stage8",
    "stage9": "11_stage9",
}


def _active_run_root(root: Path) -> Path:
    configured = os.environ.get("THESIS_REPRO_RUN_ROOT") or os.environ.get("CREDIT_REPRO_RUN_PATH")
    return Path(configured) if configured else Path(root) / "runs" / "unbound"

def stage_dir(root: Path, stage: str) -> Path:
    return _active_run_root(Path(root)) / _RUN_STAGE_DIRS.get(stage, stage)

def final_root(root: Path) -> Path:
    return _active_run_root(Path(root))
