from __future__ import annotations
import os
from pathlib import Path

V43_STAGE2_RUNTIME_PATH = Path("03_simulator")
V43_BP_RATE_SOURCE_PATH = Path("contracts/scientific/v43_rate_source")
V43_BP_RATE_PRODUCTION_PATH = Path("contracts/scientific/v43_rate_production")

def stage_dir(root: Path, stage: str) -> Path:
    run_root = os.environ.get("THESIS_REPRO_RUN_ROOT") or os.environ.get("CREDIT_REPRO_RUN_PATH")
    return (Path(run_root) if run_root else Path(root) / "runs" / "unbound") / stage

def final_root(root: Path) -> Path:
    run_root = os.environ.get("THESIS_REPRO_RUN_ROOT") or os.environ.get("CREDIT_REPRO_RUN_PATH")
    return Path(run_root) if run_root else Path(root) / "runs" / "unbound"
