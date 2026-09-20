"""Concrete runtime-asset resolution for fresh and historical execution.

Fresh producers must resolve through the active run namespace.  The fallback
paths are intentionally reference-only and are used by the published replay
lane, never by a fresh compute parent.
"""
from __future__ import annotations

import os
from pathlib import Path


def _root(root: Path) -> Path:
    return Path(root).resolve()


def _fresh(root: Path) -> Path | None:
    value = os.environ.get("THESIS_REPRO_RUN_ROOT")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else _root(root) / path


def active_action_contract(root: Path) -> Path:
    fresh = _fresh(root)
    if fresh is not None:
        candidates = (fresh / "03_stage2/input_source/candidate_action_contract_v4_3.json", fresh / "03_stage2/candidate_action_contract_v4_3.json")
        for path in candidates:
            if path.is_file():
                return path
    root = _root(root)
    candidates = (root / "contracts/scientific/v43_action_contract.json", root / "frozen/original_release/rl/stage2/candidate_action_contract_v4_3.json")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("active V4.3 action contract is unavailable")


def active_oracle_registry(root: Path) -> Path:
    fresh = _fresh(root)
    if fresh is not None:
        for path in (fresh / "02_oracle/work/contracts/oracle_backend_registry.yaml", fresh / "02_oracle/work/contracts/oracle_components/oracle_backend_registry.yaml"):
            if path.is_file():
                return path
    root = _root(root)
    path = root / "contracts/scientific/final_freeze/oracle_backend_registry.yaml"
    if path.is_file():
        return path
    raise FileNotFoundError("active Oracle registry is unavailable")


def active_stage2_run_config(root: Path) -> Path:
    fresh = _fresh(root)
    if fresh is not None:
        path = fresh / "03_stage2/01_contract/resolved_run_config.json"
        if path.is_file():
            return path
    path = _root(root) / "frozen/original_release/rl/stage2/v4_3_runtime/01_contract/resolved_run_config.json"
    if path.is_file():
        return path
    raise FileNotFoundError("active Stage2 run config is unavailable")


def active_stage2_eval_panel(root: Path) -> Path:
    fresh = _fresh(root)
    if fresh is not None:
        for relative in ("03_stage2/stage2/evaluation_financial_grid.parquet", "03_stage2/stage2/evaluation_grid.parquet"):
            path = fresh / relative
            if path.is_file():
                return path
    path = _root(root) / "frozen/original_release/rl/stage2/phase_eval_candidate.parquet"
    if path.is_file():
        return path
    raise FileNotFoundError("active Stage2 evaluation panel is unavailable")


def active_stage2_eval_ids(root: Path) -> Path:
    fresh = _fresh(root)
    if fresh is not None:
        for relative in ("03_stage2/input_source/input_splits/canonical_evaluation_row_ids.parquet", "03_stage2/input_splits/canonical_evaluation_row_ids.parquet"):
            path = fresh / relative
            if path.is_file():
                return path
    path = _root(root) / "frozen/original_release/rl/stage2/input_splits/canonical_evaluation_row_ids.parquet"
    if path.is_file():
        return path
    raise FileNotFoundError("active Stage2 evaluation IDs are unavailable")


def active_stage2_history(root: Path) -> Path:
    fresh = _fresh(root)
    if fresh is not None:
        path = fresh / "03_stage2/input_source/input_splits/canonical_business_plan_history.parquet"
        if path.is_file():
            return path
    path = _root(root) / "frozen/original_release/rl/stage2/input_splits/canonical_business_plan_history.parquet"
    if path.is_file():
        return path
    raise FileNotFoundError("active Stage2 BusinessPlan history is unavailable")
