from __future__ import annotations

from pathlib import Path


def source_path(root: Path, run_id: str, profile: str) -> Path:
    root = Path(root)
    if profile == "MANUSCRIPT_FROZEN":
        return root / "frozen_replay/v1.3/frozen_inputs/stage8/canonical_itt_observations.parquet"
    if profile == "CLEAN_REEXECUTION":
        run_root = root / "runs" / run_id
        run_info = run_root / "RUN_INFO.json"
        if not run_info.is_file():
            raise FileNotFoundError(f"BLOCKED_CLEAN_RUN_METADATA_MISSING:{run_info}")
        try:
            metadata = __import__("json").loads(run_info.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"CLEAN_RUN_METADATA_INVALID:{type(exc).__name__}") from exc
        if metadata.get("run_id") != run_id or metadata.get("created_this_run") is not True:
            raise ValueError("STALE_RUN_REJECTED:created_this_run_or_run_id")
        if metadata.get("status") not in {"PASS", "READY", "COMPLETED"} or metadata.get("upstream_status") not in {None, "PASS"}:
            raise ValueError("STALE_RUN_REJECTED:upstream_status")
        stage8 = root / "runs" / run_id / "stage8"
        for name in ("canonical_itt_observations.parquet", "llm_stage8_multi_oracle_scores_all_populations.parquet"):
            candidate = stage8 / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"BLOCKED_CLEAN_RUN_INPUT_MISSING:{stage8}")
    raise ValueError(f"unknown analysis profile: {profile}")


def primary_filter(frame, *, reasoning_regime, model_key, budget, execution_policy="STRICT"):
    return frame.loc[
        (frame.reasoning_regime == reasoning_regime)
        & (frame.model_key == model_key)
        & (frame.budget == budget)
        & (frame.execution_policy == execution_policy)
        & (frame.action_space == "FREE8")
        & (frame.replicate == 1)
        & (frame.information_condition == "IC-b")
        & (frame.analysis_population.astype(str).str.lower() == "itt")
    ].copy()
