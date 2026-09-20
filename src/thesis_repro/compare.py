"""Descriptive fresh-vs-frozen comparison, never a fresh compute parent."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .frozen import verify_frozen
from .paths import ROOT, load_json, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_state(run_dir: Path, relative: str) -> dict[str, Any]:
    path = run_dir / relative
    return {"state": "present" if path.is_file() else "missing", "path": relative, "sha256": _sha256(path) if path.is_file() else None}


def _c3e_comparison(root: Path, run_dir: Path) -> dict[str, Any]:
    fresh_path = run_dir / "07_c3e/c3e_actions.parquet"
    frozen_path = root / "frozen/original_release/rl/C3E_E2_7SEED_BALANCED_DFEBAFA6/C3E_firm_actions.parquet"
    if not fresh_path.is_file() or not frozen_path.is_file():
        return {"status": "UNAVAILABLE", "reason": "fresh or frozen C3-E action table missing"}
    fresh = pd.read_parquet(fresh_path)
    frozen = pd.read_parquet(frozen_path)
    key = "row_id" if "row_id" in fresh and "row_id" in frozen else None
    if key is None:
        return {"status": "UNAVAILABLE", "reason": "no shared C3-E firm key"}
    left = fresh.set_index(key)
    right = frozen.set_index(key)
    common = left.index.intersection(right.index)
    action_column = "action_id" if "action_id" in left and "action_id" in right else "action"
    if action_column not in left or action_column not in right:
        return {"status": "UNAVAILABLE", "reason": "no shared C3-E action column"}
    agreement = float((left.loc[common, action_column].astype(str).to_numpy() == right.loc[common, action_column].astype(str).to_numpy()).mean()) if len(common) else None
    probability_columns = sorted(set(left.columns).intersection(right.columns).intersection({"p_A0", "p_DL", "p_RF", "p_CX", "p_WC1", "p_WC2", "p_OE", "p_MX1", "p_MX2"}))
    probability_mae = float((left.loc[common, probability_columns].astype(float) - right.loc[common, probability_columns].astype(float)).abs().to_numpy().mean()) if probability_columns and len(common) else None
    return {"status": "PASS", "fresh_rows": int(len(fresh)), "frozen_rows": int(len(frozen)), "common_firms": int(len(common)), "action_agreement": agreement, "changed_firm_count": int(len(common) - round((agreement or 0.0) * len(common))), "probability_mae": probability_mae, "fresh_sha256": _sha256(fresh_path), "frozen_sha256": _sha256(frozen_path), "comparison_only": True}


def _stage9_comparison(root: Path, run_dir: Path) -> dict[str, Any]:
    fresh_path = run_dir / "11_stage9/llm_stage9_primary_contrast_summary.csv"
    frozen_path = root / "frozen/original_release/evaluation/llm_final_evaluation_20260913/stage9/BASELINE_STRICT_ITT/llm_stage9_primary_contrast_summary.csv"
    if not fresh_path.is_file() or not frozen_path.is_file():
        return {"status": "UNAVAILABLE", "reason": "fresh or frozen Stage9 summary missing"}
    fresh = pd.read_csv(fresh_path)
    frozen = pd.read_csv(frozen_path)
    keys = [key for key in ("contrast", "backend", "model_key", "mode", "information_condition", "budget", "analysis_population") if key in fresh and key in frozen]
    estimate = "mean_effect" if "mean_effect" in fresh and "mean_effect" in frozen else None
    if not keys or estimate is None:
        return {"status": "UNAVAILABLE", "reason": "no compatible Stage9 estimand identity"}
    merged = fresh[keys + [estimate]].merge(frozen[keys + [estimate]], on=keys, suffixes=("_fresh", "_frozen"), how="inner")
    if merged.empty:
        return {"status": "PASS", "matched_estimands": 0, "comparison_only": True}
    diff = pd.to_numeric(merged[f"{estimate}_fresh"], errors="coerce") - pd.to_numeric(merged[f"{estimate}_frozen"], errors="coerce")
    return {"status": "PASS", "matched_estimands": int(len(merged)), "estimate_mae": float(diff.abs().mean()), "estimate_rank_correlation": float(pd.to_numeric(merged[f"{estimate}_fresh"], errors="coerce").corr(pd.to_numeric(merged[f"{estimate}_frozen"], errors="coerce"), method="spearman")) if len(merged) > 1 else None, "comparison_only": True}


def compare_run(run_id: str, *, root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    run_dir = root / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"unknown run: {run_id}")
    manifest = load_json(run_dir / "run_manifest.json")
    frozen = verify_frozen(root)
    fresh = {
        "oracle": _artifact_state(run_dir, "02_oracle/work/ledgers/stage1_oracle_backends_full_development.json"),
        "c3e": _artifact_state(run_dir, "07_c3e/release.json"),
        "stage8": _artifact_state(run_dir, "10_stage8/metadata.json"),
        "stage9": _artifact_state(run_dir, "11_stage9/metadata.json"),
    }
    complete = all(item["state"] == "present" for item in fresh.values())
    report = {
        "schema_version": "fresh_vs_frozen_v3",
        "run_id": run_id,
        "frozen_status": frozen["status"],
        "fresh_completion_state": manifest.get("completion_state"),
        "comparison_status": "PASS" if complete and frozen["status"] == "PASS" else "INPUT_REQUIRED",
        "fresh_artifacts": fresh,
        "c3e": _c3e_comparison(root, run_dir),
        "stage9": _stage9_comparison(root, run_dir),
        "frozen_reference_counts": {"stage8": 96600, "primary": 96, "supplemental": 722, "parent": 96, "registry": 914, "raw_llm_generations": 48300},
        "comparison_semantics": "replication_comparison_only; exact equality is not implied",
    }
    comparison_dir = run_dir / "14_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_json(comparison_dir / "fresh_vs_frozen.json", report)
    (comparison_dir / "FRESH_VS_FROZEN.md").write_text(
        f"# Fresh vs frozen: `{run_id}`\n\nComparison status: `{report['comparison_status']}`\n\nThis is a descriptive comparison after fresh computation; it is not an identity proof.\n",
        encoding="utf-8",
    )
    return report
