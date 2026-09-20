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
    if "row_id" not in fresh or "row_id" not in frozen:
        return {"status": "FAILED", "reason": "no shared C3-E firm identity"}

    actions = "action_id" if "action_id" in fresh.columns and "action_id" in frozen.columns else "action" if "action" in fresh.columns and "action" in frozen.columns else None
    if actions is None:
        return {"status": "FAILED", "reason": "no compatible C3-E action column"}

    action_names = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")

    def normalized(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        out = frame.copy()
        out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)
        out = out.rename(columns={actions: "action_id"})
        found: dict[str, str] = {}
        for action in action_names:
            for candidate in (f"probability__{action}", f"p_{action}", action):
                if candidate in out.columns:
                    found[action] = candidate
                    break
        if len(found) != len(action_names):
            raise ValueError(f"{label} C3-E probability columns are incomplete")
        for action, column in found.items():
            out[f"probability__{action}"] = pd.to_numeric(out[column], errors="raise")
        if out.duplicated("row_id").any():
            raise ValueError(f"{label} C3-E identity is duplicated")
        return out[["row_id", "action_id", *[f"probability__{action}" for action in action_names]]]

    try:
        left = normalized(fresh, "fresh")
        right = normalized(frozen, "frozen")
    except (KeyError, ValueError) as exc:
        return {"status": "FAILED", "reason": str(exc)}
    merged = left.merge(right, on="row_id", how="inner", suffixes=("_fresh", "_frozen"), validate="one_to_one")
    if len(merged) == 0 or len(merged) != len(left) or len(merged) != len(right):
        return {"status": "FAILED", "reason": "fresh and frozen C3-E identities do not close exactly", "fresh_rows": len(left), "frozen_rows": len(right), "common_firms": len(merged)}
    agreement = float(merged["action_id_fresh"].astype(str).eq(merged["action_id_frozen"].astype(str)).mean())
    probability_columns = [f"probability__{action}" for action in action_names]
    probability_mae = float((merged[[f"{column}_fresh" for column in probability_columns]].to_numpy(dtype=float) - merged[[f"{column}_frozen" for column in probability_columns]].to_numpy(dtype=float)).__abs__().mean())
    return {"status": "PASS", "fresh_rows": int(len(fresh)), "frozen_rows": int(len(frozen)), "common_firms": int(len(merged)), "action_agreement": agreement, "changed_firm_count": int((~merged["action_id_fresh"].astype(str).eq(merged["action_id_frozen"].astype(str))).sum()), "probability_mae": probability_mae, "fresh_sha256": _sha256(fresh_path), "frozen_sha256": _sha256(frozen_path), "comparison_only": True}


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
    synthetic = manifest.get("execution_class") == "SYNTHETIC_E2E_ACCEPTANCE"
    report = {
        "schema_version": "fresh_vs_frozen_v3",
        "run_id": run_id,
        "frozen_status": frozen["status"],
        "fresh_completion_state": manifest.get("completion_state"),
        "comparison_status": "PASS" if complete and frozen["status"] == "PASS" else ("SYNTHETIC_COMPARISON_NOT_APPLICABLE" if synthetic else "INPUT_REQUIRED"),
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
