"""Finalize the Stage1 Alpha backend with the canonical monotone-bin contract.

The statistical pipeline deliberately fits only populated development bins.
This module is the production boundary that completes the otherwise undefined
empty bins, writes the versioned contract, and re-scores every exported
firm-year row with that same contract. It belongs to the Alpha producer; it is
not a post-hoc verification artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from credit_recourse.contracts.v43_alpha_contract import (
    EXPECTED_ALPHA_CONTENT_HASH,
    EXPECTED_ALPHA_CONTRACT_SHA256,
)
from credit_recourse.oracle.fresh_runtime import oracle_execution_profile
from credit_recourse.oracle.backends.alpha.modules.monotone_bins import (
    contract_hash,
    freeze_monotone_params,
    validate_monotone_params,
)
from credit_recourse.oracle.backends.alpha.modules.oracle_alpha_scorer import build_alpha_scorer


PARAMS_NAME = "oracle_alpha_params.json"
OUTPUT_PARQUET_NAME = "oracle_firm_year_output_alpha.parquet"
OUTPUT_CSV_NAME = "oracle_firm_year_output_alpha.csv"
RANK_SHOCK_NAME = "rank_shock_base_panel_alpha.parquet"
FILE_MANIFEST_NAME = "stage4_alpha_file_manifest.csv"
KEY = "거래소코드"
RNG_SEED = 42


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_canonical_contract(path: Path, value: dict[str, Any]) -> None:
    """Write the frozen contract with platform-independent canonical CRLF bytes."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_bytes(payload.replace("\n", "\r\n").encode("utf-8"))


def finalize_contract_file(path: Path) -> tuple[dict[str, Any], str]:
    """Convert a freshly fitted legacy Alpha parameter file into V4.3."""
    path = Path(path)
    params = json.loads(path.read_text(encoding="utf-8"))
    if "oracle_alpha_contract_version" in params or "empty_bin_repair" in params:
        validate_monotone_params(params)
        corrected = params
    else:
        corrected = freeze_monotone_params(params)
    if oracle_execution_profile() == "synthetic":
        # The synthetic acceptance fixture deliberately fits its own
        # deterministic panel.  It must retain the real Alpha producer and
        # monotone-bin semantics, but it cannot truthfully claim byte identity
        # with the licensed-data V4.3 release contract.
        corrected["empty_bin_repair"]["fit_source"] = (
            "SYNTHETIC_E2E_ACCEPTANCE deterministic fixture development rows"
        )
        corrected["oracle_alpha_contract_hash"] = contract_hash(corrected)
        validate_monotone_params(corrected)
    _write_canonical_contract(path, corrected)
    digest = file_sha256(path)
    if oracle_execution_profile() == "production" and digest != EXPECTED_ALPHA_CONTRACT_SHA256:
        raise ValueError(
            "Fresh Alpha producer did not reproduce the canonical V4.3 contract: "
            f"expected={EXPECTED_ALPHA_CONTRACT_SHA256} actual={digest}"
        )
    if oracle_execution_profile() == "production" and corrected.get("oracle_alpha_contract_hash") != EXPECTED_ALPHA_CONTENT_HASH:
        raise ValueError("Fresh Alpha producer changed the V4.3 semantic contract hash")
    return corrected, digest


def _score_frame(frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    selected = list(params["selected_variables"])
    missing = [name for name in selected if name not in frame.columns]
    if missing:
        raise KeyError(f"Alpha output is missing selected variables: {missing}")
    scorer = build_alpha_scorer(params)
    scored = [scorer(values) for values in frame[selected].to_dict("records")]
    out = frame.copy()
    for variable in selected:
        out[f"{variable}_iso_score"] = [row["item_scores"][variable] for row in scored]
        out[f"{variable}_imputed"] = [row["imputed"][variable] for row in scored]

    records = params.get("selected_variable_records") or []
    financial = [str(row["variable_id"]) for row in records if row.get("source") == "financial"]
    nonfinancial = [str(row["variable_id"]) for row in records if row.get("source") == "nonfinancial"]
    if not financial:
        financial = [name for name in selected if name.startswith("R")]
    if not nonfinancial:
        nonfinancial = [name for name in selected if name not in financial]
    weights = {str(key): float(value) for key, value in params["optimized_weights"].items()}
    out["fin_raw_alpha"] = sum(out[f"{name}_iso_score"] * weights[name] for name in financial)
    out["nonfin_raw_alpha"] = sum(out[f"{name}_iso_score"] * weights[name] for name in nonfinancial)
    out["fin_score_alpha"] = [row["fin_score"] for row in scored]
    out["nonfin_score_alpha"] = [row["nonfin_score"] for row in scored]
    out["R_score_alpha"] = [row["R_score"] for row in scored]
    out["R_grade_alpha_raw"] = [row["R_grade_raw"] for row in scored]
    out["R_grade_alpha"] = [row["R_grade"] for row in scored]
    out["R_grade_alpha_num"] = [row["R_grade_num"] for row in scored]
    out["R_PD_alpha"] = [row["R_PD"] for row in scored]
    return out


def _calc_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame[frame["grade_base"].notna() & frame["R_grade_alpha"].notna()]
    if valid.empty:
        return {}
    observed = pd.to_numeric(valid.get("rating_num_10", valid["rating_num"]), errors="coerce")
    predicted = pd.to_numeric(valid["R_grade_alpha_num"], errors="coerce")
    observed_grade = valid.get("grade_base_10", valid["grade_base"])
    exact = float((observed_grade == valid["R_grade_alpha"]).mean())
    within1 = float((np.abs(observed - predicted) <= 1).mean())
    mae = float(np.abs(observed - predicted).mean())
    rho = float(spearmanr(valid["R_score_alpha"], observed).statistic)
    target = (observed <= 4).astype(int)
    try:
        ar = float((2 * roc_auc_score(target, valid["R_score_alpha"]) - 1) * 100)
    except Exception:
        ar = float("nan")
    try:
        qk = float(cohen_kappa_score(observed.astype(int), predicted.astype(int), weights="quadratic"))
    except Exception:
        qk = float("nan")
    return {
        "n": int(len(valid)),
        "exact": round(exact, 4),
        "within1": round(within1, 4),
        "mae": round(mae, 3),
        "rho": round(rho, 4),
        "AR": round(ar, 1) if np.isfinite(ar) else None,
        "QK": round(qk, 3) if np.isfinite(qk) else None,
    }


def _rewrite_diagnostics(output_dir: Path, frame: pd.DataFrame, params: dict[str, Any]) -> list[dict[str, Any]]:
    grade_order = list(params["grade_order"])
    grade_num = {grade: index + 1 for index, grade in enumerate(grade_order)}
    metrics: list[dict[str, Any]] = []
    for split in ("dev", "oot"):
        row = _calc_metrics(frame[frame["split_stage4"] == split])
        row.update({"specification": "alpha", "split": split})
        metrics.append(row)
        valid = frame[
            frame["split_stage4"].eq(split)
            & frame["grade_base"].notna()
            & frame["R_grade_alpha"].notna()
        ]
        if not valid.empty:
            labels = sorted(
                set(valid["grade_base"]) | set(valid["R_grade_alpha"]),
                key=lambda grade: grade_num.get(str(grade), 99),
            )
            confusion = pd.DataFrame(0, index=labels, columns=labels)
            for observed, predicted in zip(valid["grade_base"], valid["R_grade_alpha"]):
                confusion.loc[observed, predicted] += 1
            confusion.to_csv(output_dir / f"confusion_matrix_alpha_{split}.csv", encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(output_dir / "preliminary_dev_oot_metrics_alpha.csv", index=False)

    distribution = []
    for split in ("dev", "oot"):
        subset = frame[frame["split_stage4"] == split]
        for grade in grade_order:
            distribution.append({
                "split": split,
                "grade": grade,
                "observed": int(subset["grade_base"].eq(grade).sum()),
                "predicted_alpha": int(subset["R_grade_alpha"].eq(grade).sum()),
            })
    pd.DataFrame(distribution).to_csv(output_dir / "grade_distribution_alpha.csv", index=False)
    return metrics


def _rewrite_boundary_jump_test(output_dir: Path, frame: pd.DataFrame, params: dict[str, Any]) -> None:
    scorer = build_alpha_scorer(params)
    selected = list(params["selected_variables"])
    dev = frame[frame["split_stage4"].eq("dev")]
    sample = dev.sample(n=min(50, len(dev)), random_state=RNG_SEED).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        original_values = {name: row[name] for name in selected}
        original = scorer(original_values)
        for variable in selected:
            value = row[variable]
            if pd.isna(value) or value == 0:
                continue
            for perturbation in (-0.05, -0.01, -0.005, 0.005, 0.01, 0.05):
                changed_values = dict(original_values)
                changed_values[variable] = value * (1 + perturbation)
                changed = scorer(changed_values)
                rows.append({
                    "firm_id": row.get(KEY, ""),
                    "year": int(row["year"]),
                    "variable_id": variable,
                    "perturbation": perturbation,
                    "original_value": round(float(value), 6),
                    "perturbed_value": round(float(changed_values[variable]), 6),
                    "item_score_before": round(original["item_scores"][variable], 2),
                    "item_score_after": round(changed["item_scores"][variable], 2),
                    "item_score_delta": round(changed["item_scores"][variable] - original["item_scores"][variable], 2),
                    "R_score_before": round(original["R_score"], 4),
                    "R_score_after": round(changed["R_score"], 4),
                    "R_score_delta": round(changed["R_score"] - original["R_score"], 4),
                    "R_grade_before": original["R_grade"],
                    "R_grade_after": changed["R_grade"],
                    "boundary_crossed": original["R_grade"] != changed["R_grade"],
                    "excessive_jump_flag": (
                        abs(changed["item_scores"][variable] - original["item_scores"][variable]) > 20
                        and abs(perturbation) <= 0.01
                    ),
                })
    pd.DataFrame(rows).to_csv(output_dir / "boundary_jump_test_alpha.csv", index=False)


def _rewrite_rank_shock(output_dir: Path, scored: pd.DataFrame) -> None:
    path = output_dir / RANK_SHOCK_NAME
    if not path.exists():
        return
    rank = pd.read_parquet(path)
    join_keys = [name for name in (KEY, "year") if name in rank.columns and name in scored.columns]
    if len(join_keys) != 2:
        raise KeyError("Rank-shock panel cannot be aligned to the Alpha output")
    values = scored[join_keys + ["R_score_alpha", "R_grade_alpha"]]
    if values.duplicated(join_keys).any():
        raise ValueError("Alpha output has duplicate firm-year keys")
    base = rank.drop(columns=["R_score_alpha", "R_grade_alpha"], errors="ignore")
    rewritten = base.merge(values, on=join_keys, how="left", validate="one_to_one")
    if rewritten[["R_score_alpha", "R_grade_alpha"]].isna().any().any():
        raise ValueError("Rank-shock panel has rows missing from the Alpha output")
    ordered = [name for name in rank.columns if name in rewritten.columns]
    rewritten[ordered].to_parquet(path)


def finalize_alpha_backend(output_dir: Path) -> dict[str, Any]:
    """Finalize contract and all score-bearing Stage1 Alpha outputs in place."""
    output_dir = Path(output_dir).resolve()
    params_path = output_dir / PARAMS_NAME
    output_path = output_dir / OUTPUT_PARQUET_NAME
    if not params_path.is_file() or not output_path.is_file():
        raise FileNotFoundError(f"Incomplete Alpha backend output: {output_dir}")
    params, digest = finalize_contract_file(params_path)
    frame = _score_frame(pd.read_parquet(output_path), params)
    frame.to_parquet(output_path)
    frame.to_csv(output_dir / OUTPUT_CSV_NAME, index=False, encoding="utf-8-sig")
    _rewrite_rank_shock(output_dir, frame)
    metrics = _rewrite_diagnostics(output_dir, frame, params)
    _rewrite_boundary_jump_test(output_dir, frame, params)
    (output_dir / "bin_score_table_isotonic_alpha.json").write_text(
        json.dumps(params["bin_score_table_isotonic"], ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    files = [path for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != FILE_MANIFEST_NAME]
    pd.DataFrame([{"filename": path.name, "size": path.stat().st_size} for path in files]).to_csv(
        output_dir / FILE_MANIFEST_NAME, index=False
    )
    return {
        "status": "PASS",
        "producer_step": "alpha_monotone_empty_bin_contract_finalization",
        "contract_sha256": digest,
        "contract_content_hash": params["oracle_alpha_contract_hash"],
        "rows_rescored": int(len(frame)),
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(finalize_alpha_backend(args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
