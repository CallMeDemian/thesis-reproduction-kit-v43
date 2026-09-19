"""Dynamic current selected-variable contract for fresh Oracle/RL runs.

The current scientific contract is the selected-variable master produced by the
preserved Stage00_04/Stage1 Oracle artifacts. Package/config copies are snapshots
that must match that artifact; they never define a fixed fresh-run R-code list.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def validate_dynamic_selected_variable_records(
    records: list[dict[str, Any]],
    *,
    min_total: int = 2,
    max_total: int = 11,
) -> dict[str, Any]:
    """Validate the actual Stage00_04 universe without legacy cardinalities."""
    if not isinstance(records, list) or not all(isinstance(x, dict) for x in records):
        raise TypeError("selected-variable artifact must be a list of records")
    ids = [str(x.get("variable_id", "")).strip() for x in records]
    categories = [str(x.get("category", "")).strip() for x in records]
    sources = [str(x.get("source", "")).strip() for x in records]
    if any(not x for x in ids):
        raise ValueError("selected-variable artifact contains a blank variable_id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"selected-variable artifact contains duplicate ids: {ids}")
    if any(not x for x in categories) or len(categories) != len(set(categories)):
        raise ValueError(f"selected-variable categories must be nonblank and unique: {categories}")
    if not (int(min_total) <= len(ids) <= int(max_total)):
        raise ValueError(
            f"selected-variable count outside dynamic contract: {len(ids)} not in {min_total}..{max_total}"
        )
    unknown_sources = sorted(set(sources) - {"financial", "nonfinancial"})
    if unknown_sources:
        raise ValueError(f"selected-variable artifact has unknown sources: {unknown_sources}")
    n_financial = sum(x == "financial" for x in sources)
    n_nonfinancial = sum(x == "nonfinancial" for x in sources)
    if n_financial < 1 or n_nonfinancial < 1:
        raise ValueError(
            "dynamic selected-variable contract requires both blocks: "
            f"financial={n_financial}, nonfinancial={n_nonfinancial}"
        )
    if "kospi_dummy" in ids:
        raise ValueError("kospi_dummy is diagnostic-only and cannot be selected")
    ineligible = [
        x.get("variable_id") for x in records
        if "selected_eligible" in x and not bool(x.get("selected_eligible"))
    ]
    if ineligible:
        raise ValueError(f"ineligible variables survived Stage00_04 selection: {ineligible}")
    return {
        "contract_version": "dynamic_optional_category_selected_variables_v1",
        "selected_count": len(ids),
        "financial_count": n_financial,
        "nonfinancial_count": n_nonfinancial,
        "selected_variables": ids,
        "selected_categories": categories,
    }

def read_selected_variable_master(path: Path) -> pd.DataFrame:
    """Read selected_variable_master.csv with strict schema presence.

    The project writes this file with UTF-8 BOM in several places.  A missing
    variable_id column is a hard contract failure because downstream archive
    readers use that column to distinguish the current R133 oracle from stale
    R136 source snapshots.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"selected_variable_master missing: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "variable_id" not in df.columns:
        raise KeyError(f"selected_variable_master has no variable_id column: {path}; columns={list(df.columns)}")
    return df


def selected_variable_ids_from_master(path: Path) -> list[str]:
    df = read_selected_variable_master(path)
    values = [str(x).strip() for x in df["variable_id"].dropna().tolist() if str(x).strip()]
    if len(values) != len(set(values)):
        dupes = sorted({x for x in values if values.count(x) > 1})
        raise ValueError(f"selected_variable_master has duplicate variable_id values: {path}; duplicates={dupes}")
    if not values:
        raise ValueError(f"selected_variable_master has no selected variables: {path}")
    return values


def resolve_stage1_selected_variable_master(project_root: Path) -> Path:
    """Resolve the authoritative fresh-run Stage1 selected-variable artifact."""
    root = Path(project_root).resolve()
    from credit_recourse.oracle.fresh_runtime import resolve_fresh_oracle_runtime

    s4 = resolve_fresh_oracle_runtime(root).inputs_root / "stage00_04_variable_selection"
    candidates = [s4 / "selected_variable_master.csv", s4 / "outputs" / "selected_variable_master.csv"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Fresh selected-variable contract requires the actual Stage00_04/Stage1 artifact; "
        f"checked={[str(p) for p in candidates]}"
    )


def verify_current_selected_variable_master(
    path: Path,
    *,
    expected_ids: list[str],
    expected_source_path: Path,
) -> dict[str, Any]:
    """Verify one snapshot against the actual Stage1-selected universe."""
    errors: list[str] = []
    try:
        ids = selected_variable_ids_from_master(path)
    except Exception as exc:
        return {
            "path": str(Path(path)),
            "status": "FAIL",
            "selected_variables": [],
            "expected_selected_variables": list(expected_ids),
            "expected_source_path": str(expected_source_path),
            "errors": [repr(exc)],
        }
    if ids != list(expected_ids):
        errors.append(
            "selected_variable_master variable_id order/content differs from actual Stage1 artifact: "
            f"expected={list(expected_ids)} found={ids} path={path} source={expected_source_path}"
        )
    return {
        "path": str(Path(path)),
        "status": "PASS" if not errors else "FAIL",
        "selected_variables": ids,
        "expected_selected_variables": list(expected_ids),
        "expected_source_path": str(expected_source_path),
        "dynamic_stage1_artifact_is_source_of_truth": True,
        "errors": errors,
    }


def verify_package_selected_variable_masters(
    package_root: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Verify the single live selected-variable master against Stage1."""
    package_root = Path(package_root).resolve()
    root = Path(project_root).resolve() if project_root is not None else package_root.parents[1]
    try:
        reference_path = resolve_stage1_selected_variable_master(root)
        expected = selected_variable_ids_from_master(reference_path)
    except Exception as exc:
        return {
            "status": "FAIL",
            "dynamic_stage1_artifact_is_source_of_truth": True,
            "checks": [],
            "errors": [repr(exc)],
        }
    # There is deliberately one runtime configuration owner.  Historical
    # package snapshots are archived and must not be treated as alternate
    # authorities during a live verification.
    from credit_recourse.oracle.fresh_runtime import resolve_fresh_oracle_runtime

    paths = [resolve_fresh_oracle_runtime(root).config_root / "selected_variable_master.csv"]
    checks = [
        verify_current_selected_variable_master(
            p,
            expected_ids=expected,
            expected_source_path=reference_path,
        )
        for p in paths
    ]
    errors = [err for check in checks for err in check["errors"]]
    return {
        "status": "PASS" if not errors else "FAIL",
        "authoritative_stage1_master": str(reference_path),
        "selected_variables": expected,
        "dynamic_stage1_artifact_is_source_of_truth": True,
        "checks": checks,
        "errors": errors,
    }
