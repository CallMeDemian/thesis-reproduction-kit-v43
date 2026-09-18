"""Canonical Stage00_02 growth-audit input contract for Stage00_04.

The legacy candidate pool is intentionally retained by Stage00_02 for v2
reproduction.  Corrected Oracle development must instead consume the expanded
ratio panel and candidate pool together with both growth-audit tables.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


GROWTH_CANDIDATE_CONTRACT_VERSION = "stage00_02_canonical_candidate_pool_to_stage00_04_v2"
GROWTH_CATEGORY = "성장성"
KEY_COLUMNS = ["거래소코드", "year"]


def as_bool_series(series: pd.Series) -> pd.Series:
    """Coerce serialized bool values without treating non-empty strings as true."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes", "y", "t"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_nonempty(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing canonical {label}: {path}")


def _normalized_reason(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _normalized_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame[KEY_COLUMNS].copy()
    keys["거래소코드"] = (
        keys["거래소코드"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    keys["year"] = pd.to_numeric(keys["year"], errors="coerce").astype("Int64")
    return keys


def load_canonical_financial_inputs(stage2_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the corrected financial ratio panel and candidate eligibility.

    Every growth candidate must be covered by exactly one Stage00_02 audit row.
    ``main_eligible_v3_2`` is the authoritative selection flag.  Denominator
    instability is applied again as a fail-safe so no downstream transformation
    can silently resurrect an excluded ratio.
    """

    stage2_dir = Path(stage2_dir).resolve()
    growth_dir = stage2_dir / "growth_audit"
    paths = {
        "legacy_ratio_panel": stage2_dir / "engineered_financial_ratios.parquet",
        "expanded_ratio_panel": stage2_dir / "engineered_financial_ratios_canonical.parquet",
        "expanded_candidate_pool": stage2_dir / "candidate_ratio_pool_canonical.csv",
        "existing_growth_audit": growth_dir / "growth_audit_existing.csv",
        "new_growth_audit": growth_dir / "growth_audit_new.csv",
    }
    for label, path in paths.items():
        _require_nonempty(path, label)

    base_keys = pd.read_parquet(paths["legacy_ratio_panel"], columns=KEY_COLUMNS)
    ratios = pd.read_parquet(paths["expanded_ratio_panel"])
    candidates = pd.read_csv(paths["expanded_candidate_pool"])
    existing_audit = pd.read_csv(paths["existing_growth_audit"])
    new_audit = pd.read_csv(paths["new_growth_audit"])
    growth_audit = pd.concat([existing_audit, new_audit], ignore_index=True, sort=False)

    for label, frame, required in [
        ("expanded ratio panel", ratios, KEY_COLUMNS),
        ("expanded candidate pool", candidates, ["category", "ratio_id", "main_eligible_v3_2", "exclusion_reason_v3_2", "selected_eligible", "expected_good_direction", "quality_gate_scope"]),
        ("growth audit", growth_audit, ["ratio_id", "main_eligible", "denom_instability", "exclusion_reason"]),
    ]:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(f"{label} missing required columns: {missing}")

    if base_keys.duplicated(KEY_COLUMNS).any() or ratios.duplicated(KEY_COLUMNS).any():
        raise ValueError("Stage00_02 base/expanded ratio panels must be unique by firm-year")
    base_universe = pd.MultiIndex.from_frame(_normalized_key_frame(base_keys))
    expanded_universe = pd.MultiIndex.from_frame(_normalized_key_frame(ratios))
    if len(base_universe) != len(expanded_universe) or not base_universe.sort_values().equals(expanded_universe.sort_values()):
        raise ValueError("Expanded growth ratio panel changed the canonical Stage00_02 firm-year universe")

    candidates["ratio_id"] = candidates["ratio_id"].astype(str)
    growth_audit["ratio_id"] = growth_audit["ratio_id"].astype(str)
    if candidates["ratio_id"].duplicated().any():
        duplicate_ids = sorted(candidates.loc[candidates["ratio_id"].duplicated(False), "ratio_id"].unique())
        raise ValueError(f"Expanded candidate pool contains duplicate ratio_id values: {duplicate_ids}")
    if growth_audit["ratio_id"].duplicated().any():
        duplicate_ids = sorted(growth_audit.loc[growth_audit["ratio_id"].duplicated(False), "ratio_id"].unique())
        raise ValueError(f"Growth audit contains duplicate ratio_id values: {duplicate_ids}")

    growth_mask = candidates["category"].astype(str).eq(GROWTH_CATEGORY)
    growth_ids = set(candidates.loc[growth_mask, "ratio_id"])
    audit_ids = set(growth_audit["ratio_id"])
    if growth_ids != audit_ids:
        raise ValueError(
            "Growth candidate/audit ratio_id universe mismatch: "
            f"missing_audit={sorted(growth_ids - audit_ids)}, extra_audit={sorted(audit_ids - growth_ids)}"
        )

    missing_ratio_columns = sorted(set(candidates["ratio_id"]) - set(map(str, ratios.columns)))
    if missing_ratio_columns:
        raise KeyError(f"Expanded ratio panel missing candidate columns: {missing_ratio_columns}")

    audit_index = growth_audit.set_index("ratio_id", drop=False)
    expanded_eligible = as_bool_series(candidates.loc[growth_mask, "main_eligible_v3_2"])
    audited_eligible = as_bool_series(candidates.loc[growth_mask, "ratio_id"].map(audit_index["main_eligible"]))
    expanded_reason = _normalized_reason(candidates.loc[growth_mask, "exclusion_reason_v3_2"])
    audited_reason = _normalized_reason(candidates.loc[growth_mask, "ratio_id"].map(audit_index["exclusion_reason"]))
    if not expanded_eligible.reset_index(drop=True).equals(audited_eligible.reset_index(drop=True)):
        raise ValueError("Expanded pool main_eligible_v3_2 disagrees with Stage00_02 growth audit")
    if not expanded_reason.reset_index(drop=True).equals(audited_reason.reset_index(drop=True)):
        raise ValueError("Expanded pool exclusion_reason_v3_2 disagrees with Stage00_02 growth audit")

    candidates["selected_eligible"] = as_bool_series(
        candidates.get("selected_eligible", pd.Series(False, index=candidates.index))
    )
    candidates["growth_main_eligible_v3_2"] = False
    candidates["growth_denom_instability"] = False
    candidates["growth_exclusion_reason_v3_2"] = ""
    candidates["selected_eligibility_source"] = "stage00_02_quality_pool"

    growth_ratio_ids = candidates.loc[growth_mask, "ratio_id"]
    denom_instability = as_bool_series(growth_ratio_ids.map(audit_index["denom_instability"]))
    candidates.loc[growth_mask, "growth_main_eligible_v3_2"] = expanded_eligible.to_numpy()
    candidates.loc[growth_mask, "growth_denom_instability"] = denom_instability.to_numpy()
    candidates.loc[growth_mask, "growth_exclusion_reason_v3_2"] = audited_reason.to_numpy()
    candidates.loc[growth_mask, "selected_eligible"] = expanded_eligible.to_numpy()
    candidates.loc[growth_mask, "selected_eligibility_source"] = "stage00_02_growth_audit_main_eligible_v3_2"

    denom_mask = growth_mask & as_bool_series(candidates["growth_denom_instability"])
    candidates.loc[denom_mask, "selected_eligible"] = False
    if candidates.loc[denom_mask, "growth_exclusion_reason_v3_2"].map(lambda value: "denom_instability" not in str(value)).any():
        raise ValueError("Denominator-instability audit rows must carry exclusion_reason=denom_instability")
    if (denom_mask & as_bool_series(candidates["selected_eligible"])).any():
        raise AssertionError("Denominator-unstable growth candidate survived the Stage00_04 fail-safe")
    expected_scope = "full_available_panel_including_2024_retained_by_research_contract"
    observed_scopes = set(candidates["quality_gate_scope"].dropna().astype(str))
    if observed_scopes != {expected_scope}:
        raise ValueError(f"Unexpected Stage00_02 quality-gate scope: {sorted(observed_scopes)}")

    candidates["candidate_pool_contract_version"] = GROWTH_CANDIDATE_CONTRACT_VERSION
    contract = {
        "contract_version": GROWTH_CANDIDATE_CONTRACT_VERSION,
        "status": "PASS",
        "selection_rule": "growth selected_eligible := main_eligible_v3_2 AND NOT denom_instability",
        "quality_gate_scope": expected_scope,
        "firm_year_rows": int(len(ratios)),
        "candidate_count": int(len(candidates)),
        "main_candidate_count": int((~candidates["category"].astype(str).eq("기타/복합")).sum()),
        "growth_candidate_count": int(growth_mask.sum()),
        "growth_selected_eligible_count": int(as_bool_series(candidates.loc[growth_mask, "selected_eligible"]).sum()),
        "denominator_instability_blocked_ratio_ids": sorted(candidates.loc[denom_mask, "ratio_id"].tolist()),
        "input_artifacts": {
            label: {"path": str(path), "sha256": _sha256(path), "size_bytes": int(path.stat().st_size)}
            for label, path in paths.items()
        },
    }
    return ratios, candidates, contract
