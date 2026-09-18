from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.oracle.contracts.rating_scale import (
    AGENCY_NAME,
    AGENCY_PRIORITY,
    ALLOWED_MAIN_AGENCY_CODES,
    ICR_ALLOWED_SECURITY_CODE,
    add_rating_scale_columns,
)

KEY_KR = "거래소코드"
YEAR_KR = "회계년도"
RATING_KR = "신용등급"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    cols = list(cols)
    lowered = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        hit = lowered.get(str(cand).strip().lower())
        if hit is not None:
            return hit
    for cand in candidates:
        c_low = str(cand).strip().lower()
        for col in cols:
            if c_low and c_low in str(col).strip().lower():
                return col
    return None


def _require_exact_col(cols: list[str], expected: str, *, semantic_name: str, source_file: Path) -> str:
    expected_folded = str(expected).strip().casefold()
    matches = [
        col for col in cols
        if str(col).strip().casefold() == expected_folded
    ]
    duplicate_like = [
        col for col in cols
        if re.fullmatch(re.escape(expected_folded) + r"\.\d+", str(col).strip().casefold())
    ]
    if duplicate_like:
        matches.extend(duplicate_like)
    if len(matches) != 1:
        raise ValueError(
            f"{source_file.name}: expected exactly one raw column {expected!r} for "
            f"{semantic_name}, found {matches}. Substring fallback is forbidden."
        )
    return matches[0]


def _raw_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).strftime("%Y/%m/%d")
    return str(value).strip()


def _parse_rating_date_value(value: Any) -> tuple[pd.Timestamp, str, str, pd.Timestamp, pd.Timestamp]:
    """Parse a rating event date without inventing an unknown day.

    ``rating_date`` is populated only for day-precision observations.  Month-
    and year-precision observations retain a null exact date plus explicit
    lower/upper period bounds for conservative as-of selection.
    """
    raw = _raw_text(value)
    if not raw:
        return pd.NaT, raw, "missing", pd.NaT, pd.NaT
    if isinstance(value, (pd.Timestamp, datetime)):
        exact = pd.Timestamp(value).normalize()
        return exact, raw, "day", exact, exact
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            exact = pd.to_datetime(float(value), unit="D", origin="1899-12-30").normalize()
            if 1900 <= int(exact.year) <= 2100:
                return exact, raw, "day", exact, exact
        except Exception:
            pass
    match = re.fullmatch(
        r"\s*((?:19|20)\d{2})(?:[./-](\d{1,2}))?(?:[./-](\d{1,2}))?(?:\s+00:00:00)?\s*",
        raw,
    )
    if match:
        year = int(match.group(1))
        month_text = match.group(2)
        day_text = match.group(3)
        if month_text is None or int(month_text) == 0:
            return (
                pd.NaT, raw, "year", pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
            )
        month = int(month_text)
        if not 1 <= month <= 12:
            return pd.NaT, raw, "invalid", pd.NaT, pd.NaT
        if day_text is None or int(day_text) == 0:
            last_day = calendar.monthrange(year, month)[1]
            return (
                pd.NaT, raw, "month", pd.Timestamp(year, month, 1),
                pd.Timestamp(year, month, last_day),
            )
        day = int(day_text)
        try:
            exact = pd.Timestamp(year, month, day)
        except ValueError:
            return pd.NaT, raw, "invalid", pd.NaT, pd.NaT
        return exact, raw, "day", exact, exact
    try:
        exact = pd.Timestamp(pd.to_datetime(raw, errors="raise")).normalize()
    except Exception:
        return pd.NaT, raw, "invalid", pd.NaT, pd.NaT
    return exact, raw, "day", exact, exact


def _parse_snapshot_period(value: Any) -> tuple[Any, pd.Timestamp, str]:
    raw = _raw_text(value)
    year = _norm_year(value)
    if pd.isna(year):
        return pd.NA, pd.NaT, raw
    year = int(year)
    match = re.search(r"(?:19|20)\d{2}[./-](\d{1,2})", raw)
    month = int(match.group(1)) if match and 1 <= int(match.group(1)) <= 12 else 12
    cutoff = pd.Timestamp(year, month, calendar.monthrange(year, month)[1])
    return year, cutoff, raw


def _join_unique(values: pd.Series) -> str:
    unique = sorted({str(x).strip() for x in values if pd.notna(x) and str(x).strip()})
    return " | ".join(unique)


def _first_nonempty(values: pd.Series) -> Any:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return value
    return pd.NA


def _norm_code(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"[^0-9]", "", s)
    return digits.zfill(6) if digits else s


def _norm_year(x: Any) -> Any:
    if pd.isna(x):
        return pd.NA
    try:
        y = int(float(x))
        if 1900 <= y <= 2100:
            return y
    except Exception:
        pass
    m = re.search(r"(19|20)\d{2}", str(x))
    return int(m.group(0)) if m else pd.NA


def _read_excel_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="calamine")
    except Exception:
        return pd.read_excel(path)


def _raw_rating_files(raw_dir: Path) -> list[Path]:
    """Find raw rating workbooks recursively.

    Final data may keep KOSPI/KOSDAQ and KONEX workbooks either directly under
    ``data/raw/rating_sample`` or in subfolders such as ``kospi_kosdaq`` and
    ``konex_optional``.  Use recursive discovery so Stage0 rebuild does not
    silently miss KONEX or market-specific folders.
    """
    patterns = ["*신용평가에 관한 사항*.xlsx", "*신용평가*.xlsx", "*rating*.xlsx"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(raw_dir.glob(pat))
        files.extend(raw_dir.rglob(pat))
    out = sorted({f.resolve(): f for f in files if f.is_file() and not f.name.startswith("~$")}.values())
    return out


def _build_rating_event_and_state_panels(valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if valid.empty:
        raise ValueError("No eligible dated long-term ICR rating rows remain after raw filtering")

    event_keys = [
        "firm_id", "agency_code", "security_type", "rating_date_precision",
        "rating_date_lower_bound", "rating_date_upper_bound", "grade_base_notch",
    ]
    work = valid.sort_values(
        ["firm_id", "rating_date_upper_bound", "rating_date_lower_bound", "agency_pri", "raw_snapshot_year", "_source_file"],
        ascending=[True, True, True, True, True, True],
    ).copy()
    events = (
        work.groupby(event_keys, as_index=False, dropna=False)
        .agg(
            rating_date=("rating_date", "first"),
            rating_date_raw=("rating_date_raw", _first_nonempty),
            rating_date_raw_variants=("rating_date_raw", _join_unique),
            agency_name=("agency_name", _first_nonempty),
            agency_metadata_raw=("agency_metadata_raw", _first_nonempty),
            agency_metadata_variants=("agency_metadata_raw", _join_unique),
            grade_raw=("grade_base", _first_nonempty),
            grade_raw_variants=("grade_base", _join_unique),
            grade_base_raw=("grade_base_raw", _first_nonempty),
            grade_10=("grade_base_10", _first_nonempty),
            rating_num_10=("rating_num_10", "first"),
            grade_7=("grade_base_7", _first_nonempty),
            rating_num_7=("rating_num_7", "first"),
            rating_num_notch=("rating_num_notch", "first"),
            source_file=("_source_file", _join_unique),
            raw_snapshot_year=("raw_snapshot_year", "min"),
            raw_snapshot_year_last=("raw_snapshot_year", "max"),
            raw_snapshot_occurrence_count=("raw_snapshot_year", "size"),
            raw_snapshot_years=("raw_snapshot_year", lambda s: "|".join(str(int(x)) for x in sorted(set(s.dropna().astype(int))))),
            firm_name=("firm_name", _first_nonempty),
            market=("market", _first_nonempty),
            security_name=("security_name", _first_nonempty),
            agency_pri=("agency_pri", "first"),
        )
        .rename(columns={
            "agency_code": "agency_code",
            "security_type": "security_type",
            "grade_base_notch": "grade_notch",
        })
    )
    events["rating_event_id"] = events.apply(
        lambda row: hashlib.sha256(
            "|".join([
                str(row["firm_id"]),
                str(int(row["agency_code"])),
                str(row["rating_date_precision"]),
                str(pd.Timestamp(row["rating_date_lower_bound"]).isoformat()),
                str(pd.Timestamp(row["rating_date_upper_bound"]).isoformat()),
                str(row["grade_notch"]),
                str(int(row["security_type"])),
            ]).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    event_columns = [
        "rating_event_id", "firm_id", "rating_date", "rating_date_raw",
        "rating_date_raw_variants", "rating_date_precision",
        "rating_date_lower_bound", "rating_date_upper_bound", "agency_code",
        "agency_name", "agency_metadata_raw", "agency_metadata_variants",
        "grade_raw", "grade_raw_variants", "grade_notch", "grade_base_raw",
        "grade_10", "rating_num_10", "grade_7", "rating_num_7",
        "rating_num_notch", "security_type", "security_name", "source_file",
        "raw_snapshot_year", "raw_snapshot_year_last",
        "raw_snapshot_occurrence_count", "raw_snapshot_years", "firm_name",
        "market", "agency_pri",
    ]
    events = events[event_columns].sort_values(
        ["firm_id", "rating_date_upper_bound", "rating_date_lower_bound", "agency_pri", "rating_event_id"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    if events["rating_event_id"].duplicated().any():
        raise ValueError("rating event deduplication produced duplicate rating_event_id values")

    targets = (
        work[["firm_id", "raw_snapshot_year", "raw_snapshot_cutoff"]]
        .dropna(subset=["firm_id", "raw_snapshot_year", "raw_snapshot_cutoff"])
        .groupby(["firm_id", "raw_snapshot_year"], as_index=False)
        .agg(rating_state_cutoff=("raw_snapshot_cutoff", "max"))
        .rename(columns={"raw_snapshot_year": "year"})
    )
    state_rows: list[dict[str, Any]] = []
    no_prior_event = 0
    events_by_firm = {firm: group.copy() for firm, group in events.groupby("firm_id", sort=False)}
    for target in targets.itertuples(index=False):
        firm_events = events_by_firm.get(str(target.firm_id))
        if firm_events is None or firm_events.empty:
            no_prior_event += 1
            continue
        cutoff = pd.Timestamp(target.rating_state_cutoff)
        eligible = firm_events[
            pd.to_datetime(firm_events["rating_date_upper_bound"], errors="coerce") <= cutoff
        ].copy()
        if eligible.empty:
            no_prior_event += 1
            continue
        chosen = eligible.sort_values(
            ["rating_date_upper_bound", "rating_date_lower_bound", "agency_pri", "rating_event_id"],
            ascending=[False, False, True, True],
        ).iloc[0]
        lower = pd.Timestamp(chosen["rating_date_lower_bound"])
        upper = pd.Timestamp(chosen["rating_date_upper_bound"])
        exact = pd.to_datetime(chosen["rating_date"], errors="coerce")
        observed_current_year = int(lower.year) == int(target.year) and int(upper.year) == int(target.year)
        state_rows.append({
            "거래소코드": str(target.firm_id),
            "year": int(target.year),
            "raw_snapshot_year": int(target.year),
            "rating_state_cutoff": cutoff,
            "rating_event_id": chosen["rating_event_id"],
            "source_rating_date": exact,
            "source_rating_date_raw": chosen["rating_date_raw"],
            "source_rating_date_precision": chosen["rating_date_precision"],
            "source_rating_period_start": lower,
            "source_rating_period_end": upper,
            "rating_age_days": int((cutoff - exact).days) if pd.notna(exact) else pd.NA,
            "rating_age_days_min": int((cutoff - upper).days),
            "rating_age_days_max": int((cutoff - lower).days),
            "rating_observed_in_current_year": bool(observed_current_year),
            "carried_forward": bool(not observed_current_year),
            "rating_observation_status": (
                "observed_event_in_current_firm_year"
                if observed_current_year else "carried_forward_asof"
            ),
            "representative_rating_contract": "latest_eligible_event_then_agency_priority_tiebreak",
            "rating_year_assignment_source": "event_timeline_asof_firm_year_cutoff",
            "회사명": chosen["firm_name"],
            "시장": chosen["market"],
            "평가일": exact,
            "평가사구분": int(chosen["agency_code"]),
            "평가사명": chosen["agency_name"],
            "평가사명 및 등급": chosen["agency_metadata_raw"],
            "증권명": chosen["security_name"],
            "증권구분": int(chosen["security_type"]),
            "grade_base": chosen["grade_raw"],
            "grade_base_raw": chosen["grade_base_raw"],
            "grade_base_notch": chosen["grade_notch"],
            "rating_num_notch": chosen["rating_num_notch"],
            "grade_base_10": chosen["grade_10"],
            "rating_num_10": chosen["rating_num_10"],
            "grade_base_7": chosen["grade_7"],
            "rating_num_7": chosen["rating_num_7"],
            "agency_pri": int(chosen["agency_pri"]),
            "_source_file": chosen["source_file"],
            "source_raw_snapshot_year_first": int(chosen["raw_snapshot_year"]),
            "source_raw_snapshot_year_last": int(chosen["raw_snapshot_year_last"]),
            "source_raw_snapshot_occurrence_count": int(chosen["raw_snapshot_occurrence_count"]),
        })
    states = pd.DataFrame(state_rows)
    if states.empty:
        raise ValueError("Rating event timeline produced no firm-year as-of states")
    states = states.sort_values(["거래소코드", "year"]).reset_index(drop=True)
    if states.duplicated(["거래소코드", "year"]).any():
        raise ValueError("Rating state builder produced duplicate firm-year rows")
    future = pd.to_datetime(states["source_rating_period_end"], errors="coerce") > pd.to_datetime(
        states["rating_state_cutoff"], errors="coerce"
    )
    if bool(future.any()):
        raise ValueError(f"Future rating events leaked into earlier firm-year states: {int(future.sum())}")

    diagnostics = {
        "rating_event_rows": int(len(events)),
        "rating_event_duplicate_snapshot_rows_removed": int(len(valid) - len(events)),
        "firm_year_targets": int(len(targets)),
        "firm_year_states": int(len(states)),
        "firm_year_without_prior_event": int(no_prior_event),
        "carried_forward_states": int(states["carried_forward"].sum()),
        "rating_observed_in_current_year_states": int(states["rating_observed_in_current_year"].sum()),
        "future_event_state_rows": 0,
        "event_date_precision_distribution": events["rating_date_precision"].value_counts(dropna=False).to_dict(),
        "representative_rating_contract": "latest rating_date_upper_bound DESC, rating_date_lower_bound DESC, agency priority ASC only as tie-break",
        "firm_year_cutoff_contract": "raw fiscal snapshot period end; event upper bound must be <= cutoff",
    }
    return states, events, diagnostics


def build_rating_sample(
    raw_rating_dir: Path, *, return_events: bool = False
) -> tuple[pd.DataFrame, dict[str, Any]] | tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    files = _raw_rating_files(raw_rating_dir)
    if not files:
        raise FileNotFoundError(f"No raw rating Excel files found under {raw_rating_dir}")

    frames = []
    file_rows = []
    source_contract = {
        "firm_id_source": KEY_KR,
        "raw_snapshot_year_source": YEAR_KR,
        "rating_value_source": RATING_KR,
        "rating_date_source": "평가일",
        "agency_code_source": "평가사구분",
        "agency_metadata_source": "평가사명 및 등급",
        "security_type_source": "증권구분",
    }
    for fp in files:
        raw = _read_excel_any(fp)
        if raw.empty:
            file_rows.append({"file": fp.name, "rows_raw": 0, "rows_used": 0, "status": "empty"})
            continue
        cols = list(raw.columns)
        found = {
            "code": _require_exact_col(cols, KEY_KR, semantic_name="firm_id", source_file=fp),
            "year": _require_exact_col(cols, YEAR_KR, semantic_name="raw_snapshot_year", source_file=fp),
            "rating": _require_exact_col(cols, RATING_KR, semantic_name="rating_value", source_file=fp),
            "security_code": _require_exact_col(cols, "증권구분", semantic_name="security_type", source_file=fp),
            "agency_code": _require_exact_col(cols, "평가사구분", semantic_name="agency_code", source_file=fp),
            "agency_metadata": _require_exact_col(cols, "평가사명 및 등급", semantic_name="agency_metadata", source_file=fp),
            "eval_date": _require_exact_col(cols, "평가일", semantic_name="rating_date", source_file=fp),
        }
        name_col = _find_col(cols, ["회사명", "기업명", "종목명", "firm_name", "company"])
        market_col = _find_col(cols, ["시장", "market"])
        sec_name_col = _find_col(cols, ["증권명", "security_name"])

        snapshot_parts = raw[found["year"]].apply(_parse_snapshot_period)
        snapshot_frame = pd.DataFrame(
            snapshot_parts.tolist(), index=raw.index,
            columns=["raw_snapshot_year", "raw_snapshot_cutoff", "raw_snapshot_year_raw"],
        )
        date_parts = raw[found["eval_date"]].apply(_parse_rating_date_value)
        date_frame = pd.DataFrame(
            date_parts.tolist(), index=raw.index,
            columns=[
                "rating_date", "rating_date_raw", "rating_date_precision",
                "rating_date_lower_bound", "rating_date_upper_bound",
            ],
        )
        df = pd.DataFrame({
            "firm_id": raw[found["code"]].map(_norm_code),
            "grade_base": raw[found["rating"]],
            "security_type": pd.to_numeric(raw[found["security_code"]], errors="coerce"),
            "agency_code": pd.to_numeric(raw[found["agency_code"]], errors="coerce"),
            "agency_metadata_raw": raw[found["agency_metadata"]].astype("string"),
            "_source_file": fp.name,
            "firm_name": raw[name_col].astype("string") if name_col else pd.Series(pd.NA, index=raw.index, dtype="string"),
            "market": raw[market_col].astype("string") if market_col else pd.Series(pd.NA, index=raw.index, dtype="string"),
            "security_name": raw[sec_name_col].astype("string") if sec_name_col else pd.Series(pd.NA, index=raw.index, dtype="string"),
        })
        df = pd.concat([df, snapshot_frame, date_frame], axis=1)
        frames.append(df)
        file_rows.append({
            "file": fp.name,
            "rows_raw": int(len(raw)),
            "rows_used": int(len(df)),
            "status": "read_strict_schema",
            "selected_raw_columns": {key: str(value) for key, value in found.items()},
        })

    if not frames:
        raise ValueError(f"Raw rating files were found but none contained rows: {raw_rating_dir}")
    raw_all = pd.concat(frames, ignore_index=True, sort=False)
    counts: dict[str, Any] = {
        "raw_all": int(len(raw_all)),
        "files": file_rows,
        "raw_source_columns": source_contract,
        "rating_value_substring_fallback_allowed": False,
        "agency_population_contract": "eligible external ICR sources with agency codes 10/20/30/60/90; fixed before fresh results",
    }

    df = raw_all[raw_all["security_type"].eq(ICR_ALLOWED_SECURITY_CODE)].copy()
    counts["icr_security_code_40"] = int(len(df))
    df = df[df["agency_code"].isin(ALLOWED_MAIN_AGENCY_CODES)].copy()
    counts["allowed_agency_codes_10_20_30_60_90"] = int(len(df))
    df = df[df["grade_base"].notna()].copy()
    counts["rating_notna"] = int(len(df))
    df = add_rating_scale_columns(df, source_col="grade_base")
    df = df[df["rating_num_10"].notna()].copy()
    counts["valid_long_term_grade"] = int(len(df))

    df = df[
        (df["firm_id"] != "")
        & df["raw_snapshot_year"].notna()
        & df["raw_snapshot_cutoff"].notna()
    ].copy()
    counts["valid_firm_snapshot_key"] = int(len(df))
    usable_date = (
        df["rating_date_precision"].isin(["day", "month", "year"])
        & df["rating_date_lower_bound"].notna()
        & df["rating_date_upper_bound"].notna()
    )
    counts["unusable_rating_date_rows"] = int((~usable_date).sum())
    df = df.loc[usable_date].copy()
    counts["usable_rating_event_date"] = int(len(df))
    df["raw_snapshot_year"] = df["raw_snapshot_year"].astype(int)
    df["agency_code"] = pd.to_numeric(df["agency_code"], errors="coerce").astype("Int64")
    df["security_type"] = pd.to_numeric(df["security_type"], errors="coerce").astype("Int64")
    df["agency_name"] = df["agency_code"].map(AGENCY_NAME).astype("string")
    df["agency_pri"] = df["agency_code"].map(AGENCY_PRIORITY).fillna(99).astype(int)

    sample, events, timeline_counts = _build_rating_event_and_state_panels(df)
    counts.update(timeline_counts)
    counts["dedup_firm_year"] = int(len(sample))
    counts["grade_distribution_10"] = sample["grade_base_10"].value_counts(dropna=False).to_dict()
    counts["agency_distribution"] = sample["평가사구분"].astype("string").value_counts(dropna=False).to_dict()
    counts["rating_state_provenance_columns"] = [
        "rating_event_id", "rating_state_cutoff", "source_rating_date",
        "source_rating_date_raw", "source_rating_date_precision",
        "source_rating_period_start", "source_rating_period_end",
        "carried_forward", "rating_age_days", "rating_age_days_min",
        "rating_age_days_max", "rating_observed_in_current_year",
    ]
    if return_events:
        return sample, events, counts
    return sample, counts




def _canonical_columns(panel_path: Path) -> dict[str, str | None]:
    cols = list(pd.read_parquet(panel_path).columns)
    return {
        "code": _find_col(cols, ["stock_code", "financial__stock_code", "rating__stock_code", "code", "거래소코드", "corp_code", "firm_id"]),
        "year": _find_col(cols, ["fiscal_year", "year", "회계년도"]),
        "rating": _find_col(cols, ["rating__rating", "rating", "grade_base", "신용등급"]),
        "security_code": _find_col(cols, ["rating__증권구분", "증권구분", "security_type_code"]),
        "agency_code": _find_col(cols, ["rating__평가사구분", "평가사구분", "agency_code"]),
        "eval_date": _find_col(cols, ["rating__평가일", "평가일", "evaluation_date"]),
        "event_id": _find_col(cols, ["rating__rating_event_id", "rating_event_id"]),
        "state_cutoff": _find_col(cols, ["rating__rating_state_cutoff", "rating_state_cutoff"]),
        "source_period_end": _find_col(cols, ["rating__source_rating_period_end", "source_rating_period_end"]),
        "source_precision": _find_col(cols, ["rating__source_rating_date_precision", "source_rating_date_precision"]),
        "carried_forward": _find_col(cols, ["rating__carried_forward", "carried_forward"]),
        "observed_current_year": _find_col(cols, ["rating__observed_in_current_year", "rating_observed_in_current_year"]),
    }


def validate_stage0_contract(stage0_dir: Path) -> tuple[bool, list[str], dict[str, Any]]:
    panel_path = stage0_dir / "canonical_panel" / "stage0_canonical_panel.parquet"
    if not panel_path.exists():
        return False, [f"missing canonical panel: {panel_path}"], {}
    found = _canonical_columns(panel_path)
    errors = []
    for k in [
        "code", "year", "rating", "security_code", "agency_code", "eval_date",
        "event_id", "state_cutoff", "source_period_end", "source_precision",
        "carried_forward", "observed_current_year",
    ]:
        if found.get(k) is None:
            errors.append(f"stage0 canonical panel missing required rating metadata column: {k}")
    if errors:
        return False, errors, {"inferred_columns": found}

    cols = list(dict.fromkeys(v for v in found.values() if v))
    df = pd.read_parquet(panel_path, columns=cols)
    work = pd.DataFrame({
        "거래소코드": df[found["code"]].map(_norm_code),
        "year": df[found["year"]].map(_norm_year),
        "grade_base": df[found["rating"]],
        "증권구분": pd.to_numeric(df[found["security_code"]], errors="coerce"),
        "평가사구분": pd.to_numeric(df[found["agency_code"]], errors="coerce"),
        "평가일": pd.to_datetime(df[found["eval_date"]], errors="coerce"),
        "rating_event_id": df[found["event_id"]].astype("string"),
        "rating_state_cutoff": pd.to_datetime(df[found["state_cutoff"]], errors="coerce"),
        "source_rating_period_end": pd.to_datetime(df[found["source_period_end"]], errors="coerce"),
        "source_rating_date_precision": df[found["source_precision"]].astype("string"),
        "carried_forward": df[found["carried_forward"]].fillna(False).astype(bool),
        "rating_observed_in_current_year": df[found["observed_current_year"]].fillna(False).astype(bool),
    })
    bad_sec = int((work["증권구분"].notna() & ~work["증권구분"].eq(ICR_ALLOWED_SECURITY_CODE)).sum())
    bad_ag = int((work["평가사구분"].notna() & ~work["평가사구분"].isin(ALLOWED_MAIN_AGENCY_CODES)).sum())
    scaled = add_rating_scale_columns(work, source_col="grade_base")
    bad_grade = int(scaled["rating_num_10"].isna().sum())
    dup = int(work.duplicated(["거래소코드", "year"]).sum())
    future = int((work["source_rating_period_end"] > work["rating_state_cutoff"]).fillna(False).sum())
    bad_flags = int((work["carried_forward"] == work["rating_observed_in_current_year"]).sum())
    if bad_sec:
        errors.append(f"non-ICR 증권구분 rows in Stage0 canonical: {bad_sec}")
    if bad_ag:
        errors.append(f"disallowed 평가사구분 rows in Stage0 canonical: {bad_ag}")
    if bad_grade:
        errors.append(f"invalid/non-long-term rating rows in Stage0 canonical: {bad_grade}")
    if dup:
        errors.append(f"duplicate firm-year rows in Stage0 canonical: {dup}")
    if future:
        errors.append(f"future rating events in Stage0 as-of states: {future}")
    if bad_flags:
        errors.append(f"Stage0 carry-forward/current-year observation flags are not complements: {bad_flags}")
    event_path = stage0_dir / "canonical_panel" / "rating_event_panel.parquet"
    event_rows = 0
    duplicate_events = 0
    if not event_path.exists():
        errors.append(f"missing rating event panel: {event_path}")
    else:
        event_panel = pd.read_parquet(event_path)
        event_rows = int(len(event_panel))
        required_event = {
            "rating_event_id", "firm_id", "rating_date", "rating_date_raw",
            "rating_date_precision", "rating_date_lower_bound",
            "rating_date_upper_bound", "agency_code", "agency_name",
            "grade_raw", "grade_notch", "grade_10", "rating_num_10",
            "security_type", "source_file", "raw_snapshot_year",
        }
        missing_event = sorted(required_event - set(event_panel.columns))
        if missing_event:
            errors.append(f"rating_event_panel missing columns: {missing_event}")
        elif not event_panel.empty:
            duplicate_events = int(event_panel["rating_event_id"].duplicated().sum())
            if duplicate_events:
                errors.append(f"duplicate rating_event_id rows in rating_event_panel: {duplicate_events}")
            precision = event_panel["rating_date_precision"].astype("string")
            exact = pd.to_datetime(event_panel["rating_date"], errors="coerce")
            invalid_precision = int((~precision.isin(["day", "month", "year"])).sum())
            invented_partial_dates = int(((precision != "day") & exact.notna()).sum())
            missing_exact_dates = int(((precision == "day") & exact.isna()).sum())
            if invalid_precision:
                errors.append(f"rating_event_panel invalid date precision rows: {invalid_precision}")
            if invented_partial_dates:
                errors.append(f"rating_event_panel invented exact dates for partial observations: {invented_partial_dates}")
            if missing_exact_dates:
                errors.append(f"rating_event_panel missing exact day dates: {missing_exact_dates}")
    meta = {
        "inferred_columns": found,
        "rows": int(len(work)),
        "bad_security_rows": bad_sec,
        "bad_agency_rows": bad_ag,
        "bad_grade_rows": bad_grade,
        "duplicate_firm_year_rows": dup,
        "future_event_state_rows": future,
        "bad_carry_forward_flag_rows": bad_flags,
        "rating_event_rows": event_rows,
        "duplicate_rating_event_rows": duplicate_events,
    }
    return len(errors) == 0, errors, meta


def repair_stage0_canonical(stage0_dir: Path, raw_rating_dir: Path) -> dict[str, Any]:
    panel_path = stage0_dir / "canonical_panel" / "stage0_canonical_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing Stage0 canonical panel: {panel_path}")
    sample, rating_events, counts = build_rating_sample(raw_rating_dir, return_events=True)
    panel = pd.read_parquet(panel_path)
    cols = _canonical_columns(panel_path)
    code_col, year_col = cols.get("code"), cols.get("year")
    if code_col is None or year_col is None:
        raise KeyError(f"Cannot infer canonical code/year columns for repair: {cols}")

    panel["__repair_code"] = panel[code_col].map(_norm_code)
    panel["__repair_year"] = panel[year_col].map(_norm_year)

    # A stale Stage0 built before the final rating-sampling contract can already
    # contain duplicate firm-year rows.  Repair must collapse the base panel before
    # merging the repaired rating sample; otherwise a correctly de-duplicated
    # rating sample is multiplied back into duplicate canonical rows.
    panel["__nonnull_count"] = panel.notna().sum(axis=1)
    sort_cols = ["__repair_code", "__repair_year", "__nonnull_count"]
    panel = (
        panel.sort_values(sort_cols, ascending=[True, True, False])
        .drop_duplicates(["__repair_code", "__repair_year"], keep="first")
        .drop(columns=["__nonnull_count"], errors="ignore")
        .copy()
    )
    sample_key = sample.rename(columns={"거래소코드": "__repair_code", "year": "__repair_year"})

    # Drop legacy rating columns that violate the final contract, then replace from repaired sample.
    drop_prefixes = ("rating__",)
    drop_exact = {
        "rating", "grade_base", "rating_num", "grade_base_10", "grade_base_7",
        "grade_base_notch", "rating_num_10", "rating_num_7", "rating_num_notch",
        "증권구분", "평가사구분", "평가일", "증권명", "평가사명", "평가사명 및 등급",
        "raw_snapshot_year", "rating_state_cutoff", "rating_event_id",
        "source_rating_date", "source_rating_date_raw", "source_rating_date_precision",
        "source_rating_period_start", "source_rating_period_end", "rating_age_days",
        "rating_age_days_min", "rating_age_days_max", "rating_observed_in_current_year",
        "carried_forward", "rating_observation_status", "representative_rating_contract",
        "rating_year_assignment_source", "source_raw_snapshot_year_first",
        "source_raw_snapshot_year_last", "source_raw_snapshot_occurrence_count",
    }
    drop_cols = [c for c in panel.columns if str(c).startswith(drop_prefixes) or str(c) in drop_exact]
    panel = panel.drop(columns=drop_cols, errors="ignore")

    rename_map = {
        "회사명": "rating__firm_name_raw",
        "시장": "rating__market",
        "평가일": "rating__평가일",
        "평가사구분": "rating__평가사구분",
        "평가사명": "rating__평가사명",
        "증권명": "rating__증권명",
        "증권구분": "rating__증권구분",
        "grade_base_notch": "rating__rating",
        "grade_base_raw": "rating__grade_base_raw",
        "grade_base_10": "rating__grade_base_10",
        "rating_num_10": "rating__rating_num_10",
        "grade_base_7": "rating__grade_base_7",
        "rating_num_7": "rating__rating_num_7",
        "rating_num_notch": "rating__rating_num_notch",
        "agency_pri": "rating__agency_pri",
        "_source_file": "rating__source_file",
        "raw_snapshot_year": "rating__raw_snapshot_year",
        "rating_state_cutoff": "rating__rating_state_cutoff",
        "rating_event_id": "rating__rating_event_id",
        "source_rating_date": "rating__source_rating_date",
        "source_rating_date_raw": "rating__source_rating_date_raw",
        "source_rating_date_precision": "rating__source_rating_date_precision",
        "source_rating_period_start": "rating__source_rating_period_start",
        "source_rating_period_end": "rating__source_rating_period_end",
        "rating_age_days": "rating__rating_age_days",
        "rating_age_days_min": "rating__rating_age_days_min",
        "rating_age_days_max": "rating__rating_age_days_max",
        "rating_observed_in_current_year": "rating__observed_in_current_year",
        "carried_forward": "rating__carried_forward",
        "rating_observation_status": "rating__observation_status",
        "representative_rating_contract": "rating__representative_rating_contract",
        "rating_year_assignment_source": "rating__year_assignment_source",
        "평가사명 및 등급": "rating__agency_metadata_raw",
    }
    sample_cols = ["__repair_code", "__repair_year"] + [c for c in rename_map if c in sample_key.columns]
    sample_key = sample_key[sample_cols].rename(columns=rename_map)
    merged = panel.merge(sample_key, on=["__repair_code", "__repair_year"], how="inner")
    # Enforce the canonical one-row-per-firm-year invariant after merge as well.
    # This protects against accidental duplicate raw keys or legacy canonical aliases.
    merged = merged.drop_duplicates(["__repair_code", "__repair_year"], keep="first").copy()
    merged = merged.drop(columns=["__repair_code", "__repair_year"], errors="ignore")

    # Preserve common unprefixed aliases for old Stage00-01 readers while keeping explicit scale columns.
    if "rating__rating" in merged.columns:
        merged["grade_base"] = merged["rating__grade_base_10"]
        merged["rating"] = merged["rating__rating"]
    if "rating__rating_num_10" in merged.columns:
        merged["rating_num_10"] = pd.to_numeric(merged["rating__rating_num_10"], errors="coerce").astype("Int64")
        merged["rating_num"] = merged["rating_num_10"]
    for c in ["grade_base_10", "grade_base_7", "rating_num_7", "rating_num_notch", "grade_base_notch"]:
        rc = f"rating__{c}"
        if rc in merged.columns:
            merged[c] = merged[rc]

    if merged.empty:
        raise ValueError("Stage0 rating repair produced empty canonical panel; check raw rating keys vs canonical keys")

    backup = panel_path.with_suffix(panel_path.suffix + f".bak_rating_contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    panel_path.replace(backup)
    merged.to_parquet(panel_path, index=False)
    event_path = stage0_dir / "canonical_panel" / "rating_event_panel.parquet"
    event_tmp = event_path.with_suffix(".tmp.parquet")
    rating_events.to_parquet(event_tmp, index=False)
    event_tmp.replace(event_path)

    ok, errors, val_meta = validate_stage0_contract(stage0_dir)
    if not ok:
        raise ValueError("Stage0 repair validation failed: " + "; ".join(errors))

    manifest_update = {
        "semantic_contract_version": "v10_core_semantic_correction_v3_rating_event_asof",
        "contract_version": "stage0_rating_event_asof_v34",
        "status": "PASS",
        "rating_sample_counts": counts,
        "rating_lineage_contract": {
            "rating_value_source": "신용등급",
            "agency_metadata_source": "평가사명 및 등급",
            "firm_year_state": "latest eligible event as of fiscal-period cutoff",
            "representative_selection": "evaluation date first; agency priority only as tie-break",
            "partial_date_policy": "preserve precision and bounds; never invent an exact day",
            "reward_observation_policy": "tplus1 new rating event only; carry-forward is not observed zero",
        },
    }
    for manifest_path in [
        stage0_dir / "stage0_manifest.json",
        stage0_dir / "canonical_panel" / "stage0_manifest.json",
    ]:
        payload: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        payload.update(manifest_update)
        manifest_tmp = manifest_path.with_suffix(".tmp.json")
        manifest_tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        manifest_tmp.replace(manifest_path)

    meta = {
        "stage": "stage0_rating_contract_repair",
        "status": "PASS",
        "semantic_contract_version": "v10_core_semantic_correction_v3_rating_event_asof",
        "created_utc": _now(),
        "raw_rating_dir": str(raw_rating_dir),
        "stage0_dir": str(stage0_dir),
        "backup": str(backup),
        "rows_after_repair": int(len(merged)),
        "rating_event_rows": int(len(rating_events)),
        "rating_event_panel": str(event_path),
        "rating_sample_counts": counts,
        "validation": val_meta,
        "contract": "strict 신용등급 source + event dedup + fiscal-period as-of state + latest evaluation date first + agency priority tie-break + explicit carry-forward provenance + rating_num_10/7/notch scales",
    }
    out_meta = stage0_dir / "canonical_panel" / "stage0_rating_contract_repair_metadata.json"
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--stage0-dir", default=None)
    p.add_argument("--raw-rating-dir", default=None)
    p.add_argument("--force", action="store_true", help="Repair even if Stage0 currently validates.")
    args = p.parse_args(argv)

    root = Path(args.project_root).resolve()
    stage0_dir = Path(args.stage0_dir).resolve() if args.stage0_dir else root / "data" / "final_freeze" / "stage0_oracle_foundation"
    raw_rating_dir = Path(args.raw_rating_dir).resolve() if args.raw_rating_dir else root / "data" / "raw" / "rating_sample"

    ok, errors, meta = validate_stage0_contract(stage0_dir)
    if ok and not args.force:
        result = {"stage": "stage0_rating_contract_repair", "status": "SKIP_ALREADY_VALID", "created_utc": _now(), "validation": meta}
    else:
        result = repair_stage0_canonical(stage0_dir, raw_rating_dir)
        result["pre_repair_errors"] = errors
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
