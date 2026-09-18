"""Read only canonical financial fields, with the existing registry precedence."""
from __future__ import annotations

from dataclasses import fields
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from credit_recourse.contracts.account_registry import ACCOUNT_REGISTRY, aliases_for, _find_column
from .firm_state import FirmState


def firm_key(value):
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if s[:1].upper() == "A" and s[1:].isdigit():
        s = s[1:]
    return s.zfill(6) if s.isdigit() else s


def read_financial_panel(path: Path):
    """Project parquet BEFORE reading values: no scores/features are loaded.

    The coalesce order is resolve_mapping_value's alias/code order. Exact
    selected column is retained for each field/row; unobserved stays null.
    """
    columns = pq.ParquetFile(path).schema.names
    hits = {field: list(dict.fromkeys(c for token in aliases_for(field)
                                    if (c := _find_column(columns, token)) is not None))
            for field in ACCOUNT_REGISTRY}
    keycols = ["firm_id", "fiscal_year"]
    if "canonical_business_plan_history_row_id" in columns:
        keycols.append("canonical_business_plan_history_row_id")
    wanted = list(dict.fromkeys(keycols + [c for group in hits.values() for c in group]))
    raw = pd.read_parquet(path, columns=wanted)
    out = raw[keycols].copy()
    out["firm_id"] = out.firm_id.map(firm_key)
    out["fiscal_year"] = pd.to_numeric(out.fiscal_year, errors="raise").astype(int)
    if out.duplicated(["firm_id", "fiscal_year"]).any():
        raise ValueError(f"Duplicate canonical financial key: {path}")
    lineage = {}
    for field, choices in hits.items():
        value = pd.Series(float("nan"), index=raw.index)
        source = pd.Series(None, index=raw.index, dtype=object)
        for c in choices:
            available = raw[c].notna() & source.isna()
            value.loc[available] = pd.to_numeric(raw.loc[available, c], errors="raise")
            source.loc[available] = c
        out[field] = value
        out["source__"+field] = source
        lineage[field] = {"precedence": choices, "selected_column_counts": source.value_counts().to_dict(),
                          "unobserved_count": int(source.isna().sum())}
    return out.sort_values(["firm_id", "fiscal_year"]).reset_index(drop=True), lineage


def state_from_record(row):
    names = {f.name for f in fields(FirmState)} - {"firm_id", "year", "sector", "rating_num", "rating_grade"}
    values = {k: float(row[k]) for k in names if k in row and row[k] is not None
              and math.isfinite(float(row[k]))}
    return FirmState(firm_id=firm_key(row["firm_id"]), year=int(row["fiscal_year"]),
                     sector="Unknown", **values)
