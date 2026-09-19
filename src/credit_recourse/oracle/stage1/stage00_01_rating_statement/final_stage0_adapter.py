
from __future__ import annotations

from pathlib import Path
import json
import os
import re
import gc
import shutil
import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # import-safe smoke; materialization still requires parquet support
    pa = None
    pq = None

from credit_recourse.oracle.contracts.rating_scale import (
    add_rating_scale_columns, AGENCY_NAME, AGENCY_PRIORITY, ALLOWED_MAIN_AGENCY_CODES,
    ICR_ALLOWED_SECURITY_CODE, normalize_grade_components, GRADE2NUM_10,
)

K_CODE = "\uac70\ub798\uc18c\ucf54\ub4dc"
K_YEAR = "\ud68c\uacc4\ub144\ub3c4"
K_NAME = "\ud68c\uc0ac\uba85"
K_MARKET = "\uc2dc\uc7a5"

K_BS = "\uc7ac\ubb34\uc0c1\ud0dc\ud45c"
K_IS = "\uc190\uc775\uacc4\uc0b0\uc11c"
K_CF = "\ud604\uae08\ud750\ub984\ud45c"
K_EQ = "\uc790\ubcf8\ubcc0\ub3d9\ud45c"
K_RE = "\uc774\uc775\uc789\uc5ec\uae08\ucc98\ubd84\uacc4\uc0b0\uc11c"
K_RATIO = "\uc7ac\ubb34\ube44\uc728"

STMT_TYPES = [K_BS, K_IS, K_CF, K_EQ, K_RE, K_RATIO]

ALIASES = {
    K_BS: [K_BS, "balance", "balance_sheet", "bs", "statement_of_financial_position", "\uc7ac\ubb34\uc0c1\ud0dc", "\ub300\ucc28\ub300\uc870\ud45c"],
    K_IS: [K_IS, "income", "income_statement", "is", "profit_loss", "pnl", "\uc190\uc775", "\ud3ec\uad04\uc190\uc775"],
    K_CF: [K_CF, "cash", "cashflow", "cash_flow", "cf", "\ud604\uae08\ud750\ub984"],
    K_EQ: [K_EQ, "equity", "capital_change", "changes_in_equity", "\uc790\ubcf8\ubcc0\ub3d9"],
    K_RE: [K_RE, "retained", "appropriation", "retained_earnings", "\uc774\uc775\uc789\uc5ec\uae08"],
    K_RATIO: [K_RATIO, "ratio", "financial_ratio", "ratios", "\uc7ac\ubb34\ube44\uc728"],
}

def _find_col(cols, candidates):
    cols = list(cols)
    for cand in candidates:
        for c in cols:
            if str(c).strip() == cand:
                return c
    for cand in candidates:
        for c in cols:
            if cand.lower() in str(c).lower():
                return c
    return None

def _grade_num(g):
    if pd.isna(g):
        return pd.NA
    _, _, notch, status = normalize_grade_components(g)
    if status != "ok":
        return pd.NA
    return GRADE2NUM_10.get(str(notch).replace("+", "").replace("-", ""), pd.NA)

def _norm_code(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"[^0-9]", "", s)
    return digits.zfill(6) if digits else s

def _norm_year(x):
    if pd.isna(x):
        return pd.NA
    text = str(x).strip()
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return int(match.group(0))
    try:
        year = int(float(x))
        return year if 1900 <= year <= 2100 else pd.NA
    except Exception:
        return pd.NA

def _canon_statement_type(x):
    s = str(x).strip()
    sl = s.lower()
    for canon, aliases in ALIASES.items():
        for a in aliases:
            if str(a).lower() in sl:
                return canon
    return None



def _norm_market(x):
    if pd.isna(x):
        return "UNKNOWN"
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "<na>", "unknown", "unknown_market"}:
        return "UNKNOWN"
    su = s.upper()
    if "KOSPI" in su or "코스피" in s or "유가" in s:
        return "KOSPI"
    if "KOSDAQ" in su or "코스닥" in s:
        return "KOSDAQ"
    if "KONEX" in su or "코넥스" in s:
        return "KONEX"
    return "UNKNOWN"


def _sector7_from_industry_text(x):
    if pd.isna(x):
        return "UNKNOWN"
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "<na>", "unknown"}:
        return "UNKNOWN"
    if any(k in s for k in ["금융", "보험", "증권", "은행", "캐피탈"]):
        return "금융"
    if any(k in s for k in ["제조", "의약", "화학", "전자", "자동차", "기계", "금속", "식품"]):
        return "제조"
    if any(k in s for k in ["건설", "부동산", "토목"]):
        return "건설·부동산"
    if any(k in s for k in ["운수", "물류", "항공", "해운"]):
        return "운수·물류"
    if any(k in s for k in ["전기", "가스", "에너지", "수도"]):
        return "에너지·유틸리티"
    if any(k in s for k in ["정보", "통신", "소프트웨어", "플랫폼", "서비스"]):
        return "정보통신·서비스"
    if any(k in s for k in ["도소매", "유통", "소비", "음식", "숙박"]):
        return "소비재·유통"
    return "UNKNOWN"


def _infer_market_from_filename(name: str) -> str:
    return _norm_market(name)


def _read_excel_with_fallback(fp: Path, **kwargs) -> pd.DataFrame:
    errors = []
    for engine in ["calamine", "openpyxl", None]:
        try:
            if engine is None:
                return pd.read_excel(fp, **kwargs)
            return pd.read_excel(fp, engine=engine, **kwargs)
        except Exception as exc:
            errors.append(f"{engine or 'default'}:{type(exc).__name__}:{exc}")
    raise RuntimeError(f"Failed to read Excel file {fp}: {' | '.join(errors)}")


def _find_project_root_for_raw(stage0_dir: Path) -> Path:
    configured = os.environ.get("THESIS_REPRO_ORACLE_RAW_ROOT")
    if configured:
        return Path(configured).resolve().parent
    for cand in [stage0_dir, *stage0_dir.parents, Path.cwd(), *Path.cwd().parents]:
        if (cand / "data" / "raw").exists():
            return cand
    return stage0_dir.parents[2] if len(stage0_dir.parents) >= 3 else Path.cwd()


def _scan_stage00_01_raw_context(stage0_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build dated context plus genuinely static context from raw workbooks.

    Industry classification is time varying. It is retained only in
    ``code_year`` and is never copied into ``code_static``.
    """
    project_root = _find_project_root_for_raw(stage0_dir)
    raw_root = Path(os.environ.get("THESIS_REPRO_ORACLE_RAW_ROOT", project_root / "data" / "raw")).resolve()
    raw_roots = [raw_root / "raw_all", raw_root / "raw_nonfinancial"]
    rows = []
    static_rows = []
    for raw_root in raw_roots:
        if not raw_root.exists():
            continue
        for fp in sorted(raw_root.rglob("*.xlsx")):
            market = _infer_market_from_filename(fp.name)
            is_general = "일반사항" in fp.name
            if not is_general:
                # Financial statement filename provenance can recover market only, keyed rows if workbook has code/year.
                pass
            try:
                usecols = None
                df = _read_excel_with_fallback(fp, nrows=None if is_general else None)
            except Exception as exc:
                print(f"[adapter][WARN] raw context scan skipped {fp.name}: {exc}", flush=True)
                continue
            code_col = _find_col(df.columns, [K_CODE, "종목코드", "회사코드", "stock_code", "code"])
            year_col = _find_col(df.columns, [K_YEAR, "year", "사업연도", "결산년도"])
            name_col = _find_col(df.columns, [K_NAME, "기업명", "종목명", "firm_name"])
            industry_code_col = _find_col(df.columns, ["산업코드", "표준산업코드", "업종코드"])
            industry_name_col = _find_col(df.columns, ["산업명", "업종명", "산업분류", "업종"])
            business_code_col = _find_col(df.columns, ["업종코드", "상장업종코드"])
            if code_col is None:
                continue
            w = pd.DataFrame({K_CODE: df[code_col].map(_norm_code)})
            w = w[w[K_CODE] != ""].copy()
            if w.empty:
                continue
            if name_col is not None:
                w[K_NAME] = df.loc[w.index, name_col].astype("string")
            w[K_MARKET] = market
            if industry_code_col is not None:
                w["산업코드"] = df.loc[w.index, industry_code_col].astype("string")
            if industry_name_col is not None:
                w["산업명"] = df.loc[w.index, industry_name_col].astype("string")
            if business_code_col is not None:
                w["업종코드"] = df.loc[w.index, business_code_col].astype("string")
            w["stage00_01_context_source"] = str(fp.relative_to(project_root)) if fp.is_relative_to(project_root) else str(fp)
            if year_col is not None:
                wy = w.copy()
                wy["year"] = df.loc[w.index, year_col].map(_norm_year)
                wy = wy[pd.notna(wy["year"])].copy()
                if not wy.empty:
                    wy["year"] = wy["year"].astype("int64")
                    rows.append(wy)
            static_rows.append(w[[c for c in [K_CODE, K_MARKET, K_NAME, "stage00_01_context_source"] if c in w.columns]].copy())
            del df
            gc.collect()
    cols = [K_CODE, "year", K_MARKET, "산업코드", "산업명", "업종코드", K_NAME, "stage00_01_context_source"]
    static_cols = [K_CODE, K_MARKET, K_NAME, "stage00_01_context_source"]
    code_year = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(columns=cols)
    code_static = pd.concat(static_rows, ignore_index=True, sort=False) if static_rows else pd.DataFrame(columns=static_cols)
    if not code_year.empty:
        # Column-wise last-non-null keeps a dated general-info industry value
        # even when another statement workbook contributes only market context
        # for the same firm-year.
        code_year = (code_year[[c for c in cols if c in code_year.columns]]
                     .groupby([K_CODE, "year"], as_index=False, dropna=False)
                     .last())
    if not code_static.empty:
        code_static = (code_static[[c for c in static_cols if c in code_static.columns]]
                       .groupby(K_CODE, as_index=False, dropna=False)
                       .last())
    return code_year, code_static


def _merge_context_backward_asof(out: pd.DataFrame, code_year: pd.DataFrame) -> pd.DataFrame:
    """Attach latest raw context whose source year is not after target year."""
    if code_year.empty:
        return out
    left = out.copy()
    left["_stage00_01_row_order"] = range(len(left))
    right_cols = [
        c for c in [K_CODE, "year", K_MARKET, "산업코드", "산업명", "업종코드", "stage00_01_context_source"]
        if c in code_year.columns
    ]
    right = code_year[right_cols].copy().rename(columns={"year": "_ctx_source_year"})
    right = right.rename(columns={
        c: f"_ctx_{c}" for c in right.columns if c not in [K_CODE, "_ctx_source_year"]
    })
    left["year"] = pd.to_numeric(left["year"], errors="coerce").astype("Int64")
    right["_ctx_source_year"] = pd.to_numeric(right["_ctx_source_year"], errors="coerce").astype("Int64")
    left_valid = left[left[K_CODE].notna() & left["year"].notna()].copy()
    left_invalid = left.drop(index=left_valid.index).copy()
    right = right[right[K_CODE].notna() & right["_ctx_source_year"].notna()].copy()
    if right.empty or left_valid.empty:
        return left.drop(columns=["_stage00_01_row_order"], errors="ignore")
    left_valid = left_valid.sort_values(["year", K_CODE], kind="mergesort")
    right = right.sort_values(["_ctx_source_year", K_CODE], kind="mergesort")
    merged = pd.merge_asof(
        left_valid,
        right,
        left_on="year",
        right_on="_ctx_source_year",
        by=K_CODE,
        direction="backward",
        allow_exact_matches=True,
    )
    if not left_invalid.empty:
        for c in merged.columns:
            if c not in left_invalid.columns:
                left_invalid[c] = pd.NA
        merged = pd.concat([merged, left_invalid[merged.columns]], ignore_index=True)
    merged = (merged.sort_values("_stage00_01_row_order", kind="mergesort")
                    .drop(columns=["_stage00_01_row_order"])
                    .reset_index(drop=True))
    future = merged["_ctx_source_year"].notna() & (merged["_ctx_source_year"] > merged["year"])
    if bool(future.any()):
        raise AssertionError("Stage00_01 backward-as-of context attached future raw general information")
    return merged


def _repair_stage00_01_context(fy: pd.DataFrame, stage0_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    out = fy.copy()
    out[K_CODE] = out[K_CODE].map(_norm_code)
    out["year"] = out["year"].map(_norm_year).astype("Int64")
    if K_MARKET not in out.columns:
        out[K_MARKET] = "UNKNOWN"
    out[K_MARKET] = out[K_MARKET].map(_norm_market).astype("string")
    if "sector_7" not in out.columns:
        out["sector_7"] = "UNKNOWN"
    invalid_sector = {"KOSPI", "KOSDAQ", "KONEX", "UNKNOWN_MARKET", "UNKNOWN", "", "nan", "None", "<NA>"}
    code_year, code_static = _scan_stage00_01_raw_context(stage0_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_year.to_csv(out_dir / "stage00_01_raw_context_code_year_lookup.csv", index=False, encoding="utf-8-sig")
    code_static.to_csv(out_dir / "stage00_01_raw_context_code_static_lookup.csv", index=False, encoding="utf-8-sig")

    if not code_year.empty:
        out = _merge_context_backward_asof(out, code_year)
    if not code_static.empty:
        ctx_cols = [c for c in [K_CODE, K_MARKET, K_NAME, "stage00_01_context_source"] if c in code_static.columns]
        ctx = code_static[ctx_cols].rename(columns={c: f"_ctx_static_{c}" for c in ctx_cols if c != K_CODE})
        out = out.merge(ctx, on=K_CODE, how="left")

    columns_added = []
    for col in ["산업코드", "산업명", "업종코드", "stage00_01_context_source"]:
        if col not in out.columns:
            out[col] = pd.NA
            columns_added.append(col)
        cy = f"_ctx_{col}"
        if cy in out.columns:
            out[col] = out[col].where(out[col].notna() & out[col].astype(str).str.strip().ne(""), out[cy])
        # Industry fields intentionally have no code-static fallback. If no
        # same-year/past snapshot exists, they remain missing.
    # Market repair: preserve known panel market, otherwise use code-year then static filename/context.
    if "_ctx_시장" in out.columns:
        out["_ctx_시장"] = out["_ctx_시장"].map(_norm_market)
        out[K_MARKET] = out[K_MARKET].where(out[K_MARKET] != "UNKNOWN", out["_ctx_시장"])
    if "_ctx_static_시장" in out.columns:
        out["_ctx_static_시장"] = out["_ctx_static_시장"].map(_norm_market)
        out[K_MARKET] = out[K_MARKET].where(out[K_MARKET] != "UNKNOWN", out["_ctx_static_시장"])
    out[K_MARKET] = out[K_MARKET].map(_norm_market).astype("string")

    # sector_7 is industry-derived; never accept market labels as sector values.
    sector_raw = out["sector_7"].astype("string")
    needs_sector = sector_raw.isna() | sector_raw.astype(str).isin(invalid_sector)
    industry_sector = out.get("산업명", pd.Series(pd.NA, index=out.index)).map(_sector7_from_industry_text)
    out.loc[needs_sector, "sector_7"] = industry_sector[needs_sector]
    out["sector_7"] = out["sector_7"].fillna("UNKNOWN").astype("string")

    asof_context_rows = int(out.get("_ctx_source_year", pd.Series(pd.NA, index=out.index)).notna().sum())
    out = out.drop(columns=[c for c in out.columns if c.startswith("_ctx_") or c.startswith("_ctx_static_")], errors="ignore")
    meta = {
        "status": "PASS",
        "rows": int(len(out)),
        "market_known": int(out[K_MARKET].astype(str).ne("UNKNOWN").sum()),
        "sector_known": int(~out["sector_7"].astype(str).isin(list(invalid_sector)).sum()) if False else int(out["sector_7"].astype(str).ne("UNKNOWN").sum()),
        "columns_added": columns_added,
        "columns_present": [c for c in [K_CODE, "year", K_MARKET, "sector_7", "산업코드", "산업명", "업종코드", "stage00_01_context_source"] if c in out.columns],
        "raw_context": {"code_year_lookup_rows": int(len(code_year)), "code_static_lookup_rows": int(len(code_static))},
        "temporal_context_contract": "target_year_exact_or_latest_source_year_le_target; no_static_industry_fallback",
        "asof_context_rows": asof_context_rows,
        "future_context_rows": 0,
    }
    (out_dir / "stage00_01_context_contract_repair.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, meta


def ensure_stage00_01_context_contract(stage0_dir: Path, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    panel_p = out_dir / "firm_year_panel_v1.parquet"
    if not panel_p.exists():
        raise FileNotFoundError(f"Stage00_01 context guard missing firm-year panel: {panel_p}")
    fy = pd.read_parquet(panel_p)
    repaired, meta = _repair_stage00_01_context(fy, Path(stage0_dir), out_dir)
    required = [K_CODE, "year", K_MARKET, "sector_7"]
    missing = [c for c in required if c not in repaired.columns]
    if missing:
        raise RuntimeError(f"Stage00_01 context contract repair failed; missing columns: {missing}")
    repaired.to_parquet(panel_p, index=False)
    return meta

def _safe_read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    # read only required physical columns; this is critical for large statement_items_panel.
    unique_cols = []
    for c in columns:
        if c not in unique_cols:
            unique_cols.append(c)
    return pd.read_parquet(path, columns=unique_cols)


def ensure_stage00_01_statement_coverage_contract(out_dir: Path) -> dict:
    """Materialize six statement-presence flags from actual cleaned panels.

    Stage00_03 historically looked for ``has_*`` columns in the Stage00_01
    firm-year panel.  The memory-lite adapter emitted the six cleaned panels but
    omitted those flags, silently turning statement_coverage_count into a
    constant zero.  Reconstruct membership deterministically from the cleaned
    parquet keys; no raw reread or Stage0 mutation is required.
    """
    out_dir = Path(out_dir)
    panel_path = out_dir / "firm_year_panel_v1.parquet"
    clean_dir = out_dir / "cleaned_statement_panels"
    if not panel_path.exists():
        raise FileNotFoundError(f"Stage00_01 coverage contract missing firm-year panel: {panel_path}")
    fy = pd.read_parquet(panel_path)
    if K_CODE not in fy.columns or "year" not in fy.columns:
        raise KeyError("Stage00_01 coverage contract requires 거래소코드/year keys")
    fy[K_CODE] = fy[K_CODE].map(_norm_code)
    fy["year"] = pd.to_numeric(fy["year"], errors="coerce").astype("Int64")
    target_keys = pd.MultiIndex.from_arrays([fy[K_CODE].astype(str), fy["year"]])
    flag_counts: dict[str, int] = {}
    for statement_type in STMT_TYPES:
        path = clean_dir / f"{statement_type}_clean.parquet"
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Stage00_01 coverage contract missing cleaned panel: {path}")
        keys = pd.read_parquet(path, columns=[K_CODE, "year"])
        keys[K_CODE] = keys[K_CODE].map(_norm_code)
        keys["year"] = pd.to_numeric(keys["year"], errors="coerce").astype("Int64")
        keys = keys.dropna(subset=[K_CODE, "year"]).drop_duplicates([K_CODE, "year"])
        observed_keys = pd.MultiIndex.from_frame(keys[[K_CODE, "year"]])
        flag = f"has_{statement_type}"
        fy[flag] = target_keys.isin(observed_keys).astype("int8")
        flag_counts[flag] = int(fy[flag].sum())
    coverage_cols = [f"has_{statement_type}" for statement_type in STMT_TYPES]
    coverage = fy[coverage_cols].sum(axis=1)
    if int(coverage.nunique(dropna=True)) < 2:
        raise ValueError(
            "Stage00_01 statement coverage remains constant after cleaned-panel membership materialization: "
            f"counts={coverage.value_counts(dropna=False).to_dict()}"
        )
    fy.to_parquet(panel_path, index=False)
    meta = {
        "contract_version": "stage00_01_cleaned_panel_membership_coverage_v1",
        "status": "PASS",
        "source": "cleaned_statement_panels firm-year membership",
        "statement_types": STMT_TYPES,
        "coverage_columns": coverage_cols,
        "flag_positive_rows": flag_counts,
        "coverage_value_counts": {str(k): int(v) for k, v in coverage.value_counts().sort_index().items()},
        "coverage_unique_count": int(coverage.nunique(dropna=True)),
        "raw_or_future_data_used": False,
    }
    (out_dir / "stage00_01_statement_coverage_contract.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta



def _materialize_precomputed_ratio_from_raw_excel(stage0_dir: Path, clean_dir: Path, fy_lookup: pd.DataFrame) -> dict:
    """Materialize raw precomputed financial-ratio Excel files into 재무비율_clean.parquet.

    Stage0 canonical statement_items_panel does not contain financial_ratio rows.
    The original raw_all Excel directory still contains 재무비율 files, so we expose
    them under the legacy Stage00-01 cleaned_statement_panels contract.
    """
    out_p = clean_dir / f"{K_RATIO}_clean.parquet"

    configured_raw_root = os.environ.get("THESIS_REPRO_ORACLE_RAW_ROOT")
    project_root = None
    if configured_raw_root:
        raw_dir = Path(configured_raw_root).resolve() / "raw_all"
    for cand in [stage0_dir, *stage0_dir.parents, Path.cwd(), *Path.cwd().parents]:
        if (cand / "data" / "raw" / "raw_all").exists():
            project_root = cand
            break
    if project_root is None:
        # Fallback for the standard final_freeze layout:
        # <root>/archive/DEPLOYED_RELEASE/stage0_oracle_foundation
        project_root = stage0_dir.parents[2] if len(stage0_dir.parents) >= 3 else Path.cwd()

    if project_root is not None and not configured_raw_root:
        raw_dir = project_root / "data" / "raw" / "raw_all"
    files = sorted([
        f for f in raw_dir.glob("*재무비율*.xlsx")
        if ("코스피" in f.name or "코스닥" in f.name or "코넥스" in f.name)
    ])

    print(f"[adapter] raw ratio dir: {raw_dir}", flush=True)
    print(f"[adapter] raw ratio files found: {len(files)}", flush=True)
    for f in files:
        print(f"[adapter] raw ratio candidate: {f.name}", flush=True)

    if not files:
        pd.DataFrame(columns=[K_NAME, K_CODE, K_YEAR, "year"]).to_parquet(out_p, index=False)
        print(f"[adapter][WARN] no raw ratio xlsx found under {raw_dir}", flush=True)
        return {"ratio_raw_files": 0, "ratio_rows": 0, "ratio_cols": 4, "ratio_note": f"no raw ratio xlsx under {raw_dir}"}

    frames = []
    total_raw_rows = 0

    for fp in files:
        print(f"[adapter] reading raw ratio file: {fp.name}", flush=True)
        try:
            df = pd.read_excel(fp, engine="calamine")
        except Exception:
            df = pd.read_excel(fp)

        if df.empty:
            continue

        code_col = _find_col(df.columns, [K_CODE, "종목코드", "회사코드", "stock_code", "code"])
        year_col = _find_col(df.columns, [K_YEAR, "year", "사업연도", "결산년도"])
        name_col = _find_col(df.columns, [K_NAME, "기업명", "종목명", "firm_name"])

        if code_col is None or year_col is None:
            print(f"[adapter][WARN] skip ratio file without code/year: {fp.name}", flush=True)
            continue

        work = pd.DataFrame({
            K_CODE: df[code_col].map(_norm_code),
            "year": df[year_col].map(_norm_year),
        })
        work = work[(work[K_CODE] != "") & pd.notna(work["year"])].copy()
        work["year"] = work["year"].astype("int64")
        work[K_YEAR] = work["year"].astype(str)

        if name_col is not None:
            work[K_NAME] = df.loc[work.index, name_col].astype("string")

        id_cols = {code_col, year_col}
        if name_col is not None:
            id_cols.add(name_col)

        ratio_cols = []
        for c in df.columns:
            if c in id_cols:
                continue
            cs = str(c).strip()
            if not cs or cs.lower().startswith("unnamed"):
                continue

            s = pd.to_numeric(df.loc[work.index, c], errors="coerce")
            # Keep only columns with at least one numeric observation.
            if int(s.notna().sum()) == 0:
                continue

            safe_name = cs.replace("\n", " ").replace("\r", " ").strip()
            new_col = f"[RAW_RATIO]{safe_name}(NICE원자료)"
            work[new_col] = s
            ratio_cols.append(new_col)

        if ratio_cols:
            keep = [K_CODE, K_YEAR, "year"]
            if K_NAME in work.columns:
                keep.insert(0, K_NAME)
            keep += ratio_cols
            frames.append(work[keep].copy())
            total_raw_rows += len(work)

        del df, work
        gc.collect()

    if not frames:
        pd.DataFrame(columns=[K_NAME, K_CODE, K_YEAR, "year"]).to_parquet(out_p, index=False)
        return {"ratio_raw_files": len(files), "ratio_rows": 0, "ratio_cols": 4, "ratio_note": "raw ratio files found but no numeric ratio columns parsed"}

    ratio = pd.concat(frames, ignore_index=True, sort=False)

    # Prefer canonical firm name from fy_lookup when available.
    if K_NAME in ratio.columns:
        ratio = ratio.drop(columns=[K_NAME], errors="ignore")

    before_filter_rows = len(ratio)
    ratio = ratio.merge(fy_lookup, on=[K_CODE, "year"], how="inner")
    ratio[K_YEAR] = ratio["year"].astype(str)
    filtered_rows = before_filter_rows - len(ratio)
    print(f"[adapter] {K_RATIO}: rating-universe inner filter dropped {filtered_rows:,} rows", flush=True)

    front = [K_NAME, K_CODE, K_YEAR, "year"] if K_NAME in ratio.columns else [K_CODE, K_YEAR, "year"]
    ratio_value_cols = [c for c in ratio.columns if c not in front]

    # Multiple raw files / markets can overlap. Collapse to one firm-year row.
    agg = {c: "last" for c in ratio_value_cols}
    if K_NAME in ratio.columns:
        agg[K_NAME] = "last"

    ratio = ratio.groupby([K_CODE, "year"], as_index=False, dropna=False).agg(agg)
    ratio[K_YEAR] = ratio["year"].astype(str)

    front = [K_NAME, K_CODE, K_YEAR, "year"] if K_NAME in ratio.columns else [K_CODE, K_YEAR, "year"]
    rest = [c for c in ratio.columns if c not in front]
    ratio = ratio[front + rest]

    ratio.to_parquet(out_p, index=False)
    print(f"[adapter] {K_RATIO}: rows={len(ratio):,}, cols={len(ratio.columns):,} from raw Excel", flush=True)

    return {
        "ratio_raw_files": len(files),
        "ratio_raw_rows_read": int(total_raw_rows),
        "ratio_rows": int(len(ratio)),
        "ratio_cols": int(len(ratio.columns)),
        "ratio_output": str(out_p),
    }


def materialize_stage00_01(stage0_dir: Path, out_dir: Path) -> dict:
    can = stage0_dir / "canonical_panel"
    if not can.exists():
        raise FileNotFoundError(f"Missing Stage0 canonical_panel: {can}")

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = out_dir / "cleaned_statement_panels"
    clean_dir.mkdir(exist_ok=True)

    panel_p = can / "stage0_canonical_panel.parquet"
    stmt_p = can / "statement_items_panel.parquet"
    if not panel_p.exists():
        raise FileNotFoundError(f"Missing canonical panel: {panel_p}")
    if not stmt_p.exists():
        raise FileNotFoundError(f"Missing statement items panel: {stmt_p}")

    # ---------- firm-year panel ----------
    if pq is None:
        raise ModuleNotFoundError("pyarrow is required to materialize Stage00_01 parquet schemas")
    panel_cols = pq.read_schema(panel_p).names
    firm_col = _find_col(panel_cols, ["stock_code", "financial__stock_code", "rating__stock_code", "code", K_CODE, "corp_code", "firm_id"])
    year_col = _find_col(panel_cols, ["fiscal_year", "year"])
    name_col = _find_col(panel_cols, ["financial__firm_name_raw", "rating__firm_name_raw", "firm_name_raw", "firm_name", K_NAME])
    market_col = _find_col(panel_cols, ["financial__market", "rating__market", "market", K_MARKET])
    rating_col = _find_col(panel_cols, ["rating__rating", "rating", "grade_base", "신용등급"])
    # Only explicit 10-grade columns are accepted as numeric rating input.
    # Do not substring-match legacy rating__rating_num, because that can bind
    # rating__rating_num_notch and leak the 22-notch scale into Stage00_01.
    rnum_col = _find_col(panel_cols, ["rating__rating_num_10", "rating_num_10"])
    security_col = _find_col(panel_cols, ["rating__증권구분", "증권구분", "security_type_code"])
    agency_col = _find_col(panel_cols, ["rating__평가사구분", "평가사구분", "agency_code"])
    eval_date_col = _find_col(panel_cols, ["rating__평가일", "평가일", "evaluation_date"])
    security_name_col = _find_col(panel_cols, ["rating__증권명", "증권명", "security_name"])
    agency_name_col = _find_col(panel_cols, ["rating__평가사명", "평가사명", "agency", "agency_name"])
    rating_event_id_col = _find_col(panel_cols, ["rating__rating_event_id", "rating_event_id"])
    rating_state_cutoff_col = _find_col(panel_cols, ["rating__rating_state_cutoff", "rating_state_cutoff"])
    source_rating_date_col = _find_col(panel_cols, ["rating__source_rating_date", "source_rating_date"])
    source_rating_date_raw_col = _find_col(panel_cols, ["rating__source_rating_date_raw", "source_rating_date_raw"])
    source_rating_precision_col = _find_col(panel_cols, ["rating__source_rating_date_precision", "source_rating_date_precision"])
    source_rating_period_start_col = _find_col(panel_cols, ["rating__source_rating_period_start", "source_rating_period_start"])
    source_rating_period_end_col = _find_col(panel_cols, ["rating__source_rating_period_end", "source_rating_period_end"])
    carried_forward_col = _find_col(panel_cols, ["rating__carried_forward", "carried_forward"])
    observed_current_year_col = _find_col(panel_cols, ["rating__observed_in_current_year", "rating_observed_in_current_year"])
    rating_age_days_col = _find_col(panel_cols, ["rating__rating_age_days", "rating_age_days"])
    rating_age_days_min_col = _find_col(panel_cols, ["rating__rating_age_days_min", "rating_age_days_min"])
    rating_age_days_max_col = _find_col(panel_cols, ["rating__rating_age_days_max", "rating_age_days_max"])

    needed_panel_cols = [c for c in [firm_col, year_col, name_col, market_col, rating_col, rnum_col,
                                     security_col, agency_col, eval_date_col, security_name_col, agency_name_col,
                                     rating_event_id_col, rating_state_cutoff_col, source_rating_date_col,
                                     source_rating_date_raw_col, source_rating_precision_col,
                                     source_rating_period_start_col, source_rating_period_end_col,
                                     carried_forward_col, observed_current_year_col, rating_age_days_col,
                                     rating_age_days_min_col, rating_age_days_max_col] if c is not None]
    if firm_col is None or year_col is None:
        raise KeyError(f"Cannot infer firm/year columns from canonical panel: {panel_cols}")
    if rating_col is None and rnum_col is None:
        raise KeyError(f"Cannot infer rating columns from canonical panel: {panel_cols}")

    panel = _safe_read_parquet_columns(panel_p, needed_panel_cols)

    fy = pd.DataFrame({
        K_CODE: panel[firm_col].map(_norm_code),
        "year": panel[year_col].map(_norm_year).astype("Int64"),
    })
    if name_col:
        fy[K_NAME] = panel[name_col].astype("string")
    if market_col:
        fy[K_MARKET] = panel[market_col].astype("string")
    if rating_col:
        fy["grade_base"] = panel[rating_col].astype("string")
    if rnum_col:
        fy["rating_num"] = pd.to_numeric(panel[rnum_col], errors="coerce")
    else:
        fy["rating_num"] = fy["grade_base"].map(_grade_num)
    if security_col:
        fy["증권구분"] = pd.to_numeric(panel[security_col], errors="coerce").astype("Int64")
    if agency_col:
        fy["평가사구분"] = pd.to_numeric(panel[agency_col], errors="coerce").astype("Int64")
        fy["평가사명"] = fy["평가사구분"].map(AGENCY_NAME).astype("string")
        fy["agency_pri"] = fy["평가사구분"].map(AGENCY_PRIORITY).fillna(99).astype(int)
    elif agency_name_col:
        fy["평가사명"] = panel[agency_name_col].astype("string")
        fy["agency_pri"] = 99
    else:
        fy["agency_pri"] = 99
    if eval_date_col:
        fy["평가일"] = pd.to_datetime(panel[eval_date_col], errors="coerce")
    else:
        fy["평가일"] = pd.NaT
    if security_name_col:
        fy["증권명"] = panel[security_name_col].astype("string")
    provenance_columns = {
        "rating_event_id": rating_event_id_col,
        "rating_state_cutoff": rating_state_cutoff_col,
        "source_rating_date": source_rating_date_col,
        "source_rating_date_raw": source_rating_date_raw_col,
        "source_rating_date_precision": source_rating_precision_col,
        "source_rating_period_start": source_rating_period_start_col,
        "source_rating_period_end": source_rating_period_end_col,
        "carried_forward": carried_forward_col,
        "rating_observed_in_current_year": observed_current_year_col,
        "rating_age_days": rating_age_days_col,
        "rating_age_days_min": rating_age_days_min_col,
        "rating_age_days_max": rating_age_days_max_col,
    }
    missing_provenance = [name for name, col in provenance_columns.items() if col is None]
    if missing_provenance:
        raise ValueError(f"Stage00_01 requires Stage0 rating-event/as-of provenance columns: {missing_provenance}")
    for name, col in provenance_columns.items():
        if name in {"rating_state_cutoff", "source_rating_date", "source_rating_period_start", "source_rating_period_end"}:
            fy[name] = pd.to_datetime(panel[col], errors="coerce")
        elif name in {"carried_forward", "rating_observed_in_current_year"}:
            fy[name] = panel[col].fillna(False).astype(bool)
        elif name in {"rating_age_days", "rating_age_days_min", "rating_age_days_max"}:
            fy[name] = pd.to_numeric(panel[col], errors="coerce").astype("Int64")
        else:
            fy[name] = panel[col].astype("string")

    # Stage00_01 must validate the Stage0 rating-sampling contract instead of silently reinterpreting it.
    contract_errors = []
    if "증권구분" in fy.columns:
        bad_sec = fy[fy["증권구분"].notna() & ~fy["증권구분"].eq(ICR_ALLOWED_SECURITY_CODE)]
        if len(bad_sec):
            contract_errors.append(f"Stage0 rating sampling violation: non-ICR 증권구분 rows={len(bad_sec)}")
    else:
        contract_errors.append("Stage0 canonical panel lacks 증권구분 metadata required for ICR validation")
    if "평가사구분" in fy.columns:
        bad_ag = fy[fy["평가사구분"].notna() & ~fy["평가사구분"].isin(ALLOWED_MAIN_AGENCY_CODES)]
        if len(bad_ag):
            contract_errors.append(f"Stage0 rating sampling violation: disallowed agency-code rows={len(bad_ag)}")
    else:
        contract_errors.append("Stage0 canonical panel lacks 평가사구분 metadata required for agency-code validation")
    if rating_col is None:
        contract_errors.append("Stage0 canonical panel lacks raw rating/grade_base column")
    future_rating = (
        pd.to_datetime(fy["source_rating_period_end"], errors="coerce")
        > pd.to_datetime(fy["rating_state_cutoff"], errors="coerce")
    ).fillna(False)
    if bool(future_rating.any()):
        contract_errors.append(f"Stage0 rating as-of violation: future events rows={int(future_rating.sum())}")
    bad_flags = fy["carried_forward"] == fy["rating_observed_in_current_year"]
    if bool(bad_flags.any()):
        contract_errors.append(f"Stage0 rating provenance flag violation rows={int(bad_flags.sum())}")
    if contract_errors:
        raise ValueError("; ".join(contract_errors))

    fy = add_rating_scale_columns(fy, source_col="grade_base")
    fy["grade_base"] = fy["grade_base_10"].astype("string")
    fy["rating_num"] = fy["rating_num_10"].astype("Int64")
    fy, context_meta = _repair_stage00_01_context(fy, stage0_dir, out_dir)
    fy["split"] = "dev"
    fy.loc[fy["year"] >= 2020, "split"] = "oot"
    fy["eligible_for_stage2"] = fy["rating_num_10"].notna()
    # Dedup is strictly a Stage0 responsibility.  Stage00_01 is an adapter and
    # must fail fast if duplicate firm-year rows leak through the Stage0 contract.
    fy = fy.dropna(subset=["year"]).copy()
    dup_mask = fy.duplicated([K_CODE, "year"], keep=False)
    if bool(dup_mask.any()):
        dup_sample = fy.loc[dup_mask, [K_CODE, "year", "평가사구분", "평가일"]].head(20).to_dict(orient="records")
        raise ValueError(f"Stage00_01 refuses to silently deduplicate Stage0 firm-year duplicates: count={int(dup_mask.sum())}, sample={dup_sample}")
    fy = fy.sort_values([K_CODE, "year"], ascending=[True, True])
    fy.to_parquet(out_dir / "firm_year_panel_v1.parquet", index=False)

    merge_cols = [K_CODE, "year"]
    if K_NAME in fy.columns:
        merge_cols.append(K_NAME)
    fy_lookup = fy[merge_cols].copy()

    del panel
    gc.collect()

    # ---------- statement items: stream, rating-universe filter, then pivot ----------
    # Patch reservation ORA00-003:
    # Stage0 can contain tens of millions of long statement rows.  Do not create
    # the expensive account-name string column for the full raw universe.  Stream
    # the frozen Stage0 statement_items_panel, keep only firm-years present in
    # the Stage00_01 rating universe, write filtered long shards, and pivot each
    # statement type separately.
    stmt_cols = pq.read_schema(stmt_p).names
    firm2 = _find_col(stmt_cols, ["stock_code", "financial__stock_code", "rating__stock_code", "code", K_CODE, "corp_code", "firm_id"])
    year2 = _find_col(stmt_cols, ["fiscal_year", "year", K_YEAR])
    stmt_col = _find_col(stmt_cols, ["statement_type", "statement", "stmt_type"])
    item_code_col = _find_col(stmt_cols, ["item_code", "account_code", "item_id"])
    item_name_col = _find_col(stmt_cols, ["item_name_raw", "item_name", "account_name"])
    val_col = _find_col(stmt_cols, ["value_numeric", "value", "amount", "value_raw"])

    inferred = {
        "firm": firm2,
        "year": year2,
        "statement_type": stmt_col,
        "item_code": item_code_col,
        "item_name": item_name_col,
        "value": val_col,
    }
    bad = [k for k, v in inferred.items() if v is None]
    if bad:
        raise KeyError(f"statement_items_panel missing inferred columns {bad}; columns={stmt_cols}")

    need_stmt_cols = [firm2, year2, stmt_col, item_code_col, item_name_col, val_col]
    unique_need_stmt_cols = []
    for c in need_stmt_cols:
        if c not in unique_need_stmt_cols:
            unique_need_stmt_cols.append(c)

    # Normalize the firm-year lookup once and use a string key for fast vectorized
    # membership checks inside each parquet batch.
    fy_lookup[K_CODE] = fy_lookup[K_CODE].map(_norm_code).astype("object")
    fy_lookup["year"] = pd.to_numeric(fy_lookup["year"].map(_norm_year), errors="coerce").astype("Int64")
    fy_lookup = fy_lookup.dropna(subset=["year"]).copy()
    fy_lookup["year"] = fy_lookup["year"].astype("int64")
    fy_lookup["_key"] = fy_lookup[K_CODE].astype("object") + "|" + fy_lookup["year"].astype(str).astype("object")
    lookup_keys = set(fy_lookup["_key"].dropna().tolist())
    if not lookup_keys:
        raise ValueError("Stage00_01 has no rating-universe firm-year keys to filter statement items")

    tmp_long_dir = out_dir / "_stage00_01_statement_long_shards"
    if tmp_long_dir.exists():
        shutil.rmtree(tmp_long_dir)
    tmp_long_dir.mkdir(parents=True, exist_ok=True)

    raw_audit_p = out_dir / "financial_statement_items_raw.parquet"
    raw_audit_p.parent.mkdir(parents=True, exist_ok=True)
    if raw_audit_p.exists():
        raw_audit_p.unlink()

    stmt_to_idx = {stmt: idx for idx, stmt in enumerate(STMT_TYPES)}
    stmt_counts_raw: dict[str, int] = {}
    stmt_counts_filtered: dict[str, int] = {}
    total_statement_rows = 0
    kept_statement_rows = 0
    raw_audit_writer = None
    batch_size = 250_000

    def _append_raw_audit(df: pd.DataFrame):
        nonlocal raw_audit_writer
        if df.empty:
            return
        if pa is None:
            raise ModuleNotFoundError("pyarrow is required to write Stage00_01 raw audit parquet")
        table = pa.Table.from_pandas(df, preserve_index=False)
        if raw_audit_writer is None:
            raw_audit_writer = pq.ParquetWriter(raw_audit_p, table.schema, compression="snappy")
        raw_audit_writer.write_table(table)

    try:
        pf = pq.ParquetFile(stmt_p)
        for batch_idx, batch in enumerate(pf.iter_batches(columns=unique_need_stmt_cols, batch_size=batch_size), start=1):
            raw = batch.to_pandas()
            total_statement_rows += int(len(raw))
            if raw.empty:
                continue

            firm_s = raw[firm2].map(_norm_code).astype("object")
            year_s = pd.to_numeric(raw[year2].map(_norm_year), errors="coerce").astype("Int64")
            stmt_s_all = raw[stmt_col].map(_canon_statement_type).astype("object")
            key_s = firm_s + "|" + year_s.astype("string").astype("object")

            raw_counts = stmt_s_all.value_counts(dropna=False).to_dict()
            for k, v in raw_counts.items():
                kk = str(k)
                stmt_counts_raw[kk] = stmt_counts_raw.get(kk, 0) + int(v)

            # rating-universe statement filter: this must happen before `_col`
            # construction to avoid materializing account-name strings for
            # non-sample firm-years.
            mask = key_s.isin(lookup_keys) & stmt_s_all.isin(STMT_TYPES) & firm_s.astype(str).ne("") & year_s.notna()
            if not bool(mask.any()):
                del raw, firm_s, year_s, stmt_s_all, key_s, mask
                if batch_idx % 20 == 0:
                    gc.collect()
                continue

            sub = pd.DataFrame({
                "_firm": firm_s.loc[mask].astype("object"),
                "_year": year_s.loc[mask].astype("int64"),
                "_stmt": stmt_s_all.loc[mask].astype("object"),
                "_icode": raw.loc[mask, item_code_col].map(lambda x: "" if pd.isna(x) else str(x).strip()).astype("object"),
                "_iname": raw.loc[mask, item_name_col].map(lambda x: "" if pd.isna(x) else str(x).strip()).astype("object"),
                "_value": pd.to_numeric(raw.loc[mask, val_col], errors="coerce"),
            })
            sub["_col"] = "[" + sub["_icode"] + "]" + sub["_iname"] + "(IFRS)(천원)"
            sub = sub[["_firm", "_year", "_stmt", "_icode", "_iname", "_value", "_col"]]

            kept_statement_rows += int(len(sub))
            filt_counts = sub["_stmt"].value_counts(dropna=False).to_dict()
            for k, v in filt_counts.items():
                kk = str(k)
                stmt_counts_filtered[kk] = stmt_counts_filtered.get(kk, 0) + int(v)

            _append_raw_audit(sub)
            for stmt in STMT_TYPES:
                stmt_sub = sub.loc[sub["_stmt"].eq(stmt), ["_firm", "_year", "_col", "_value"]]
                if stmt_sub.empty:
                    continue
                shard_p = tmp_long_dir / f"stmt_{stmt_to_idx[stmt]:02d}_{batch_idx:06d}.parquet"
                stmt_sub.to_parquet(shard_p, index=False)

            if batch_idx % 20 == 0:
                print(
                    f"[adapter] statement stream batch={batch_idx:,} total_rows={total_statement_rows:,} kept_rows={kept_statement_rows:,}",
                    flush=True,
                )
                gc.collect()
            del raw, firm_s, year_s, stmt_s_all, key_s, mask, sub
    finally:
        if raw_audit_writer is not None:
            raw_audit_writer.close()

    if raw_audit_writer is None:
        pd.DataFrame(columns=["_firm", "_year", "_stmt", "_icode", "_iname", "_value", "_col"]).to_parquet(raw_audit_p, index=False)

    print(
        f"[adapter] rating-universe statement filter: kept {kept_statement_rows:,} / {total_statement_rows:,} rows",
        flush=True,
    )

    # Keep merge keys plain/object + int64 so downstream merges cannot be tripped
    # by nullable/StringDtype alignment differences.
    fy_lookup_merge = fy_lookup.drop(columns=["_key"], errors="ignore").copy()
    fy_lookup_merge[K_CODE] = fy_lookup_merge[K_CODE].astype("object")
    fy_lookup_merge["year"] = fy_lookup_merge["year"].astype("int64")

    for stmt in STMT_TYPES:
        print(f"[adapter] materializing {stmt} ...", flush=True)
        shard_files = sorted(tmp_long_dir.glob(f"stmt_{stmt_to_idx[stmt]:02d}_*.parquet"))
        if not shard_files:
            pd.DataFrame(columns=[K_NAME, K_CODE, K_YEAR, "year"]).to_parquet(
                clean_dir / f"{stmt}_clean.parquet", index=False
            )
            print(f"[adapter] {stmt}: empty", flush=True)
            continue

        sub = pd.concat((pd.read_parquet(p) for p in shard_files), ignore_index=True, sort=False)
        sub = sub.dropna(subset=["_firm", "_year", "_col"])
        sub["_firm"] = sub["_firm"].astype("object")
        sub["_year"] = pd.to_numeric(sub["_year"], errors="coerce").astype("int64")
        # Reduce duplicate pressure before pivot.
        sub = sub.drop_duplicates(["_firm", "_year", "_col"], keep="last")

        wide = sub.pivot(index=["_firm", "_year"], columns="_col", values="_value").reset_index()
        wide.columns.name = None
        wide = wide.rename(columns={"_firm": K_CODE, "_year": "year"})
        wide[K_CODE] = wide[K_CODE].astype("object")
        wide["year"] = wide["year"].astype("int64")
        wide[K_YEAR] = wide["year"].astype(str)

        before_filter_rows = len(wide)
        wide = wide.merge(fy_lookup_merge, on=[K_CODE, "year"], how="inner")
        filtered_rows = before_filter_rows - len(wide)
        if filtered_rows:
            print(f"[adapter] {stmt}: rating-universe inner filter dropped {filtered_rows:,} rows", flush=True)

        front = [K_NAME, K_CODE, K_YEAR, "year"] if K_NAME in wide.columns else [K_CODE, K_YEAR, "year"]
        rest = [c for c in wide.columns if c not in front]
        wide = wide[front + rest]

        out_p = clean_dir / f"{stmt}_clean.parquet"
        wide.to_parquet(out_p, index=False)
        print(f"[adapter] {stmt}: rows={len(wide):,}, cols={len(wide.columns):,}", flush=True)

        del sub, wide
        gc.collect()

    shutil.rmtree(tmp_long_dir, ignore_errors=True)

    # Stage0 canonical statement_items_panel has no financial_ratio rows.
    # Recover raw precomputed financial ratios from data/raw/raw_all/*재무비율*.xlsx.
    ratio_meta = _materialize_precomputed_ratio_from_raw_excel(stage0_dir, clean_dir, fy_lookup_merge)
    coverage_meta = ensure_stage00_01_statement_coverage_contract(out_dir)

    meta = {
        "stage": "stage00_01_final_stage0_adapter_memlite",
        "status": "PASS",
        "semantic_contract_version": "v10_core_semantic_correction_v3_rating_event_asof",
        "read_only_stage0": str(stage0_dir),
        "output": str(out_dir),
        "firm_year_rows": int(len(fy)),
        "context_contract_repair": context_meta,
        "cleaned_statement_files": int(len(list(clean_dir.glob("*_clean.parquet")))),
        "precomputed_ratio_materialization": ratio_meta,
        "statement_coverage_contract": coverage_meta,
        "statement_type_canonical_counts_raw": {str(k): int(v) for k, v in stmt_counts_raw.items()},
        "statement_type_canonical_counts_after_rating_filter": {str(k): int(v) for k, v in stmt_counts_filtered.items()},
        "statement_rows_total_before_rating_filter": int(total_statement_rows),
        "statement_rows_kept_after_rating_filter": int(kept_statement_rows),
        "methodology_note": "Stage0 is read-only to this adapter and must already carry the v3 rating-event/as-of contract. The memory-lite adapter streams statement rows, filters to the corrected rating-state universe before constructing expensive account-name string columns, writes filtered long shards, and pivots statement types independently."
    }
    (out_dir / "stage00_01_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return meta
