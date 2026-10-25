from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import ROOT, load_json, write_json


def _safe_extract(archive_path: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError(f"unsafe archive member: {info.filename}")
            archive.extract(info, destination)
            if not info.is_dir():
                extracted.append(target)
    return extracted


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_header(path: Path) -> list[str]:
    import pandas as pd
    try:
        frame = pd.read_excel(path, nrows=0, engine="calamine")
    except Exception:
        frame = pd.read_excel(path, nrows=0)
    return [str(column) for column in frame.columns]


def _expected_schema(group_root: str, relative: Path) -> dict[str, Any] | None:
    """Columns the raw loaders require, taken from the loader code itself.

    raw_all statement files: build_stage0_foundation_from_raw finds a firm-code and a year column with find_col.
    rating_sample files: rating_contract_repair requires exact column names.
    Other files (ratio sheets, nonfinancial tables) have no column contract in the loaders and stay NOT_CHECKED.
    """
    if group_root.endswith("raw_all"):
        from credit_recourse.oracle.stage0 import build_stage0_foundation_from_raw as stage0
        if "재무비율" in relative.name or stage0.infer_statement_type(relative) is None:
            return None
        return {"loader": "credit_recourse.oracle.stage0.build_stage0_foundation_from_raw", "find_col": {
            "firm_code": [stage0.K_CODE, "종목코드", "회사코드", "stock_code", "code"],
            "year": [stage0.K_YEAR, "사업연도", "결산년도", "fiscal_year", "year"]}}
    if group_root.endswith("rating_sample"):
        from credit_recourse.oracle.stage0 import rating_contract_repair as rating
        return {"loader": "credit_recourse.oracle.stage0.rating_contract_repair", "exact": [
            rating.KEY_KR, rating.YEAR_KR, rating.RATING_KR, "증권구분", "평가사구분", "평가사명 및 등급", "평가일"]}
    return None


def _check_schema(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    try:
        columns = _read_header(path)
    except Exception as exc:
        return {"schema_status": "SCHEMA_MISMATCH", "schema_error": f"unreadable: {type(exc).__name__}"}
    missing = [name for name in expected.get("exact", []) if name not in columns]
    if "find_col" in expected:
        from credit_recourse.oracle.stage0.build_stage0_foundation_from_raw import find_col
        missing += [role for role, candidates in expected["find_col"].items() if find_col(columns, candidates) is None]
    return {"schema_status": "SCHEMA_MISMATCH" if missing else "SCHEMA_MATCH", "schema_loader": expected["loader"], **({"schema_missing": missing} if missing else {})}


def verify_input_contract(*, write: bool = True) -> dict[str, Any]:
    """Check restored private inputs against the authoritative inventory.

    The report is intentionally file-level and machine readable.  Missing raw
    inputs are a normal, explicit state for a clean clone; callers must not
    silently fall back to frozen artifacts.
    """
    inventory_path = ROOT / "contracts/scientific/input_inventory.json"
    raw_root = ROOT / "data/raw"
    inventory = load_json(inventory_path)
    ignored = set(inventory.get("ignored_names", [])) | {"input_receipt.json", "input_contract_report.json"}
    records: list[dict[str, Any]] = []
    required_ok = True
    receipt_hashes: dict[str, str] = {}
    receipt_path = raw_root / "input_receipt.json"
    if receipt_path.is_file():
        try:
            receipt = load_json(receipt_path)
            for archive in receipt.get("archives", []):
                for item in archive.get("files", []):
                    receipt_hashes[str(item.get("path", "")).replace("\\", "/")] = str(item.get("sha256", ""))
        except (OSError, ValueError, TypeError):
            receipt_hashes = {}
    for group in inventory.get("inputs", []):
        target = ROOT / str(group["target_root"])
        if not str(target.relative_to(ROOT)).startswith("data\\raw") and not str(target.relative_to(ROOT)).startswith("data/raw"):
            continue
        for expected in group.get("files", []):
            relative = Path(str(expected["relative_path"]))
            path = target / relative
            row: dict[str, Any] = {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "expected_size_bytes": int(expected["size_bytes"]),
                "optional": bool(expected.get("optional", False)),
                "presence_status": "MISSING",
                "size_status": "NOT_CHECKED",
                "hash_status": "NOT_CONTRACTED",
                "schema_status": "NOT_CHECKED",
            }
            if not path.is_file():
                row["presence_status"] = "OPTIONAL_MISSING" if row["optional"] else "MISSING"
                row["status"] = row["presence_status"]
                if not row["optional"]:
                    required_ok = False
            else:
                row["presence_status"] = "PRESENT"
                actual_size = path.stat().st_size
                row["actual_size_bytes"] = actual_size
                if actual_size != row["expected_size_bytes"]:
                    row["size_status"] = "SIZE_MISMATCH"
                    row["status"] = "SIZE_MISMATCH"
                    if not row["optional"]:
                        required_ok = False
                else:
                    row["size_status"] = "SIZE_MATCH"
                    row["sha256"] = _hash(path)
                    expected_hash = receipt_hashes.get(row["path"])
                    if expected_hash:
                        row["hash_status"] = "HASH_MATCH" if row["sha256"] == expected_hash else "HASH_MISMATCH"
                        if row["hash_status"] == "HASH_MISMATCH" and not row["optional"]:
                            required_ok = False
                    row["status"] = "HASH_MISMATCH" if row["hash_status"] == "HASH_MISMATCH" else "SIZE_MATCH"
                    expected_schema = _expected_schema(str(group["target_root"]).replace("\\", "/"), relative)
                    if expected_schema is not None and path.suffix.lower() == ".xlsx":
                        row.update(_check_schema(path, expected_schema))
                        if row["schema_status"] == "SCHEMA_MISMATCH" and not row["optional"]:
                            required_ok = False
            records.append(row)

    present_paths = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in raw_root.rglob("*")
        if path.is_file() and path.name not in ignored
    } if raw_root.is_dir() else set()
    expected_paths = {row["path"] for row in records}
    extras = sorted(present_paths - expected_paths)
    report: dict[str, Any] = {
        "schema_version": "input_contract_report_v2",
        "inventory": str(inventory_path.relative_to(ROOT)).replace("\\", "/"),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "INPUT_CONTRACT_PASS" if required_ok and records else "INPUT_REQUIRED",
        "required_file_count": sum(not row["optional"] for row in records),
        "present_file_count": sum(row["status"] == "SIZE_MATCH" for row in records),
        "missing_required_count": sum(row["status"] == "MISSING" for row in records),
        "size_mismatch_count": sum(row["status"] == "SIZE_MISMATCH" for row in records),
        "optional_missing_count": sum(row["status"] == "OPTIONAL_MISSING" for row in records),
        "status_vocabulary": ["PRESENT", "MISSING", "SIZE_MATCH", "SIZE_MISMATCH", "HASH_MATCH", "HASH_MISMATCH", "OPTIONAL_MISSING", "SCHEMA_MATCH", "SCHEMA_MISMATCH"],
        "schema_mismatch_count": sum(row.get("schema_status") == "SCHEMA_MISMATCH" for row in records),
        "comparison_basis": {
            "presence_and_size": "each file is compared with relative_path and size_bytes in the inventory",
            "hash": "receipt comparison: sha256 is compared with data/raw/input_receipt.json, which `data restore` writes from the same archives. "
                    "This is self-referential: it shows the files did not change after restore, not that they are the original licensed files "
                    "(the inventory records no per-file hash).",
            "schema": "header check of xlsx files against the columns the raw loaders require (raw_all statement files: firm-code and year "
                      "columns as found by build_stage0_foundation_from_raw.find_col; rating_sample files: the exact columns required by "
                      "rating_contract_repair). Files without a loader column contract stay NOT_CHECKED.",
        },
        "konex_files": {
            "inventory_entries": sum("konex_optional/" in row["path"] for row in records),
            "marked_optional": sum(row["optional"] for row in records if "konex_optional/" in row["path"]),
            "note": "The inventory description mentions optional KONEX files, but no inventory entry carries optional=true, so data doctor "
                    "treats the files under raw_nonfinancial/konex_optional/ as required.",
        },
        "unexpected_raw_files": extras,
        "files": records,
    }
    if write:
        write_json(raw_root / "input_contract_report.json", report)
    return report


def data_doctor() -> dict[str, Any]:
    report = verify_input_contract(write=True)
    report["raw_root"] = str((ROOT / "data/raw").relative_to(ROOT))
    report["raw_data_present"] = report["present_file_count"] > 0
    return report


def validate_stage2_source(root: Path | None = None) -> dict[str, Any]:
    root = Path(root or ROOT).resolve()
    source_root = root / "data/raw/stage2_input_source"
    contract_path = root / "contracts/scientific/stage2_producer_input_contract.json"
    contract = load_json(contract_path)
    required = list(contract["required_files"])
    missing = [relative for relative in required if not (source_root / relative).is_file() or (source_root / relative).stat().st_size <= 0]
    forbidden = [str(path.relative_to(source_root)).replace("\\", "/") for path in source_root.rglob("*") if path.is_file() and any(token in str(path).replace("\\", "/") for token in ("checkpoint", "oracle_score", "C3E_firm"))]
    schema_errors: list[str] = []
    try:
        import pandas as pd
        evaluation = pd.read_parquet(source_root / "input_splits/phase_eval.parquet")
        if len(evaluation) != 575:
            schema_errors.append(f"phase_eval_rows:{len(evaluation)}")
        year_column = "fiscal_year" if "fiscal_year" in evaluation.columns else "year" if "year" in evaluation.columns else None
        if year_column is None or set(pd.to_numeric(evaluation[year_column], errors="coerce").dropna().astype(int)) != {2024}:
            schema_errors.append("phase_eval_year:not_2024")
    except Exception as exc:
        if not (source_root / "input_splits/phase_eval.parquet").is_file():
            schema_errors.append("phase_eval_missing")
        else:
            schema_errors.append(f"phase_eval_unreadable:{type(exc).__name__}")
    return {"status": "PASS" if not missing and not forbidden and not schema_errors else "INPUT_REQUIRED", "role": contract["role"], "source_status": contract["source_status"], "source_root": str(source_root.relative_to(root)).replace("\\", "/"), "required_files": required, "missing": missing, "forbidden_substitutions": forbidden, "schema_errors": schema_errors}


def restore_data(raw_all: Path, raw_nonfinancial: Path, ratings: Path, *, stage2_source: Path | None = None, c6ex_permutation: Path | None = None) -> dict[str, Any]:
    raw_root = ROOT / "data/raw"
    if raw_root.exists():
        for child in raw_root.iterdir():
            if child.name != ".gitkeep":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    mapping = [("raw_all", raw_all), ("raw_nonfinancial", raw_nonfinancial), ("rating_sample", ratings)]
    receipt: dict[str, Any] = {"schema_version": "input_receipt_v1", "created_at": datetime.now(timezone.utc).isoformat(), "archives": [], "status": "PASS"}
    for name, archive in mapping:
        if not archive.is_file():
            receipt["status"] = "FAIL"
            receipt["archives"].append({"name": name, "path": str(archive), "status": "MISSING"})
            continue
        files = _safe_extract(archive, raw_root / name)
        receipt["archives"].append({"name": name, "path": str(archive), "status": "PASS", "file_count": len(files), "files": [{"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "sha256": _hash(path)} for path in files]})
    if stage2_source is not None:
        stage2_root = raw_root / "stage2_input_source"
        files = _safe_extract(stage2_source, stage2_root) if stage2_source.is_file() else []
        validation = validate_stage2_source(ROOT)
        receipt["stage2_source"] = {"path": str(stage2_source), "role": "EXPERIMENTAL_INPUT_NOT_RESULT", "status": "PASS" if stage2_source.is_file() and validation["status"] == "PASS" else "INPUT_REQUIRED", "file_count": len(files), "validation": validation}
        if validation["status"] != "PASS":
            receipt["status"] = "INPUT_REQUIRED"
    if c6ex_permutation is not None:
        destination = ROOT / "data/design/C6EX_permutation.parquet"
        if not c6ex_permutation.is_file():
            receipt["c6ex_permutation"] = {"status": "INPUT_REQUIRED", "reason": "file_missing"}
            receipt["status"] = "INPUT_REQUIRED"
        else:
            expected = "a5385d4c811e73fbf39a46cf299959d72ada8f3931ed3a02a2ba07e4f1c0bd40"
            actual = _hash(c6ex_permutation)
            if actual != expected:
                receipt["c6ex_permutation"] = {"status": "FAILED", "expected_sha256": expected, "actual_sha256": actual}
                receipt["status"] = "FAIL"
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(c6ex_permutation, destination)
                receipt["c6ex_permutation"] = {"status": "PASS", "path": str(destination.relative_to(ROOT)).replace("\\", "/"), "sha256": actual}
    write_json(raw_root / "input_receipt.json", receipt)
    contract = verify_input_contract(write=True)
    receipt["input_contract_status"] = contract["status"]
    receipt["input_contract_report"] = "data/raw/input_contract_report.json"
    write_json(raw_root / "input_receipt.json", receipt)
    return receipt
