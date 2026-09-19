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


def restore_data(raw_all: Path, raw_nonfinancial: Path, ratings: Path) -> dict[str, Any]:
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
    write_json(raw_root / "input_receipt.json", receipt)
    contract = verify_input_contract(write=True)
    receipt["input_contract_status"] = contract["status"]
    receipt["input_contract_report"] = "data/raw/input_contract_report.json"
    write_json(raw_root / "input_receipt.json", receipt)
    return receipt
