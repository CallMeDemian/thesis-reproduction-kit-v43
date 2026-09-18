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


def data_doctor() -> dict[str, Any]:
    inventory = ROOT / "contracts/scientific/input_inventory.json"
    raw = ROOT / "data/raw"
    return {
        "status": "PASS" if inventory.is_file() and raw.is_dir() else "FAIL",
        "inventory": str(inventory.relative_to(ROOT)),
        "raw_root": str(raw.relative_to(ROOT)),
        "raw_file_count": sum(1 for path in raw.rglob("*") if path.is_file() and path.name != ".gitkeep"),
        "raw_data_present": any(path.is_file() and path.name != ".gitkeep" for path in raw.rglob("*")),
    }


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
    return receipt

