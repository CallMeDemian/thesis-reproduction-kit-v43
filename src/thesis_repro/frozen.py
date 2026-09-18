from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

from .paths import ROOT, load_json, sha256_file

ZIP_RELATIVE = Path("frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip")
ZIP_SHA256 = "931631d52541bb6dfdd5ed9fda767e2e48ccb8b7521f70ae771db68df2c1831d"
STAGE8_MEMBER = "THESIS_REPRO_KIT_v2.1.1_FINAL/frozen_replay/v1.3/frozen_inputs/stage8/canonical_itt_observations.parquet"
REGISTRY_MEMBER = "THESIS_REPRO_KIT_v2.1.1_FINAL/evidence/RESULT_REGISTRY.csv"
STAGE8_SHA256 = "aedfadbf1728a14d3e4cb45bfa8a022d2c10313c52606e7bd1aee68899de8a0d"
REGISTRY_SHA256 = "6d43643da844315c3a531cdb07c5dddc8f5fef5fc91b00f117f830b6c95e1b56"


def _member_sha256(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen() -> dict[str, Any]:
    path = ROOT / ZIP_RELATIVE
    result: dict[str, Any] = {"status": "PASS", "checks": []}
    if not path.is_file():
        return {"status": "FAIL", "error": f"missing frozen distribution: {path}"}
    actual_zip = sha256_file(path)
    result["checks"].append({"name": "distribution_sha256", "expected": ZIP_SHA256, "actual": actual_zip, "status": "PASS" if actual_zip == ZIP_SHA256 else "FAIL"})
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for member in (STAGE8_MEMBER, REGISTRY_MEMBER):
            if member not in names:
                result["checks"].append({"name": member, "status": "FAIL", "error": "missing member"})
                continue
            expected = STAGE8_SHA256 if member == STAGE8_MEMBER else REGISTRY_SHA256
            actual = _member_sha256(archive, member)
            result["checks"].append({"name": member, "expected": expected, "actual": actual, "status": "PASS" if actual == expected else "FAIL"})
            if member == REGISTRY_MEMBER:
                with archive.open(member) as handle:
                    count = sum(1 for _ in csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))
                result["registry_rows"] = count
                result["checks"].append({"name": "registry_row_count", "expected": 914, "actual": count, "status": "PASS" if count == 914 else "FAIL"})
    result["stage8_expected_rows"] = 96600
    result["primary_expected_rows"] = 96
    result["supplemental_expected_rows"] = 722
    result["parent_expected_rows"] = 96
    if any(item.get("status") == "FAIL" for item in result["checks"]):
        result["status"] = "FAIL"
    return result

