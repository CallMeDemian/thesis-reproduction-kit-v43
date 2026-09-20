from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pandas as pd

from credit_recourse.analysis.final_v43.paths import frozen_v13_root

from .paths import ROOT, load_json, sha256_file
from .release import load_certified_release

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


def verify_frozen(root: Path = ROOT) -> dict[str, Any]:
    path = Path(root).resolve() / ZIP_RELATIVE
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


def verify_v13_evidence(root: Path = ROOT) -> dict[str, Any]:
    """Verify the certified V1.3 evidence package used by published replay."""
    package = frozen_v13_root(root)
    manifest_path = package / "RELEASE_MANIFEST.json"
    stage8 = package / "frozen_inputs/stage8/canonical_itt_observations.parquet"
    supplemental = package / "canonical_v13/V13_SUPPLEMENTAL_RESULTS.csv"
    parent = package / "canonical_v13/parent_gate_sensitivity_v13.csv"
    parent_flags = package / "canonical_v13/parent_gate_firm_flags_v13.csv"
    table_contract = package / "contracts/THESIS_TABLE_CONTRACT.csv"
    checks: list[dict[str, Any]] = []
    if not manifest_path.is_file():
        return {"status": "FAIL", "error": f"missing V1.3 manifest: {manifest_path}"}
    manifest = load_json(manifest_path)

    actual_stage8_sha = sha256_file(stage8) if stage8.is_file() else None
    checks.append({"name": "stage8_sha256", "expected": manifest.get("source_sha256"), "actual": actual_stage8_sha, "status": "PASS" if actual_stage8_sha == manifest.get("source_sha256") else "FAIL"})
    stage8_rows = pq.read_table(stage8).num_rows if stage8.is_file() else 0
    checks.append({"name": "stage8_rows", "expected": 96600, "actual": stage8_rows, "status": "PASS" if stage8_rows == 96600 else "FAIL"})
    firms = pq.read_table(stage8, columns=["firm_key"]).column("firm_key").to_pylist() if stage8.is_file() else []
    checks.append({"name": "stage8_firms", "expected": 575, "actual": len(set(firms)), "status": "PASS" if len(set(firms)) == 575 else "FAIL"})
    supplemental_rows = len(pd.read_csv(supplemental)) if supplemental.is_file() else 0
    parent_rows = len(pd.read_csv(parent)) if parent.is_file() else 0
    parent_flag_rows = len(pd.read_csv(parent_flags)) if parent_flags.is_file() else 0
    contract = pd.read_csv(table_contract) if table_contract.is_file() else pd.DataFrame()
    checks.extend([
        {"name": "supplemental_rows", "expected": 722, "actual": supplemental_rows, "status": "PASS" if supplemental_rows == 722 else "FAIL"},
        {"name": "parent_rows", "expected": 96, "actual": parent_rows, "status": "PASS" if parent_rows == 96 else "FAIL"},
        {"name": "parent_flag_rows", "expected": 4600, "actual": parent_flag_rows, "status": "PASS" if parent_flag_rows == 4600 else "FAIL"},
        {"name": "table_contract_rows", "expected": 914, "actual": len(contract), "status": "PASS" if len(contract) == 914 and not contract.duplicated("row_id").any() else "FAIL"},
    ])
    return {"status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL", "package": str(package), "manifest": manifest, "stage8_rows": stage8_rows, "supplemental_rows": supplemental_rows, "parent_rows": parent_rows, "registry_rows": len(contract), "checks": checks}


def reproduce_frozen(root: Path = ROOT, run_id: str = "thesis-reproduction") -> dict[str, Any]:
    """Run the certified V1.3 replay in an isolated copy, then retain its receipt."""
    root = Path(root).resolve()
    release = load_certified_release(root)
    distribution = verify_frozen(root)
    evidence = verify_v13_evidence(root)
    run_dir = root / "runs" / run_id / "final"
    run_dir.mkdir(parents=True, exist_ok=True)
    if distribution.get("status") != "PASS" or evidence.get("status") != "PASS":
        receipt = {"status": "FAIL", "release": release, "distribution": distribution, "evidence": evidence}
        (run_dir / "reproduction_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return receipt

    package = frozen_v13_root(root)
    markers = ["PRIMARY_REPRODUCTION_PASS", "SUPPLEMENTAL_REPRODUCTION_PASS", "PARENT_GATE_PASS", "THESIS_CONTRACT_PASS", "PACKAGE_INTEGRITY_PASS"]
    with tempfile.TemporaryDirectory(prefix="thesis_v13_replay_") as temp:
        temp_root = Path(temp)
        temp_package = temp_root / package.name
        shutil.copytree(package, temp_package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        env = {key: value for key, value in os.environ.items() if not key.startswith("THESIS_REPRO_")}
        env["PYTHONPATH"] = os.pathsep.join(str(Path(item)) for item in sys.path if item and Path(item).is_dir())
        env["PYTHONNOUSERSITE"] = "1"
        script = temp_package / "reproduction/reproduce_all_v13.py"
        completed = subprocess.run([sys.executable, str(script)], cwd=temp_package, env=env, capture_output=True, text=True)
        output = f"{completed.stdout}\n{completed.stderr}".strip()
        found = {marker: marker in output for marker in markers}
        generated = temp_root / "repro_audit/v13_refreeze/THESIS_REPRO_KIT_v1.3_FINAL"
        if completed.returncode == 0 and generated.is_dir():
            shutil.copytree(generated, run_dir / "v13_reproduction", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        status = "PASS" if completed.returncode == 0 and all(found.values()) else "FAIL"
        receipt = {"status": status, "release": release, "distribution": distribution, "evidence": evidence, "command": [sys.executable, "reproduction/reproduce_all_v13.py"], "returncode": completed.returncode, "markers": found, "stdout": completed.stdout[-12000:], "stderr": completed.stderr[-12000:], "execution_scope": "PUBLISHED_RESULT_REPRODUCTION", "bootstrap_stream": "FROZEN_PRODUCTION_STREAM"}
    (run_dir / "reproduction_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt

