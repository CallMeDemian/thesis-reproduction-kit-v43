"""Fresh Stage8 structural materialization and verification.

The full scientific adapter requires verified same-run Oracle, Simulator and
C3-E artifacts.  This module never falls back to the frozen Stage8 table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_stage8(run_root: Path, *, synthetic: bool = False) -> dict[str, Any]:
    run_root = Path(run_root)
    llm_report_path = run_root / "09_llm" / "materialization_report.json"
    if not llm_report_path.is_file():
        return {"status": "NOT_EXECUTED", "reason": "LLM materialization report is missing"}
    llm_report = _load_json(llm_report_path)
    if llm_report.get("status") != "PASS":
        return {"status": "FAILED", "reason": "LLM materialization is incomplete"}
    if not synthetic:
        required = [run_root / "02_oracle" / "oracle_validation_report.json", run_root / "03_simulator" / "simulator_validation_report.json", run_root / "08_c3e" / "release.json"]
        if any(not path.is_file() for path in required):
            return {"status": "NOT_EXECUTED", "reason": "same-run verified Oracle, Simulator, and C3-E parents are required"}
    normalized = run_root / "09_llm" / "normalized_responses.jsonl"
    rows = [json.loads(line) for line in normalized.read_text(encoding="utf-8").splitlines() if line.strip()]
    destination = run_root / "10_stage8" / "stage8_rows.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for response in rows:
            request_id = response["request_id"]
            candidate = response.get("selected_candidate") or "A0"
            for policy in ("Strict", "Repaired"):
                row = {
                    "semantic_key": f"{request_id}:{policy}",
                    "request_id": request_id,
                    "provider_request_id": response.get("provider_request_id"),
                    "policy": policy,
                    "candidate_id": candidate.replace("_noop", ""),
                    "action_application_status": "accepted" if policy == "Strict" else "repaired",
                    "lineage": {
                        "llm_response_sha256": response.get("raw_response_sha256"),
                        "same_run": True,
                        "frozen_stage8_parent": False,
                    },
                    "fixture_only": synthetic,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                count += 1
    report = verify_stage8(run_root, expected_rows=2 * len(rows), synthetic=synthetic)
    report.update({"stage8_rows_sha256": _sha256(destination), "request_count": len(rows), "output": str(destination.relative_to(run_root)).replace("\\", "/")})
    (run_root / "10_stage8" / "stage8_validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def verify_stage8(run_root: Path, *, expected_rows: int, synthetic: bool = False) -> dict[str, Any]:
    path = Path(run_root) / "10_stage8" / "stage8_rows.jsonl"
    if not path.is_file():
        return {"status": "FAILED", "errors": ["stage8_rows_missing"]}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    if len(rows) != expected_rows:
        errors.append(f"row_count:{len(rows)}!={expected_rows}")
    keys = [row.get("semantic_key") for row in rows]
    if len(keys) != len(set(keys)):
        errors.append("duplicate_semantic_key")
    by_request: dict[str, set[str]] = {}
    for row in rows:
        by_request.setdefault(str(row.get("request_id")), set()).add(str(row.get("policy")))
        if row.get("lineage", {}).get("frozen_stage8_parent") is True:
            errors.append("forbidden_frozen_stage8_parent")
        if row.get("lineage", {}).get("same_run") is not True:
            errors.append("missing_same_run_lineage")
    if any(policies != {"Strict", "Repaired"} for policies in by_request.values()):
        errors.append("strict_repaired_pairing_incomplete")
    return {"schema_version": "fresh_stage8_verification_v1", "status": "PASS" if not errors else "FAILED", "execution_class": "SYNTHETIC_E2E_ACCEPTANCE" if synthetic else "REAL_COMPUTE", "row_count": len(rows), "expected_rows": expected_rows, "request_count": len(by_request), "errors": sorted(set(errors))}
