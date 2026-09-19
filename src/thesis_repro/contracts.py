from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator

from .paths import ROOT, load_json, sha256_file

EXPECTED_REQUESTS = 48_300
FIRM_COUNT = 575
REGIMES = ("baseline", "high")
MATRIX_RELATIVE = Path("frozen/evidence/llm/experiment_matrix.csv")


@dataclass(frozen=True)
class LogicalRequest:
    request_id: str
    generation_regime: str
    cell_id: str
    model: str
    condition: str
    information_condition: str
    budget: str
    replicate_id: int
    firm_ordinal: int
    parent_request_id: str | None
    c3e_reference: str
    response_contract: str


def matrix_rows() -> list[dict[str, str]]:
    path = ROOT / MATRIX_RELATIVE
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def contract_status() -> str:
    return load_json(ROOT / "contracts/llm/final_as_executed_generation_contract.json")["status"]


def historical_snapshot_ready() -> bool:
    """Return whether the migrated historical semantic snapshot is complete."""
    contract = ROOT / "contracts/llm/final_as_executed_generation_contract.json"
    if not contract.is_file():
        return False
    payload = load_json(contract)
    return payload.get("historical_evidence", {}).get("source_snapshot_status") == "MIGRATED_AND_REQUEST_ID_RECONCILED"


def fresh_contract_ready() -> bool:
    path = ROOT / "contracts/llm/fresh_replication_contract.json"
    if not path.is_file():
        return False
    payload = load_json(path)
    return payload.get("status") == "READY_FOR_DRY_RENDER"


def live_runner_ready() -> bool:
    return (ROOT / "src/thesis_repro/live_llm.py").is_file()


def live_execution_authorized() -> bool:
    return os.environ.get("THESIS_REPRO_ENABLE_LIVE_LLM") == "I_APPROVE_FRESH_REPLICATION"


def iter_logical_requests(run_id: str | None = None) -> Iterator[LogicalRequest]:
    rows = matrix_rows()
    if len(rows) != 42:
        raise ValueError(f"expected 42 frozen matrix cells, found {len(rows)}")
    for regime in REGIMES:
        for row in rows:
            for firm in range(1, FIRM_COUNT + 1):
                namespace = f"fresh:{run_id}" if run_id else "historical-plan"
                request_id = f"{namespace}:{regime}:{row['cell_id']}:firm-{firm:04d}"
                parent = row.get("parent_cell_id") or None
                parent_id = f"{namespace}:{regime}:{parent}:firm-{firm:04d}" if parent else None
                yield LogicalRequest(
                    request_id=request_id,
                    generation_regime=regime,
                    cell_id=row["cell_id"],
                    model=row["model"],
                    condition=row["condition"],
                    information_condition=row["info"],
                    budget=row["budget"],
                    replicate_id=int(row["replicate"]),
                    firm_ordinal=firm,
                    parent_request_id=parent_id,
                    c3e_reference="fresh.<run_id>.C3E",
                    response_contract="contracts/scientific/final_freeze/llm_response_schema_v1.json",
                )


def render_requests(limit: int | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
    values = [asdict(item) for item in iter_logical_requests(run_id=run_id)]
    if limit is not None:
        values = values[:limit]
    return values


def contract_report(run_id: str | None = None) -> dict[str, Any]:
    matrix = ROOT / MATRIX_RELATIVE
    generated = sum(1 for _ in iter_logical_requests(run_id=run_id))
    ids = set()
    unresolved = 0
    for item in iter_logical_requests(run_id=run_id):
        if item.request_id in ids:
            raise ValueError(f"duplicate request id: {item.request_id}")
        ids.add(item.request_id)
        if item.parent_request_id and item.parent_request_id not in ids and item.cell_id not in {"C4R", "C6-E", "C6-EX"}:
            unresolved += 1
    return {
        "expected": EXPECTED_REQUESTS,
        "generated": generated,
        "unique_ids": len(ids),
        "matrix_cells": 42,
        "firms": FIRM_COUNT,
        "regimes": list(REGIMES),
        "matrix_sha256": sha256_file(matrix),
        "source_status": contract_status(),
        "request_namespace": f"fresh:{run_id}" if run_id else "historical-plan",
        "historical_snapshot_ready": historical_snapshot_ready(),
        "fresh_contract_ready": fresh_contract_ready(),
        "live_runner_ready": live_runner_ready(),
        "live_execution_authorized": live_execution_authorized(),
        "unresolved_parent_count": unresolved,
        "semantic_basis": "migrated historical logical_requests/provider_inputs/experiment_matrix metadata" if historical_snapshot_ready() else "experiment_matrix contract only",
    }
