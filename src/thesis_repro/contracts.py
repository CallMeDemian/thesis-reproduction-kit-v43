from __future__ import annotations

import csv
import hashlib
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


def iter_logical_requests() -> Iterator[LogicalRequest]:
    rows = matrix_rows()
    if len(rows) != 42:
        raise ValueError(f"expected 42 frozen matrix cells, found {len(rows)}")
    for regime in REGIMES:
        for row in rows:
            for firm in range(1, FIRM_COUNT + 1):
                request_id = f"v43:{regime}:{row['cell_id']}:firm-{firm:04d}"
                parent = row.get("parent_cell_id") or None
                parent_id = f"v43:{regime}:{parent}:firm-{firm:04d}" if parent else None
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


def render_requests(limit: int | None = None) -> list[dict[str, Any]]:
    values = [asdict(item) for item in iter_logical_requests()]
    if limit is not None:
        values = values[:limit]
    return values


def contract_report() -> dict[str, Any]:
    matrix = ROOT / MATRIX_RELATIVE
    generated = sum(1 for _ in iter_logical_requests())
    ids = set()
    unresolved = 0
    for item in iter_logical_requests():
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
        "unresolved_parent_count": unresolved,
        "semantic_warning": "The exact historical provider request snapshot is unavailable; identities are a deterministic product plan, not a historical byte replay.",
    }

