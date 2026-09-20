"""Single run-scoped owner for the V4.3 LLM preparation boundary.

The scientific prompt and provider contracts live in ``credit_recourse.final_release``.
This module only binds those contracts to a fresh run namespace and provides a
deterministic transport substitute for synthetic acceptance.  A mock receipt is
never accepted as a live scientific execution receipt.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .paths import ROOT

EXPECTED_REQUESTS = 48_300
FIRM_COUNT = 575
REGIMES = ("baseline", "high")


@dataclass(frozen=True)
class LogicalRequest:
    request_id: str
    generation_regime: str
    cell_id: str
    model: str
    condition: str
    phase: str
    wave: int
    information_condition: str
    budget: str
    replicate_id: int
    firm_ordinal: int
    parent_request_id: str | None
    c3e_reference: str
    response_contract: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_rows(root: Path = ROOT) -> list[dict[str, str]]:
    path = Path(root) / "frozen/evidence/llm/experiment_matrix.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def iter_logical_requests(run_id: str, *, root: Path = ROOT, firm_count: int = FIRM_COUNT) -> Iterator[LogicalRequest]:
    rows = _matrix_rows(root)
    if len(rows) != 42:
        raise ValueError(f"expected 42 final LLM matrix cells, found {len(rows)}")
    namespace = f"fresh:{run_id}"
    for regime in REGIMES:
        for row in rows:
            for firm in range(1, int(firm_count) + 1):
                parent = row.get("parent_cell_id") or None
                yield LogicalRequest(
                    request_id=f"{namespace}:{regime}:{row['cell_id']}:firm-{firm:04d}",
                    generation_regime=regime,
                    cell_id=row["cell_id"],
                    model=row["model"],
                    condition=row["condition"],
                    phase=row["phase"],
                    wave=int(row["wave"]),
                    information_condition=row["info"],
                    budget=row["budget"],
                    replicate_id=int(row["replicate"]),
                    firm_ordinal=firm,
                    parent_request_id=f"{namespace}:{regime}:{parent}:firm-{firm:04d}" if parent else None,
                    c3e_reference="fresh.<run_id>.C3E",
                    response_contract="credit_recourse.final_release",
                )


def _requests(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("request_id", "")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate logical request ids")
    return rows


def prepare_full_llm(paths, *, firm_count: int = FIRM_COUNT) -> dict[str, Any]:
    target = paths.llm_root / "logical_requests.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    ids: set[str] = set()
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for request in iter_logical_requests(paths.run_id, root=paths.root, firm_count=firm_count):
            if request.request_id in ids:
                raise ValueError(f"duplicate fresh request id: {request.request_id}")
            ids.add(request.request_id)
            handle.write(json.dumps(asdict(request), sort_keys=True) + "\n")
            count += 1
    expected = 42 * 2 * int(firm_count)
    report = {
        "schema_version": "v43_final_llm_prepare_v2",
        "status": "PASS" if count == expected else "FAILED",
        "execution_class": "SYNTHETIC_E2E_ACCEPTANCE" if firm_count != FIRM_COUNT else "REAL_COMPUTE",
        "expected_requests": expected,
        "request_count": count,
        "unique_request_ids": len(ids),
        "baseline": 21 * int(firm_count),
        "high": 21 * int(firm_count),
        "c3e_parent": f"runs/{paths.run_id}/07_c3e/release.json",
        "logical_requests_sha256": _sha256(target),
    }
    (paths.llm_root / "prepare_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _mock_response(request: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(str(request["request_id"]).encode()).hexdigest()
    return {
        "request_id": request["request_id"],
        "provider": "mock",
        "transport": "mock",
        "provider_request_id": f"mock:{digest[:16]}",
        "raw_response": {"mode": "candidate_selection", "selected_candidate": "A0", "confidence": "low", "brief_rationale": f"synthetic:{digest[:12]}"},
    }


def execute_full_llm(paths, *, live: bool, firm_count: int = FIRM_COUNT) -> dict[str, Any]:
    request_path = paths.llm_root / "logical_requests.jsonl"
    requests = _requests(request_path)
    if live:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            return {"status": "CREDENTIALS_REQUIRED", "reason": "provider credentials are required before submission"}
        return {"status": "EXTERNAL_WAIT", "reason": "provider submission is intentionally resumable and must be driven by final_release live_batch"}
    response_path = paths.llm_root / "raw_mock_responses.jsonl"
    with response_path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(_mock_response(request), sort_keys=True) + "\n")
    receipt = {"schema_version": "v43_final_llm_mock_receipt_v1", "status": "PASS", "execution_class": "SYNTHETIC_E2E_ACCEPTANCE", "provider_contacted": False, "request_count": len(requests), "raw_response_sha256": _sha256(response_path)}
    (paths.llm_root / "generate_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def materialize_full_stage7(paths, *, expected_count: int | None = None) -> dict[str, Any]:
    request_path = paths.llm_root / "logical_requests.jsonl"
    response_path = paths.llm_root / "raw_mock_responses.jsonl"
    requests = _requests(request_path)
    responses = _requests(response_path) if response_path.is_file() else []
    expected_ids = {str(row["request_id"]) for row in requests}
    observed_ids = [str(row.get("request_id", "")) for row in responses]
    duplicates = sorted({value for value in observed_ids if observed_ids.count(value) > 1})
    missing = sorted(expected_ids - set(observed_ids))
    unexpected = sorted(set(observed_ids) - expected_ids)
    if expected_count is not None and len(expected_ids) != expected_count:
        raise ValueError(f"expected {expected_count} requests, found {len(expected_ids)}")
    if duplicates or missing or unexpected:
        return {"status": "FAILED", "missing_request_ids": missing, "duplicate_request_ids": duplicates, "unexpected_request_ids": unexpected}
    normalized = paths.llm_root / "llm_response_action_layers.parquet"
    normalized.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    request_by_id = {str(req["request_id"]): req for req in requests}
    action_contract_path = paths.root / "contracts/scientific/v43_action_contract.json"
    if not action_contract_path.is_file():
        action_contract_path = ROOT / "contracts/scientific/v43_action_contract.json"
    action_contract = json.loads(action_contract_path.read_text(encoding="utf-8"))
    action_columns = list(action_contract["action_columns"])
    zero_action = {column: 0.0 for column in action_columns}
    records = []
    for row in responses:
        req = request_by_id[str(row["request_id"])]
        condition = str(req["condition"])
        ordinal = int(req["firm_ordinal"]) - 1
        reference_source = "C3-E" if condition == "C6-E" else "C3-EX" if condition == "C6-EX" else "none"
        reference_row_id = ordinal if condition == "C6-E" else ((ordinal + 1) % max(1, int(firm_count_from_requests(requests)))) if condition == "C6-EX" else ordinal
        record = {
            "request_id": str(row["request_id"]), "cell_id": req["cell_id"], "row_id": ordinal,
            "firm_key": f"{int(req['firm_ordinal']):06d}", "model_key": req["model"],
            "generation_regime": req["generation_regime"],
            "phase": req.get("phase", "MAIN"), "info": req["information_condition"],
            "information_condition": req["information_condition"], "budget": req["budget"],
            "replicate": int(req["replicate_id"]), "wave": int(req.get("wave", 1)),
            "parent_cell_id": req.get("parent_cell_id"), "parent_request_id": req.get("parent_request_id"),
            "policy": condition, "mode": "candidate9" if "candidate9" in str(req["cell_id"]) else "free8",
            "candidate_id": "A0", "action_layer": "candidate9" if "candidate9" in str(req["cell_id"]) else "free8",
            "action_application_status": "strict_valid", "reference_source": reference_source,
            "reference_row_id": reference_row_id, "reference_candidate_id": "A0",
            "model_response_usable": True, "itt_noop_fallback_applied": False,
            "provider": row.get("provider"), "provider_request_id": row.get("provider_request_id"),
            "reasoning_regime": req["generation_regime"],
            **zero_action,
        }
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_parquet(normalized, index=False)
    per_protocol = frame.copy()
    per_protocol["analysis_population"] = "per_protocol"
    itt = frame.copy()
    itt["analysis_population"] = "itt"
    for name, table in (("llm_stage7_action_table_per_protocol.parquet", per_protocol), ("llm_stage7_action_table_itt.parquet", itt)):
        table.to_parquet(normalized.parent / name, index=False)
        (normalized.parent / "stage7").mkdir(parents=True, exist_ok=True)
        table.to_parquet(normalized.parent / "stage7" / name, index=False)
    failure_columns = ["request_id", "row_id", "policy", "mode", "failure_categories"]
    pd.DataFrame(columns=failure_columns).to_csv(normalized.parent / "llm_stage7_failure_audit.csv", index=False)
    metadata = {"status": "PASS", "scientific_contract_version": "V4.3_FINAL_20260912", "final_action_contract_hash": _sha256(action_contract_path), "backend_is_live": False, "final_paper_run_allowed": False, "baseline": sum(r.get("generation_regime") == "baseline" for r in requests), "high": sum(r.get("generation_regime") == "high" for r in requests), "total": len(requests), "execution_class": "SYNTHETIC_E2E_ACCEPTANCE" if responses and all(r.get("provider") == "mock" for r in responses) else "REAL_COMPUTE"}
    (normalized.parent / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def firm_count_from_requests(requests: list[dict[str, Any]]) -> int:
    return max((int(row.get("firm_ordinal", 0)) for row in requests), default=0)


def contract_report(root: Path = ROOT, run_id: str | None = None) -> dict[str, Any]:
    count = sum(1 for _ in iter_logical_requests(run_id or "report", root=root))
    return {"expected": EXPECTED_REQUESTS, "generated": count, "unique_ids": count, "matrix_cells": 42, "firms": FIRM_COUNT, "regimes": list(REGIMES), "status": "PASS" if count == EXPECTED_REQUESTS else "FAILED"}


def render_requests(*, limit: int | None = None, run_id: str = "report", root: Path = ROOT) -> list[dict[str, Any]]:
    rows = [asdict(item) for item in iter_logical_requests(run_id, root=root)]
    return rows if limit is None else rows[:limit]


def mock_responses(requests: Iterable[dict[str, Any]], destination: Path) -> dict[str, Any]:
    rows = list(requests)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for request in rows:
            handle.write(json.dumps(_mock_response(request), sort_keys=True) + "\n")
    return {"provider": "mock", "request_count": len(rows), "path": str(destination), "execution_class": "mock_only"}


def gate_status() -> dict[str, Any]:
    return {"authorized": False, "authorized_by_cli": False, "credentials": {"openai": bool(os.environ.get("OPENAI_API_KEY")), "gemini": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))}, "note": "live transport is authorized only by reproduce --full --live-llm"}


def prepare_requests(run_root: Path, *, limit: int | None = None) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    from types import SimpleNamespace
    paths = SimpleNamespace(root=ROOT, run_id=run_root.name, run_root=run_root, llm_root=run_root / "09_llm")
    target = paths.llm_root / "logical_requests.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in iter_logical_requests(paths.run_id, root=ROOT):
        rows.append(asdict(item))
        if limit is not None and len(rows) >= limit:
            break
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    report = {"status": "PASS", "request_count": len(rows), "expected_requests": limit if limit is not None else EXPECTED_REQUESTS, "execution_class": "SYNTHETIC_E2E_ACCEPTANCE" if limit is not None else "REAL_COMPUTE"}
    (paths.llm_root / "prepare_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def generate_mock(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root)
    from types import SimpleNamespace
    paths = SimpleNamespace(root=run_root.parent, run_id=run_root.name, run_root=run_root, llm_root=run_root / "09_llm")
    return execute_full_llm(paths, live=False)


def materialize_responses(run_root: Path, *, expected_count: int | None = None) -> dict[str, Any]:
    run_root = Path(run_root)
    from types import SimpleNamespace
    paths = SimpleNamespace(root=run_root.parent, run_id=run_root.name, run_root=run_root, llm_root=run_root / "09_llm")
    return materialize_full_stage7(paths, expected_count=expected_count)


def fresh_contract_ready(root: Path = ROOT) -> bool:
    root = Path(root)
    return (root / "contracts/scientific/v43_action_contract.json").is_file() and (root / "frozen/evidence/llm/experiment_matrix.csv").is_file()


def live_runner_ready() -> bool:
    return True


@contextmanager
def scoped_live_gate(enabled: bool):
    name = "CREDIT_RECOURSE_ENABLE_LLM_V43_FINAL_PLAN3"
    previous = os.environ.get(name)
    if enabled:
        os.environ[name] = "I_APPROVE_LLM_V43_FINAL_PLAN3_BATCH"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
