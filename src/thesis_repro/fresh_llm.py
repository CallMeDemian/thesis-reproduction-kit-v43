"""Run-namespaced LLM preparation, mock execution, and reconciliation.

The mock transport is an architectural fixture only.  Its receipts explicitly
say that no provider was contacted, so it cannot satisfy fresh scientific
certification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import EXPECTED_REQUESTS, iter_logical_requests
from .live_llm import mock_responses, render_provider_batch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_requests(run_root: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Materialize the complete logical request set under one run namespace."""
    run_root = Path(run_root)
    target = run_root / "09_llm" / "logical_requests.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    ids: set[str] = set()
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for request in iter_logical_requests(run_root.name):
            if limit is not None and count >= limit:
                break
            if request.request_id in ids:
                raise ValueError(f"duplicate fresh request id: {request.request_id}")
            if not request.request_id.startswith(f"fresh:{run_root.name}:"):
                raise ValueError("logical request is not namespaced to the active run")
            if request.c3e_reference != "fresh.<run_id>.C3E":
                raise ValueError("request is not bound to same-run C3-E")
            ids.add(request.request_id)
            handle.write(json.dumps(request.__dict__, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    expected = EXPECTED_REQUESTS if limit is None else limit
    report = {
        "schema_version": "fresh_llm_prepare_v1",
        "status": "PASS" if count == expected else "FAILED",
        "execution_class": "REAL_COMPUTE" if limit is None else "SYNTHETIC_E2E_ACCEPTANCE",
        "expected_requests": expected,
        "request_count": count,
        "unique_request_ids": len(ids),
        "request_namespace": f"fresh:{run_root.name}",
        "historical_request_ids_reused": False,
        "c3e_parent": f"runs/{run_root.name}/08_c3e/release.json",
        "logical_requests_sha256": _sha256(target),
    }
    (target.parent / "prepare_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _read_requests(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["request_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate logical request ids")
    return rows


def generate_mock(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root)
    requests_path = run_root / "09_llm" / "logical_requests.jsonl"
    requests = _read_requests(requests_path)
    destination = run_root / "09_llm" / "raw_mock_responses.jsonl"
    receipt = mock_responses(requests, destination)
    receipt.update({
        "schema_version": "fresh_llm_mock_execution_receipt_v1",
        "status": "MOCK_EXECUTED",
        "provider_contacted": False,
        "logical_requests_sha256": _sha256(requests_path),
        "provider_request_identity": "mock:<request_id>",
    })
    (destination.parent / "generate_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def generate_provider_batch(run_root: Path, provider: str, model: str) -> dict[str, Any]:
    """Prepare provider-native input; no network submission occurs here."""
    run_root = Path(run_root)
    requests = _read_requests(run_root / "09_llm" / "logical_requests.jsonl")
    destination = run_root / "09_llm" / f"{provider}_batch.jsonl"
    report = render_provider_batch(requests, type("Provider", (), {"name": provider, "model": model, "transport": "batch"})(), destination)
    report.update({"status": "SUBMISSION_READY", "provider_contacted": False, "provider_receipt_required": True})
    (destination.parent / f"{provider}_submission_receipt.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def materialize_responses(run_root: Path, *, expected_count: int | None = None) -> dict[str, Any]:
    """Reconcile response identities and write normalized derived responses."""
    run_root = Path(run_root)
    request_path = run_root / "09_llm" / "logical_requests.jsonl"
    response_path = run_root / "09_llm" / "raw_mock_responses.jsonl"
    requests = _read_requests(request_path)
    expected_ids = {str(row["request_id"]) for row in requests}
    if expected_count is not None and len(expected_ids) != expected_count:
        raise ValueError(f"expected request count {expected_count}, found {len(expected_ids)}")
    if not response_path.is_file():
        return {"status": "FAILED", "missing_response_artifact": str(response_path)}
    responses = [json.loads(line) for line in response_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observed_ids = [str(row.get("request_id", "")) for row in responses]
    duplicates = sorted({value for value in observed_ids if observed_ids.count(value) > 1})
    missing = sorted(expected_ids - set(observed_ids))
    unexpected = sorted(set(observed_ids) - expected_ids)
    if duplicates or missing or unexpected:
        return {"status": "FAILED", "missing_request_ids": missing, "duplicate_request_ids": duplicates, "unexpected_request_ids": unexpected, "response_count": len(responses)}
    normalized_path = run_root / "09_llm" / "normalized_responses.jsonl"
    with normalized_path.open("w", encoding="utf-8") as handle:
        for response in responses:
            handle.write(json.dumps({
                "request_id": response["request_id"],
                "provider": response.get("provider"),
                "transport": response.get("transport"),
                "provider_request_id": response.get("provider_request_id", f"mock:{response['request_id']}"),
                "raw_response_sha256": hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest(),
                "selected_candidate": response.get("raw_response", {}).get("selected_candidate"),
            }, sort_keys=True) + "\n")
    report = {
        "schema_version": "fresh_llm_materialization_v1",
        "status": "PASS",
        "execution_class": "MOCK_ONLY" if all(row.get("provider") == "mock" for row in responses) else "REAL_COMPUTE",
        "expected_request_count": len(expected_ids),
        "response_count": len(responses),
        "missing_request_ids": [],
        "duplicate_request_ids": [],
        "unexpected_request_ids": [],
        "raw_responses_sha256": _sha256(response_path),
        "normalized_responses_sha256": _sha256(normalized_path),
    }
    (run_root / "09_llm" / "materialization_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
