from __future__ import annotations

from thesis_repro.contracts import render_requests
from thesis_repro.live_llm import gate_status, mock_responses


def test_fresh_request_ids_are_namespaced_and_parented():
    rows = render_requests(limit=20, run_id="negative-test")
    assert rows
    assert all(row["request_id"].startswith("fresh:negative-test:") for row in rows)
    assert all((row["parent_request_id"] or "").startswith("fresh:negative-test:") or row["parent_request_id"] is None for row in rows)
    assert all("fresh.<run_id>" in row["c3e_reference"] for row in rows)


def test_live_gate_defaults_closed():
    assert gate_status()["authorized"] is False


def test_mock_transport_preserves_cardinality(tmp_path):
    rows = render_requests(limit=3, run_id="mock-test")
    report = mock_responses(rows, tmp_path / "responses.jsonl")
    assert report["request_count"] == 3
    assert len((tmp_path / "responses.jsonl").read_text(encoding="utf-8").splitlines()) == 3
