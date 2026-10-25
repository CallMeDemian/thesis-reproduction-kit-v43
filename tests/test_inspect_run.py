import json

import pytest

from thesis_repro import cli


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_inspect_run_prefers_run_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    _write(tmp_path / "runs/synthetic_check/run_manifest.json", {"completion_state": "PASS"})
    _write(tmp_path / "runs/synthetic_check/final/REPRODUCTION_RECEIPT.json", {"status": "IGNORED"})
    assert cli._inspect("run", "synthetic_check") == {"completion_state": "PASS"}


def test_inspect_run_reads_reproduction_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    _write(tmp_path / "runs/published_check/final/REPRODUCTION_RECEIPT.json", {"status": "PASS"})
    result = cli._inspect("run", "published_check")
    assert result["receipt"] == {"status": "PASS"}
    assert result["source"] == "runs/published_check/final/REPRODUCTION_RECEIPT.json"


def test_inspect_run_without_outputs_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        cli._inspect("run", "missing")
