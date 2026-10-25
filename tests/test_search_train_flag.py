import sys
import types
from dataclasses import dataclass

import pytest

from thesis_repro import search


@dataclass
class _Config:
    run_id: str = ""
    epochs: int = 1
    seed: int = 0

    def validate(self):
        return self


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = []
    module = types.ModuleType("credit_recourse.rl.v43_stage5_search")
    module.SearchConfig = _Config
    module.prepare_run = lambda root, config, dry_run: calls.append((config.run_id, dry_run)) or {"run_id": config.run_id}
    monkeypatch.setitem(sys.modules, "credit_recourse.rl.v43_stage5_search", module)
    return calls


def _catalog(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text("run_id,epochs,seed\nB27,2,1\nDT06,3,2\n", encoding="utf-8")
    return path


def test_without_train_flag_nothing_is_trained(tmp_path, fake_runtime):
    result = search.reproduce_search(tmp_path, catalog_path=_catalog(tmp_path), source_run="fresh")
    assert result["status"] == "PLANNED"
    assert result["planned_runs"] == ["B27", "DT06"]
    assert result["training_runs"] == 0
    assert fake_runtime == []
    assert not (tmp_path / "runs").exists()


def test_train_flag_starts_training(tmp_path, fake_runtime, monkeypatch):
    monkeypatch.setattr(search, "build_search_baseline", lambda root, source_run: {"status": "BASELINE_WRITTEN"})
    result = search.reproduce_search(tmp_path, catalog_path=_catalog(tmp_path), source_run="fresh", execute=True)
    assert result["status"] == "TRAINING_COMPLETED"
    assert result["comparison_with_historical_selection"] == "NOT_CHECKED"
    assert fake_runtime == [("B27", False), ("DT06", False)]


def test_cli_passes_train_flag(monkeypatch):
    from thesis_repro import cli

    seen = {}
    monkeypatch.setattr(search, "reproduce_search", lambda root, **kw: seen.update(kw) or {"status": "PLANNED"})
    cli.main(["search", "--rl", "--source-run", "fresh"])
    assert seen["execute"] is False
