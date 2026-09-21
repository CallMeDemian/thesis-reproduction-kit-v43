from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from credit_recourse.final_release.common import sha256_file
from credit_recourse.final_release.executor import freeze_release, prepare_wave
from credit_recourse.final_release.llm_contract import industry_binding_status
from thesis_repro.llm_runtime import prepare_real_plan3_config


def _write_binding_fixture(
    root: Path,
    *,
    role: str,
    frozen: bool = False,
    rows: int = 575,
    duplicate_firm: bool = False,
    blank_code: bool = False,
    hash_value: str | None = None,
) -> tuple[Path, Path]:
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "firm_key": [f"{index:06d}::2024" for index in range(rows)],
            "induty_code": ["C10" for _ in range(rows)],
            "industry_display_value": ["Manufacturing" for _ in range(rows)],
        }
    )
    if duplicate_firm and len(frame) > 1:
        frame.loc[1, "firm_key"] = frame.loc[0, "firm_key"]
    if blank_code and len(frame) > 0:
        frame.loc[0, "induty_code"] = ""
    artifact = config / "FY2024_INDUSTRY_BINDING.parquet"
    frame.to_parquet(artifact, index=False)
    manifest = {
        "schema_version": "fresh_industry_binding_v1",
        "status": "PASS",
        "role": role,
        "frozen": frozen,
        "binding_artifact": artifact.name,
        "binding_artifact_sha256": hash_value or sha256_file(artifact),
        "rows": rows,
        "unique_firm_key": int(frame["firm_key"].nunique()),
        "unresolved_count": 0,
    }
    (config / "FY2024_INDUSTRY_BINDING_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (config / "information_contract.json").write_text(
        json.dumps(
            {
                "industry_binding_manifest": "FY2024_INDUSTRY_BINDING_MANIFEST.json",
                "valid_industry_nonmissing": 575,
                "valid_industry_missing": 0,
                "valid_industry_coverage_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return config, artifact


def test_fresh_industry_binding_is_ready_with_active_config_root(monkeypatch, tmp_path):
    config, _ = _write_binding_fixture(tmp_path, role="FRESH_EXOGENOUS_INFORMATION_INPUT")
    monkeypatch.setenv("THESIS_REPRO_LLM_CONFIG_ROOT", str(config))
    result = industry_binding_status(tmp_path)
    assert result["ready"] is True
    assert result["fresh_authorized"] is True
    assert result["api_key_leak_scan"] == "NOT_APPLICABLE_FRESH_RUNTIME"


def test_fresh_role_is_not_historical_without_active_config_root(monkeypatch, tmp_path):
    config, _ = _write_binding_fixture(tmp_path, role="FRESH_EXOGENOUS_INFORMATION_INPUT")
    frozen_root = tmp_path / "frozen" / "evidence" / "llm"
    frozen_root.mkdir(parents=True)
    for item in config.iterdir():
        item.replace(frozen_root / item.name)
    monkeypatch.delenv("THESIS_REPRO_LLM_CONFIG_ROOT", raising=False)
    result = industry_binding_status(tmp_path)
    assert result["ready"] is False
    assert result["fresh_authorized"] is False


def test_unknown_industry_role_is_rejected(monkeypatch, tmp_path):
    config, _ = _write_binding_fixture(tmp_path, role="COMPUTED_FINANCIAL_OUTPUT")
    monkeypatch.setenv("THESIS_REPRO_LLM_CONFIG_ROOT", str(config))
    assert industry_binding_status(tmp_path)["ready"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hash_value": "tampered"},
        {"rows": 574},
        {"duplicate_firm": True},
        {"blank_code": True},
    ],
    ids=["hash-mismatch", "574-rows", "duplicate-firm", "blank-code"],
)
def test_industry_binding_integrity_gate_rejects_invalid_fixtures(monkeypatch, tmp_path, kwargs):
    config, _ = _write_binding_fixture(tmp_path, role="FRESH_EXOGENOUS_INFORMATION_INPUT", **kwargs)
    monkeypatch.setenv("THESIS_REPRO_LLM_CONFIG_ROOT", str(config))
    assert industry_binding_status(tmp_path)["ready"] is False


def test_historical_frozen_industry_binding_remains_accepted(monkeypatch, tmp_path):
    config, _ = _write_binding_fixture(
        tmp_path,
        role="FROZEN_EXOGENOUS_INFORMATION_INPUT",
        frozen=True,
    )
    frozen_root = tmp_path / "frozen" / "evidence" / "llm"
    frozen_root.mkdir(parents=True)
    for item in config.iterdir():
        item.replace(frozen_root / item.name)
    leak = tmp_path / "repro" / "manifests" / "runtime_gates"
    leak.mkdir(parents=True)
    (leak / "api_key_leak_scan_full.json").write_text(
        json.dumps({"status": "PASS", "key_literal_hits": 0}), encoding="utf-8"
    )
    monkeypatch.delenv("THESIS_REPRO_LLM_CONFIG_ROOT", raising=False)
    result = industry_binding_status(tmp_path)
    assert result["ready"] is True
    assert result["historical_authorized"] is True


def test_real_plan3_prepare_freeze_and_wave1_accept_fresh_binding(monkeypatch, tmp_path):
    """Exercise the production preparation chain with preserved contract inputs."""
    source_repo = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    for relative in ("src", "contracts/scientific", "frozen/evidence/llm"):
        shutil.copytree(source_repo / relative, repo / relative)

    distribution = source_repo / "frozen" / "distribution" / "THESIS_REPRO_KIT_v2.1.1_FINAL.zip"
    assert distribution.is_file(), "the preserved exact permutation fixture is required"
    run_root = repo / "runs" / "integration"
    design_root = repo / "data" / "design"
    design_root.mkdir(parents=True)
    stage2_root = run_root / "03_stage2"
    (stage2_root / "input_source" / "input_splits").mkdir(parents=True)
    c3e_root = run_root / "07_c3e"
    c3e_root.mkdir(parents=True)
    llm_root = run_root / "09_llm"

    with zipfile.ZipFile(distribution) as archive:
        members = archive.namelist()
        member_map = {
            "permutation": next(name for name in members if name.endswith("configs/current/llm/C6EX_permutation.parquet")),
            "panel": next(name for name in members if name.endswith("stage2_candidate_projection/phase_eval_candidate.parquet")),
            "ids": next(name for name in members if name.endswith("stage2_candidate_projection/input_splits/canonical_evaluation_row_ids.parquet")),
        }
        (design_root / "C6EX_permutation.parquet").write_bytes(archive.read(member_map["permutation"]))
        panel_path = stage2_root / "input_source" / "phase_eval_candidate.parquet"
        panel_path.write_bytes(archive.read(member_map["panel"]))
        panel = pd.read_parquet(panel_path)
        panel["industry_class"] = [f"C{10 + (index % 5)}" for index in range(len(panel))]
        panel.to_parquet(panel_path, index=False)
        (stage2_root / "input_source" / "input_splits" / "canonical_evaluation_row_ids.parquet").write_bytes(archive.read(member_map["ids"]))

    action_contract = repo / "contracts" / "scientific" / "v43_action_contract.json"
    shutil.copy2(action_contract, stage2_root / "input_source" / "candidate_action_contract_v4_3.json")
    action_ids = ["A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2"]
    actions = pd.DataFrame({"row_id": range(575), "action_id": [action_ids[index % len(action_ids)] for index in range(575)]})
    actions.to_parquet(c3e_root / "C3E_firm_actions.parquet", index=False)
    pd.DataFrame({"row_id": range(575), "p": [1.0] * 575}).to_parquet(c3e_root / "C3E_firm_probabilities.parquet", index=False)
    (c3e_root / "C3E_definition.json").write_text("{}\n", encoding="utf-8")
    (c3e_root / "release.json").write_text(json.dumps({"release_hash": "fresh-c3e-integration"}), encoding="utf-8")

    paths = SimpleNamespace(root=repo, run_root=run_root, stage2_root=stage2_root, llm_root=llm_root, c3e_root=c3e_root)
    monkeypatch.setenv("THESIS_REPRO_RUN_ROOT", str(run_root))
    monkeypatch.setenv("THESIS_REPRO_C3E_ROOT", str(c3e_root))
    prepared = prepare_real_plan3_config(paths)
    assert prepared["status"] == "PASS", prepared
    assert prepared["industry_binding"]["role"] == "FRESH_EXOGENOUS_INFORMATION_INPUT"
    config_root = llm_root / "config"
    monkeypatch.setenv("THESIS_REPRO_LLM_CONFIG_ROOT", str(config_root))
    assert industry_binding_status(repo)["ready"] is True

    monkeypatch.setattr(
        "credit_recourse.final_release.semantic_fixture.run_semantic_fixture",
        lambda root: {"status": "PASS", "schema_version": "synthetic-test"},
    )
    monkeypatch.setenv("THESIS_REPRO_LLM_OUTPUT_ROOT", str(tmp_path / "llm-output"))
    _, release = freeze_release(repo)
    assert release["status"] == "FROZEN_READY_WAVE1"
    wave1 = prepare_wave(repo, release["protocol_release_hash"], 1)
    assert wave1["status"] == "PREPARED"
    assert wave1["api_calls_executed"] == 0
