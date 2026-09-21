from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from credit_recourse.final_release.common import ContractError
from credit_recourse.final_release.common import canonical_hash, sha256_file
from credit_recourse.final_release.contract import _semantic_design
from credit_recourse.final_release.llm_contract import build_c6ex_materialization, validate_llm_contract
from thesis_repro import cli
from thesis_repro.llm_runtime import execute_real_llm


def _c6ex_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    permutation = pd.DataFrame({"row_id": [0, 1, 2, 3], "donor_row_id": [1, 2, 3, 0]})
    actions = pd.DataFrame({"row_id": [0, 1, 2, 3], "action_id": ["A0", "A1", "A0", "A0"]})
    return permutation, actions


def test_c6ex_valid_derangement_counts_observed_collisions():
    materialized, diagnostics = build_c6ex_materialization(*_c6ex_frames(), expected_rows=4)
    assert diagnostics == {
        "firm_count": 4,
        "self_donor_collision_count": 0,
        "action_label_collision_count": 2,
        "distribution_preserved": True,
    }
    assert int(materialized["action_label_collision"].sum()) == 2


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.assign(donor_row_id=[0, 2, 3, 1]),
        lambda frame: frame.assign(donor_row_id=[1, 1, 3, 0]),
        lambda frame: frame.assign(donor_row_id=[1, 2, 3, 99]),
    ],
    ids=["self-donor", "duplicate-donor", "unknown-donor"],
)
def test_c6ex_derangement_rejects_invalid_donor_relation(mutator):
    permutation, actions = _c6ex_frames()
    with pytest.raises(ContractError):
        build_c6ex_materialization(mutator(permutation), actions, expected_rows=4)


def _make_run_local_plan3(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    config = tmp_path / "plan3"
    config.mkdir()
    source = repo / "frozen/evidence/llm"
    for name in ("final_design.json", "design_release.json", "experiment_matrix.csv", "feature_dictionary.csv", "information_contract.json", "model_contract.json", "prompt_contract.json", "retry_policy.json", "analysis_contract.json"):
        shutil.copy2(source / name, config / name)

    action_labels = ["A0", "A1", "A0"]
    actions = pd.DataFrame({"row_id": list(range(575)), "action_id": [action_labels[index % 3] for index in range(575)]})
    permutation = pd.DataFrame({"row_id": list(range(575)), "donor_row_id": [(index + 1) % 575 for index in range(575)]})
    materialized, diagnostics = build_c6ex_materialization(permutation, actions)
    permutation.to_parquet(config / "C6EX_permutation.parquet", index=False)
    materialized.to_parquet(config / "C6EX_materialized.parquet", index=False)

    c3e_root = tmp_path / "c3e"
    c3e_root.mkdir()
    (c3e_root / "C3E_definition.json").write_text("{}\n", encoding="utf-8")
    actions.to_parquet(c3e_root / "C3E_firm_actions.parquet", index=False)
    pd.DataFrame({"row_id": list(range(575)), "p": [1.0] * 575}).to_parquet(c3e_root / "C3E_firm_probabilities.parquet", index=False)
    c3e_hash = "fresh-c3e-test-hash"
    (c3e_root / "release.json").write_text(json.dumps({"release_hash": c3e_hash}) + "\n", encoding="utf-8")

    design = json.loads((config / "final_design.json").read_text(encoding="utf-8"))
    design["active_reference_policy"]["release_hash"] = c3e_hash
    design["design_release_hash"] = "fresh-design-test-hash"
    (config / "final_design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    release = json.loads((config / "design_release.json").read_text(encoding="utf-8"))
    release["c3e_release_hash"] = c3e_hash
    release["design_release_hash"] = design["design_release_hash"]
    release["final_design_semantic_hash"] = canonical_hash(_semantic_design(design))

    source_paths = {
        "experiment_matrix": config / "experiment_matrix.csv",
        "feature_dictionary": config / "feature_dictionary.csv",
        "information_contract": config / "information_contract.json",
        "model_contract": config / "model_contract.json",
        "prompt_contract": config / "prompt_contract.json",
        "retry_policy": config / "retry_policy.json",
        "analysis_contract": config / "analysis_contract.json",
        "action_contract": repo / "contracts/scientific/v43_action_contract.json",
        "c3e_definition": c3e_root / "C3E_definition.json",
        "c3e_probabilities": c3e_root / "C3E_firm_probabilities.parquet",
        "c3e_actions": c3e_root / "C3E_firm_actions.parquet",
        "c6ex_permutation": config / "C6EX_permutation.parquet",
        "c6ex_materialized": config / "C6EX_materialized.parquet",
        "c6ex_manifest": config / "C6EX_manifest.json",
    }
    runtime_names = {
        "runtime_contract": "src/credit_recourse/final_release/contract.py",
        "runtime_llm_contract": "src/credit_recourse/final_release/llm_contract.py",
        "runtime_parser": "src/credit_recourse/final_release/parsing.py",
        "runtime_prompting": "src/credit_recourse/final_release/prompting.py",
        "runtime_provider_adapters": "src/credit_recourse/final_release/providers.py",
        "runtime_retry_policy": "src/credit_recourse/final_release/retry_policy.py",
        "runtime_ledger": "src/credit_recourse/final_release/ledger.py",
        "runtime_executor": "src/credit_recourse/final_release/executor.py",
        "runtime_collection": "src/credit_recourse/final_release/collection.py",
        "runtime_doctor": "src/credit_recourse/final_release/doctor.py",
        "runtime_failure_coder": "src/credit_recourse/final_release/failure_coder.py",
        "runtime_action_validation": "src/credit_recourse/final_release/action_validation.py",
        "runtime_evaluation_bridge": "src/credit_recourse/final_release/evaluation_bridge.py",
        "runtime_semantic_fixture": "src/credit_recourse/final_release/semantic_fixture.py",
        "runtime_v43_production_bundle": "src/credit_recourse/simulator/v43_production_bundle.py",
        "stage8_evaluator": "src/credit_recourse/eval/final_stage8_llm_multi_oracle_eval/pipeline.py",
        "stage8_failure_enrichment": "src/credit_recourse/eval/final_stage8_llm_multi_oracle_eval/failure_enrichment.py",
        "stage9_evaluator": "src/credit_recourse/eval/final_stage9_llm_rl_comparison/pipeline.py",
        "stage9_revision_metrics": "src/credit_recourse/eval/final_stage9_llm_rl_comparison/revision_metrics.py",
    }
    source_paths.update({name: repo / path for name, path in runtime_names.items()})
    release["source_hashes"] = {name: sha256_file(path) for name, path in source_paths.items() if path.is_file()}
    (config / "design_release.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "v43_c6ex_fresh_materialization_v1",
        "status": "PASS",
        "c3e_release_hash": c3e_hash,
        "permutation_sha256": sha256_file(config / "C6EX_permutation.parquet"),
        "materialized_sha256": sha256_file(config / "C6EX_materialized.parquet"),
        "parent_historical_permutation_sha256": sha256_file(config / "C6EX_permutation.parquet"),
        "distribution_preserved": diagnostics["distribution_preserved"],
        "firm_count": diagnostics["firm_count"],
        "self_donor_collision_count": diagnostics["self_donor_collision_count"],
        "action_label_collision_count": diagnostics["action_label_collision_count"],
    }
    (config / "C6EX_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    release["source_hashes"]["c6ex_manifest"] = sha256_file(config / "C6EX_manifest.json")
    (config / "design_release.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    return config, c3e_root


def test_run_local_c6ex_manifest_loads_and_validates(monkeypatch, tmp_path):
    config, c3e_root = _make_run_local_plan3(tmp_path)
    monkeypatch.setenv("THESIS_REPRO_LLM_CONFIG_ROOT", str(config))
    monkeypatch.setenv("THESIS_REPRO_C3E_ROOT", str(c3e_root))
    result = validate_llm_contract(Path(__file__).resolve().parents[1])
    assert result["status"] == "PASS"
    assert result["c6ex_action_label_collisions"] == 191


@pytest.mark.parametrize("field", ["permutation_sha256", "materialized_sha256"])
def test_run_local_c6ex_hash_mismatch_rejected(monkeypatch, tmp_path, field):
    config, c3e_root = _make_run_local_plan3(tmp_path)
    manifest_path = config / "C6EX_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("THESIS_REPRO_LLM_CONFIG_ROOT", str(config))
    monkeypatch.setenv("THESIS_REPRO_C3E_ROOT", str(c3e_root))
    with pytest.raises(ContractError):
        validate_llm_contract(Path(__file__).resolve().parents[1])


def test_run_local_c6ex_action_distribution_mismatch_rejected(monkeypatch, tmp_path):
    config, c3e_root = _make_run_local_plan3(tmp_path)
    materialized_path = config / "C6EX_materialized.parquet"
    materialized = pd.read_parquet(materialized_path)
    materialized.loc[0, "own_C3E_action"] = "UNDECLARED_ACTION"
    materialized["action_label_collision"] = materialized["own_C3E_action"].eq(materialized["C6EX_reference_action"])
    materialized.to_parquet(materialized_path, index=False)
    manifest_path = config / "C6EX_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["materialized_sha256"] = sha256_file(materialized_path)
    manifest["action_label_collision_count"] = int(materialized["action_label_collision"].sum())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    release_path = config / "design_release.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["source_hashes"]["c6ex_materialized"] = sha256_file(materialized_path)
    release["source_hashes"]["c6ex_manifest"] = sha256_file(manifest_path)
    release_path.write_text(json.dumps(release), encoding="utf-8")
    monkeypatch.setenv("THESIS_REPRO_LLM_CONFIG_ROOT", str(config))
    monkeypatch.setenv("THESIS_REPRO_C3E_ROOT", str(c3e_root))
    with pytest.raises(ContractError):
        validate_llm_contract(Path(__file__).resolve().parents[1])


def _live_paths(tmp_path: Path):
    run_root = tmp_path / "runs" / "micro"
    llm_root = run_root / "09_llm"
    llm_root.mkdir(parents=True)
    (llm_root / "real_llm_release_manifest.json").write_text(
        json.dumps({"status": "PREPARED", "baseline_hash": "baseline", "high_hash": "high"}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        root=tmp_path,
        run_id="micro",
        run_root=run_root,
        llm_root=llm_root,
        c3e_root=run_root / "07_c3e",
    )


def _patch_live_control_plane(monkeypatch, tmp_path, polls, retries=None):
    paths = _live_paths(tmp_path)
    submissions = []
    retry_calls = []
    retry_results = list(retries or [])
    bound = set()

    def submit(root, release_hash, wave, *, approved, attempt_index=1, **kwargs):
        key = (release_hash, wave, attempt_index)
        if key not in bound:
            submissions.append(key)
            bound.add(key)
        return {"status": "SUBMITTED"}

    def poll(root, release_hash, wave, *, attempt_index=1):
        value = polls.pop(0) if polls else {"status": "COLLECTED", "all_terminal_or_downloaded": True}
        return value

    def retry(root, release_hash, wave, attempt_index):
        retry_calls.append((release_hash, wave, attempt_index))
        return retry_results.pop(0) if retry_results else {"status": "PREPARED"}

    monkeypatch.setattr("credit_recourse.final_release.live_batch.submit_wave", submit)
    monkeypatch.setattr("credit_recourse.final_release.live_batch.poll_and_download", poll)
    monkeypatch.setattr("credit_recourse.final_release.retry.prepare_retry", retry)
    monkeypatch.setattr("credit_recourse.final_release.executor.load_release", lambda root, release: (tmp_path, {"status": "FROZEN_READY_WAVE1"}))
    monkeypatch.setattr("credit_recourse.final_release.executor.generation_status", lambda root, release: {"status": "GENERATION_INCOMPLETE", "ledger_states": {}})
    monkeypatch.setattr("credit_recourse.final_release.executor.prepare_wave", lambda *args, **kwargs: {"status": "PREPARED"})
    monkeypatch.setattr("credit_recourse.high_reasoning.runner.prepare", lambda *args, **kwargs: {"status": "PREPARED"})
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("CREDIT_RECOURSE_ENABLE_LLM_V43_FINAL_PLAN3", "I_APPROVE_LLM_V43_FINAL_PLAN3_BATCH")
    return paths, submissions, retry_calls


def test_live_pending_job_returns_external_wait_without_retry(monkeypatch, tmp_path):
    paths, submissions, retry_calls = _patch_live_control_plane(
        monkeypatch, tmp_path, [{"all_terminal_or_downloaded": False}]
    )
    result = execute_real_llm(paths, resume=False)
    assert result["status"] == "EXTERNAL_WAIT"
    assert submissions == [("baseline", 1, 1)]
    assert retry_calls == []


def test_live_resume_polls_existing_job_without_duplicate_submission(monkeypatch, tmp_path):
    paths, submissions, retry_calls = _patch_live_control_plane(
        monkeypatch,
        tmp_path,
        [
            {"all_terminal_or_downloaded": False},
            {"status": "COLLECTED", "all_terminal_or_downloaded": True},
        ],
    )
    assert execute_real_llm(paths, resume=False)["status"] == "EXTERNAL_WAIT"
    assert execute_real_llm(paths, resume=True)["status"] == "PASS"
    assert submissions.count(("baseline", 1, 1)) == 1
    assert retry_calls == []


def test_live_terminal_retry_submits_only_prepared_retry(monkeypatch, tmp_path):
    paths, submissions, retry_calls = _patch_live_control_plane(
        monkeypatch,
        tmp_path,
        [
            {"status": "RETRY_REQUIRED", "all_terminal_or_downloaded": True},
            {"status": "COLLECTED", "all_terminal_or_downloaded": True},
        ],
        retries=[{"status": "PREPARED"}],
    )
    result = execute_real_llm(paths, resume=True)
    assert result["status"] == "PASS"
    assert retry_calls == [("baseline", 1, 2)]
    assert ("baseline", 1, 2) in submissions


def test_live_no_retry_needed_never_submits_attempt_two(monkeypatch, tmp_path):
    paths, submissions, retry_calls = _patch_live_control_plane(
        monkeypatch,
        tmp_path,
        [{"status": "RETRY_REQUIRED", "all_terminal_or_downloaded": True}],
        retries=[{"status": "NO_RETRY_NEEDED"}],
    )
    result = execute_real_llm(paths, resume=True)
    assert result["status"] == "FAILED"
    assert retry_calls == [("baseline", 1, 2)]
    assert all(attempt == 1 for _, _, attempt in submissions)


def test_live_success_does_not_prepare_retry_and_high_wave2_stays_high(monkeypatch, tmp_path):
    paths, submissions, retry_calls = _patch_live_control_plane(monkeypatch, tmp_path, [])
    result = execute_real_llm(paths, resume=True)
    assert result["status"] == "PASS"
    assert retry_calls == []
    assert ("baseline", 2, 1) in submissions
    assert ("high", 2, 1) in submissions
    assert submissions.index(("high", 2, 1)) > submissions.index(("baseline", 2, 1))
    assert all(release_hash == "high" for release_hash, wave, _ in submissions if release_hash == "high" and wave == 2)


@pytest.mark.parametrize("completion_state,accepted", [("PASS", True), ("PASS_WITH_QUALIFICATION", True), ("INPUT_REQUIRED", False), ("EXTERNAL_WAIT", False)])
def test_full_reproduction_cli_uses_completion_state(monkeypatch, completion_state, accepted):
    monkeypatch.setattr(cli, "execute", lambda *args, **kwargs: {"completion_state": completion_state})
    if accepted:
        cli.main(["reproduce", "--full", "--run-id", "micro"])
    else:
        with pytest.raises(SystemExit) as exc:
            cli.main(["reproduce", "--full", "--run-id", "micro"])
        assert exc.value.code == 1
