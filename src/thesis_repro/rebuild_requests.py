"""Offline rebuild of the frozen Baseline and High LLM provider requests.

``credit_recourse.final_release.contract.load_design`` checks every released
source hash against the checkout bytes and stops when one differs.  On this
kit that strict path cannot pass: some expected hashes were recorded from CRLF
bytes and some from LF bytes, several runtime sources were edited after the
release (see ``provenance/SCIENTIFIC_CODE_CHANGES.csv``), and the release used
the frozen action contract rather than ``contracts/scientific/v43_action_contract.json``.

This command keeps that strict path untouched and instead:

1. reports each released source hash against the checkout bytes, the
   LF-normalized bytes and the CRLF-normalized bytes (content drift is a
   warning, not a failure);
2. builds the design bundle from ``frozen/evidence/llm`` with the frozen
   action contract (SHA-256 ``f722e189...``);
3. takes the C6-EX reference table, the C3-E release files and the Stage2
   evaluation panel from the certified distribution ZIP in ``frozen/`` into
   the run folder (exact bytes, hash checked);
4. rebuilds every provider batch line with the release code and compares it
   byte for byte with the stored shard line.

The verdict is the byte comparison only.  No provider is contacted and no
network library is imported.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

FROZEN_ACTION_CONTRACT = "frozen/original_release/rl/stage2/candidate_action_contract_v4_3.json"
FROZEN_ACTION_CONTRACT_SHA256 = "f722e189a265e111c018341254a648ddb9cecc05aa8c12c3547a023ea4c458c0"
LLM_CONFIG = "frozen/evidence/llm"
FINAL_PLAN3 = "frozen/original_release/llm/final_plan3"
BASELINE_RELEASE = "2cf6d6d0e4250e66ce882ab95f9d641f2c73711ffbc6429e9a203dcc2ee680a2"
HIGH_RELEASE = "d5d878bf50718bac65ede73df3fc88c16723870b6c90a790557ff35b063b7ba7"
ZIP_PREFIX = "THESIS_REPRO_KIT_v2.1.1_FINAL/"
C3E_MEMBER_DIR = ZIP_PREFIX + "archive/DEPLOYED_RELEASE/stage5_candidate_iql/C3E_E2_7SEED_BALANCED_DFEBAFA6/"
ZIP_MEMBERS = {
    "c6ex_permutation": ZIP_PREFIX + "configs/current/llm/C6EX_permutation.parquet",
    "c6ex_materialized": ZIP_PREFIX + "configs/current/llm/C6EX_materialized.parquet",
    "c6ex_manifest": ZIP_PREFIX + "configs/current/llm/C6EX_manifest.json",
    "c3e_definition": C3E_MEMBER_DIR + "C3E_definition.json",
    "c3e_probabilities": C3E_MEMBER_DIR + "C3E_firm_probabilities.parquet",
    "c3e_actions": C3E_MEMBER_DIR + "C3E_firm_actions.parquet",
    "phase_eval_candidate": ZIP_PREFIX + "archive/DEPLOYED_RELEASE/stage2_candidate_projection/phase_eval_candidate.parquet",
}
CONFIG_SOURCES = {
    "experiment_matrix": "experiment_matrix.csv",
    "model_contract": "model_contract.json",
    "prompt_contract": "prompt_contract.json",
    "feature_dictionary": "feature_dictionary.csv",
    "information_contract": "information_contract.json",
    "retry_policy": "retry_policy.json",
    "analysis_contract": "analysis_contract.json",
}
RUNTIME_SOURCES = {
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
CODE_CHANGE_RECORD = "provenance/SCIENTIFIC_CODE_CHANGES.csv"
LFS_POINTER_HEAD = b"version https://git-lfs.github.com/spec/v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _line_ending_hashes(data: bytes) -> dict[str, str]:
    lf = data.replace(b"\r\n", b"\n")
    return {"checkout": _sha(data), "lf": _sha(lf), "crlf": _sha(lf.replace(b"\n", b"\r\n"))}


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(len(LFS_POINTER_HEAD)) == LFS_POINTER_HEAD


def _extract_zip_inputs(root: Path, target: Path) -> dict[str, Any]:
    from .llm_runtime import CERTIFIED_DISTRIBUTION_RELATIVE, CERTIFIED_DISTRIBUTION_SHA256
    distribution = root / CERTIFIED_DISTRIBUTION_RELATIVE
    if not distribution.is_file():
        return {"status": "INPUT_REQUIRED", "reason": "certified distribution ZIP missing", "path": str(CERTIFIED_DISTRIBUTION_RELATIVE)}
    if _is_lfs_pointer(distribution):
        return {"status": "INPUT_REQUIRED", "reason": "certified distribution ZIP is a Git LFS pointer; run: git lfs pull", "path": str(CERTIFIED_DISTRIBUTION_RELATIVE)}
    digest = hashlib.sha256()
    with distribution.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    if digest.hexdigest() != CERTIFIED_DISTRIBUTION_SHA256:
        return {"status": "FAILED", "reason": "CERTIFIED_DISTRIBUTION_HASH_MISMATCH", "actual_sha256": digest.hexdigest()}
    target.mkdir(parents=True, exist_ok=True)
    extracted = {}
    with zipfile.ZipFile(distribution) as archive:
        for name, member in ZIP_MEMBERS.items():
            data = archive.read(member)
            path = target / Path(member).name
            path.write_bytes(data)
            extracted[name] = {"member": member, "path": path, "sha256": _sha(data)}
    return {"status": "PASS", "distribution_sha256": digest.hexdigest(), "files": extracted}


def _source_report(root: Path, release: dict[str, Any], design: dict[str, Any], extracted: dict[str, Any]) -> list[dict[str, Any]]:
    paths: dict[str, Path | None] = {name: root / LLM_CONFIG / rel for name, rel in CONFIG_SOURCES.items()}
    paths.update({name: root / rel for name, rel in RUNTIME_SOURCES.items()})
    paths["action_contract"] = root / FROZEN_ACTION_CONTRACT
    authority = Path(str(design.get("authority_document", "")))
    paths["authority_document"] = authority if authority.is_absolute() else root / authority
    for name in ("c6ex_permutation", "c6ex_materialized", "c6ex_manifest", "c3e_definition", "c3e_probabilities", "c3e_actions"):
        paths[name] = extracted[name]["path"] if name in extracted else None
    rows = []
    for name, expected in release["source_hashes"].items():
        path = paths.get(name)
        row: dict[str, Any] = {"name": name, "expected_sha256": expected, "path": str(path.relative_to(root)).replace("\\", "/") if path and path.is_relative_to(root) else str(path)}
        if path is None or not path.is_file():
            row.update(status="MISSING", severity="WARN")
        elif _is_lfs_pointer(path):
            row.update(status="LFS_POINTER", severity="WARN", note="run: git lfs pull")
        else:
            hashes = _line_ending_hashes(path.read_bytes())
            match = next((kind for kind in ("checkout", "lf", "crlf") if hashes[kind] == expected), None)
            if match:
                row.update(status="MATCH", matched_bytes=match, severity="OK")
            else:
                row.update(status="CONTENT_DIFFERS", severity="WARN", note=f"content changed after the release; see {CODE_CHANGE_RECORD}" if name.startswith(("runtime_", "stage8_", "stage9_")) else "content differs from the released bytes")
            row["checkout_sha256"] = hashes["checkout"]
        if name in extracted:
            row["source"] = f"certified distribution ZIP member {extracted[name]['member']}"
        rows.append(row)
    return rows


def _frames_equal(left, right) -> bool:
    """Same columns, same row order and the same values (missing values compared as missing)."""
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in left.columns:
        a = left[column].reset_index(drop=True)
        b = right[column].reset_index(drop=True)
        a = [None if value is None or value != value else str(value) for value in a.astype(object)]
        b = [None if value is None or value != value else str(value) for value in b.astype(object)]
        if a != b:
            return False
    return True


def _stored_lines(path: Path) -> list[bytes]:
    return [line for line in path.read_bytes().splitlines(keepends=True) if line.strip()]


def _parent_texts(ledger: Path) -> dict[str, str]:
    connection = sqlite3.connect(str(ledger))
    try:
        rows = connection.execute(
            "SELECT l.request_id, a.raw_visible_text FROM logical_requests l "
            "JOIN attempts a ON a.request_id = l.request_id AND a.attempt_index = l.selected_attempt_index"
        ).fetchall()
    finally:
        connection.close()
    out = {}
    for request_id, text in rows:
        if text is not None:
            out[str(request_id)] = text.decode("utf-8") if isinstance(text, bytes) else str(text)
    return out


def _prepared_shard_hashes(directory: Path) -> dict[str, str]:
    hashes = {}
    for manifest in sorted((directory / "manifests").glob("wave*_prepared.json")):
        for batch in json.loads(manifest.read_text(encoding="utf-8")).get("provider_batches", []):
            hashes[str(batch["path"]).replace("\\", "/")] = str(batch["sha256"])
    return hashes


def rebuild_requests(root: Path, run_id: str = "rebuild-requests", *, verify: bool = True) -> dict[str, Any]:
    import pandas as pd

    from credit_recourse.final_release import contract as design_contract
    from credit_recourse.final_release.common import canonical_bytes, canonical_hash, load_json
    from credit_recourse.final_release.executor import _inventory, _render
    from credit_recourse.final_release.providers import build_batch_record
    from credit_recourse.high_reasoning.runner import HIGH_CAP, _record as high_record

    root = Path(root).resolve()
    out = root / "runs" / run_id / "rebuild_requests"
    out.mkdir(parents=True, exist_ok=True)
    action_path = root / FROZEN_ACTION_CONTRACT
    if not action_path.is_file() or _is_lfs_pointer(action_path):
        return {"status": "INPUT_REQUIRED", "reason": "frozen action contract missing or a Git LFS pointer; run: git lfs pull", "path": FROZEN_ACTION_CONTRACT}
    action_sha = _sha(action_path.read_bytes())
    if action_sha != FROZEN_ACTION_CONTRACT_SHA256:
        return {"status": "FAILED", "reason": "frozen action contract hash differs", "actual_sha256": action_sha}
    zipped = _extract_zip_inputs(root, out / "config")
    if zipped["status"] != "PASS":
        return zipped
    extracted = zipped["files"]
    base = root / LLM_CONFIG
    design_json = load_json(base / "final_design.json")
    release = load_json(base / "design_release.json")
    sources = _source_report(root, release, design_json, extracted)
    design = design_contract.DesignBundle(
        root=root, design=design_json, release=release, matrix=design_contract._csv(base / "experiment_matrix.csv"),
        models=load_json(base / "model_contract.json"), prompts=load_json(base / "prompt_contract.json"),
        information=load_json(base / "information_contract.json"), retry=load_json(base / "retry_policy.json"),
        action_contract=load_json(action_path), design_release_hash=release["design_release_hash"])

    runs = {"BASELINE": root / FINAL_PLAN3 / BASELINE_RELEASE, "HIGH": root / FINAL_PLAN3 / HIGH_RELEASE}
    for directory in runs.values():
        for name in ("logical_requests.parquet", "firm_payload_source.parquet", "generation_ledger.sqlite"):
            path = directory / name
            if not path.is_file() or _is_lfs_pointer(path):
                return {"status": "INPUT_REQUIRED", "reason": "frozen LLM run file missing or a Git LFS pointer; run: git lfs pull", "path": str(path.relative_to(root)).replace("\\", "/")}

    checks: dict[str, Any] = {}
    cohort = pd.read_parquet(runs["BASELINE"] / "firm_payload_source.parquet")
    stored_base = pd.read_parquet(runs["BASELINE"] / "logical_requests.parquet")
    rebuilt_base = _inventory(design, cohort)
    checks["baseline_inventory_identical"] = _frames_equal(rebuilt_base, stored_base)
    identity = {"namespace": "LLM_HIGH_REASONING_FULL_REPLICATION_V1", "baseline_release_hash": BASELINE_RELEASE, "reasoning_arm": "HIGH",
                "openai_effort": "high", "gemini_thinking_level": "high", "max_output_tokens": HIGH_CAP,
                "matrix_hash": _sha((runs["BASELINE"] / "logical_requests.parquet").read_bytes())}
    checks["high_release_hash_rebuilt"] = canonical_hash(identity) == HIGH_RELEASE

    previous = os.environ.get("THESIS_REPRO_LLM_CONFIG_ROOT")
    os.environ["THESIS_REPRO_LLM_CONFIG_ROOT"] = str(out / "config")
    shards = []
    mismatches: list[dict[str, Any]] = []
    total = matched = 0
    try:
        for label, directory in runs.items():
            inventory = pd.read_parquet(directory / "logical_requests.parquet").set_index("request_id")
            firms = pd.read_parquet(directory / "firm_payload_source.parquet").set_index("firm_key")
            parents = _parent_texts(directory / "generation_ledger.sqlite")
            prepared = _prepared_shard_hashes(directory)
            models = load_json(directory / "release.json")["runtime_binding"]["models"] if label == "HIGH" else None
            files = sorted(p for p in (directory / "provider_batches").rglob("*.jsonl") if not p.name.endswith("_output.jsonl"))
            for path in files:
                rebuilt_bytes = []
                stored = _stored_lines(path)
                for line in stored:
                    record = json.loads(line)
                    rid = str(record.get("custom_id") or record.get("key"))
                    item = inventory.loc[rid]
                    parent = parents.get(str(item["parent_request_id"])) if int(item["wave"]) == 2 else None
                    rendered = _render(design, firms.loc[item["firm_key"]].to_dict(), {**item.to_dict(), "request_id": rid}, parent)
                    if label == "BASELINE":
                        model = design.models["models"][str(item["model_key"])]
                        rebuilt = build_batch_record(provider=model["provider"], exact_model_id=model["exact_model_id"], request_id=rid, system_text=rendered.system, user_text=rendered.user, max_output_tokens=model["max_output_tokens"])
                    else:
                        rebuilt = high_record(models[str(item["model_key"])], rid, rendered.system, rendered.user)
                    payload = canonical_bytes(rebuilt) + b"\n"
                    rebuilt_bytes.append(payload)
                    total += 1
                    if payload == line:
                        matched += 1
                    elif len(mismatches) < 20:
                        mismatches.append({"run": label, "request_id": rid, "shard": str(path.relative_to(root)).replace("\\", "/")})
                relative = str(path.relative_to(directory)).replace("\\", "/")
                rebuilt_sha = _sha(b"".join(rebuilt_bytes))
                shards.append({"run": label, "shard": relative, "lines": len(stored), "stored_sha256": _sha(path.read_bytes()),
                               "rebuilt_sha256": rebuilt_sha, "prepared_manifest_sha256": prepared.get(relative),
                               "rebuilt_equals_stored": rebuilt_sha == _sha(path.read_bytes()),
                               "rebuilt_equals_prepared_manifest": prepared.get(relative) == rebuilt_sha if relative in prepared else None})
    finally:
        if previous is None:
            os.environ.pop("THESIS_REPRO_LLM_CONFIG_ROOT", None)
        else:
            os.environ["THESIS_REPRO_LLM_CONFIG_ROOT"] = previous

    manifest_checked = [s for s in shards if s["rebuilt_equals_prepared_manifest"] is not None]
    report = {
        "schema_version": "v43_offline_request_rebuild_v1",
        "status": "PASS" if total and matched == total and all(s["rebuilt_equals_stored"] for s in shards) else "FAILED",
        "verdict_basis": "byte equality of every rebuilt provider batch line with the stored shard line",
        "provider_contacted": False,
        "requests_rebuilt": total,
        "requests_byte_identical": matched,
        "shards": len(shards),
        "shards_byte_identical": sum(1 for s in shards if s["rebuilt_equals_stored"]),
        "shards_checked_against_prepared_manifest": len(manifest_checked),
        "shards_equal_prepared_manifest": sum(1 for s in manifest_checked if s["rebuilt_equals_prepared_manifest"]),
        "action_contract": {"path": FROZEN_ACTION_CONTRACT, "sha256": action_sha},
        "inputs_from_certified_distribution": {name: {"member": v["member"], "sha256": v["sha256"]} for name, v in extracted.items()},
        "inventory_checks": checks,
        "source_hashes": {
            "matched": sum(1 for r in sources if r["status"] == "MATCH"),
            "matched_by_bytes": {kind: sum(1 for r in sources if r.get("matched_bytes") == kind) for kind in ("checkout", "lf", "crlf")},
            "warnings": [r for r in sources if r["severity"] == "WARN"],
        },
        "first_mismatches": mismatches,
        "note": "The strict load_design path is unchanged. Source-hash warnings do not decide the verdict.",
    }
    (out / "source_hash_report.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    pd.DataFrame(shards).to_csv(out / "shard_comparison.csv", index=False, lineterminator="\n")
    (out / "REBUILD_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    report["outputs"] = str(out.relative_to(root)).replace("\\", "/")
    return report
