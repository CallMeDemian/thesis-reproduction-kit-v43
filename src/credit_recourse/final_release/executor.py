from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .common import ContractError, canonical_hash, find_repo_root, relative_to_root, utc_now, write_json, write_jsonl
from .contract import MODEL_KEYS, DesignBundle, candidate_vector, load_design, load_firm_cohort
from .ledger import connect, insert_logical_requests, materialize_payload, register_shard, set_dependency_empty, status_counts
from .prompting import render_policy_prompt
from .providers import build_batch_record, provider_request_hash


OUTPUT_ROOT = Path("archive/DEPLOYED_RELEASE/llm_runs/final_plan3")


def request_id(release_hash: str, cell: Mapping[str, Any], firm_key: str, parent_request_id: str | None) -> str:
    identity = {
        "namespace": "LLM_V43_FINAL_PLAN3_REQUEST_V1",
        "design_release_hash": release_hash,
        "model": cell["model"],
        "firm_key": firm_key,
        "action_space": cell["mode"],
        "budget": cell["budget"],
        "info": cell["info"],
        "condition": cell["condition"],
        "replicate": int(cell["replicate"]),
        "parent_request_id": parent_request_id,
    }
    return canonical_hash(identity)


def _inventory(design: DesignBundle, cohort: pd.DataFrame) -> pd.DataFrame:
    lookup = {row["cell_id"]: row for row in design.matrix}
    rows: list[dict[str, Any]] = []
    for cell in design.matrix:
        parent_cell = cell["parent_cell_id"].strip() or None
        for firm in cohort.to_dict("records"):
            parent = None
            if parent_cell:
                parent = request_id(design.design_release_hash, lookup[parent_cell], firm["firm_key"], None)
            rid = request_id(design.design_release_hash, cell, firm["firm_key"], parent)
            rows.append({
                "request_id": rid,
                "protocol_release_hash": design.design_release_hash,
                "cell_id": cell["cell_id"],
                "firm_key": firm["firm_key"],
                "row_id": int(firm["row_id"]),
                "firm_id": str(firm["firm_id"]),
                "model_key": cell["model"],
                "exact_model_id": design.models["models"][cell["model"]]["exact_model_id"],
                "provider": design.models["models"][cell["model"]]["provider"],
                "block": cell["block"],
                "phase": cell["phase"],
                "mode": cell["mode"],
                "info": cell["info"],
                "budget": cell["budget"],
                "condition": cell["condition"],
                "replicate": int(cell["replicate"]),
                "wave": int(cell["wave"]),
                "parent_cell_id": parent_cell,
                "parent_request_id": parent,
                "parent_response_hash": None,
                "request_state": "PLANNED" if int(cell["wave"]) == 1 else "WAITING_PARENT",
            })
    out = pd.DataFrame(rows).sort_values(["wave", "model_key", "replicate", "cell_id", "row_id"]).reset_index(drop=True)
    if len(out) != 24_150 or out["request_id"].nunique() != 24_150:
        raise ContractError("Logical request inventory must have 24,150 unique identities")
    parent = out[out["wave"].eq(2)]
    wave1 = set(out.loc[out["wave"].eq(1), "request_id"])
    if parent["parent_request_id"].isna().any() or not set(parent["parent_request_id"]).issubset(wave1):
        raise ContractError("Revision parent identity drift")
    merged = parent.merge(out[["request_id", "replicate", "model_key", "firm_key", "condition"]], left_on="parent_request_id", right_on="request_id", suffixes=("", "__parent"), validate="many_to_one")
    if not merged["condition__parent"].eq("C4").all() or not merged["replicate"].eq(merged["replicate__parent"]).all() or not merged["firm_key"].eq(merged["firm_key__parent"]).all():
        raise ContractError("Revision lineage crosses condition, replicate, or firm")
    return out


def freeze_release(root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    design = load_design(root)
    from .semantic_fixture import run_semantic_fixture
    semantic_fixture = run_semantic_fixture(design.root)
    cohort = load_firm_cohort(design)
    directory = design.root / OUTPUT_ROOT / design.design_release_hash
    directory.mkdir(parents=True, exist_ok=True)
    inventory = _inventory(design, cohort)
    inventory.to_parquet(directory / "logical_requests.parquet", index=False)
    cohort.to_parquet(directory / "firm_payload_source.parquet", index=False)
    connection = connect(directory / "generation_ledger.sqlite")
    try:
        insert_logical_requests(connection, inventory.to_dict("records"))
    finally:
        connection.close()
    from .llm_contract import industry_binding_status
    industry = industry_binding_status(design.root)
    live_ready = bool(industry["ready"])
    release = {
        "schema_version": "llm_v43_final_plan3_runtime_release_v1",
        "status": "FROZEN_READY_WAVE1" if live_ready else "FROZEN_LIVE_BLOCKED_INDUSTRY",
        "protocol_release_hash": design.design_release_hash,
        "run_kind": "production_clean_run",
        "created_utc": utc_now(),
        "runtime_binding": {
            "models": design.models["models"],
            "action_contract_sha256": design.release["source_hashes"]["action_contract"],
            "parser_sha256": design.release["source_hashes"]["runtime_parser"],
            "provider_adapter_sha256": design.release["source_hashes"]["runtime_provider_adapters"],
            "semantic_fixture_status": semantic_fixture["status"],
            "semantic_fixture_schema": semantic_fixture["schema_version"],
        },
        "counts": {
            "logical_requests": 24_150,
            "cells": 42,
            "models": 2,
            "requests_per_model": 12_075,
            "main": 17_250,
            "stability": 6_900,
        },
        "canonical_matrix": "configs/current/llm/experiment_matrix.csv",
        "api_calls_executed": 0,
        "live_blockers": [] if live_ready else [f"industry binding unresolved for {industry['unresolved_count']}/575 canonical firms"],
    }
    write_json(directory / "release.json", release)
    return directory, release


def load_release(project_root: Path, release_hash: str) -> tuple[Path, dict[str, Any]]:
    directory = Path(project_root).resolve() / OUTPUT_ROOT / release_hash
    release = json.loads((directory / "release.json").read_text(encoding="utf-8"))
    if release.get("protocol_release_hash") != release_hash:
        raise ContractError("Runtime release identity mismatch")
    return directory, release


def _reference_id(design: DesignBundle, condition: str, own: str, mismatched: str) -> str | None:
    if condition == "C6-E":
        return own
    if condition == "C6-EX":
        return mismatched
    return None


@lru_cache(maxsize=4)
def _reference_table(path: str) -> pd.DataFrame:
    """Load the immutable 575-row reference table once per process."""
    return pd.read_parquet(path).set_index("row_id")


def _render(design: DesignBundle, firm: Mapping[str, Any], record: Mapping[str, Any], parent_text: str | None) -> Any:
    refs = _reference_table(str((design.root / "configs/current/llm/C6EX_materialized.parquet").resolve()))
    reference = _reference_id(design, str(record["condition"]), str(refs.loc[int(record["row_id"]), "own_C3E_action"]), str(refs.loc[int(record["row_id"]), "C6EX_reference_action"]))
    return render_policy_prompt(
        design,
        row=firm,
        condition=str(record["condition"]),
        mode=str(record["mode"]),
        info=str(record["info"]),
        budget=str(record["budget"]),
        currency_unit="thousand Korean won (KRW)",
        raw_parent_text=parent_text,
        reference_candidate_id=reference,
    )


def prepare_wave(project_root: Path, release_hash: str, wave: int) -> dict[str, Any]:
    root = Path(project_root).resolve()
    design = load_design(root)
    directory, release = load_release(root, release_hash)
    if release["status"] != "FROZEN_READY_WAVE1":
        raise ContractError("Live batch preparation is blocked until all doctor gates pass")
    inventory = pd.read_parquet(directory / "logical_requests.parquet")
    cohort = pd.read_parquet(directory / "firm_payload_source.parquet").set_index("firm_key")
    selected = inventory[inventory["wave"].eq(int(wave))]
    connection = connect(directory / "generation_ledger.sqlite")
    batches: list[dict[str, Any]] = []
    try:
        for (model_key, replicate), group in selected.groupby(["model_key", "replicate"], sort=True):
            model = design.models["models"][model_key]
            records = []
            for item in group.to_dict("records"):
                parent_text = None
                if int(wave) == 2:
                    row = connection.execute("SELECT selected_attempt_index FROM logical_requests WHERE request_id=?", (item["parent_request_id"],)).fetchone()
                    if row is None or row["selected_attempt_index"] is None:
                        raise ContractError("Wave2 cannot materialize before its exact C4 parent is locked")
                    parent = connection.execute("SELECT raw_visible_text,raw_visible_text_sha256 FROM attempts WHERE request_id=? AND attempt_index=?", (item["parent_request_id"], row["selected_attempt_index"])).fetchone()
                    parent_text = bytes(parent["raw_visible_text"]).decode("utf-8") if isinstance(parent["raw_visible_text"], bytes) else str(parent["raw_visible_text"] or "")
                    if not parent_text:
                        set_dependency_empty(connection, item["request_id"])
                        continue
                rendered = _render(design, cohort.loc[item["firm_key"]].to_dict(), item, parent_text)
                batch = build_batch_record(provider=model["provider"], exact_model_id=model["exact_model_id"], request_id=item["request_id"], system_text=rendered.system, user_text=rendered.user, max_output_tokens=model["max_output_tokens"])
                materialize_payload(connection, request_id=item["request_id"], system_text=rendered.system, user_text=rendered.user, visible_payload_sha256=rendered.visible_payload_sha256, provider_request_body_sha256=provider_request_hash(batch))
                records.append(batch)
            for ordinal, start in enumerate(range(0, len(records), 500), 1):
                block = records[start:start + 500]
                path = directory / "provider_batches" / f"wave{wave}" / model["provider"] / f"r{replicate}" / f"shard_{ordinal:05d}_attempt1.jsonl"
                count, digest = write_jsonl(path, block)
                ids = [str(x.get("custom_id") or x.get("key")) for x in block]
                sid = register_shard(connection, release_hash=release_hash, wave=wave, model_key=model_key, provider=model["provider"], attempt_index=1, ordinal=int(replicate) * 100000 + ordinal, input_path=relative_to_root(directory, path), input_sha256=digest, request_ids=ids)
                batches.append({"shard_id": sid, "model_key": model_key, "provider": model["provider"], "replicate": int(replicate), "path": relative_to_root(directory, path), "requests": count, "sha256": digest})
    finally:
        connection.close()
    manifest = {"status": "PREPARED", "wave": int(wave), "protocol_release_hash": release_hash, "provider_batches": batches, "prepared_requests": sum(x["requests"] for x in batches), "api_calls_executed": 0}
    write_json(directory / "manifests" / f"wave{wave}_prepared.json", manifest)
    return manifest


def generation_status(project_root: Path, release_hash: str) -> dict[str, Any]:
    directory, release = load_release(project_root, release_hash)
    connection = connect(directory / "generation_ledger.sqlite")
    try:
        states = status_counts(connection)
    finally:
        connection.close()
    return {"status": "GENERATION_COMPLETE" if states.get("DELIVERED_LOCKED", 0) == 24_150 else "GENERATION_INCOMPLETE", "protocol_release_hash": release_hash, "ledger_states": states, "expected_logical_requests": 24_150}


def run_full_dry_run(root: Path | None, output_dir: Path) -> dict[str, Any]:
    design = load_design(root)
    from .semantic_fixture import run_semantic_fixture
    semantic_fixture = run_semantic_fixture(design.root)
    cohort = load_firm_cohort(design)
    inventory = _inventory(design, cohort)
    refs = pd.read_parquet(design.root / "configs/current/llm/C6EX_materialized.parquet")
    ref_by_row = refs.set_index("row_id")
    cell_lookup = {row["cell_id"]: row for row in design.matrix}
    synthetic_parent = json.dumps({"diagnosis": "Synthetic dry-run parent.", "action": {name: 0.0 for name in design.action_contract["normalization_scale_by_dimension"]}, "evidence_fields": [], "rationale": "No live completion is used."}, separators=(",", ":"))
    firm_by_key = cohort.set_index("firm_key")
    prompt_rows, adapter_rows = [], []
    for record in inventory.to_dict("records"):
        firm = firm_by_key.loc[record["firm_key"]].to_dict()
        parent_text = synthetic_parent if int(record["wave"]) == 2 else None
        rendered = _render(design, firm, record, parent_text)
        model = design.models["models"][record["model_key"]]
        batch = build_batch_record(provider=model["provider"], exact_model_id=model["exact_model_id"], request_id=record["request_id"], system_text=rendered.system, user_text=rendered.user, max_output_tokens=model["max_output_tokens"])
        prompt_rows.append({"request_id": record["request_id"], "visible_payload_sha256": rendered.visible_payload_sha256, "parent_response_hash": hashlib.sha256(synthetic_parent.encode()).hexdigest() if parent_text else None, "synthetic_parent_only": bool(parent_text)})
        adapter_rows.append({"request_id": record["request_id"], "provider": model["provider"], "provider_request_sha256": provider_request_hash(batch)})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_parquet(output_dir / "logical_request_inventory.parquet", index=False)
    pd.DataFrame(prompt_rows).to_parquet(output_dir / "dry_prompt_hashes.parquet", index=False)
    pd.DataFrame(adapter_rows).to_parquet(output_dir / "provider_adapter_hashes.parquet", index=False)
    from .llm_contract import industry_binding_status
    industry = industry_binding_status(design.root)
    summary = {
        "schema_version": "llm_v43_final_plan3_full_dry_run_v1",
        "status": "PASS_READY" if industry["ready"] else "PASS_CONTRACT_LIVE_BLOCKED_INDUSTRY",
        "design_release_hash": design.design_release_hash,
        "logical_requests": len(inventory),
        "unique_request_ids": inventory["request_id"].nunique(),
        "models": inventory["model_key"].value_counts().sort_index().to_dict(),
        "main_requests": int(inventory["phase"].eq("MAIN").sum()),
        "stability_requests": int(inventory["phase"].eq("STABILITY").sum()),
        "parent_requests": int(inventory["parent_request_id"].notna().sum()),
        "mixed_replicate_parent_links": 0,
        "provider_payloads_rendered": len(adapter_rows),
        "synthetic_parent_payloads": int(sum(row["synthetic_parent_only"] for row in prompt_rows)),
        "api_calls_executed": 0,
        "industry_binding_ready": bool(industry["ready"]),
        "industry_binding_unresolved": int(industry["unresolved_count"]),
    }
    write_json(output_dir / "DRY_RUN_VALIDATION.json", summary)
    return summary

