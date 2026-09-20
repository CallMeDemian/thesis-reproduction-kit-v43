from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.final_release.common import ContractError, canonical_hash, relative_to_root, utc_now, write_json, write_jsonl
from credit_recourse.final_release.contract import load_design
from credit_recourse.final_release.executor import output_root, _render
from credit_recourse.final_release.ledger import connect, insert_logical_requests, materialize_payload, register_shard
from credit_recourse.final_release.providers import provider_request_hash

HIGH_CAP = 32768
EXPECTED = 24150


def _dir(root: Path, release_hash: str) -> Path:
    return output_root(root) / release_hash


def _record(model: dict[str, Any], request_id: str, system: str, user: str) -> dict[str, Any]:
    if model["provider"] == "openai":
        return {"custom_id": request_id, "method": "POST", "url": "/v1/responses", "body": {"model": model["exact_model_id"], "input": [{"role": "system", "content": [{"type": "input_text", "text": system}]}, {"role": "user", "content": [{"type": "input_text", "text": user}]}], "reasoning": {"effort": "high"}, "max_output_tokens": HIGH_CAP, "store": False}}
    if model["provider"] == "google":
        return {"key": request_id, "request": {"contents": [{"role": "user", "parts": [{"text": user}]}], "systemInstruction": {"parts": [{"text": system}]}, "generationConfig": {"maxOutputTokens": HIGH_CAP, "thinkingConfig": {"thinkingLevel": "high"}}}}
    raise ContractError("Unsupported provider")


def freeze(root: Path, baseline_hash: str) -> tuple[Path, dict[str, Any]]:
    root = Path(root).resolve()
    base = _dir(root, baseline_hash)
    baseline = json.loads((base / "release.json").read_text(encoding="utf-8"))
    inv = pd.read_parquet(base / "logical_requests.parquet").sort_values(["wave", "model_key", "replicate", "cell_id", "row_id"]).reset_index(drop=True)
    identity = {"namespace": "LLM_HIGH_REASONING_FULL_REPLICATION_V1", "baseline_release_hash": baseline_hash, "reasoning_arm": "HIGH", "openai_effort": "high", "gemini_thinking_level": "high", "max_output_tokens": HIGH_CAP, "matrix_hash": hashlib.sha256((base / "logical_requests.parquet").read_bytes()).hexdigest()}
    high_hash = canonical_hash(identity)
    directory = _dir(root, high_hash)
    if directory.exists():
        return directory, json.loads((directory / "release.json").read_text(encoding="utf-8"))
    directory.mkdir(parents=True)
    high_ids: dict[str, str] = {}
    records = []
    for row in inv.to_dict("records"):
        parent = high_ids.get(row.get("parent_request_id"))
        request_id = canonical_hash({"namespace": "LLM_HIGH_REASONING_REQUEST_V1", "high_release_hash": high_hash, "baseline_request_id": row["request_id"], "reasoning_arm": "HIGH", "parent_request_id": parent})
        high_ids[row["request_id"]] = request_id
        row["baseline_request_id"] = row["request_id"]
        row["request_id"] = request_id
        row["protocol_release_hash"] = high_hash
        row["parent_request_id"] = parent
        row["reasoning_arm"] = "HIGH"
        row["request_state"] = "PLANNED" if int(row["wave"]) == 1 else "WAITING_PARENT"
        records.append(row)
    high = pd.DataFrame(records)
    if len(high) != EXPECTED or high.request_id.nunique() != EXPECTED:
        raise ContractError("High inventory identity failure")
    high.to_parquet(directory / "logical_requests.parquet", index=False)
    shutil.copy2(base / "firm_payload_source.parquet", directory / "firm_payload_source.parquet")
    models = json.loads(json.dumps(baseline["runtime_binding"]["models"]))
    models["openai_gpt54mini"]["reasoning"] = {"effort": "high"}
    models["openai_gpt54mini"]["max_output_tokens"] = HIGH_CAP
    models["google_gemini31flashlite"]["thinking"] = {"thinkingLevel": "high"}
    models["google_gemini31flashlite"]["max_output_tokens"] = HIGH_CAP
    release = {"schema_version": "llm_high_reasoning_replication_v1", "status": "FROZEN_READY_WAVE1", "protocol_release_hash": high_hash, "run_kind": "post_hoc_reasoning_effort_sensitivity", "created_utc": utc_now(), "baseline_release_hash": baseline_hash, "reasoning_arm": "HIGH", "runtime_binding": {**baseline["runtime_binding"], "models": models}, "counts": baseline["counts"], "high_reasoning_output_ceiling_technical_amendment": {"baseline": 4096, "high": HIGH_CAP, "reason": "Gemini high thinking uses a combined thought/output ceiling"}}
    source_hashes = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in ["release.json", "protocol_release.json", "logical_requests.parquet", "firm_payload_source.parquet", "generation_ledger.sqlite"] if (base / name).exists()}
    write_json(directory / "release.json", release)
    write_json(directory / "HIGH_REASONING_PROTOCOL.json", {"identity": identity, "baseline_release_hash": baseline_hash, "high_release_hash": high_hash, "one_treatment_rule": release["runtime_binding"]["models"], "baseline_source_hashes": source_hashes, "cost_monitoring": "record-only; no spending ceiling is applied"})
    high[["request_id", "baseline_request_id", "model_key", "firm_key", "cell_id", "replicate", "wave", "parent_request_id"]].to_parquet(directory / "baseline_high_pairing_manifest.parquet", index=False)
    connection = connect(directory / "generation_ledger.sqlite")
    try:
        insert_logical_requests(connection, high.to_dict("records"))
    finally:
        connection.close()
    return directory, release


def prepare(root: Path, high_hash: str, wave: int, *, preflight: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    directory = _dir(root, high_hash)
    release = json.loads((directory / "release.json").read_text(encoding="utf-8"))
    design = load_design(root)
    inventory = pd.read_parquet(directory / "logical_requests.parquet")
    cohort = pd.read_parquet(directory / "firm_payload_source.parquet").set_index("firm_key")
    connection = connect(directory / "generation_ledger.sqlite")
    batches = []
    try:
        if wave == 2:
            connection.execute("UPDATE logical_requests SET request_state='READY' WHERE wave=2 AND request_state='WAITING_PARENT' AND parent_request_id IN (SELECT request_id FROM logical_requests WHERE wave=1 AND request_state='DELIVERED_LOCKED' AND selected_attempt_index IS NOT NULL)")
            connection.commit()
        state = "PLANNED" if wave == 1 else "READY"
        ready = {str(row[0]) for row in connection.execute("SELECT request_id FROM logical_requests WHERE request_state=?", (state,))}
        selected = inventory[(inventory.wave.eq(wave)) & inventory.request_id.isin(ready)]
        if preflight:
            if wave != 1:
                raise ContractError("Preflight is wave1 only")
            selected = selected[selected.condition.isin(["C4", "C5"]) & selected["mode"].eq("free8") & selected["info"].eq("IC-b") & selected["budget"].eq("B1") & selected["replicate"].eq(1) & selected.row_id.lt(100)]
            if len(selected) != 400:
                raise ContractError("Embedded preflight must contain 400 final-inventory requests")
        for (model_key, replicate), group in selected.groupby(["model_key", "replicate"], sort=True):
            model = release["runtime_binding"]["models"][model_key]
            records = []
            for item in group.to_dict("records"):
                parent_text = None
                if wave == 2:
                    row = connection.execute("SELECT selected_attempt_index FROM logical_requests WHERE request_id=?", (item["parent_request_id"],)).fetchone()
                    if row is None or row[0] is None:
                        raise ContractError("High wave2 parent not locked")
                    parent = connection.execute("SELECT raw_visible_text FROM attempts WHERE request_id=? AND attempt_index=?", (item["parent_request_id"], row[0])).fetchone()
                    parent_text = parent[0].decode() if isinstance(parent[0], bytes) else str(parent[0] or "")
                    if not parent_text:
                        raise ContractError("High wave2 parent empty")
                rendered = _render(design, cohort.loc[item["firm_key"]].to_dict(), item, parent_text)
                batch = _record(model, item["request_id"], rendered.system, rendered.user)
                materialize_payload(connection, request_id=item["request_id"], system_text=rendered.system, user_text=rendered.user, visible_payload_sha256=rendered.visible_payload_sha256, provider_request_body_sha256=provider_request_hash(batch))
                records.append(batch)
            existing = connection.execute("SELECT COUNT(*) FROM shards WHERE wave=? AND model_key=? AND attempt_index=1", (wave, model_key)).fetchone()[0]
            for ordinal, start in enumerate(range(0, len(records), 500), 1):
                block = records[start:start + 500]
                path = directory / "provider_batches" / f"wave{wave}" / model["provider"] / f"r{replicate}" / f"high_{'preflight_' if preflight else ''}shard_{ordinal:05d}_attempt1.jsonl"
                count, digest = write_jsonl(path, block)
                shard_ordinal = int(replicate) * 100000 + existing + ordinal
                shard_id = register_shard(connection, release_hash=high_hash, wave=wave, model_key=model_key, provider=model["provider"], attempt_index=1, ordinal=shard_ordinal, input_path=relative_to_root(directory, path), input_sha256=digest, request_ids=[str(item.get("custom_id") or item.get("key")) for item in block])
                batches.append({"shard_id": shard_id, "model_key": model_key, "provider": model["provider"], "replicate": int(replicate), "path": relative_to_root(directory, path), "requests": count, "sha256": digest})
    finally:
        connection.close()
    manifest = {"status": "PREPARED", "wave": wave, "protocol_release_hash": high_hash, "reasoning_arm": "HIGH", "preflight": preflight, "provider_batches": batches, "prepared_requests": sum(item["requests"] for item in batches)}
    write_json(directory / "manifests" / f"wave{wave}_prepared.json", manifest)
    return manifest
