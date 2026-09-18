from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import ContractError, iter_jsonl, relative_to_root, utc_now, write_json, write_jsonl
from .retry_policy import is_retryable_class
from .ledger import connect, register_shard, status_counts
from .executor import load_release


def prepare_retry(project_root: Path, release_hash: str, wave: int, attempt_index: int) -> dict[str, Any]:
    if int(attempt_index) not in {2, 3}:
        raise ContractError("Retry attempt_index must be 2 or 3")
    root = Path(project_root).resolve()
    directory, _ = load_release(root, release_hash)
    original = json.loads((directory / "manifests" / f"wave{int(wave)}_prepared.json").read_text(encoding="utf-8"))
    if original.get("status") != "PREPARED":
        raise ContractError(f"Original wave {wave} is not fully prepared")
    ledger = connect(directory / "generation_ledger.sqlite")
    batches: list[dict[str, Any]] = []
    total = 0
    try:
        for item in original["provider_batches"]:
            source = directory / item["path"]
            eligible: list[dict[str, Any]] = []
            for record in iter_jsonl(source):
                request_id = str(record.get("custom_id") or record.get("key"))
                logical = ledger.execute("SELECT selected_attempt_index FROM logical_requests WHERE request_id=?", (request_id,)).fetchone()
                attempts = ledger.execute("SELECT attempt_index,attempt_state,retry_class FROM attempts WHERE request_id=? ORDER BY attempt_index", (request_id,)).fetchall()
                if logical is None:
                    raise ContractError(f"Retry source contains unknown request: {request_id}")
                if logical["selected_attempt_index"] is not None:
                    continue
                if (
                    [row["attempt_index"] for row in attempts] == list(range(1, int(attempt_index)))
                    and all(
                        row["attempt_state"] == "NO_CONFIRMED_COMPLETION"
                        and is_retryable_class(row["retry_class"])
                        for row in attempts
                    )
                ):
                    eligible.append(record)
            if not eligible:
                continue
            if len(eligible) > 500:
                raise ContractError("Retry shard exceeds 500 requests")
            provider = str(item["provider"])
            target = directory / "provider_batches" / f"wave{int(wave)}" / provider / f"r{int(item.get('replicate', 1))}" / f"shard_{int(item['ordinal']):05d}_attempt{int(attempt_index)}.jsonl"
            count, digest = write_jsonl(target, eligible)
            request_ids = [str(row.get("custom_id") or row.get("key")) for row in eligible]
            ledger_ordinal = int(item.get("ledger_ordinal", int(item.get("replicate", 1)) * 100000 + int(item["ordinal"])))
            sid = register_shard(
                ledger,
                release_hash=release_hash,
                wave=int(wave),
                model_key=str(item["model_key"]),
                provider=provider,
                attempt_index=int(attempt_index),
                ordinal=ledger_ordinal,
                input_path=relative_to_root(directory, target),
                input_sha256=digest,
                request_ids=request_ids,
            )
            total += count
            batches.append({**item, "shard_id": sid, "ledger_ordinal": ledger_ordinal, "path": relative_to_root(directory, target), "requests": count, "sha256": digest, "attempt_index": int(attempt_index)})
        result = {
            "schema_version": "llm_v43_retry_preparation_v2",
            "status": "PREPARED" if total else "NO_RETRY_NEEDED",
            "protocol_release_hash": release_hash,
            "wave": int(wave),
            "attempt_index": int(attempt_index),
            "prepared_requests": total,
            "maximum_requests_per_shard": 500,
            "provider_batches": batches,
            "ledger_states": status_counts(ledger),
            "prepared_utc": utc_now(),
        }
    finally:
        ledger.close()
    write_json(directory / "manifests" / f"wave{int(wave)}_attempt{int(attempt_index)}_retry_prepared.json", result)
    return result
