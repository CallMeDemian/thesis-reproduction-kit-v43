from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from .contract import load_design
from .parsing import parse_policy_response


def find_asset_roots(root: Path) -> list[Path]:
    candidates = [
        root / "release_assets/LLM_FINAL_RAW_RESPONSES",
        root / "release_assets/installed/LLM_FINAL_RAW_RESPONSES",
    ]
    return [path for path in candidates if path.is_dir()]


def ledgers(root: Path) -> list[Path]:
    found = []
    for asset_root in find_asset_roots(root):
        found.extend(sorted(asset_root.rglob("generation_ledger.sqlite")))
    return found


def schema_audit(root: Path) -> dict:
    result = {"schema_version": "v21_llm_replay_schema_audit", "ledgers": []}
    for ledger in ledgers(root):
        with sqlite3.connect(str(ledger)) as connection:
            tables = [row[0] for row in connection.execute("select name from sqlite_master where type='table'")]
            required = {"logical_requests", "attempts", "parsed_responses", "payloads"}
            missing = sorted(required - set(tables))
            counts = {}
            for table in sorted(required - set(missing)):
                counts[table] = int(connection.execute(f"select count(*) from {table}").fetchone()[0])
            result["ledgers"].append({"path": str(ledger), "tables": sorted(tables), "required_tables_missing": missing, "counts": counts})
    result["status"] = "PASS" if result["ledgers"] and not any(x["required_tables_missing"] for x in result["ledgers"]) else "MISSING_ASSET"
    return result


def replay_parser_sample(root: Path, run_id: str, limit: int = 5) -> dict:
    design = load_design(root)
    ledger_paths = ledgers(root)
    if not ledger_paths:
        raise FileNotFoundError("LLM_FINAL_RAW_RESPONSES asset is not installed")
    matrix = {row["cell_id"]: row for row in design.matrix}
    records = []
    for ledger in ledger_paths:
        with sqlite3.connect(str(ledger)) as connection:
            query = """select l.request_id,l.cell_id,l.firm_key,l.model_key,l.wave,l.parent_request_id,
                       a.attempt_index,a.raw_visible_text,a.raw_provider_json,p.parse_json
                       from logical_requests l join attempts a on a.request_id=l.request_id
                       and a.attempt_index=l.selected_attempt_index left join parsed_responses p
                       on p.request_id=l.request_id and p.selected_attempt_index=l.selected_attempt_index
                       where a.raw_visible_text is not null order by l.request_id limit ?"""
            rows = connection.execute(query, (limit,)).fetchall()
        for request_id, cell_id, firm_key, model_key, wave, parent_id, attempt_index, raw, provider, stored_parse in rows:
            cell = matrix.get(cell_id)
            if not cell:
                raise ValueError(f"request identity missing from matrix: {cell_id}")
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            parsed = parse_policy_response(design, raw_visible_text=text, mode=cell["mode"], budget=cell["budget"])
            records.append({
                "request_id": request_id, "firm_key": firm_key, "model_key": model_key, "wave": wave,
                "parent_request_id": parent_id, "attempt_index": attempt_index, "reasoning_regime": "HIGH" if "high" in str(ledger).lower() else "BASELINE",
                "provider_raw_available": bool(provider), "stored_parser_available": stored_parse is not None,
                "candidate_id": parsed.candidate_id_req, "accepted": parsed.a_accepted is not None,
                "itt": parsed.a_itt, "parser_state": parsed.parser_state,
            })
            if len(records) >= limit:
                break
        if len(records) >= limit:
            break
    out = root / "runs" / run_id
    for stage in ("stage7", "stage8", "stage9"):
        (out / stage).mkdir(parents=True, exist_ok=True)
    with (out / "stage7/replayed_actions.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {"run_id": run_id, "mode": "LLMActionReplay", "created_this_run": True, "provider_api_calls": 0, "raw_provider_payload_to_parser": True, "records": len(records), "stage8_status": "READY_FOR_FULL_MATERIALIZATION", "stage9_status": "READY_FOR_FULL_MATERIALIZATION"}
    (out / "stage7/REPLAY_METADATA.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata
