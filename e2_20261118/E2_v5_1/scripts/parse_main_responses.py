from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd


E2 = Path(__file__).resolve().parents[1]
TASK = E2.parent
WORKSPACE = TASK.parent
MAIN = WORKSPACE / "archive/DEPLOYED_RELEASE/llm_runs/final_plan3/d5d878bf50718bac65ede73df3fc88c16723870b6c90a790557ff35b063b7ba7"
KIT = TASK / "A_reproduction/E1_score_gate_replay/runroot_zip/THESIS_REPRO_KIT_v2.1.1_FINAL"
E1_PROCESS = TASK / "source_inputs/github_release_ablation_20261102_readonly/contents/ablation_20261102/E1/scripts/e1_process.py"
PROMPT_CONTRACT = TASK / "source_inputs/github_main/frozen/evidence/llm/prompt_contract.json"
ACTION_CONTRACT = KIT / "archive/DEPLOYED_RELEASE/stage2_candidate_projection/candidate_action_contract_v4_3.json"
PILOT_OUTPUTS = E2 / "responses"
MAIN_OUTPUTS = E2 / "responses_main"
MANIFEST = E2 / "manifests_main/e2_main_execution_manifest.csv"
JOB_STATE = E2 / "logs_main/provider_jobs_main.json"
OUTPUT = E2 / "results"
MODELS = {"openai": "gpt-5.4-mini-2026-03-17", "google": "gemini-3.1-flash-lite"}
MODEL_KEYS = {"openai": "openai_gpt54mini", "google": "google_gemini31flashlite"}
RATES = {"openai": {"input": 0.375, "output": 2.25}, "google": {"input": 0.125, "output": 0.75}}
Q_METRICS = {"pretax_income", "interest_expense", "total_debt", "cash", "current_liabilities"}
Q_DIRECTIONS = {"increase", "decrease", "no_material_change"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def request_id(provider: str, line: dict[str, Any]) -> str | None:
    return str(line.get("custom_id" if provider == "openai" else "key") or "") or None


def load_e1_processor():
    spec = importlib.util.spec_from_file_location("e1_process_main", E1_PROCESS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen E1 parser at {E1_PROCESS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_design():
    if str(KIT / "src") not in sys.path:
        sys.path.insert(0, str(KIT / "src"))
    return SimpleNamespace(root=KIT,
                           prompts=json.loads(PROMPT_CONTRACT.read_text(encoding="utf-8")),
                           action_contract=json.loads(ACTION_CONTRACT.read_text(encoding="utf-8")))


def load_ledger_rows(ids: list[str]) -> dict[str, dict[str, Any]]:
    db = MAIN / "generation_ledger.sqlite"
    conn = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out: dict[str, dict[str, Any]] = {}
    try:
        for start in range(0, len(ids), 400):
            batch = ids[start:start + 400]
            slots = ",".join("?" for _ in batch)
            query = f"""
            SELECT lr.request_id, lr.request_state, lr.selected_raw_sha256,
                   p.visible_payload_sha256,
                   a.raw_visible_text_sha256, a.raw_visible_text, a.completion_kind,
                   a.attempt_state, a.provider_job_id
            FROM logical_requests lr
            LEFT JOIN payloads p ON p.request_id=lr.request_id
            LEFT JOIN attempts a ON a.request_id=lr.request_id AND a.attempt_index=lr.selected_attempt_index
            WHERE lr.request_id IN ({slots})
            """
            for row in conn.execute(query, batch):
                value = dict(row)
                raw = value.get("raw_visible_text")
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                value["raw_visible_text"] = raw or ""
                out[str(value["request_id"])] = value
    finally:
        conn.close()
    if set(ids) != set(out):
        raise RuntimeError(f"Locked source ledger missing {len(set(ids)-set(out))} IDs")
    return out


def load_raw_outputs(manifest: pd.DataFrame) -> tuple[dict[str, tuple[str, dict[str, Any], Path]], list[dict[str, Any]], list[dict[str, Any]]]:
    records: dict[str, tuple[str, dict[str, Any], Path]] = {}
    file_hashes: list[dict[str, Any]] = []
    duplicate_audit: list[dict[str, Any]] = []
    for provider in MODELS:
        for folder, patterns in [(PILOT_OUTPUTS, [f"pilot_{provider}_attempt1_*output.jsonl", f"pilot_{provider}_attempt1_*error.jsonl"]),
                                 (MAIN_OUTPUTS, [f"main_{provider}_*_output.jsonl", f"main_{provider}_*_error.jsonl"])]:
            paths = sorted({path for pattern in patterns for path in folder.glob(pattern)})
            for path in paths:
                file_hashes.append({"path": str(path.relative_to(WORKSPACE)), "bytes": path.stat().st_size,
                                    "sha256": sha256_file(path)})
                for line in read_jsonl(path):
                    rid = request_id(provider, line)
                    if not rid:
                        raise RuntimeError(f"Response line has no request ID: {path}")
                    if rid in records:
                        prior_provider, prior_line, prior_path = records[rid]
                        prior_error = "_error.jsonl" in prior_path.name
                        current_error = "_error.jsonl" in path.name
                        prior_retry = "retry" in prior_path.stem.lower()
                        current_retry = "retry" in path.stem.lower()
                        # A single frozen infrastructure retry may add a successful output
                        # for an ID previously present in the initial batch's error file.
                        # Retain both artifact hashes, but normalize the retry output.
                        allowed = (
                            prior_provider == provider
                            and prior_error and current_retry
                        ) or (
                            prior_provider == provider
                            and prior_retry and not prior_error and current_error
                        )
                        if not allowed:
                            raise RuntimeError(f"Unexpected duplicate output request ID: {rid} ({prior_path.name}, {path.name})")
                        selected_line, selected_path = (line, path) if current_retry else (prior_line, prior_path)
                        duplicate_audit.append({
                            "request_id": rid,
                            "provider": provider,
                            "prior_path": str(prior_path.relative_to(WORKSPACE)),
                            "prior_sha256": sha256_bytes(canonical(prior_line).encode("utf-8")),
                            "new_path": str(path.relative_to(WORKSPACE)),
                            "new_sha256": sha256_bytes(canonical(line).encode("utf-8")),
                            "selected_path": str(selected_path.relative_to(WORKSPACE)),
                            "resolution": "single_frozen_retry_record_over_initial_error",
                        })
                        records[rid] = (provider, selected_line, selected_path)
                        continue
                    records[rid] = (provider, line, path)
    return records, file_hashes, duplicate_audit


def q_prompt(provider: str, request: dict[str, Any]) -> str:
    if provider == "openai":
        return next(item["content"][0]["text"] for item in request["body"]["input"] if item.get("role") == "user")
    return request["request"]["contents"][0]["parts"][0]["text"]


def q_expected_ids(provider: str, request: dict[str, Any]) -> list[int]:
    import re
    prompt = q_prompt(provider, request)
    return [int(item) for item in re.findall(r'"candidate"\s*:\s*"[^"]+"\s*,\s*"candidate_id"\s*:\s*(\d+)', prompt)]


def q_validate(text: str, expected_ids: list[int]) -> tuple[bool, list[str], int, dict[str, Any] | None]:
    errors: list[str] = []
    mismatch = 0
    try:
        obj = json.loads(text)
    except Exception:
        return False, ["invalid_json_or_non_json_text"], 0, None
    if not isinstance(obj, dict) or set(obj) != {"forecasts"}:
        return False, ["top_level_schema_mismatch"], 0, None
    forecasts = obj.get("forecasts")
    if not isinstance(forecasts, list) or len(forecasts) != len(expected_ids):
        return False, ["forecast_count_mismatch"], 0, None
    observed_ids = []
    for index, row in enumerate(forecasts):
        if not isinstance(row, dict) or set(row) != {"candidate_id", *Q_METRICS}:
            errors.append(f"row_{index}_field_set_mismatch")
            continue
        cid = row.get("candidate_id")
        if isinstance(cid, bool) or not isinstance(cid, int):
            errors.append(f"row_{index}_candidate_id_not_integer")
        else:
            observed_ids.append(cid)
        for metric in Q_METRICS:
            item = row.get(metric)
            if not isinstance(item, dict) or set(item) != {"direction", "change_thousand_krw"}:
                errors.append(f"row_{index}_{metric}_field_set_mismatch")
                continue
            direction, amount = item.get("direction"), item.get("change_thousand_krw")
            if direction not in Q_DIRECTIONS:
                errors.append(f"row_{index}_{metric}_invalid_direction")
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                errors.append(f"row_{index}_{metric}_invalid_amount")
                continue
            if not math.isfinite(float(amount)):
                errors.append(f"row_{index}_{metric}_nonfinite_amount")
                continue
            if direction == "increase" and not amount > 0:
                mismatch += 1
            elif direction == "decrease" and not amount < 0:
                mismatch += 1
            elif direction == "no_material_change" and amount != 0:
                mismatch += 1
    if observed_ids != expected_ids:
        errors.append("candidate_ids_or_order_mismatch")
    return not errors, errors, mismatch, obj if not errors else None


def usage_tokens(provider: str, line: dict[str, Any]) -> tuple[int, int]:
    response = line.get("response") if isinstance(line.get("response"), dict) else {}
    body = response.get("body") if provider == "openai" and isinstance(response.get("body"), dict) else response
    usage = body.get("usage") or body.get("usageMetadata") or {}
    if provider == "openai":
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    return int(usage.get("promptTokenCount", 0)), int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))


def main() -> None:
    manifest = pd.read_csv(MANIFEST, dtype={"request_id": str, "firm_key": str})
    if len(manifest) != 10350 or manifest.request_id.nunique() != 10350:
        raise RuntimeError("Main execution manifest must contain 10,350 unique logical requests")
    counts = manifest.groupby(["condition", "generation_source"]).size().to_dict()
    expected = {
        ("N", "existing_main_reuse"): 2300,
        ("R", "existing_main_reuse"): 2300,
        ("T", "pilot_reuse"): 200, ("RT", "pilot_reuse"): 200, ("Q", "pilot_reuse"): 200,
        ("T", "new_request"): 2100, ("RT", "new_request"): 2100, ("Q", "new_request"): 950,
    }
    if counts != expected:
        raise RuntimeError(f"Unexpected execution matrix: {counts}")

    jobs = json.loads(JOB_STATE.read_text(encoding="utf-8"))
    summary = jobs.get("last_poll_summary", {})
    accepted_jobs = summary.get("jobs", [])
    if len(accepted_jobs) < 24 or not all(j.get("terminal") for j in accepted_jobs):
        raise RuntimeError("All initial and any authorized retry provider chunks must be terminal before parsing")
    if any(j.get("terminal_failure") for j in accepted_jobs):
        raise RuntimeError("A provider batch itself failed; inspect job state and apply only the frozen infrastructure retry rule")

    raw_records, file_hashes, duplicate_audit = load_raw_outputs(manifest)
    output_artifacts = {
        str(artifact["path"]): str(artifact["sha256"])
        for job in summary["jobs"] for artifact in job.get("artifacts", [])
        if artifact.get("kind") in {"output", "error"}
    }
    observed_artifacts = {Path(item["path"]).name: item["sha256"] for item in file_hashes
                          if str(Path(item["path"]).parent).endswith("responses_main")}
    if output_artifacts != observed_artifacts:
        raise RuntimeError(f"Provider output artifact/hash mismatch: provider={len(output_artifacts)}, local={len(observed_artifacts)}")
    (E2 / "logs_main/main_response_duplicate_audit.json").write_text(
        json.dumps({"duplicates_resolved": len(duplicate_audit), "records": duplicate_audit}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    new_ids = set(manifest.loc[manifest.generation_source.eq("new_request"), "request_id"].astype(str))
    pilot_ids = set(manifest.loc[manifest.generation_source.eq("pilot_reuse"), "reused_pilot_request_id"].astype(str))
    observed_new = {rid for rid, (_, _, path) in raw_records.items() if path.parent == MAIN_OUTPUTS}
    observed_pilot = {rid for rid, (_, _, path) in raw_records.items() if path.parent == PILOT_OUTPUTS}
    if observed_new != new_ids:
        raise RuntimeError(f"New API response ID mismatch: expected {len(new_ids)}, found {len(observed_new)}, missing={len(new_ids-observed_new)}, extra={len(observed_new-new_ids)}")
    if not pilot_ids.issubset(observed_pilot):
        raise RuntimeError(f"Pilot response IDs missing: {len(pilot_ids-observed_pilot)}")

    source_ids = manifest.loc[manifest.generation_source.eq("existing_main_reuse"), "source_main_request_id"].astype(str).tolist()
    source_rows = load_ledger_rows(source_ids)
    proc = load_e1_processor()
    design = load_design()

    # Build the frozen request map for pilot Q structure validation and per-row payload hashes.
    pilot_request_map: dict[str, dict[str, Any]] = {}
    for provider in MODELS:
        request_path = E2 / "requests" / f"{provider}_pilot_mixed.jsonl"
        if not request_path.exists():
            continue
        for request in read_jsonl(request_path):
            rid = str(request.get("custom_id") if provider == "openai" else request.get("key"))
            pilot_request_map[rid] = request
    main_request_map: dict[str, dict[str, Any]] = {}
    for provider in MODELS:
        for path in sorted((E2 / "requests_main").glob(f"{provider}_*_chunk*.jsonl")):
            for request in read_jsonl(path):
                rid = str(request.get("custom_id") if provider == "openai" else request.get("key"))
                main_request_map[rid] = request
    expected_new_ids = set(manifest.loc[manifest.generation_source.eq("new_request"), "request_id"].astype(str))
    if set(main_request_map) != expected_new_ids:
        raise RuntimeError(f"Frozen new request ID mismatch: expected {len(expected_new_ids)}, found {len(main_request_map)}")
    for row in manifest[manifest.generation_source.isin(["new_request", "pilot_reuse"])].itertuples(index=False):
        source_id = str(row.request_id) if row.generation_source == "new_request" else str(row.reused_pilot_request_id)
        request = main_request_map.get(source_id) if row.generation_source == "new_request" else pilot_request_map.get(source_id)
        if request is None:
            raise RuntimeError(f"Frozen request payload missing for {row.request_id}")
        body = request["body"] if row.provider == "openai" else request["request"]
        body_hash = sha256_bytes(canonical(body).encode("utf-8"))
        if body_hash != str(row.provider_request_body_sha256):
            raise RuntimeError(f"Frozen request payload hash mismatch for {row.request_id}")

    parsed_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    duplicate_table_mismatch = []
    table_digest_by_firm: dict[tuple[str, str, int, int], set[str]] = {}
    for row in manifest.itertuples(index=False):
        provider = str(row.provider)
        condition = str(row.condition)
        rid = str(row.request_id)
        line = None
        raw_text = None
        completion = None
        raw_hash = None
        payload_hash = None
        source_path = None
        parsed = None
        if row.generation_source == "existing_main_reuse":
            src_id = str(row.source_main_request_id)
            src = source_rows[src_id]
            raw_text = src["raw_visible_text"]
            completion = str(src.get("completion_kind") or "usable_output")
            raw_hash = sha256_bytes(raw_text.encode("utf-8"))
            payload_hash = str(src.get("visible_payload_sha256") or "")
            if raw_hash != str(row.source_main_raw_sha256) or raw_hash != str(row.selected_raw_sha256):
                raise RuntimeError(f"Locked raw source hash mismatch: {rid}")
            if src.get("request_state") != "DELIVERED_LOCKED":
                raise RuntimeError(f"Unlocked main source row: {src_id}")
            usable, action, errors = parse_raw_action(proc, design, raw_text, completion)
            outcome = "DELIVERED"
            returned_model = str(row.exact_model_id)
            model_ok = True
            retry_class = None
            input_tokens = output_tokens = 0
        else:
            source_id = str(row.reused_pilot_request_id) if row.generation_source == "pilot_reuse" else rid
            observed = raw_records.get(source_id)
            if observed is None:
                raise RuntimeError(f"Missing raw response for {rid} (source {source_id})")
            observed_provider, line, source_path = observed
            if observed_provider != provider:
                raise RuntimeError(f"Provider mismatch for {rid}")
            raw_line_bytes = canonical(line).encode("utf-8")
            raw_hash = sha256_bytes(raw_line_bytes)
            proc_row = proc.process_line(provider, line, design)
            outcome = str(proc_row.get("outcome"))
            completion = proc_row.get("completion_kind")
            raw_text = proc_row.get("raw_visible_text")
            returned_model = proc_row.get("returned_model")
            model_ok = bool(proc_row.get("returned_model_ok")) if proc_row.get("returned_model_ok") is not None else False
            retry_class = proc_row.get("retry_class")
            usable = bool(proc_row.get("strict_usable")) if outcome == "DELIVERED" else False
            action = proc_row.get("strict_itt_action") if outcome == "DELIVERED" else None
            errors = proc_row.get("error_codes") or []
            input_tokens, output_tokens = usage_tokens(provider, line)
            if row.generation_source == "pilot_reuse":
                expected_hash = str(row.pilot_raw_record_sha256)
                if expected_hash and raw_hash != expected_hash:
                    raise RuntimeError(f"Pilot raw record hash mismatch: {rid}")
            else:
                if returned_model and str(returned_model) != str(row.exact_model_id):
                    model_ok = False
            if condition == "Q":
                request = (pilot_request_map.get(source_id) if row.generation_source == "pilot_reuse"
                           else main_request_map.get(rid))
                if request is None:
                    raise RuntimeError(f"Missing frozen Q request payload: {rid}")
                expected_ids = q_expected_ids(provider, request)
                raw_response = line.get("response") if isinstance(line.get("response"), dict) else {}
                raw_body = raw_response.get("body") if provider == "openai" and isinstance(raw_response.get("body"), dict) else raw_response
                usage = raw_body.get("usage") or raw_body.get("usageMetadata") or {}
                if outcome == "DELIVERED" and model_ok and completion == "usable_output" and str(raw_text or "").strip():
                    q_valid, q_errors, direction_sign_mismatches, forecast_obj = q_validate(str(raw_text), expected_ids)
                else:
                    q_valid, q_errors, direction_sign_mismatches, forecast_obj = False, [str(completion or retry_class or "not_delivered")], 0, None
                reasoning_tokens = (int((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
                                    if provider == "openai" else int(usage.get("thoughtsTokenCount", 0)))
                q_rows.append({
                    "request_id": rid, "provider": provider, "condition": condition,
                    "row_id": int(row.row_id), "firm_key": str(row.firm_key), "model_key": str(row.model_key),
                    "replicate": int(row.replicate), "generation_source": str(row.generation_source),
                    "delivered": outcome == "DELIVERED", "returned_model": returned_model,
                    "returned_model_ok": model_ok, "completion_kind": completion,
                    "json_schema_valid": q_valid, "schema_errors": "|".join(q_errors),
                    "direction_sign_mismatch_count": direction_sign_mismatches,
                    "expected_candidate_ids": json.dumps(expected_ids),
                    "forecast_json": json.dumps(forecast_obj, ensure_ascii=False, sort_keys=True) if forecast_obj is not None else None,
                    "input_tokens": input_tokens, "output_tokens_including_reasoning": output_tokens,
                    "reasoning_tokens": reasoning_tokens, "response_sha256": raw_hash,
                    "source_path": str(source_path.relative_to(WORKSPACE)) if source_path else None,
                })
        if condition != "Q":
            if action is None:
                action_json = None
            else:
                action_json = json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            parsed_rows.append({
                "request_id": rid, "provider": provider, "condition": condition,
                "row_id": int(row.row_id), "firm_key": str(row.firm_key), "model_key": str(row.model_key),
                "replicate": int(row.replicate), "generation_source": str(row.generation_source),
                "source_main_request_id": str(row.source_main_request_id) if pd.notna(row.source_main_request_id) else None,
                "reused_pilot_request_id": str(row.reused_pilot_request_id) if pd.notna(row.reused_pilot_request_id) else None,
                "outcome": outcome, "retry_class": retry_class, "completion_kind": completion,
                "returned_model": returned_model, "returned_model_ok": model_ok,
                "strict_usable": bool(usable), "strict_itt_action_json": action_json,
                "parse_error_codes": "|".join(errors), "input_tokens": input_tokens,
                "output_tokens_including_reasoning": output_tokens, "response_sha256": raw_hash,
                "payload_sha256": payload_hash,
                "source_path": str(source_path.relative_to(WORKSPACE)) if source_path else "generation_ledger.sqlite",
                "source_raw_sha256_expected": str(row.source_main_raw_sha256) if row.generation_source == "existing_main_reuse" else None,
            })

        if condition in {"T", "RT"}:
            key = (str(row.model_key), str(row.firm_key), int(row.row_id), int(row.replicate))
            table_digest_by_firm.setdefault(key, set()).add(str(row.simulator_table_sha256))

    if any(len(x) != 1 for x in table_digest_by_firm.values()):
        duplicate_table_mismatch = [k for k, v in table_digest_by_firm.items() if len(v) != 1]
        raise RuntimeError(f"T/RT table hashes do not match: {len(duplicate_table_mismatch)} row keys")

    parsed = pd.DataFrame(parsed_rows)
    q = pd.DataFrame(q_rows)
    if len(parsed) != 9200 or len(q) != 1150:
        raise RuntimeError(f"Parsed row mismatch: policy={len(parsed)}, Q={len(q)}")
    if parsed.request_id.nunique() != len(parsed) or q.request_id.nunique() != len(q):
        raise RuntimeError("Duplicate logical request IDs in parsed outputs")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(OUTPUT / "main_policy_responses.csv", index=False, encoding="utf-8-sig")
    parsed.to_parquet(OUTPUT / "main_policy_responses.parquet", index=False)
    q.to_csv(OUTPUT / "main_q_responses.csv", index=False, encoding="utf-8-sig")
    q.to_parquet(OUTPUT / "main_q_responses.parquet", index=False)
    file_hashes_out = {"at_local": pd.Timestamp.now().isoformat(), "files": file_hashes}
    (E2 / "logs_main/main_response_hashes.json").write_text(json.dumps(file_hashes_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    policy_summary = parsed.groupby(["model_key", "condition"], as_index=False).agg(
        requests=("request_id", "size"), delivered=("outcome", lambda x: int(x.eq("DELIVERED").sum())),
        strict_usable=("strict_usable", "sum"), infra_unresolved=("outcome", lambda x: int(x.ne("DELIVERED").sum())),
        output_limit=("completion_kind", lambda x: int(x.eq("output_limit").sum())),
        refusal=("completion_kind", lambda x: int(x.eq("provider_refusal").sum())),
        invalid=("strict_usable", lambda x: int((~x.astype(bool)).sum())),
    )
    policy_summary.to_csv(OUTPUT / "main_policy_quality_summary.csv", index=False, encoding="utf-8-sig")

    token_rows = []
    all_rows = parsed
    for (provider, condition, generation_source), group in pd.concat([
        all_rows[["provider", "condition", "generation_source", "input_tokens", "output_tokens_including_reasoning", "request_id"]],
        q[["provider", "condition", "generation_source", "input_tokens", "output_tokens_including_reasoning", "request_id"]],
    ], ignore_index=True).groupby(["provider", "condition", "generation_source"], sort=True):
        inp = int(group.input_tokens.sum())
        out = int(group.output_tokens_including_reasoning.sum())
        cost = inp / 1_000_000 * RATES[provider]["input"] + out / 1_000_000 * RATES[provider]["output"]
        token_rows.append({"provider": provider, "condition": condition, "generation_source": generation_source,
                           "requests": len(group), "input_tokens": inp, "output_tokens_including_reasoning": out,
                           "estimated_batch_cost_usd": round(cost, 6),
                           "input_rate_per_million": RATES[provider]["input"], "output_rate_per_million": RATES[provider]["output"]})
    pd.DataFrame(token_rows).to_csv(OUTPUT / "main_cost_actual_tokens.csv", index=False, encoding="utf-8-sig")

    unresolved = int(parsed.outcome.ne("DELIVERED").sum())
    q_unresolved = int((~q.delivered.fillna(False)).sum())
    model_mismatches = int((~parsed.returned_model_ok.fillna(False)).sum() + (~q.returned_model_ok.fillna(False)).sum())
    report = {
        "status": "RESPONSE_INTEGRITY_COMPLETE" if unresolved == 0 and q_unresolved == 0 and model_mismatches == 0 else "INFRA_RETRY_REVIEW_REQUIRED",
        "scores_computed": False,
        "logical_requests": len(manifest), "policy_requests": len(parsed), "q_requests": len(q),
        "new_api_requests": len(new_ids), "new_api_response_ids": len(observed_new),
        "pilot_reused_requests": len(pilot_ids), "locked_main_reuse_requests": len(source_ids),
        "response_files_hashed": len(file_hashes), "unresolved_policy_infrastructure": unresolved,
        "unresolved_q_infrastructure": q_unresolved,
        "returned_model_mismatches": model_mismatches,
        "strict_policy_invalid_model_outputs": int((~parsed.strict_usable.astype(bool) & parsed.outcome.eq("DELIVERED")).sum()),
        "q_schema_valid": int(q.json_schema_valid.fillna(False).sum()),
        "q_direction_amount_internal_mismatched_metrics": int(q.direction_sign_mismatch_count.sum()),
        "t_rt_table_hash_mismatches": len(duplicate_table_mismatch),
        "request_manifest_counts": {f"{a}|{b}": int(n) for (a, b), n in counts.items()},
        "provider_job_summary_at_utc": summary.get("at_utc"),
        "raw_file_hashes": file_hashes,
        "cost_csv": "results/main_cost_actual_tokens.csv",
        "scoring_gate": "No α/β/γ scores created or read by this script. Start scoring only when status is RESPONSE_INTEGRITY_COMPLETE.",
    }
    (E2 / "logs_main/main_response_integrity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "logical_requests", "policy_requests", "q_requests", "new_api_requests",
                                             "unresolved_policy_infrastructure", "unresolved_q_infrastructure",
                                             "returned_model_mismatches",
                                             "strict_policy_invalid_model_outputs", "q_schema_valid",
                                             "q_direction_amount_internal_mismatched_metrics", "response_files_hashed"]}, ensure_ascii=False, indent=2))


def parse_raw_action(proc: Any, design: Any, raw: str, completion_kind: str | None) -> tuple[bool, dict[str, float], list[str]]:
    from credit_recourse.final_release.parsing import parse_policy_response
    parsed = parse_policy_response(design, raw_visible_text=raw, mode="free8", budget="B1",
                                   provider_refusal=completion_kind == "provider_refusal",
                                   output_limit=completion_kind == "output_limit")
    action = {k: float(parsed.a_itt.get(k, 0.0)) for k in [
        "growth_capex_reduction_pct", "deleveraging_total_debt_pct", "refinancing_short_debt_pct",
        "inv_turnover_chg", "ar_turnover_chg", "ap_turnover_chg", "cogs_ratio_chg", "sga_ratio_chg"]}
    return bool(parsed.policy_usable), action, list(parsed.error_codes)


if __name__ == "__main__":
    main()
