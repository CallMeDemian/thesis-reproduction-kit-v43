from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_pilot as prep


OUT = prep.OUT
E1 = prep.E1
CONTRACT = prep.KIT / "archive/DEPLOYED_RELEASE/stage2_candidate_projection/candidate_action_contract_v4_3.json"
RATES = {
    "openai": {"input": 0.375, "output": 2.25},
    "google": {"input": 0.125, "output": 0.75},
}
PRICE_SOURCES = {
    "openai_standard": "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
    "openai_batch_discount": "https://developers.openai.com/api/docs/guides/batch",
    "google_batch": "https://ai.google.dev/gemini-api/docs/pricing?hl=en",
}


def load_e1_usage() -> tuple[dict[str, dict], dict[str, dict]]:
    usage: dict[str, dict] = {}
    with (E1 / "runs/e1_selected.jsonl").open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            u = x.get("usage") or {}
            if x["provider"] == "openai":
                input_tokens = int(u.get("input_tokens", 0))
                output_tokens = int(u.get("output_tokens", 0))
            else:
                input_tokens = int(u.get("promptTokenCount", 0))
                output_tokens = int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0))
            usage[x["request_id"]] = {"provider": x["provider"], "input_tokens": input_tokens,
                                      "output_tokens": output_tokens, "returned_model": x.get("returned_model")}
    manifest = pd.read_csv(E1 / "requests/E1_request_manifest.csv")
    records = prep.read_jsonl_batches()
    chars: dict[str, dict] = {}
    for row in manifest.itertuples(index=False):
        item = records[row.e1_request_id]
        chars[row.e1_request_id] = {
            "system_chars": len(prep.get_system_text(row.provider, item)),
            "user_chars": len(prep.get_user_text(row.provider, item)),
        }
    return usage, chars


def row_to_source_condition(cond: str) -> str:
    return "E1-RL" if cond in {"R", "RT"} else "E1-C4R"


def main() -> None:
    req = pd.read_parquet(OUT / "request_manifest.parquet")
    sample = pd.read_csv(OUT / "sample_100_firms.csv")
    draft = pd.read_csv(OUT / "draft_action_audit_all_2300.csv")
    all_invalid = pd.read_csv(OUT / "prior_draft_47_cases.csv")
    assert len(req) == 1000 and req.request_id.nunique() == 1000
    assert len(sample) == 100 and len(draft) == 2300 and len(all_invalid) == 47
    assert req.groupby(["provider", "condition"]).size().eq(100).all()
    assert req.exact_model_id.groupby(req.provider).nunique().eq(1).all()
    assert set(req.loc[req.provider == "openai", "exact_model_id"]) == {"gpt-5.4-mini-2026-03-17"}
    assert set(req.loc[req.provider == "google", "exact_model_id"]) == {"gemini-3.1-flash-lite"}

    # Check mixed batch settings and body IDs, then create prompt diff invariants.
    sources: dict[tuple[str, int, str], tuple[str, str, dict]] = {}
    id_seen: set[str] = set()
    for provider in ["openai", "google"]:
        p = OUT / f"requests/{provider}_pilot_mixed.jsonl"
        lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]
        assert len(lines) == 500
        for item in lines:
            if provider == "openai":
                rid = item["custom_id"]
                body = item["body"]
                assert item["method"] == "POST" and item["url"] == "/v1/responses"
                assert body["model"] == "gpt-5.4-mini-2026-03-17"
                assert body["reasoning"] == {"effort": "high"}
                assert body["max_output_tokens"] == 32768
                system = next(x["content"][0]["text"] for x in body["input"] if x.get("role") == "system")
                user = next(x["content"][0]["text"] for x in body["input"] if x.get("role") == "user")
            else:
                rid = item["key"]
                rq = item["request"]
                assert rq["generationConfig"]["maxOutputTokens"] == 32768
                assert rq["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"
                system = rq["systemInstruction"]["parts"][0]["text"]
                user = rq["contents"][0]["parts"][0]["text"]
            assert rid not in id_seen
            id_seen.add(rid)
            m = req.loc[req.request_id == rid].iloc[0]
            sources[(provider, int(m.row_id), str(m.condition))] = (system, user, item)
    assert id_seen == set(req.request_id)

    table_rows_count = 0
    table_hash_pairs = 0
    for provider in ["openai", "google"]:
        for row_id in sample.row_id.astype(int).tolist():
            _, n, _ = sources[(provider, row_id, "N")]
            _, r, _ = sources[(provider, row_id, "R")]
            _, t, _ = sources[(provider, row_id, "T")]
            _, rt, _ = sources[(provider, row_id, "RT")]
            assert "ADDITIONAL REFERENCE PROGRAM" not in n and "ADDITIONAL REFERENCE PROGRAM" not in t
            assert "ADDITIONAL REFERENCE PROGRAM" in r and "ADDITIONAL REFERENCE PROGRAM" in rt
            assert "Below is the change in financial-statement amounts" not in n
            assert "Below is the change in financial-statement amounts" not in r
            assert t.startswith(n + "\n\n") and rt.startswith(r + "\n\n")
            t_suffix = t[len(n) + 2:]
            rt_suffix = rt[len(r) + 2:]
            assert t_suffix == rt_suffix
            assert t_suffix.startswith(prep.TABLE_CAPTION + "\n\n")
            assert sum(line.startswith("| ") for line in t_suffix.splitlines()) == 11  # header + 10 body rows
            assert "Oracle" not in t_suffix and "P3" not in t_suffix and "P5" not in t_suffix
            assert "alpha" not in t_suffix.lower() and "beta" not in t_suffix.lower() and "gamma" not in t_suffix.lower()
            table_rows_count += 10
            # Q must never contain the earlier draft, RL reference, or Simulator result table.
            _, q, _ = sources[(provider, row_id, "Q")]
            assert "ADDITIONAL REFERENCE PROGRAM" not in q
            assert "Below is the change in financial-statement amounts" not in q
            assert "No Simulator result or policy reference is supplied." in q
            # The exact common condition instruction is used in all four revision conditions.
            assert prep.COMMON_TASK in n and prep.COMMON_TASK in r and prep.COMMON_TASK in t and prep.COMMON_TASK in rt
            # The table-payload hash is shared byte-for-byte between T and RT.
            hm = req.query("provider == @provider and row_id == @row_id and condition in ['T','RT']")
            assert hm.table_payload_sha256.nunique() == 1
            table_hash_pairs += 1
    assert table_rows_count == 2000

    # Validate the existing B1 Simulator rows used for the 200 selected firm x model tables.
    b1 = pd.read_parquet(prep.B1_TABLES)
    chosen_ids = set(sample.row_id.astype(int))
    chosen = b1[b1.row_id.isin(chosen_ids) & b1.replicate.eq(1)].copy()
    assert len(chosen) == 2000
    qa = {
        "selected_tables": int(chosen.table_id.nunique()),
        "table_rows": int(len(chosen)),
        "all_action_valid": bool(chosen.action_valid.all()),
        "all_simulator_ok": bool(chosen.simulator_ok.all()),
        "all_currency_KRW_thousand": bool(chosen.currency_unit.eq("KRW_thousand").all()),
        "all_one_year_horizon": bool(chosen.horizon.eq("decision_year_plus_one").all()),
        "max_normalized_action_T": float(chosen.action_T.max()),
        "action_T_exceeds_1_by_more_than_1e-8": int((chosen.action_T > 1.0 + 1e-8).sum()),
        "max_abs_balance_identity_residual_KRW_thousand": float(chosen.identity_error_t1.abs().max()),
        "max_abs_A0_identity_residual_KRW_thousand": float(chosen.identity_error_A0.abs().max()),
        "max_abs_residual_within_10_KRW_thousand_tolerance": bool(chosen.identity_error_t1.abs().max() <= 10.0),
        "financial_effect_status_counts": chosen.financial_effect_status.value_counts(dropna=False).to_dict(),
        "financial_items_missing_count_max": int(chosen.financial_items_missing_count.max()),
        "selected_strict_invalid_draft_table_rows": int(chosen.loc[chosen.row_order.eq(2), "draft_source_strict_usable"].eq(False).sum()),
    }
    if not qa["all_action_valid"] or not qa["all_simulator_ok"] or not qa["all_currency_KRW_thousand"] or not qa["all_one_year_horizon"]:
        raise ValueError("selected B1 table failed basic Simulator contract checks")
    if qa["action_T_exceeds_1_by_more_than_1e-8"] or not qa["max_abs_residual_within_10_KRW_thousand_tolerance"]:
        raise ValueError("selected B1 table failed action budget/accounting tolerance")
    (OUT / "table_integrity.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

    # Approximate cost from selected E1 usage and exact new prompt character lengths.
    e1_usage, e1_chars = load_e1_usage()
    e1_manifest = pd.read_csv(E1 / "requests/E1_request_manifest.csv").set_index("e1_request_id")
    estimates = []
    for row in req.itertuples(index=False):
        source_id = str(row.source_e1_request_id)
        hist = e1_usage[source_id]
        char = e1_chars[source_id]
        source_chars = char["system_chars"] + char["user_chars"]
        ratio = (int(row.input_chars) / source_chars) if source_chars else 1.0
        input_est = int(math.ceil(hist["input_tokens"] * ratio))
        source_condition = str(e1_manifest.loc[source_id, "e1_condition"])
        estimates.append({
            "request_id": row.request_id, "provider": row.provider, "condition": row.condition,
            "historical_source_condition": source_condition, "input_tokens_est": input_est,
            "output_tokens_est": int(hist["output_tokens"]),
        })
    est = pd.DataFrame(estimates)
    # For Q, which has a distinct forecasting task, use the provider's historical 75th-percentile
    # E1-C4R output length as a conservative proxy; do not inspect Simulator answers.
    q_proxy = {}
    e1_manifest_df = pd.read_csv(E1 / "requests/E1_request_manifest.csv").set_index("e1_request_id")
    for provider in ["openai", "google"]:
        ids = e1_manifest_df[(e1_manifest_df.provider == provider) & (e1_manifest_df.e1_condition == "E1-C4R")].index
        vals = [e1_usage[x]["output_tokens"] for x in ids if x in e1_usage]
        q_proxy[provider] = int(np.quantile(vals, 0.75, method="linear"))
        est.loc[(est.provider == provider) & (est.condition == "Q"), "output_tokens_est"] = q_proxy[provider]
    cost_rows = []
    for provider in ["openai", "google"]:
        g = est[est.provider == provider]
        in_tok = int(g.input_tokens_est.sum())
        out_tok = int(g.output_tokens_est.sum())
        point = in_tok / 1_000_000 * RATES[provider]["input"] + out_tok / 1_000_000 * RATES[provider]["output"]
        # 20% more input tokens, historical p90 output lengths for each source condition.
        p90_by_cond = {}
        for cond in ["E1-C4R", "E1-RL"]:
            ids = e1_manifest_df[(e1_manifest_df.provider == provider) & (e1_manifest_df.e1_condition == cond)].index
            vals = [e1_usage[x]["output_tokens"] for x in ids if x in e1_usage]
            p90_by_cond[cond] = int(np.quantile(vals, 0.90, method="linear"))
        upper_out = 0
        for cond in ["N", "T", "Q"]:
            upper_out += 100 * p90_by_cond["E1-C4R"]
        for cond in ["R", "RT"]:
            upper_out += 100 * p90_by_cond["E1-RL"]
        upper_in = int(math.ceil(in_tok * 1.20))
        upper = upper_in / 1_000_000 * RATES[provider]["input"] + upper_out / 1_000_000 * RATES[provider]["output"]
        max_output = 500 * 32768
        worst = upper_in / 1_000_000 * RATES[provider]["input"] + max_output / 1_000_000 * RATES[provider]["output"]
        cost_rows.append({
            "provider": provider, "requests": int(len(g)), "input_tokens_est": in_tok,
            "output_tokens_est": out_tok, "point_estimate_usd": round(point, 4),
            "conservative_p90_usd": round(upper, 4), "output_cap_single_attempt_usd": round(worst, 4),
            "retry_max_3_attempts_estimate_usd": round(3 * worst, 4),
            "batch_input_usd_per_million": RATES[provider]["input"],
            "batch_output_usd_per_million": RATES[provider]["output"],
            "q_output_proxy_tokens_per_request": q_proxy[provider],
        })
    total = {k: round(sum(row[k] for row in cost_rows), 4) for k in ["point_estimate_usd", "conservative_p90_usd", "output_cap_single_attempt_usd", "retry_max_3_attempts_estimate_usd"]}
    cost = {"estimate_method": "Selected E1 usage; input tokens scaled by exact character-count ratio; T/RT table contexts therefore approximate; Q output proxied by E1-C4R p75. Prices are posted paid-tier Batch rates and not an account billing quote.",
            "rates_source_urls": PRICE_SOURCES, "by_provider": cost_rows, "total": total}
    (OUT / "preflight_cost_estimate.json").write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# E2 v5.1 Pilot Preflight",
        "",
        "## Fixed input checks",
        "",
        f"- Sample: {len(sample)} firms, stratified allocation `{sample.stratum_C3E_action.value_counts().to_dict()}`.",
        f"- Logical requests: {len(req)} = 2 providers × 5 conditions × 100 firms; unique request IDs: {req.request_id.nunique()}.",
        f"- Original C4 drafts strictly parsed: {len(draft)}; strict-invalid: {len(all_invalid)}; selected sample invalid firm/model drafts: {qa['selected_strict_invalid_draft_table_rows']}.",
        f"- T/RT shared table input: {table_hash_pairs} firm/model pairs × 10 rows = {table_rows_count} row appearances; T and RT text tables match byte-for-byte.",
        "- Exact models/settings checked in request JSONL; pre-call metadata check confirms both exact model IDs are available. No generation smoke call was sent.",
        "- New Oracle score calculation/reading is not part of this preflight.",
        "",
        "## Simulator table integrity (selected 100 firms × 2 models)",
        "",
        f"- {qa['selected_tables']} tables / {qa['table_rows']} rows; every action valid and every Simulator call successful.",
        f"- Unit/horizon: all `KRW_thousand`, `decision_year_plus_one`; missing financial items max={qa['financial_items_missing_count_max']}.",
        f"- Largest normalized action budget T={qa['max_normalized_action_T']:.12f}; no value exceeds 1 by more than 1e-8.",
        f"- Largest accounting identity residual: {qa['max_abs_balance_identity_residual_KRW_thousand']:.9g} KRW thousand (Simulator check tolerance 10); A0 residual max={qa['max_abs_A0_identity_residual_KRW_thousand']:.9g}.",
        f"- Financial-effect statuses: `{qa['financial_effect_status_counts']}`. Zero effects are explicitly separate from `NOT_SIMULATED`; selected tables have no unavailable simulation rows.",
        "- Amounts in prompts use signed KRW-thousand changes, rounded to integer thousands (Decimal half-up); raw source floats remain in the frozen B1 parquet.",
        "",
        "## Historical-token cost estimate",
        "",
        "| Provider | Input tokens est. | Output tokens est. | Point estimate | Historical p90 estimate | Single-attempt cap scenario |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in cost_rows:
        lines.append(f"| {row['provider']} | {row['input_tokens_est']:,} | {row['output_tokens_est']:,} | ${row['point_estimate_usd']:.2f} | ${row['conservative_p90_usd']:.2f} | ${row['output_cap_single_attempt_usd']:.2f} |")
    lines += [
        f"| Total | {sum(x['input_tokens_est'] for x in cost_rows):,} | {sum(x['output_tokens_est'] for x in cost_rows):,} | ${total['point_estimate_usd']:.2f} | ${total['conservative_p90_usd']:.2f} | ${total['output_cap_single_attempt_usd']:.2f} |",
        "",
        "Rates: GPT-5.4 mini standard input/output $0.75/$4.50 per million; OpenAI Batch is 50% cheaper, so this estimate uses $0.375/$2.25. Gemini 3.1 Flash-Lite Batch paid-tier text input/output is $0.125/$0.75 per million. Retries are excluded from the point estimate; the three-attempt cap scenario is recorded in `preflight_cost_estimate.json`.",
        "",
        f"Pricing references: [OpenAI model pricing]({PRICE_SOURCES['openai_standard']}), [OpenAI Batch discount]({PRICE_SOURCES['openai_batch_discount']}), [Gemini API pricing]({PRICE_SOURCES['google_batch']}).",
        "",
        "These are token-based paid-tier estimates, not an account invoice. Actual provider usage will be recorded from the batch responses.",
    ]
    (OUT / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Final pre-call artifact hashes: protocol, requests, exact manifest, sampling and audit files.
    freeze_files = [
        OUT / "E2_V5_1_PROTOCOL.md", OUT / "request_manifest.parquet", OUT / "request_manifest.csv",
        OUT / "sample_100_firms.csv", OUT / "draft_action_audit_all_2300.csv", OUT / "prior_draft_47_cases.csv",
        OUT / "parent_prompt_request_links.csv", OUT / "prompt_diff.md", OUT / "prompt_templates/E2_PROMPT_TEMPLATE_RULES.md",
        OUT / "prompt_templates/COMMON_TASK.txt", OUT / "prompt_templates/TABLE_CAPTION.txt",
        OUT / "requests/openai_pilot_mixed.jsonl", OUT / "requests/google_pilot_mixed.jsonl",
        OUT / "preflight_report.md", OUT / "preflight_cost_estimate.json", OUT / "table_integrity.json",
        OUT / "ORACLE_SEAM_CROSSWALK.md", OUT / "STRESS_IMPLEMENTATION.md", OUT / "model_preflight.json",
        OUT / "requirements_api_lock.txt", OUT / "scripts/prepare_pilot.py", OUT / "scripts/preflight_validate.py",
        OUT / "scripts/model_metadata_preflight.py", OUT / "scripts/run_pilot_batch.py", OUT / "source_hashes.csv",
    ]
    hashes = [{"path": str(p.relative_to(prep.WORKSPACE)), "bytes": p.stat().st_size,
               "sha256": prep.sha256_file(p)} for p in freeze_files]
    pd.DataFrame(hashes).to_csv(OUT / "pre_call_hashes.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"preflight": "PASS", "requests": len(req), "strict_invalid_original_C4": len(all_invalid),
                      "selected_draft_invalid": qa["selected_strict_invalid_draft_table_rows"],
                      "cost_estimate_usd": total, "table_integrity": qa,
                      "pre_call_hash_files": len(hashes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
