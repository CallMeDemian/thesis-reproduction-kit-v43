"""E1 parse replay gate (runbook Phase 1.6).

1. Replays the September High outputs of the E1 cell (C4R and C6-E, both models, 2,300 rows) through
   ``e1_process.process_line`` and compares ``strict_usable`` and ``strict_itt_action_json`` with the
   frozen canonical observations (STRICT, HIGH, FREE8, B1, IC-b, replicate 1).
2. Feeds synthetic refusal, truncated, empty, malformed and error responses and checks each maps as
   ``apply_itt_rule`` specifies (A0 for completed-but-unusable, retry for infrastructure failure).

Usage (kit root): python ablation_20261102/E1/scripts/e1_parse_replay.py --kit-root . --out <json>
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_process import load_processing_context, process_line  # noqa: E402

HIGH = "frozen/original_release/llm/final_plan3/d5d878bf50718bac65ede73df3fc88c16723870b6c90a790557ff35b063b7ba7"
V13 = "frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN"
PROVIDER = {"openai_gpt54mini": "openai", "google_gemini31flashlite": "google"}


def iter_outputs(root: Path):
    for provider in ("openai", "google"):
        for path in sorted((root / HIGH / "provider_batches").glob(f"wave*/{provider}/r1/shard_*_attempt1_output.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                first = handle.readline()
                if first.startswith("version https://git-lfs"):
                    continue  # LFS pointer (file not pulled); completeness is enforced by the 2,300-row check
                handle.seek(0)
                for text in handle:
                    if text.strip():
                        yield provider, path, json.loads(text)


def replay(root: Path, design) -> dict:
    obs = pd.read_parquet(root / V13 / "frozen_inputs/stage8/canonical_itt_observations.parquet")
    cell = obs[
        (obs.execution_policy == "STRICT") & (obs.reasoning_regime == "HIGH") & (obs.action_space == "FREE8")
        & (obs.budget == "B1") & (obs.information_condition == "IC-b") & (obs.replicate == 1)
        & obs.policy_condition.isin(["C4R", "C6-E"])
    ].set_index("request_id")
    assert len(cell) == 2300, len(cell)
    want = set(cell.index)
    seen: dict[str, int] = {}
    mismatches: list[dict] = []
    stats = {"rows": 0, "usable_match": 0, "action_match": 0, "action_string_match": 0, "model_ok": 0,
             "not_delivered": 0, "unusable_in_september": 0}
    for provider, path, line in iter_outputs(root):
        rid = str(line.get("custom_id") or line.get("key") or "")
        if rid not in want:
            continue
        seen[rid] = seen.get(rid, 0) + 1
        row = cell.loc[rid]
        if PROVIDER[row["model_key"]] != provider:
            mismatches.append({"request_id": rid, "why": "provider mismatch"})
            continue
        res = process_line(provider, line, design)
        stats["rows"] += 1
        if res["outcome"] != "DELIVERED":
            stats["not_delivered"] += 1
            mismatches.append({"request_id": rid, "why": "not delivered", "retry": res["retry_class"]})
            continue
        stats["model_ok"] += bool(res["returned_model_ok"])
        usable_ok = bool(res["strict_usable"]) == bool(row["strict_usable"])
        frozen_action = json.loads(row["strict_itt_action_json"])
        action_ok = res["strict_itt_action"] == frozen_action
        string_ok = json.dumps(res["strict_itt_action"], sort_keys=True, separators=(",", ":")) == row["strict_itt_action_json"]
        stats["usable_match"] += usable_ok
        stats["action_match"] += action_ok
        stats["action_string_match"] += string_ok
        stats["unusable_in_september"] += (not bool(row["strict_usable"]))
        if not (usable_ok and action_ok):
            mismatches.append({"request_id": rid, "why": "value mismatch", "usable_ok": usable_ok, "action_ok": action_ok})
    missing = sorted(want - set(seen))
    duplicated = sorted(r for r, n in seen.items() if n > 1)
    ok = (not mismatches and not missing and not duplicated and stats["rows"] == 2300
          and stats["usable_match"] == 2300 and stats["action_match"] == 2300 and stats["model_ok"] == 2300)
    return {"pass": ok, **stats, "missing_rows": len(missing), "duplicate_rows": len(duplicated),
            "mismatch_count": len(mismatches), "mismatch_examples": mismatches[:5]}


def _openai_line(rid, *, text=None, status=200, body_status="completed", incomplete=None, refusal=None, error=None, model="gpt-5.4-mini-2026-03-17"):
    content = []
    if text is not None:
        content.append({"type": "output_text", "text": text})
    if refusal is not None:
        content.append({"type": "refusal", "refusal": refusal})
    body = {"id": "resp_x", "model": model, "status": body_status,
            "output": [{"type": "message", "role": "assistant", "content": content}] if content else []}
    if incomplete:
        body["incomplete_details"] = {"reason": incomplete}
    line = {"id": "batch_req_x", "custom_id": rid, "response": {"status_code": status, "request_id": "r", "body": body}, "error": error}
    return line


def _google_line(rid, *, text=None, finish="STOP", error=None, model="gemini-3.1-flash-lite"):
    if error is not None:
        return {"key": rid, "error": error}
    parts = [{"text": text}] if text is not None else []
    return {"key": rid, "response": {"candidates": [{"content": {"parts": parts, "role": "model"}, "finishReason": finish}],
                                     "modelVersion": model, "responseId": "g_x"}}


def synthetic(design, good_action: dict) -> dict:
    good = json.dumps({"diagnosis": "d", "action": good_action, "evidence_fields": ["derived__debt_to_assets"], "rationale": "r"})
    bad_bounds = copy.deepcopy(good_action)
    bad_bounds["growth_capex_reduction_pct"] = 9.0
    bad_cap = json.dumps({"diagnosis": "d", "action": bad_bounds, "evidence_fields": [], "rationale": "r"})
    cases = []
    for provider, mk in (("openai", _openai_line), ("google", _google_line)):
        def line(**kw):
            return mk("rid", **kw)
        trunc_kw = {"incomplete": "max_output_tokens", "body_status": "incomplete", "text": good[:40]} if provider == "openai" \
            else {"finish": "MAX_TOKENS", "text": good[:40]}
        refusal_kw = {"refusal": "I cannot help with that."} if provider == "openai" else {"finish": "SAFETY", "text": ""}
        cases += [
            (provider, "valid_action", line(text=good), "DELIVERED", True, "ACCEPTED"),
            (provider, "provider_refusal", line(**refusal_kw), "DELIVERED", False, "CONFIRMED_UNUSABLE_COMPLETION"),
            (provider, "truncated_max_tokens", line(**trunc_kw), "DELIVERED", False, "CONFIRMED_UNUSABLE_COMPLETION"),
            (provider, "empty_text", line(text=""), "DELIVERED", False, "CONFIRMED_UNUSABLE_COMPLETION"),
            (provider, "not_json", line(text="Sorry, here is some prose."), "DELIVERED", False, "CONFIRMED_UNUSABLE_COMPLETION"),
            (provider, "bound_violation", line(text=bad_cap), "DELIVERED", False, "CONFIRMED_UNUSABLE_COMPLETION"),
        ]
        if provider == "openai":
            cases += [
                (provider, "http_500_error", mk("rid", status=500, text=None, error=None), "RETRY", None, "UNRESOLVED_INFRA"),
                (provider, "explicit_error_object", mk("rid", status=0, error={"code": "server_error", "message": "x"}), "RETRY", None, "UNRESOLVED_INFRA"),
                (provider, "http_429", mk("rid", status=429), "RETRY", None, "UNRESOLVED_INFRA"),
            ]
        else:
            cases += [
                (provider, "error_object", _google_line("rid", error={"code": 503, "message": "unavailable"}), "RETRY", None, "UNRESOLVED_INFRA"),
                (provider, "error_429", _google_line("rid", error={"code": 429, "message": "RESOURCE_EXHAUSTED"}), "RETRY", None, "UNRESOLVED_INFRA"),
            ]
    a0 = {k: 0.0 for k in good_action}
    results = []
    for provider, name, ln, outcome, usable, status in cases:
        res = process_line(provider, ln, design)
        ok = res["outcome"] == outcome and res["action_application_status"] == status
        if outcome == "DELIVERED":
            ok = ok and bool(res["strict_usable"]) == usable
            ok = ok and (res["strict_itt_action"] == (good_action if usable else a0))
        else:
            ok = ok and res["strict_itt_action"] is None
        results.append({"provider": provider, "case": name, "outcome": res["outcome"], "status": res["action_application_status"],
                        "completion_kind": res.get("completion_kind"), "pass": ok})
    return {"pass": all(r["pass"] for r in results), "cases": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.kit_root.resolve()
    design = load_processing_context(root)
    rep = replay(root, design)
    good_action = {"growth_capex_reduction_pct": 0.0, "deleveraging_total_debt_pct": 0.0,
                   "refinancing_short_debt_pct": 0.1, "inv_turnover_chg": 0.0, "ar_turnover_chg": 0.0,
                   "ap_turnover_chg": 0.0, "cogs_ratio_chg": 0.0, "sga_ratio_chg": 0.0}
    syn = synthetic(design, good_action)
    result = {"september_replay": rep, "synthetic": syn, "status": "PASS" if rep["pass"] and syn["pass"] else "FAIL"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\r\n")
    print(json.dumps({"status": result["status"], "september_replay": {k: v for k, v in rep.items() if k != "mismatch_examples"},
                      "synthetic_pass": syn["pass"], "synthetic_failed": [c for c in syn["cases"] if not c["pass"]]}, indent=1))
    sys.exit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
