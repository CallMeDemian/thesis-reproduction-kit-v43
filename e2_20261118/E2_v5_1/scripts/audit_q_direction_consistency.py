from __future__ import annotations

import json
import sys
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

E2 = Path(__file__).resolve().parents[1]
KIT = E2.parents[0] / "A_reproduction/E1_score_gate_replay/runroot_zip/THESIS_REPRO_KIT_v2.1.1_FINAL"
E1_RELEASE = E2.parents[0] / "source_inputs/github_release_ablation_20261102_readonly/contents/ablation_20261102/E1"
MODELS = {"openai": "gpt-5.4-mini-2026-03-17", "google": "gemini-3.1-flash-lite"}


def main() -> None:
    if str(KIT / "src") not in sys.path:
        sys.path.insert(0, str(KIT / "src"))
    from credit_recourse.final_release.providers import normalize_google, normalize_openai
    with (E2 / "request_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        condition_by_id = {row["request_id"]: row["condition"] for row in csv.DictReader(f)}

    totals = Counter()
    by_metric = Counter()
    by_type = Counter()
    for provider, expected in MODELS.items():
        responses = E2 / "responses" / f"pilot_{provider}_attempt1_output.jsonl"
        with responses.open(encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                request_id = raw.get("custom_id") if provider == "openai" else raw.get("key")
                if condition_by_id.get(request_id) != "Q":
                    continue
                norm = normalize_openai(raw, expected) if provider == "openai" else normalize_google(raw, expected)
                if not norm.get("delivered") or str(norm.get("returned_model") or "") != expected:
                    continue
                totals[(provider, "delivered")] += 1
                if norm.get("output_limit"):
                    totals[(provider, "output_limit")] += 1
                    continue
                if norm.get("provider_refusal"):
                    totals[(provider, "provider_refusal")] += 1
                    continue
                text = norm.get("raw_visible_text")
                if not str(text or "").strip():
                    totals[(provider, "delivered_empty")] += 1
                    continue
                try:
                    obj = json.loads(text)
                except Exception:
                    totals[(provider, "invalid_json")] += 1
                    continue
                forecasts = obj.get("forecasts") if isinstance(obj, dict) else None
                if not isinstance(forecasts, list):
                    totals[(provider, "invalid_forecasts")] += 1
                    continue
                totals[(provider, "valid_json")]+=1
                request_mismatches = 0
                for row in forecasts:
                    if not isinstance(row, dict):
                        continue
                    for metric in ("pretax_income", "interest_expense", "total_debt", "cash", "current_liabilities"):
                        value = row.get(metric)
                        if not isinstance(value, dict):
                            continue
                        direction = value.get("direction")
                        amount = value.get("change_thousand_krw")
                        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                            continue
                        violation = None
                        if direction == "increase" and amount <= 0:
                            violation = "increase_not_positive"
                        elif direction == "decrease" and amount >= 0:
                            violation = "decrease_not_negative"
                        elif direction == "no_material_change" and amount != 0:
                            violation = "no_change_nonzero"
                        if violation:
                            request_mismatches += 1
                            by_metric[(provider, metric)] += 1
                            by_type[(provider, violation)] += 1
                if request_mismatches:
                    totals[(provider, "responses_with_direction_amount_mismatch")] += 1
                totals[(provider, "metric_direction_amount_mismatches")] += request_mismatches
    rows = []
    for provider in MODELS:
        for metric in ("pretax_income", "interest_expense", "total_debt", "cash", "current_liabilities"):
            rows.append({"provider": provider, "metric": metric,
                         "direction_amount_mismatch_count": by_metric[(provider, metric)]})
    pd.DataFrame(rows).to_csv(E2 / "results/pilot_q_direction_consistency.csv", index=False, encoding="utf-8-sig")
    output = {
        "scope": "Q output internal direction/sign consistency only; no Simulator, Oracle or actual-answer comparison",
        "totals": {f"{p}.{k}": v for (p, k), v in sorted(totals.items())},
        "mismatch_types": {f"{p}.{k}": v for (p, k), v in sorted(by_type.items())},
    }
    (E2 / "results/pilot_q_direction_consistency.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
