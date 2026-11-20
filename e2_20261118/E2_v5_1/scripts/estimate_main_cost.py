from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[3]
E2 = WORKSPACE / "e2_20261118/E2_v5_1"


def main() -> None:
    with (E2 / "request_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        req_manifest = pd.DataFrame(csv.DictReader(f))
    with (E2 / "results/pilot_cost_actual_tokens.csv").open(encoding="utf-8-sig", newline="") as f:
        rate_rows = {r["provider"]: r for r in csv.DictReader(f)}
    request_counts = {("openai", "T"): 1050, ("openai", "RT"): 1050, ("openai", "Q"): 475,
                      ("google", "T"): 1050, ("google", "RT"): 1050, ("google", "Q"): 475}
    estimates = []
    provider_totals = defaultdict(float)
    provider_variance = defaultdict(float)
    output_cap_scenario = defaultdict(float)
    for provider in ["openai", "google"]:
        condition_by_id = dict(zip(req_manifest.loc[req_manifest.provider == provider, "request_id"],
                                   req_manifest.loc[req_manifest.provider == provider, "condition"]))
        usage = defaultdict(list)
        path = E2 / f"responses/pilot_{provider}_attempt1_output.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rid = obj.get("custom_id") if provider == "openai" else obj.get("key")
                cond = condition_by_id[rid]
                response = obj.get("response") or {}
                if provider == "openai":
                    meta = (response.get("body") or {}).get("usage") or {}
                    input_tokens = int(meta.get("input_tokens", 0))
                    output_tokens = int(meta.get("output_tokens", 0))
                else:
                    meta = response.get("usageMetadata") or {}
                    input_tokens = int(meta.get("promptTokenCount", 0))
                    output_tokens = int(meta.get("candidatesTokenCount", 0)) + int(meta.get("thoughtsTokenCount", 0))
                usage[cond].append((input_tokens, output_tokens))

        rate = rate_rows[provider]
        in_rate = float(rate["input_rate_per_million"])
        out_rate = float(rate["output_rate_per_million"])
        for condition in ["T", "RT", "Q"]:
            samples = usage[condition]
            if len(samples) != 100:
                raise ValueError(f"Expected 100 pilot usage records for {provider}/{condition}; found {len(samples)}")
            costs = [i * in_rate / 1_000_000 + o * out_rate / 1_000_000 for i, o in samples]
            count = request_counts[(provider, condition)]
            mean_input = sum(i for i, _ in samples) / len(samples)
            mean_output = sum(o for _, o in samples) / len(samples)
            mean_cost = sum(costs) / len(costs)
            variance = sum((x - mean_cost) ** 2 for x in costs) / (len(costs) - 1)
            estimate = count * mean_cost
            provider_totals[provider] += estimate
            provider_variance[provider] += count * variance
            output_cap = count * (mean_input * in_rate / 1_000_000 + 32768 * out_rate / 1_000_000)
            output_cap_scenario[provider] += output_cap
            estimates.append({
                "provider": provider, "condition": condition, "pilot_requests": len(samples),
                "new_requests": count, "pilot_mean_input_tokens": mean_input,
                "pilot_mean_output_tokens_including_reasoning": mean_output,
                "input_rate_usd_per_million": in_rate, "output_rate_usd_per_million": out_rate,
                "estimated_cost_usd": estimate,
                "simulated_32768_output_cap_scenario_usd": output_cap,
            })
    total = sum(provider_totals.values())
    se = math.sqrt(sum(provider_variance.values()))
    upper = total + 1.645 * se
    rows = estimates + [
        {"provider": p, "condition": "TOTAL", "pilot_requests": 300, "new_requests": 2575,
         "pilot_mean_input_tokens": "", "pilot_mean_output_tokens_including_reasoning": "",
         "input_rate_usd_per_million": "", "output_rate_usd_per_million": "",
         "estimated_cost_usd": provider_totals[p],
         "simulated_32768_output_cap_scenario_usd": output_cap_scenario[p]}
        for p in ["openai", "google"]
    ]
    rows.append({"provider": "ALL", "condition": "TOTAL", "pilot_requests": 600, "new_requests": 5150,
                 "pilot_mean_input_tokens": "", "pilot_mean_output_tokens_including_reasoning": "",
                 "input_rate_usd_per_million": "", "output_rate_usd_per_million": "",
                 "estimated_cost_usd": total,
                 "simulated_32768_output_cap_scenario_usd": sum(output_cap_scenario.values())})
    pd.DataFrame(rows).to_csv(E2 / "results/main_cost_estimate.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "Pilot mean token usage by provider × condition; pilot Batch rates; no retries included.",
        "point_estimate_usd": total,
        "approx_one_sided_95pct_upper_usd": upper,
        "standard_error_under_independent_call_costs_usd": se,
        "simulated_all_requests_at_output_cap_with_pilot_mean_input_usd": sum(output_cap_scenario.values()),
        "provider_estimates_usd": dict(provider_totals),
        "batch_rates_source": str(E2 / "results/pilot_cost_actual_tokens.csv"),
        "new_request_count": 5150,
        "assumptions": [
            "Policy generation includes reasoning tokens in output usage.",
            "The 32,768-token output scenario is not a hard bill cap and assumes pilot mean input length.",
            "Actual charge may differ because of token variability, rate changes, or infrastructure retries.",
        ],
    }
    (E2 / "results/main_cost_estimate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
