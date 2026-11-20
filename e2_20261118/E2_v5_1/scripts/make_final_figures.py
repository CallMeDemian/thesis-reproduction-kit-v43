from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


E2 = Path(__file__).resolve().parents[1]
RESULTS = E2 / "results"


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

    condition_rows = [r for r in rows("main_condition_values.csv")
                      if r["model"] == "ALL" and r["evaluator"] == "alpha"]
    by_condition = {r["condition"]: r for r in condition_rows}
    order = ["N", "R", "T", "RT"]
    vals = [as_float(by_condition[c], "policy_value_vs_A0") for c in order]
    lows = [as_float(by_condition[c], "ci95_low") for c in order]
    highs = [as_float(by_condition[c], "ci95_high") for c in order]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.bar(order, vals, color=["#7d91a6", "#567a9d", "#c8a456", "#628b72"])
    ax.errorbar(range(4), vals, yerr=[[v-l for v, l in zip(vals, lows)], [h-v for v, h in zip(vals, highs)]],
                fmt="none", ecolor="#222", capsize=4)
    ax.set_ylabel("α policy value vs A0")
    ax.set_title("E2 policy value (575 firms; two models and repeats averaged)")
    ax.axhline(0, color="#555", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS / "main_2x2_alpha.png", dpi=180)
    plt.close(fig)

    comp_rows = [r for r in rows("main_r8_policy_decomposition.csv")
                 if r["source"] == "main" and r["model_key"] == "ALL" and r["evaluator"] == "alpha"]
    comp = {r["policy"]: r for r in comp_rows}
    cvals = [as_float(comp[c], "C") for c in order]
    avals = [as_float(comp[c], "A") for c in order]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.bar(order, cvals, label="Composition", color="#d0a657")
    ax.bar(order, avals, bottom=cvals, label="Assignment", color="#618971")
    ax.set_ylabel("α value change vs A0")
    ax.set_title("E2 composition and assignment decomposition")
    ax.legend(frameon=False)
    ax.axhline(0, color="#555", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS / "main_composition_assignment.png", dpi=180)
    plt.close(fig)

    stress_rows = [r for r in rows("main_sensitivity_contrasts.csv")
                   if r["model"] == "ALL" and r["evaluator"] == "alpha" and r["contrast"] == "RT_minus_R"]
    stress = {r["scenario"]: r for r in stress_rows}
    s_order = ["S0", "S1", "S2"]
    svals = [as_float(stress[s], "difference") for s in s_order]
    sl = [as_float(stress[s], "ci95_low") for s in s_order]
    sh = [as_float(stress[s], "ci95_high") for s in s_order]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.bar(s_order, svals, color=["#628b72", "#b8845a", "#8d7ca8"])
    ax.errorbar(range(3), svals, yerr=[[v-l for v, l in zip(svals, sl)], [h-v for v, h in zip(svals, sh)]],
                fmt="none", ecolor="#222", capsize=4)
    ax.set_ylabel("α RT−R")
    ax.set_title("Fixed-action simulator sensitivity")
    ax.axhline(0, color="#555", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS / "main_stress_sensitivity.png", dpi=180)
    plt.close(fig)

    print("Created main_2x2_alpha.png, main_composition_assignment.png, main_stress_sensitivity.png")


if __name__ == "__main__":
    main()
