"""F5: E1a, E1b and E1c means with 95% intervals (alpha, beta, gamma; per model) from results/T5_E1.csv.

Usage: python 55_figure_F5.py [--table <T5_E1.csv>] [--out <dir>]   (defaults: results/T5_E1.csv, figures/)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import common as c

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 9})
SYM = {"alpha": "α", "beta": "β", "gamma": "γ"}
ROWS = [("E1a", "E1a  학습된 행동 내용 (E1-RL - E1-OE), 343개 기업"), ("E1b", "E1b  고정 OE를 보여 준 것 (E1-OE - E1-C4R), 575개 기업"),
        ("E1c", "E1c  새 배치의 A3 (E1-RL - E1-C4R), 575개 기업")]


def main(table: Path, out: Path) -> None:
    t = pd.read_csv(table)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    lo, hi = t.ci_low.min() - 0.02, t.ci_high.max() + 0.02
    for ax, model in zip(axes, ("GPT", "GEMINI")):
        y = 0
        ticks, labels = [], []
        for cid, title in ROWS:
            ax.text(lo, -y + 0.45, title, fontsize=8.5, fontweight="bold", va="bottom")
            for oracle in c.ORACLES:
                r = t[(t.contrast_id == cid) & (t.model == model) & (t.oracle == oracle)].iloc[0]
                sig = pd.notna(r.holm_p) and r.holm_p < 0.05
                ax.plot([r.ci_low, r.ci_high], [-y, -y], color="black", lw=1.3)
                ax.plot(r.estimate, -y, "o", color="black" if sig else "white", mec="black", ms=6)
                ax.text(hi + 0.005, -y, f"{r.estimate:+.3f}  Holm p={'N/A' if pd.isna(r.holm_p) else format(r.holm_p, '.3f')}", va="center", fontsize=7.5)
                ticks.append(-y)
                labels.append(SYM[oracle])
                y += 1
            y += 1.0
        ax.set_ylim(-y + 0.6, 1.3)
        ax.axvline(0, color="#8c8c8c", ls="--", lw=0.9)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_xlim(lo, hi)
        ax.set_title({"GPT": "GPT-5.4-mini", "GEMINI": "Gemini 3.1 Flash-Lite"}[model], loc="left", fontweight="bold")
        ax.set_xlabel("정책가치 차이 [95% 신뢰구간]")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.01, -0.03, "탐색적 실험(새 배치, 자유 작성 · High · B1 · Strict ITT). 채운 점: 해당 묶음 Holm p < .05. 구간은 보정 전. 343개 값을 575개 평균처럼 읽지 않는다.", fontsize=8)
    fig.subplots_adjust(right=0.86, wspace=0.55)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "F5_E1.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "F5_E1.svg", bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=c.RESULTS / "T5_E1.csv")
    ap.add_argument("--out", type=Path, default=c.FIGURES)
    a = ap.parse_args()
    main(a.table, a.out)
    print("F5 written to", a.out)
