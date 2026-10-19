"""Figures F1-F4: monochrome, Korean font, PNG (300 dpi) and SVG. Data come from the T1-T4 result files."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import common as c

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "svg.fonttype": "none",
                     "axes.edgecolor": "black", "text.color": "black", "axes.labelcolor": "black",
                     "xtick.color": "black", "ytick.color": "black", "font.size": 9})
GREY = "#8c8c8c"
LIGHT = "#e6e6e6"
ROWS = [("GPT", "BASELINE"), ("GPT", "HIGH"), ("GEMINI", "BASELINE"), ("GEMINI", "HIGH")]
LABEL = {"GPT": "GPT-5.4-mini", "GEMINI": "Gemini 3.1 Flash-Lite", "BASELINE": "Baseline", "HIGH": "High"}


def save(fig, name):
    c.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(c.FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(c.FIGURES / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def f1():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, dashed=False, bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec="black",
                           lw=1.4 if bold else 1.0, ls="--" if dashed else "-")
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, fontweight="bold" if bold else "normal")
        return (x, y, w, h)

    c4 = box(0.3, 2.35, 2.0, 1.1, "C4\n직접 생성", bold=True)
    c5 = box(0.3, 0.3, 2.0, 1.1, "C5\n진단 유도 생성\n(C4와 독립)")
    c4r = box(4.2, 4.3, 2.4, 1.1, "C4R\n참조 없는 재검토")
    c6e = box(4.2, 2.35, 2.4, 1.1, "C6-E\nC3-E 행동을 참조로 제시")
    c6x = box(4.2, 0.4, 2.4, 1.1, "C6-EX\n다른 기업의 C3-E 행동을\n참조로 제시")
    c3e = box(7.8, 2.35, 1.9, 1.1, "C3-E\nRL 정책의 행동\n(고정, 외부 기준선)", dashed=True)

    def arrow(p, q, style="-"):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black", ls=style, shrinkA=0, shrinkB=0))

    arrow((2.3, 3.2), (4.2, 4.85))     # C4 -> C4R
    arrow((2.3, 2.9), (4.2, 2.9))      # C4 -> C6-E
    arrow((2.3, 2.6), (4.2, 0.95))     # C4 -> C6-EX
    arrow((7.8, 2.9), (6.6, 2.9), "--")  # C3-E reference -> C6-E
    arrow((8.2, 2.35), (6.6, 1.1), "--")  # C3-E reference -> C6-EX

    def contrast(p, q, label, off=(0, 0)):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="<->", lw=1.0, color=GREY, ls=":", shrinkA=0, shrinkB=0,
                                                         connectionstyle="arc3,rad=0.25"))
        mx, my = (p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=GREY, lw=0.8))

    contrast((1.3, 1.4), (1.3, 2.35), "A1", (-0.55, 0))              # C5 vs C4
    contrast((1.0, 3.45), (4.2, 5.15), "A2", (0.40, -0.52))          # C4R vs C4 (arc above the process arrow)
    contrast((5.4, 4.3), (5.4, 3.45), "A3", (0.5, 0))                 # C6-E vs C4R
    contrast((5.4, 2.35), (5.4, 1.5), "A4", (0.5, 0))                 # C6-E vs C6-EX
    ax.text(0.2, 5.85, "실선 화살표: 같은 C4 초안을 받아 수정", fontsize=8, va="center")
    ax.text(0.2, 5.6, "점선 화살표: 참조로 제시되는 C3-E 행동", fontsize=8, va="center")
    ax.text(0.2, 5.35, "회색 점선 양방향 표시: 대비(처치 - 대조)", fontsize=8, va="center")
    ax.text(0.2, 5.1, "(대비는 화살표가 아니다)", fontsize=8, va="center")
    save(fig, "F1_structure")


def forest(df, contrasts, name, titles):
    fig, axes = plt.subplots(len(contrasts), 1, figsize=(8.2, 2.6 * len(contrasts)), sharex=True)
    axes = np.atleast_1d(axes)
    lo = min(df.ci_low.min(), 0) - 0.03
    hi = df.ci_high.max() + 0.03
    for ax, cid, title in zip(axes, contrasts, titles):
        s = df[(df.contrast_id == cid) & (df.budget == "B1") & (df.oracle == "alpha")]
        ys = list(range(len(ROWS)))[::-1]
        for y, (m, r) in zip(ys, ROWS):
            row = s[(s.model == m) & (s.reasoning_regime == r)].iloc[0]
            ax.plot([row.ci_low, row.ci_high], [y, y], color="black", lw=1.4)
            ax.plot(row.estimate, y, "o", color="black" if row.holm_p < 0.05 else "white", mec="black", ms=6)
            txt = f"{row.estimate:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]"
            p60 = f"  60개 묶음 p={row.holm_p_60:.3f}" if r == "HIGH" and not pd.isna(row.holm_p_60) else ""
            ax.text(hi + 0.01, y, f"{txt}   주 Holm p={row.holm_p:.3f}{p60}", va="center", fontsize=7.5)
        ax.axvline(0, color=GREY, ls="--", lw=0.9)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{LABEL[m]} · {LABEL[r]}" for m, r in ROWS])
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlim(lo, hi)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("정책가치 차이 [95% 신뢰구간]  (자유 작성 · B1 · Oracle-α · Strict ITT · 575개 기업)")
    fig.text(0.01, -0.02, "채운 점: 주 보정 Holm p < .05. 구간은 보정 전이다. 60개 묶음 p는 High 설정의 추가 보정값이다.", fontsize=8)
    fig.subplots_adjust(right=0.58, hspace=0.45)
    save(fig, name)


def f2(t1):
    forest(t1, ["A1", "A2"], "F2a_forest_A1A2", ["A1  진단 유도 (C5 - C4)", "A2  자기 재검토 (C4R - C4)"])
    forest(t1, ["A3", "A4"], "F2b_forest_A3A4", ["A3  참조정보 추가 패키지의 증분 (C6-E - C4R)", "A4  참조 연결 (C6-E - C6-EX)"])


def f3(t1):
    combos = [(m, r, b, o) for m in ("GPT", "GEMINI") for r in ("BASELINE", "HIGH") for b in ("B1", "BINF") for o in c.ORACLES]
    cids = ["A1", "A2", "A3", "A4"]
    fig, ax = plt.subplots(figsize=(15, 3.6))
    for i, cid in enumerate(cids):
        for j, (m, r, b, o) in enumerate(combos):
            row = t1[(t1.contrast_id == cid) & (t1.model == m) & (t1.reasoning_regime == r) & (t1.budget == b) & (t1.oracle == o)].iloc[0]
            pos = row.estimate > 0
            sig = row.holm_p < 0.05
            ax.add_patch(plt.Rectangle((j, len(cids) - 1 - i), 1, 1, fc=LIGHT if pos else "white", ec="black", lw=1.6 if sig else 0.4))
            mark = "*" if sig else ""
            if r == "HIGH":
                mark += "†" if row.holm_p_60 < 0.05 else ""
            ax.text(j + 0.5, len(cids) - 1 - i + 0.5, f"{row.estimate:+.2f}{mark}", ha="center", va="center", fontsize=7)
    ax.set_xlim(0, len(combos))
    ax.set_ylim(0, len(cids))
    ax.set_yticks([len(cids) - 1 - i + 0.5 for i in range(len(cids))])
    ax.set_yticklabels(["A1 진단 유도", "A2 자기 재검토", "A3 참조 제시", "A4 참조 연결"])
    ax.set_xticks([j + 0.5 for j in range(len(combos))])
    ax.set_xticklabels([f"{m[:3]}\n{'Base' if r == 'BASELINE' else 'High'}\n{b}\n{ {'alpha': 'α', 'beta': 'β', 'gamma': 'γ'}[o] }" for m, r, b, o in combos], fontsize=7)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.01, -0.04, "회색 칸: 평균 차이 양수, 흰 칸: 음수. 굵은 테두리와 *: 주 보정 Holm p < .05. †: High 설정에서 60개 묶음 보정으로도 p < .05. "
                          "Strict ITT, 자유 작성, 575개 기업. Oracle 사이 크기는 비교하지 않는다.", fontsize=8)
    save(fig, "F3_heatmap")


def f4(t1, supp):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    ys = list(range(len(ROWS)))[::-1]
    rep = supp[(supp.holm_family_id == "REPAIRED_SUPPLEMENTAL") & (supp.action_space == "FREE8") & (supp.budget == "B1") & (supp.oracle == "ALPHA")]
    stab = supp[(supp.holm_family_id == "STABILITY_SUPPLEMENTAL") & (supp.execution_policy == "STRICT") & (supp.budget == "B1") & (supp.oracle == "ALPHA")]

    def draw(ax, rows, title, labels):
        for y, (m, r) in zip(ys, ROWS):
            for k, (lab, getter, dy, fill) in enumerate(labels):
                row = getter(m, r)
                yy = y + dy
                ax.plot([row[1], row[2]], [yy, yy], color="black", lw=1.2)
                ax.plot(row[0], yy, "o" if fill else "s", color="black" if fill else "white", mec="black", ms=5.5)
        ax.axvline(0, color=GREY, ls="--", lw=0.9)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{LABEL[m]}\n{LABEL[r]}" for m, r in ROWS], fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    def strict(cid):
        return lambda m, r: tuple(t1[(t1.contrast_id == cid) & (t1.model == m) & (t1.reasoning_regime == r) & (t1.budget == "B1") & (t1.oracle == "alpha")].iloc[0][["estimate", "ci_low", "ci_high"]])

    def repaired(lhs, rhs):
        def g(m, r):
            x = rep[(rep.model == m) & (rep.reasoning_regime == r) & (rep.lhs == lhs) & (rep.rhs == rhs)]
            if len(x) != 1:
                raise SystemExit(f"F4: repaired row missing {m}/{r}/{lhs}/{rhs}")
            return tuple(x.iloc[0][["estimate", "ci_low", "ci_high"]])
        return g

    def run2(lhs, rhs):
        def g(m, r):
            x = stab[(stab.model == m) & (stab.reasoning_regime == r) & (stab.lhs == lhs) & (stab.rhs == rhs) & (stab.replicate == 2)]
            if len(x) != 1:
                raise SystemExit(f"F4: run2 row missing {m}/{r}")
            return tuple(x.iloc[0][["estimate", "ci_low", "ci_high"]])
        return g

    draw(axes[0], None, "A2  Strict(●)와 Repaired(□)", [("Strict", strict("A2"), 0.14, True), ("Repaired", repaired("C4R", "C4"), -0.14, False)])
    draw(axes[1], None, "A3  Strict(●)와 Repaired(□)", [("Strict", strict("A3"), 0.14, True), ("Repaired", repaired("C6-E", "C4R"), -0.14, False)])
    draw(axes[2], None, "A3  Run1(●)과 Run2(□), Strict", [("Run1", strict("A3"), 0.14, True), ("Run2", run2("C6-E", "C4R"), -0.14, False)])
    for ax in axes:
        ax.set_xlabel("정책가치 차이 [95% 신뢰구간]")
    fig.text(0.01, -0.06, "자유 작성 · B1 · Oracle-α · 575개 기업 · 구간은 보정 전. Repaired는 같은 응답을 다른 규칙으로 처리한 것이며 의도의 복원으로 읽지 않는다.", fontsize=8)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "F4_robustness")


def main():
    t1 = pd.read_csv(c.RESULTS / "T1_primary_A1_A4.csv")
    supp = pd.read_csv(c.SUPP_PATH)
    f1()
    f2(t1)
    f3(t1)
    f4(t1, supp)


if __name__ == "__main__":
    main()
    print(sorted(p.name for p in c.FIGURES.iterdir()))
