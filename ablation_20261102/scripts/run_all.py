"""Run the ablation pipeline in order: inputs -> T1 -> primary stream check -> T2 -> T3 -> T4/R9/R10 -> figures -> (E1 outputs) -> QA.

Usage (kit root, Python 3.12 with the kit's pinned packages, PYTHONUTF8=1):
    python ablation_20261102/scripts/run_all.py

The E1 outputs (F5, descriptive tables) need results/T5_E1.csv, which comes from
E1/scripts/e1_observations.py followed by E1/scripts/e1_analyze.py once every E1 request is collected.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["00_inputs.py", "10_primary_T1.py", "15_primary_stream_check.py", "20_s1_T2.py", "30_ref1_T3.py", "40_robust_T4.py", "50_figures.py"]
POST_E1 = ["55_figure_F5.py", "60_descriptive.py"]


def run(step: str) -> int:
    print(f"== {step}", flush=True)
    return subprocess.run([sys.executable, "-X", "utf8", str(HERE / step)], cwd=HERE).returncode


def main() -> int:
    steps = list(STEPS)
    if (HERE.parent / "results" / "T5_E1.csv").exists():
        steps += POST_E1
    for step in steps:
        if run(step):
            print(f"STOP: {step} failed", flush=True)
            return 1
    print("== 90_qa.py", flush=True)
    return subprocess.run([sys.executable, "-X", "utf8", str(HERE / "90_qa.py"), "--report", str(HERE.parent / "QA_REPORT.md")], cwd=HERE).returncode


if __name__ == "__main__":
    sys.exit(main())
