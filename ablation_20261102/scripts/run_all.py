"""Run the ablation pipeline in order: inputs -> T1 -> T2 -> T3 -> T4/R9/R10 -> figures -> QA.

Usage (kit root, Python 3.12 with the kit's pinned packages, PYTHONUTF8=1):
    python ablation_20261102/scripts/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["00_inputs.py", "10_primary_T1.py", "20_s1_T2.py", "30_ref1_T3.py", "40_robust_T4.py", "50_figures.py"]


def main() -> int:
    for step in STEPS:
        print(f"== {step}", flush=True)
        r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / step)], cwd=HERE)
        if r.returncode:
            print(f"STOP: {step} failed", flush=True)
            return r.returncode
    print("== 90_qa.py", flush=True)
    return subprocess.run([sys.executable, "-X", "utf8", str(HERE / "90_qa.py"), "--report", str(HERE.parent / "QA_REPORT.md")], cwd=HERE).returncode


if __name__ == "__main__":
    sys.exit(main())
