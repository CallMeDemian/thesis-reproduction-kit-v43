from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FreshRuntimePaths:
    root: Path
    run_id: str
    run_root: Path
    raw_root: Path
    oracle_root: Path
    stage2_root: Path
    rl_encoder_root: Path
    rl_bc_root: Path
    rl_iql_root: Path
    c3e_root: Path
    stage6_root: Path
    llm_root: Path
    stage8_root: Path
    stage9_root: Path
    results_root: Path
    thesis_outputs_root: Path
    comparison_root: Path
    release_root: Path
    logs_root: Path

    @classmethod
    def from_run(cls, root: Path, run_id: str, *, create: bool = True):
        run_root = Path(root) / "runs" / run_id
        values = {"root": Path(root), "run_id": run_id, "run_root": run_root,
                  "raw_root": run_root / "01_inputs", "oracle_root": run_root / "02_oracle",
                  "stage2_root": run_root / "03_stage2", "rl_encoder_root": run_root / "04_rl_encoder",
                  "rl_bc_root": run_root / "05_rl_bc", "rl_iql_root": run_root / "06_rl_iql",
                  "c3e_root": run_root / "07_c3e", "stage6_root": run_root / "08_stage6",
                  "llm_root": run_root / "09_llm",
                  "stage8_root": run_root / "10_stage8", "stage9_root": run_root / "11_stage9",
                  "results_root": run_root / "12_results", "thesis_outputs_root": run_root / "13_thesis_outputs",
                  "comparison_root": run_root / "14_comparison",
                  "release_root": run_root / "15_release", "logs_root": run_root / "logs"}
        result = cls(**values)
        if create:
            for path in values.values():
                if isinstance(path, Path):
                    path.mkdir(parents=True, exist_ok=True)
        return result

    def as_dict(self):
        return {name: str(value).replace("\\", "/") for name, value in self.__dict__.items() if name != "root"}
