"""Single source of truth for the fresh scientific DAG."""
from __future__ import annotations

STAGES = {
    "OracleClean": ["VerifyInputs", "Oracle", "VerifyOracle"],
    "OracleRLClean": ["VerifyInputs", "Oracle", "VerifyOracle", "Simulator", "RLDataset", "RLExecutionGate", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL"],
    "OracleRLLLMClean": ["VerifyInputs", "Oracle", "VerifyOracle", "Simulator", "RLDataset", "RLExecutionGate", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL", "LLMPrepare", "LLMGenerate", "LLMMaterialize", "Stage8", "Stage9", "VerifyResults"],
    "FullClean": ["VerifyInputs", "Oracle", "VerifyOracle", "Simulator", "RLDataset", "RLExecutionGate", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL", "LLMPrepare", "LLMGenerate", "LLMMaterialize", "Stage8", "Stage9", "VerifyResults", "ThesisOutputs", "CompareFrozen", "VerifyAll"],
}

STAGE_DIRS = {
    "VerifyInputs": "01_inputs", "Oracle": "02_oracle", "VerifyOracle": "02_oracle",
    "Simulator": "03_simulator", "RLDataset": "04_rl_dataset", "RLEncoder": "05_rl_encoder",
    "RLExecutionGate": "05_rl_encoder", "RLBehaviorClone": "06_rl_bc", "RLIQL": "07_rl_iql",
    "C3E": "08_c3e", "Stage6": "08_c3e", "VerifyRL": "08_c3e", "LLMPrepare": "09_llm",
    "LLMGenerate": "09_llm", "LLMMaterialize": "09_llm", "Stage8": "10_stage8",
    "Stage9": "11_stage9", "VerifyResults": "12_results", "ThesisOutputs": "13_thesis_outputs",
    "CompareFrozen": "14_comparison", "VerifyAll": "15_release",
}

RUN_DIRS = ["00_run", *[f"{i:02d}_{name}" for i, name in enumerate(("inputs", "oracle", "simulator", "rl_dataset", "rl_encoder", "rl_bc", "rl_iql", "c3e", "llm", "stage8", "stage9", "results", "thesis_outputs", "comparison", "release"), start=1)], "logs"]

def stages_for(mode: str) -> list[str]:
    try:
        return list(STAGES[mode])
    except KeyError as exc:
        raise ValueError(f"unknown mode: {mode}") from exc

def stage_dependencies(mode: str, stage: str) -> list[str]:
    stages = stages_for(mode)
    if stage not in stages:
        raise ValueError(f"stage {stage!r} is not part of mode {mode!r}")
    return stages[:stages.index(stage)]
