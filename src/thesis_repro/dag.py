"""Single source of truth for the fresh scientific DAG."""
from __future__ import annotations

STAGES = {
    "OracleClean": ["VerifyInputs", "Oracle", "VerifyOracle"],
    "OracleRLClean": ["VerifyInputs", "Oracle", "VerifyOracle", "Stage2", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL"],
    "OracleRLLLMClean": ["VerifyInputs", "Oracle", "VerifyOracle", "Stage2", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL", "LLMPrepare", "LLMGenerate", "LLMMaterialize", "Stage8", "Stage9", "VerifyResults"],
    "FullClean": ["VerifyInputs", "Oracle", "VerifyOracle", "Stage2", "RLEncoder", "RLBehaviorClone", "RLIQL", "C3E", "Stage6", "VerifyRL", "LLMPrepare", "LLMGenerate", "LLMMaterialize", "Stage8", "Stage9", "VerifyResults", "ThesisOutputs", "CompareFrozen", "VerifyAll"],
}

STAGE_DIRS = {
    "VerifyInputs": "01_inputs", "Oracle": "02_oracle", "VerifyOracle": "02_oracle",
    "Stage2": "03_stage2", "RLEncoder": "04_rl_encoder", "RLBehaviorClone": "05_rl_bc", "RLIQL": "06_rl_iql",
    "C3E": "07_c3e", "Stage6": "08_stage6", "VerifyRL": "07_c3e", "LLMPrepare": "09_llm",
    "LLMGenerate": "09_llm", "LLMMaterialize": "09_llm", "Stage8": "10_stage8",
    "Stage9": "11_stage9", "VerifyResults": "12_results", "ThesisOutputs": "13_thesis_outputs",
    "CompareFrozen": "14_comparison", "VerifyAll": "15_release",
}

RUN_DIRS = ["00_run", "01_inputs", "02_oracle", "03_stage2", "04_rl_encoder", "05_rl_bc", "06_rl_iql", "07_c3e", "08_stage6", "09_llm", "10_stage8", "11_stage9", "12_results", "13_thesis_outputs", "14_comparison", "15_release", "logs"]

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
