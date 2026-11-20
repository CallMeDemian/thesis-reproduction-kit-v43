from __future__ import annotations

from pathlib import Path
from typing import Any

CANONICAL_STAGE_DIRS: dict[str, str] = {
    "stage0": "stage0_oracle_foundation",
    "stage1_inputs": "stage1_oracle_inputs",
    "stage1_backends": "stage1_oracle_backends",
    "stage2": "stage2_candidate_projection",
    "stage3": "stage3_acd_ssl",
    "stage4": "stage4_candidate_bc",
    "stage5": "stage5_candidate_iql",
    "stage6": "stage6_candidate_selector_eval",
    "stage7": "stage7_llm_action_generation",
    "stage8": "stage8_llm_multi_oracle_eval",
    "stage9": "stage9_llm_rl_comparison",
}

# Current V4.3 runtime artifacts are owned by their formal stage. These paths
# intentionally contain no repair-run or dated development namespace.
V43_STAGE2_RUNTIME_PATH = Path(
    "data/final_freeze/stage2_candidate_projection/v4_3_runtime"
)
V43_HISTORICAL_CONTEXT_PATH = Path(
    "data/final_freeze/stage2_candidate_projection/runtime_inputs/"
    "historical_financial_context_v4_3"
)
V43_BP_RATE_SOURCE_PATH = Path(
    "data/final_freeze/stage2_candidate_projection/runtime_inputs/"
    "business_plan_rate_v4/source"
)
V43_BP_RATE_PRODUCTION_PATH = Path(
    "data/final_freeze/stage2_candidate_projection/runtime_inputs/"
    "business_plan_rate_v4/production"
)
V43_STAGE5_SEARCH_PATH = Path(
    "data/final_freeze/stage5_candidate_iql/search_v4_3"
)
V43_STAGE6_REFERENCE_PATH = Path(
    "data/final_freeze/stage6_candidate_selector_eval/reference_surface_v4_3"
)

def final_root(project_root: Path) -> Path:
    return Path(project_root).resolve() / "data" / "final_freeze"


def stage_dir(project_root: Path, stage_key: str) -> Path:
    name = CANONICAL_STAGE_DIRS.get(stage_key, stage_key)
    return final_root(project_root) / name


def as_manifest(project_root: Path) -> dict[str, Any]:
    root = final_root(project_root)
    return {
        "contract_version": "canonical_stage_paths_v1_runner_paths_are_source_of_truth",
        "final_root": str(root),
        "canonical_stage_dirs": CANONICAL_STAGE_DIRS,
        "policy": "Production runners and stage boundary verifiers use only canonical current-run paths.",
    }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    args = ap.parse_args()
    print(json.dumps(as_manifest(Path(args.project_root)), ensure_ascii=False, indent=2))
