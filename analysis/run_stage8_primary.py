from pathlib import Path
import json

from credit_recourse.eval.final_stage8_llm_multi_oracle_eval.pipeline import run_stage8

RUNTIME = Path(r"C:\Users\Demian\Desktop\REPRO_KIT_semantic_repair\analysis\rl_reference_gate_runtime")
print(json.dumps(run_stage8(project_root=RUNTIME), ensure_ascii=False, indent=2))
