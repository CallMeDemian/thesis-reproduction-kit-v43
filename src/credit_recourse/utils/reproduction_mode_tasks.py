from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
RELATIVE = Path('configs/current/final_freeze/reproduction_modes.yaml')
ALLOWED_TASKS = {'VerifyInputs', 'Oracle', 'RL', 'LLMConfirmatory', 'ConfirmatoryAnalysis', 'ThesisOutputs', 'VerifyAll'}
ALLOWED_MODES = {'OracleClean', 'OracleRLClean', 'OracleRLLLMClean', 'FullClean'}

def load_mode_tasks(project_root: Path, mode: str) -> list[str]:
    path = project_root.resolve() / RELATIVE
    payload = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if payload.get('schema_version') != 'reproduction_modes_v6_fresh_only':
        raise ValueError('unsupported reproduction mode schema')
    modes = payload.get('modes') or {}
    if set(modes) != ALLOWED_MODES or mode not in modes:
        raise ValueError(f'unknown mode: {mode}')
    tasks = list(modes[mode].get('tasks') or [])
    if not tasks or any((task not in ALLOWED_TASKS for task in tasks)):
        raise ValueError(f'invalid task list for mode={mode}: {tasks}')
    if len(tasks) != len(set(tasks)):
        raise ValueError(f'duplicate tasks for mode={mode}: {tasks}')
    return tasks

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--mode', choices=sorted(ALLOWED_MODES), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(load_mode_tasks(Path(args.project_root), args.mode)))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
