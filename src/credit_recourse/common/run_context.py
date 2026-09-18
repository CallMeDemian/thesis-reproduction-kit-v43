from __future__ import annotations
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
ARTIFACT_ROOT_ENV = 'CREDIT_RECOURSE_ARTIFACT_ROOT'
ANALYSIS_ROOT_ENV = 'CREDIT_RECOURSE_ANALYSIS_ROOT'
_ALLOWED_ID = re.compile('^[A-Za-z][A-Za-z0-9_-]{1,31}$')
_ALLOWED_STATUS = {'INCOMPLETE', 'PASS', 'PASS_WITH_SKIPS', 'FAIL'}

class ReproductionContractError(RuntimeError):
    """Raised when a lineage, path, or immutable-output contract is violated."""

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def validate_identifier(value: str, *, label: str='identifier') -> str:
    value = str(value).strip()
    if not _ALLOWED_ID.fullmatch(value):
        raise ReproductionContractError(f"Invalid {label}={value!r}. Use 2-32 characters: leading letter, then letters, digits, '_' or '-'.")
    return value

def ensure_within(path: Path, parent: Path, *, label: str='path') -> Path:
    path = Path(path).resolve()
    parent = Path(parent).resolve()
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise ReproductionContractError(f'{label} escapes required root: path={path}, root={parent}') from exc
    return path

def iter_files(root: Path, *, exclude_relative: Iterable[str]=()) -> Iterable[Path]:
    root = Path(root).resolve()
    excluded = {str(Path(x)).replace('\\', '/') for x in exclude_relative}
    for path in sorted(root.rglob('*'), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in excluded:
            continue
        yield path

def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', delete=False, dir=str(path.parent), suffix='.tmp') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        tmp = Path(handle.name)
    tmp.replace(path)

def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def resolve_artifact_root(project_root: Path | str, *, explicit: Path | str | None=None, must_exist: bool=True) -> Path:
    """Resolve the current clean-run artifact root from an explicit contract."""
    candidate: Path | None = None
    source = ''
    if explicit is not None and str(explicit).strip():
        candidate = Path(explicit)
        source = 'explicit'
    elif os.environ.get(ARTIFACT_ROOT_ENV, '').strip():
        candidate = Path(os.environ[ARTIFACT_ROOT_ENV])
        source = ARTIFACT_ROOT_ENV
    else:
        raise ReproductionContractError(
            f'Artifact-root resolution is explicit-only. Pass an artifact root or set {ARTIFACT_ROOT_ENV}.'
        )
    candidate = candidate.expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_root).resolve() / candidate
    candidate = candidate.resolve()
    if must_exist and (not candidate.exists()):
        raise FileNotFoundError(f'Artifact root from {source} does not exist: {candidate}')
    return candidate

def resolve_analysis_root(project_root: Path | str, *, explicit: Path | str | None=None, must_exist: bool=False) -> Path:
    candidate: Path
    if explicit is not None and str(explicit).strip():
        candidate = Path(explicit)
    elif os.environ.get(ANALYSIS_ROOT_ENV, '').strip():
        candidate = Path(os.environ[ANALYSIS_ROOT_ENV])
    else:
        # Analysis has one repository-level owner.  Frozen data/artifacts stay
        # under archive/DEPLOYED_RELEASE; do not recreate a second analysis owner.
        candidate = Path(project_root).resolve() / 'analysis'
    if not candidate.is_absolute():
        candidate = Path(project_root).resolve() / candidate
    candidate = candidate.resolve()
    if must_exist and (not candidate.exists()):
        raise FileNotFoundError(candidate)
    return candidate

@dataclass(frozen=True)
class RunPaths:
    project_root: Path
    lineage: str
    run_id: str
    run_root: Path

    @classmethod
    def create(cls, project_root: Path | str, lineage: str, run_id: str) -> 'RunPaths':
        project_root = Path(project_root).resolve()
        run_id = validate_identifier(run_id, label='run_id')
        lineage_map = {'FreshRun': 'fresh', 'RLReproduction': 'rlrep'}
        if lineage not in lineage_map:
            raise ReproductionContractError(f'Unsupported lineage: {lineage}')
        run_root = project_root / 'data' / 'runs' / lineage_map[lineage] / run_id
        return cls(project_root, lineage, run_id, run_root)

    @property
    def status_path(self) -> Path:
        return self.run_root / 'status.json'

    @property
    def work_root(self) -> Path:
        return self.run_root / '.work'

    @property
    def manifests_root(self) -> Path:
        return self.run_root / 'manifests'

    def initialize(self, *, config: dict[str, Any]) -> None:
        if self.run_root.exists():
            raise FileExistsError(f'Run ID already exists and cannot be overwritten: {self.run_root}')
        self.run_root.mkdir(parents=True)
        self.work_root.mkdir()
        self.manifests_root.mkdir()
        write_json_atomic(self.status_path, {'run_id': self.run_id, 'lineage': self.lineage, 'status': 'INCOMPLETE', 'created_utc': utc_now(), 'updated_utc': utc_now(), 'required_tasks': {}, 'optional_tasks': {}})
        write_json_atomic(self.run_root / 'run_config.json', config)

    def set_status(self, status: str, *, required_tasks: dict[str, str] | None=None, optional_tasks: dict[str, str] | None=None, message: str | None=None) -> None:
        if status not in _ALLOWED_STATUS:
            raise ReproductionContractError(f'Unsupported run status: {status}')
        current = read_json(self.status_path)
        if current.get('status') in {'PASS', 'PASS_WITH_SKIPS'}:
            raise ReproductionContractError(f'Completed run is immutable: {self.run_root}')
        current['status'] = status
        current['updated_utc'] = utc_now()
        if required_tasks is not None:
            current['required_tasks'] = required_tasks
        if optional_tasks is not None:
            current['optional_tasks'] = optional_tasks
        if message:
            current['message'] = message
        write_json_atomic(self.status_path, current)
