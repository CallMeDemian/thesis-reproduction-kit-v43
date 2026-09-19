"""Call-time resolver for fresh Oracle scientific runtime paths.

The resolver deliberately reads environment bindings on every call.  Fresh
verification must never retain a module-level path derived from an earlier
run or silently fall back to a frozen release tree.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable


class CallTimePath:
    """Path-like binding whose environment value is read on each operation."""

    def __init__(self, env_name: str, fallback: Callable[[], Path]):
        self.env_name = env_name
        self.fallback = fallback

    def resolve(self) -> Path:
        configured = os.environ.get(self.env_name)
        return Path(configured).resolve() if configured else Path(self.fallback()).resolve()

    def __truediv__(self, value):
        return self.resolve() / value

    def __fspath__(self):
        return os.fspath(self.resolve())

    def __str__(self):
        return str(self.resolve())

    def __repr__(self):
        return repr(self.resolve())

    def __getattr__(self, name):
        return getattr(self.resolve(), name)


@dataclass(frozen=True)
class FreshOracleRuntime:
    project_root: Path
    work_root: Path
    config_root: Path
    raw_root: Path

    @property
    def stage0_root(self) -> Path:
        return self.work_root / "stage0_oracle_foundation"

    @property
    def inputs_root(self) -> Path:
        return self.work_root / "stage1_oracle_inputs"

    @property
    def backends_root(self) -> Path:
        return self.work_root / "stage1_oracle_backends"

    @property
    def ledgers_root(self) -> Path:
        return self.work_root / "ledgers"

    @property
    def registry_path(self) -> Path:
        return self.config_root / "oracle_backend_registry.yaml"


def resolve_fresh_oracle_runtime(project_root: Path) -> FreshOracleRuntime:
    root = Path(project_root).resolve()
    work = Path(os.environ.get("THESIS_REPRO_ORACLE_WORK_ROOT", root / "runs" / "unbound" / "oracle_work")).resolve()
    config = Path(os.environ.get("THESIS_REPRO_ORACLE_CONFIG_ROOT", root / "contracts" / "oracle_components")).resolve()
    raw = Path(os.environ.get("THESIS_REPRO_ORACLE_RAW_ROOT", root / "data" / "raw")).resolve()
    return FreshOracleRuntime(root, work, config, raw)


def run_relative_path(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(Path(project_root).resolve()).as_posix()


def assert_fresh_path(project_root: Path, path: Path, *, label: str) -> None:
    resolved = path.resolve()
    root = Path(project_root).resolve()
    forbidden = ("data/final_freeze", "configs/current", "archive/DEPLOYED_RELEASE", "frozen/")
    text = resolved.as_posix()
    if any(token in text for token in forbidden):
        raise ValueError(f"{label} resolves to forbidden legacy/frozen root: {resolved}")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes project root: {resolved}") from exc
