"""Call-time resolver for fresh Oracle scientific runtime paths.

The resolver deliberately reads environment bindings on every call.  Fresh
verification must never retain a module-level path derived from an earlier
run or silently fall back to a frozen release tree.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import yaml


FORBIDDEN_FRESH_ROOTS = (
    "data/final_freeze",
    "configs/current",
    "archive/DEPLOYED_RELEASE",
    "frozen/",
)


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


def oracle_execution_profile() -> str:
    """Return the explicit Oracle acceptance profile for this process.

    Production fresh replication is the default.  The synthetic profile is
    reserved for the deterministic architecture fixture and never authorizes
    a licensed-data run to accept a non-canonical fitted Alpha contract.
    """
    profile = str(os.environ.get("THESIS_REPRO_ORACLE_PROFILE", "production")).strip().lower()
    if profile not in {"production", "synthetic"}:
        raise ValueError(f"unknown Oracle execution profile: {profile!r}")
    return profile


def run_relative_path(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(Path(project_root).resolve()).as_posix()


def assert_fresh_path(project_root: Path, path: Path, *, label: str) -> None:
    resolved = path.resolve()
    root = Path(project_root).resolve()
    text = resolved.as_posix()
    if any(token in text for token in FORBIDDEN_FRESH_ROOTS):
        raise ValueError(f"{label} resolves to forbidden legacy/frozen root: {resolved}")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes project root: {resolved}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _backend_binding(runtime: FreshOracleRuntime, name: str) -> dict[str, str]:
    output_root = (runtime.backends_root / name).resolve()
    params = {
        "alpha": "oracle_alpha_params.json",
        "beta": "benchmark_beta_params.json",
        "gamma": "benchmark_gamma_params.json",
    }[name]
    output = {
        "alpha": "oracle_firm_year_output_alpha.parquet",
        "beta": "benchmark_firm_year_output_beta.parquet",
        "gamma": "benchmark_firm_year_output_gamma.parquet",
    }[name]
    binding = {
        "path": output_root.as_posix(),
        "params": (output_root / params).as_posix(),
        "output": (output_root / output).as_posix(),
    }
    if name == "alpha":
        binding["metrics"] = (output_root / "preliminary_dev_oot_metrics_alpha.csv").as_posix()
    if name == "gamma":
        binding["model"] = (output_root / "benchmark_gamma_model.joblib").as_posix()
    return binding


def materialize_fresh_oracle_registry(project_root: Path, runtime: FreshOracleRuntime | None = None) -> Path:
    """Create the run-local Oracle registry from immutable semantic metadata.

    The frozen registry is used only as a contract metadata source.  Every
    artifact binding in the materialized registry is rewritten to the active
    run namespace before it is written.  This function is intentionally
    callable at Stage1 startup, before any backend output exists.
    """
    root = Path(project_root).resolve()
    runtime = runtime or resolve_fresh_oracle_runtime(root)
    source = root / "contracts" / "scientific" / "final_freeze" / "oracle_backend_registry.yaml"
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"immutable Oracle registry metadata is missing: {source}")
    source_sha = file_sha256(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    registry = copy.deepcopy(payload)
    backends = registry.setdefault("backends", {})
    bindings: dict[str, dict[str, str]] = {}
    for name in ("alpha", "beta", "gamma"):
        if name not in backends:
            raise ValueError(f"immutable Oracle registry is missing backend metadata: {name}")
        binding = _backend_binding(runtime, name)
        bindings[name] = binding
        backends[name] = {**backends[name], **binding}
        for label, value in binding.items():
            assert_fresh_path(root, Path(value), label=f"fresh Oracle registry {name}.{label}")
    registry["provenance"] = {
        "source_contract_path": source.relative_to(root).as_posix(),
        "source_contract_sha256": source_sha,
        "materialized_run_id": runtime.work_root.parent.parent.name,
        "generated_path_bindings": bindings,
        "frozen_registry_used_as": "immutable_contract_metadata_only",
        "fresh_compute_parent_policy": "same_run_artifacts_only",
    }
    path = runtime.registry_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(registry, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path
