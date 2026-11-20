"""Environment and required-artifact checks for the thesis reproduction kit."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


CORE_PATHS = (
    "data/final_freeze/configs/final_oracle_rl_contract.json",
    "data/final_freeze/stage1_oracle_backends/alpha/oracle_alpha_params.json",
    "data/final_freeze/stage2_candidate_projection/candidate_action_contract_v4_3.json",
    "src/credit_recourse/rl/pipelines/final_stage2_raw_action_source_precompute/pipeline.py",
    "src/credit_recourse/rl/contracts/avs256_acd_v2.py",
    "data/final_freeze/stage2_candidate_projection/action_sources/stage2_raw_action_source_panel.parquet",
    "data/final_freeze/stage2_candidate_projection/v4_3_runtime/V43_STAGE5_SEARCH_BASELINE.json",
    "data/final_freeze/stage3_acd_ssl/deployed_ensemble_e2_7seed/DEPLOYMENT_MANIFEST.json",
    "data/final_freeze/stage4_candidate_bc/deployed_ensemble_e2_7seed/DEPLOYMENT_MANIFEST.json",
    "data/final_freeze/stage5_candidate_iql/C3E_E2_7SEED_BALANCED_DFEBAFA6/C3E_definition.json",
    "configs/current/contract_manifest.json",
    "configs/current/llm/design_release.json",
    "repro/manifests/reproduction_manifest.json",
    "repro/expected/golden_cases.json",
)


def _requirements(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        rows[name.strip()] = version.strip()
    return rows


def inspect_environment(project_root: Path, *, strict_versions: bool = False) -> dict[str, Any]:
    root = Path(project_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    lock = root / "requirements.lock.txt"
    expected = _requirements(lock) if lock.is_file() else {}
    if not lock.is_file():
        errors.append("requirements.lock.txt is missing")

    packages: list[dict[str, Any]] = []
    for name, version in expected.items():
        try:
            observed = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed = None
        status = "PASS" if observed == version else ("WARN" if observed else "FAIL")
        if observed is None:
            errors.append(f"missing package: {name}=={version}")
        elif observed != version:
            message = f"package version drift: {name} expected={version} observed={observed}"
            (errors if strict_versions else warnings).append(message)
            if strict_versions:
                status = "FAIL"
        packages.append({"distribution": name, "expected": version, "observed": observed, "status": status})

    python_status = "PASS" if sys.version_info[:2] == (3, 12) else "WARN"
    if python_status == "WARN":
        message = f"reference Python is 3.12; observed={platform.python_version()}"
        (errors if strict_versions else warnings).append(message)

    artifacts = []
    for relative in CORE_PATHS:
        path = root / relative
        present = path.is_file()
        artifacts.append({"path": relative, "present": present})
        if not present:
            errors.append(f"required artifact missing: {relative}")

    try:
        import torch

        torch_info: dict[str, Any] = {
            "installed": True,
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_build": str(torch.version.cuda) if torch.version.cuda else None,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        if not torch.cuda.is_available():
            warnings.append("CUDA unavailable: VERIFY_EXISTING works, reference RL training requires CUDA")
    except Exception as exc:
        torch_info = {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
        errors.append("PyTorch could not be imported")

    api = {
        "openai_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "google_key_present": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
    }
    if not any(api.values()):
        warnings.append("No LLM API key found; GENERATE is unavailable, REPLAY remains available when raw responses exist")

    return {
        "schema_version": "thesis_reproduction_doctor_v1",
        "status": "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
        "project_root": str(root),
        "python": {"status": python_status, "version": platform.python_version(), "executable": sys.executable},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": packages,
        "torch": torch_info,
        "artifacts": artifacts,
        "llm_api": api,
        "errors": errors,
        "warnings": warnings,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
