from __future__ import annotations

import csv
import importlib.util
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .frozen import REGISTRY_MEMBER, reproduce_frozen
from .paths import ROOT, write_json
from .release import load_certified_release


def _load_output_builder(root: Path):
    path = root / "scripts/build_thesis_outputs_v21.py"
    spec = importlib.util.spec_from_file_location("thesis_repro_build_outputs_v21", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load certified output builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frozen_registry(root: Path) -> pd.DataFrame:
    distribution = root / "frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip"
    with zipfile.ZipFile(distribution) as archive:
        with archive.open(REGISTRY_MEMBER) as handle:
            return pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8-sig"))


def _compare_registry(root: Path, run_id: str) -> dict[str, Any]:
    generated_path = root / "runs" / run_id / "analysis/RESULT_REGISTRY.csv"
    generated = pd.read_csv(generated_path)
    frozen = _frozen_registry(root)
    generated_ids = set(generated["result_id"].astype(str))
    frozen_ids = set(frozen["result_id"].astype(str))
    checks = [
        {"name": "row_count", "expected": 914, "actual": len(generated), "status": "PASS" if len(generated) == 914 else "FAIL"},
        {"name": "unique_result_ids", "expected": 914, "actual": generated["result_id"].nunique(), "status": "PASS" if generated["result_id"].nunique() == 914 else "FAIL"},
        {"name": "result_id_set", "expected": len(frozen_ids), "actual": len(generated_ids & frozen_ids), "status": "PASS" if generated_ids == frozen_ids else "FAIL"},
    ]
    if "result_class" in frozen and "result_class" in generated:
        checks.append({"name": "result_class_counts", "expected": frozen["result_class"].value_counts().to_dict(), "actual": generated["result_class"].value_counts().to_dict(), "status": "PASS" if frozen["result_class"].value_counts().to_dict() == generated["result_class"].value_counts().to_dict() else "FAIL"})
    for column in ("N", "estimate"):
        if column in frozen and column in generated:
            left = frozen.set_index("result_id")[column].sort_index()
            right = generated.set_index("result_id")[column].sort_index()
            if column == "N":
                equal = left.index.equals(right.index) and left.astype("Int64").equals(right.astype("Int64"))
            else:
                equal = left.index.equals(right.index) and np.isclose(
                    left.astype(float).to_numpy(),
                    right.astype(float).to_numpy(),
                    atol=1e-12,
                    rtol=1e-12,
                    equal_nan=True,
                ).all()
            checks.append({"name": f"{column}_match", "status": "PASS" if equal else "FAIL"})
    return {"status": "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL", "generated": str(generated_path), "frozen_registry_rows": len(frozen), "checks": checks}


def reproduce(root: Path = ROOT, run_id: str = "thesis-reproduction") -> dict[str, Any]:
    root = Path(root).resolve()
    release = load_certified_release(root)
    replay = reproduce_frozen(root, run_id)
    if replay.get("status") != "PASS":
        return {"status": "FAIL", "release": release, "replay": replay, "scope": "PUBLISHED_RESULT_REPRODUCTION"}
    builder = _load_output_builder(root)
    outputs = Path(builder.build(root, run_id, "MANUSCRIPT_FROZEN"))
    comparison = _compare_registry(root, run_id)
    final_dir = root / "runs" / run_id / "final"
    shutil.copytree(root / "runs" / run_id / "analysis", final_dir / "analysis", dirs_exist_ok=True)
    shutil.copytree(outputs, final_dir / "thesis_outputs", dirs_exist_ok=True)
    receipt = {
        "status": "PASS" if comparison.get("status") == "PASS" else "FAIL",
        "scope": "PUBLISHED_RESULT_REPRODUCTION",
        "release": release,
        "replay": replay,
        "analysis": {"primary_rows": 96, "supplemental_rows": 722, "parent_rows": 96, "registry_rows": 914, "estimate_source": "RECOMPUTED_FIRM_LEVEL_PRIMARY_ESTIMATE", "bootstrap_ci_source": "FROZEN_PRODUCTION_STREAM"},
        "registry_comparison": comparison,
        "outputs": str(final_dir),
        "api_calls": 0,
        "training_runs": 0,
    }
    write_json(final_dir / "REPRODUCTION_RECEIPT.json", receipt)
    return receipt
