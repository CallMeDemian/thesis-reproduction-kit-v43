"""Build a machine-readable audit of the reachable fresh scientific runtime."""

from __future__ import annotations

import ast
import json
import re
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
AUDIT = ROOT / "runs/_runtime_audit"
ENTRYPOINTS = {
    "oracle": [
        "credit_recourse.oracle.stage0.build_stage0_foundation_from_raw",
        "credit_recourse.oracle.stage1.run_stage1_oracle_development",
    ],
    "simulator": ["credit_recourse.simulator.v43_production_bundle"],
    "rl": [
        "credit_recourse.rl.v43_one_pass_data",
        "credit_recourse.rl.v43_one_pass_grid",
        "credit_recourse.rl.v43_runtime",
        "credit_recourse.rl.v43_training",
    ],
    "llm": [
        "credit_recourse.final_release.final_release.executor",
        "credit_recourse.final_release.final_release.live_batch",
        "credit_recourse.final_release.final_release.collection",
        "credit_recourse.final_release.final_release.evaluation_bridge",
        "credit_recourse.final_release.final_release.providers",
    ],
    "evaluation": [
        "credit_recourse.eval.final_stage8_llm_multi_oracle_eval.pipeline",
        "credit_recourse.eval.final_stage9_llm_rl_comparison.pipeline",
        "credit_recourse.analysis.final_v43.runner",
    ],
}
LEGACY_TOKENS = (
    "archive/DEPLOYED_RELEASE",
    "configs/current",
    "data/final_freeze",
    "frozen/original_release",
    "frozen/evidence",
)


def module_path(module: str) -> Path | None:
    path = SRC / (module.replace(".", "/") + ".py")
    if path.is_file():
        return path
    init = SRC / module.replace(".", "/") / "__init__.py"
    return init if init.is_file() else None


def imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name.startswith("credit_recourse."))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("credit_recourse."):
            found.add(node.module)
    return found


def reachable() -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    graph: dict[str, list[str]] = {}
    missing: list[dict[str, str]] = []
    for family, roots in ENTRYPOINTS.items():
        queue = deque(roots)
        seen: set[str] = set()
        while queue:
            module = queue.popleft()
            if module in seen:
                continue
            seen.add(module)
            path = module_path(module)
            if path is None:
                missing.append({"family": family, "module": module, "reason": "internal module has no source file"})
                graph[module] = []
                continue
            deps = sorted(imports(path))
            graph[module] = deps
            queue.extend(deps)
    return graph, missing


def legacy_references() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in [*sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "contracts").rglob("*.json")), *sorted((ROOT / "contracts").rglob("*.yaml"))]:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for token in LEGACY_TOKENS:
                if token in line:
                    if token.startswith("frozen/"):
                        classification = "A_LEGITIMATE_FROZEN_REFERENCE"
                    elif "contract" in line.lower() or path.parts[-2:-1] == ("contracts",):
                        classification = "B_IMMUTABLE_CONTRACT_TO_PROMOTE_OR_REBIND"
                    elif "archive/DEPLOYED_RELEASE" in line and ("historical" in line.lower() or "source" in line.lower()):
                        classification = "E_HISTORICAL_ONLY_IMPLEMENTATION"
                    else:
                        classification = "D_OBSOLETE_LEGACY_PATH_TO_REMOVE"
                    rows.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "line": line_number, "token": token, "classification": classification, "text": line.strip()})
    return rows


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    graph, missing = reachable()
    legacy = legacy_references()
    inventory = json.loads((ROOT / "contracts/scientific/input_inventory.json").read_text(encoding="utf-8"))
    required_contracts = sorted(str(path.relative_to(ROOT)).replace("\\", "/") for path in (ROOT / "contracts").rglob("*") if path.is_file() and path.suffix in {".json", ".yaml", ".yml", ".csv"})
    required_raw = [{"target_root": item.get("target_root"), "file_count": item.get("file_count"), "files": [file.get("relative_path") for file in item.get("files", [])]} for item in inventory.get("inputs", []) if str(item.get("target_root", "")).startswith("data/raw")]
    plan = {
        "fresh_allowed_roots": ["data/raw", "contracts", "src", "runs/<run_id>"],
        "frozen_policy": "comparison/reference/verification only",
        "rebinding": {"archive/DEPLOYED_RELEASE": "runs/<run_id>/stage outputs", "configs/current": "contracts/scientific and contracts/llm", "data/final_freeze": "runs/<run_id>/stage8 or stage9", "frozen/original_release": "never a fresh compute parent", "frozen/evidence": "comparison/reference only"},
        "missing_internal_modules": missing,
    }
    outputs = {
        "internal_import_closure": {"schema_version": "runtime_import_closure_v1", "entrypoints": ENTRYPOINTS, "graph": graph, "reachable_module_count": len(graph), "missing_module_count": len(missing), "status": "FAIL" if missing else "PASS"},
        "legacy_path_references": {"schema_version": "legacy_path_reference_audit_v1", "references": legacy, "counts_by_classification": {key: sum(row["classification"] == key for row in legacy) for key in sorted({row["classification"] for row in legacy})}},
        "missing_internal_modules": {"schema_version": "missing_internal_modules_v1", "status": "FAIL" if missing else "PASS", "missing": missing},
        "required_static_contracts": {"schema_version": "required_static_contracts_v1", "contracts": required_contracts, "count": len(required_contracts)},
        "required_raw_inputs": {"schema_version": "required_raw_inputs_v1", "inputs": required_raw, "required_file_count": sum(len(item["files"]) for item in required_raw)},
        "runtime_rebinding_plan": {"schema_version": "runtime_rebinding_plan_v1", **plan},
    }
    for name, payload in outputs.items():
        (AUDIT / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FAIL" if missing else "PASS", "reachable_module_count": len(graph), "missing_module_count": len(missing), "legacy_reference_count": len(legacy), "audit_root": str(AUDIT.relative_to(ROOT)).replace("\\", "/")}, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
