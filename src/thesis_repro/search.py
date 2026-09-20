"""Original-search reproduction over an explicitly supplied catalog."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any


def _read_catalog(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload.get("rows", payload.get("configs", []))
    if suffix in {".xlsx", ".xls"}:
        import pandas as pd
        return pd.read_excel(path).to_dict("records")
    raise ValueError(f"unsupported search catalog format: {path.suffix}")


def build_search_baseline(root: Path, source_run: str) -> dict[str, Any]:
    root = Path(root).resolve()
    run_root = root / "runs" / source_run
    candidates = {
        "reward_components": run_root / "03_stage2/stage2/reward_components.parquet",
        "stage3_checkpoint": run_root / "04_rl_encoder/seed_2/final_epoch.pt",
        "stage4_checkpoint": run_root / "05_rl_bc/seed_2/final_epoch.pt",
    }
    release_dirs = sorted((run_root / "08_stage6").glob("FRESH_C3E_*"))
    candidates["oracle_surface"] = next((path / "firm_action_oracle_payoffs.parquet" for path in release_dirs if (path / "firm_action_oracle_payoffs.parquet").is_file()), run_root / "08_stage6/firm_action_oracle_payoffs.parquet")
    candidates["c2_decisions"] = next((path for path in (run_root / "08_stage6/C2_fixed_rule_decisions.parquet", *[p / "C2_fixed_rule_decisions.parquet" for p in release_dirs]) if path.is_file()), run_root / "08_stage6/C2_fixed_rule_decisions.parquet")
    missing = [name for name, path in candidates.items() if not path.is_file()]
    if missing:
        return {"status": "INPUT_REQUIRED", "reason": "COMPLETED_FRESH_RUN_REQUIRED_FOR_SEARCH_BASELINE", "source_run": source_run, "missing": missing}
    immutable = {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in candidates.values()}
    upstream_identity = {"source_run": source_run, "artifacts": immutable}
    evaluation_identity = {"source_run": source_run, "evaluation": "V4.3_2024_C3E_STAGE6"}
    baseline = {
        "schema_version": "v43_stage5_search_baseline_v2",
        "status": "READY_FOR_V43_STAGE5_HYPERPARAMETER_SEARCH",
        "upstream_identity": upstream_identity,
        "upstream_baseline_hash": hashlib.sha256(json.dumps(upstream_identity, sort_keys=True).encode()).hexdigest(),
        "evaluation_identity": evaluation_identity,
        "evaluation_contract_hash": hashlib.sha256(json.dumps(evaluation_identity, sort_keys=True).encode()).hexdigest(),
        "immutable_files": immutable,
        "artifacts": {name: str(path.relative_to(root)).replace("\\", "/") for name, path in candidates.items()},
    }
    destination = run_root / "03_stage2/V43_STAGE5_SEARCH_BASELINE.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "path": str(destination.relative_to(root)).replace("\\", "/"), "source_run": source_run, "baseline": baseline}


def reproduce_search(root: Path, *, mode: str = "rl", catalog_path: Path | None = None, execute: bool = True, run_id: str = "search-reproduction", source_run: str | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    candidates = [Path(catalog_path)] if catalog_path else [root / "contracts/search/rl_search_catalog.csv", root / "data/search/RL_HYPERPARAMETER_SEARCH.xlsx", root / "data/search/RL_HYPERPARAMETER_SEARCH.csv", root / "data/search/RL_HYPERPARAMETER_SEARCH.json"]
    catalog = next((path.resolve() for path in candidates if path.is_file()), None)
    if catalog is None:
        return {"status": "INPUT_REQUIRED", "scope": "ORIGINAL_SEARCH_REPRODUCTION", "mode": mode, "required_artifact": "original RL hyperparameter search catalog", "preserved_runtime": "src/credit_recourse/rl/v43_stage5_search.py", "reason": "EXTERNAL_SEARCH_CATALOG_REQUIRED", "training_runs": 0}
    rows = _read_catalog(catalog)
    if not rows:
        return {"status": "FAILED", "reason": "search catalog contains no configurations", "catalog": str(catalog)}
    from credit_recourse.rl.v43_stage5_search import SearchConfig, prepare_run
    if mode != "rl":
        return {"status": "FAILED", "reason": "only the preserved RL Stage5 search runtime is executable"}
    if execute:
        if not source_run:
            return {"status": "INPUT_REQUIRED", "reason": "COMPLETED_FRESH_RUN_REQUIRED_FOR_SEARCH_BASELINE"}
        baseline = build_search_baseline(root, source_run)
        if baseline.get("status") != "PASS":
            return baseline
        os.environ["THESIS_REPRO_SEARCH_BASELINE"] = str(root / "runs" / source_run / "03_stage2/V43_STAGE5_SEARCH_BASELINE.json")
    names = {field.name for field in fields(SearchConfig)} - {"run_id"}
    output = root / "runs" / run_id / "search" / "rl"
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "catalog_snapshot.csv"
    with snapshot.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    results = []
    for index, row in enumerate(rows):
        values: dict[str, Any] = {key: row[key] for key in names if key in row and row[key] not in ("", None)}
        values["run_id"] = str(row.get("run_id") or row.get("id") or f"catalog-{index:04d}")
        for key in names:
            if key in values:
                annotation = next(field.type for field in fields(SearchConfig) if field.name == key)
                values[key] = float(values[key]) if key not in {"epochs", "seed"} else int(float(values[key]))
        config = SearchConfig(**values).validate()
        if execute:
            results.append(prepare_run(root, config, dry_run=False))
        else:
            results.append({"run_id": config.run_id, "status": "PLANNED"})
    leaderboard = output / "reconstructed_leaderboard.json"
    leaderboard.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    selected = {"status": "PASS", "historical_reference_selection": ["B27", "M2_S0935", "DT06", "T15_REWARD_S088"], "selection_rule_status": "EXTERNAL_SELECTION_RULE_NOT_DISTRIBUTED", "selection_basis": "historical reference only; reconstructed leaderboard is reported separately"}
    (output / "selected_configurations.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "scope": "ORIGINAL_SEARCH_REPRODUCTION", "mode": mode, "catalog": str(catalog.relative_to(root)).replace("\\", "/"), "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(), "run_count": len(results), "outputs": str(output.relative_to(root)).replace("\\", "/")}
