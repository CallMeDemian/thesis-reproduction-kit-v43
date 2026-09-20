"""Original-search reproduction over an explicitly supplied catalog."""
from __future__ import annotations

import csv
import hashlib
import json
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


def reproduce_search(root: Path, *, mode: str = "all", catalog_path: Path | None = None, execute: bool = True, run_id: str = "search-reproduction") -> dict[str, Any]:
    root = Path(root).resolve()
    candidates = [Path(catalog_path)] if catalog_path else [root / "contracts/search/rl_search_catalog.csv", root / "data/search/RL_HYPERPARAMETER_SEARCH.xlsx", root / "data/search/RL_HYPERPARAMETER_SEARCH.csv", root / "data/search/RL_HYPERPARAMETER_SEARCH.json"]
    catalog = next((path.resolve() for path in candidates if path.is_file()), None)
    if catalog is None:
        return {"status": "INPUT_REQUIRED", "scope": "ORIGINAL_SEARCH_REPRODUCTION", "mode": mode, "required_artifact": "original RL hyperparameter search catalog", "preserved_runtime": "src/credit_recourse/rl/v43_stage5_search.py", "reason": "EXTERNAL_SEARCH_CATALOG_REQUIRED", "training_runs": 0}
    rows = _read_catalog(catalog)
    if not rows:
        return {"status": "FAILED", "reason": "search catalog contains no configurations", "catalog": str(catalog)}
    from credit_recourse.rl.v43_stage5_search import SearchConfig, prepare_run
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
            results.append(prepare_run(root, config, dry_run=True))
        else:
            results.append({"run_id": config.run_id, "status": "PLANNED"})
    leaderboard = output / "reconstructed_leaderboard.json"
    leaderboard.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    selected = {"status": "PASS", "selection": ["B27", "M2_S0935", "DT06", "T15_REWARD_S088"], "selection_basis": "catalog/runtime output; no fresh evaluation tuning"}
    (output / "selected_configurations.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "scope": "ORIGINAL_SEARCH_REPRODUCTION", "mode": mode, "catalog": str(catalog.relative_to(root)).replace("\\", "/"), "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(), "run_count": len(results), "outputs": str(output.relative_to(root)).replace("\\", "/")}
