"""Verify migrated original assets and the exact two-regime LLM cardinality."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def raw_request_ids(paths: list[Path]) -> set[str]:
    ids = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                ids.add(row["request_id"])
    return ids


def main() -> int:
    manifest = json.loads((ROOT / "provenance/ASSET_PACK_MANIFEST.json").read_text(encoding="utf-8"))
    checked = 0
    for pack in manifest["packs"]:
        if pack["status"] != "COPIED_AND_HASH_VERIFIED":
            raise AssertionError(f"pack not migrated: {pack['pack_id']}")
        for item in pack["files"]:
            path = ROOT / item["planned_target_path"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise AssertionError(f"hash mismatch: {path}")
            checked += 1

    llm_root = ROOT / "frozen/original_release/llm/final_plan3"
    regimes = {
        "baseline": "2cf6d6d0e4250e66ce882ab95f9d641f2c73711ffbc6429e9a203dcc2ee680a2",
        "high": "d5d878bf50718bac65ede73df3fc88c16723870b6c90a790557ff35b063b7ba7",
    }
    llm_report = {}
    for name, run_id in regimes.items():
        run = llm_root / run_id
        logical = pq.read_table(run / "logical_requests.parquet", columns=["request_id"]).column("request_id").to_pylist()
        logical_ids = set(logical)
        raw_paths = sorted((run / "raw_responses").glob("*.jsonl"))
        raw_ids = raw_request_ids(raw_paths)
        input_paths = [p for p in run.glob("provider_batches/**/*.jsonl") if not p.name.endswith("_output.jsonl")]
        output_paths = [p for p in run.glob("provider_batches/**/*_output.jsonl")]
        input_count = sum(jsonl_count(p) for p in input_paths)
        output_count = sum(jsonl_count(p) for p in output_paths)
        assert len(logical_ids) == 24150
        assert len(raw_ids) == 24150
        assert logical_ids == raw_ids
        assert input_count == 24150
        assert output_count == 24150
        llm_report[name] = {
            "run_id": run_id,
            "logical_requests": len(logical_ids),
            "normalized_raw_responses": len(raw_ids),
            "provider_inputs": input_count,
            "provider_outputs": output_count,
        }

    rl_root = ROOT / "frozen/original_release/rl/C3E_E2_7SEED_BALANCED_DFEBAFA6"
    assert len(list(rl_root.rglob("*.pt"))) == 28
    print(json.dumps({"status": "PASS", "hash_verified_files": checked, "llm": llm_report, "rl_checkpoints": 28}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
