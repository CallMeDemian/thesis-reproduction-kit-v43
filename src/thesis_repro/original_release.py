from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from .paths import ROOT

CANONICAL_LF_PATHS = {
    "provenance/RL_PARENT_GRAPH.json",
    "frozen/release/submission/THESIS_V43_SUBMISSION_FINAL_20260917/artifact_registry.json",
    "frozen/release/submission/THESIS_V43_SUBMISSION_FINAL_20260917/claim_registry.json",
    "frozen/release/submission/THESIS_V43_SUBMISSION_FINAL_20260917/contract_registry.json",
    "frozen/release/submission/THESIS_V43_SUBMISSION_FINAL_20260917/source_registry.json",
    "frozen/release/submission/THESIS_V43_SUBMISSION_FINAL_20260917/table_registry.csv",
}


def _sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.relative_to(ROOT).as_posix() in CANONICAL_LF_PATHS:
        data = data.replace(b"\r\n", b"\n")
    digest = hashlib.sha256(data)
    return digest.hexdigest()


def _root_hash(manifest: dict[str, Any]) -> str:
    sha_by_id = {str(item["logical_id"]): str(item["sha256"]) for item in manifest["records"]}
    sha_by_id.update({str(item["logical_id"]): str(item["sha256"]) for item in manifest["pack_index"]})
    rows = []
    for item in sorted(manifest["records"], key=lambda x: str(x["logical_id"])):
        parents = sorted(str(parent) for parent in item["parents"])
        parent_hashes = sorted(sha_by_id[parent] for parent in parents)
        rows.append(f"{item['logical_id']}\t{item['sha256']}\t{','.join(parents)}\t{','.join(parent_hashes)}")
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()


def verify_original_release() -> dict[str, Any]:
    import pyarrow.parquet as pq
    manifest = json.loads((ROOT / "frozen/original_release/ORIGINAL_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    expected_root = (ROOT / "frozen/original_release/RELEASE_ROOT.txt").read_text(encoding="utf-8").strip()
    checks: list[dict[str, Any]] = []
    actual_root = _root_hash(manifest)
    checks.append({"name": "release_root", "expected": expected_root, "actual": actual_root, "status": "PASS" if expected_root == actual_root == manifest["release_root_sha256"] else "FAIL"})
    verified = 0
    for item in manifest["records"]:
        path = ROOT / str(item["path"])
        ok = path.is_file() and _sha256(path) == item["sha256"]
        if ok:
            verified += 1
        checks.append({"name": f"artifact:{item['logical_id']}", "status": "PASS" if ok else "FAIL"})
    graph = json.loads((ROOT / "provenance/RL_PARENT_GRAPH.json").read_text(encoding="utf-8"))
    checks.append({"name": "rl_parent_graph", "expected": "PASS", "actual": graph["summary"]["status"], "status": "PASS" if graph["summary"]["status"] == "PASS" and graph["summary"]["closed_member_count"] == 28 else "FAIL"})
    c3e = json.loads((ROOT / "frozen/original_release/rl/C3E_E2_7SEED_BALANCED_DFEBAFA6/C3E_definition.json").read_text(encoding="utf-8"))
    checks.append({"name": "c3e_identity", "status": "PASS" if c3e.get("decision_sha256") == "dfebafa6b50e55c2f5019766e0e540af9964b2a9ab7d4e49805d03e8011b792e" else "FAIL"})
    stage8 = next(item for item in manifest["records"] if item["logical_id"] == "stage8.canonical.itt")
    checks.append({"name": "stage8_canonical_sha256", "status": "PASS" if stage8["sha256"] == "aedfadbf1728a14d3e4cb45bfa8a022d2c10313c52606e7bd1aee68899de8a0d" else "FAIL"})
    with (ROOT / "frozen/release/submission/THESIS_V43_SUBMISSION_FINAL_20260917/table_registry.csv").open(encoding="utf-8-sig") as handle:
        table_rows = list(csv.DictReader(handle))
    checks.append({"name": "release_table_registry", "status": "PASS" if len(table_rows) > 0 and len({row["table_id"] for row in table_rows}) == len(table_rows) else "FAIL", "rows": len(table_rows)})
    stage8_rows = pq.read_table(ROOT / str(stage8["path"])).num_rows
    checks.append({"name": "stage8_row_count", "expected": 96600, "actual": stage8_rows, "status": "PASS" if stage8_rows == 96600 else "FAIL"})
    llm_counts = {}
    for regime, run_id in {
        "baseline": "2cf6d6d0e4250e66ce882ab95f9d641f2c73711ffbc6429e9a203dcc2ee680a2",
        "high": "d5d878bf50718bac65ede73df3fc88c16723870b6c90a790557ff35b063b7ba7",
    }.items():
        run = ROOT / "frozen/original_release/llm/final_plan3" / run_id
        logical = set(pq.read_table(run / "logical_requests.parquet", columns=["request_id"]).column("request_id").to_pylist())
        raw = set()
        for raw_path in (run / "raw_responses").glob("*.jsonl"):
            with raw_path.open(encoding="utf-8") as handle:
                raw.update(json.loads(line)["request_id"] for line in handle)
        inputs = sum(sum(1 for _ in p.open(encoding="utf-8")) for p in run.glob("provider_batches/**/*.jsonl") if not p.name.endswith("_output.jsonl"))
        outputs = sum(sum(1 for _ in p.open(encoding="utf-8")) for p in run.glob("provider_batches/**/*_output.jsonl"))
        llm_counts[regime] = {"logical": len(logical), "raw": len(raw), "provider_inputs": inputs, "provider_outputs": outputs, "request_ids_match": logical == raw}
        checks.append({"name": f"llm_{regime}_cardinality", "status": "PASS" if llm_counts[regime] == {"logical": 24150, "raw": 24150, "provider_inputs": 24150, "provider_outputs": 24150, "request_ids_match": True} else "FAIL"})
    distribution = ROOT / "frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip"
    with zipfile.ZipFile(distribution) as archive:
        with archive.open("THESIS_REPRO_KIT_v2.1.1_FINAL/evidence/RESULT_REGISTRY.csv") as raw_handle:
            registry_rows = list(csv.DictReader(io.TextIOWrapper(raw_handle, encoding="utf-8-sig")))
    registry_ids = [row["result_id"] for row in registry_rows]
    checks.append({"name": "result_registry_914_unique", "expected": 914, "actual": len(registry_rows), "unique": len(set(registry_ids)), "status": "PASS" if len(registry_rows) == 914 and len(set(registry_ids)) == 914 else "FAIL"})
    result = {
        "status": "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL",
        "release_id": manifest["release_id"],
        "release_root_sha256": actual_root,
        "artifact_count": manifest["artifact_count"],
        "verified_artifacts": verified,
        "rl_parent_chains": graph["summary"],
        "stage8_rows": stage8_rows,
        "registry_rows": len(registry_rows),
        "llm": llm_counts,
        "checks": checks,
    }
    return result
