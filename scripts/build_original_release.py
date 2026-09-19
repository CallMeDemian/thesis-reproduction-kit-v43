"""Build the unified original scientific release manifest and root hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_record(records: list[dict[str, object]], logical_id: str, path: str, role: str, parents: list[str], level: str) -> None:
    file_path = ROOT / path
    if not file_path.is_file():
        raise FileNotFoundError(path)
    records.append(
        {
            "logical_id": logical_id,
            "path": path,
            "sha256": sha256(file_path),
            "byte_size": file_path.stat().st_size,
            "semantic_role": role,
            "parents": sorted(parents),
            "reproducibility_level": level,
        }
    )


def main() -> int:
    pack_manifest = json.loads((ROOT / "provenance/ASSET_PACK_MANIFEST.json").read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    pack_index = []
    for pack in pack_manifest["packs"]:
        if pack["status"] != "COPIED_AND_HASH_VERIFIED":
            raise ValueError(f"asset pack is not verified: {pack['pack_id']}")
        parent_id = f"pack:{pack['pack_id']}"
        pack_hash_rows = []
        for item in pack["files"]:
            path = item["planned_target_path"]
            add_record(records, f"{parent_id}:{path}", path, item["semantic_role"], [parent_id], "R1_EXACT_ARTIFACT")
            pack_hash_rows.append(f"{path}\t{item['sha256']}")
        pack_index.append(
            {
                "logical_id": parent_id,
                "path": pack["planned_target_root"],
                "sha256": hashlib.sha256(("\n".join(sorted(pack_hash_rows)) + "\n").encode()).hexdigest(),
                "byte_size": pack["total_bytes"],
                "semantic_role": pack["pack_id"],
                "file_count": pack["file_count"],
                "reproducibility_level": "R1_EXACT_ARTIFACT",
            }
        )

    add_record(records, "source.release.bundle", "frozen/original_release/source/REPRO_KIT_semantic_repair.bundle", "source_snapshot", [], "R0_PROVENANCE")
    add_record(records, "provenance.source_commit_verification", "provenance/source_commit_verification.json", "source_provenance", [], "R0_PROVENANCE")
    add_record(records, "provenance.scientific_dag_closure", "provenance/SCIENTIFIC_DAG_CLOSURE.json", "scientific_dag_closure", [], "R0_PROVENANCE")
    add_record(records, "provenance.rl_parent_graph", "provenance/RL_PARENT_GRAPH.json", "rl_parent_graph", ["pack:rl_stage2_compute_parent", "pack:rl_stage3_e2_parent", "pack:rl_stage4_bc_parent", "pack:rl_stage6_selector_eval", "pack:rl_c3e_e2_final"], "R2_FROZEN_REPLAY")
    add_record(records, "stage8.canonical.itt", "frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN/frozen_inputs/stage8/canonical_itt_observations.parquet", "stage8_canonical_96600", [], "R2_FROZEN_REPLAY")
    add_record(records, "frozen.distribution.result_registry_914", "frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip", "result_registry_914_and_frozen_distribution", [], "R2_FROZEN_REPLAY")
    for name in ["artifact_registry.json", "source_registry.json", "contract_registry.json", "claim_registry.json", "table_registry.csv"]:
        add_record(records, f"release.registry.{name}", f"frozen/release/submission/THESIS_V43_SUBMISSION_FINAL_20260917/{name}", "release_registry_or_binding", [], "R1_EXACT_ARTIFACT")

    records = sorted(records, key=lambda item: str(item["logical_id"]))
    sha_by_id = {str(item["logical_id"]): str(item["sha256"]) for item in records}
    sha_by_id.update({str(item["logical_id"]): str(item["sha256"]) for item in pack_index})
    canonical_rows = []
    for item in records:
        parents = sorted(str(p) for p in item["parents"])
        parent_hashes = sorted(sha_by_id[p] for p in parents)
        canonical_rows.append(f"{item['logical_id']}\t{item['sha256']}\t{','.join(parents)}\t{','.join(parent_hashes)}")
    root_hash = hashlib.sha256(("\n".join(canonical_rows) + "\n").encode()).hexdigest()
    certification_path = ROOT / "provenance/REMOTE_ASSET_CERTIFICATION.json"
    remote_certification: object = json.loads(certification_path.read_text(encoding="utf-8")) if certification_path.is_file() else "provenance/REMOTE_ASSET_CERTIFICATION.json (pending final remote run)"
    manifest = {
        "schema_version": "original_scientific_release_manifest_v1",
        "release_id": "THESIS_V43_ORIGINAL_SCIENTIFIC_RELEASE",
        "audit_date": "2026-09-19",
        "source_snapshot": "provenance.source_commit_verification",
        "environment": "pack:simulator_release",
        "restricted_input_manifests": ["pack:simulator_release", "pack:rl_stage2_compute_parent"],
        "sections": {
            "oracle": [p["logical_id"] for p in pack_index if "oracle" in str(p["logical_id"])],
            "simulator": ["pack:simulator_release"],
            "rl_stage2_to_stage6": [p["logical_id"] for p in pack_index if "rl_" in str(p["logical_id"])],
            "llm": ["pack:llm_final_plan3"],
            "stage8": ["stage8.canonical.itt"],
            "stage9": ["pack:stage8_stage9_evaluation"],
            "release_registry": [x["logical_id"] for x in records if str(x["logical_id"]).startswith("release.registry.")],
        },
        "pack_index": pack_index,
        "records": records,
        "root_hash_basis": "sorted logical_id + sha256 + sorted parent logical IDs + sorted parent SHA256 values; release metadata and root files excluded",
        "release_root_sha256": root_hash,
        "artifact_count": len(records),
        "total_bytes": sum(int(item["byte_size"]) for item in records),
        "dag_closure": "historical_selected_compute_parents_closed; restricted raw input and fresh execution remain external/open",
        "remote_lfs_certification": remote_certification,
        "frozen_replay_status": "CERTIFIED",
    }
    (ROOT / "frozen/original_release/ORIGINAL_RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (ROOT / "frozen/original_release/RELEASE_ROOT.txt").write_text(root_hash + "\n", encoding="utf-8")
    (ROOT / "frozen/original_release/release.json").write_text(json.dumps({k: manifest[k] for k in ["schema_version", "release_id", "release_root_sha256", "artifact_count", "total_bytes", "dag_closure", "remote_lfs_certification", "source_snapshot", "frozen_replay_status"]}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"release_root_sha256": root_hash, "artifact_count": len(records), "total_bytes": manifest["total_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
