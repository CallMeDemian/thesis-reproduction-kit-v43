"""Build byte-level migration and original-asset manifests.

This script is deliberately read-only with respect to the source repository.
It hashes source trees and the current product tree, then writes manifests to
the product repository. Large scientific assets are inventoried with their
planned destination but are not copied unless a separate migration command is
explicitly run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable


EXCLUDED_DIRS = {".git", ".pytest_cache", "__pycache__", ".venv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def record_tree(
    source_root: Path,
    source_rel: str,
    planned_target: str,
    semantic_role: str,
    *,
    selector=None,
) -> dict[str, object]:
    root = source_root / Path(source_rel)
    if not root.exists():
        return {
            "pack_id": semantic_role,
            "source_relative_root": source_rel.replace("\\", "/"),
            "planned_target_root": planned_target,
            "status": "MISSING",
            "file_count": 0,
            "total_bytes": 0,
            "files": [],
        }

    entries = []
    for path in files_under(root):
        rel = path.relative_to(root).as_posix()
        if selector is not None and not selector(rel):
            continue
        entries.append(
            {
                "source_relative_path": f"{source_rel.replace('\\', '/')}/{rel}",
                "planned_target_path": f"{planned_target.rstrip('/')}/{rel}",
                "byte_size": path.stat().st_size,
                "sha256": sha256(path),
                "semantic_role": semantic_role,
                "migration_status": "INVENTORIED_NOT_COPIED",
            }
        )
    return {
        "pack_id": semantic_role,
        "source_relative_root": source_rel.replace("\\", "/"),
        "planned_target_root": planned_target,
        "status": "FOUND_IN_SOURCE_MIGRATION_PENDING",
        "file_count": len(entries),
        "total_bytes": sum(int(item["byte_size"]) for item in entries),
        "files": entries,
    }


def record_file_list(
    source_root: Path,
    files: list[tuple[str, str]],
    pack_id: str,
) -> dict[str, object]:
    entries = []
    for source_rel, target_rel in files:
        path = source_root / Path(source_rel)
        if not path.is_file():
            raise FileNotFoundError(f"missing source artifact for {pack_id}: {source_rel}")
        entries.append(
            {
                "source_relative_path": source_rel.replace("\\", "/"),
                "planned_target_path": target_rel.replace("\\", "/"),
                "byte_size": path.stat().st_size,
                "sha256": sha256(path),
                "semantic_role": pack_id,
                "migration_status": "INVENTORIED_NOT_COPIED",
            }
        )
    return {
        "pack_id": pack_id,
        "source_relative_root": "multiple_release_critical_paths",
        "planned_target_root": str(Path(files[0][1]).parent).replace("\\", "/") if files else "",
        "status": "FOUND_IN_SOURCE_MIGRATION_PENDING",
        "file_count": len(entries),
        "total_bytes": sum(int(item["byte_size"]) for item in entries),
        "files": entries,
    }


def build_asset_manifest(source_root: Path, previous: dict[str, object] | None = None) -> dict[str, object]:
    evaluation_selector = lambda rel: any(
        "stage8" in part.lower() or "stage9" in part.lower()
        for part in rel.replace("\\", "/").split("/")
    )
    packs = [
        record_tree(
            source_root,
            "archive/DEPLOYED_RELEASE/stage0_oracle_foundation",
            "frozen/original_release/oracle/stage0_oracle_foundation",
            "oracle_stage0_foundation",
        ),
        record_tree(
            source_root,
            "archive/DEPLOYED_RELEASE/stage1_oracle_inputs",
            "frozen/original_release/oracle/stage1_oracle_inputs",
            "oracle_stage1_inputs",
        ),
        record_tree(
            source_root,
            "archive/DEPLOYED_RELEASE/stage1_oracle_backends",
            "frozen/original_release/oracle/stage1_oracle_backends",
            "oracle_stage1_backends",
        ),
        record_tree(
            source_root,
            "archive/DEPLOYED_RELEASE/stage5_candidate_iql/C3E_E2_7SEED_BALANCED_DFEBAFA6",
            "frozen/original_release/rl/C3E_E2_7SEED_BALANCED_DFEBAFA6",
            "rl_c3e_e2_final",
        ),
        record_tree(
            source_root,
            "archive/DEPLOYED_RELEASE/llm_runs/final_plan3",
            "frozen/original_release/llm/final_plan3",
            "llm_final_plan3",
        ),
        record_tree(
            source_root,
            "data/reproduction/llm_final_evaluation_20260913",
            "frozen/original_release/evaluation/llm_final_evaluation_20260913",
            "stage8_stage9_evaluation",
            selector=evaluation_selector,
        ),
        record_file_list(
            source_root,
            [
                (
                    f"src/credit_recourse/simulator/{path}",
                    f"frozen/original_release/simulator/src/credit_recourse/simulator/{path}",
                )
                for path in sorted(
                    str(p.relative_to(source_root / "src/credit_recourse/simulator")).replace("\\", "/")
                    for p in files_under(source_root / "src/credit_recourse/simulator")
                )
            ]
            + [
                (path, f"frozen/original_release/simulator/{path}")
                for path in [
                    "configs/current/contract_manifest.json",
                    "configs/current/final_freeze/final_action_contract.yaml",
                    "configs/current/final_freeze/final_candidate_library.yaml",
                    "configs/current/final_freeze/final_oracle_rl_contract.json",
                    "configs/current/final_freeze/frozen_python_environment.json",
                    "configs/current/final_freeze/reproduction_input_inventory.json",
                    "configs/current/final_freeze/simulator_ratio_alias_map.json",
                    "configs/current/final_freeze/stage2_action_source_map.json",
                    "configs/current/final_freeze/v43_rl.json",
                ]
            ],
            "simulator_release",
        ),
        record_tree(
            source_root,
            "archive/DEPLOYED_RELEASE/stage3_acd_ssl",
            "frozen/original_release/rl/stage3_e2",
            "rl_stage3_e2_parent",
        ),
        record_tree(
            source_root,
            "archive/DEPLOYED_RELEASE/stage4_candidate_bc",
            "frozen/original_release/rl/stage4_bc",
            "rl_stage4_bc_parent",
        ),
        record_tree(
            source_root,
            "archive/DEPLOYED_RELEASE/stage6_candidate_selector_eval",
            "frozen/original_release/rl/stage6",
            "rl_stage6_selector_eval",
        ),
        record_file_list(
            source_root,
            [
                (f"archive/DEPLOYED_RELEASE/stage2_candidate_projection/{path}", f"frozen/original_release/rl/stage2/{path}")
                for path in [
                    "candidate_action_contract_v4_3.json",
                    "candidate_action_contract_v4_3_manifest.json",
                    "candidate_library_metadata.json",
                    "candidate_projection_diagnostics.csv",
                    "candidate_projection_diagnostics.json",
                    "counterfactual_transitions_metadata__P50.json",
                    "final_candidate_library__P50.yaml",
                    "phase3_iql_candidate__P50.parquet",
                    "phase3_iql_candidate.parquet",
                    "phase3_iql_counterfactual_candidate__P50.parquet",
                    "phase3_iql_observed_factual__P50.parquet",
                    "input_splits/canonical_transition_manifest.json",
                    "input_splits/canonical_transition_row_ids.parquet",
                    "input_splits/canonical_evaluation_cohort_manifest.json",
                    "input_splits/canonical_evaluation_row_ids.parquet",
                    "runtime_inputs/business_plan_rate_v4/production/CANONICAL_INPUT_MANIFEST.json",
                    "runtime_inputs/business_plan_rate_v4/production/production_selected_rate_ledger.parquet",
                    "runtime_inputs/historical_financial_context_v4_3/preparation_manifest.json",
                    "runtime_inputs/historical_financial_context_v4_3/actual_context.parquet",
                    "runtime_inputs/historical_financial_context_v4_3/actual_financial_states.parquet",
                    "v4_3_runtime/01_contract/encoder_contract.json",
                    "v4_3_runtime/01_contract/resolved_run_config.json",
                    "v4_3_runtime/02_data/stage3_rows.parquet",
                    "v4_3_runtime/02_data/stage4_rows.parquet",
                    "v4_3_runtime/02_data/stage5_rows.parquet",
                ]
            ],
            "rl_stage2_compute_parent",
        ),
    ]
    previous_files = {
        item["source_relative_path"]: item
        for pack in (previous or {}).get("packs", [])
        for item in pack.get("files", [])
    }
    for pack in packs:
        for item in pack["files"]:
            old_item = previous_files.get(item["source_relative_path"])
            if old_item and old_item.get("migration_status") == "COPIED_AND_HASH_VERIFIED":
                item["migration_status"] = old_item["migration_status"]
                item["destination_relative_path"] = old_item.get("destination_relative_path", item["planned_target_path"])
        if pack["files"] and all(item["migration_status"] == "COPIED_AND_HASH_VERIFIED" for item in pack["files"]):
            pack["status"] = "COPIED_AND_HASH_VERIFIED"
    return {
        "schema_version": "original_asset_pack_manifest_v1",
        "audit_date": "2026-09-19",
        "source_repository": "CallMeDemian/REPRO_KIT_semantic_repair",
        "source_root_policy": "source_relative_paths_are_relative_to_the_original_repository_root",
        "copy_policy": "source trees are read-only; each pack records whether it is hash-verified in the product repository",
        "packs": packs,
        "summary": {
            "pack_count": len(packs),
            "file_count": sum(int(pack["file_count"]) for pack in packs),
            "total_bytes": sum(int(pack["total_bytes"]) for pack in packs),
        },
    }


def source_candidate(target_rel: str) -> str | None:
    mapping = [
        ("frozen/distribution/", "dist/"),
        ("frozen/evidence/v1.3/", "repro_audit/FINAL_RELEASE/"),
        ("frozen/release/submission/", "repro/releases/"),
        ("src/credit_recourse/", "src/credit_recourse/"),
    ]
    for target_prefix, source_prefix in mapping:
        if target_rel.startswith(target_prefix):
            return source_prefix + target_rel[len(target_prefix) :]
    return None


def build_migration_manifest(
    source_root: Path,
    target_root: Path,
    asset_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    asset_mapping = {
        item["planned_target_path"]: item["source_relative_path"]
        for pack in (asset_manifest or {}).get("packs", [])
        for item in pack.get("files", [])
    }
    entries = []
    for path in files_under(target_root):
        target_rel = path.relative_to(target_root).as_posix()
        candidate = source_candidate(target_rel) or asset_mapping.get(target_rel)
        source_path = source_root / candidate if candidate else None
        old_hash = sha256(source_path) if source_path and source_path.is_file() else None
        new_hash = sha256(path)
        if old_hash and old_hash == new_hash:
            action = "COPIED_BYTE_IDENTICAL"
        elif old_hash:
            action = "CURATED_OR_TRANSFORMED_FROM_SOURCE"
        else:
            action = "NEW_PRODUCT_FILE"
        entries.append(
            {
                "new_relative_path": target_rel,
                "new_byte_size": path.stat().st_size,
                "new_sha256": new_hash,
                "old_relative_path": candidate,
                "old_byte_size": source_path.stat().st_size if source_path and source_path.is_file() else None,
                "old_sha256": old_hash,
                "semantic_role": target_rel.split("/", 1)[0],
                "migration_action": action,
            }
        )
    return {
        "schema_version": "migration_manifest_v1",
        "audit_date": "2026-09-19",
        "source_repository": "CallMeDemian/REPRO_KIT_semantic_repair",
        "target_repository": "CallMeDemian/thesis-reproduction-kit-v43",
        "source_tree_is_read_only": True,
        "entries": entries,
        "summary": {
            "new_file_count": len(entries),
            "byte_identical_count": sum(item["migration_action"] == "COPIED_BYTE_IDENTICAL" for item in entries),
            "curated_or_transformed_count": sum(item["migration_action"] == "CURATED_OR_TRANSFORMED_FROM_SOURCE" for item in entries),
            "new_product_file_count": sum(item["migration_action"] == "NEW_PRODUCT_FILE" for item in entries),
        },
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    if not source_root.is_dir() or not target_root.is_dir():
        print("source and target roots must be directories", file=sys.stderr)
        return 2
    asset_manifest_path = target_root / "provenance/ASSET_PACK_MANIFEST.json"
    previous = json.loads(asset_manifest_path.read_text(encoding="utf-8")) if asset_manifest_path.exists() else None
    asset_manifest = build_asset_manifest(source_root, previous)
    for pack in asset_manifest["packs"]:
        for item in pack["files"]:
            target = target_root / Path(str(item["planned_target_path"]))
            if target.is_file() and sha256(target) == item["sha256"]:
                item["migration_status"] = "COPIED_AND_HASH_VERIFIED"
                item["destination_relative_path"] = item["planned_target_path"]
        if pack["files"] and all(item["migration_status"] == "COPIED_AND_HASH_VERIFIED" for item in pack["files"]):
            pack["status"] = "COPIED_AND_HASH_VERIFIED"
    write_json(asset_manifest_path, asset_manifest)
    write_json(target_root / "provenance/MIGRATION_MANIFEST.json", build_migration_manifest(source_root, target_root, asset_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
