"""Build the exact parent graph for the preserved 28-member C3-E cohort."""

from __future__ import annotations

import csv
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


def tree_digest(root: Path) -> str:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append(f"{path.relative_to(root).as_posix()}\t{sha256(path)}")
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()


def manifest_map(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["sha256"]: item for item in data["artifacts"]}


def main() -> int:
    stage3 = manifest_map(ROOT / "frozen/original_release/rl/stage3_e2/deployed_ensemble_e2_7seed/DEPLOYMENT_MANIFEST.json")
    stage4 = manifest_map(ROOT / "frozen/original_release/rl/stage4_bc/deployed_ensemble_e2_7seed/DEPLOYMENT_MANIFEST.json")
    stage2_root = ROOT / "frozen/original_release/rl/stage2"
    stage2_digest = tree_digest(stage2_root)
    stage6_root = ROOT / "frozen/original_release/rl/stage6/C3E_E2_7SEED_BALANCED_DFEBAFA6"
    stage6_file = stage6_root / "firm_action_oracle_payoffs.parquet"
    stage6_hash = sha256(stage6_file)
    rows = list(csv.DictReader((ROOT / "frozen/original_release/rl/C3E_E2_7SEED_BALANCED_DFEBAFA6/member_provenance.csv").open(encoding="utf-8-sig")))
    members = []
    for row in rows:
        stage5_rel = Path(row["canonical_checkpoint_path"]).relative_to("data/final_freeze/stage5_candidate_iql")
        stage5_path = ROOT / "frozen/original_release/rl" / stage5_rel
        s3 = stage3[row["stage3_checkpoint_hash"]]
        s4 = stage4[row["stage4_checkpoint_hash"]]
        stage3_path = ROOT / "frozen/original_release/rl/stage3_e2" / Path(s3["canonical_path"]).relative_to("data/final_freeze/stage3_acd_ssl")
        stage4_path = ROOT / "frozen/original_release/rl/stage4_bc" / Path(s4["canonical_path"]).relative_to("data/final_freeze/stage4_candidate_bc")
        checks = {
            "stage2_parent_pack": stage2_root.is_dir() and any(stage2_root.iterdir()),
            "stage3": stage3_path.is_file() and sha256(stage3_path) == row["stage3_checkpoint_hash"],
            "stage4": stage4_path.is_file() and sha256(stage4_path) == row["stage4_checkpoint_hash"],
            "stage5": stage5_path.is_file() and sha256(stage5_path) == row["canonical_checkpoint_sha256"],
            "stage6": stage6_file.is_file() and stage6_hash == "2be6cc91f3b81a2640141d5dda995ea2f23d9158961b59bfd4549c8b3d39d70a",
        }
        members.append(
            {
                "logical_id": f"rl.member.{row['configuration']}.seed{row['seed']}",
                "configuration": row["configuration"],
                "seed": int(row["seed"]),
                "actor_semantic_hash": row["actor_semantic_hash"],
                "decision_sha256": row["decision_sha256"],
                "parent_logical_ids": [
                    "rl.stage2.compute_parent",
                    f"rl.stage3.e2.seed{row['seed']}",
                    f"rl.stage4.bc.seed{row['seed']}",
                    "rl.stage6.selector_eval",
                ],
                "parents": [
                    {
                        "logical_id": "rl.stage2.compute_parent",
                        "path": "frozen/original_release/rl/stage2",
                        "sha256": stage2_digest,
                        "role": "candidate_projection_and_counterfactual_inputs",
                    },
                    {
                        "logical_id": f"rl.stage3.e2.seed{row['seed']}",
                        "path": stage3_path.relative_to(ROOT).as_posix(),
                        "sha256": row["stage3_checkpoint_hash"],
                        "role": "E2_encoder_checkpoint",
                    },
                    {
                        "logical_id": f"rl.stage4.bc.seed{row['seed']}",
                        "path": stage4_path.relative_to(ROOT).as_posix(),
                        "sha256": row["stage4_checkpoint_hash"],
                        "role": "behavior_cloning_checkpoint",
                    },
                    {
                        "logical_id": f"rl.stage5.c3e.{row['configuration']}.seed{row['seed']}",
                        "path": stage5_path.relative_to(ROOT).as_posix(),
                        "sha256": row["canonical_checkpoint_sha256"],
                        "role": "preserved_final_actor",
                    },
                    {
                        "logical_id": "rl.stage6.selector_eval",
                        "path": stage6_file.relative_to(ROOT).as_posix(),
                        "sha256": stage6_hash,
                        "role": "C3-E selector/payoff evaluation",
                    },
                ],
                "checks": checks,
                "closure_status": "CLOSED" if all(checks.values()) else "MISSING_PARENT",
                "reproducibility_level": "R2_FROZEN_REPLAY",
            }
        )
    graph = {
        "schema_version": "rl_parent_graph_v1",
        "audit_date": "2026-09-19",
        "release_id": "C3E_E2_7SEED_BALANCED_DFEBAFA6",
        "shared_parent_packs": {
            "stage2": {"path": "frozen/original_release/rl/stage2", "sha256": stage2_digest},
            "stage6": {"path": stage6_file.relative_to(ROOT).as_posix(), "sha256": stage6_hash},
        },
        "members": members,
        "summary": {
            "member_count": len(members),
            "closed_member_count": sum(m["closure_status"] == "CLOSED" for m in members),
            "missing_parent_member_count": sum(m["closure_status"] == "MISSING_PARENT" for m in members),
            "actor_checkpoint_count": 28,
            "decision_sha256": "dfebafa6b50e55c2f5019766e0e540af9964b2a9ab7d4e49805d03e8011b792e",
            "status": "PASS" if len(members) == 28 and all(m["closure_status"] == "CLOSED" for m in members) else "FAIL",
        },
    }
    out = ROOT / "provenance/RL_PARENT_GRAPH.json"
    out.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(graph["summary"], indent=2))
    return 0 if graph["summary"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
