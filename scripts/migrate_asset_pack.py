"""Copy one inventoried asset pack and verify every source/destination hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--pack-id", required=True)
    args = parser.parse_args()
    manifest_path = args.target_root / "provenance/ASSET_PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pack = next(item for item in manifest["packs"] if item["pack_id"] == args.pack_id)
    copied = []
    for item in pack["files"]:
        source = args.source_root / Path(item["source_relative_path"])
        target = args.target_root / Path(item["planned_target_path"])
        actual_source_hash = sha256(source)
        if actual_source_hash != item["sha256"]:
            raise RuntimeError(f"source hash changed: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        actual_target_hash = sha256(target)
        if actual_target_hash != item["sha256"]:
            raise RuntimeError(f"destination hash mismatch: {target}")
        item["migration_status"] = "COPIED_AND_HASH_VERIFIED"
        item["destination_relative_path"] = item["planned_target_path"]
        copied.append(item["planned_target_path"])
    pack["status"] = "COPIED_AND_HASH_VERIFIED"
    manifest["migration_receipts"] = manifest.get("migration_receipts", [])
    manifest["migration_receipts"].append(
        {
            "pack_id": args.pack_id,
            "status": "COPIED_AND_HASH_VERIFIED",
            "file_count": len(copied),
            "target_root": pack["planned_target_root"],
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"pack_id": args.pack_id, "status": pack["status"], "file_count": len(copied)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
