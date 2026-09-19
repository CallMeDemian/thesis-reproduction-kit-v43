"""Fresh Oracle-Alpha contract namespace.

The complete parameter payload is an input artifact of an actual run.  This
module only defines the stable path/version seam used by verifiers and makes
the absence of that payload explicit instead of resolving an archive path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ALPHA_CONTRACT_VERSION = "fresh_v43_alpha_contract_v1"
CANONICAL_ALPHA_CONTRACT_PATH = Path("contracts/scientific/v43_alpha_contract.json")
CANONICAL_V43_ORACLE_REGISTRY_PATH = Path("contracts/scientific/v43_oracle_registry.json")
ALPHA_CONTRACT_PROMOTION_MANIFEST_PATH = Path("runs/_runtime_audit/alpha_contract_promotion_manifest.json")
ALPHA_STRICT_VERIFIER_PATH = Path("runs/_runtime_audit/alpha_strict_verifier_report.json")
EMPTY_BIN_REPAIR_RULE = "fail_if_empty_unless_explicitly_repaired"
SELECTED_VARIABLES = ()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v43_alpha_contract(root: Path):
    path = Path(root) / CANONICAL_ALPHA_CONTRACT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"fresh Alpha parameter artifact is required: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, file_sha256(path)


EXPECTED_ALPHA_CONTENT_HASH = "UNMATERIALIZED_FRESH_ALPHA_CONTENT"
EXPECTED_ALPHA_CONTRACT_SHA256 = "UNMATERIALIZED_FRESH_ALPHA_SHA256"
