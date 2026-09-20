from __future__ import annotations

from pathlib import Path


V13_PACKAGE = Path("frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN")


def frozen_v13_root(root: Path) -> Path:
    """Return the certified V1.3 evidence package, never a legacy compute root."""
    return Path(root).resolve() / V13_PACKAGE
