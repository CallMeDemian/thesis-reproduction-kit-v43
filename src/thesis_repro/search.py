from __future__ import annotations

from pathlib import Path
from typing import Any


def reproduce_search(root: Path, *, mode: str = "all") -> dict[str, Any]:
    """Expose the preserved historical search runtime without inventing a new grid.

    The final source contains the exact Stage5 search implementation, but this
    distribution does not contain a certified, immutable search baseline.  A
    missing baseline is an input boundary, never permission to tune a new one.
    """
    root = Path(root).resolve()
    baseline_candidates = sorted(root.glob("runs/**/V43_STAGE5_SEARCH_BASELINE.json"))
    return {
        "status": "INPUT_REQUIRED",
        "scope": "ORIGINAL_SEARCH_REPRODUCTION",
        "mode": mode,
        "required_artifact": "V43_STAGE5_SEARCH_BASELINE.json",
        "candidates": [str(path.relative_to(root)).replace("\\", "/") for path in baseline_candidates],
        "preserved_runtime": "src/credit_recourse/rl/v43_stage5_search.py",
        "preserved_audit": "src/credit_recourse/rl/v43_optimizer_audit.py",
        "reason": "no certified immutable historical search baseline is present; refusing to expand the search space or start a new optimization campaign",
        "training_runs": 0,
    }
