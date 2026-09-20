from __future__ import annotations

import pandas as pd

from .paths import frozen_v13_root


EXPECTED_COUNTS = {
    "PP_CONTRAST": 336,
    "SUPPLEMENTAL_HISTORICAL_ROW": 276,
    "EXECUTION_SENSITIVITY": 48,
    "REASONING_REGIME_INTERACTION": 24,
    "RQ4_ACTION_SPACE": 18,
    "RQ5_INFORMATION": 12,
    "BUDGET_INTERACTION": 8,
}


def load(root):
    path = frozen_v13_root(root) / "canonical_v13/V13_SUPPLEMENTAL_RESULTS.csv"
    frame = pd.read_csv(path)
    if len(frame) != 722 or frame.analysis_id.value_counts().to_dict() != EXPECTED_COUNTS:
        raise ValueError("supplemental 722-row contract mismatch")
    if frame.key_json.map(lambda x: __import__("hashlib").sha256(str(x).encode("utf-8")).hexdigest()).ne(frame.key_sha256).any():
        raise ValueError("supplemental canonical key hash mismatch")
    return frame
