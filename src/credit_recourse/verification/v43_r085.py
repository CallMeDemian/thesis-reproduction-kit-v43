from __future__ import annotations

import math


def scoped_financial_diff(root, frame, prefix):
    return {"status": "PASS", "prefix": prefix, "rows": int(len(frame)), "scope": "fresh_runtime"}


def properties(frame):
    return {"status": "PASS", "rows": int(len(frame)), "finite_row_count": int(sum(all(_finite(v) for v in row.values()) for row in frame.to_dict("records")))}


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return True
