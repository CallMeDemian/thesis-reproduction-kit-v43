from __future__ import annotations

import pandas as pd

from .paths import frozen_v13_root


def load(root):
    path = frozen_v13_root(root) / "canonical_v13/parent_gate_sensitivity_v13.csv"
    frame = pd.read_csv(path)
    if len(frame) != 96:
        raise ValueError("parent gate must contain 96 descriptive rows")
    return frame
