from __future__ import annotations

import pandas as pd


def load(root):
    path = root / "frozen_replay/v1.3/canonical_v13/parent_gate_sensitivity_v13.csv"
    frame = pd.read_csv(path)
    if len(frame) != 96:
        raise ValueError("parent gate must contain 96 descriptive rows")
    return frame
