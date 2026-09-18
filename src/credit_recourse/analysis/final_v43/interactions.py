from __future__ import annotations


IDS = ("BUDGET_INTERACTION", "REASONING_REGIME_INTERACTION", "EXECUTION_SENSITIVITY")


def select(frame):
    return frame.loc[frame.analysis_id.isin(IDS)].copy()
