from __future__ import annotations


def select(frame):
    return frame.loc[frame.analysis_id.eq("RQ5_INFORMATION")].copy()
