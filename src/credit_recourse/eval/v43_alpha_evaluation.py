from __future__ import annotations

import pandas as pd


def _prepare_stage6_context_lookup(*args, **kwargs):
    return {}


def _target_operating_loss_freq_3y(*args, **kwargs):
    return float("nan")


def _target_cap_change_count_3y(*args, **kwargs):
    return float("nan")


def _state_to_frame_dict(state, *args, **kwargs):
    if hasattr(state, "to_dict"):
        return state.to_dict()
    return dict(state) if isinstance(state, dict) else {}
