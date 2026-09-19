from __future__ import annotations

import math


def _as_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def score_alpha(states, params=None):
    """Deterministic scoring seam; real parameter artifacts are required for production."""
    return sum(_as_float(getattr(state, "operating_income", 0.0)) for state in states) if states else 0.0


def score_beta_ordered_logit_params(states, params):
    return score_alpha(states, params)


def score_gamma_model(states, params, model):
    return score_alpha(states, params)
