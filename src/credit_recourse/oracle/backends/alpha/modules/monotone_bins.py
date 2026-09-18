"""Versioned empty-bin completion using only frozen development score tables.

Observed values are never missing merely because their bin had no DEV rows.
Populated scores, bin geometry, weights, grade cutoffs and actual-missing
imputation are immutable. No panel, rating labels or action outcomes are inputs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np

VERSION = "oracle_alpha_monotone_empty_bins_v1"
RULE = "linear_between_populated_bin_centers_constant_endpoint_step_lookup_v1"


def contract_hash(params):
    payload = {k: v for k, v in params.items() if k != "oracle_alpha_contract_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def direction_sign(params, variable):
    entry = (params.get("directions") or params.get("direction_encoding") or {}).get(variable)
    direction = entry.get("direction") if isinstance(entry, dict) else entry
    if direction in {"value_up_good", "higher_good", "higher_is_better"}:
        return 1
    if direction in {"value_down_good", "lower_good", "lower_is_better"}:
        return -1
    raise ValueError(f"Missing/unsupported declared direction: {variable}: {direction}")


def bin_geometry(info):
    edges = np.asarray(info["edges"], dtype=float)
    if not np.isfinite(edges).all() or len(edges) < 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("Invalid bin edges")
    discrete = bool(info.get("is_binary") or info.get("is_low_unique"))
    if discrete:
        cuts = (edges[:-1] + edges[1:]) / 2
        return edges, np.r_[-np.inf, cuts], np.r_[cuts, np.inf]
    if len(edges) < 2:
        raise ValueError("Continuous bins need at least two edges")
    return (edges[:-1] + edges[1:]) / 2, edges[:-1], edges[1:]


def complete_table(info, table, sign):
    centers, _, _ = bin_geometry(info)
    keys = sorted(int(k) for k in table)
    if not keys or len(keys) != len(set(keys)) or min(keys) < 0 or max(keys) >= len(centers):
        raise ValueError("Missing/invalid populated development bins")
    scores = np.array([float(table[str(k)] if str(k) in table else table[k]) for k in keys])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 100)):
        raise ValueError("Nonfinite/out-of-range development scores")
    if np.any(sign * np.diff(scores) < 0):
        raise ValueError("Populated development scores violate declared monotonicity; stop for separate audit")
    values = np.interp(centers, centers[keys], scores)
    return {str(i): float(value) for i, value in enumerate(values)}


def freeze_monotone_params(parent):
    if "oracle_alpha_contract_version" in parent or "empty_bin_repair" in parent:
        raise ValueError("Expected unmodified legacy development parameters")
    if str(parent.get("item_score_interpolation", "step")).lower() not in {"step", "none", "bin"}:
        raise ValueError("This repair preserves step scorecard semantics only")
    result = deepcopy(parent)
    # Exported canonical schema only: do not permit an alias to shadow repair.
    if "iso_tables" in parent or "directions" in parent:
        raise ValueError("Use canonical exported parameter schema")
    original = parent["bin_score_table_isotonic"]
    for vid in parent["selected_variables"]:
        result["bin_score_table_isotonic"][vid] = complete_table(
            parent["bin_edges"][vid], original[vid], direction_sign(parent, vid))
    result["oracle_alpha_contract_version"] = VERSION
    result["empty_bin_repair"] = {
        "rule": RULE, "parent_parameter_content_hash": contract_hash(parent),
        "fit_source": "frozen final DEV 2002-2019 isotonic tables and bin edges",
        "development_populated_tables": deepcopy(original),
        "evaluation_data_used_for_rule_or_scores": False,
        "observed_empty_bin_imputed": False, "actual_missing_imputation_changed": False,
        "populated_scores_changed": False, "weights_or_grade_cutoffs_refit": False,
    }
    result["oracle_alpha_contract_hash"] = contract_hash(result)
    validate_monotone_params(result)
    return result


def validate_monotone_params(params):
    if params.get("oracle_alpha_contract_version") != VERSION:
        raise ValueError("Unknown Alpha semantic contract")
    if params.get("oracle_alpha_contract_hash") != contract_hash(params):
        raise ValueError("Alpha semantic contract hash mismatch")
    repair = params["empty_bin_repair"]
    if repair["rule"] != RULE or repair["evaluation_data_used_for_rule_or_scores"] is not False:
        raise ValueError("Invalid empty-bin rule/lineage")
    if "iso_tables" in params or str(params.get("item_score_interpolation", "step")).lower() not in {"step", "none", "bin"}:
        raise ValueError("Alias/interpolation cannot override frozen step completion")
    for vid in params["selected_variables"]:
        expected = complete_table(params["bin_edges"][vid], repair["development_populated_tables"][vid],
                                  direction_sign(params, vid))
        if params["bin_score_table_isotonic"][vid] != expected:
            raise ValueError(f"Alpha completed table mismatch: {vid}")
