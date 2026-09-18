"""Continuous observed-action reconstruction, independent of candidate/scoring code."""
from __future__ import annotations

from dataclasses import fields
import math

from .business_plan import calibrate_business_plan
from .executed_primitives import (ExecutedFinancialPrimitives, OPERATING_FIELDS,
                                  capex_baseline, decompose_debt, operating_levels, stable_hash)
from .firm_state import FirmState

RECONSTRUCTION_VERSION = "historical_v4_1_reconstruction_v1"
ELIGIBILITY_RULE = {
    "version": RECONSTRUCTION_VERSION,
    "year": "consecutive observed opening/closing years; split by actual closing year: DEV<=2019, OOT=2020..2023, POST>=2024",
    "bp": "canonical calibrated history for same firm, last 3 available observations <=base_year; nonempty; no target-year data",
    "capex": "finite source target CAPEX, abs(raw acquisition cash-flow amount) as existing Stage2 source contract; no missing replacement",
    "debt": "all four debt accounts finite/nonnegative at both dates; legacy percentage mask retained; explicit observed 0->0 amount is valid although legacy ratio is undefined",
    "operating": "all six legacy observation masks true; finite nonnegative operating levels; strictly positive opening/closing turnover denominators and opening revenue",
    "other": "finite nonnegative opening PPE; no negative reported stocks; no candidate bounds or nearest-candidate projection",
    "zero_growth": "amounts remain defined; normalized growth adjustment is null when BP growth is exactly zero",
    "maintenance": "actual maintenance=min(actual_capex,BP_maintenance); maintenance shortfall stored separately and included in CAPEX recomposition",
    "attrition_precedence": ["missing_capex", "missing_debt_stack", "missing_business_plan", "other"],
    "selection_on_simulation_success_or_scores": False,
}


def observed(value):
    return value is not None and math.isfinite(float(value))


def calibrated_plan(history, *, firm_id, base_year):
    if any(s.firm_id != firm_id for s in history):
        raise ValueError("Business-plan history firm mismatch")
    selected = sorted((s for s in history if s.year <= base_year), key=lambda s: s.year)[-3:]
    if len({s.year for s in selected}) != len(selected):
        raise ValueError("Duplicate canonical firm/year history")
    if not selected:
        return None, []
    # No rating prediction or target observation: canonical calibrator's existing
    # missing-grade default is used, exactly as the frozen financial history path.
    return calibrate_business_plan(selected), [s.year for s in selected]


def reconstruct(opening: FirmState, closing: FirmState | None, bp, legacy_masks: dict,
                *, bp_years: list[int], bp_source_hash: str):
    """Return (auditable row, complete primitive or None); missing never means zero."""
    if closing is not None and (closing.firm_id != opening.firm_id or closing.year != opening.year + 1):
        raise ValueError("Historical reconstruction requires exact consecutive firm/year alignment")
    if any(y > opening.year for y in bp_years) or (bp is not None and not bp_years):
        raise ValueError("Invalid canonical BP temporal lineage")
    row = {"row_id": f"{opening.firm_id}:{opening.year}:{opening.year+1}",
           "firm_id": opening.firm_id, "base_year": opening.year, "target_year": opening.year + 1,
           "split": "DEV" if opening.year + 1 <= 2019 else "OOT" if opening.year + 1 <= 2023 else "POST",
           "producer_version": RECONSTRUCTION_VERSION, "bp_source_hash": bp_source_hash,
           "bp_history_years": ",".join(map(str, bp_years)),
           "bp_max_history_year": max(bp_years) if bp_years else None,
           "eligibility_rule_hash": stable_hash(ELIGIBILITY_RULE)}
    for label, state in (("opening", opening), ("closing", closing)):
        for f in fields(FirmState):
            if f.name in {"firm_id", "year", "sector", "rating_num", "rating_grade"}:
                continue
            value = getattr(state, f.name) if state is not None else None
            row[f"{label}__{f.name}"] = float(value) if observed(value) else None
    row.update({"legacy_action_observed__" + k: bool(v) for k, v in legacy_masks.items()})
    row.update({"bp__" + k: v for k, v in bp.to_dict().items()} if bp is not None else {})
    values = {f.name: None for f in fields(ExecutedFinancialPrimitives)
              if f.name not in {"source_kind", "version"}}
    reasons = []
    missing_capex = closing is None or not observed(closing.capex)
    if missing_capex:
        reasons.append("missing_capex")
    capex_ready = (not missing_capex and bp is not None and observed(opening.ppe)
                   and opening.ppe >= 0 and observed(opening.revenue) and opening.revenue > 0)
    row["observation_mask_capex"] = not missing_capex
    if capex_ready:
        base = capex_baseline(opening, bp)
        actual = abs(float(closing.capex))
        actual_maintenance = min(base["maintenance_capex"], actual)
        actual_growth = max(actual - base["maintenance_capex"], 0.0)
        values.update(executed_capex=actual,
                      growth_capex_adjustment=base["bp_growth_capex"] - actual_growth,
                      maintenance_capex_adjustment=base["maintenance_capex"] - actual_maintenance)
        row.update(base, actual_capex_source_signed=float(closing.capex), actual_capex=actual,
                   actual_maintenance_capex=actual_maintenance, actual_growth_capex=actual_growth,
                   capex_below_maintenance=actual < base["maintenance_capex"],
                   zero_bp_growth_capex=base["bp_growth_capex"] == 0,
                   observed_growth_capex_action=(values["growth_capex_adjustment"] / base["bp_growth_capex"]
                                                if base["bp_growth_capex"] > 0 else None),
                   total_capex_adjustment=base["bp_capex"]-actual)
        row["capex_roundtrip_error"] = (base["maintenance_capex"]-values["maintenance_capex_adjustment"]
                                       +base["bp_growth_capex"]-values["growth_capex_adjustment"]-actual)
    stacks = ("short_term_debt", "long_term_debt", "current_portion_long_debt", "bonds")
    debt_observed = closing is not None and all(observed(getattr(s, f)) for s in (opening, closing) for f in stacks)
    if not debt_observed:
        reasons.append("missing_debt_stack")
    debt_valid = debt_observed and all(getattr(s, f) >= 0 for s in (opening, closing) for f in stacks)
    row["observation_mask_debt_stack"] = debt_observed
    if debt_valid:
        old = {"short": opening.short_term_debt, "gross_ltd": opening.long_term_debt+opening.current_portion_long_debt,
               "bond": opening.bonds}
        new = {"short": closing.short_term_debt, "gross_ltd": closing.long_term_debt+closing.current_portion_long_debt,
               "bond": closing.bonds}
        # Preserve masks; an exactly observed zero pair is a known amount delta,
        # not a missing legacy percentage replaced with a zero action.
        for label, field, mask in (("short", "short_term_debt", "short_debt_pct"),
                                   ("gross_ltd", "long_term_debt", "long_debt_pct"),
                                   ("bond", "bonds", "bond_pct")):
            zero_pair = getattr(opening, field) == 0 and getattr(closing, field) == 0
            ok = bool(legacy_masks.get(mask, False)) or zero_pair
            row["observation_mask_"+label+"_principal"] = ok
            row["legacy_ratio_undefined_observed_zero_pair_"+label] = zero_pair and not legacy_masks.get(mask, False)
            debt_valid = debt_valid and ok
        delta = {k: new[k]-old[k] for k in old}
        row.update({"opening_"+k: old[k] for k in old})
        row.update({"closing_"+k: new[k] for k in old})
        row.update({"observed_delta_"+k: delta[k] for k in old})
        if debt_valid:
            values.update(decompose_debt(delta["short"], delta["gross_ltd"], delta["bond"]))
            rf = values["refi_short_to_ltd_amount"]+values["refi_short_to_bond_amount"]
            row.update(observed_rf_total=rf, observed_rf_to_ltd=values["refi_short_to_ltd_amount"],
                       observed_rf_to_bond=values["refi_short_to_bond_amount"])
            row["observed_rf_over_opening_short"] = rf/old["short"] if old["short"] > 0 else None
            row["observed_rf_ltd_over_opening_short"] = values["refi_short_to_ltd_amount"]/old["short"] if old["short"] > 0 else None
            row["observed_rf_bond_over_opening_short"] = values["refi_short_to_bond_amount"]/old["short"] if old["short"] > 0 else None
            names = {"short": "short_debt_principal_change", "gross_ltd": "gross_ltd_principal_change", "bond": "bond_principal_change"}
            recovered = {"short": values[names["short"]]-rf,
                         "gross_ltd": values[names["gross_ltd"]]+values["refi_short_to_ltd_amount"],
                         "bond": values[names["bond"]]+values["refi_short_to_bond_amount"]}
            for k, name in names.items():
                row["residual_"+name] = values[name]
                row[name+"_over_opening_stack"] = values[name]/old[k] if old[k] > 0 else None
                row["debt_roundtrip_error_"+k] = recovered[k]-delta[k]
    if bp is None:
        reasons.append("missing_business_plan")
    old_levels = operating_levels(opening)
    new_levels = operating_levels(closing) if closing is not None else {k: None for k in OPERATING_FIELDS}
    for name in OPERATING_FIELDS:
        mask = name + "_chg"
        ok = (bool(legacy_masks.get(mask, False)) and observed(old_levels[name]) and observed(new_levels[name])
              and old_levels[name] >= 0 and new_levels[name] >= 0)
        # Zero turnover with nonzero numerator cannot reproduce an account level.
        if "turnover" in name:
            ok = ok and old_levels[name] > 0 and new_levels[name] > 0
        row["observation_mask_"+name] = ok
        if ok:
            values[name+"_change"] = new_levels[name]-old_levels[name]
            values[name+"_target"] = new_levels[name]
            row["observed_"+name+"_change"] = values[name+"_change"]
    rev_ok = (closing is not None and bool(legacy_masks.get("revenue_growth", False))
              and observed(opening.revenue) and opening.revenue > 0 and observed(closing.revenue) and closing.revenue >= 0)
    row["observation_mask_revenue_growth"] = rev_ok
    if rev_ok:
        values["revenue_growth"] = (closing.revenue-opening.revenue)/opening.revenue
        row["observed_revenue_growth"] = values["revenue_growth"]
    other = (not debt_valid and debt_observed) or (not capex_ready and bp is not None and not missing_capex)
    other = other or not rev_ok or any(values[k+"_change"] is None for k in OPERATING_FIELDS)
    if other:
        reasons.append("other")
    row.update({"excluded_"+r: r in reasons for r in ELIGIBILITY_RULE["attrition_precedence"]})
    row["reconstruction_status"] = "ELIGIBLE" if not reasons else "EXCLUDED"
    row["reconstruction_reason"] = ";".join(reasons) if reasons else "complete_observed_primitives"
    row["primary_exclusion_reason"] = reasons[0] if reasons else "none"
    row.update({"primitive__"+k: v for k, v in values.items()})
    row.update({"observation_mask_primitive__"+k: observed(v) for k, v in values.items()})
    primitive = ExecutedFinancialPrimitives(**values, source_kind="historical") if not reasons else None
    row["primitive_hash"] = primitive.content_hash if primitive is not None else None
    return row, primitive
