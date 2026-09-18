"""Semantic v4 action operators for the equal-intensity nine-action contract.

This module is deliberately independent of Oracle code.  It consumes only a
factual :class:`FirmState`, a training-history :class:`BusinessPlan`, a frozen
candidate vector, and frozen historical DL/RF allocation constants.

The legacy ``ppe_pct``/independent-debt-percentage simulator remains available
for reproducing v3 artifacts.  New v4 consumers must use this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from credit_recourse.simulator.action import ACTION_BOUNDS
from credit_recourse.simulator.business_plan import BusinessPlan
from credit_recourse.simulator.dl_interest_timing import (
    DLInterestTiming,
    dl_interest_savings,
)
from credit_recourse.simulator.financial_simulator import (
    operating_plan_from_deltas, classify_sustainability,
    SimulationResult,
    _finite_ratio,
    _safe_get,
)
from credit_recourse.simulator.firm_state import FirmState
from credit_recourse.simulator.executed_primitives import (
    ExecutedFinancialPrimitives, PRIMITIVE_VERSION, OPERATING_FIELDS, capex_baseline, operating_levels,
)


ACTION_SEMANTIC_CONTRACT_VERSION_V4 = "candidate_action_v4_equal_intensity_semantic9"
SEMANTIC_FINANCIAL_SIMULATOR_V4_VERSION = "semantic_financial_simulator_v4_v7_bounded_endpoint_realization"
CANDIDATE_POLICY_MODE = "CANDIDATE_POLICY_MODE"
HISTORICAL_REPLAY_MODE = "HISTORICAL_REPLAY_MODE"
HISTORICAL_REPLAY_EXECUTION_VERSION = "historical_v4_1_replay_observed_financing_v1"
V4_ACTION_DIMENSIONS: tuple[str, ...] = (
    "growth_capex_reduction_pct",
    "deleveraging_total_debt_pct",
    "refinancing_short_debt_pct",
    "inv_turnover_chg",
    "ar_turnover_chg",
    "ap_turnover_chg",
    "cogs_ratio_chg",
    "sga_ratio_chg",
)


@dataclass(frozen=True)
class SemanticActionV4:
    growth_capex_reduction_pct: float = 0.0
    deleveraging_total_debt_pct: float = 0.0
    refinancing_short_debt_pct: float = 0.0
    inv_turnover_chg: float = 0.0
    ar_turnover_chg: float = 0.0
    ap_turnover_chg: float = 0.0
    cogs_ratio_chg: float = 0.0
    sga_ratio_chg: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SemanticActionV4":
        # Revenue growth belongs to BusinessPlan/scenario state.  Accepting a
        # zero compatibility column keeps frozen candidate tables readable;
        # any non-zero candidate input is a contract violation.
        for key in ("action__revenue_growth", "revenue_growth"):
            if key in row:
                raw_growth = float(row.get(key) or 0.0)
                if not math.isfinite(raw_growth) or abs(raw_growth) > 1.0e-15:
                    raise ValueError(
                        "revenue_growth is exogenous BusinessPlan state, not a v4 candidate action"
                    )
        values: dict[str, float] = {}
        for name in V4_ACTION_DIMENSIONS:
            raw = row.get(f"action__{name}", row.get(name, 0.0))
            value = float(raw or 0.0)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite v4 action value: {name}={raw!r}")
            values[name] = value
        action = cls(**values)
        if action.deleveraging_total_debt_pct > 0 and action.refinancing_short_debt_pct > 0:
            raise ValueError("A v4 candidate cannot deleverage and refinance in the same vector")
        if not 0.0 <= action.growth_capex_reduction_pct <= 1.0:
            raise ValueError("growth_capex_reduction_pct must be in [0,1]")
        if not 0.0 <= action.deleveraging_total_debt_pct <= 1.0:
            raise ValueError("deleveraging_total_debt_pct must be in [0,1]")
        if not 0.0 <= action.refinancing_short_debt_pct <= 1.0:
            raise ValueError("refinancing_short_debt_pct must be in [0,1]")
        return action


def _normalized_weights(raw: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, float]:
    out = {name: max(0.0, float(raw.get(name, 0.0))) for name in names}
    total = sum(out.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"Invalid frozen allocation weights: {out}")
    return {name: value / total for name, value in out.items()}


class SemanticFinancialSimulatorV4:
    """One-year financial simulator with fixed CX/DL/RF v4 semantics."""

    def __init__(
        self,
        *,
        dl_allocation: Mapping[str, Any],
        rf_allocation: Mapping[str, Any],
        contract_hash: str,
        intensity_scale_by_dimension: Mapping[str, Any],
        liquidity_reserve_cash_to_cogs: float,
        accounting_incremental_residual_abs_tolerance: float,
        accounting_incremental_residual_rel_tolerance: float,
        liquidity_cash_abs_tolerance: float = 1.0e-6,
        preserve_current_non_current_residual: bool = True,
        dl_interest_timing: DLInterestTiming | None = None,
    ) -> None:
        self.dl = _normalized_weights(
            dl_allocation, ("short_debt_weight", "gross_ltd_weight", "bond_weight")
        )
        self.rf = _normalized_weights(rf_allocation, ("long_weight", "bond_weight"))
        self.contract_hash = str(contract_hash)
        # None preserves v4. Only the hashed v4.1 factory opts into timing;
        # the primitive never receives a candidate ID or an evaluation score.
        self.dl_interest_timing = dl_interest_timing
        self.preserve_current_non_current_residual = bool(preserve_current_non_current_residual)
        self.reserve_ratio = float(liquidity_reserve_cash_to_cogs)
        if not math.isfinite(self.reserve_ratio) or self.reserve_ratio < 0.0:
            raise ValueError("liquidity reserve ratio must be finite and non-negative")
        self.accounting_abs_tol = float(accounting_incremental_residual_abs_tolerance)
        self.accounting_rel_tol = float(accounting_incremental_residual_rel_tolerance)
        self.liquidity_cash_abs_tol = float(liquidity_cash_abs_tolerance)
        if not math.isfinite(self.liquidity_cash_abs_tol) or self.liquidity_cash_abs_tol < 0.0:
            raise ValueError("liquidity cash absolute tolerance must be finite and non-negative")
        if not math.isfinite(self.accounting_abs_tol) or self.accounting_abs_tol < 0.0:
            raise ValueError("accounting absolute tolerance must be finite and non-negative")
        if not math.isfinite(self.accounting_rel_tol) or self.accounting_rel_tol < 0.0:
            raise ValueError("accounting relative tolerance must be finite and non-negative")
        if set(intensity_scale_by_dimension) != set(V4_ACTION_DIMENSIONS):
            raise ValueError("v4 intensity scale keys must exactly match action dimensions")
        self.intensity_scale = {
            name: float(intensity_scale_by_dimension[name]) for name in V4_ACTION_DIMENSIONS
        }
        if any((not math.isfinite(v)) or v <= 0.0 for v in self.intensity_scale.values()):
            raise ValueError("v4 intensity scales must be finite and positive")

    @staticmethod
    def _earnings(bp, *, revenue, cogs, sga, interest, fixed_cash_dividends=None):
        operating_income = revenue - cogs - sga
        non_op_income = revenue * bp.non_op_income_ratio
        non_interest_cost = float(bp.non_interest_financial_cost)
        if not math.isfinite(non_interest_cost) or non_interest_cost < 0.:
            raise ValueError('Non-interest financial cost must be finite and non-negative')
        broad_financial_cost = interest + non_interest_cost
        pretax = operating_income - broad_financial_cost + non_op_income
        tax = max(0.0, pretax) * bp.tax_rate
        net_income = pretax - tax
        dividends = max(0.0, net_income) * bp.dividend_payout
        if fixed_cash_dividends is not None:
            dividends = float(fixed_cash_dividends)
            if not math.isfinite(dividends) or dividends < 0.0:
                raise ValueError("Fixed business-plan dividends must be finite and non-negative")
        return {"revenue": revenue, "cogs": cogs, "sga": sga,
            "gross_profit": revenue-cogs, "operating_income": operating_income,
            "interest_expense": interest, "pure_interest_expense": interest,
            "financial_cost": broad_financial_cost, "non_interest_financial_cost": non_interest_cost,
            "non_op_income": non_op_income,
            "pretax_income": pretax, "tax_expense": tax, "net_income": net_income,
            "cash_dividends": dividends}

    @classmethod
    def _income_and_ocf(cls, state: FirmState, bp: BusinessPlan, *, revenue: float,
                        cogs: float, sga: float, depreciation: float, amortization: float,
                        interest: float, ar_t1: float, inv_t1: float, ap_t1: float,
                        fixed_cash_dividends: float | None = None):
        income = cls._earnings(bp, revenue=revenue, cogs=cogs, sga=sga, interest=interest,
                               fixed_cash_dividends=fixed_cash_dividends)
        income.update(depreciation=depreciation, amortization=amortization)
        ar_change = ar_t1 - _safe_get(state.receivables)
        inv_change = inv_t1 - _safe_get(state.inventory)
        ap_change = ap_t1 - _safe_get(state.payables)
        wc_change = -(ar_change + inv_change) + ap_change
        ocf = income["net_income"] + depreciation + amortization + wc_change
        return income, {"operating_cf": ocf, "ar_t1": ar_t1, "inv_t1": inv_t1,
            "ap_t1": ap_t1, "ar_change": ar_change, "inv_change": inv_change,
            "ap_change": ap_change, "wc_change": wc_change}

    def baseline_dividend_plan(self, state, bp):
        """A0 earnings set the dividend amount before any candidate action."""
        baseline, _ = self._candidate_plan(state, bp, SemanticActionV4())
        revenue = max(0.0, _safe_get(state.revenue)) * (1.0 + bp.revenue_growth)
        short = max(0.0, _safe_get(state.short_term_debt))
        gross_ltd = max(0.0, _safe_get(state.long_term_debt)) + max(0.0, _safe_get(state.current_portion_long_debt))
        bonds = max(0.0, _safe_get(state.bonds))
        interest, _ = self._interest(baseline, short_old=short, short_new=short,
            existing_gross_ltd_old=gross_ltd, existing_gross_ltd_new=gross_ltd,
            existing_bond_old=bonds, existing_bond_new=bonds,
            new_long_tranche=0.0, new_bond_tranche=0.0)
        return self._earnings(baseline, revenue=revenue, cogs=revenue*baseline.cogs_ratio,
            sga=revenue*baseline.sga_ratio, interest=interest)["cash_dividends"]

    @staticmethod
    def _interest(
        bp: BusinessPlan,
        *,
        short_old: float,
        short_new: float,
        existing_gross_ltd_old: float,
        existing_gross_ltd_new: float,
        existing_bond_old: float,
        existing_bond_new: float,
        new_long_tranche: float,
        new_bond_tranche: float,
    ) -> tuple[float, dict[str, float]]:
        parts = {
            "existing_short_debt_interest": 0.5 * (short_old + short_new) * bp.rate_short,
            "existing_gross_ltd_interest": 0.5
            * (existing_gross_ltd_old + existing_gross_ltd_new)
            * bp.rate_long,
            "existing_bond_interest": 0.5
            * (existing_bond_old + existing_bond_new)
            * bp.rate_bond,
            # New RF tranches are issued during the year and never reprice the
            # pre-existing stock.  Half-year exposure is the common timing rule.
            "new_refinancing_long_tranche_interest": 0.5 * new_long_tranche * bp.rate_long,
            "new_refinancing_bond_tranche_interest": 0.5 * new_bond_tranche * bp.rate_bond,
        }
        return float(sum(parts.values())), parts

    def _candidate_plan(self, state_t, bp, action):
        # Reuse the existing factual-ratio endpoint rules for unaffected WC/OE
        # primitives; CX and debt never pass through legacy endpoint semantics.
        names = ("inv_turnover_chg", "ar_turnover_chg", "ap_turnover_chg",
                 "cogs_ratio_chg", "sga_ratio_chg")
        deltas = {name: max(ACTION_BOUNDS[name][0], min(ACTION_BOUNDS[name][1], getattr(action, name)))
                  for name in names}
        bp_eff, endpoint_audit = operating_plan_from_deltas(state_t, bp, deltas)
        endpoint_delta_by_field = {
            "cogs_ratio": action.cogs_ratio_chg,
            "sga_ratio": action.sga_ratio_chg,
            "ar_turnover": action.ar_turnover_chg,
            "inv_turnover": action.inv_turnover_chg,
            "ap_turnover": action.ap_turnover_chg,
        }
        for field_name, requested_delta in endpoint_delta_by_field.items():
            if abs(float(requested_delta)) <= 1.0e-15:
                baseline = float(endpoint_audit[field_name]["baseline"])
                setattr(bp_eff, field_name, baseline)
                endpoint_audit[field_name]["target_unclipped"] = baseline
                endpoint_audit[field_name]["target_effective"] = baseline
                endpoint_audit[field_name]["zero_delta_baseline_preserved"] = True
            else:
                endpoint_audit[field_name]["zero_delta_baseline_preserved"] = False
                baseline = float(endpoint_audit[field_name]["baseline"])
                endpoint = float(endpoint_audit[field_name]["target_effective"])
                realized_delta = endpoint - baseline
                requested = float(requested_delta)
                endpoint_repaired = False
                if realized_delta * requested < -1.0e-15:
                    endpoint = baseline
                    endpoint_repaired = True
                elif abs(realized_delta) > abs(requested) + 1.0e-15:
                    endpoint = baseline + requested
                    endpoint_repaired = True
                if endpoint_repaired:
                    setattr(bp_eff, field_name, endpoint)
                    endpoint_audit[field_name]["target_effective"] = endpoint
                endpoint_audit[field_name]["bounded_realization_repair"] = endpoint_repaired
                endpoint_audit[field_name]["realized_delta"] = endpoint - baseline
        return bp_eff, endpoint_audit

    def compile_candidate(self, state_t, bp, action, *, fixed_cash_dividends=None):
        """Compile frozen candidate operators, including their feasibility limits."""
        return self._run(state_t, bp, action, compile_only=True, fixed_cash_dividends=fixed_cash_dividends)

    def simulate_primitives(self, state_t, bp, primitives, *, execution_mode=CANDIDATE_POLICY_MODE,
                            source_capex_context=None, fixed_cash_dividends=None):
        """Execute observed or compiled amounts; no nearest-candidate projection."""
        if not isinstance(primitives, ExecutedFinancialPrimitives):
            raise TypeError("Expected ExecutedFinancialPrimitives")
        return self._run(state_t, bp, None, primitives=primitives, execution_mode=execution_mode,
                         source_capex_context=source_capex_context, fixed_cash_dividends=fixed_cash_dividends)

    def simulate(self, state_t, bp, action, *, fixed_cash_dividends=None):
        if isinstance(action, ExecutedFinancialPrimitives):
            return self.simulate_primitives(state_t, bp, action, fixed_cash_dividends=fixed_cash_dividends)
        return self._run(state_t, bp, action, fixed_cash_dividends=fixed_cash_dividends)

    def _run(self, state_t, bp, action, *, primitives=None, compile_only=False,
             execution_mode=CANDIDATE_POLICY_MODE, source_capex_context=None, fixed_cash_dividends=None):
        if execution_mode not in {CANDIDATE_POLICY_MODE, HISTORICAL_REPLAY_MODE}:
            raise ValueError("Unknown financial execution mode")
        if execution_mode == HISTORICAL_REPLAY_MODE and (
            primitives is None or primitives.source_kind != "historical" or compile_only
        ):
            raise ValueError("Historical replay requires observed historical primitives; candidates cannot opt out of feasibility")
        from_primitives = primitives is not None
        if action is not None and not isinstance(action, SemanticActionV4):
            action = SemanticActionV4.from_mapping(action)
        if action is None:
            action = SemanticActionV4()  # diagnostics only, never a projected action

        if primitives is None:
            bp_eff, endpoint_audit = self._candidate_plan(state_t, bp, action)
        else:
            bp_eff = BusinessPlan(**bp.to_dict())
            baseline_levels = operating_levels(state_t)
            if primitives.source_kind == "candidate":
                _, baseline_audit = self._candidate_plan(state_t, bp, SemanticActionV4())
                baseline_levels = {name: baseline_audit[name]["baseline"] for name in OPERATING_FIELDS}
            if any(v is None for v in baseline_levels.values()):
                raise ValueError("Unobserved operating baseline cannot execute primitives")
            endpoint_audit = {}
            for name in OPERATING_FIELDS:
                target = getattr(primitives, name + "_target")
                delta = getattr(primitives, name + "_change")
                baseline = baseline_levels[name]
                if not math.isclose(baseline + delta, target, rel_tol=1e-12, abs_tol=1e-10):
                    raise ValueError("Operating primitive roundtrip mismatch: " + name)
                setattr(bp_eff, name, target)
                endpoint_audit[name] = {"baseline": baseline, "target_unclipped": target,
                                       "target_effective": target, "historical_clipping": False}
            bp_eff.revenue_growth = primitives.revenue_growth
        revenue_t = max(0.0, _safe_get(state_t.revenue))
        revenue = revenue_t * (1.0 + bp_eff.revenue_growth)
        cogs = revenue * bp_eff.cogs_ratio
        sga = revenue * bp_eff.sga_ratio
        ppe_t = max(0.0, _safe_get(state_t.ppe))
        intang_t = max(0.0, _safe_get(state_t.intangibles))
        raw_depreciation = max(0.0, ppe_t * bp_eff.depreciation_rate)
        raw_amortization = max(0.0, intang_t * bp_eff.amortization_rate)
        # A calibrated rate can be unstable when the historical asset base is
        # near zero. Non-cash consumption may not exceed the opening stock; new
        # CAPEX is not depreciated in the same one-period projection.
        depreciation = min(ppe_t, raw_depreciation)
        amortization = min(intang_t, raw_amortization)

        # CX: reduce only positive growth/discretionary CAPEX.  Maintenance is
        # capped at the business-plan CAPEX so a low-CAPEX plan is never raised
        # merely to manufacture a maintenance floor.
        capex_parts = capex_baseline(state_t, bp_eff)
        baseline_capex = capex_parts["bp_capex"]
        maintenance_capex = capex_parts["maintenance_capex"]
        growth_capex = capex_parts["bp_growth_capex"]
        if primitives is None:
            executed_capex = maintenance_capex + (
                1.0 - action.growth_capex_reduction_pct
            ) * growth_capex
        else:
            executed_capex = primitives.executed_capex
        ppe_t1 = max(0.0, ppe_t + executed_capex - depreciation)

        short_old = max(0.0, _safe_get(state_t.short_term_debt))
        cpld_old = max(0.0, _safe_get(state_t.current_portion_long_debt))
        long_old = max(0.0, _safe_get(state_t.long_term_debt))
        gross_ltd_old = long_old + cpld_old
        bond_old = max(0.0, _safe_get(state_t.bonds))
        total_debt_old = short_old + gross_ltd_old + bond_old
        opening_other_current_assets_raw = (
            _safe_get(state_t.current_assets)
            - _safe_get(state_t.cash)
            - _safe_get(state_t.short_term_investments)
            - _safe_get(state_t.receivables)
            - _safe_get(state_t.inventory)
        )

        short_new = short_old
        existing_gross_ltd_new = gross_ltd_old
        existing_bond_new = bond_old
        new_long_tranche = 0.0
        new_bond_tranche = 0.0
        requested_repayment = 0.0
        realized_repayment = 0.0
        cash_feasibility_scale = 1.0
        balance_feasibility_scale = 1.0
        refinanced_principal = 0.0
        allocation_redistributed_amount = 0.0
        repayment_by_stack = {"short": 0.0, "gross_ltd": 0.0, "bond": 0.0}
        rf_bond_access = bool(bond_old > 1.0e-9)
        rf_effective_long_weight = self.rf["long_weight"]
        rf_effective_bond_weight = self.rf["bond_weight"]

        # Preliminary cash uses unchanged-principal interest; it is used only to
        # determine DL feasibility and cannot be increased circularly by DL's
        # own interest saving.
        interest_pre, _ = self._interest(
            bp_eff,
            short_old=short_old,
            short_new=short_old,
            existing_gross_ltd_old=gross_ltd_old,
            existing_gross_ltd_new=gross_ltd_old,
            existing_bond_old=bond_old,
            existing_bond_new=bond_old,
            new_long_tranche=0.0,
            new_bond_tranche=0.0,
        )
        ar_t1 = revenue / bp_eff.ar_turnover if bp_eff.ar_turnover > 0 else 0.0
        inv_t1 = cogs / bp_eff.inv_turnover if bp_eff.inv_turnover > 0 else 0.0
        ap_t1 = cogs / bp_eff.ap_turnover if bp_eff.ap_turnover > 0 else 0.0
        income_pre, ocf_pre = self._income_and_ocf(
            state_t,
            bp_eff,
            revenue=revenue,
            cogs=cogs,
            sga=sga,
            depreciation=depreciation,
            amortization=amortization,
            interest=interest_pre,
            ar_t1=ar_t1,
            inv_t1=inv_t1,
            ap_t1=ap_t1,
            fixed_cash_dividends=fixed_cash_dividends,
        )
        projected_pre_financing_cash = (
            max(0.0, _safe_get(state_t.cash))
            + ocf_pre["operating_cf"]
            - executed_capex
            - income_pre["cash_dividends"]
        )
        required_cash_for_nonnegative_current_assets = max(
            0.0,
            -(
                max(0.0, _safe_get(state_t.short_term_investments))
                + ar_t1
                + inv_t1
                + opening_other_current_assets_raw
            ),
        )
        required_liquidity_reserve = max(
            0.0,
            self.reserve_ratio * cogs,
            required_cash_for_nonnegative_current_assets,
        )
        available_repayment_cash = max(
            0.0, projected_pre_financing_cash - required_liquidity_reserve
        )

        if primitives is None and action.deleveraging_total_debt_pct > 0.0 and total_debt_old > 0.0:
            requested_repayment = action.deleveraging_total_debt_pct * total_debt_old
            target_repayment = min(
                requested_repayment, available_repayment_cash, total_debt_old
            )
            balance_feasibility_scale = min(
                1.0, total_debt_old / requested_repayment
                if requested_repayment > 0.0 else 1.0,
            )
            cash_feasibility_scale = min(
                1.0, available_repayment_cash / requested_repayment
                if requested_repayment > 0.0 else 1.0,
            )
            balances = {"short": short_old, "gross_ltd": gross_ltd_old, "bond": bond_old}
            capacities = dict(balances)
            weights = {
                "short": self.dl["short_debt_weight"],
                "gross_ltd": self.dl["gross_ltd_weight"],
                "bond": self.dl["bond_weight"],
            }
            remaining = target_repayment
            for _ in range(len(capacities) + 1):
                active = [name for name, cap in capacities.items() if cap > 1.0e-10]
                if remaining <= 1.0e-10 or not active:
                    break
                active_weight = sum(weights[name] for name in active)
                if active_weight > 1.0e-12:
                    shares = {name: weights[name] / active_weight for name in active}
                else:
                    active_capacity = sum(capacities[name] for name in active)
                    shares = {name: capacities[name] / active_capacity for name in active}
                allocated_now = 0.0
                for name in active:
                    amount = min(capacities[name], remaining * shares[name])
                    repayment_by_stack[name] += amount
                    capacities[name] -= amount
                    allocated_now += amount
                remaining -= allocated_now
                if allocated_now <= 1.0e-10:
                    break
            if remaining > max(1.0e-7, 1.0e-10 * max(target_repayment, 1.0)):
                raise ValueError(
                    f"DL constrained allocation failed: target={target_repayment}, remaining={remaining}"
                )
            preferred = {name: target_repayment * weights[name] for name in weights}
            allocation_redistributed_amount = 0.5 * sum(
                abs(repayment_by_stack[name] - preferred[name]) for name in weights
            )
            short_new -= repayment_by_stack["short"]
            existing_gross_ltd_new -= repayment_by_stack["gross_ltd"]
            existing_bond_new -= repayment_by_stack["bond"]
            realized_repayment = sum(repayment_by_stack.values())
        elif primitives is None and action.refinancing_short_debt_pct > 0.0 and short_old > 0.0:
            refinanced_principal = min(
                short_old, action.refinancing_short_debt_pct * short_old
            )
            short_new = short_old - refinanced_principal
            # Bond issuance requires an outstanding opening bond. If access is
            # absent, the frozen bond preference is deterministically reassigned
            # to the long-bank tranche; current v4's frozen bond weight is zero.
            if self.rf["bond_weight"] > 0.0 and not rf_bond_access:
                rf_effective_long_weight = 1.0
                rf_effective_bond_weight = 0.0
            new_long_tranche = refinanced_principal * rf_effective_long_weight
            new_bond_tranche = refinanced_principal * rf_effective_bond_weight

        if primitives is None:
            primitives = ExecutedFinancialPrimitives(
                growth_capex_adjustment=growth_capex - max(0.0, executed_capex - maintenance_capex),
                maintenance_capex_adjustment=0.0, executed_capex=executed_capex,
                short_debt_principal_change=-repayment_by_stack["short"],
                gross_ltd_principal_change=-repayment_by_stack["gross_ltd"],
                bond_principal_change=-repayment_by_stack["bond"],
                refi_short_to_ltd_amount=new_long_tranche,
                refi_short_to_bond_amount=new_bond_tranche,
                revenue_growth=bp_eff.revenue_growth, source_kind="candidate",
                **{name + "_change": endpoint_audit[name]["target_effective"] - endpoint_audit[name]["baseline"]
                   for name in OPERATING_FIELDS},
                **{name + "_target": endpoint_audit[name]["target_effective"] for name in OPERATING_FIELDS},
            )
        if execution_mode == HISTORICAL_REPLAY_MODE and source_capex_context is not None:
            primitives.validate_source_capex(source_capex_context)
        elif (execution_mode == HISTORICAL_REPLAY_MODE and self.dl_interest_timing is not None
              and self.dl_interest_timing.repayment_timing_fraction == 1.0):
            raise ValueError("V4.3 historical execution requires frozen source CAPEX context")
        else:
            primitives.validate_capex(capex_parts)
        # Both input languages enter this SAME amount-based principal executor.
        short_new, existing_gross_ltd_new, existing_bond_new, new_long_tranche, new_bond_tranche = (
            primitives.apply_debt(short_old, gross_ltd_old, bond_old)
        )
        repayment_by_stack = primitives.repayment_by_stack()
        realized_repayment = sum(repayment_by_stack.values())
        refinanced_principal = primitives.rf_total
        if compile_only:
            return primitives

        interest, interest_parts = self._interest(
            bp_eff,
            short_old=short_old,
            short_new=(short_old - primitives.rf_total + max(primitives.short_debt_principal_change, 0.0)),
            existing_gross_ltd_old=gross_ltd_old,
            existing_gross_ltd_new=(gross_ltd_old + max(primitives.gross_ltd_principal_change, 0.0)),
            existing_bond_old=bond_old,
            existing_bond_new=(bond_old + max(primitives.bond_principal_change, 0.0)),
            new_long_tranche=new_long_tranche,
            new_bond_tranche=new_bond_tranche,
        )
        income, ocf = self._income_and_ocf(
            state_t,
            bp_eff,
            revenue=revenue,
            cogs=cogs,
            sga=sga,
            depreciation=depreciation,
            amortization=amortization,
            interest=interest,
            ar_t1=ar_t1,
            inv_t1=inv_t1,
            ap_t1=ap_t1,
            fixed_cash_dividends=fixed_cash_dividends,
        )

        dl_saving_diagnostics = None
        if self.dl_interest_timing is not None:
            # Allocation and pre-saving capacity are final at this point.
            # Never feed these savings back into repayment_by_stack.
            dl_saving_diagnostics = dl_interest_savings(
                repayment_by_stack, bp_eff, interest, self.dl_interest_timing
            )
            income_without_saving, ocf_without_saving = income, ocf
            interest = dl_saving_diagnostics["interest_expense_after_dl_saving"]
            if dl_saving_diagnostics["total_dl_interest_saving"] > 0.0:
                income, ocf = self._income_and_ocf(
                    state_t, bp_eff, revenue=revenue, cogs=cogs, sga=sga,
                    depreciation=depreciation, amortization=amortization,
                    interest=interest, ar_t1=ar_t1, inv_t1=inv_t1, ap_t1=ap_t1,
                    fixed_cash_dividends=fixed_cash_dividends,
                )
            for field, label in (("pretax_income", "pretax_income"),
                                 ("tax_expense", "tax"),
                                 ("net_income", "net_income"),
                                 ("cash_dividends", "dividend")):
                dl_saving_diagnostics[f"{label}_change_from_interest_saving"] = (
                    income[field] - income_without_saving[field]
                )
            ocf_delta = ocf["operating_cf"] - ocf_without_saving["operating_cf"]
            retained_delta = (dl_saving_diagnostics["net_income_change_from_interest_saving"]
                              - dl_saving_diagnostics["dividend_change_from_interest_saving"])
            dl_saving_diagnostics.update({
                "ocf_change_from_interest_saving": ocf_delta,
                "ending_cash_change_from_interest_saving": (
                    ocf_delta - dl_saving_diagnostics["dividend_change_from_interest_saving"]
                ),
                "retained_earnings_change_from_interest_saving": retained_delta,
                "total_equity_change_from_interest_saving": retained_delta,
                "principal_financing_cf_change_from_interest_saving": 0.0,
            })

        reclass_rate = float(bp_eff.reclass_rate)
        if not 0.0 <= reclass_rate < 1.0:
            raise ValueError(f"Invalid reclass_rate={reclass_rate}")
        # Only the existing gross LTD stock is eligible for t+1 CPLD.  New RF
        # long tranches have contractual maturity >12 months at first recognition.
        cpld_t1 = existing_gross_ltd_new * reclass_rate
        long_t1 = existing_gross_ltd_new * (1.0 - reclass_rate) + new_long_tranche
        bond_t1 = existing_bond_new + new_bond_tranche
        gross_ltd_t1 = cpld_t1 + long_t1
        total_debt_t1 = short_new + gross_ltd_t1 + bond_t1
        # Preserve the action-principal endpoint before ordinary operating
        # liquidity financing. RF neutrality and DL feasibility are audited on
        # this layer, not on a later shortfall-financing tranche.
        total_debt_t1_before_liquidity_financing = total_debt_t1
        action_principal_change = total_debt_t1_before_liquidity_financing - total_debt_old
        financing_cf_before_liquidity = action_principal_change - income["cash_dividends"]
        investing_cf = -executed_capex
        cash_before_liquidity_financing = (
            max(0.0, _safe_get(state_t.cash))
            + ocf["operating_cf"]
            + investing_cf
            + financing_cf_before_liquidity
        )
        current_assets_before_liquidity_financing = (
            cash_before_liquidity_financing
            + max(0.0, _safe_get(state_t.short_term_investments))
            + ar_t1
            + inv_t1
            + opening_other_current_assets_raw
        )
        # A closing-date short-term tranche may fund an ordinary operating
        # shortfall. It is computed after DL feasibility, is never eligible for
        # repayment by that same action, and has no within-period interest
        # exposure. This is distinct from a forbidden repayment-cancelling plug.
        # Do not manufacture a short-term borrowing tranche from round-off at
        # an exactly binding DL cash-capacity boundary. Material shortfalls are
        # still financed and remain subject to the no-repayment-reborrow guard.
        cash_floor_shortfall = (
            -cash_before_liquidity_financing
            if cash_before_liquidity_financing < -self.liquidity_cash_abs_tol
            else 0.0
        )
        current_asset_floor_shortfall = (
            -current_assets_before_liquidity_financing
            if current_assets_before_liquidity_financing < -self.liquidity_cash_abs_tol
            else 0.0
        )
        liquidity_shortfall_financing = max(
            cash_floor_shortfall, current_asset_floor_shortfall
        )
        short_new += liquidity_shortfall_financing
        total_debt_t1 += liquidity_shortfall_financing
        debt_change = total_debt_t1 - total_debt_old
        financing_cf = financing_cf_before_liquidity + liquidity_shortfall_financing
        cash_t1 = cash_before_liquidity_financing + liquidity_shortfall_financing
        replay_audit = {
            "execution_mode": execution_mode,
            "historical_replay_execution_version": HISTORICAL_REPLAY_EXECUTION_VERSION,
            "observed_financing_primitives": {k: v for k, v in primitives.to_dict().items()
                if k in {"short_debt_principal_change", "gross_ltd_principal_change", "bond_principal_change",
                         "refi_short_to_ltd_amount", "refi_short_to_bond_amount"}},
            "pre_plug_cash": cash_before_liquidity_financing,
            "pre_plug_current_assets": current_assets_before_liquidity_financing,
            "cash_floor_shortfall": cash_floor_shortfall,
            "current_asset_floor_shortfall": current_asset_floor_shortfall,
            "required_cash_plug": liquidity_shortfall_financing,
            "cash_plug_to_assets": (liquidity_shortfall_financing / state_t.total_assets
                if state_t.total_assets is not None and math.isfinite(state_t.total_assets) and state_t.total_assets > 0 else None),
            "post_plug_short_debt": short_new,
            "incremental_unobserved_financing": liquidity_shortfall_financing,
            "candidate_repayment_reborrow_gate_applied": execution_mode == CANDIDATE_POLICY_MODE,
        } if execution_mode == HISTORICAL_REPLAY_MODE else None

        def closing_error(message):
            error = ValueError(message)
            if replay_audit is not None:
                error.historical_replay_diagnostics = replay_audit
            return error

        if cash_t1 < -self.liquidity_cash_abs_tol:
            raise closing_error(
                "v4 liquidity financing failed to close negative cash: "
                f"cash_before={cash_before_liquidity_financing}, "
                f"liquidity_financing={liquidity_shortfall_financing}, cash_t1={cash_t1}"
            )
        cash_t1 = max(0.0, cash_t1)
        cash_reborrow_plug_used = bool(
            realized_repayment > self.liquidity_cash_abs_tol
            and liquidity_shortfall_financing > self.liquidity_cash_abs_tol
        )
        if cash_reborrow_plug_used and execution_mode == CANDIDATE_POLICY_MODE:
            raise ValueError(
                "DL feasibility invariant failed: realized repayment was followed by "
                "short-debt liquidity re-borrowing"
            )
        interest_parts["new_liquidity_shortfall_tranche_interest"] = 0.0

        sti_t1 = max(0.0, _safe_get(state_t.short_term_investments))
        intang_t1 = max(0.0, intang_t - amortization)
        other_ca = opening_other_current_assets_raw
        other_nca = _safe_get(state_t.non_current_assets) - ppe_t - intang_t
        other_cl = _safe_get(state_t.current_liabilities) - short_old - cpld_old - _safe_get(state_t.payables)
        other_ncl = _safe_get(state_t.non_current_liabilities) - long_old - bond_old
        if _safe_get(state_t.current_assets) == 0.0 and _safe_get(state_t.non_current_assets) == 0.0:
            other_ca = 0.0
            other_nca = _safe_get(state_t.total_assets) - _safe_get(state_t.cash) - sti_t1 - _safe_get(state_t.receivables) - _safe_get(state_t.inventory) - ppe_t - intang_t
        if _safe_get(state_t.current_liabilities) == 0.0 and _safe_get(state_t.non_current_liabilities) == 0.0:
            other_cl = 0.0
            other_ncl = _safe_get(state_t.total_liabilities) - short_old - cpld_old - long_old - bond_old - _safe_get(state_t.payables)

        current_assets = cash_t1 + sti_t1 + ar_t1 + inv_t1 + other_ca
        non_current_assets = ppe_t1 + intang_t1 + other_nca
        current_liabilities = short_new + cpld_t1 + ap_t1 + other_cl
        non_current_liabilities = long_t1 + bond_t1 + other_ncl
        total_equity = _safe_get(state_t.total_equity) + income["net_income"] - income["cash_dividends"]
        total_assets_pre = current_assets + non_current_assets
        total_liabilities = current_liabilities + non_current_liabilities
        accounting_residual = total_assets_pre - (total_liabilities + total_equity)
        opening_reported_identity_residual = _safe_get(state_t.total_assets) - (
            _safe_get(state_t.total_liabilities) + _safe_get(state_t.total_equity)
        )
        opening_accounting_residual = (
            _safe_get(state_t.current_assets) + _safe_get(state_t.non_current_assets)
        ) - (
            _safe_get(state_t.current_liabilities)
            + _safe_get(state_t.non_current_liabilities)
            + _safe_get(state_t.total_equity)
        )
        incremental_accounting_residual = accounting_residual - opening_accounting_residual
        accounting_scale = max(
            abs(_safe_get(state_t.total_assets)),
            abs(total_assets_pre),
            abs(total_liabilities + total_equity),
            1.0,
        )
        accounting_tolerance = max(
            self.accounting_abs_tol, self.accounting_rel_tol * accounting_scale
        )
        if abs(incremental_accounting_residual) > accounting_tolerance:
            raise closing_error(
                "v4 incremental accounting residual exceeds contract tolerance: "
                f"opening={opening_accounting_residual}, closing_pre={accounting_residual}, "
                f"incremental={incremental_accounting_residual}, tolerance={accounting_tolerance}, "
                f"opening_cash={_safe_get(state_t.cash)}, closing_cash={cash_t1}, "
                f"operating_cf={ocf['operating_cf']}, investing_cf={investing_cf}, "
                f"financing_cf={financing_cf}, opening_payables={_safe_get(state_t.payables)}, "
                f"closing_payables={ap_t1}, debt_change={debt_change}, "
                f"net_income={income['net_income']}, dividends={income['cash_dividends']}"
            )
        # Carry the small inherited opening residual explicitly through the
        # unmodelled non-current-asset line. Action-induced residual is gated
        # above and may never be silently absorbed.
        reconciled_other_nca = other_nca - accounting_residual
        reconciled_non_current_assets = non_current_assets - accounting_residual
        if reconciled_other_nca < -accounting_tolerance or reconciled_non_current_assets < -accounting_tolerance:
            raise closing_error(
                "NONPHYSICAL_STATE after inherited-residual reconciliation: "
                f"other_nca={reconciled_other_nca}, non_current_assets={reconciled_non_current_assets}, "
                f"tolerance={accounting_tolerance}"
            )
        other_nca = max(0.0, reconciled_other_nca)
        non_current_assets = max(0.0, reconciled_non_current_assets)
        total_assets = current_assets + non_current_assets
        if current_assets < -accounting_tolerance or total_assets < -accounting_tolerance:
            raise closing_error(
                f"NONPHYSICAL_STATE current_assets={current_assets}, total_assets={total_assets}"
            )

        retained_t1 = _safe_get(state_t.retained_earnings) + income["net_income"] - income["cash_dividends"]
        state_t1 = FirmState(
            firm_id=state_t.firm_id,
            year=int(state_t.year) + 1,
            sector=state_t.sector,
            rating_num=state_t.rating_num,
            rating_grade=state_t.rating_grade,
            revenue=revenue,
            cogs=cogs,
            gross_profit=income["gross_profit"],
            sga=sga,
            operating_income=income["operating_income"],
            financial_cost=income["financial_cost"],
            pure_interest_expense=interest,
            non_interest_financial_cost=income["non_interest_financial_cost"],
            pretax_income=income["pretax_income"],
            tax_expense=income["tax_expense"],
            net_income=income["net_income"],
            comprehensive_income=income["net_income"],
            depreciation=depreciation,
            amortization=amortization,
            interest_income=income["non_op_income"] * 0.5,
            dividend_income=income["non_op_income"] * 0.5,
            total_assets=total_assets,
            current_assets=current_assets,
            non_current_assets=non_current_assets,
            cash=cash_t1,
            short_term_investments=sti_t1,
            receivables=ar_t1,
            inventory=inv_t1,
            ppe=ppe_t1,
            intangibles=intang_t1,
            total_liabilities=total_liabilities,
            current_liabilities=current_liabilities,
            non_current_liabilities=non_current_liabilities,
            short_term_debt=short_new,
            current_portion_long_debt=cpld_t1,
            long_term_debt=long_t1,
            bonds=bond_t1,
            payables=ap_t1,
            total_equity=total_equity,
            capital_stock=state_t.capital_stock,
            retained_earnings=retained_t1,
            ending_capital_stock=state_t.ending_capital_stock,
            ending_capital_surplus=state_t.ending_capital_surplus,
            ending_other_capital=state_t.ending_other_capital,
            ending_oci=state_t.ending_oci,
            ending_retained_earnings=retained_t1,
            cash_dividends=income["cash_dividends"],
            operating_cf=ocf["operating_cf"],
            investing_cf=investing_cf,
            financing_cf=financing_cf,
            capex=executed_capex,
        )
        accounting = state_t1.accounting_identity_check(tol=10.0)
        sustainability = classify_sustainability(state_t1, income)
        realized_growth_capex_reduction_pct = (
            (growth_capex - max(0.0, executed_capex - maintenance_capex)) / growth_capex
            if growth_capex > 1.0e-10 else 0.0
        )
        realized_action_by_dimension = {
            "growth_capex_reduction_pct": realized_growth_capex_reduction_pct,
            "deleveraging_total_debt_pct": (
                realized_repayment / total_debt_old if total_debt_old > 1.0e-10 else 0.0
            ),
            "refinancing_short_debt_pct": (
                refinanced_principal / short_old if short_old > 1.0e-10 else 0.0
            ),
            "inv_turnover_chg": endpoint_audit["inv_turnover"]["target_effective"] - endpoint_audit["inv_turnover"]["baseline"],
            "ar_turnover_chg": endpoint_audit["ar_turnover"]["target_effective"] - endpoint_audit["ar_turnover"]["baseline"],
            "ap_turnover_chg": endpoint_audit["ap_turnover"]["target_effective"] - endpoint_audit["ap_turnover"]["baseline"],
            "cogs_ratio_chg": endpoint_audit["cogs_ratio"]["target_effective"] - endpoint_audit["cogs_ratio"]["baseline"],
            "sga_ratio_chg": endpoint_audit["sga_ratio"]["target_effective"] - endpoint_audit["sga_ratio"]["baseline"],
        }
        nominal_normalized_intensity = sum(
            abs(float(getattr(action, dim))) / self.intensity_scale[dim]
            for dim in V4_ACTION_DIMENSIONS
        )
        realized_normalized_intensity = sum(
            abs(float(realized_action_by_dimension[dim])) / self.intensity_scale[dim]
            for dim in V4_ACTION_DIMENSIONS
        )
        realized_intensity_ratio = (
            realized_normalized_intensity / nominal_normalized_intensity
            if nominal_normalized_intensity > 1.0e-12 else 1.0
        )
        diagnostics = {
            "action_semantic_contract_version": ACTION_SEMANTIC_CONTRACT_VERSION_V4,
            "simulator_producer_version": SEMANTIC_FINANCIAL_SIMULATOR_V4_VERSION,
            "candidate_action_contract_hash": self.contract_hash,
            "action_clipped": action.to_dict(),
            "realized_action_vector": {f"action__{dim}": float(realized_action_by_dimension[dim]) for dim in V4_ACTION_DIMENSIONS},
            "nominal_normalized_intensity": nominal_normalized_intensity,
            "realized_normalized_intensity": realized_normalized_intensity,
            "realized_intensity_ratio": realized_intensity_ratio,
            "action_endpoint_audit": endpoint_audit,
            "is": income,
            "ocf_components": ocf,
            "icf_components": {
                "baseline_capex": baseline_capex,
                "maintenance_capex": maintenance_capex,
                "growth_capex": growth_capex,
                "growth_capex_reduction_pct": action.growth_capex_reduction_pct,
                "executed_capex": executed_capex,
                "total_capex": executed_capex,
                "investing_cf": investing_cf,
                "ppe_t1": ppe_t1,
                "forced_asset_sale": False,
            },
            "noncash_charge_components": {
                "raw_depreciation": raw_depreciation,
                "effective_depreciation": depreciation,
                "depreciation_capped_at_opening_ppe": bool(raw_depreciation > ppe_t),
                "raw_amortization": raw_amortization,
                "effective_amortization": amortization,
                "amortization_capped_at_opening_intangibles": bool(raw_amortization > intang_t),
            },
            "fcf_components": {
                "short_debt_old": short_old,
                "short_debt_new": short_new,
                "gross_ltd_old": gross_ltd_old,
                "gross_ltd_new": gross_ltd_t1,
                "bond_old": bond_old,
                "bond_new": bond_t1,
                "total_interest_bearing_debt_old": total_debt_old,
                "total_interest_bearing_debt_new": total_debt_t1,
                "total_interest_bearing_debt_new_before_liquidity_financing": total_debt_t1_before_liquidity_financing,
                "action_principal_change": action_principal_change,
                "debt_change": debt_change,
                "requested_repayment": requested_repayment,
                "realized_repayment": realized_repayment,
                "repayment_by_stack": repayment_by_stack,
                "allocation_redistributed_amount": allocation_redistributed_amount,
                "dl_realized_fraction_of_requested": (realized_repayment / requested_repayment if requested_repayment > 1.0e-12 else 1.0),
                "debt_action_timing": "DL closing-date; RF half-year transfer",
                "projected_pre_financing_cash": projected_pre_financing_cash,
                "required_liquidity_reserve": required_liquidity_reserve,
                "required_cash_for_nonnegative_current_assets": required_cash_for_nonnegative_current_assets,
                "opening_other_current_assets_raw": opening_other_current_assets_raw,
                "current_assets_before_liquidity_financing": current_assets_before_liquidity_financing,
                "cash_floor_shortfall": cash_floor_shortfall,
                "current_asset_floor_shortfall": current_asset_floor_shortfall,
                "liquidity_cash_abs_tolerance": self.liquidity_cash_abs_tol,
                "available_repayment_cash": available_repayment_cash,
                "balance_feasibility_scale": balance_feasibility_scale,
                "cash_feasibility_scale": cash_feasibility_scale,
                "realized_intensity_ratio": realized_intensity_ratio,
                "refinanced_principal": refinanced_principal,
                "rf_bond_access": rf_bond_access,
                "rf_effective_long_weight": rf_effective_long_weight,
                "rf_effective_bond_weight": rf_effective_bond_weight,
                "new_long_tranche": new_long_tranche,
                "new_bond_tranche": new_bond_tranche,
                "new_tranche_immediate_cpld": 0.0,
                "principal_neutrality_error": total_debt_t1_before_liquidity_financing - total_debt_old
                if refinanced_principal > 0.0
                else None,
                "interest_components": interest_parts,
                "interest_expense": interest,
                "cash_before_liquidity_financing": cash_before_liquidity_financing,
                "liquidity_shortfall_financing": liquidity_shortfall_financing,
                "liquidity_financing_used": bool(liquidity_shortfall_financing > 1.0e-10),
                "liquidity_financing_interest_timing": "closing_date_zero_within_period_exposure",
                "financing_cf": financing_cf,
                "cash_reborrow_plug_used": cash_reborrow_plug_used,
            },
            "balance_sheet_closing": {
                "accounting_residual_reconciled_to_other_non_current_assets": accounting_residual,
                "opening_accounting_residual": opening_accounting_residual,
                "opening_reported_identity_residual": opening_reported_identity_residual,
                "incremental_accounting_residual": incremental_accounting_residual,
                "accounting_residual_tolerance": accounting_tolerance,
                "other_non_current_assets_after_reconciliation": other_nca,
                "nonphysical_state": False,
                "cash": cash_t1,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "total_equity": total_equity,
            },
            "bp_effective": bp_eff.to_dict(),
            "oracle_used": False,
        }
        if fixed_cash_dividends is not None:
            diagnostics["dividend_plan"] = {"policy": "fixed_A0_cash_dividend_amount",
                "baseline_amount": float(fixed_cash_dividends),
                "executed_amount": income["cash_dividends"],
                "action_induced_change": income["cash_dividends"]-float(fixed_cash_dividends),
                "payout_ratio_used_for_A0_plan_only": bp.dividend_payout}
        diagnostics["executed_financial_primitives"] = primitives.to_dict()
        if replay_audit is not None:
            diagnostics["historical_replay"] = replay_audit
        diagnostics["executed_primitive_hash"] = primitives.content_hash
        diagnostics["primitive_executor_version"] = PRIMITIVE_VERSION
        if from_primitives:
            diagnostics["action_clipped"] = None
            diagnostics["realized_action_vector"] = None
            diagnostics["nominal_normalized_intensity"] = None
            diagnostics["realized_normalized_intensity"] = None
            diagnostics["realized_intensity_ratio"] = None
            diagnostics["historical_candidate_projection_used"] = False
            diagnostics["icf_components"]["growth_capex_reduction_pct"] = (
                primitives.growth_capex_adjustment / growth_capex if growth_capex > 0 else None
            )
        if source_capex_context is not None:
            diagnostics["icf_components"]["growth_capex_reduction_pct"] = None
            diagnostics["historical_source_capex_attribution"] = {
                **source_capex_context,
                "growth_capex_adjustment": primitives.growth_capex_adjustment,
                "maintenance_capex_adjustment": primitives.maintenance_capex_adjustment,
                "execution_rule": "observed_absolute_executed_capex",
                "current_bp_capex_validation_used": False,
            }
        if dl_saving_diagnostics is not None:
            diagnostics["simulator_action_semantic_contract_version"] = self.dl_interest_timing.semantic_version
            diagnostics["simulator_producer_version"] = self.dl_interest_timing.producer_version
            diagnostics["dl_interest_saving"] = dl_saving_diagnostics
            diagnostics["fcf_components"]["debt_action_timing"] = (
                "DL capped half-year savings after pre-saving capacity allocation; "
                "RF unchanged half-year transfer"
            )
            if self.dl_interest_timing.effective_interest_saving_rate_cap is None:
                diagnostics["fcf_components"]["debt_action_timing"] = (
                    "DL uncapped half-year savings after pre-saving capacity allocation; "
                    "RF unchanged half-year transfer"
                )
            if self.dl_interest_timing.repayment_timing_fraction == 1.0:
                diagnostics["fcf_components"]["debt_action_timing"] = (
                    "DL uncapped full-year savings after pre-saving capacity allocation; "
                    "RF unchanged half-year transfer"
                )
        return SimulationResult(
            state_t1=state_t1,
            diagnostics=diagnostics,
            sustainability=sustainability,
            plug_used=(
                "operating_liquidity_shortfall_financing"
                if liquidity_shortfall_financing > 1.0e-10
                else "none"
            ),
            plug_amount=liquidity_shortfall_financing,
            accounting_check=accounting,
        )

