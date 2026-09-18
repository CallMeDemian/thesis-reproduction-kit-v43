"""Financial amounts shared by candidate compilation and observed reconstruction.

No policy IDs, learned scores, projection, or candidate bounds belong here.
Debt changes are residuals AFTER the explicitly represented neutral transfer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from .business_plan import BusinessPlan
from .firm_state import FirmState

PRIMITIVE_VERSION = "executed_financial_primitives_v1"
OPERATING_FIELDS = ("inv_turnover", "ar_turnover", "ap_turnover", "cogs_ratio", "sga_ratio")


def operating_levels(state: FirmState) -> dict[str, float | None]:
    pairs = {"inv_turnover": (state.cogs, state.inventory),
             "ar_turnover": (state.revenue, state.receivables),
             "ap_turnover": (state.cogs, state.payables),
             "cogs_ratio": (state.cogs, state.revenue),
             "sga_ratio": (state.sga, state.revenue)}
    return {name: (float(num / den) if num is not None and den is not None
                  and math.isfinite(num) and math.isfinite(den) and den > 0 else None)
            for name, (num, den) in pairs.items()}


def stable_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def capex_baseline(state: FirmState, bp: BusinessPlan) -> dict[str, float]:
    """Frozen v4.1 CX arithmetic, also used by the historical producer."""
    revenue = max(0.0, float(state.revenue or 0.0))
    ppe = max(0.0, float(state.ppe or 0.0))
    depreciation = min(ppe, max(0.0, ppe * bp.depreciation_rate))
    baseline = max(0.0, revenue * bp.capex_to_revenue)
    maintenance = min(baseline, depreciation)
    return {"bp_capex": baseline, "maintenance_capex": maintenance,
            "bp_growth_capex": max(baseline - maintenance, 0.0)}


def decompose_debt(short_delta: float, gross_delta: float, bond_delta: float) -> dict[str, float]:
    """Maximal principal-neutral matching; attribution, not transaction identification."""
    if not all(math.isfinite(x) for x in (short_delta, gross_delta, bond_delta)):
        raise ValueError("Missing/nonfinite debt delta is not zero financing")
    long_increase, bond_increase = max(gross_delta, 0.0), max(bond_delta, 0.0)
    capacity = long_increase + bond_increase
    rf = min(max(-short_delta, 0.0), capacity)
    # Divide first: when bond increase is exactly zero, the fraction is exactly
    # one. Multiply-then-divide could exceed rf by one ULP and invent negative RF.
    to_long = rf * (long_increase / capacity) if capacity > 0.0 else 0.0
    to_bond = rf - to_long
    return {"refi_short_to_ltd_amount": to_long, "refi_short_to_bond_amount": to_bond,
            "short_debt_principal_change": short_delta + rf,
            "gross_ltd_principal_change": gross_delta - to_long,
            "bond_principal_change": bond_delta - to_bond}


@dataclass(frozen=True)
class ExecutedFinancialPrimitives:
    # Positive adjustment = restraint. Maintenance shortfall is a separate
    # positive amount, not disguised as growth restraint or discarded.
    growth_capex_adjustment: float
    maintenance_capex_adjustment: float
    executed_capex: float
    short_debt_principal_change: float
    gross_ltd_principal_change: float
    bond_principal_change: float
    refi_short_to_ltd_amount: float
    refi_short_to_bond_amount: float
    inv_turnover_change: float
    ar_turnover_change: float
    ap_turnover_change: float
    revenue_growth: float
    cogs_ratio_change: float
    sga_ratio_change: float
    # Redundant exact endpoints preserve floating-point candidate parity.
    # They are checked against deltas, never an independent financial producer.
    inv_turnover_target: float
    ar_turnover_target: float
    ap_turnover_target: float
    cogs_ratio_target: float
    sga_ratio_target: float
    source_kind: str
    version: str = PRIMITIVE_VERSION

    def __post_init__(self):
        if self.version != PRIMITIVE_VERSION or self.source_kind not in {"candidate", "historical"}:
            raise ValueError("Unknown primitive contract/source")
        for k, v in asdict(self).items():
            if k not in {"source_kind", "version"} and (v is None or not math.isfinite(v)):
                raise ValueError(f"Unobserved/nonfinite primitive: {k}")
        if self.executed_capex < 0 or min(self.refi_short_to_ltd_amount, self.refi_short_to_bond_amount) < 0:
            raise ValueError("Negative executed CAPEX or reverse RF is not this schema")
        if self.revenue_growth < -1:
            raise ValueError("Negative revenue endpoint")
        # Signed reported cost ratios can be inherited (e.g. expense reversals).
        # Do not impose a new endpoint floor on the frozen candidate semantics.
        # Historical physical/observation eligibility is enforced by its producer.

    def to_dict(self):
        return asdict(self)

    @property
    def content_hash(self):
        return stable_hash(self.to_dict())

    @property
    def rf_total(self):
        return self.refi_short_to_ltd_amount + self.refi_short_to_bond_amount

    def debt_deltas(self):
        return {"short": self.short_debt_principal_change - self.rf_total,
                "gross_ltd": self.gross_ltd_principal_change + self.refi_short_to_ltd_amount,
                "bond": self.bond_principal_change + self.refi_short_to_bond_amount}

    def repayment_by_stack(self):
        return {"short": max(-self.short_debt_principal_change, 0.0),
                "gross_ltd": max(-self.gross_ltd_principal_change, 0.0),
                "bond": max(-self.bond_principal_change, 0.0)}

    def validate_capex(self, baseline):
        """Candidate roundtrip against the current business-plan baseline."""
        self._validate_capex_roundtrip(baseline)

    def validate_source_capex(self, context):
        """Validate frozen historical attribution; execution uses absolute CAPEX."""
        if self.source_kind != "historical":
            raise ValueError("Source CAPEX context requires historical primitives")
        if context["primitive_hash"] != self.content_hash:
            raise ValueError("Frozen historical primitive hash mismatch")
        for key in ("bp_capex", "maintenance_capex", "bp_growth_capex", "actual_capex"):
            if not math.isfinite(float(context[key])) or float(context[key]) < 0:
                raise ValueError("Invalid frozen source CAPEX: " + key)
        if not math.isclose(context["maintenance_capex"] + context["bp_growth_capex"],
                            context["bp_capex"], rel_tol=1e-12, abs_tol=1e-7):
            raise ValueError("Frozen source CAPEX baseline mismatch")
        if context["actual_capex"] != self.executed_capex:
            raise ValueError("Observed absolute CAPEX mismatch")
        self._validate_capex_roundtrip(context)

    def _validate_capex_roundtrip(self, baseline):
        recomposed = (baseline["maintenance_capex"] - self.maintenance_capex_adjustment
                      + baseline["bp_growth_capex"] - self.growth_capex_adjustment)
        if not math.isclose(recomposed, self.executed_capex, rel_tol=1e-12, abs_tol=1e-7):
            raise ValueError("CAPEX primitive roundtrip mismatch")

    def apply_debt(self, short, gross_ltd, bond):
        endpoints = (short + self.short_debt_principal_change - self.rf_total,
                     gross_ltd + self.gross_ltd_principal_change,
                     bond + self.bond_principal_change)
        if any(x < -1e-7 or not math.isfinite(x) for x in endpoints):
            raise ValueError("Executed principal exceeds opening stock; no clipping allowed")
        return (*endpoints, self.refi_short_to_ltd_amount, self.refi_short_to_bond_amount)
