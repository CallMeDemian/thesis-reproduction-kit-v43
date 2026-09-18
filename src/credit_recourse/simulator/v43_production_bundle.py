"""Single production construction seam for the frozen v4.3 simulation stack."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from credit_recourse.rl.common.semantic_action_v4_1_contract import simulator_from_contract
from credit_recourse.simulator.business_plan import BusinessPlan, calibrate_business_plan_non_rate, NON_RATE_CALIBRATION_VERSION
from credit_recourse.simulator.business_plan_interest_rate_v4 import BusinessPlanRateResolverV4
from credit_recourse.simulator.executed_primitives import ExecutedFinancialPrimitives, capex_baseline
from credit_recourse.simulator.semantic_action_v4 import (
    CANDIDATE_POLICY_MODE,
    HISTORICAL_REPLAY_MODE,
    SEMANTIC_FINANCIAL_SIMULATOR_V4_VERSION,
    SemanticActionV4,
)

from credit_recourse.simulator.financial_cost_v43 import FinancialCostSources
from credit_recourse.contracts.v43_action_contract import CANONICAL_ACTION_CONTRACT_PATH

BUNDLE_VERSION = "V43ProductionSimulationBundle/5_fixed_dividend_repaired_broad_financial_cost"
CANDIDATE_DIVIDEND_POLICY = "fixed_A0_cash_dividend_amount"
CALIBRATED_BP_RATE_V4_MODE = "calibrated_bp_rate_v4"
CONTRACT_RELATIVE_PATH = CANONICAL_ACTION_CONTRACT_PATH


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


@dataclass(frozen=True)
class V43ProductionSimulationBundle:
    """The only V4.3 top-level simulator construction path.

    Its semantic simulator is created exclusively through
    ``simulator_from_contract``.  It accepts the frozen nine archetypes and
    historical executed primitives as distinct input languages, which both
    reach the same amount-based primitive executor.
    """
    project_root: Path
    action_contract: Mapping[str, Any]
    action_contract_sha256: str
    _rate_resolver: BusinessPlanRateResolverV4
    _simulator: Any
    _financial_cost_sources: Any = None

    @classmethod
    def from_project_root(cls, project_root: Path) -> "V43ProductionSimulationBundle":
        root = Path(project_root).resolve()
        path = root / CONTRACT_RELATIVE_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Frozen v4.3 action contract is missing: {path}")
        contract = json.loads(path.read_text(encoding="utf-8"))
        expected = "simulator_action_semantic_contract_v4_3_dl_interest_full_year"
        if contract.get("simulator_action_semantic_contract_version") != expected:
            raise ValueError("V4.3 production bundle refuses a non-full-year action contract")
        required = {"A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2"}
        candidates = contract.get("fixed_candidates") or {}
        if set(candidates) != required:
            raise ValueError("Frozen V4.3 candidate contract is not the approved 9-action universe")
        return cls(
            project_root=root,
            action_contract=contract,
            action_contract_sha256=_sha256(path),
            _rate_resolver=BusinessPlanRateResolverV4.from_frozen_artifacts(root),
            _simulator=simulator_from_contract(contract),
            _financial_cost_sources=FinancialCostSources.from_project_root(root),
        )

    def build_plan(self, state: Any, history: Iterable[Any]):
        history = list(history)
        key = str(state.firm_id).removesuffix(".0").zfill(6)
        if any(str(s.firm_id).removesuffix(".0").zfill(6) != key for s in history):
            raise ValueError("Business-plan history firm mismatch")
        history = sorted((s for s in history if int(s.year) <= int(state.year)), key=lambda s: int(s.year))
        if len({int(s.year) for s in history}) != len(history):
            raise ValueError("Duplicate canonical firm/year history")
        history = history[-3:]
        if not history or int(history[-1].year) != int(state.year):
            raise ValueError(f"Canonical history must end at decision year: {key}/{state.year}")
        base = calibrate_business_plan_non_rate(history)
        if self._financial_cost_sources is None:
            raise ValueError('Production bundle requires verified financial-cost sources')
        amount, _ = self._financial_cost_sources.calibrate(state,history,
            max(0.,float(state.revenue or 0.))*(1.+base.revenue_growth))
        base = replace(base,non_interest_financial_cost=amount)
        lineage = self._rate_resolver.resolve(key, int(state.year))
        rate = lineage.result.selected_rate
        return replace(base, rate_short=rate, rate_long=rate, rate_bond=rate), lineage

    def _candidate_action(self, candidate_id: str) -> SemanticActionV4:
        try:
            vector = self.action_contract["fixed_candidates"][str(candidate_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown frozen V4.3 candidate ID: {candidate_id!r}") from exc
        return SemanticActionV4.from_mapping(vector)

    def _common_lineage(self, *, bp: BusinessPlan, rate_lineage: Any, execution_mode: str, primitive_hash: str | None = None) -> dict[str, Any]:
        payload = json.dumps(bp.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        result = {
            "action_contract_version": self.action_contract["simulator_action_semantic_contract_version"],
            "action_contract_sha256": self.action_contract_sha256,
            "simulator_version": SEMANTIC_FINANCIAL_SIMULATOR_V4_VERSION,
            "simulator_factory_version": "simulator_from_contract",
            "production_bundle_version": BUNDLE_VERSION,
            "non_rate_calibration_version": NON_RATE_CALIBRATION_VERSION,
            "execution_mode": execution_mode,
            "bp_rate_mode": CALIBRATED_BP_RATE_V4_MODE,
            "business_plan_signature": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "bp_rate_selected": rate_lineage.result.selected_rate,
            "bp_rate_selection_source": rate_lineage.result.selection_source,
            "bp_rate_source_year": rate_lineage.result.source_observation_year,
            "bp_rate_contract_id": rate_lineage.bp_rate_contract_id,
            "bp_rate_contract_sha256": rate_lineage.contract_sha256,
        }
        if primitive_hash is not None:
            result["executed_primitive_hash"] = primitive_hash
        return result

    def simulate_candidate(self, state: Any, history: Iterable[Any], candidate_id: str):
        history = list(history)
        bp, rate_lineage = self.build_plan(state, history)
        action = self._candidate_action(candidate_id)
        dividend_plan = self._simulator.baseline_dividend_plan(state, bp)
        result = self._simulator.simulate(state, bp, action, fixed_cash_dividends=dividend_plan)
        primitives = ExecutedFinancialPrimitives(**result.diagnostics["executed_financial_primitives"])
        lineage = self._common_lineage(bp=bp, rate_lineage=rate_lineage, execution_mode=CANDIDATE_POLICY_MODE, primitive_hash=primitives.content_hash)
        lineage.update(candidate_dividend_policy=CANDIDATE_DIVIDEND_POLICY, baseline_cash_dividend_plan=dividend_plan)
        _, cost_lineage = self._financial_cost_sources.calibrate(state,
            [h for h in history if h.year <= state.year],max(0.,float(state.revenue or 0.))*(1.+bp.revenue_growth))
        lineage.update(cost_lineage)
        return result, bp, rate_lineage, action, primitives, lineage

    def simulate_action(
        self,
        state: Any,
        history: Iterable[Any],
        action_mapping: Mapping[str, Any],
        *,
        action_label: str = "LLM_FREE8",
    ):
        """Execute one externally validated eight-dimensional action.

        This is the canonical Stage8 seam for LLM actions. It applies the same
        business plan, broad-financial-cost, fixed-dividend, and accounting
        semantics as frozen candidate execution. No clipping, projection, or
        candidate substitution occurs here.
        """
        history = list(history)
        bp, rate_lineage = self.build_plan(state, history)
        action = SemanticActionV4.from_mapping(action_mapping)
        dividend_plan = self._simulator.baseline_dividend_plan(state, bp)
        result = self._simulator.simulate(
            state, bp, action, fixed_cash_dividends=dividend_plan
        )
        primitives = ExecutedFinancialPrimitives(
            **result.diagnostics["executed_financial_primitives"]
        )
        lineage = self._common_lineage(
            bp=bp,
            rate_lineage=rate_lineage,
            execution_mode=CANDIDATE_POLICY_MODE,
            primitive_hash=primitives.content_hash,
        )
        lineage.update(
            candidate_dividend_policy=CANDIDATE_DIVIDEND_POLICY,
            baseline_cash_dividend_plan=dividend_plan,
            external_action_label=str(action_label),
            external_action_strict_validation=True,
            external_action_posthoc_repair=False,
        )
        _, cost_lineage = self._financial_cost_sources.calibrate(
            state,
            [h for h in history if h.year <= state.year],
            max(0.0, float(state.revenue or 0.0)) * (1.0 + bp.revenue_growth),
        )
        lineage.update(cost_lineage)
        return result, bp, rate_lineage, action, primitives, lineage
    def simulate_historical(self, state: Any, history: Iterable[Any], primitives: ExecutedFinancialPrimitives, source_capex_context: Mapping[str, Any]):
        if not isinstance(primitives, ExecutedFinancialPrimitives):
            raise TypeError("historical V4.3 execution requires ExecutedFinancialPrimitives")
        if primitives.source_kind != "historical":
            raise ValueError("Historical execution refuses candidate primitives")
        if (str(source_capex_context["firm_id"]).removesuffix(".0").zfill(6) != str(state.firm_id).removesuffix(".0").zfill(6)
                or int(source_capex_context["base_year"]) != int(state.year)):
            raise ValueError("Historical source CAPEX key mismatch")
        history = list(history)
        bp, rate_lineage = self.build_plan(state, history)
        result = self._simulator.simulate_primitives(state, bp, primitives, execution_mode=HISTORICAL_REPLAY_MODE,
                                                     source_capex_context=dict(source_capex_context))
        lineage = self._common_lineage(bp=bp, rate_lineage=rate_lineage, execution_mode=HISTORICAL_REPLAY_MODE, primitive_hash=primitives.content_hash)
        _, cost_lineage = self._financial_cost_sources.calibrate(state,
            [h for h in history if h.year <= state.year],max(0.,float(state.revenue or 0.))*(1.+bp.revenue_growth))
        lineage.update(cost_lineage)
        return result, bp, rate_lineage, lineage


def financial_record(state, candidate_id, payload):
    """Lossless common financial serialization for every production consumer."""
    if candidate_id is None:
        result, bp, rate, lineage = payload
        action = None
        primitives = ExecutedFinancialPrimitives(**result.diagnostics["executed_financial_primitives"])
    else:
        result, bp, rate, action, primitives, lineage = payload
    return {
        "firm_id": str(state.firm_id).removesuffix(".0").zfill(6),
        "base_year": int(state.year), "candidate_id": candidate_id,
        **{"state__" + k: v for k, v in state.to_dict().items()},
        "state__pure_interest_expense": lineage.get("decision_pure_interest_expense",state.pure_interest_expense),
        "state__non_interest_financial_cost": lineage.get("decision_non_interest_financial_cost",state.non_interest_financial_cost),
        **{"bp__" + k: v for k, v in bp.to_dict().items()},
        **{"action__" + k: v for k, v in (action.to_dict() if action else {}).items()},
        **{"primitive__" + k: v for k, v in primitives.to_dict().items()},
        **{"sim__" + k: v for k, v in result.state_t1.to_dict().items()},
        **rate.as_dict(), **lineage,
        "diagnostics_json": json.dumps(result.diagnostics, sort_keys=True, ensure_ascii=False, default=str),
        "accounting_check_json": json.dumps(result.accounting_check, sort_keys=True, ensure_ascii=False, default=str),
        "sustainability": str(result.sustainability),
        "plug_used": result.plug_used, "plug_amount": result.plug_amount,
    }
