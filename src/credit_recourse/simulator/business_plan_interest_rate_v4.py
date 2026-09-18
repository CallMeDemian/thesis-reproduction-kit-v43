"""Production resolver for the frozen provider-observed BP-rate v4 contract.

The resolver is deliberately a lookup over the frozen v4 selection ledger.
It does not re-estimate rates, inspect ``financial_cost``, or use replay row
identifiers.  The only lookup key is ``(firm_id, base_year)``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping



RATE_CONTRACT_VERSION = "business_plan_interest_rate_v4_provider_observed"
RESOLVER_VERSION = "BusinessPlanRateResolverV4/2"
_FREEZE = "archive/DEPLOYED_RELEASE/stage2_candidate_projection/runtime_inputs/business_plan_rate_v4/source"


def _key(value: object) -> str:
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw.zfill(6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name}: {value!r}")
    return result


@dataclass(frozen=True)
class BorrowingRateV4Result:
    selected_rate: float
    selection_source: str
    source_observation_year: int | None
    provider_rate: float | None
    rating_grade_used: str | None
    ceiling: float


@dataclass(frozen=True)
class BorrowingRateV4Lineage:
    firm_id: str
    base_year: int
    result: BorrowingRateV4Result
    bp_rate_contract_id: str
    contract_sha256: str
    ledger_sha256: str
    resolver_version: str

    def as_dict(self) -> dict[str, object]:
        r = self.result
        return {
            "business_plan_rate": r.selected_rate,
            "business_plan_rate_short": r.selected_rate,
            "business_plan_rate_long": r.selected_rate,
            "business_plan_rate_bond": r.selected_rate,
            "business_plan_rate_source": r.selection_source,
            "business_plan_rate_source_observation_year": r.source_observation_year,
            "business_plan_rate_provider_rate": r.provider_rate,
            "business_plan_rate_rating_grade_used": r.rating_grade_used,
            "business_plan_rate_ceiling": r.ceiling,
            "business_plan_rate_contract_id": self.bp_rate_contract_id,
            "business_plan_rate_contract_sha256": self.contract_sha256,
            "business_plan_rate_ledger_sha256": self.ledger_sha256,
            "business_plan_rate_resolver_version": self.resolver_version,
            "external_contemporaneous_initialization_fallback": (
                r.selection_source == "external_contemporaneous_initialization_fallback"
            ),
        }


class BusinessPlanRateResolverV4:
    """Fail-closed lookup of frozen V4 rates by firm and decision/base year."""

    def __init__(
        self,
        selected_rate_rows: Iterable[Mapping[str, object]],
        *,
        contract: Mapping[str, object],
        contract_sha256: str,
        ledger_sha256: str,
        rating_grade_by_ordinal: Mapping[int, str] | None = None,
        valid_provider_years: Iterable[int] | None = None,
    ) -> None:
        if str(contract.get("version")) != RATE_CONTRACT_VERSION:
            raise ValueError("Unexpected BP-rate v4 contract identity")
        self.contract = dict(contract)
        self.contract_sha256 = str(contract_sha256)
        self.ledger_sha256 = str(ledger_sha256)
        self._grade_by_ordinal = {int(k): str(v) for k, v in (rating_grade_by_ordinal or {}).items()}
        self._valid_provider_years = sorted({int(y) for y in (valid_provider_years or [])})
        grouped: dict[tuple[str, int], list[Mapping[str, object]]] = {}
        for raw in selected_rate_rows:
            if "firm_id" not in raw or "base_year" not in raw:
                raise ValueError("V4 ledger row lacks firm_id or base_year")
            grouped.setdefault((_key(raw["firm_id"]), int(raw["base_year"])), []).append(raw)
        self._rows: dict[tuple[str, int], Mapping[str, object]] = {}
        for identity, rows in grouped.items():
            baseline = self._canonical(rows[0])
            if any(self._canonical(row) != baseline for row in rows[1:]):
                raise ValueError(f"Conflicting duplicate V4 rate lookup key: {identity}")
            self._rows[identity] = rows[0]

    @staticmethod
    def _canonical(row: Mapping[str, object]) -> tuple[object, ...]:
        def optional_number(value: object) -> float | None:
            if value is None:
                return None
            try:
                if math.isnan(float(value)):
                    return None
            except (TypeError, ValueError):
                return None
            return _number(value, name="duplicate-ledger value")

        return (
            _number(row.get("selected_rate"), name="selected_rate"),
            str(row.get("selection_source")),
            optional_number(row.get("provider_rate")),
            optional_number(row.get("rating_ordinal")),
            _number(row.get("ceiling"), name="ceiling"),
            _number(row.get("rate_short"), name="rate_short"),
            _number(row.get("rate_long"), name="rate_long"),
            _number(row.get("rate_bond"), name="rate_bond"),
        )

    @classmethod
    def from_frozen_artifacts(cls, project_root: Path) -> "BusinessPlanRateResolverV4":
        root = Path(project_root).resolve()
        freeze = root / _FREEZE
        contract_path = freeze / "bp_rate_v4_contract.json"
        original_ledger_path = freeze / "selected_rate_lineage.parquet"
        production_freeze = root / "archive/DEPLOYED_RELEASE/stage2_candidate_projection/runtime_inputs/business_plan_rate_v4/production"
        production_ledger_path = production_freeze / "production_selected_rate_ledger.parquet"
        production_preflight = production_freeze / "production_lookup_preflight.json"
        ledger_path = production_ledger_path if production_ledger_path.is_file() else original_ledger_path
        mapping_path = freeze / "rating_mapping.json"
        provider_path = freeze / "provider_rate_observation_ledger.parquet"
        if not all(path.is_file() for path in (contract_path, ledger_path, mapping_path, provider_path)):
            raise FileNotFoundError("Frozen BP-rate v4 artifacts are incomplete")
        if ledger_path == production_ledger_path:
            review = json.loads(production_preflight.read_text(encoding="utf-8")) if production_preflight.is_file() else {}
            if review.get("status") != "PASS" or review.get("candidate_coverage_missing") != 0 or not review.get("original_test3_selection_bitwise_equal"):
                raise ValueError("BP-rate V4 production lookup panel is not admitted")
        import pandas as pd

        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        grades = {
            int(row["rating_num_notch"]): str(row["grade_notch"])
            for row in mapping.get("grades", [])
            if row.get("rating_num_notch") is not None and row.get("grade_notch") is not None
        }
        provider = pd.read_parquet(provider_path)
        valid_years = provider.loc[provider.source_valid.fillna(False), "fiscal_year"].dropna().astype(int).tolist()
        ledger = pd.read_parquet(ledger_path)
        return cls(
            ledger.to_dict("records"),
            contract=contract,
            contract_sha256=_sha256(contract_path),
            ledger_sha256=_sha256(ledger_path),
            rating_grade_by_ordinal=grades,
            valid_provider_years=valid_years,
        )

    def _source_observation_year(self, *, base_year: int, source: str) -> int | None:
        if source == "external_contemporaneous_initialization_fallback":
            return None
        if source == "provider_observed_valid":
            return base_year
        eligible = [year for year in self._valid_provider_years if year <= base_year]
        return max(eligible) if eligible else None

    def resolve(self, firm_id: object, base_year: int) -> BorrowingRateV4Lineage:
        identity = (_key(firm_id), int(base_year))
        row = self._rows.get(identity)
        if row is None:
            raise KeyError(f"Missing frozen BP-rate v4 lookup: firm_id={identity[0]} base_year={identity[1]}")
        rate = _number(row.get("selected_rate"), name="selected_rate")
        ceiling = _number(row.get("ceiling"), name="ceiling")
        if rate <= 0.0 or rate > ceiling:
            raise ValueError(f"Frozen V4 rate outside model domain: {identity}")
        stack_rates = [_number(row.get(name), name=name) for name in ("rate_short", "rate_long", "rate_bond")]
        if any(abs(value - rate) > 1e-12 for value in stack_rates):
            raise ValueError(f"Artificial debt-stack spread in frozen V4 ledger: {identity}")
        source = str(row.get("selection_source"))
        raw_provider = row.get("provider_rate")
        provider_rate = None if raw_provider is None or (isinstance(raw_provider, float) and math.isnan(raw_provider)) else _number(raw_provider, name="provider_rate")
        ordinal = row.get("rating_ordinal")
        rating_grade = None
        try:
            if ordinal is not None and not math.isnan(float(ordinal)):
                rating_grade = self._grade_by_ordinal.get(int(ordinal))
        except (TypeError, ValueError):
            pass
        return BorrowingRateV4Lineage(
            firm_id=identity[0], base_year=identity[1],
            result=BorrowingRateV4Result(
                selected_rate=rate, selection_source=source,
                source_observation_year=self._source_observation_year(base_year=identity[1], source=source),
                provider_rate=provider_rate, rating_grade_used=rating_grade, ceiling=ceiling,
            ),
            bp_rate_contract_id=RATE_CONTRACT_VERSION,
            contract_sha256=self.contract_sha256, ledger_sha256=self.ledger_sha256,
            resolver_version=RESOLVER_VERSION,
        )

