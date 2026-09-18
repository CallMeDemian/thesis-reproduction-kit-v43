"""V4.3 production encoder rules, fixed independently of evaluation outcomes."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from credit_recourse.contracts.stage_paths import (
    V43_BP_RATE_PRODUCTION_PATH,
    V43_BP_RATE_SOURCE_PATH,
)
from credit_recourse.contracts.v43_action_contract import CANONICAL_ACTION_CONTRACT_PATH

SCHEMA_VERSION = "V43EncoderContract/3_one_pass_2022"
from credit_recourse.simulator.financial_cost_v43 import SOURCE_PATH
PURE_INTEREST_PATH = SOURCE_PATH
CALIBRATION_PATH = "src/credit_recourse/simulator/business_plan.py"
ACTION_IDS = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")
ACTION_PATH = CANONICAL_ACTION_CONTRACT_PATH.as_posix()
RATE_PATH = (V43_BP_RATE_PRODUCTION_PATH / "production_selected_rate_ledger.parquet").as_posix()
RATE_CONTRACT_PATH = (V43_BP_RATE_SOURCE_PATH / "bp_rate_v4_contract.json").as_posix()
KEYS = ("firm_id", "fiscal_year")
DEBT_FIELDS = ("short_term_debt", "current_portion_long_debt", "long_term_debt", "bonds")
ASSET_FIELDS = (
    "revenue", "current_assets", "non_current_assets", "cash", "short_term_investments",
    "receivables", "inventory", "ppe", "intangibles", "total_liabilities",
    "current_liabilities", "non_current_liabilities", *DEBT_FIELDS, "payables",
    "total_equity", "capital_stock", "retained_earnings", "operating_cf", "investing_cf",
    "financing_cf", "capex", "depreciation", "amortization", "cash_dividends",
)
REVENUE_FIELDS = (
    "cogs", "sga", "gross_profit", "operating_income", "financial_cost", "pure_interest_expense", "pretax_income",
    "tax_expense", "net_income", "operating_cf", "capex", "depreciation", "amortization",
)
RATIOS = {
    "current_ratio": ("current_assets", "current_liabilities"),
    "cash_to_cogs": ("cash", "cogs"),
    "cash_to_current_liabilities": ("cash", "current_liabilities"),
    "interest_coverage": ("operating_income", "pure_interest_expense"),
    "inv_turnover": ("cogs", "inventory"),
    "ar_turnover": ("revenue", "receivables"),
    "ap_turnover": ("cogs", "payables"),
    "capex_to_ppe": ("capex", "ppe"),
    "depreciation_to_ppe": ("depreciation", "ppe"),
}
HISTORY_FEATURES = (
    "log_assets", "log_revenue", "operating_income_to_revenue", "operating_cf_to_assets",
    "total_debt_to_assets", "cash_to_current_liabilities", "current_ratio", "inv_turnover",
    "ar_turnover", "ap_turnover", "capex_to_revenue", "pure_interest_expense_to_total_debt",
)
CATEGORICAL_COLUMNS = ("sector", "rate_source")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    block: str
    formula: str
    sources: tuple[str, ...]
    transform: str = "asinh"
    reason: str = "financial meaning and decision-time availability; no outcome selection"
    time: str = "fiscal_year <= decision_year"
    raw_dtype: str = "float64; unobserved/undefined=NaN"
    tensor_dtype: str = "float32 plus explicit bool missing mask"


def feature_specs():
    rows = []
    def add(name, block, formula, sources, transform="asinh"):
        rows.append(FeatureSpec(name, block, formula, tuple(sources), transform))
    for field in ("total_assets", "revenue"):
        add("log_" + ("assets" if field == "total_assets" else field), "scale_growth",
            "sign(x)*log1p(abs(x)); x in source thousand KRW", (field,), "identity")
    for field in ASSET_FIELDS:
        block = ("debt" if field in DEBT_FIELDS or "liabilit" in field else
                 "investment" if field in ("ppe", "capex", "depreciation", "amortization", "intangibles") else
                 "working_capital" if field in ("receivables", "inventory", "payables") else
                 "cash_flow" if field.endswith("_cf") else "liquidity_scale")
        add(field + "_to_assets", block, field + "/positive(total_assets)", (field, "total_assets"))
    for field in REVENUE_FIELDS:
        name = "broad_financial_cost_to_revenue" if field == "financial_cost" else field + "_to_revenue"
        block = "interest_burden" if field == "pure_interest_expense" else "profitability_cost_cashflow"
        add(name, block, field + "/positive(revenue)", (field, "revenue"))
    for name, (num, den) in RATIOS.items():
        add(name, "interest_burden" if name == "interest_coverage" else "working_capital" if "turnover" in name else "financial_ratios",
            num + "/positive(" + den + ")", (num, den))
    add("total_debt_to_assets", "debt", "sum(all four observed debt stacks)/positive(assets)", (*DEBT_FIELDS, "total_assets"))
    add("gross_ltd_to_assets", "debt", "(noncurrent_LTD+CPLD)/positive(assets)", ("long_term_debt", "current_portion_long_debt", "total_assets"))
    for field in (*DEBT_FIELDS, "pure_interest_expense", "operating_cf"):
        add(field + "_to_total_debt", "interest_burden" if field == "pure_interest_expense" else "debt_structure", field + "/positive(complete total_debt)", (field, *DEBT_FIELDS))
    add("net_debt_to_assets", "debt", "(complete total_debt-cash)/positive(assets)", (*DEBT_FIELDS, "cash", "total_assets"))
    add("net_working_capital_to_assets", "working_capital", "(AR+inventory-AP)/positive(assets)", ("receivables", "inventory", "payables", "total_assets"))
    add("bp_rate", "interest_rate", "frozen BP-rate V4 lookup(firm_id,year); absent lookup is missing", ("selected_rate",))
    add("rate_source_age", "interest_rate", "decision year minus source year; unknown external initialization age remains missing", ("source_observation_year",), "identity")
    add("decision_year", "context", "(fiscal_year-2000)/25", ("fiscal_year",), "identity")
    for name in HISTORY_FEATURES:
        for suffix, formula in (
            ("mean3", "mean(last <=3 finite observations)"),
            ("change_per_year", "(latest-previous)/(actual year gap); requires 2 valid observations"),
            ("std3", "population SD of last <=3 finite observations; requires >=2"),
            ("count3", "number of last <=3 finite observations"),
            ("age", "decision year minus latest finite observation year; missing if none"),
        ):
            add("history__" + name + "__" + suffix, "history", formula, (name,),
                "identity" if suffix in ("count3", "age") else "asinh")
    formulas = {
        "opening_reserve_to_assets": ("max(0,reserve_ratio*COGS,cash-current_assets)/assets", ("cash", "cogs", "current_assets", "total_assets")),
        "opening_repayment_cash_to_assets": ("max(cash-opening_reserve,0)/assets; excludes projected cash flow and DL interest saving", ("cash", "cogs", "current_assets", "total_assets")),
        "dl_requested_principal_to_assets": ("frozen DL fraction*complete gross principal/assets", (*DEBT_FIELDS, "total_assets")),
        "dl_cash_coverage": ("min(opening_cash/requested_DL_principal,1); observed zero request gives 0", (*DEBT_FIELDS, "cash", "cogs", "current_assets")),
        "rf_transferable_short_to_assets": ("max(SD,0)*frozen RF fraction/assets", ("short_term_debt", "total_assets")),
        "near_maturity_share": ("(SD+CPLD)/positive(complete total_debt)", DEBT_FIELDS),
        "bp_capex_to_assets": ("frozen non-rate BP calibration(last 3 rows) baseline CAPEX/assets; missing without calibration observations", ("revenue", "capex", "total_assets")),
        "maintenance_capex_to_assets": ("min(BP CAPEX,depreciation limited to PPE)/assets", ("revenue", "capex", "ppe", "depreciation", "total_assets")),
        "discretionary_capex_to_assets": ("max(BP CAPEX-maintenance CAPEX,0)/assets", ("revenue", "capex", "ppe", "depreciation", "total_assets")),
        "cx_discretionary_share": ("discretionary CAPEX/positive(BP CAPEX)", ("revenue", "capex", "ppe", "depreciation")),
        "ar_stock_to_assets": ("max(AR,0)/assets", ("receivables", "total_assets")),
        "inventory_stock_to_assets": ("max(inventory,0)/assets", ("inventory", "total_assets")),
        "ap_stock_to_assets": ("max(AP,0)/assets", ("payables", "total_assets")),
        "ap_turnover_headroom": ("max(current AP turnover-0.5,0); frozen floor", ("cogs", "payables")),
        "oe_cost_base_to_assets": ("(max(COGS,0)+max(SGA,0))/assets", ("cogs", "sga", "total_assets")),
    }
    for name, (formula, fields) in formulas.items():
        add("ability__" + name, "actionability", formula, fields)
    for candidate in ACTION_IDS[1:]:
        add("ability__" + candidate + "__joint_fraction", "actionability",
            "minimum decision-time coverage of contract-active primitives; unknown constituent => missing",
            ("frozen_candidate_vector", "opening stocks", "factual operating ratios", "BP CAPEX inputs"), "identity")
    if len({s.name for s in rows}) != len(rows):
        raise ValueError("Duplicate V43 feature name")
    return tuple(rows)


@dataclass(frozen=True)
class V43EncoderContract:
    action_contract_sha256: str
    rate_contract_sha256: str
    rate_ledger_sha256: str
    action_contract: dict
    pure_interest_source_sha256: str
    non_rate_calibration_sha256: str
    version: str = SCHEMA_VERSION

    @classmethod
    def from_project_root(cls, root):
        root = Path(root)
        action = json.loads((root / ACTION_PATH).read_text(encoding="utf-8"))
        if tuple(action["candidate_ids"]) != ACTION_IDS or len(action["action_columns"]) != 8:
            raise ValueError("V43 requires frozen nine classes and eight semantic primitives")
        if action["simulator_action_semantic_contract_version"] != "simulator_action_semantic_contract_v4_3_dl_interest_full_year":
            raise ValueError("Wrong action contract")
        if any(action["target_nominal_normalized_intensity"][a] != (0.0 if a == "A0" else 1.0) for a in ACTION_IDS):
            raise ValueError("Frozen nominal intensities changed")
        return cls(file_sha256(root / ACTION_PATH), file_sha256(root / RATE_CONTRACT_PATH), file_sha256(root / RATE_PATH), action,
                   file_sha256(root / PURE_INTEREST_PATH), file_sha256(root / CALIBRATION_PATH))

    @property
    def specs(self):
        return feature_specs()

    @property
    def continuous_columns(self):
        return tuple(s.name for s in self.specs)

    def to_dict(self):
        return {
            "pure_interest_source": {"path": PURE_INTEREST_PATH, "sha256": self.pure_interest_source_sha256,
                "code": "U01B550010000", "unit": "thousand KRW", "join": "same firm and fiscal year only",
                "absent_or_conflicted": "missing; never broad-cost substitution",
                "interest_coverage_zero_denominator": "undefined ratio stays NaN; zero expense/revenue and expense/debt remain observed zero"},
            "non_rate_calibration_sha256": self.non_rate_calibration_sha256,
            "revision_reason": "User-requested semantic corrections: preserve observed BP zeros and distinguish pure interest from broad financial cost; no outcome-based feature tuning",
            "version": self.version, "feature_specs": [asdict(s) for s in self.specs],
            "categorical_columns": CATEGORICAL_COLUMNS, "action_ids": ACTION_IDS,
            "action_columns": self.action_contract["action_columns"],
            "action_contract_sha256": self.action_contract_sha256,
            "rate_contract_sha256": self.rate_contract_sha256, "rate_ledger_sha256": self.rate_ledger_sha256,
            "action_representation": "9-class one-hot + 8 primitive masks + 8 directions; nominal nonnoop=1",
            "architecture": {"d_model": 256, "n_heads": 8, "n_layers": 4, "ff_mult": 4, "actor_logits": 9, "critic": "explicit state * action embedding interaction"},
            "statistics": {"fit_population": "unique union of common rl_fit_allowed Stage3/4/5 decision keys; t<=2022 and outcome<=2023", "center": "training median after fixed transform", "scale": "training IQR if >1e-6 else 1", "clipping": "none", "all_missing_training": "center 0/scale 1 are tensor storage constants; missing mask retained", "categorical": "training vocabulary only; 0=missing,1=unseen"},
            "temporal": {"history": "same firm <=decision year; <=3 latest finite observations per feature", "eval_base_year": 2024, "train_transition_year_max": 2022, "train_outcome_year_max": 2023, "evaluation_rollout_year": 2025, "target_supplement": "missing canonical next states use same-year authoritative actual accounts only as targets; excluded from other decision histories", "resolution": "annual decision-time contract; no within-year filing-date claim"},
            "missingness": "partial debt sums and undefined ratios remain NaN; tensor zero placeholders always have bool masks; missing action never maps to noop",
            "actionability_rules": {"scope": "opening-stock/endpoint resource coverage, not full simulated execution feasibility", "DL": "opening cash above cash/COGS and current-asset reserve; no future OCF or interest saving", "RF": "positive observed SD and complete maturity stacks", "CX": "observed discretionary BP CAPEX >0", "WC": "observed positive stocks and numerator; AP decrement constrained by 0.5 turnover floor", "OE": "observed COGS/revenue and SGA/revenue headroom above zero floors", "MX": "minimum constituent fractions for own frozen vector; no assumed cash released by another primitive"},
            "forbidden": ["candidate outcome", "Alpha", "Oracle/rank/ceiling", "reward", "future history", "legacy 10D action", "statistics outside rl_fit_allowed"],
            "adoption_policy": "financial semantics fixed before new matrices and diagnostics; no outcome-based adoption or tail tuning",
        }

    @property
    def schema_hash(self):
        return content_hash(self.to_dict())
