"""Minimal canonical account registry used by the fresh simulator boundary.

The registry deliberately resolves only decision-time accounting fields.  It does
not import a historical path or silently substitute broad financial cost for
pure interest expense.
"""
from __future__ import annotations

import re

ACCOUNT_REGISTRY = (
    "revenue", "cogs", "gross_profit", "sga", "operating_income", "financial_cost",
    "pure_interest_expense", "non_interest_financial_cost", "pretax_income", "tax_expense",
    "net_income", "comprehensive_income", "depreciation", "amortization", "interest_income",
    "dividend_income", "total_assets", "current_assets", "non_current_assets", "cash",
    "short_term_investments", "receivables", "inventory", "ppe", "intangibles",
    "total_liabilities", "current_liabilities", "non_current_liabilities", "short_term_debt",
    "current_portion_long_debt", "long_term_debt", "bonds", "payables", "total_equity",
    "capital_stock", "retained_earnings", "ending_capital_stock", "ending_capital_surplus",
    "ending_other_capital", "ending_oci", "ending_retained_earnings", "cash_dividends",
    "operating_cf", "investing_cf", "financing_cf", "capex",
)

ITEM_CODE_MAP = {name: name for name in ACCOUNT_REGISTRY}
ITEM_CODE_MAP.update({"U01A820000000": "current_liabilities", "U01B550010000": "pure_interest_expense"})
REVERSE_ITEM_CODE_MAP = {value: key for key, value in ITEM_CODE_MAP.items()}


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def aliases_for(field: str) -> tuple[str, ...]:
    if field not in ACCOUNT_REGISTRY:
        return (field,)
    return tuple(dict.fromkeys((field, f"raw__{field}", f"sim__{field}", f"next__{field}", REVERSE_ITEM_CODE_MAP.get(field, ""))))


def _find_column(columns, token: str):
    if not token:
        return None
    wanted = _norm(token)
    for column in columns:
        if _norm(column) == wanted:
            return column
    return None


def resolved_field_values(values: dict, *, next_state: bool = False) -> dict:
    columns = list(values)
    result = {}
    for field in ACCOUNT_REGISTRY:
        choices = aliases_for(field)
        if next_state:
            choices = tuple(f"next__{field}" if item == field else item for item in choices) + choices
        selected = next((column for token in choices if (column := _find_column(columns, token)) is not None), None)
        if selected is not None:
            result[field] = values[selected]
    return result
