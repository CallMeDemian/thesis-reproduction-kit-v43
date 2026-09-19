from __future__ import annotations
import json
from pathlib import Path

CANONICAL_ACTION_CONTRACT_PATH = Path("contracts/scientific/v43_action_contract.json")
ACTION_IDS = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")
ACTION_DIMENSIONS = ("growth_capex_reduction_pct", "deleveraging_total_debt_pct", "refinancing_short_debt_pct", "inv_turnover_chg", "ar_turnover_chg", "ap_turnover_chg", "cogs_ratio_chg", "sga_ratio_chg")

def load_action_contract(root: Path) -> dict:
    payload = json.loads((Path(root) / CANONICAL_ACTION_CONTRACT_PATH).read_text(encoding="utf-8"))
    expected = tuple(f"action__{name}" for name in ACTION_DIMENSIONS)
    columns = tuple(payload.get("action_columns", ()))
    if tuple(payload.get("candidate_ids", ())) != ACTION_IDS or columns != expected:
        raise ValueError("fresh V4.3 action contract must be exactly 8D with 9 canonical actions")
    if any("revenue_growth" in name for name in columns):
        raise ValueError("revenue_growth must remain an exogenous scenario, never a policy action")
    return payload
