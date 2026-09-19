from __future__ import annotations

import pandas as pd

from credit_recourse.simulator.firm_state import load_firm_state_from_columns


def _row_to_firm_state(row: pd.Series):
    values = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    firm_id = values.get("firm_id", values.get("corp_code", values.get("id", "unknown")))
    year = values.get("fiscal_year", values.get("year", 0))
    sector = values.get("sector", values.get("sector_7", "Unknown"))
    return load_firm_state_from_columns(values, str(firm_id), int(float(year)), str(sector))
