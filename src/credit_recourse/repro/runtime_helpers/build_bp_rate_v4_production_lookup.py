from __future__ import annotations


def key(firm_id, year):
    return str(firm_id), int(year)


def select_missing(rows, lookup):
    return [row for row in rows if key(row.get("firm_id"), row.get("base_year", row.get("fiscal_year"))) not in lookup]
