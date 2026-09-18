from __future__ import annotations

import re


def _token(value) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").upper()


def result_id(row) -> str:
    if row.get("result_class") == "PRIMARY":
        return "PRIMARY_{}_{}_{}_{}_{}_{}_{}_{}".format(
            _token(row["model"]), _token(row["reasoning_regime"]), _token(row["execution_policy"]),
            _token(row["action_space"]), _token(row["budget"]), _token(row["contrast"]), _token(row["oracle"]),
            _token(row.get("replicate", 1)),
        )
    return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        _token(row.get("analysis_id", row.get("result_class", "RESULT"))),
        _token(row.get("model", "NA")), _token(row.get("reasoning_regime", "NA")),
        _token(row.get("execution_policy", "NA")), _token(row.get("action_space", "NA")),
        _token(row.get("budget", "NA")), _token(row.get("lhs", row.get("contrast", "NA"))),
        _token(row.get("rhs", "NA")), _token(row.get("oracle", "NA")),
        _token(row.get("information_condition", row.get("info", "NA"))),
        _token(row.get("replicate", "NA")), _token(row.get("population", "NA")),
    )
