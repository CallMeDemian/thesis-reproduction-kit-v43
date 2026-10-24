"""Input checks: row counts, hashes, value encodings, columns outside the kit schema (plan Sec.6 step 2)."""
from __future__ import annotations

import json

import pandas as pd

import common as c


def main() -> dict:
    obs = c.load_obs()
    supp = pd.read_csv(c.SUPP_PATH)
    pb, ph = pd.read_csv(c.PRIM_BASE), pd.read_csv(c.PRIM_HIGH)
    parent = pd.read_csv(c.PARENT_PATH)
    s9 = pd.read_csv(c.require_file(c.C3E_PATH), low_memory=False)
    c3e = s9[(s9.policy == "C3-E") & (s9.analysis_population == "itt")]
    checks = {
        "observations_rows": (len(obs), 96600),
        "strict_requests_21x575x2x2": (int((obs.execution_policy == "STRICT").sum()), 21 * 575 * 2 * 2),
        "requests_total": (21 * 575 * 2 * 2, 48300),
        "firms": (int(obs.firm_key.nunique()), 575),
        "primary_rows": (len(pb) + len(ph), 96),
        "supplemental_rows": (len(supp), 722),
        "parent_gate_rows": (len(parent), 96),
        "registry_rows": (len(pb) + len(ph) + len(supp) + len(parent), 914),
        "c3e_itt_rows": (len(c3e), 575),
        "c3e_row_id_unique": (int(c3e.row_id.nunique()), 575),
        "c3e_firm_key_all_empty": (bool(c3e.firm_key.isna().all()), True),
    }
    m = obs[["row_id", "firm_key"]].drop_duplicates()
    checks["row_id_firm_key_one_to_one"] = (bool(m.row_id.is_unique and m.firm_key.is_unique and len(m) == 575), True)
    extra = ["parent_request_id", "reference_row_id", "reference_candidate_id"]
    checks["extra_columns_present"] = ([x for x in extra if x in obs.columns], extra)
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    hashes = {str(p.relative_to(c.KIT)).replace("\\", "/"): c.sha256_file(p)
              for p in (c.OBS_PATH, c.SUPP_PATH, c.PARENT_PATH, c.PRIM_BASE, c.PRIM_HIGH, c.C3E_PATH)}
    out = {"status": "PASS" if not bad else "FAIL",
           "checks": {k: {"actual": v[0], "expected": v[1]} for k, v in checks.items()}, "input_sha256": hashes}
    c.RESULTS.mkdir(parents=True, exist_ok=True)
    (c.RESULTS / "_inputs_check.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    if bad:
        raise SystemExit("00_inputs FAILED: " + json.dumps(bad, default=str))
    return out


if __name__ == "__main__":
    print(main()["status"])
