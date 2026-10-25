"""E1 evaluation gate: the September C4, C4R and C6-E rows of the E1 cell (both models, 3,450 rows)
pushed through the E1 scoring path must reproduce every frozen delta_R_score_alpha/beta/gamma
(np.isclose, atol = rtol = 1e-9). Also compares the path's own A0 scores with the frozen noop scores.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_score as sc  # noqa: E402

V13 = "frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-root", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N rows only (gate result is then not valid)")
    args = ap.parse_args()
    kit = args.kit_root.resolve()
    sc.setup(kit, args.run_root)
    obs = pd.read_parquet(kit / V13 / "frozen_inputs/stage8/canonical_itt_observations.parquet")
    cell = obs[
        (obs.execution_policy == "STRICT") & (obs.reasoning_regime == "HIGH") & (obs.action_space == "FREE8")
        & (obs.budget == "B1") & (obs.information_condition == "IC-b") & (obs.replicate == 1)
        & obs.policy_condition.isin(["C4", "C4R", "C6-E"])
    ].reset_index(drop=True)
    assert len(cell) == 3450, len(cell)
    if args.limit:
        cell = cell.head(args.limit)
    acts = pd.DataFrame([json.loads(s) for s in cell["strict_itt_action_json"]])
    actions = pd.DataFrame({"row_id": cell["row_id"].astype(int).to_numpy()})
    for d in sc.DIMENSIONS:
        actions[f"action__{d}"] = acts[d].astype(float).to_numpy()
    t0 = time.time()
    res = sc.score_with_noop(actions, args.run_root)
    secs = round(time.time() - t0, 1)
    report = {"rows": len(cell), "seconds": secs, "tolerance": {"atol": 1e-9, "rtol": 1e-9}, "oracles": {}}
    ok = True
    for o in sc.ORACLES:
        d_ok = np.isclose(res[f"delta_R_score_{o}"].to_numpy(), cell[f"delta_R_score_{o}"].to_numpy(), atol=1e-9, rtol=1e-9)
        r_ok = np.isclose(res[f"R_score_{o}"].to_numpy(), cell[f"R_score_{o}"].to_numpy(), atol=1e-9, rtol=1e-9)
        n_ok = np.isclose(res[f"noop_R_score_{o}"].to_numpy(), cell[f"noop_R_score_{o}"].to_numpy(), atol=1e-9, rtol=1e-9)
        diff = np.abs(res[f"delta_R_score_{o}"].to_numpy() - cell[f"delta_R_score_{o}"].to_numpy())
        report["oracles"][o] = {"delta_match": int(d_ok.sum()), "score_match": int(r_ok.sum()), "noop_match": int(n_ok.sum()),
                                "max_abs_delta_diff": float(np.nanmax(diff)), "nan_rows": int(res[f"delta_R_score_{o}"].isna().sum())}
        ok &= bool(d_ok.all() and r_ok.all() and n_ok.all())
    report["status"] = "PASS" if ok and not args.limit else ("DEBUG_ONLY" if args.limit else "FAIL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\r\n")
    print(json.dumps(report, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
