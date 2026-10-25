"""Build the E1 observations (one row per model x firm x condition) and score them.

Run only after every request is collected (runs/ALL_DELIVERED.txt exists). Input: runs/e1_selected.jsonl (lowest
delivered attempt per request, Strict ITT already applied by e1_process). Scoring: the path validated by
e1_score_gate.py. Output: E1/observations/e1_observations.parquet and .csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_score as sc  # noqa: E402

EXPECTED = {"openai": "gpt-5.4-mini-2026-03-17", "google": "gemini-3.1-flash-lite"}


def tokens(provider: str, usage) -> dict:
    u = usage or {}
    if provider == "openai":
        det = u.get("output_tokens_details") or {}
        return {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"), "reasoning_tokens": det.get("reasoning_tokens")}
    return {"input_tokens": u.get("promptTokenCount"), "output_tokens": u.get("candidatesTokenCount"), "reasoning_tokens": u.get("thoughtsTokenCount")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-root", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    args = ap.parse_args()
    kit = args.kit_root.resolve()
    e1 = kit / "ablation_20261102/E1"
    if not (e1 / "runs/ALL_DELIVERED.txt").exists():
        raise SystemExit("E1 OBSERVATIONS BLOCKED: not every request is collected (no runs/ALL_DELIVERED.txt)")
    man = pd.read_csv(e1 / "requests/E1_request_manifest.csv", dtype={"row_id": int, "firm_key": str})
    sel = [json.loads(t) for t in (e1 / "runs/e1_selected.jsonl").open(encoding="utf-8")]
    if len(sel) != 3450 or {r["request_id"] for r in sel} != set(man.e1_request_id):
        raise SystemExit("E1 OBSERVATIONS BLOCKED: selected set differs from the manifest")
    if any(r["returned_model"] != EXPECTED[r["provider"]] for r in sel):
        raise SystemExit("E1 OBSERVATIONS BLOCKED: returned_model differs from expected_returned_model_version")
    rows = []
    sd = {r["request_id"]: r for r in sel}
    for m in man.itertuples():
        r = sd[m.e1_request_id]
        a = r["strict_itt_action"]
        if a is None:
            raise SystemExit("E1 OBSERVATIONS BLOCKED: delivered row without ITT action")
        row = {"model_key": m.model_key, "row_id": int(m.row_id), "firm_key": m.firm_key, "e1_condition": m.e1_condition,
               "e1_request_id": m.e1_request_id, "source_request_id": m.source_request_id, "parent_request_id": m.parent_request_id,
               "attempt": r["attempt"], "provider_request_id": r.get("provider_request_id"), "returned_model": r["returned_model"],
               "completion_kind": r["completion_kind"], "strict_usable": bool(r["strict_usable"]),
               "action_application_status": r["action_application_status"],
               "strict_itt_action_json": json.dumps(a, sort_keys=True, separators=(",", ":")),
               "reference_candidate_id": m.reference_candidate_id, "own_C3E_action": m.own_C3E_action,
               **tokens(r["provider"], r.get("usage"))}
        for d in sc.DIMENSIONS:
            row[f"action__{d}"] = float(a[d])
        rows.append(row)
    obs = pd.DataFrame(rows)
    sc.setup(kit, args.run_root)
    scored = sc.score_with_noop(obs[["row_id", *sc.ACTION_COLS]], args.run_root)
    for o in sc.ORACLES:
        for k in ("R_score", "noop_R_score", "delta_R_score"):
            obs[f"{k}_{o}"] = scored[f"{k}_{o}"].to_numpy()
    for o in sc.ORACLES:
        if obs[f"delta_R_score_{o}"].isna().any():
            raise SystemExit("E1 OBSERVATIONS BLOCKED: NaN in delta_R_score")
    out = e1 / "observations"
    out.mkdir(exist_ok=True)
    obs.to_parquet(out / "e1_observations.parquet", index=False)
    obs.drop(columns=[]).to_csv(out / "e1_observations.csv", index=False, lineterminator="\n")
    summary = {"rows": len(obs), "per_model_condition": obs.groupby(["model_key", "e1_condition"]).size().to_dict().__repr__(),
               "strict_unusable_rows": int((~obs.strict_usable).sum()),
               "sha256_parquet": hashlib.sha256((out / "e1_observations.parquet").read_bytes()).hexdigest()}
    (out / "e1_observations_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\r\n")
    print(json.dumps({k: summary[k] for k in ("rows", "strict_unusable_rows", "sha256_parquet")}))


if __name__ == "__main__":
    main()
