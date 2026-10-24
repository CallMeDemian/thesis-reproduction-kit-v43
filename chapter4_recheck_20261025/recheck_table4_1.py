"""Recheck of thesis Table 4-1 (direction agreement, AUC) and the upgrade share used in section 4.2.

Runs the kit's Loop-B1 code (credit_recourse.oracle.verification.verify_stage1_substrate_validation) on the
deployed Oracle firm-year outputs inside the distribution zip, which is a regular file (no Git LFS needed).

Usage (kit root, Python 3.11 kit environment):
    python chapter4_recheck_20261025/recheck_table4_1.py
Also reports the cases whose three years (t-1, t, t+1) are all 2020 or later (Table 4-1 note).
Writes chapter4_recheck_20261025/table4_1_direction_auc.json and exits 1 if a value differs from the thesis.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
sys.path.insert(0, str(KIT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_recourse.oracle.verification import verify_stage1_substrate_validation as b1  # noqa: E402

ZIP = KIT / "frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip"
PREFIX = "THESIS_REPRO_KIT_v2.1.1_FINAL/archive/DEPLOYED_RELEASE/stage1_oracle_backends/"
CODE = KIT / "src/credit_recourse/oracle/verification/verify_stage1_substrate_validation.py"

# Thesis (2026-10-25 revision): Table 4-1 and section 4.1 / 4.2.
EXPECTED = {
    "alpha": {"movers": 165, "agreement_pct": 70.9, "ci95_pct": [63.6, 77.3], "auc": 0.787},
    "beta": {"movers": 165, "agreement_pct": 72.7, "ci95_pct": [65.5, 78.9], "auc": 0.777},
    "gamma": {"movers": 165, "agreement_pct": 68.5, "ci95_pct": [61.0, 75.1], "auc": 0.782},
}
EXPECTED_UPGRADE_SHARE_PCT = 53.9  # "약 54%" in section 4.2
# Table 4-1 note: cases whose three years (t-1, t, t+1) are all 2020 or later.
EXPECTED_FULL_OOT = {"alpha": (116, 73.3, 0.798), "beta": (116, 75.9, 0.751), "gamma": (116, 70.7, 0.794)}
EXPECTED_OOT_BY_YEAR = {2020: 22, 2021: 27, 2022: 48, 2023: 68}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mover_rating_years(panel: pd.DataFrame, score_col: str) -> list[int]:
    """Year t+1 of every mover, built with the same conditions as b1._lead_pairs (which does not return years)."""
    years: list[int] = []
    for _, g in panel.sort_values([b1.KEY, "year"]).groupby(b1.KEY):
        yrs = g["year"].to_numpy()
        rat = g["rating_num_10"].to_numpy(dtype=float)
        sc = g[score_col].to_numpy(dtype=float)
        for i in range(1, len(g) - 1):
            consecutive = yrs[i + 1] - yrs[i] == 1 and yrs[i] - yrs[i - 1] == 1
            observed = not np.isnan([rat[i], rat[i + 1], sc[i], sc[i - 1]]).any()
            if consecutive and observed and rat[i + 1] != rat[i]:
                years.append(int(yrs[i + 1]))
    return years


def main() -> int:
    if not ZIP.exists():
        raise SystemExit(f"missing {ZIP}")
    head = ZIP.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs"):
        raise SystemExit(f"{ZIP.name} is a Git LFS pointer; run: git lfs pull --include \"{ZIP.relative_to(KIT).as_posix()}\"")
    out = {"source_zip": ZIP.relative_to(KIT).as_posix(), "source_zip_sha256": sha256(ZIP.read_bytes()),
           "loop_b1_code": CODE.relative_to(KIT).as_posix(), "loop_b1_code_sha256": sha256(CODE.read_bytes()),
           "python": platform.python_version(), "pandas": pd.__version__,
           "definition": b1.evaluate_stage1_backend_panel.__doc__.strip() + " Lead pairs: Oracle score change t-1 -> t "
                         "versus real 10-grade rating change t -> t+1; split = split of year t+1; rating-unchanged rows excluded; "
                         "score change 0 among movers counts as a non-match; Wilson 95% interval.",
           "members": {}, "oracles": {}, "checks": {}}
    ok = True
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(ZIP) as z:
        bd = Path(tmp)
        for name, (rel, _col) in b1.BACKENDS.items():
            data = z.read(PREFIX + rel)
            (bd / rel).parent.mkdir(parents=True, exist_ok=True)
            (bd / rel).write_bytes(data)
            out["members"][PREFIX + rel] = sha256(data)
        errors: list[str] = []
        rmap = b1._load_authoritative_rating_map(bd, errors)
        for name, (rel, col) in b1.BACKENDS.items():
            panel, score_col, _status = b1._load_backend_score_panel(name, bd / rel, col, rmap, errors)
            res = b1.evaluate_stage1_backend_panel(panel, score_col)
            pairs = b1._lead_pairs(panel, score_col)
            oot_movers = pairs[(pairs["split"] == "oot") & (pairs["d_rating"] != 0)]
            rec = {}
            for split in ("oot", "dev", "all"):
                s = res[split]
                rec[split] = {"movers": s["n_movers"], "matches": s["lead_direction_agreement_matches"],
                              "agreement": s["lead_direction_agreement"], "ci95": s["lead_direction_agreement_ci95"],
                              "auc": s["upgrade_vs_downgrade_auc"], "score_change_zero_movers": s["n_score_delta_zero_movers"]}
            rec["oot"]["upgrades"] = int((oot_movers["improvement"] > 0).sum())
            years = [y for y in mover_rating_years(panel, score_col) if 2020 <= y <= 2023]
            rec["oot"]["movers_by_rating_year"] = {y: years.count(y) for y in sorted(set(years))}
            full = b1._lead_pairs(panel[panel["year"] >= 2020], score_col)
            fm = full[full["d_rating"] != 0]
            fa = b1.compute_direction_agreement(fm["d_score_lead"], fm["improvement"])
            rec["oot_all_three_years_2020_or_later"] = {
                "movers": fa["n_movers"], "matches": fa["n_matches"], "agreement": fa["direction_agreement"],
                "ci95": fa["direction_agreement_ci95"],
                "auc": b1._auc_upgrade_vs_downgrade(fm["d_score_lead"].to_numpy(), fm["improvement"].to_numpy())}
            rec["verdict"] = res["verdict"]
            out["oracles"][name] = rec
            exp, got = EXPECTED[name], rec["oot"]
            fo, fe = rec["oot_all_three_years_2020_or_later"], EXPECTED_FULL_OOT[name]
            checks = {
                "movers_by_rating_year": got["movers_by_rating_year"] == EXPECTED_OOT_BY_YEAR,
                "all_three_years_2020_or_later": (fo["movers"], round(100 * fo["agreement"], 1), round(fo["auc"], 3)) == fe,
                "movers": got["movers"] == exp["movers"],
                "agreement_pct": round(100 * got["agreement"], 1) == exp["agreement_pct"],
                "ci95_pct": [round(100 * v, 1) for v in got["ci95"]] == exp["ci95_pct"],
                "auc": round(got["auc"], 3) == exp["auc"],
            }
            out["checks"][name] = checks
            ok &= all(checks.values())
        if errors:
            raise SystemExit("; ".join(errors))
    oot = out["oracles"]["alpha"]["oot"]
    share = round(100 * oot["upgrades"] / oot["movers"], 1)
    out["upgrade_share_oot_pct"] = share
    out["checks"]["upgrade_share_oot_pct"] = share == EXPECTED_UPGRADE_SHARE_PCT
    ok &= out["checks"]["upgrade_share_oot_pct"]
    out["status"] = "PASS" if ok else "FAIL"
    (HERE / "table4_1_direction_auc.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, rec in out["oracles"].items():
        o = rec["oot"]
        print(f"{name:5s} OOT movers {o['movers']}  agreement {100 * o['agreement']:.1f}% "
              f"[{100 * o['ci95'][0]:.1f}, {100 * o['ci95'][1]:.1f}]  AUC {o['auc']:.3f}  ({o['matches']}/{o['movers']})")
    for name, rec in out["oracles"].items():
        f = rec["oot_all_three_years_2020_or_later"]
        print(f"{name:5s} all three years 2020+  movers {f['movers']}  agreement {100 * f['agreement']:.1f}%  AUC {f['auc']:.3f}")
    print(f"OOT movers by rating year {out['oracles']['alpha']['oot']['movers_by_rating_year']}")
    print(f"upgrade share among OOT movers {share}%  ->  {out['status']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
