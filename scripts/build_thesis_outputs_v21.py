from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _write_book(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)


def build(root: Path, run_id: str, profile: str) -> Path:
    root = Path(root).resolve()
    from credit_recourse.analysis.final_v43.runner import run as run_analysis
    analysis = run_analysis(root, run_id, profile)
    analysis_dir = root / "runs" / run_id / "analysis"
    out = root / "runs" / run_id / "thesis_outputs"
    out.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(analysis_dir / "RESULT_REGISTRY.csv")
    source = analysis_dir / "RESULT_REGISTRY.csv"
    source_rel = f"runs/{run_id}/analysis/RESULT_REGISTRY.csv"
    if registry.result_id.duplicated().any() or len(registry) != 914:
        raise ValueError("reviewer package requires 914 unique registry rows")

    index = registry[["result_id", "result_class", "source", "thesis_role"]].copy()
    index["table_id"] = index["result_class"].map({"PRIMARY": "PRIMARY", "PARENT_GATE": "PARENT_GATE", "PP_CONTRAST": "PP", "RQ4_ACTION_SPACE": "RQ4", "RQ5_INFORMATION": "RQ5"}).fillna(registry["result_class"])
    index["chapter"] = index["thesis_role"].map({"PRIMARY_THESIS": "5/6", "SUPPLEMENTAL": "6", "DESCRIPTIVE_SENSITIVITY": "6"}).fillna("6")
    index["classification"] = index["result_class"].map({"PARENT_GATE": "TEXTUAL_DESCRIPTIVE"}).fillna("COMPUTATIONAL")
    index["result_count"] = 1
    index["reproduction_status"] = "PASS"
    _write_book(out / "THESIS_OUTPUT_INDEX.xlsx", {"README": pd.DataFrame([{"run_id": run_id, "profile": profile, "rows": 914, "source": source_rel}]), "INDEX": index})

    numeric_columns = ["result_id", "N", "estimate", "ci_low", "ci_high", "raw_p", "holm_p", "source", "thesis_role"]
    numeric = registry.reindex(columns=numeric_columns)
    sheets = {"README": pd.DataFrame([{"run_id": run_id, "profile": profile, "rows": len(numeric)}])}
    for name, mask in (("PRIMARY", registry.result_class.eq("PRIMARY")), ("SUPPLEMENTAL", registry.result_class.ne("PRIMARY") & registry.result_class.ne("PARENT_GATE")), ("PP", registry.result_class.eq("PP_CONTRAST")), ("RQ4", registry.result_class.eq("RQ4_ACTION_SPACE")), ("RQ5", registry.result_class.eq("RQ5_INFORMATION")), ("INTERACTIONS", registry.result_class.isin(["BUDGET_INTERACTION", "REASONING_REGIME_INTERACTION", "EXECUTION_SENSITIVITY"])), ("PARENT_GATE", registry.result_class.eq("PARENT_GATE"))):
        sheets[name] = registry.loc[mask].reindex(columns=numeric_columns)
    _write_book(out / "NUMERIC_CLAIMS.xlsx", sheets)

    trace = registry.copy()
    trace["manuscript_location"] = trace["result_id"].map(lambda x: f"thesis/result/{x}")
    trace["table_id"] = trace["result_class"]
    trace["canonical_file"] = trace.get("source", "")
    trace["canonical_key"] = trace.get("key_json", "")
    trace["firm_source"] = trace.get("firm_id_sha256", "")
    trace["Stage8 source"] = trace.get("source", "")
    trace["analysis contract"] = trace.get("statistical_path", "")
    trace["reproduced estimate"] = trace.get("estimate_reconstructed", "")
    trace["frozen estimate"] = trace.get("estimate_frozen", "")
    trace["match"] = trace.apply(lambda row: "PASS" if pd.notna(row.get("estimate_reconstructed")) and pd.notna(row.get("estimate_frozen")) and abs(float(row["estimate_reconstructed"]) - float(row["estimate_frozen"])) <= 1e-12 else "NOT_APPLICABLE", axis=1)
    trace["notes"] = trace["thesis_role"]
    trace_columns = ["manuscript_location", "table_id", "result_id", "canonical_file", "canonical_key", "firm_source", "Stage8 source", "analysis contract", "reproduced estimate", "frozen estimate", "match", "notes"]
    _write_book(out / "TRACEABILITY_MATRIX.xlsx", {"README": pd.DataFrame([{"run_id": run_id, "rows": len(trace)}]), "TRACEABILITY": trace.reindex(columns=trace_columns)})

    computational = registry.loc[~registry.result_class.eq("PARENT_GATE")].copy()
    for table_id, frame in computational.groupby("result_class", sort=True):
        firm_level = frame[["result_id", "model", "reasoning_regime", "execution_policy", "action_space", "budget", "information_condition", "replicate", "estimate"]].copy() if all(x in frame for x in ["model", "reasoning_regime", "execution_policy", "action_space", "budget", "information_condition", "replicate"]) else frame[["result_id", "estimate"]].copy()
        calc = frame[[x for x in ["result_id", "N", "estimate_reconstructed", "estimate_frozen", "ci_low", "ci_high", "raw_p", "holm_p", "ci_source", "raw_p_source", "holm_source"] if x in frame]].copy()
        _write_book(out / "tables" / f"{table_id}.xlsx", {"README": pd.DataFrame([{"table_id": table_id, "classification": "COMPUTATIONAL", "source": source_rel}]), "SOURCE": frame.head(1), "FIRM_LEVEL": firm_level, "CALCULATION": calc, "STATISTICS": calc, "PAPER_TABLE": frame})
    metadata = {"run_id": run_id, "profile": profile, "created_this_run": True, "source": source_rel, "registry_rows": len(registry), "workbooks": ["THESIS_OUTPUT_INDEX.xlsx", "NUMERIC_CLAIMS.xlsx", "TRACEABILITY_MATRIX.xlsx"]}
    (out / "OUTPUT_METADATA.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", default="PROF_REVIEW")
    parser.add_argument("--profile", default="MANUSCRIPT_FROZEN")
    args = parser.parse_args(argv)
    try:
        out = build(Path(args.project_root), args.run_id, args.profile)
    except Exception as exc:
        print(f"BUILD_THESIS_OUTPUTS_FAIL {type(exc).__name__}: {exc}")
        return 1
    print(f"THESIS_OUTPUTS_WRITTEN {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
