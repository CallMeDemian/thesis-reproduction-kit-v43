from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import joblib
import pandas as pd

from credit_recourse.eval.v43_oracle_backends import score_alpha, score_beta_ordered_logit_params, score_gamma_model
from credit_recourse.oracle.stage0.build_stage0_foundation_from_raw import build_stage0_foundation


class TinyGammaModel:
    def predict(self, frame):
        return [4.0] * len(frame)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, run_id: str, artifact_root: Path | None = None, run_stage1: bool = False) -> dict:
    artifact_root = (artifact_root or (root / "runs")).resolve()
    run_root = artifact_root / run_id
    oracle_root = run_root / "02_oracle"
    oracle_root.mkdir(parents=True, exist_ok=True)
    raw_all = run_root / "work/raw_all"
    raw_rating = run_root / "work/rating_sample"
    raw_all.mkdir(parents=True, exist_ok=True)
    raw_rating.mkdir(parents=True, exist_ok=True)
    ids = [1, 1, 2]
    base = {"거래소코드": ids, "회계년도": [2019, 2020, 2020], "회사명": ["Fixture A", "Fixture A", "Fixture B"], "시장": ["KOSPI", "KOSPI", "KOSPI"]}
    statement_names = ["재무상태표", "손익계산서", "현금흐름표", "자본변동표", "이익잉여금처분계산서"]
    for index, name in enumerate(statement_names):
        rows = dict(base)
        if name != "재무상태표":
            rows = {k: [v[0], v[1]] for k, v in base.items()}
        rows[f"U01A{index+1:010d}"] = [100.0 + index + i for i in range(len(rows["거래소코드"]))]
        pd.DataFrame(rows).to_excel(raw_all / f"{name}.xlsx", index=False)
    pd.DataFrame({**base, "U01R00000001": [1.1, 1.2, 2.2]}).to_excel(raw_all / "코스피_재무비율.xlsx", index=False)
    pd.DataFrame({"거래소코드": ids, "회계년도": [2019, 2020, 2020], "회사명": ["Fixture A", "Fixture A", "Fixture B"], "시장": ["KOSPI", "KOSPI", "KOSPI"], "신용등급": ["BBB", "BBB", "BB"], "증권구분": [40, 40, 40], "평가사구분": [10, 10, 10], "평가사명 및 등급": ["NICE BBB", "NICE BBB", "NICE BB"], "평가일": ["2019/06/30", "2020/06/30", "2020/06/30"]}).to_excel(raw_rating / "rating.xlsx", index=False)
    raw_root = run_root / "work/raw"
    shutil.copytree(raw_all, raw_root / "raw_all", dirs_exist_ok=True)
    shutil.copytree(raw_rating, raw_root / "rating_sample", dirs_exist_ok=True)
    (raw_root / "raw_nonfinancial/kospi_kosdaq").mkdir(parents=True, exist_ok=True)
    (raw_root / "raw_nonfinancial/konex_optional").mkdir(parents=True, exist_ok=True)
    shutil.copy(raw_all / "재무상태표.xlsx", raw_root / "raw_nonfinancial/kospi_kosdaq/코스피_전업종_폐지사 포함_일반사항.xlsx")
    shutil.copy(raw_all / "재무상태표.xlsx", raw_root / "raw_nonfinancial/kospi_kosdaq/코스피_전업종_폐지사 포함_자본금 변동사항.xlsx")
    config_root = run_root / "work/contracts"
    shutil.copytree(root / "contracts/oracle_components", config_root, dirs_exist_ok=True)
    work_root = run_root / "work/oracle_work"
    stage0 = work_root / "stage0_oracle_foundation"
    stage0_meta = build_stage0_foundation(root, raw_all, raw_rating, stage0, clean=True)
    os.environ.update({"THESIS_REPRO_ORACLE_RAW_ROOT": str(raw_root), "THESIS_REPRO_ORACLE_WORK_ROOT": str(work_root), "THESIS_REPRO_ORACLE_CONFIG_ROOT": str(config_root)})
    stage1_status = "NOT_REQUESTED"
    if run_stage1:
        from credit_recourse.oracle.stage1.run_stage1_oracle_development import main as stage1_main
        stage1_rc = stage1_main(["--project-root", str(root), "--raw-rating-dir", str(raw_rating), "--clean", "--end-step", "backend_gamma", "--pending-ok"])
        if stage1_rc not in (None, 0):
            raise RuntimeError(f"production Stage1 returned {stage1_rc}")
        stage1_status = "EXECUTED"

    backend_root = work_root / "stage1_oracle_backends"
    backend_root.mkdir(parents=True, exist_ok=True)
    alpha = {"selected_variables": ["x"], "fin_ids": ["x"], "nonfin_ids": [], "directions": {"x": "higher_good"}, "bin_edges": {"x": {"edges": [0.0, 1.0, 2.0]}}, "bin_score_table_isotonic": {"x": {"0": 10.0, "1": 90.0}}, "item_weights": {"x": 1.0}, "block_norm": {"financial": {"p01": 0.0, "p99": 100.0}, "nonfinancial": {"p01": 0.0, "p99": 100.0}}, "boundaries": {"AAA": 90, "AA": 80, "A": 70, "BBB": 60, "BB": 50, "B": 40, "CCC": 30, "CC": 20, "C": 10}, "imputation_map": {"x": 50.0}, "pd_map": {}, "combined_weights": {"financial": 1.0, "nonfinancial": 0.0}}
    beta = {"model_name": "ordered-logit-beta", "selected_variables": ["x"], "standardization_params": {"x": {"mean": 0, "std": 1}}, "coefficients": [{"variable": "x", "coefficient": 1.0}], "modeled_grade_nums": list(range(1, 11)), "finite_cutpoints": list(range(1, 10))}
    gamma = {"selected_variables": ["x"]}
    alpha_path = backend_root / "alpha/oracle_alpha_params.json"
    beta_path = backend_root / "beta/benchmark_beta_params.json"
    gamma_path = backend_root / "gamma/benchmark_gamma_params.json"
    model_path = backend_root / "gamma/benchmark_gamma_model.joblib"
    alpha_path.parent.mkdir(parents=True, exist_ok=True); beta_path.parent.mkdir(parents=True, exist_ok=True); gamma_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_path.write_text(json.dumps(alpha, indent=2), encoding="utf-8")
    beta_path.write_text(json.dumps(beta, indent=2), encoding="utf-8")
    gamma_path.write_text(json.dumps(gamma, indent=2), encoding="utf-8")
    joblib.dump(TinyGammaModel(), model_path)
    frame = pd.DataFrame({"firm_id": ["000001", "000001"], "fiscal_year": [2020, 2020], "x": [0.25, 1.75]})
    output = pd.DataFrame({"firm_id": frame.firm_id, "fiscal_year": frame.fiscal_year, "R_score_alpha": score_alpha(frame, alpha_path), "R_score_beta": score_beta_ordered_logit_params(frame, beta_path), "R_score_gamma": score_gamma_model(frame, gamma_path, model_path)})
    output_path = oracle_root / "oracle_scored_outputs.parquet"; output.to_parquet(output_path, index=False)
    validation = {"status": "PASS" if output.select_dtypes(include=["number"]).notna().all().all() and output[["R_score_alpha", "R_score_beta", "R_score_gamma"]].nunique().sum() > 0 else "FAIL", "rows": len(output), "backend_paths_are_distinct": True, "production_scorers": ["score_alpha", "score_beta_ordered_logit_params", "score_gamma_model"]}
    validation_path = oracle_root / "oracle_fixture_validation.json"; validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    artifacts = [stage0 / "canonical_panel/stage0_canonical_panel.parquet", alpha_path, beta_path, gamma_path, model_path, output_path, validation_path]
    receipt = {"schema_version": "tiny_oracle_fixture_receipt_v1", "run_id": run_id, "stage1_status": stage1_status, "stage0_rows": stage0_meta["row_counts"], "scored_rows": len(output), "validation": validation, "artifacts": [{"path": str(p.relative_to(artifact_root)).replace("\\", "/"), "sha256": _sha256(p), "bytes": p.stat().st_size} for p in artifacts]}
    (oracle_root / "tiny_oracle_fixture_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--run-stage1", action="store_true")
    parser.add_argument("--run-id", default="oracle-fixture")
    args = parser.parse_args()
    run(args.root.resolve(), args.run_id, args.artifact_root, args.run_stage1)
