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
from thesis_repro.execution_context import ExecutionContext, scoped_oracle_compatibility


class TinyGammaModel:
    def predict(self, frame):
        return [4.0] * len(frame)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, run_id: str, artifact_root: Path | None = None, run_stage1: bool = False, prepare_only: bool = False) -> dict:
    artifact_root = (artifact_root or (root / "runs")).resolve()
    run_root = artifact_root / run_id
    oracle_root = run_root / "02_oracle"
    oracle_root.mkdir(parents=True, exist_ok=True)
    raw_all = run_root / "work/raw_all"
    raw_rating = run_root / "work/rating_sample"
    raw_all.mkdir(parents=True, exist_ok=True)
    raw_rating.mkdir(parents=True, exist_ok=True)
    # 213 x 22 = 4,686 firm-years, just above the production eligibility
    # contract while keeping this deterministic E2E fixture tractable.
    fixture_firms = range(1, 214)
    # Include one pre-development year so real lagged growth ratios are
    # defined for the first 2002 development row.
    years = [year for year in range(2001, 2024) for _firm in fixture_firms]
    ids = [firm for _year in range(2001, 2024) for firm in fixture_firms]
    signal = [firm * 10.0 + (year - 2002) for year, firm in zip(years, ids)]
    # Keep all ten grades represented in both Dev and OOT while retaining
    # deterministic year-to-year movers.  The previous divisor (4) saturated
    # most OOT firms at AAA, making the real pre-registered correlation gates
    # fail for a fixture-distribution reason rather than an execution issue.
    grade_num = [max(1, min(10, 10 - ((year - 2002 + firm) // 20))) for year, firm in zip(years, ids)]
    quality = [11 - value for value in grade_num]
    base = {"거래소코드": ids, "회계년도": years, "회사명": [f"Fixture {i}" for i in ids], "시장": ["KOSPI"] * len(ids)}
    statement_codes = {
        "재무상태표": ["U01A100000000", "U01A110000000", "U01A111038600", "U01A111038700", "U01A111045400", "U01A111050000", "U01A600000000", "U01A611000000", "U01A615000000", "U01A800000000", "U01A810000000", "U01A811000000", "U01A811012800", "U01A811026000", "U01A811026700", "U01A811030700"],
        "손익계산서": ["U01B100000000", "U01B200000000", "U01B201014400", "U01B350014100", "U01B350014300", "U01B430000000", "U01B550000000", "U01B700000000", "U01B800000000", "U01B840000000", "U01B900000000"],
        "현금흐름표": ["U01D100000000", "U01D200000000", "U01D206012400", "U01D300000000"],
        "자본변동표": ["U01A600000000"], "이익잉여금처분계산서": ["U01A615000000"],
    }
    for index, name in enumerate(statement_codes):
        indices = list(range(len(ids)))
        if name == "자본변동표":
            indices = [j for j in indices if not ((ids[j] * 3 + years[j]) % 43 == 0)]
        rows = {k: [v[j] for j in indices] for k, v in base.items()}
        for code_index, code in enumerate(statement_codes[name]):
            # Keep the fixture deterministic but avoid making every candidate
            # an exact copy of one latent linear signal.  Production Stage00_04
            # must exercise its real collinearity gate, not spend unbounded time
            # replacing perfectly duplicated synthetic variables.
            values = [
                1000.0 + 7.0 * signal[j] + 11.0 * code_index + 3.0 * index
                + 7.0 * ((ids[j] * (code_index + 3) + years[j] * (index + 5)) % 29 - 14)
                for j in indices
            ]
            # Give the six immutable Alpha financial variables a deterministic
            # but distinct credit signal.  This makes the real selector recover
            # the promoted V4.3 variable contract instead of relying on a toy
            # post-selection overwrite.
            if code == "U01B100000000":  # revenue
                values = [10000.0 + 20.0 * signal[j] for j in indices]
            elif code == "U01A100000000":  # total assets
                values = [10000.0 + 20.0 * signal[j] * (1.35 + 0.015 * quality[j]) * (1.0 + 0.25 * ((ids[j] * 7 + years[j] * 3) % 17) / 16.0) for j in indices]
            elif code == "U01A110000000":  # non-current assets; drives R182
                values = [10000.0 + 20.0 * signal[j] * (0.70 + 0.50 * quality[j] + 0.002 * (years[j] - 2001)) for j in indices]
            elif code == "U01A615000000":  # retained earnings; drives R064
                values = [10000.0 + 20.0 * signal[j] * (0.30 - 0.018 * quality[j]) for j in indices]
            elif code == "U01A611000000":  # capital stock; drives R157
                values = [10000.0 + 20.0 * signal[j] / (0.55 + 0.06 * quality[j]) for j in indices]
            elif code == "U01A111038600":  # current assets
                values = [10000.0 + 20.0 * signal[j] * (0.62 + 0.025 * quality[j]) for j in indices]
            elif code == "U01A111038700":  # inventory
                values = [10000.0 + 20.0 * signal[j] * (0.08 - 0.003 * quality[j]) for j in indices]
            elif code == "U01A811026000":  # current liabilities
                values = [10000.0 + 20.0 * signal[j] * (0.34 - 0.012 * quality[j]) for j in indices]
            elif code == "U01B700000000":  # pretax income; drives R006
                values = [10000.0 + 20.0 * signal[j] * (0.02 + 0.012 * quality[j]) for j in indices]
            elif code == "U01B550000000":  # financial cost; drives R085
                values = [10000.0 + 20.0 * signal[j] * (0.28 - 0.014 * quality[j]) for j in indices]
            elif code == "U01B430000000":  # operating income / EBITDA proxy
                values = [10000.0 + 20.0 * signal[j] * (0.04 + 0.01 * quality[j] + 0.002 * (years[j] - 2002)) for j in indices]
            if code in {"U01B430000000", "U01B840000000"}:
                values = [-abs(value) if ids[j] == 10 and years[j] % 4 == 0 else value for j, value in zip(indices, values)]
            rows[code] = values
        pd.DataFrame(rows).to_excel(raw_all / f"{name}.xlsx", index=False)
    ratio_rows = {**base, "U01R00000001": [1.1 + 0.01 * value for value in signal]}
    for code_index, code in enumerate(sorted({code for codes in statement_codes.values() for code in codes})):
        ratio_rows[code] = [
            1000.0 + 7.0 * value + 13.0 * code_index
            + 7.0 * ((firm * (code_index + 5) + year * 3) % 31 - 15)
            for value, firm, year in zip(signal, ids, years)
        ]
        if code == "U01B100000000":
            ratio_rows[code] = [10000.0 + 20.0 * signal[j] for j in range(len(ids))]
        elif code == "U01A100000000":
            ratio_rows[code] = [10000.0 + 20.0 * signal[j] * (1.35 + 0.015 * quality[j]) * (1.0 + 0.25 * ((ids[j] * 7 + years[j] * 3) % 17) / 16.0) for j in range(len(ids))]
        elif code == "U01A110000000":
            ratio_rows[code] = [10000.0 + 20.0 * signal[j] * (0.70 + 0.50 * quality[j] + 0.002 * (years[j] - 2001)) for j in range(len(ids))]
        elif code == "U01A615000000":
            ratio_rows[code] = [10000.0 + 20.0 * signal[j] * (0.30 - 0.018 * quality[j]) for j in range(len(ids))]
        elif code == "U01A611000000":
            ratio_rows[code] = [10000.0 + 20.0 * signal[j] / (0.55 + 0.06 * quality[j]) for j in range(len(ids))]
        elif code == "U01A111038600":
            ratio_rows[code] = [10000.0 + 20.0 * signal[j] * (0.62 + 0.025 * quality[j]) for j in range(len(ids))]
        elif code == "U01A111038700":
            ratio_rows[code] = [10000.0 + 20.0 * signal[j] * (0.08 - 0.003 * quality[j]) for j in range(len(ids))]
        elif code == "U01A811026000":
            ratio_rows[code] = [10000.0 + 20.0 * signal[j] * (0.34 - 0.012 * quality[j]) for j in range(len(ids))]
        elif code == "U01B700000000":
            ratio_rows[code] = [10000.0 + 20.0 * signal[j] * (0.32 - 0.012 * quality[j]) for j in range(len(ids))]
        elif code == "U01B550000000":
            ratio_rows[code] = [10000.0 + 20.0 * signal[j] * (0.28 - 0.014 * quality[j]) for j in range(len(ids))]
        elif code == "U01B430000000":
            ratio_rows[code] = [10000.0 + 20.0 * signal[j] * (0.04 + 0.01 * quality[j] + 0.002 * (years[j] - 2002)) for j in range(len(ids))]
    pd.DataFrame(ratio_rows).to_excel(raw_all / "코스피_재무비율.xlsx", index=False)
    grade_labels = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
    grades = [grade_labels[n - 1] for n in grade_num]
    pd.DataFrame({**base, "신용등급": grades, "증권구분": [40] * len(ids), "평가사구분": [10] * len(ids), "평가사명 및 등급": [f"NICE {grade}" for grade in grades], "평가일": [f"{year}/06/30" for year in years]}).to_excel(raw_rating / "rating.xlsx", index=False)
    raw_root = run_root / "work/raw"
    shutil.copytree(raw_all, raw_root / "raw_all", dirs_exist_ok=True)
    shutil.copytree(raw_rating, raw_root / "rating_sample", dirs_exist_ok=True)
    (raw_root / "raw_nonfinancial/kospi_kosdaq").mkdir(parents=True, exist_ok=True)
    (raw_root / "raw_nonfinancial/konex_optional").mkdir(parents=True, exist_ok=True)
    general = pd.DataFrame({**base, "산업명": ["정보통신서비스" if firm % 2 else "금속기계전자" for firm in ids], "종업원": [100 + int(value) + ((firm * year) % 23) for value, firm, year in zip(signal, ids, years)], "설립일": ["1990/01/01"] * len(ids), "상장일": ["1995/01/01"] * len(ids)})
    capital = pd.DataFrame({**base, "변동일": [f"{year}/06/30" if ((firm + year) % 10) < (2 + grade_num[index] // 3) else None for index, (year, firm) in enumerate(zip(years, ids))]})
    for market in ["코스피", "코스닥"]:
        general.to_excel(raw_root / f"raw_nonfinancial/kospi_kosdaq/{market}_전업종_폐지사 포함_일반사항.xlsx", index=False)
        capital.to_excel(raw_root / f"raw_nonfinancial/kospi_kosdaq/{market}_전업종_폐지사 포함_자본금 변동사항.xlsx", index=False)
    general.to_excel(raw_root / "raw_nonfinancial/konex_optional/코넥스_전업종_일반사항.xlsx", index=False)
    capital.to_excel(raw_root / "raw_nonfinancial/konex_optional/코넥스_전업종_자본금 변동사항.xlsx", index=False)
    if prepare_only:
        receipt = {"schema_version": "tiny_oracle_fixture_raw_inputs_v1", "run_id": run_id, "fixture_kind": "raw_inputs_only", "raw_root": str(raw_root), "raw_all": str(raw_all), "raw_rating": str(raw_rating), "raw_file_count": len(list(raw_root.rglob("*.xlsx")))}
        print(json.dumps(receipt, indent=2))
        return receipt
    config_root = run_root / "work/contracts"
    shutil.copytree(root / "contracts/oracle_components", config_root, dirs_exist_ok=True)
    shutil.copy2(
        root / "contracts/scientific/final_freeze/final_oracle_rl_contract.json",
        config_root / "final_oracle_rl_contract.json",
    )
    work_root = run_root / "work/oracle_work"
    stage0 = work_root / "stage0_oracle_foundation"
    stage0_meta = build_stage0_foundation(root, raw_all, raw_rating, stage0, clean=True)
    os.environ.update({"THESIS_REPRO_PROJECT_ROOT": str(root), "THESIS_REPRO_RUN_ID": run_id, "THESIS_REPRO_ORACLE_RAW_ROOT": str(raw_root), "THESIS_REPRO_ORACLE_WORK_ROOT": str(work_root), "THESIS_REPRO_ORACLE_CONFIG_ROOT": str(config_root)})
    stage1_status = "NOT_REQUESTED"
    stage1_input_mode = "not_requested"
    if run_stage1:
        from credit_recourse.oracle.stage1.run_stage1_oracle_development import main as stage1_main
        from credit_recourse.oracle.fresh_runtime import resolve_fresh_oracle_runtime
        from thesis_repro.stages.oracle import _verify_production_oracle
        # This fixture always runs Stage1 from the generated run-local raw
        # inputs.  Frozen Stage1 input trees are evidence-only and must never
        # become a silent parent of a fresh CI computation.
        stage1_input_mode = "fresh_synthetic_raw_stage1"
        # This is an explicit fixture acceptance profile. It runs the real
        # Stage0/Stage1 producers and verifiers, but does not claim that a
        # tiny synthetic panel reproduces the licensed-data Alpha bytes.
        fixture_context = ExecutionContext.from_profile("synthetic", run_id, "OracleClean")
        stage1_common = ["--project-root", str(root), "--raw-rating-dir", str(raw_rating)]
        # The fixture is an end-to-end acceptance, not a partial Stage1
        # prefix. A single production invocation is required so the final
        # ledger can truthfully assert final_result_allowed=true.
        windows = [[*stage1_common, "--clean"]]
        with scoped_oracle_compatibility(fixture_context):
            for window in windows:
                stage1_rc = stage1_main(window)
                if stage1_rc != 0:
                    raise RuntimeError(f"production Stage1 returned {stage1_rc} for window={window}")
        stage1_status = "E2E_PASS"

    backend_root = work_root / "stage1_oracle_backends"
    if run_stage1:
        ledger = work_root / "ledgers/stage1_oracle_backends_full_development.json"
        report = json.loads(ledger.read_text(encoding="utf-8"))
        if report.get("status") != "PASS" or report.get("final_result_allowed") is not True:
            raise RuntimeError("Stage1 ledger did not PASS with final_result_allowed=true")
        runtime = resolve_fresh_oracle_runtime(root)
        # The production verifiers retain a scoped compatibility binding for
        # the fixture profile.  Keep the verification call inside the same
        # explicit context as Stage1; otherwise the Alpha strict verifier
        # would correctly interpret the synthetic parameter hash as a
        # production canonical-hash mismatch.
        with scoped_oracle_compatibility(fixture_context):
            verification = _verify_production_oracle(root, runtime, write_validation_report=True, context=fixture_context)
        if verification.get("status") != "PASS" or verification.get("final_result_allowed") is not True:
            raise RuntimeError("production Oracle verification did not PASS")
        if verification.get("fixture_verification_contract", {}).get("contract") != "SYNTHETIC_E2E_ACCEPTANCE":
            raise RuntimeError("synthetic Oracle fixture did not record its explicit fixture verification contract")
        validation = {"status": "PASS", "mode": "production_stage0_stage1_e2e", "acceptance_profile": "SYNTHETIC_E2E_ACCEPTANCE", "stage1_input_mode": stage1_input_mode, "stage1_rc": 0, "backend_root": str(backend_root), "stage1_report": {"status": report.get("status"), "final_result_allowed": report.get("final_result_allowed"), "execution_profile": report.get("execution_profile")}, "verification": verification}
        validation_path = oracle_root / "oracle_fixture_validation.json"; validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        artifacts = [stage0 / "canonical_panel/stage0_canonical_panel.parquet", stage0 / "stage0_validation.json", ledger, validation_path]
        output = pd.read_parquet(backend_root / "alpha/oracle_firm_year_output_alpha.parquet")
    else:
        # Scorer unit fixture is deliberately isolated from production backend directories.
        unit_root = oracle_root / "scorer_unit_fixture"
        unit_root.mkdir(parents=True, exist_ok=True)
        alpha = {"selected_variables": ["x"], "fin_ids": ["x"], "nonfin_ids": [], "directions": {"x": "higher_good"}, "bin_edges": {"x": {"edges": [0.0, 1.0, 2.0]}}, "bin_score_table_isotonic": {"x": {"0": 10.0, "1": 90.0}}, "item_weights": {"x": 1.0}, "block_norm": {"financial": {"p01": 0.0, "p99": 100.0}, "nonfinancial": {"p01": 0.0, "p99": 100.0}}, "boundaries": {"AAA": 90, "AA": 80, "A": 70, "BBB": 60, "BB": 50, "B": 40, "CCC": 30, "CC": 20, "C": 10}, "imputation_map": {"x": 50.0}, "pd_map": {}, "combined_weights": {"financial": 1.0, "nonfinancial": 0.0}}
        beta = {"model_name": "ordered-logit-beta", "selected_variables": ["x"], "standardization_params": {"x": {"mean": 0, "std": 1}}, "coefficients": [{"variable": "x", "coefficient": 1.0}], "modeled_grade_nums": list(range(1, 11)), "finite_cutpoints": list(range(1, 10))}
        gamma = {"selected_variables": ["x"]}
        alpha_path = unit_root / "alpha_params.json"; beta_path = unit_root / "beta_params.json"; gamma_path = unit_root / "gamma_params.json"; model_path = unit_root / "gamma_model.joblib"
        alpha_path.write_text(json.dumps(alpha, indent=2), encoding="utf-8"); beta_path.write_text(json.dumps(beta, indent=2), encoding="utf-8"); gamma_path.write_text(json.dumps(gamma, indent=2), encoding="utf-8"); joblib.dump(TinyGammaModel(), model_path)
        frame = pd.DataFrame({"firm_id": ["000001", "000001"], "fiscal_year": [2020, 2020], "x": [0.25, 1.75]})
        output = pd.DataFrame({"firm_id": frame.firm_id, "fiscal_year": frame.fiscal_year, "R_score_alpha": score_alpha(frame, alpha_path), "R_score_beta": score_beta_ordered_logit_params(frame, beta_path), "R_score_gamma": score_gamma_model(frame, gamma_path, model_path)})
        output_path = oracle_root / "oracle_scored_outputs.parquet"; output.to_parquet(output_path, index=False)
        validation = {"status": "PASS" if output.select_dtypes(include=["number"]).notna().all().all() and output[["R_score_alpha", "R_score_beta", "R_score_gamma"]].nunique().sum() > 0 else "FAIL", "rows": len(output), "production_backend_dir_untouched": not any(backend_root.rglob("*.json")), "production_scorers": ["score_alpha", "score_beta_ordered_logit_params", "score_gamma_model"]}
        validation_path = oracle_root / "oracle_fixture_validation.json"; validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        artifacts = [stage0 / "canonical_panel/stage0_canonical_panel.parquet", alpha_path, beta_path, gamma_path, model_path, output_path, validation_path]
    receipt = {"schema_version": "tiny_oracle_fixture_receipt_v2", "run_id": run_id, "fixture_kind": "production_stage0_stage1_e2e" if run_stage1 else "scorer_unit_fixture", "stage1_status": stage1_status, "stage0_rows": stage0_meta["row_counts"], "scored_rows": len(output), "validation": validation, "artifacts": [{"path": str(p.relative_to(artifact_root)).replace("\\", "/"), "sha256": _sha256(p), "bytes": p.stat().st_size} for p in artifacts]}
    (oracle_root / "tiny_oracle_fixture_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--run-stage1", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-id", default="oracle-fixture")
    args = parser.parse_args()
    run(args.root.resolve(), args.run_id, args.artifact_root, args.run_stage1, args.prepare_only)
