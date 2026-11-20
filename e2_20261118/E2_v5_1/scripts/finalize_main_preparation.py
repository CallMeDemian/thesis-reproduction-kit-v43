from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[3]
E2 = WORKSPACE / "e2_20261118/E2_v5_1"
MANIFEST_DIR = E2 / "manifests_main"
REQUEST_DIR = E2 / "requests_main"
LOG_DIR = E2 / "logs_main"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest = pd.read_parquet(MANIFEST_DIR / "e2_main_execution_manifest.parquet")
    reuse = pd.read_csv(MANIFEST_DIR / "existing_NR_reuse_audit.csv")
    files = pd.read_csv(MANIFEST_DIR / "new_request_files.csv")
    new_counts = manifest.query("generation_source=='new_request'").groupby(["provider", "condition"]).size()
    pilot_counts = manifest.query("generation_source=='pilot_reuse'").groupby(["provider", "condition"]).size()
    reuse_counts = manifest.query("generation_source=='existing_main_reuse'").groupby(["source_main_condition", "provider", "replicate"]).size()
    expected_new = {(p, c): n for p in ["openai", "google"] for c, n in [("T", 1050), ("RT", 1050), ("Q", 475)]}
    if {k: int(v) for k, v in new_counts.to_dict().items()} != expected_new:
        raise ValueError("The frozen request count matrix differs from the authorized plan")
    if {k: int(v) for k, v in pilot_counts.to_dict().items()} != {
        (p, c): 100 for p in ["openai", "google"] for c in ["T", "RT", "Q"]
    }:
        raise ValueError("Pilot reuse count matrix differs from 100 firms × 2 models × T/RT/Q")
    if len(manifest) != 10350 or manifest.request_id.nunique() != 10350:
        raise ValueError("Unified execution manifest is not exactly 10,350 unique rows")
    if len(reuse) != 4600 or len(files) != 24:
        raise ValueError("Existing reuse audit or request files are incomplete")
    if not (reuse.ledger_request_state == "DELIVERED_LOCKED").all():
        raise ValueError("An N/R selected output is not locked in the generation ledger")
    if not (reuse.raw_sha256 == reuse.selected_raw_sha256).all():
        raise ValueError("An N/R selected raw hash does not match its response hash")

    prompt_diff = """# N/R 및 신규 T/RT 프롬프트 차이

기존 N/R은 원 C4R/C6-E TASK 및 실행 시점의 프롬프트·응답을 그대로 재사용했다. 두 TASK lead-in은 서로 다르다. T/RT는 v5.1에서 고정한 공통 TASK 문구를 사용하고 표 이외의 새 행동 결정 지시를 추가하지 않았다. 따라서 전체 575개 표본의 T−N, RT−R 및 상호작용은 결과표만의 순수 효과가 아니라 공통화 TASK와 생성 시점 차이가 함께 들어간 패키지 비교로 해석한다. 파일럿 100개 기업 N/R/T/RT는 같은 시기의 같은 공통 TASK로 생성되어 동시기 보조 2×2 분석에 사용한다.

- 기존 C4R/C6-E 프롬프트는 각 회차의 동일 C4 초안 원문을 부모 ID와 응답 해시로 연결했다.
- C6-E의 기업별 RL 참조 블록은 동결 E1-RL 참조 블록과 동일했다.
- 파일럿 T/RT/Q 600개 요청의 user/body hash는 재조립된 프롬프트와 일치했다.
- 모델·HIGH 추론·32,768 출력 상한·도구 미사용·행동 계약은 고정했다.
"""
    (LOG_DIR / "main_prompt_difference.md").write_text(prompt_diff, encoding="utf-8")

    summary = {
        "status": "REQUESTS_PREPARED_NO_ORACLE_SCORES_READ",
        "manifest_rows_unique": int(len(manifest)),
        "existing_NR_reuse_rows": int(len(reuse)),
        "existing_NR_locked_rows": int(reuse.ledger_request_state.eq("DELIVERED_LOCKED").sum()),
        "existing_NR_parquet_state": {f"{c}|{s}": int(n) for (c, s), n in reuse.groupby(["condition", "parquet_request_state"]).size().items()},
        "new_calls_by_provider_condition": {f"{p}|{c}": int(n) for (p, c), n in new_counts.items()},
        "pilot_reuse_by_provider_condition": {f"{p}|{c}": int(n) for (p, c), n in pilot_counts.items()},
        "pilot_responses_reused": int(manifest.generation_source.eq("pilot_reuse").sum()),
        "new_request_count": int(manifest.generation_source.eq("new_request").sum()),
        "strict_invalid_C4_drafts": 47,
        "C6E_reference_block_matches_E1_RL": True,
        "normalized_C4R_C6E_body_matches_outside_task_reference": True,
        "new_request_file_count": int(len(files)),
        "cost_point_usd": 69.84,
        "cost_approx_one_sided_95pct_upper_usd": 70.49,
        "simulated_output_cap_scenario_usd": 260.89,
        "pre_call_score_status": "No Oracle or policy-value scores read/calculated",
    }
    (LOG_DIR / "main_preparation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    frozen_paths = [
        E2 / "E2_V5_1_PROTOCOL.md", E2 / "E2_MAIN_REUSE_PROTOCOL.md",
        E2 / "request_manifest.csv", E2 / "sample_100_firms.csv",
        E2 / "results/pilot_cost_actual_tokens.csv",
        E2 / "results/main_cost_estimate.csv", E2 / "results/main_cost_estimate.json",
        E2 / "responses/pilot_openai_attempt1_output.jsonl", E2 / "responses/pilot_google_attempt1_output.jsonl",
        E2 / "scripts/prepare_main.py", E2 / "scripts/finalize_main_preparation.py", E2 / "scripts/estimate_main_cost.py",
        E2 / "scripts/run_main_batches.py",
        MANIFEST_DIR / "e2_main_execution_manifest.csv", MANIFEST_DIR / "e2_main_execution_manifest.parquet",
        MANIFEST_DIR / "existing_NR_reuse_audit.csv", MANIFEST_DIR / "t_rt_parent_table_links.csv",
        MANIFEST_DIR / "new_request_files.csv", LOG_DIR / "main_source_hashes.csv", LOG_DIR / "main_prompt_difference.md",
        WORKSPACE / "e2_20261118/B1_diagnostics/e2_simulator_10row_tables.parquet",
    ] + [REQUEST_DIR / name for name in sorted(p.name for p in REQUEST_DIR.glob("*.jsonl"))]
    rows = [{"path": str(p.relative_to(WORKSPACE)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in frozen_paths]
    pd.DataFrame(rows).to_csv(LOG_DIR / "pre_call_hashes_main.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
