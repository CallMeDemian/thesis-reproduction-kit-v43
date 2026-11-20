# 실험 원자료에서 보고서까지

## 기존 연구와 R1–R9

`v1.0.4 THESIS_REPRO_KIT_v2.1.1_FINAL.zip` + `thesis-v43-ablation-20261102 THESIS_V43_ABLATION_20261102.zip`
→ canonical STRICT ITT·C3-E 행동/확률·E1 관측·동결 Simulator/Oracle
→ 원 11/17 structural scripts 및 11/18 tech-review scripts
→ `R2_R8_replay_workspace/analysis_20261117/` 배열/모델 + `logs/replay/`
→ `E2_REPRODUCTION_REPORT.md`, R6 부록, 이번 통합보고서.

`EXPERIMENT_INVENTORY.csv`는 R 이름과 스크립트를 교차 기재한다. R과 e번호는 같은 계산을 가리킬 수 있으므로 목록 행을 독립 실험 수로 합산하지 않는다.

## 행동 수용 및 적응형 게이트

공개 pinned Git/LFS의 HIGH 원요청·원응답·원장·28 actor/critic 진단
→ `analysis/rl_reference_gate_runtime/stage7_full_materialized/` 행동표
→ 별도 분석 Stage8(all populations, hash 519ef1a1…)
→ `rl_reference_gate_analysis.py`, `rl_llm_interaction_gate.py`, `policy_advice_taking_financial_state.py`, `policy_advice_taking_mechanism_v2.py`
→ firm-level 행동/OOF/음성대조 원장 및 집계 CSV
→ 네 원 연구보고서와 통합보고서.

이 Stage8은 canonical ITT 캡슐과 다른 파일이다. 두 소스를 합치지 않고 FILE_CATALOG에 각 해시를 보존한다. 계약 역시 repository `v43_action_contract.json`과 stage2의 candidate contract를 경로별로 구분한다.

## P0–P5/B1

같은 동결 기업·행동·Simulator/7평가기
→ `B1_diagnostics/b1_run.py` + `b1_finalize.py`
→ policy surface, P5 고정 행동, cost-allocation counterfactual, 10행 표
→ paired 대비·구성/배정 CSV 및 `b1_artifact_sha256.json`
→ `E2_B1_RESULTS.md`.

## E2 행별 연결

최종 규칙 `E2_MAIN_REUSE_PROTOCOL.md` + v5.1 설계/사전 hash
→ `manifests_main/e2_main_execution_manifest.csv`의 `request_id, row_id, firm_key, model_key, replicate, condition, generation_source`
→ 부모 C4 ID/raw/payload hash + 기존 N/R의 `source_main_request_id/selected_attempt/raw/payload hash`
→ pilot 및 신규 `requests/*.jsonl`, `requests_main/*.jsonl` + body/user/system/table hash
→ `responses/*.jsonl`, `responses_main/*.jsonl`, 기존 `generation_ledger.sqlite`의 잠긴 선택 응답
→ `logs_main/main_response_integrity.json` + `main_response_hashes.json`
→ `parse_main_responses.py` → policy/Q response tables(무효 A0와 Q 구조불가 분리)
→ `score_action_manifest.csv`의 action hash 및 고정 100 derangement
→ `score_main_and_analyze.py` → 38,750 saved score rows, H1/H2–H6, P0–P5, Q, S0/S1/S2
→ `make_action_diagnostics.py`, 집계 CSV/그림
→ `E2_FINAL_REPORT.md` → 통합보고서.

N/R 원응답은 새로 복사한 선택 출력으로 바꾸지 않고 원 SQLite와 그 공개 LFS 객체를 해시로 연결했다. CSV/parquet의 중복 표현과 기존 manifest는 연구 산출물 정체성을 위해 보존한다. 같은 파일의 디렉터리 중복본은 catalog alias로 복원한다.

H1→main_h1_primary/by_model; H2/H3→main_factorial_contrasts; H4→main_rt_vs_p0_p5/main_p3_gap_recovery; H5→main_r8_decomposition_contrasts/main_action_distribution_summary; H6→main_q_forecast_accuracy/main_q_structure_summary; T9→main_sensitivity_contrasts/main_sensitivity_change_vs_s0. 이 파일들이 논문 표의 우선 수치 원천이다.
