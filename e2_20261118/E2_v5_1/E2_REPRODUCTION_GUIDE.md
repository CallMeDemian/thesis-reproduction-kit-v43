# E2 분석 재실행 안내

## 입력 및 고정 기준

- 승인된 실행 프로토콜: `E2_MAIN_REUSE_PROTOCOL.md`
- 원본 v5.1 프로토콜과 설계서: `E2_V5_1_PROTOCOL.md`, `source_inputs/E2_최종_실험설계서_v5.1_20261118.md`
- 10,350행 분석 매니페스트: `manifests_main/e2_main_execution_manifest.csv`
- 신규 5,150건 원요청: `requests_main/*.jsonl`
- 기존·파일럿·신규 원응답: `responses/*.jsonl`, `responses_main/*.jsonl`
- 응답 무결성과 raw hash: `logs_main/main_response_integrity.json`, `logs_main/main_response_hashes.json`
- 동결 Simulator·Oracle 입력과 E1 관측자료: `../A_reproduction/E1_score_gate_replay/runroot_zip/THESIS_REPRO_KIT_v2.1.1_FINAL/`, `../source_inputs/github_release_ablation_20261102_readonly/contents/ablation_20261102/E1/observations/e1_observations.parquet`

## 실행 순서

작업 디렉터리는 저장소 루트로 둔다. 분석용 Python 환경은 Python 3.11.15, NumPy 2.3.3, pandas 2.3.3으로 기록돼 있다. 기존 키트와 동결 자료의 상대 경로가 필요하다.

```powershell
$py = 'e2_20261118\A_reproduction\R2_R8_replay_workspace\.venv\Scripts\python.exe'
& $py 'e2_20261118\E2_v5_1\scripts\parse_main_responses.py'
& $py 'e2_20261118\E2_v5_1\scripts\score_main_and_analyze.py'
& $py 'e2_20261118\E2_v5_1\scripts\make_action_diagnostics.py'
```

파서는 α·β·γ 점수를 계산하지 않고 `RESPONSE_INTEGRITY_COMPLETE` 상태인지 기록한다. 점수 분석기는 이 상태와 신규 응답 ID 5,150개를 확인한 뒤에만 동작한다. 이미 생성된 결과를 다시 채점할 때는 분석기 출력 파일이 갱신된다.

그림 생성은 matplotlib이 설치된 Python 환경을 사용한다. 이번 실행 환경은 Python 3.12.10, matplotlib 3.11.0, NumPy 2.4.4였다.

```powershell
& 'C:\Users\Demian\AppData\Local\Programs\Python\Python312\python.exe' `
  'e2_20261118\E2_v5_1\scripts\make_final_figures.py'
```

`make_final_artifact_manifest.py`는 원요청·원응답, 프로토콜, 코드, 결과표, 그림, 주요 로그를 SHA-256과 함께 `E2_FINAL_ARTIFACT_MANIFEST.csv`에 기록한다. 원응답별 별도 해시는 `logs_main/main_response_hashes.json`에도 있다.

## 주요 분석 코드

- `scripts/parse_main_responses.py`: 재사용 원장 및 공급자 출력 연결, 응답 스키마·행동계약·모델 ID·사용량 무결성 점검
- `scripts/score_main_and_analyze.py`: α·β·γ 정책가치, H1·2×2·P0–P5, 100개 기업 재배정 구성·배정 가치, Q, S0/S1/S2 민감도, 회계 QA
- `scripts/make_action_diagnostics.py`: RL 참조 수용·수정, P3 행동 유사도, C3-E/OE 하위집단, A0보다 낮은 점수 비율
- `scripts/make_final_figures.py`: 최종 보고용 PNG 세 개
- `scripts/make_final_artifact_manifest.py`: 재현 자료 SHA-256 매니페스트
