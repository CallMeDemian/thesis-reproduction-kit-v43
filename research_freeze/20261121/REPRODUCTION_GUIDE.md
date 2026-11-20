# 재현·검증 안내 — 새 실험 없는 증거 동결

## 1. 읽기와 경량 검증

먼저 INTEGRATED_RESEARCH_REPORT_KO.md → EXPERIMENT_INVENTORY.csv → EVIDENCE_LINEAGE.md → LIMITATIONS_AND_GAPS.md 순서로 읽는다. Release의 RELEASE_MANIFEST.json은 최종 tag commit과 asset SHA-256을 기록하고, FILE_CATALOG.json은 필요한 각 경로의 원천을 기록한다. SHA256SUMS.txt는 자기 자신을 제외한 모든 첨부파일을 검증한다. FILE_CATALOG 안의 hash는 line ending을 정규화하지 않은 실제 바이트다.

GitHub 자동 Source ZIP만으로 원자료가 모두 들어 있다고 가정하지 않는다. 아래의 정식 첨부 ZIP을 사용한다.

- THESIS_V43_RESEARCH_CODE_REPORTS_v1.1.0.zip
- THESIS_V43_RESEARCH_DATA_RESULTS_v1.1.0.zip
- THESIS_V43_RESEARCH_RAW_REQUESTS_RESPONSES_v1.1.0.zip

검증 도구는 Python 3.11 이상 표준 라이브러리만 필요하다. release assets를 한 디렉터리에 내려받은 뒤 실행한다. 원 실험 코드는 실행하지 않는다.

```powershell
python restore_and_verify.py --assets-dir C:\freeze_assets --checksums-only
```

## 2. 의존성 복원

v1.0.4의 certified `THESIS_REPRO_KIT_v2.1.1_FINAL.zip`, ablation release의 `THESIS_V43_ABLATION_20261102.zip`을 같은 assets 디렉터리에 둔다. 버전/URL/전체 ZIP SHA-256/member는 catalog에 있다. Human baseline은 별도 source-list Release를 참조하며 post-study 계산의 필수 입력은 아니다.

catalog의 `existing_public_git`는 baseline commit `b4ed3bf5504d792a0a06f66c7c3712601ae857bf`에 고정돼 있다. 별도 clone을 그 SHA로 checkout하고 Git LFS 내용을 가져온다. 이 복원은 파일 다운로드이며 LLM 호출/학습을 하지 않는다. 이미 내려받은 clone이 있다면 재다운로드하지 않는다.

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/CallMeDemian/thesis-reproduction-kit-v43.git C:\freeze_baseline
git -C C:\freeze_baseline checkout b4ed3bf5504d792a0a06f66c7c3712601ae857bf
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
git -C C:\freeze_baseline lfs pull
python restore_and_verify.py --assets-dir C:\freeze_assets --git-root C:\freeze_baseline --output C:\freeze_restored
```

복원기는 기존 파일을 덮어쓰지 않고, 각 바이트를 검증한 뒤 누락 경로만 새 output에 만든다. alias도 복원한다. Git LFS pointer를 실제 데이터로 취급하지 않는다. 필요한 공개 LFS 객체·이전 ZIP이 없으면 구체적인 누락 목록을 출력한다. 다운로드 크기를 줄이려면 FILE_CATALOG의 `existing_public_git` 경로만 `git lfs pull --include`로 가져올 수 있다. 복원기는 네트워크·API·학습 기능이 없다.

## 3. 보존된 분석을 다시 실행할 때

전체 재실행은 이번 동결 검증의 범위가 아니다. 아래는 향후 승인된 분석 재실행을 위한 경로이며 원 동결 폴더 대신 별도 작업 복사본에서 실행해야 한다. 기존 스크립트는 결과 CSV를 덮어쓸 수 있다.

- R1/R7/R9/e29/e30: `source_inputs/tech_review_20261118/README.md`의 OBS/CON 인자를 사용한다. 입력 canonical ITT는 certified kit의 frozen_replay/v1.3/frozen_inputs/stage8에 있다.
- R2–R8: `analysis_20261117/README.md`; KIT_ROOT를 복원된 `e2_20261118/A_reproduction/E1_score_gate_replay/runroot_zip/THESIS_REPRO_KIT_v2.1.1_FINAL`로, ABL_E1을 이전 ablation ZIP의 E1 observations로 지정하고 해당 분석 폴더를 cwd로 사용한다. `HZ_SRC`는 KIT_ROOT/src다. 저장된 배열/δ·ε 모델을 우선 사용하며 재적합할 필요가 없다. R6 조건 차이는 별도 부록을 읽는다.
- 수용/게이트: `analysis/rl_reference_gate_runtime`의 보존 Stage8을 사용한다. `python analysis/rl_reference_gate_analysis.py --runtime-root analysis/rl_reference_gate_runtime --out <new-output>`; interaction은 `--rl-results`에 기존 gate result 폴더를 지정한다. 수용 v1/v2 스크립트는 경로가 고정돼 있어 전체 복사본에서 실행한다.
- E2는 `e2_20261118/E2_v5_1/E2_REPRODUCTION_GUIDE.md`와 최종 reuse protocol을 따른다. `parse_main_responses.py`는 기존 locked N/R ledger를 읽으며 공급자를 호출하지 않는다. `score_main_and_analyze.py`는 이미 저장된 행동을 Simulator로 재채점하므로 경량 무결성 검사에는 실행하지 않는다.
- 원 e26 등 일부 탐색 코드는 절대경로와 추가 history 입력을 요구한다. 코드만 보존된 항목의 실행 가능성을 새로 입증하지 않았고 누락 결과를 생성하지 않았다.

환경은 R2–R8/E2 분석 Python 3.11.15, NumPy/pandas 2.3.3, SciPy 1.16.1, scikit-learn 1.6.1, PyArrow 20.0.0, statsmodels 0.14.5; 별도 게이트 Stage8/Gamma는 Python 3.12/scikit-learn 1.8로 기록돼 있다. 정확한 원 requirements/metadata를 함께 보존한다. SDK preflight/submit/collect 스크립트는 과거 실험 정체성 보존용이며 재현을 위해 실행할 필요가 없다.

## 4. 검증의 범위

VALIDATION_REPORT.json은 기존 E2 manifest 165개, B1 hash 19개, 원응답 파일 26개, 10,350개 고유 실행 ID, 신규 요청/응답 5,150쌍, pilot 1,000개, 집계·분해 산술 및 e17 357 numeric 셀을 확인했다. 신규 paid API=0, RL 학습=0, Simulator 재실행=0, bootstrap 재실행=0이다. RELEASE_MANIFEST와 FILE_CATALOG의 전체 archive hash 검증 및 복원 결과는 별도 출판 검증 기록에 남는다.

추가 원장 확인: LINEAGE_VALIDATION.json은 과거 N/R 재사용 4,600개와 부모 C4 2,300개의 총 6,900개 locked 기록에서 선택 시도·payload hash·원문 hash를 SQLite read-only/immutable 모드로 직접 대조한 결과다. PACKAGE_VALIDATION.json은 새 ZIP member 459개의 원 바이트와 catalog를 대조한다.
