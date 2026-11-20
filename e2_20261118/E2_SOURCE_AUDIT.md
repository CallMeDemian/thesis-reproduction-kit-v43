# E2_SOURCE_AUDIT

작성일: 2026-11-18  
작업 폴더: C:/Users/Demian/Desktop/REPRO_KIT_semantic_repair/e2_20261118  
판정 기준: GitHub에 공개된 지정 재현 키트와 release를 우선 원천으로 삼았다. 문서에 기록된 수치나 9월 실행자료는 입력 계약·기업 집합·프로토콜이 맞는지 확인되지 않으면 대체 자료로 쓰지 않았다.

## 핵심 결론

- [확정] 현재 source of truth로 삼을 자료는 GitHub 저장소 CallMeDemian/thesis-reproduction-kit-v43, main commit b4ed3bf5504d792a0a06f66c7c3712601ae857bf이다.
- [확정] 동결 관측자료와 행동 계약을 읽었고, GitHub 기본 결과 재현 명령은 914개 결과 등록표와 96,600개 동결 관측값을 PASS로 재계산했다. 이 재현은 저장된 증거를 다시 분석하는 범위이며 원래 Stage8/Stage9 생성 코드를 그대로 다시 실행한 것은 아니다.
- [확정] 11월 18일 e27–e34 중 제공된 e27–e31·e33·e34를 GitHub 동결 입력으로 다시 실행했다. e27, e28, e33, e34의 핵심 출력은 묶음 결과와 일치했다.
- [최신 판정] 11월 17일 e15–e26 분석 ZIP을 Downloads에서 확보했다. R2–R5·R8을 재실행했고 R6는 일부 불일치다. 아래 “추가 자료 복구 보완” 절이 초기 탐색 상태를 대체한다.
- [확정] 2024 기준 재무상태·맥락, 8축 행동 계약, Simulator·Oracle 핵심 파일을 GitHub main의 LFS 해시와 대조했다. 공식 E1 요청 shard에는 실제 요청 본문과 C4 초안이 포함되어 있고 manifest의 payload/body hash로 추적된다.
- [미확정] E2 전용 요청 본문·10행 결과표는 아직 생성·동결하지 않았고, C2와 Stage4 행동모방 정책의 기업별 crosswalk도 없다. 이 복구 단계에서는 B1 결과표·회계 QA를 실행하지 않았다.

## 확인한 입력과 해시

| 자료 | 위치 또는 식별자 | SHA-256 / 식별값 | 판정 |
|---|---|---|---|
| 사용자 제공 v3 설계서 | Downloads/ClaudeCode_연구설계서_E2_v3_20261118.md | C7E6730F3A21350EA25F205A6EA57D67106C92A2D97EF03AB1DF61C0D720BCFA | 보존하고 v4는 별도 작성 |
| 사용자 제공 기술개선 ZIP | Downloads/기술개선_사전점검_스크립트_20261118_v2.zip | E2FF786B095515275A3E934E84A2D11AA856314959350D63CADBA0DBFEAD6D6E | 압축 내부 README와 실제 코드를 확인 |
| GitHub 저장소 | CallMeDemian/thesis-reproduction-kit-v43, main | b4ed3bf5504d792a0a06f66c7c3712601ae857bf | 기준 코드 |
| canonical ITT observations | frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN/frozen_inputs/stage8/canonical_itt_observations.parquet | AEDFADBF1728A14D3E4CB45BFA8A022D2C10313C52606E7BD1AEE68899DE8A0D | 96,600행, 575개 기업 |
| 행동 계약 | contracts/scientific/v43_action_contract.json | 5FD8FF5825FBEE64E781FAC8CE7302D36AA9C6CA7FE56F21EC8BEA7A8FC101CC | 8개 축의 단위·부호·범위·표준행동 |
| E1 프로토콜 | `source_inputs/github_release_ablation_20261102_readonly/contents/ablation_20261102/E1/E1_PROTOCOL.md` | SHA-256 `90d645a0c00eb52659d118b3c555da94d71d84d8f00e8a9454e5c0e3f85748c5`; GitHub blob `46782dc66c91ec6c6945eda80ae74dda6ef667d9` | release manifest와 실제 파일 대조 |
| 공식 E1 release ZIP | `source_inputs/github_release_ablation_20261102_readonly/THESIS_V43_ABLATION_20261102.zip` | `3f217ae100103ff83030f1dd1a31bd90821271eb2a58d0933958db65af4e3207` | 공식 asset; SHA256SUMS와 manifest 일치 |
| GitHub 기본 재현 영수증 | A_reproduction/github_published_check_receipt.json | 7A3504F74FBC8C4728CEF7EC8D99FFFFC374A5ACB60AC7720FCD455FDDF80F7C | API 호출 0, 학습 0, 재생 PASS |

## 경로와 자료 검색 결과

| 후보 | 실제 확인 | 처리 |
|---|---|---|
| C:/Users/Demian/Desktop/abl1102/thesis-reproduction-kit-v43 | 없음 | GitHub main checkout을 source of truth로 사용 |
| C:/Users/Demian/Desktop/thesis_repo | 없음 | 논문 저장소 자료는 미확보로 기록 |
| Downloads/구조적재검토_분석스크립트_20261117.zip | 93,251 B; SHA-256 `501a6836a3059a4055cd4a50c7e9a4772674cbef095e5efb32fbf59f305f2df2` | README·e15–e26·delta_train.py·fam2.py 확인, 원본 보존 후 격리 run copy에서 실행 |
| GitHub 기본 저장소 | 있음 | main b4ed3bf를 새 작업 폴더 안에 checkout |
| Git LFS 원 실험 자산 | main checkout의 일부 경로는 LFS 포인터 상태였음. 필요한 2024 기준상태·맥락·행동 계약 및 핵심 Simulator/Oracle 파일 복사본을 workspace archive에서 찾아 canonical LFS SHA-256과 대조함 | 대조한 파일만 원천 입력으로 채택. 전체 original_release가 완전히 복구됐다고 확대하지 않음 |
| Downloads의 v3와 11/18 ZIP | 있음 | 원본은 바꾸지 않고 작업 폴더에 복사·압축 해제 |

GitHub 기본 재현 실행은 Python 3.12.10, NumPy 2.4.4, pandas 3.0.2, PyArrow 24.0.0에서 수행했다. 11/17 분석 ZIP 재실행은 README의 잠금 조합을 반영한 별도 Python 3.11.15 가상환경에서 했다(세부 버전은 아래). 결과 영수증에 기본 재현의 전체 registry 대조가 기록되어 있다.

## 데이터 계약과 실제 내용

canonical 관측표는 96,600행 × 44열이다. 기업·모델·조건·반복, 제안 행동 JSON, 엄격 사용 가능 여부, α·β·γ 점수와 무행동 대비 점수 차이를 포함한다. 표에는 각 8축 행동에 대한 1년 뒤 손익·대차대조표 금액 변화 전체가 들어 있지 않다.

행동 계약은 8개 축을 다음 순서로 명시한다: 성장성 투자 축소, 총부채 축소, 단기부채 리파이낸싱, 재고회전율, 매출채권회전율, 매입채무회전율, 매출원가율, 판관비율. 각 축의 척도는 계약의 normalization scale로 정의된다. 이번 복구에서 2024 기준 기업상태·맥락, 행동 계약, Simulator 핵심 구현 및 Oracle 계약을 GitHub LFS 원천 해시와 연결했고, 키트의 단위는 천원(KRW thousand), 상태 시간축은 결정 연도 다음 해로 확인했다. 다만 E2용 10행 표는 생성하지 않아 행별 부호·반올림·회계 항등식·행동예산·복합행동 제약 QA는 완료되지 않았다.

main 저장소의 README는 frozen evidence replay와 역사적 생성자산 재구성을 구분한다. 재현 영수증도 exact historical Stage8/Stage9 producer bytes are unrecovered라고 명시한다. 따라서 기본 재현 PASS는 저장된 기업별 증거에서 등록 결과를 재계산했다는 의미다.

## E1 자료와 해석 경계

GitHub의 2026-11-02 ablation release에서 E1_PROTOCOL.md와 README를 확인했다.

- E1은 본 연구의 주 실험이 끝난 뒤 설계된 탐색적 추가 분석이다. 프로토콜은 첫 API 호출 전에 고정되었다고 기록되어 있지만, 독립적인 사전등록 표본 실험은 아니다.
- 2024 회계연도 기업 575개, 기존 모델 2종, C4R·C6-E·OE 조건을 사용했다. RL 행동과 고정 OE 행동이 다른 343개 기업에서 RL 참조와 OE 참조를 비교하는 것이 핵심 대비다.
- 같은 575개 기업과 기존 평가 자료를 사용하므로 새로운 표본외 검증으로 부르지 않는다.
- E1은 α로 C3-E 정책을 선택했으므로, 같은 평가 계열의 α 결과는 완전히 독립적인 확인으로 해석하지 않는다.
- E1 release의 8개 실제 provider request shard, 요청 manifest, 요청 생성 코드가 확보됐다. 각 요청 본문에는 C4 초안과 조건별 reference payload가 있고 manifest가 payload/body hash 및 요청 간 parent 관계를 연결한다. 별도 C4 초안 파일은 없지만 요청 shard 안에서 추적 가능하다. E1 분석표는 release 원자료로 다시 계산했으며 세부 재현 상태는 `E2_GATE1_RECOVERY_REPORT.md`에 기록했다.

## e27–e34 ZIP 실제 확인

ZIP README가 지시한 입력은 위 GitHub main의 canonical ITT observations와 행동 계약이다. e27, e28, e29, e30, e31, e33, e34 스크립트를 실제로 읽고 실행했다. e32는 README에 이미 폐기된 분석으로 기록되어 있었고 ZIP에 포함되지 않았다.

- e27 재실행 일치율: A_reproduction/R7_e27_output.txt
- e28 자기일관성 및 RL 대비: A_reproduction/R1_e28_output.txt
- e29 무행동 미만 출력 비율, e30 검출가능 차이 참고 출력: A_reproduction/e29_below_noop_output.txt, A_reproduction/e30_detectable_effect_output.txt
- e31 정책 행동 행렬: A_reproduction/e31_policy_actions.parquet, A_reproduction/e31_prep_output.txt
- e33/e34 비용축 분석: A_reproduction/R9_e33_output.txt, A_reproduction/R9_e34_output.txt

실제 실행한 e33/e34는 기업 내 중심화된 회귀 기울기를 계산한다. 기업별로 관측된 권고, 조건, 모델 및 반복에 걸쳐 매출원가율 비중과 점수의 연관성을 추정하는 것이며, 같은 고정 행동의 매출원가율·판관비율 배분만 바꿔 시뮬레이터에 다시 넣은 비교가 아니다. 따라서 그 기울기를 배분 변경의 인과효과로 표현할 수 없다.

## 재현하지 않은 대체자료

작업 폴더 옆의 analysis 및 archive 경로에는 2026년 9월 Plan-3 관련 자료가 일부 있다. 현재 자료에는 11월 E1과 동일한 데이터 해시, 완전한 575개 기업 집합, 실험 조건 및 프로토콜이 입증되어 있지 않다. 따라서 R1–R9의 대체 입력으로 사용하지 않았다.

논문과 발표의 최신 원본은 지정된 thesis_repo 경로에서 찾을 수 없었다. 이 작업에서는 문서·발표 원본을 수정하지 않았고, 동결 자료도 변경하지 않았다.

## 2026-11-18 추가 자료 복구 보완 — 이전 미확보 판정을 대체

사용자 지정 자료 경로를 다시 확인한 결과 `C:/Users/Demian/Downloads/구조적재검토_분석스크립트_20261117.zip`을 확보했다. 앞선 감사에서 “현재 Downloads 및 GitHub main에서 찾지 못했다”고 기록한 상태는 이 보완 이후의 판정으로 대체한다.

| 자료 | 크기 | SHA-256 | 처리 |
|---|---:|---|---|
| `구조적재검토_분석스크립트_20261117.zip` | 93,251 B | `501a6836a3059a4055cd4a50c7e9a4772674cbef095e5efb32fbf59f305f2df2` | 원본 보존, 별도 source_inputs 아래 추출 |
| `기술개선_사전점검_스크립트_20261118_v2.zip` | 19,298 B | `e2ff786b095515275a3e934e84a2d11aa856314959350d63cadba0dbfead6d6e` | 원본 보존 |
| `기술개선_사전점검_스크립트_20261118.zip` | 13,811 B | `f41ac1a2e806c849ad0bdcd1ecdce3bb2c87201eadd83fbf9032cf6715213f2c` | 원본 보존 |

11/17 ZIP은 56개 파일을 포함한다. README, e1–e26, `delta_train.py`, `fam2.py`, e17 요약 JSON과 보조 CSV를 확인했다. 요청 분석과 직접 연결되는 스크립트는 R2 `e15`–`e17`, R3 `e21`·`e23`, R4 `e22`, R5 `e15`, R6 `e18`·`e20`, R8 `e15`·`e16`이다. 사전 계산된 575×606 점수면·정책 재배정 배열은 ZIP에 없으므로 재현 키트의 고정 Simulator/Oracle로 별도 run workspace에서 생성했다.

고정 데이터 및 실행 연결:

- frozen distribution: `THESIS_REPRO_KIT_v2.1.1_FINAL.zip`, SHA-256 `931631d52541bb6dfdd5ed9fda767e2e48ccb8b7521f70ae771db68df2c1831d`
- 공식 E1 release의 `E1/observations/e1_observations.parquet`: 785,854 B, SHA-256 `9afbdd177b463e9c0667eddc4b7c62d3166d7c6e46ef34b02a270699972364c`
- Python 3.11.15, NumPy 2.3.3, pandas 2.3.3, SciPy 1.16.1, scikit-learn 1.6.1, joblib 1.5.3, PyYAML 6.0.3, PyArrow 20.0.0, statsmodels 0.14.5
- run copy: `A_reproduction/R2_R8_replay_workspace/analysis_20261117/`; 별도 출력 로그: `A_reproduction/R2_R8_replay_workspace/logs/`
- 원 ZIP·추출본의 분석 스크립트는 수정하지 않았다. 모든 계산 출력은 run copy에 생성했다.

R2–R6·R8 직접 사용 스크립트와 README의 SHA-256은 다음과 같다. 기준 경로는 `source_inputs/structural_review_20261117_extracted_v1/analysis_20261117/`이며, 동일 파일이 별도 run copy에 복제되어 실행됐다.

| 파일 | SHA-256 |
|---|---|
| `README.md` | `6f7ac826fe777b08809c0333b3138bf5552d36dbd72555d5c4ee044ac55707c7` |
| `e15.py` | `a3bfd352db968485950f13469a63e591ad645ef2be2acdFAF08E6B7F22030537` |
| `e16_an.py` | `15c0e8bb30cb1f4ada174f36d5c6e7398fc73a62f3cb6d9cb0727b96846b7d45` |
| `e17_summary.py` | `de75c8d15a65b4cdf8474018ab6a96e2313b46613e397426ee7a14306b1c6f18` |
| `e18_an.py` | `937106549bb7743419c85a6d883da6da04c6c869223dc7111833201d78399da0` |
| `e20.py` | `bb8d56ee8f0b92d5a6c896aee3a10524e3f20ab89e087b013b3c2a4492911738` |
| `e21.py` | `a0748ecb9d363c49699e83abcaa8cbf76b546bec85664aaeb58711c41009c03f` |
| `e22.py` | `bc750f22d261b1400fc469f7ac14b8c109ee227f1ba350fe97a940f18af657f6` |
| `e23_an.py` | `a18e79d43bc81976926d92606304b6746b23a3c7fee82b3aaea555930af291d8` |
| `delta_train.py` | `132509016314e9e461597af0217d4807e77ccbe8cccbf9b235ac8f64effc02f9` |
| `fam2.py` | `85669208e0f1ef2592652307ccc44416feda426b747abc957b65a5977b250c03` |

C2 의미는 부분 확인이다. 공식 키트의 `C3E_firm_actions.parquet`(SHA-256 `ced89c428deaab8eeaebbceece95ccd40c86eab194ce4d5c03d291198b3d975a`)에 575개 기업의 `C2_action`이 있고 값은 OE 398개, DL 177개다. 계약과 Stage 9 연결 코드는 이를 C2라고 취급하지만, 배포 키트 안에 `archive/DEPLOYED_RELEASE/stage4_candidate_bc` 산출물·체크포인트나 C2 대 Stage4 BC 기업별 교차표가 없다. 분석 스크립트의 “BC/behaviour cloning” 설명은 확인되나, C2가 실제 행동모방 정책인지 독립 검증되지 않았다. 수치 계산에는 보관된 C2 행동열을 사용하되 정책 정체에 관한 문장은 미확정으로 유지한다.

## 분석 재실행 추가 결과

- `t2.py`: 575×9 고정 후보 점수 재산출이 저장된 Oracle 점수와 일치했다. α 최대 차이 `2.84e-14`, β·γ 차이 0.
- `t3.py`: 저장된 STRICT ITT 행동 48,300건을 재시뮬레이션했고 실패 0건. α·γ 최대 차이 0, β `1.42e-14`. STRICT 표의 8,376개 `CONFIRMED_UNUSABLE_COMPLETION`은 canonical 행동이 A0이며, 비영점으로 집계된 1,434건도 부동소수점 오차 수준이다.
- `e2_gen.py`: 606행동×575기업 격자, Simulator 실패율 0%.
- `delta_train.py`·`delta_val.py`: α·δ1·δ2의 OOT 순위상관, 165건 등급이동 방향 일치율 및 AUC가 v3의 기록과 일치했다.
- `fam2.py`: ε1/ε2 OOT 순위상관 0.625/0.658, 165건 방향 일치율 66.1%/63.0%, AUC 0.706/0.688로 v3 기록과 일치했다. ε은 δ 결과를 확인한 뒤 추가된 보조 평가모형이라는 사후적 성격을 유지한다.
- `e18_gen.py`: B1 저장 LLM 출력의 α·β·γ 재산출 최대 차이 `2.84e-14`.
- e12, e16, e18, e23의 정책 재배정·출력 점수 산출 배열을 생성했다. 분석 스크립트는 모두 원본 코드로 실행했다.

## 남은 불확실성과 경계

1. R6의 2·4·8개 후보 수 비교는 v3 기대표와 전부 일치하지 않는다. 상세 수치와 후보집합 차이는 `E2_REPRODUCTION_REPORT.md`의 보완 절을 따른다. 기대표에 맞추는 수정은 하지 않았다.
2. C2의 행동열은 확보했지만 BC 정책 계보는 확보하지 못했다.
3. δ·ε 평가는 관측된 등급 변화로 제한적으로 맞춘 challenger score이며, 실제 향후 기업 성과나 진짜 신용등급 예측력을 증명하지 않는다.
4. 606행동 점수는 고정 Simulator/평가모형의 반사실적 계산이지 실제 실행 결과가 아니다. 평가모형 교체에 따른 강건성과 실제성과 일반화를 분리한다.
5. e33/e34는 관측 권고 안의 매출원가율 비중과 점수의 기업 내 연관 분석이다. 행동 배분만 바꾼 시뮬레이션이 아니며 인과효과로 해석하지 않는다.
6. 이 복구 작업은 E2 10행 결과표 생성·회계 QA, E2 파일럿·본실험을 시작하지 않았다.


