# E2_GATE1_RECOVERY_REPORT — 공식 E1 릴리스 자료 연결 및 재현 보완

작성일: 2026-11-18  
작업 폴더: `C:/Users/Demian/Desktop/REPRO_KIT_semantic_repair/e2_20261118`

## 요약

이번 보완에서 공식 E1 릴리스 자산을 확보하고 무결성을 확인했다. E1 요청·응답·관측·채점 자료가 공식 release ZIP에 들어 있었으며, 현재 작업공간의 Simulator 입력 일부도 GitHub main의 Git LFS 객체와 SHA-256이 일치했다. 이전 Gate 1 기록의 “E1 자료를 확보하지 못했다” 및 “Simulator 원입력이 없다”는 판단은 이 보고서의 확인 결과로 보완한다.

E1 분석표는 수치·분석 행 기준으로 재현됐고 독립 재채점 게이트도 3,450행에서 PASS했다. 이후 Downloads에서 11/17 분석 ZIP을 확보해 R2–R5·R8을 재현했고, R6의 일부 기대값 차이를 확인했다. 상세 최신 판정은 이 보고서의 7절과 `E2_REPRODUCTION_REPORT.md` 보완 절을 따른다.

Simulator의 핵심 입력 계약과 파일은 해시로 연결됐지만, 이번 자료 복구에서는 E2의 10행 결과표나 P0–P5 진단을 생성하지 않았다. 단위·출력 부호·결측 및 실행불가 규칙을 실제 결과표 행별로 검사한 것도 아니다. Gate 1에서 신규 LLM 호출, RL 재학습, E2 파일럿·본실험은 계속 중단 상태다.

## 1. 원천 저장소와 릴리스 무결성

GitHub 원천 식별자는 서로 구분했다.

| 저장소 | 확인한 main | 본 보고서에서의 역할 |
|---|---|---|
| `CallMeDemian/thesis-reproduction-kit-v43` | `b4ed3bf5504d792a0a06f66c7c3712601ae857bf` | 재현 키트 및 Git LFS 원천 |
| `CallMeDemian/REPRO_KIT_semantic_repair` | `324b5fbb0ac9cf322ae20cf97793165227bf45ed` | 별도 저장소. thesis 키트와 동일한 상태로 간주하지 않음 |

공식 release는 [thesis-v43-ablation-20261102](https://github.com/CallMeDemian/thesis-reproduction-kit-v43/releases/tag/thesis-v43-ablation-20261102)이며, GitHub가 자동 생성하는 소스 ZIP은 사용하지 않았다. 아래 자산을 `source_inputs/github_release_ablation_20261102_readonly/`에 저장했다.

| 자산 | 바이트 | SHA-256 | 검증 |
|---|---:|---|---|
| `THESIS_V43_ABLATION_20261102.zip` | 90,724,839 | `3f217ae100103ff83030f1dd1a31bd90821271eb2a58d0933958db65af4e3207` | GitHub release asset digest 및 `SHA256SUMS.txt`와 일치 |
| `RELEASE_MANIFEST.json` | 30,262 | `9a9bce603c954c901cd644371ba419637ec7b94eedaf6beb97dfd5a737ead8e5` | 공식 SHA256SUMS와 일치 |
| `SHA256SUMS.txt` | 270 | `9e918ba9e546c3876988f75b7dc38780dca24ea3e16718a5e3364b76e08e008a` | 공식 release 자산 |

manifest에 기록된 release commit은 `d831f2d2ff0ebe970b2e0325cc73dc295a37a1cc`, 포함 파일의 source commit은 `c15c2eb7a160506e242c255277d87a90d3703a18`이다. ZIP에는 106개 파일(압축 해제 기준 192,450,113바이트, LFS 파일 19개)이 들어 있다. 압축 내부 106개 파일 각각의 상대 경로·크기·SHA-256을 manifest와 대조했고 모두 일치했다. 실행 중 생긴 manifest 밖의 Python cache 파일은 제거했다. 최종 추출 파일 106개와 ZIP·manifest·checksums·release notes 등 원천 파일 4개, 총 110개 파일에 Windows 읽기 전용 속성을 설정했다.

## 2. 복구된 E1 자료

추출된 원천은 `source_inputs/github_release_ablation_20261102_readonly/contents/ablation_20261102/` 아래에 있다.

| 자료 | 상대 경로 | SHA-256 | 확인 내용 |
|---|---|---|---|
| 프로토콜 | `E1/E1_PROTOCOL.md` | `90d645a0c00eb52659d118b3c555da94d71d84d8f00e8a9454e5c0e3f85748c5` | 조건, 모집단, 프롬프트 변경, 분석 대비 및 QA 규칙 |
| canonical 관측값 | `E1/observations/e1_observations.parquet` | `9afbdd177b463e9c0667deddc4b7c62d3166d7c6e46ef34b02a270699972364c` | 3,450행 × 36열, 575개 기업 키 |
| 관측값 CSV 사본 | `E1/observations/e1_observations.csv` | `31cff2b9c772653184bea1b9b594ee57c03f0ae5a05b0a8d2b70db7fd3ad5bcb` | parquet의 표 형태 사본 |
| 요청 manifest / 기업 키 | `E1/requests/E1_request_manifest.csv` | `1f12775443db3a929cddf7a619a27882ad49fe5b189da23572f908854064357a` | 3,450 요청 ID, 모델·기업·조건, parent/source ID와 payload/body hash |
| 요청 빌드 보고 | `E1/requests/E1_build_report.json` | `1c420a7c3a04dbd09205577bc579cae360d067c28bf57b3d3d5868aa56fd778b` | 요청·payload 생성 QA 기록 |
| 통합 원응답 | `E1/runs/responses.jsonl` | `7f58d5d8f55dae4b4c933bfce6dd89297cb11c7b16274a67a85b084ca05cf034` | 3,450행 |
| 선택 응답 | `E1/runs/e1_selected.jsonl` | `6ecd0efccce71d61fea0c6a7eb7e20c199fc2896adb622f53cbd6c6a683fd22c` | 3,450행 |
| 공식 점수 게이트 결과 | `E1/scoring/score_gate_result.json` | `2f4355922832819a1ce5316f9da20f5769fba1d5f18e322ad445d99cee9ef170` | PASS 기록 |
| 공식 분석 스크립트 | `E1/scripts/e1_analyze.py` | `e813d7d25091d928dfefdcdad3aadac1c30d5aaa503d90771609bf2802d17e75` | T5 분석 및 wording 생성 |

실제 provider 요청 본문은 `E1/requests/batches/`의 8개 JSONL shard에 보존되어 있다. 각 shard 전체의 SHA-256은 다음과 같다. 이 파일들이 실행 당시 프롬프트의 원문이다.

| 요청 본문 | SHA-256 |
|---|---|
| `google_e1_shard_00001.jsonl` | `b925540c3a8f4425b8f0e532c07333e8fdefbe972ffcd2f78f35c5f9e80a8307` |
| `google_e1_shard_00002.jsonl` | `d592fcfca9cd2eda4ded5990589973112fe9b418f5e1c3087bbbf198047c2a88` |
| `google_e1_shard_00003.jsonl` | `5df43212bbc6d3731724fad0338acb5d42860dcb507df2ab93ed453ef3630ef6` |
| `google_e1_shard_00004.jsonl` | `8d08eca2d894f290a425121d84e5c614b5a2d38983bbdf89c9644077a2d8f329` |
| `openai_e1_shard_00001.jsonl` | `a94813354990649f4cbb54a7857ec4a1abe45b71110c9e9111ccde6e6a85f35b` |
| `openai_e1_shard_00002.jsonl` | `3cf4742bc5d51b7a3ed46b04fa07874757b5471e2e2b7304d6d9333ddb6ca58c` |
| `openai_e1_shard_00003.jsonl` | `14cf26f2557e49062f3003ce02f8efe8ab489f839e6270c2b37a9f0d2d5d3301` |
| `openai_e1_shard_00004.jsonl` | `79a3074bee040d799aa14095ff480f1e318ab6153206d62ee5e76e7dc666d167` |

별도의 C4 초안 파일은 릴리스에 없다. 프로토콜상 C4는 고정 입력이고 재실행하지 않았으며, 기업·모델별 C4 초안이 각 실제 요청 본문에 포함된다. request manifest는 `source_request_id`, `parent_request_id`, `parent_raw_sha256`, `visible_payload_sha256`, `provider_request_body_sha256`로 해당 요청의 원천을 연결한다. 따라서 초안 본문은 개별 shard 안에서 확인하고, manifest로 기업·모델·요청 관계를 추적한다.

프로토콜에 따르면 대상은 2024 회계연도 기업 575개, 모델 2종, 조건 E1-C4R/E1-RL/E1-OE이며 총 요청 3,450건이다. 조건별 1,150건, 모델별 1,725건이다. E1-RL은 기업별 C3-E 행동을 참조하고, E1-OE는 동일한 요청에서 참조 JSON만 OE로 바꾼다. 세 조건 모두 같은 기업·모델의 동결 C4 초안을 받는다. 분석 스크립트는 E1 firm key 집합을 고정 기준과 비교해 다르면 중단한다.

## 3. E1 재현 결과

### 분석표 재현

공식 `E1/scripts/e1_analyze.py`를 공식 parquet 관측값과 요청 manifest에 적용했다. 실행 위치는 `A_reproduction/E1_release_replay/`이며 결과는 `T5_E1.csv`와 `E1_wording.json`이다. 스크립트 출력은 `status=PASS`, 30개 결과 행, `placebo_gate=PASS`였다.

공식 T5 파일의 SHA-256은 `c198dd51074582c438ca977ef8766ca72465881ab813b7016148209fe59d4452`, 재실행 파일은 `256f9990755fe3943cace2a2d821cffcb96cf823441e42610f0acb0e259d74ae`다. 30개 행에서 분석 수치 및 나머지 열은 일치했다. 차이는 provenance 열 `analysis_commit`뿐이다. 공식 파일은 `a322d672eca6`, 로컬 재실행은 비커밋 실행 상태를 표시하는 `UNCOMMITTED`였다. 따라서 분석 결과는 재현 일치지만 CSV 파일 전체 바이트 동일은 아니다. `E1_wording.json`은 공식·재실행 SHA-256이 모두 `2a08eee146638c2144ae2d0d7cbb2dc2909bf3aa3a05c8bc43ff637548c60869`로 바이트 단위 일치했다.

### 독립 Simulator/Oracle 재채점

공식 E1 채점 스크립트의 요구대로 `frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip`을 별도 실행 폴더에 풀고 E1 `e1_setup_runroot.ps1`이 정한 경로 연결을 구성했다. 원본 release·Frozen 파일을 수정하지 않았다. 입력 ZIP은 `source_inputs/github_main/frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip`, SHA-256 `931631d52541bb6dfdd5ed9fda767e2e48ccb8b7521f70ae771db68df2c1831d`다. 재채점 결과는 `A_reproduction/E1_score_gate_replay.json`에 저장되어 있다.

| 평가 | α | β | γ |
|---|---:|---:|---:|
| delta 점수 일치 | 3,450 / 3,450 | 3,450 / 3,450 | 3,450 / 3,450 |
| 점수 일치 | 3,450 / 3,450 | 3,450 / 3,450 | 3,450 / 3,450 |
| 무행동 점수 일치 | 3,450 / 3,450 | 3,450 / 3,450 | 3,450 / 3,450 |
| 최대 절대 delta 차이 | 2.842170943040401e-14 | 0 | 0 |

상태는 `PASS`, 허용 오차는 atol/rtol `1e-9`, NaN은 0행이었다. 로컬 재실행 결과 SHA-256은 `55000231c89b0bc0bb5c554de531014c02bfdf1a51cb2b7478b0de5272b9ee98`이다. 공식 기록(`2f435...`)과 비교해 실행 시간 값만 다르고 모든 점수·행 수·허용 오차·상태는 일치했다. 이는 기존 C4/C4R/C6-E 동결 행동을 Simulator/Oracle 경로로 다시 채점한 결과다. 신규 LLM 생성이나 E2 결과를 뜻하지 않는다.

## 4. Simulator 입력 계약과 간접 평가정보

이전에 `github_main` 체크아웃에서 Git LFS 콘텐츠가 포인터 상태였기 때문에 Simulator 원입력을 사용할 수 없다고 판단했다. 이번에는 workspace archive에서 발견한 복사본을 GitHub main의 LFS 객체 해시와 직접 대조했다. 아래 파일은 경로·크기·SHA-256이 canonical GitHub LFS 객체와 일치하여 원천 입력으로 확인했다.

| 자료 | 확인한 파일 | SHA-256 |
|---|---|---|
| 2024 기준 기업 맥락 | `archive/DEPLOYED_RELEASE/stage2_candidate_projection/runtime_inputs/historical_financial_context_v4_3/actual_context.parquet` | `25b99f6a0eadacbbd34fab94fb896be66d5100fe5fe9ef813a5f304ddd260147` |
| 재무 상태 | `archive/DEPLOYED_RELEASE/stage2_candidate_projection/runtime_inputs/historical_financial_context_v4_3/actual_financial_states.parquet` | `97cb72a7a9e80784766b1727dddb264060863b40bffffb31759be0ba3164e1ce` |
| 8축 행동 계약 | `archive/DEPLOYED_RELEASE/stage2_candidate_projection/candidate_action_contract_v4_3.json` | `f722e189a265e111c018341254a648ddb9cecc05aa8c12c3547a023ea4c458c0` |
| 고정 후보 라이브러리 | `archive/DEPLOYED_RELEASE/stage2_candidate_projection/final_candidate_library__P50.yaml` | `c5f44780f2f47a77e667836d1499ec4b43c218796f26af597885685319f80190` |
| 금융 Simulator 핵심 구현 | `source_inputs/github_main/src/credit_recourse/simulator/financial_simulator.py` | `77cd49f23dc8fc2502d3547d353d32ad076c7ea9bccdf2a7c65f724a46f41270` |
| α Oracle 파라미터 | `archive/DEPLOYED_RELEASE/stage1_oracle_backends/alpha/oracle_alpha_params.json` | `7ddc6ebea4fe8936332b323d07438c9ec82ffc19b030c9138ff85dc116c4694b` |
| α·β·γ / RL 동결 계약 | `source_inputs/github_main/contracts/scientific/final_freeze/final_oracle_rl_contract.json` | `bd2527849b9dc37d7a60fa97be200d02dbaba25978a0d657d36001c4dd0453ff` |
| BPRate v4 production ledger | `archive/DEPLOYED_RELEASE/stage2_candidate_projection/runtime_inputs/business_plan_rate_v4/production/production_selected_rate_ledger.parquet` | `a8bf256190f0d10353f508e54b745af8ea6ca3ea3010ad6deca3f14dddac06f8` |

기준상태 parquet은 74,867행이며 2024년 행 2,677개가 있다. E1의 575개 `firm_key`는 모두 `firm_id::2024` 형식이고, 575개 기업 ID 전부가 해당 2024 재무 상태와 맥락 자료에 존재했다. 이 대조에서 2024 기업 ID 중복은 없었다. 이 연결은 9월 Plan-3 파일을 E1 대체자료로 쓰는 방식이 아니라, 11월 E1 기업 키와 GitHub LFS 해시로 확인한 2024 원입력 관계다.

계약에서 확인한 사항:

- `firm_state.py`는 모든 금액값의 단위를 원자료와 같은 **천원(KRW thousand)**으로 명시한다. 재무 상태의 시간축은 의사결정 연도 다음 해이며, main의 동결 계약은 Stage6가 전체 재무제표 Simulator 이후 Oracle 채점을 수행한다고 명시한다.
- 후보 행동 계약은 8개 축의 열, 범위, 부호, 정규화 scale과 후보 ID를 고정한다. 후보 ID는 A0/DL/RF/CX/WC1/WC2/OE/MX1/MX2이며, 이 라이브러리와 LLM이 작성하는 자유 8축 행동은 동일한 대상을 뜻한다고 가정하지 않는다.
- E1 공식 점수 게이트는 동결 Stage8 재무상태·Simulator·α/β/γ를 연결해 무행동 점수와 행동 점수를 비교했고 3,450행에서 일치했다.

E2 결과표는 Oracle 점수·등급·가중치·순위를 직접 제공하지 않도록 설계되어 있다. 다만 현재 계약에서 α 입력 선택변수로 `R006`, `R064`, `R085`, `R116`, `R157`, `R182`, `cap_change_count_3y`, `log_assets`, `operating_loss_freq_3y`가 확인된다. 예정 재무변화 열과 LLM이 보는 기초 재무정보를 조합하면 Oracle 평가변수와 겹치는 비율·재무상태를 추론할 수 있다. 따라서 결과표는 평가점수를 직접 주지 않지만 간접 평가정보를 전달할 가능성이 있다. 이번 확인은 α 계약의 변수명과 입력 자료 연결까지이며, 모든 Oracle 산식과 예정 결과 열을 완전히 crosswalk한 것은 아니다. 결과표가 평가정보와 독립이라고 해석해서는 안 된다.

핵심 Simulator 입력과 계약은 확인됐으나 **E2 10행 표의 실제 생성은 실행하지 않았다.** 표의 부호·반올림·missing 및 실행불가 표기, 표별 회계 항등식, E2-Q zero tolerance, E2 프롬프트의 행동예산이 원 E1 프롬프트와 동일한지에 대한 검증은 별도 작업이다. 따라서 본 보고서는 B1 실행 완료를 선언하지 않는다.

## 5. 11월 17일 ZIP과 R2–R6·R8 의존성 — 초기 탐색 기록

이 절의 “미확보/재현 불가” 표는 Downloads ZIP을 다시 찾기 전 첫 조사 시점의 기록이다. 실제 복구와 최신 재현 결과는 7절이 대체한다.

`구조적재검토_분석스크립트_20261117.zip`을 현재 작업공간과 다음 Windows 폴더에서 다시 찾았지만 발견하지 못했다: `C:/Users/Demian/Downloads`, `%USERPROFILE%/Downloads`, `%USERPROFILE%/OneDrive/Downloads`, `%USERPROFILE%/Desktop`. OneDrive Downloads 폴더는 존재하지 않았다. Downloads·Desktop 아래 ZIP 35개를 열어 내부 파일 목록을 검색하고, 내부 README 파일이 있으면 e15–e26·`delta_train.py`·`fam2.py`·분석 ZIP 관련 문자열도 확인했다. 해당 내부 파일명이나 README 언급이 있는 ZIP은 0개였다. 2026-11-14~2026-11-20 수정된 ZIP은 4개였으며, 이번 작업에서 확보한 공식 배포 ZIP 2개와 Downloads의 11/18 기술개선 ZIP 2개였다. 기술개선 ZIP 두 사본은 모두 19,298바이트, SHA-256 `e2ff786b095515275a3e934e84a2d11aa856314959350d63cadba0dbfead6d6e`로 동일했다. 이 범위에서도 11/17 분석 ZIP은 없었다.

thesis-reproduction-kit-v43 main b4 체크아웃에도 e15–e26 스크립트, `delta_train.py`, `fam2.py`가 없고, 공식 Ablation release에는 E1 관련 파일만 있으며 e15–e26 분석 패키지는 없다.

기존 `E2_REPRODUCTION_REPORT.md`의 R1·R7·R9 상태는 유지된다. R1(e28)과 R7(e27)은 보고된 범위에서 재현 일치, R9(e33/e34)는 관찰된 기업 내 연관 분석이 재현 일치다. e33/e34 결과를 배분 변경의 인과효과로 해석하지 않는다. 이번 E1 release 재현은 이 세 항목과 별도의 재검증이다.

| 항목 | 필요한 핵심 근거 | 현재 상태 | 판정 |
|---|---|---|---|
| R2 | e15–e17 코드, δ·ε 학습/검증 입력과 적합 파일, 평가모형별 같은 행동 점수 | 미확보. E1 release는 별도 E1 조건·점수이며 이를 대체할 수 없음 | 재현 불가 |
| R3 | e21/e23 코드, 검증기가 선택한 후보·선택 로그, 해당 후보의 α/β/γ·δ/ε 점수 | 미확보 | 재현 불가 |
| R4 | e22 코드, 606개 정책/행동 후보, 기업별 평가 점수표 | 미확보 | 재현 불가 |
| R5 | e15 분석 규칙, 동일 606개 행동×기업의 δ·ε 재적합 점수 행렬 | 미확보 | 재현 불가 |
| R6 | e18/e20 코드, 1·2·4·8·10회 생성 후보, 시드·선택 규칙·기업별 점수 | 미확보 | 재현 불가 |
| R8 | e15/e16의 정확한 분석 규칙, 기업×행동 점수 표면, 실행가능성 및 교차기업 재배정 처리 | 스크립트·분석 정의 미확보. Simulator 핵심 입력은 해시 연결됐으나 R8 점수 표면과 교차배정 규칙은 검증되지 않음 | 재현 불가; 기존 설계서 숫자 인용 금지 |

초기 조사 시점의 R2–R6·R8 판정은 결과가 반증된 것이 아니라, 계산에 필요한 원 스크립트·자료·정의가 연결되지 않았다는 의미였다. 당시 기대 결과 숫자를 분석 결과로 대체하지 않았다. 이 판정은 Downloads ZIP을 확보해 실행한 최신 판정으로 7절에서 갱신했다.

두 저장소의 GitHub 이력을 혼동하지 않았다. `REPRO_KIT_semantic_repair`의 main은 별도 commit `324b5fbb...`이고, 이 저장소에 대해 코드 검색 색인이 확인되지 않았으므로 그 원격 저장소에 e15–e26 자료가 전혀 없다고 단정하지 않는다. 초기 조사에서만 현재 작업공간·다운로드 폴더·공식 E1 release에 분석 ZIP/스크립트를 연결하지 못했으며, 이후 사용자의 다운로드 경로에서 ZIP을 확보한 최신 결과는 7절에 정리했다.

마지막 상태 점검에서 `source_inputs/github_main/.git/index`가 없고, 해당 로컬 checkout의 HEAD tree는 1,503개 파일인데 `git ls-files`에는 0개가 잡혔다. 따라서 이 checkout의 `git status`로 기존 작업 트리가 깨끗했다고 증명할 수는 없다. 이 복구 작업은 checkout 파일을 쓰거나 Git index를 복구하지 않았다. 분석에 사용한 핵심 파일은 별도로 GitHub commit/LFS SHA와 대조한 파일만 채택했다.

## 6. 기존 Gate 1 보고의 보완과 중단 기록

기존 `E2_GATE1_REPORT.md` 및 `E2_REPRODUCTION_REPORT.md`의 다음 판단을 보완한다.

1. **E1 요청·응답·관측·채점 자료:** 누락이 아니라 공식 Ablation release에 있다. 원요청 3,450건, 응답, canonical 관측값, 점수 결과가 복구됐다.
2. **독립 재채점:** 공식 고정 distribution ZIP과 현재 b4 main Simulator/Oracle 경로를 격리 runroot에 연결해 3,450건 재채점 PASS를 재현했다.
3. **Simulator 입력:** GitHub main의 LFS 포인터가 로컬 checkout에 풀리지 않았지만 workspace archive의 해당 파일은 canonical LFS 해시와 일치했다. 따라서 이 파일들은 검증된 source blob 복사본으로 쓸 수 있다.
4. **R2–R6·R8:** 11/17 분석 ZIP을 복구했다. R2–R5와 R8의 수치 재현은 일치했고 R6은 일부 값이 기대표와 달랐다.
5. **B1:** 입력 계약은 상당 부분 연결됐으나 10행 E2 표 생성과 회계 QA는 실행하지 않았다.

이번 복구에서 수행하지 않은 작업: 신규 LLM API 호출, E2 파일럿 100개 기업, 12,650건 본실험, RL 재학습, 기존 Frozen 데이터·논문·발표 원본 수정, 원격 Git push 또는 PR 병합.

**Gate 1 회복 판정(초기 상태; 7절 최신 갱신 참조):** 당시 11/17 분석 ZIP이 미확보여서 R2–R6·R8을 재현 불가로 기록했다. 이후 ZIP을 찾아 지정 분석을 재실행했으며, 최종 상태는 7절에 정리했다. B1 결과표는 미생성이다.

## 7. Downloads ZIP 복구 후 Gate 1 상태 정정

이후 사용자가 지정한 Windows Downloads 경로에서 `구조적재검토_분석스크립트_20261117.zip`을 확인했다. 파일 크기는 93,251 B, SHA-256은 `501a6836a3059a4055cd4a50c7e9a4772674cbef095e5efb32fbf59f305f2df2`이며, 56개 파일을 별도 읽기·실행 폴더로 추출했다. 앞선 Gate 보고의 “R2–R6·R8은 분석 ZIP 미확보로 재현 불가” 판정은 이 절에 의해 대체한다.

재실행 판정은 R2 일치, R3 일치, R4 일치, R5 일치, R6 일부 불일치, R8 점수 분해 일치(C2의 행동모방 정책 정체는 미확정)다. 결과와 불일치 수치는 `E2_REPRODUCTION_REPORT.md`의 업데이트 절과 `A_reproduction/R2_R8_replay_workspace/logs/replay/`에 기록했다. e17 기대표 JSON 17개 대비×7 평가모형의 mean·95% CI는 재실행 JSON과 5×10⁻⁷ 이내에서 모두 일치했다. R6은 기대표 중 Gemini 2-call α 값과 4·8-call 중간값이 동일 분석 산출로 재생되지 않았다. 이를 조정하지 않고 별도 보고했다.

### R2–R6·R8 의존성 및 실행 가능 판정

| 분석 | 필요한 코드·입력 | 이번에 연결한 입력/실행 | 판정과 잔여 자료 |
|---|---|---|---|
| R2 | e15–e17, E1 기업별 관측·행동, α·β·γ·δ·ε 점수, δ·ε 적합 자료 | e17 대비표와 606×575 점수면, δ·ε 재적합을 분리 run copy에서 재생 | 재현 일치. e17 357개 평균/CI 셀 전부 허용오차 내. 기존 575개 표본 재분석 |
| R3 | e21/e23, verifier 후보 출력과 기업별 평가모형 점수 | release의 저장 LLM 출력과 재산출된 Oracle/challenger 점수 배열 | 재현 일치. 후보 선택 원리에 대한 값이지 실제 실행 결과가 아님 |
| R4 | e22, 9개 및 606개 행동 정책, 기업×평가모형 점수 | e2_gen으로 만든 606×575 점수면과 저장 정책 행동 | 재현 일치. Oracle 선택 정책은 실제 정책 실행값이 아님 |
| R5 | e15, 행동 주효과·기업×행동 점수면 및 δ·ε | 동일 606×575 재산출 점수면 | 재현 일치. 평가모형 간 상관·최적 행동 이득 이전 지표 재현 |
| R6 | e18/e20, 저장 no-advice 후보, 선택 규칙·seed·기업별 점수 | e18/e20 실행 및 보존된 후보 출력 | 부분 불일치. Gemini 2-call 및 별도 4/8-call 표 값이 기대표와 다름. 기대표 생성 당시 후보집합·selection seed·행동별 선택 로그가 없음 |
| R8 | e15/e16, 정책별 기업행동, 교차기업 재배정 규칙, 기업×행동 점수면 | 고정 606×575 Simulator/Oracle 점수면과 저장된 C4/C4R/C6-E, RL, OE, C2 행동열 | 수치 재현 일치. 다만 C2↔Stage4 행동모방 출력 crosswalk가 없어 BC 정체는 미확정 |

R2–R5·R8에 필요한 입력은 이번 작업에서 연결해 실행했다. 남은 누락은 R6 과거 선택 provenance와 C2 정책 계보이며, 이 결손은 계산된 값의 범위를 넘어 기존 기대표 차이 또는 C2 정체를 확정하는 데만 제약이 된다.

Simulator·Oracle 재채점은 t2 고정행동 격자, t3 48,300개 저장 행동, 606×575 임의행동 격자 모두 실패 없이 완료됐다. δ·ε 재적합도 README가 고정한 Python 패키지 조합에서 수행됐다. 9월 Plan-3 자료는 11월 E1의 대체재로 사용하지 않았다. 기존 E1 공식 release와 GitHub source-of-truth 재현은 보완된 source/reproduction 보고를 따른다.

C2 행동열은 575개 기업에 대해 현재 키트 Stage6 산출물에 있으나, 해당 배포본에는 Stage4 BC 체크포인트 및 575개 기업 출력 crosswalk가 없다. 스크립트가 사용한 “BC” 라벨은 독립 검증되지 않아 E2-B나 BC 해석을 확정하지 않는다. R6 중간 표의 원래 후보집합·selection seed·행동별 점수 로그도 미확보다.

현재 Gate 1 종료 경계는 다음과 같다.

- R2–R5와 R8의 지정 스크립트 계산을 수행했고, R6 결과 차이를 문서화했다.
- E2용 10행 결과표는 새로 만들지 않았고 회계 QA도 추가로 진행하지 않았다.
- 신규 LLM API, E2 100개 기업 파일럿, E2 12,650건 본실험, RL 재학습, 동결 자료·논문·발표 원본 수정, 원격 push/PR 병합은 하지 않았다.
- 1차 관문에서 중단한다. 남은 복구 자료는 C2/Stage4 BC 출력 교차표와 R6 과거 후보 선택·seed 로그다. 이 자료가 없으면 R6 기대표 차이 및 C2 정책 정체를 확정 설명하지 않는다.

