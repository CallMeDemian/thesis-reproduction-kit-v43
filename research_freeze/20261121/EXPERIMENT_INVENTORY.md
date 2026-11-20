# 전체 실험·분석 목록

CSV가 기계 판독용 원본이며 R/e번호는 중복되는 계산이 있다. 코드만 존재하는 항목은 수행 완료로 세지 않는다.

| ID | 연구 | 보존 상태 |
|---|---|---|
| R1 | 자기 재검토·자기일관성·RL 참조 비교 | RETAINED_EXECUTION_OUTPUT |
| R2 | 7평가기 C6-E−C4R | RETAINED_EXECUTION_OUTPUT |
| R3 | 후보선택 평가계열 일반화 | RETAINED_EXECUTION_OUTPUT |
| R4 | 정책 수준 일반화 격차 | RETAINED_EXECUTION_OUTPUT |
| R5 | 행동 주효과·기업×행동 상관 | RETAINED_EXECUTION_OUTPUT |
| R6 | 호출예산·best-of-N | PARTIAL_HISTORICAL_TABLE_MISMATCH |
| R7 | 반복·모델 간 행동 일치 | RETAINED_EXECUTION_OUTPUT |
| R8 | 구성·기업별 배정 가치 | NUMERIC_REPRODUCTION_C2_IDENTITY_UNRESOLVED |
| R9 | 비용축 관찰 연관 | RETAINED_EXECUTION_OUTPUT |
| ADVICE_V1 | policy_advice_taking_financial_state | RETAINED_RAW_LINKS_FIRM_RESULTS_SUMMARIES |
| ADVICE_V2 | policy_advice_taking_mechanism_v2 | RETAINED_RAW_LINKS_FIRM_RESULTS_SUMMARIES |
| RL_SIGNAL_GATE | rl_reference_gate_results | RETAINED_RAW_LINKS_FIRM_RESULTS_SUMMARIES |
| RL_LLM_INTERACTION_GATE | rl_llm_interaction_gate | RETAINED_RAW_LINKS_FIRM_RESULTS_SUMMARIES |
| P0_P5 | 정책·구성·배정 비교/P3 보완 | RETAINED_EXECUTION_OUTPUT |
| B1_COST_COUNTERFACTUAL | 비용축 강도 고정 반사실 | RETAINED_EXECUTION_OUTPUT |
| B1_TABLE_QA | 10행 재무표·회계 QA | RETAINED_EXECUTION_OUTPUT |
| E2_PILOT | 100기업 동시기 2×2·Q/운영 품질 | RETAINED_RAW_REQUESTS_RESPONSES_AND_RESULTS |
| E2_H1 | α RT−R 개선 주 가설 | RETAINED_RAW_REQUESTS_RESPONSES_AND_RESULTS |
| E2_H2_H3 | 개별 정보·상호작용 | RETAINED_RAW_REQUESTS_RESPONSES_AND_RESULTS |
| E2_H4_P3 | P0–P5/P3 대비·점수 격차 회수 | RETAINED_RAW_REQUESTS_RESPONSES_AND_RESULTS |
| E2_H5 | 구성·배정·행동 수용/축·OE 하위집단 | RETAINED_RAW_REQUESTS_RESPONSES_AND_RESULTS |
| E2_H6_Q | 재무변화 예측 오류 | RETAINED_RAW_REQUESTS_RESPONSES_AND_RESULTS |
| E2_T8 | 반복 일치도 | RETAINED_RAW_REQUESTS_RESPONSES_AND_RESULTS |
| E2_T9 | S0/S1/S2 재무 가정 민감도 | RETAINED_RAW_REQUESTS_RESPONSES_AND_RESULTS |
| E2_INTEGRITY_COST | 파싱·원응답·회계·usage 기반 비용 | RETAINED_RAW_REQUESTS_RESPONSES_AND_RESULTS |
| STRUCT_delta_train | 도전자 평가모형 δ(변수 10개) 적합과 검증 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_delta_val | 도전자 평가모형 δ(변수 10개) 적합과 검증 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e1 | 9개 후보의 평가모형 간 전이, 후보 9개 검증 선택, RL 상위 K | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e10_lofo | 평가모형 계열 하나를 남기는 검증 선택 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e11 | RL 정책: Heuristic·행동복제(C2)·기업을 섞은 RL과 비교 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e12_an | 기업을 섞은 대조(20회)로 구성 가치와 배정 가치 분해 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e12_gen | 기업을 섞은 대조(20회)로 구성 가치와 배정 가치 분해 | RETAINED_DERIVED_ARTIFACT_NO_SEPARATE_STDOUT |
| STRUCT_e13 | LLM 비교(C6-E − C4R 등)를 일곱 평가모형에서 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e14 | 기업을 섞은 대조(20회)로 구성 가치와 배정 가치 분해 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e15 | 구조 분해: 행동의 평균 효과와 기업별 효과(기업×행동)의 평가모형 간 일치, 한 평가모형에서 찾은 최적 행동의 전이, 적용 가능성을 통제한 RL 배정 가치, 기업 단위 우월 빈도, 일곱 평가모형 공통 여지 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e16_an | 적용 가능성(상환·장기 전환·투자 축소가 그 기업에서 효과가 있는지)이 같은 기업끼리만 섞은 대조로 LLM 조건의 배정 가치를 다시 계산 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e16_gen | 적용 가능성(상환·장기 전환·투자 축소가 그 기업에서 효과가 있는지)이 같은 기업끼리만 섞은 대조로 LLM 조건의 배정 가치를 다시 계산 | RETAINED_DERIVED_ARTIFACT_NO_SEPARATE_STDOUT |
| STRUCT_e17_summary | 일곱 평가모형의 주요 대비와 95% 신뢰구간(`e17_summary.json`), 요약 그림(`fig_summary_preview.svg`, `fig_summary_widget.html`) | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e18_an | 총 행동크기 ≤ 1 조건의 저장된 LLM 출력 전부를 일곱 평가모형으로 채점. 자기 재검토, 평가모형으로 여러 출력 가운데 고르기(best-of-N), 강화학습 참조 수정을 같은 초안 기준으로 비교 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e18_gen | 총 행동크기 ≤ 1 조건의 저장된 LLM 출력 전부를 일곱 평가모형으로 채점. 자기 재검토, 평가모형으로 여러 출력 가운데 고르기(best-of-N), 강화학습 참조 수정을 같은 초안 기준으로 비교 | RETAINED_DERIVED_ARTIFACT_NO_SEPARATE_STDOUT |
| STRUCT_e19 | 강화학습 정책의 행동을 기본값으로 두고 LLM 출력을 후보로 더해 다른 계열 평가모형으로 고르는 구성이 강화학습 정책 단독보다 높은지, 선택 규칙에 따른 민감도 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e19b | 강화학습 정책의 행동을 기본값으로 두고 LLM 출력을 후보로 더해 다른 계열 평가모형으로 고르는 구성이 강화학습 정책 단독보다 높은지, 선택 규칙에 따른 민감도 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e20 | 같은 호출 수 비교: 참조 수정 한 번 대 재생성 한 번 + 다른 계열 평가모형으로 선택. 선택에 쓴 평가모형으로 채점한 값과 쓰지 않은 평가모형으로 채점한 값의 비 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e21 | 방법별로 참고한 평가모형 계열에서 잰 개선과 다른 계열에서 잰 개선(평가모형별 표준화 평균)과 그 비 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e22 | 정책의 가치를 기업 무관 몫과 기업별 몫으로 정확히 나눔(격자·후보 점수표 사용, 적용 가능성 층 안에서도). 평가모형으로 고른 정책과 강화학습 정책 비교 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e23_an | LLM 수준: 검증기로 고른 출력과 강화학습 참조의 기업별 몫을 같은 층별 대조로 계산하고, 참고한 계열과 다른 계열의 차이(평가모형 일반화 격차, 작업용 이름)를 95% 신뢰구간과 함께 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e23_gen | LLM 수준: 검증기로 고른 출력과 강화학습 참조의 기업별 몫을 같은 층별 대조로 계산하고, 참고한 계열과 다른 계열의 차이(평가모형 일반화 격차, 작업용 이름)를 95% 신뢰구간과 함께 | RETAINED_DERIVED_ARTIFACT_NO_SEPARATE_STDOUT |
| STRUCT_e24 | GPT·Codex 후속 실험 점검: 평가모형별로 C6-E와 C4R의 점수가 다른 기업 비율, 참조 증분이 어디서 나왔는지(RL 행동 복사·RL 쪽 이동·반대 이동), 141개 위약 대조, 같은 프롬프트 재실행 간 기업별 증분의 재현성과 사후 선택 상한 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e25 | 시뮬레이터 결과를 보고 고르는 방법의 여지: 후보 집합(RL+초안, RL+초안+C6-E, RL+LLM 출력 14개, RL+격자 606개)별로, 채점하는 평가모형과 다른 계열의 평가모형이 고를 때의 값과 사후 상한 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e26_ach | 달성 빈도: 실제 기업이 한 해에 행동 1단위 이상 개선한 비율과 기업별 차이 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e26_agree | 실행 간 합의 게이트: 같은 수정이 반복될 때만 LLM 수정을 쓰고 나머지는 RL | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e26_base | 상수 정책(모든 기업에 같은 행동) 대 RL C3-E, 일곱 평가모형. 평가모형별 최선 상수 행동 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e26_dec | RL의 Heuristic 대비 우위를 RL이 고른 행동별로 분해(일곱 평가모형, 실제 등급 기준) | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e26_out | 파일럿: 실제 재무제표 레버 변화(행동 단위) 대 다음 해 실제 등급 변화, 평가모형 민감도와 비교, 정책별 레버 구성과 선형 점수 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e26_out2 | 26e의 강건성(통제, 동시점, 2년, 상승 여부, 기간 분할)과 원가 대비 상대 가중 검정 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e26_out3 | 실제 등급 기준 선형 점수의 정책 간 대응 비교 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e26_out4 | 미리 정한 이질성(전환×단기차입 비중 등)과 RL의 장기 전환 선택 위치 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e26_unc | LLM 실행 간 분산이 평가모형 불일치·RL 취약성·손해 수정과 연관되는지 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e2_an | 1/4 단위 격자(606개 행동) 전체 채점, 검증 탐색의 일반화 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e2_gen | 1/4 단위 격자(606개 행동) 전체 채점, 검증 탐색의 일반화 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e3 | 선택 규칙 민감도, 같은 K에서 제안 출처(RL·LLM·무작위) 비교 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e4 | 참조의 가치 전달률(참조가 초안보다 나을 때/못할 때) | BUNDLED_RESULT_PRESERVED |
| STRUCT_e4b | 참조의 가치 전달률(참조가 초안보다 나을 때/못할 때) | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e5_stress | 가정 교란: 장기 전환 가산금리, 비용 절감 실현율 | BUNDLED_RESULT_PRESERVED |
| STRUCT_e6_an | δ로 격자·저장 정책 채점 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_e6_cat | δ로 격자·저장 정책 채점 | RETAINED_DERIVED_ARTIFACT_NO_SEPARATE_STDOUT |
| STRUCT_e6_gen | δ로 격자·저장 정책 채점 | RETAINED_DERIVED_ARTIFACT_NO_SEPARATE_STDOUT |
| STRUCT_e7_stress | 다섯 평가모형 × 네 시나리오 요약 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e8 | 다섯 평가모형 × 네 시나리오 요약 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_e9 | 도전자 평가모형 ε(변수 9개) 적합·검증·채점 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_fam2 | 도전자 평가모형 ε(변수 9개) 적합·검증·채점 | RETAINED_DERIVED_ARTIFACT_NO_SEPARATE_STDOUT |
| STRUCT_fig_gap | 평가모형 일반화 격차 그림 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_fig_summary | 일곱 평가모형의 주요 대비와 95% 신뢰구간(`e17_summary.json`), 요약 그림(`fig_summary_preview.svg`, `fig_summary_widget.html`) | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| STRUCT_t2 | 회귀 검증: 575×9 격자와 저장된 LLM 행동 48,300건을 다시 채점해 동결 점수와 비교 | RETAINED_EXECUTION_OUTPUT |
| STRUCT_t3 | 회귀 검증: 575×9 격자와 저장된 LLM 행동 48,300건을 다시 채점해 동결 점수와 비교 | CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT |
| NOOP_HARM | A0 미만 출력/평가기 손실 | RETAINED_EXECUTION_OUTPUT |
| DETECTABLE_EFFECT | 반복 평균·표준오차·MDE | RETAINED_EXECUTION_OUTPUT |
| METHOD_FEASIBILITY_1116 | confidence/harm; top-K; multi-evaluator selector; one-step interpretation | REPORT_ONLY_NO_STANDALONE_PRODUCER_FOUND |
| BRAINSTORM_1116 | 방법론 아이디어와 관련 연구 정리 | PRESERVED_PROPOSAL_NO_EXECUTION_CLAIM |
| BASELINE | 기존 V4.3 연구 | PINNED_PREVIOUS_RELEASE_DEPENDENCY |
| ABLATION_E1 | T1–T5; R9/R10; D1–D4; E1 3450 | PINNED_PREVIOUS_RELEASE_DEPENDENCY |
| HUMAN_BASELINE | 236 source documents; comparison56 | PINNED_PREVIOUS_RELEASE_DEPENDENCY |
