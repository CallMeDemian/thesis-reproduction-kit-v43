# Oracle–Simulator 입력 교차표

작성: 2026-11-18. 용도: 분석자 전용 감사자료. 이 파일은 프롬프트에 포함하지 않는다.

## 범위와 확인된 입력변수

동결 α·β·γ 백엔드의 선택 재무변수 집합은 모두 다음 9개다: `R006`, `R064`, `R085`, `R116`, `R157`, `R182`, `cap_change_count_3y`, `log_assets`, `operating_loss_freq_3y`. β·γ의 `selected_variables`와 α의 파라미터/입력 산식 계약에서 확인했다. β는 표준화·결측치 대치 후 ordered logit, γ는 같은 변수로 gradient-boosting 회귀를 쓴다. α는 선택 변수별 변환·binning과 고정 가중치를 사용한다. 평가 백엔드가 변수를 공유한다는 사실은 표의 정보가 평가정보와 겹치지 않는다는 뜻이 아니다.

기준 산식(시뮬레이션된 t+1 state 및 기준 t state):

| 변수 | 동결 산식 | 결과표 연결의 해석 |
|---|---|---|
| R006 | pretax_income / revenue | 세전이익·매출액 변화가 분자·분모에 직접 닿음. 절대 비율을 구하려면 기준 재무값 필요. |
| R064 | retained_earnings / total_assets | 표의 `total_equity`는 `retained_earnings`와 같지 않다. 자본변동으로 간접 힌트가 될 수 있으나 기초 이익잉여금과 총자산 및 배당·순이익 반영이 추가로 필요. |
| R085 | financial_cost / revenue | 표의 이자비용은 금융비용의 일부일 수 있지만 `financial_cost` 전체와 동일하다고 보장되지 않는다. 매출·전체 금융비용의 기초값 필요. |
| R116 | (current_assets − inventory) / current_liabilities | 유동자산·유동부채 변화가 분자·분모에 직접 닿는다. 재고 변화 열이 없어 완전한 당좌비율 재구성에는 재고 입력 필요. 현금은 유동자산의 일부라 중복 정보다. |
| R157 | revenue / average(capital_stock_t, capital_stock_t−1) | 매출액 변화가 분자에 직접 닿는다. 자본금은 자본총계와 다른 계정이며 표에 변화 열이 없으므로 기초·t+1 자본금 필요. |
| R182 | noncurrent_assets_t / noncurrent_assets_t−1 − 1 | 비유동자산 열은 없다. 유동자산·부채·자본만으로 다른 부채 및 총자산 구성요소를 모두 복원할 수 없으므로 직접 재계산 불가. |
| cap_change_count_3y | 최근 3년 자본금 변동 횟수 | 단일 1년 표에 자본금/주식발행 이력 열이 없다. 총차입금 변화는 자본금 변동 횟수가 아니다. 기초 3년 이력 필요. |
| log_assets | log(total_assets) | 표에 총자산 또는 비유동자산 변화가 없다. 일부 유동자산·자본 변화는 간접 단서일 뿐 전체 자산 복원에는 전체 부채·자산 정보 필요. |
| operating_loss_freq_3y | 최근 3년 영업손실 빈도 | 영업이익 변화는 한 해 손익의 부호 전환을 추정하는 데만 부분적 단서다. 기초 영업이익과 3년 이력 없이는 빈도를 재계산할 수 없다. |

## 11개 표 열과 직접/간접 연결

판정에서 **직접**은 α·β·γ 선택 변수의 산식에 명시된 계정 또는 분자·분모이고, **부분**은 비슷한/구성 계정이거나 필요한 기초값·다른 계정이 빠진 경우다. **미연결**은 선택 9개 변수의 산식에 직접 등장하지 않는다는 의미다. 재무정보가 일반적인 사업 판단에 쓸모없다는 뜻은 아니다.

| E2 표 열 (1년 변화, A0 대비) | 직접/부분 연결되는 선택변수 | 해석·누락 정보 | 근거 |
|---|---|---|---|
| 매출액 | **직접:** R006 분모, R085 분모, R157 분자 | 변화액과 기초값을 함께 보면 비율의 t+1 값을 유추 가능. | `oracle_variable_contract.py`; Simulator의 projected `revenue` |
| 영업이익 | **부분:** operating_loss_freq_3y의 연간 상태 | 3년 손실 빈도는 아님. 기초값·직전 이력 필요. | feature schema/alpha params; `semantic_action_v4.py` |
| 세전이익 | **직접:** R006 분자 | 매출 변화와 같이 있으면 R006 비율을 유추 가능. | `oracle_variable_contract.py` |
| 이자비용 | **부분:** R085 금융비용의 일부 가능성 | 순수 이자비용이 금융비용 전체와 같지 않음. 기초 금융비용/기타 금융비용 필요. | `FirmState.pure_interest_expense`, R085는 `financial_cost` |
| 총차입금 | **미연결(직접):** 선택 9개 산식에는 총차입금 없음 | 부채 변화는 일반적 위험정보. `cap_change_count_3y`와 혼동 금지. | 선택변수 목록 및 산식 계약 |
| 단기차입금 | **미연결(직접):** 선택 9개 산식에는 단기차입금 없음 | 만기·유동성 관련 간접 정보일 수 있음. | 선택변수 목록 및 산식 계약 |
| 현금및현금성자산 | **부분:** R116의 유동자산 구성요소 | 현금 변화는 유동자산 변화에 중복 포함되고, 재고가 빠져 당좌비율을 완전 복원하지 못함. | `oracle_variable_contract.py`; Simulator `current_assets` composition |
| 유동자산 | **직접:** R116 분자 구성요소 | 당좌비율에는 재고 차감 필요. | `oracle_variable_contract.py` |
| 유동부채 | **직접:** R116 분모 | 기초값 포함 시 비율 변화 유추 가능. | `oracle_variable_contract.py` |
| 자본총계 | **부분:** R064의 총자산과 회계상 연관 가능 | `retained_earnings` 또는 `capital_stock`와 동일 계정이 아니다. 총부채·기초 재무자료 없이 총자산 복원 불가. | `FirmState` 및 `oracle_variable_contract.py` |
| 영업활동현금흐름 | **미연결(직접):** 현재 선택 9개에는 OCF 입력변수 없음 | 현금창출력에 대한 일반 재무정보는 제공하지만, 이번 α·β·γ 선택변수의 산식에 직접 쓰이지 않는다. | α·β·γ 선택변수 metadata |

## 정보 노출 결론

결과표 자체에는 Oracle 점수, 등급, 가중치, 변수 이름, 행동 순위가 없다. 그러나 R006/R085/R116/R157의 계정과 일부 분자·분모가 표에 직접 나타나며, 기업 기초정보와 합치면 특정 비율의 변화를 간접 추정할 수 있다. R064·R182·log_assets·3년 빈도/횟수는 부분적 회계 단서가 될 수 있다. 따라서 “점수·가중치·등급·순위를 직접 제공하지 않는다”는 통제는 말할 수 있지만, “평가정보가 전달되지 않는다” 또는 “결과표가 평가모형과 독립적이다”라고 말할 수 없다. T/RT의 비교는 시뮬레이션 재무정보 패키지의 효과이며, 평가입력과의 중첩 가능성도 처치의 일부다.

## 근거 파일

경로는 `e2_20261118/A_reproduction/E1_score_gate_replay/runroot_zip/THESIS_REPRO_KIT_v2.1.1_FINAL/` 기준:

- `archive/DEPLOYED_RELEASE/stage1_oracle_backends/alpha/oracle_alpha_params.json`
- `archive/DEPLOYED_RELEASE/stage1_oracle_backends/beta/benchmark_beta_params.json`
- `archive/DEPLOYED_RELEASE/stage1_oracle_backends/gamma/benchmark_gamma_params.json`
- `src/credit_recourse/simulator/oracle_variable_contract.py`
- `src/credit_recourse/simulator/oracle_variables.py`
- `src/credit_recourse/simulator/semantic_action_v4.py`
- `src/credit_recourse/simulator/oracle_evaluator.py`

이 문서는 변수·계정 구조 대조를 기록하며 파일럿 Oracle 평가를 실행하거나 그 점수를 열람하지 않는다.
