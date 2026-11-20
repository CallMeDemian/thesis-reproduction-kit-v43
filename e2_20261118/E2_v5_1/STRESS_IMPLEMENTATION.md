# S1/S2 Simulator 가정 민감도 구현 계획

상태: 코드 경로와 산식을 확인했으며, 파일럿에서는 점수 재계산을 하지 않는다. 이 문서는 향후 본실험 응답의 **고정된 행동 벡터**를 사후 재채점할 때 적용할 별도 평가 계획이다. 기존 Simulator 원본은 수정하지 않는다.

## 기준 입력

- 고정 action: E2-N/R/T/RT에서 파일럿 또는 승인된 본실험이 생성한 STRICT ITT 행동, 그리고 동결된 P0–P5. 민감도 조건에서 정책·행동·표·프롬프트를 다시 만들지 않는다.
- S0: 현재 동결 FinancialSimulator 및 각 정책 행동과 일치하는 A0.
- 각 시나리오에서 A0도 같은 시나리오 설정으로 다시 실행하고 `Oracle_k(Simulator_s(i,a)) − Oracle_k(Simulator_s(i,A0))`를 계산한다.
- α·β·γ 채점은 고정된 동결 scorer를 사용한다. 결과를 LLM 프롬프트 또는 정책 선택으로 되돌리지 않는다.

## S1 — 장기 전환 가산금리 +1%p

11/17 `analysis_20261117/e5_stress.py`의 실제 식은 `_interest(bp, **kw)`의 원래 결과에 `extra = 0.01 × (new_long_tranche + new_bond_tranche) × 1.0`을 더하는 것이다. 이자비용 합계 및 상세분해 `rf_term_premium`에 더하며, 해당 연도의 모형상 세후이익·현금/재무제표 전달 경로를 거친다. 전환된 원금만 대상으로 하고 전체 차입금에 가산하지 않는다. 동결 Simulator의 원래 새 tranche 금리 비용 `0.5 × new_tranche × rate`는 유지하고, 가산분만 1년 기준으로 적용한다. 현재 고정 모델에서 `new_bond_tranche` 비중은 0이므로 효과는 새 장기 tranche에만 발생한다.

구현은 시나리오 전용 subclass/wrapper 또는 재현 가능한 monkey-patch harness로 격리하고, 동결 원 함수에 저장된 pointer를 호출한 뒤 `rf_term_premium` 항목을 덧붙인다. S0 결과는 동결 출력과 허용오차 내 일치해야 한다. S1은 각 입력 action의 새 tranche가 0일 때 S0과 동일하고, 새 long tranche가 양수일 때만 `0.01 × principal` 차이가 나는지 단위 사례로 검사한다. 이 검사는 Simulator/accounting QA이며 정책 결과를 이용한 프롬프트 조정이 아니다.

## S2 — 비용절감 실현율 50%

11/17 `e5_stress.py`의 `scaled(a, f)`는 `cogs_ratio_chg`와 `sga_ratio_chg`의 양·음 모든 변화량을 action 단계에서 0.5배로 바꾼다. 이것은 “음수 비용률 변화가 나타내는 비용절감만 절반 실현”이라는 본 실험의 민감도와 다르므로 그대로 재사용하지 않는다.

별도 평가 wrapper는 action vector와 정규화 행동예산을 원래 값으로 보존하고, 후보 BusinessPlan 구성에서 각 비용률 목표를 다음처럼 만든다:

`effective_target = baseline + (target − baseline) × 0.5`, 단 `target < baseline`인 비용절감 변화에만 적용. `target ≥ baseline`은 기존 목표를 유지한다.

COGS와 SGA 각각에 적용하고 나머지 축·제약은 건드리지 않는다. 조정된 비용률을 이용해 영업이익·세전이익·이자·세금·현금흐름 및 재무상태를 Simulator의 정상 downstream 회계 흐름으로 다시 계산한다. 표의 결과를 사후에 임의 비율로 줄이는 방식은 쓰지 않는다. action vector hash는 S0/S2 간 같아야 하며, 별도로 적용된 effective COGS/SGA rate와 비용절감액은 감사 로그에 남긴다.

QA: (1) 비용축을 사용하지 않는 action에서는 S2=S0, (2) 음수 COGS/SGA 변화는 기준 대비 목표 변화의 절반, (3) 양수 변화와 다른 6축은 그대로, (4) `T`와 제약/행동 가능성은 원 action으로 유지, (5) 계산된 재무제표가 원 회계 항등식 허용오차를 통과하는지 검증한다. 원본 Simulator 파일은 수정하지 않는다.

## 비교와 해석

S0/S1/S2에서 동일한 행동만 재평가하고, 시나리오별 기준 A0를 다시 뺀다. 주가설은 S0 α RT−R로 유지한다. 민감도는 α·β·γ의 RT−R, RT−P0, RT−P2, RT−P3와 기준 S0 대비 변화 및 기업 단위 paired bootstrap CI를 보고한다. S1/S2에서 개선이 감소하면 그 재무 가정에 대한 민감성을 뜻한다. 단독으로 Simulator 허점 악용 또는 실제 기업성과 실패를 뜻하지 않는다.

## 동결 구현 근거

- `e2_20261118/A_reproduction/R2_R8_replay_workspace/analysis_20261117/e5_stress.py`
- 동결 Simulator: `e2_20261118/A_reproduction/E1_score_gate_replay/runroot_zip/THESIS_REPRO_KIT_v2.1.1_FINAL/src/credit_recourse/simulator/semantic_action_v4.py`

기존 S2 계산을 그대로 인용하거나 기존 e5의 수치를 v5.1 민감도 결과라고 보고하지 않는다. 응답 생성 파일럿에는 이 스트레스 점수 계산이 포함되지 않는다.
