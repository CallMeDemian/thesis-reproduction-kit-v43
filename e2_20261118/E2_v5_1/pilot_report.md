# E2 v5.1 품질 파일럿 보고

상태: **COMPLETE**. 요청은 100개 기업 × 2개 모델 × N/R/T/RT/Q = 1,000개로 고정했다. 파일럿에서는 Oracle 성능점수를 계산하거나 열람하지 않았다. Q는 정답 채점 없이 JSON 구조만 검사했다.

## 조건·모델별 응답 품질

엄격 파싱은 공식 E1 frozen parser/action contract를 사용한다. 인프라 미완료는 A0로 대체하지 않고 별도 분리한다. usable rate는 전달된 요청과 전체 요청 기준을 같이 보인다.

| 모델 | 조건 | 요청 | 전달 완료 | 엄격 사용 가능 | 전체 기준 | 완료 응답 기준 | 무효/실패 원인 |
|---|---|---:|---:|---:|---:|---:|---|
| google | N | 100 | 100 | 98 | 98.0% | 98.0% | refusal=0, output_limit=0, empty=0, infra=0, model_mismatch=0 |
| google | R | 100 | 100 | 99 | 99.0% | 99.0% | refusal=0, output_limit=0, empty=0, infra=0, model_mismatch=0 |
| google | RT | 100 | 100 | 99 | 99.0% | 99.0% | refusal=0, output_limit=0, empty=0, infra=0, model_mismatch=0 |
| google | T | 100 | 100 | 99 | 99.0% | 99.0% | refusal=0, output_limit=0, empty=0, infra=0, model_mismatch=0 |
| openai | N | 100 | 100 | 98 | 98.0% | 98.0% | refusal=0, output_limit=0, empty=0, infra=0, model_mismatch=0 |
| openai | R | 100 | 100 | 99 | 99.0% | 99.0% | refusal=0, output_limit=0, empty=0, infra=0, model_mismatch=0 |
| openai | RT | 100 | 100 | 99 | 99.0% | 99.0% | refusal=0, output_limit=0, empty=0, infra=0, model_mismatch=0 |
| openai | T | 100 | 100 | 99 | 99.0% | 99.0% | refusal=0, output_limit=0, empty=0, infra=0, model_mismatch=0 |

각 모델×조건의 strict usable 비율이 95% 미만이면 운영상 경보로 기록한다. 이 기준은 연구 가설 검정이 아니다.

## Q 응답 구조

- 전달된 Q 응답: 200/200; 완전 JSON schema 유효: 199/200; output-limit=1, refusal=0, 비어 있는 완료 응답=0.
- output-limit 세부: e2v51-pilot-openai-Q-066 (incomplete/max_output_tokens, reasoning tokens=32,768). 해당 응답은 상한까지 추론 토큰을 사용하고 visible JSON을 반환하지 못했다. 인프라 재시도 대상으로 처리하지 않았다.
- 검사 항목: forecasts 배열 개수, candidate ID/순서 고정, 다섯 metric별 direction/finite signed amount, 허용 키만 확인. Simulator 예측오차는 계산하지 않았다.
- direction/부호 내부 불일치: 74개 metric 값, 이를 포함한 응답 14개. OpenAI 감소 표시에 음수가 아닌 값 72개, Google no-change 표시에 0이 아닌 값 2개. 이는 출력 내부 일관성 점검이며 Simulator 정답 대비 정확도는 아니다.

## Simulator 결과표 QA

- 선택 표본의 200개 결과표·2000행 모두 action-valid 및 Simulator 실행 성공: True/True; 단위 `KRW_thousand`, 시점 `decision_year_plus_one`.
- 정규화 행동예산 최대 T=1.000000000041; 허용오차 초과 건수 0.
- 최대 재무상태 항등식 잔차 1.52587891e-05 KRW thousand (허용 10); 결측 재무항목 최대 0.
- 결과표 상태는 0효과와 무행동 기준행을 분리했다: `{'nonzero_financial_effect': 1525, 'zero_financial_effect': 275, 'A0_baseline_zero_change': 200}`.

## 초안 예외 및 요청 통제

- 원본 C4 초안 2,300건 중 47건은 기존 STRICT ITT 처리에 따라 A0로 고정했다. 파일럿 표본에는 N 기준 모델-기업 초안 예외 3개가 있으며, 같은 원문 초안은 네 수정 조건에 공통 유지되고 Simulator 표의 초안 행에서만 A0를 사용했다.
- T/RT의 10행 표는 제출 전 문자열 동일성 검사를 통과했다. R/RT 정책 참조는 E1와 같은 JSON 블록이다. Q는 C4 원문·RL 참조·표를 받지 않는다.
- 응답별 반환 모델 ID를 요청 ID와 비교했다. 파일럿은 Oracle 점수 산출 및 조회 없이 처리했다.
- 정책 조건 무효 10건의 원인: OpenAI JSON 파싱 실패 5건, Google 행동 강도 상한/전체 응답 계약 실패 5건. 모두 모델 응답 품질이며 인프라 실패는 0건이다.
- 정책 조건 STRICT ITT 무효 응답 10건은 8개 행동축이 모두 0인 A0로 처리됨을 확인했다: PASS.

## 호출 수와 사용량 기반 비용 추정

| 공급자 | 요청 | 전달 완료 | 입력 토큰 | 출력 토큰(추론 포함) | 게시 Batch 단가 기준 비용 추정 |
|---|---:|---:|---:|---:|---:|
| openai | 500 | 500 | 2,505,782 | 4,217,232 | USD 10.4284 |
| google | 500 | 500 | 3,133,771 | 1,797,212 | USD 1.7396 |
| 합계 | 1000 | 1000 | 5,639,553 | 6,014,444 | USD 12.1681 |

비용은 응답 usage token과 사전 기록한 유료 Batch 단가로 계산한 추정치이며 청구서 금액이 아니다. 재시도는 initial attempt와 분리해 기록한다.
사전 token 추정과 비교하면 실사용 단가 추정 USD 12.1681는 point estimate USD 6.2780의 1.94배, historical p90 USD 11.1567의 1.09배다. 최악 출력상한 추정 USD 50.6584보다는 낮다.

## 파일럿 결정

- 핵심 2×2 정책 조건 N/R/T/RT는 모델×조건별 최소 strict usable rate 98.0%, 모델 ID 불일치 0, 인프라 실패 0으로 운영 품질 기준 95%를 통과했다: PASS.
- Q는 보조 진단으로 분리한다. JSON 구조 유효 199/200이며, 1건은 max-output token을 추론에 모두 사용해 응답을 내지 못했고 14개 응답에 내부 방향/부호 불일치가 있었다. 예측 정확도는 채점하지 않았으므로 이 파일럿에서 Q 품질을 확정하지 않는다.
- 파일럿은 점수 열람 전의 운영·형식 점검이며, E2의 성능 효과는 평가하지 않았다. 파일럿 응답은 본실험에 재사용하지 않으며, 본실험은 별도 승인 전까지 실행하지 않는다.

## 감사 산출물

- API 작업/원본 해시: logs/provider_jobs.json, logs/response_hashes.json; 요청/프로토콜 제출 전 hash: logs/pre_call_hashes_at_submission.csv 및 logs/provider_jobs.json의 manifest SHA-256.
- 모델·SDK/요금 추정: model_preflight.json, requirements_api_lock.txt, preflight_cost_estimate.json; 분석 기준 해시: analysis_freeze_hashes.csv.
- 결과표/평가정보 경계: ORACLE_SEAM_CROSSWALK.md; S1/S2 향후 평가 전용 구현 계획: STRESS_IMPLEMENTATION.md. 파일럿에서는 민감도 재채점하지 않았다.

완료 기록: 2026-11-18T15:37:02.301362+00:00.
