# 재현 부록 — R6 조건·복구 기록과 상태 우선순위

## R6의 기술적 정의와 남은 차이

원본 `e18_gen.py/e18_an.py`는 HIGH 10개+LOW 9개의 no-advice 저장 출력 풀(19개)에서 2/4/8개를 200회 추출하고 δ·ε 계열로 선택한다. 원본 `e20.py`는 고정 후보의 best-of-2/best-of-10과 RL 참조 수정의 같은 호출 예산 비교다. 서로 다른 후보집합의 곡선을 동일 설계로 합치지 않는다.

| 모델 | 과거 기대표 1/2/4/8/10 | e20 2/10 | e18 19-pool 2/4/8 |
|---|---|---|---|
| GPT | .785/.843/.883/.914/.918 | .843/.918 | .823/.884/.935 |
| Gemini | .805/.864/.904/.940/.955 | .880/.955 | .839/.879/.918 |

Gemini e20 2-call은 기대표보다 +.016이다. 기대표를 만든 당시 후보 ID·선택 로그·seed·행동별 점수 원장을 확보하지 못해 원인을 확정할 수 없다. 기대값을 맞추기 위해 코드·동결 데이터를 수정하지 않았다. 이 불일치는 E2 평가와 분리한다.

경로: `e2_20261118/A_reproduction/R2_R8_replay_workspace/analysis_20261117/{e18_gen.py,e18_an.py,e20.py}`, `logs/replay/{e18_an_restore_20261119.txt,e20_restore_20261119.txt,r6_restore_manifest.json}`. 원 E2 부록 `E2_v5_1/E2_REPRODUCTION_APPENDIX_R6.md`도 보존한다.

## 복구·오류·환경 기록

11/17 ZIP을 확보하기 전의 미확보 판정은 이후 source audit 보완과 R2–R8 재현 결과가 대체한다. 14개 저장 replay 로그는 모두 종료코드 0이며, e17의 357개 numeric mean/CI 셀은 묶음 JSON과 일치한다. 이는 성공한 항목의 상태이며 R6 기대표 차이를 해소했다는 뜻은 아니다.

별도 게이트 runtime의 초기 Gamma 재채점은 Python 3.14 환경에서 private `_loss` 모듈을 불러오지 못해 중단됐다. 보존된 기록은 Python 3.12/scikit-learn 1.8에서 이어서 완료했다고 명시한다. 게이트의 입력 Stage8 hash `519ef1a1aca7620905ba0a6c38bae69315b741e0566e0591a42f3fdb61f8dd74`와 canonical ITT hash `aedfadbf1728a14d3e4cb45bfa8a022d2c10313c52606e7bd1aee68899de8a0d`를 구분한다. 분석 runtime의 stale registry tag 정규화는 원 보고서에 기록되어 있으며 동결 원본을 교체하지 않는다.

## 문서 우선순위

1. 이번 통합보고서·inventory·validation은 동결 시점의 정리다.
2. E2_MAIN_REUSE_PROTOCOL은 최종 N/R 재사용 및 pilot 편입 규칙이다.
3. E2_FINAL_REPORT, main_analysis_metadata, main_response_integrity는 완료 상태다.
4. 원 v3/v4/v5/v5.1 설계, Gate1 audit, 초기 pilot report의 대기·금지·미확보 문구는 해당 시점 기록으로 남긴다.

R1–R9별 최신 판정과 상세 로그는 E2_REPRODUCTION_REPORT에 있고, 원본 설계와 당시 보고서는 수정하지 않았다. 원 보고서의 로컬 절대경로 이미지는 `results/*.png`에서 확인한다. 새 통합보고서의 수치는 source table로 직접 추적한다.
