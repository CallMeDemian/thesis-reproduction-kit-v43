# 알려진 한계와 미해결 자료

- **R6 부분 미재현:** 과거 기대표의 Gemini best-of-2 0.864에 대해 보존된 e20은 0.880이다. e18의 4/8 결과는 19개 출력 풀의 200회 부분집합이라는 다른 정의다. 과거 후보집합·seed·선택 원장이 없어 원인 확정 불가. 원 기대표·코드·복구 로그를 모두 보존했다.
- **C2/BC 정체:** Stage4가 BC인 것은 코드로 확인되나 저장 C2 행동을 575기업의 BC checkpoint 출력과 잇는 crosswalk가 없다. E2-B는 실행되지 않았고 연구 성과로 세지 않는다.
- **코드만 보존된 연구:** INVENTORY의 `CODE_ONLY_NO_RETAINED_NUMERIC_OUTPUT`는 구현이 있으나 해당 수치·stdout 또는 회귀 결과가 없다. 이들에 대해 성공/실패를 만들어내지 않았다. e32는 원 패키지에서 폐기되어 없다.
- **실제 성과·일반화:** 동일 575기업/FY2024를 반복 사용했다. C3-E 선택에 α가 쓰였고 δ·ε·P3도 후속 탐색이다. 독립 연도·기업의 정책 실행효과나 실제 등급 개선은 입증되지 않는다. e26의 실제 등급 관련 구현도 관찰 분석이다.
- **E2 비교 경계:** 과거 N/R과 신규 T/RT는 TASK 문구와 생성 시점이 다르다. 전체 표본은 패키지 비교다. 100기업 동시기 파일럿은 보조이고 일부가 본실험에 편입되어 독립 표본이 아니다. 토큰 길이/정보 내용 효과를 구분하지 않았다.
- **부정적 결과:** E2 α H1, 양의 정보 상호작용, 적응형 참조 게이트의 안정적 개선, 일반적인 LLM–RL 통합 우월성은 지지되지 않는다. 비유의는 동등성의 증거가 아니다.
- **평가 점수의 해상도:** 행동이 달라도 α 증분이 0인 행이 많다(수용/게이트 Run1 분석 약 88%). β·γ를 함께 읽는다. simulated score를 확률로 해석하지 않는다.
- **구성·배정:** 성과분해는 인과적 역할 배분이 아니다. E2 C/A CI는 100개 공통 derangement를 사용한 근사 기업 bootstrap 구간이며 R8/B1과 재배정 정의가 완전히 같지 않다.
- **RL 내부신호:** V(s)가 없어 Advantage 분석을 수행하지 않았다. 원 학습 search catalog와 정확한 selection-rule 자료, raw upstream Stage2 producer input은 과거부터 남은 외부 경계다. 동결 결과를 읽는 것과 raw-data fresh 학습 재현은 다르다.
- **환경·경로:** Python 3.11.15/scikit-learn 1.6.1의 R2–R8 harness와 Python 3.12/scikit-learn 1.8의 별도 게이트 runtime이 존재한다. 혼합 환경의 joblib 모델을 임의로 다시 직렬화하지 않는다. 일부 원 코드(e26 등)에 `/home/claude/...` 절대경로가 남아 있고 portable rerun을 검증하지 않았다. 원 코드의 경로 이동은 별도 작업 복사본에서 해야 한다.
- **과거 문서의 상태:** 초기 source audit와 pilot report에는 당시 미확보·재사용 금지·대기 문구가 남아 있다. E2_MAIN_REUSE_PROTOCOL과 최종 metadata가 최신 실행 상태다. 원문을 덮어쓰지 않고 문서 우선순위를 기록했다. 원 E2_FINAL_REPORT의 로컬 이미지 경로는 그대로 보존했으며 그림 자체는 results에 있다.
- **재배포 범위:** 사용자 지시에 따라 실험 원요청·원응답·기업별 결과를 포함한다. 공급자 암호화 응답 필드도 원 바이트로 보존했다. API credential은 포함하지 않으며 라이선스 원천 데이터·제3자 문서 원문·논문/발표 원본·가상환경·checkpoint 중복본은 새로 배포하지 않는다. 기존 공개 LFS와 이전 Release 의존성은 checksum으로 추적한다.

이번 검증은 원자료 해시·요청/응답 ID·집계 산술을 확인했다. 전체 fresh pipeline, 추가 API 호출, RL 학습, Simulator 재실행, bootstrap 재실행은 하지 않았다. 따라서 이 Release는 현재까지 보존된 증거의 동결 기준점이지 누락된 모든 upstream 원천의 완전 복구 선언은 아니다.

## 수치 출력이 발견되지 않은 구현

`STRUCT_e11`, `STRUCT_e13`, `STRUCT_e19`, `STRUCT_e19b`, `STRUCT_e24`, `STRUCT_e25`, `STRUCT_e26_ach`, `STRUCT_e26_agree`, `STRUCT_e26_base`, `STRUCT_e26_dec`, `STRUCT_e26_out`, `STRUCT_e26_out2`, `STRUCT_e26_out3`, `STRUCT_e26_out4`, `STRUCT_e26_unc`, `STRUCT_e3`, `STRUCT_e4b`, `STRUCT_e7_stress`, `STRUCT_e8`, `STRUCT_fig_gap`, `STRUCT_fig_summary`, `STRUCT_t3`. 코드·설명은 보존했으나 이 항목의 수치 결과는 이번 동결에서 추가 생성하지 않았다.
