# 기존 Release 대비 변경

`thesis-v43-reproduction-kit-v1.1.0`은 v1.0.4의 문서 patch 이후 처음으로 후속 연구 증거를 통합하는 minor release다. 기존 v1.0.0–v1.0.4와 Ablation/Human baseline Release의 tag·본문·asset은 수정하지 않는다.

- 새 통합 기술보고서, 전체 분석 inventory, 원자료–프롬프트–응답–평가–집계–보고서 연결표, 파일별 SHA-256 catalog, 재현 안내, R6 부록, 한계 문서 추가.
- R1–R9 후속 검토 및 7평가기/606격자/선택 계열 일반화 자료·코드·저장 로그 추가.
- 행동 수용/수정 v1·v2, 동일 참조 141기업 음성대조, 가치 전달/분해, 재무상태 연관 분석 추가.
- RL 내부 신호 게이트와 RL–LLM 상호작용 게이트의 OOF 부정적·불확실한 결과 추가.
- P0–P5, P3 선택적 보완, 고정 강도 비용배분 반사실, 10행 Simulator 표/회계 QA 추가.
- E2 1,000건 파일럿, 10,350행 최종 실행 매니페스트, 과거 신규 생성 5,150건의 원요청/원응답, N/R 재사용 원장 연결, 정책/Q 파싱·평가·구성/배정·P3·민감도·비용 로그 추가.
- 목록에 빠져 있던 e1–e26 세부 탐색 구현, e27–e34 진단, e29 harm/e30 검출가능 차이, 11/16 방법 타당성 검토/브레인스토밍 보존. 결과 미보존 항목은 코드만 있다고 표시.
- 기존 certified ZIP, Ablation ZIP, Human baseline source list, pinned public Git/LFS와 동일한 대용량 자료는 재첨부하지 않고 버전/asset/member/path/체크섬으로 연결.

본래 논문/발표자료·동결 결과·운영 scientific runtime을 수정하지 않았다. 새 실행 스크립트는 패키지 무결성 검사/복원용이며 연구를 재실행하지 않는다. v1.0.4부터 기존 main b4ed3bf까지 이미 존재하던 변경 이력은 BASELINE_COMMITS.json에 별도 수록한다. 이전 v1.0.4의 설명을 현재 post-study 결과로 소급 교체하지 않는다.
