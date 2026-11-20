# 원자료 보존 범위

이번 Release는 1,412개 분석 관련 경로를 catalog로 고정했다. 새 고유 파일 459개, 디렉터리 중복 alias 62개, 기존 공개 Release ZIP member 697개, pinned 공개 Git/LFS 194개다. 원래 검토한 1,994개 경로 중 582개는 post-study 분석이 읽지 않는 과거 Stage2 학습/투영 감사 중복본으로, 기존 archive에 그대로 두고 새로 배포하지 않았다. 라이선스 upstream 원천자료나 새 학습 데이터를 추가한 것이 아니다.

실험 원요청·원응답·기업별 파싱/평가 데이터는 사용자 지시에 따라 공개 첨부했다. 기존 locked N/R 원장은 공개 baseline Git LFS 객체로 연결한다. 새 연구자료의 필수 의존성은 모두 공개 Git/Release로 연결되며 비공개 동반 Release는 필요하지 않다.

논문/발표 원본 및 추출 텍스트, 가상환경·캐시·Git object 복사본은 제외했다. 역사적 학습 checkpoint와 source 데이터 중복은 재첨부하지 않았다. 기존 v1.0.4·Ablation·Human baseline은 변경하지 않고 asset 이름/전체 체크섬을 기록했다.

FILE_CATALOG의 path는 복원할 작업공간 상대경로이며, source_path/사용자 머신 경로는 새 catalog에서 제거했다. 원문 내부의 과거 절대경로는 무결성을 위해 보존한다. scientific path catalog는 자기 자신이나 새 편집 보고서의 해시를 자기참조하지 않으며, 편집 보고서와 ZIP 전체의 해시는 RELEASE_MANIFEST 및 SHA256SUMS로 고정한다.
