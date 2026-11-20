# 기술적 개선 사전 점검 스크립트 (2026-11-18, v2)

새 LLM 호출 없이 재현 키트의 동결 관측값만으로 계산한다. 대응 문서: 프로젝트 `claude/기술적개선_아이디어와_사전점검_20261118.md`, `claude/ClaudeCode_연구설계서_E2_v3_20261118.md`.

## 입력 (thesis-reproduction-kit-v43, main b4ed3bf)

- `OBS` = `frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN/frozen_inputs/stage8/canonical_itt_observations.parquet` (96,600행)
- `CON` = `contracts/scientific/v43_action_contract.json` (축별 척도, 표준행동 정의)

## 실행

```
python e27_consistency.py       OBS CON        # 재실행 일치율, 모델 간 일치율, 참조 행동과의 거리
python e28_selfconsistency.py   OBS CON        # 자기일관성 선택 대 강화학습 참조 수정
python e29_below_noop.py        OBS            # 무행동보다 점수가 낮은 권고의 비율
python e30_detectable_effect.py OBS            # 1회 실행과 2회 평균의 표준오차, 검출 가능한 최소 차이
python e31_prep.py              OBS CON OUT    # 권고별 표(행동 단위, 참조와의 겹침, 점수)를 OUT.parquet로 저장
python e33_costmix.py           OUT            # 비용 축만 쓴 권고의 배분, 같은 기업 안에서 매출원가율 비중과 점수
python e34_costmix_noref.py     OUT            # e33을 참조 없는 권고(C4, C5, C4R)로만 다시 계산
```

## 공통 조건

- 추론 높음, 엄격 규칙(규칙 위반 응답은 무행동 처리), 자유 작성. 기본은 IC-b, 총 행동크기 ≤ 1. e31·e33은 정보 조건 셋(IC-a·b·c)과 두 실행을 모두 쓴다.
- 행동 벡터는 `strict_itt_action_json`의 8개 축 값을 축별 척도(`normalization_scale_by_dimension`)로 나눈 행동 단위다.
- 1차 = `replicate` 1(MAIN), 2차 = `replicate` 2(STABILITY, 초안부터 다시 생성).
- 신뢰구간은 기업 단위 부트스트랩 95%, 다중비교 보정 전.

## 정의

- 완전 일치: 8개 축 값이 모두 같음(허용오차 1e-9).
- 행동 거리: 8개 축의 행동 단위 차이 절댓값의 합.
- 자기일관성 선택: 한 기업의 출력들 가운데 다른 출력과의 행동 거리 합이 가장 작은 출력. 동률이면 목록의 앞쪽(1차 초안).
- 검출 가능한 최소 차이: 쌍비교 차이의 표준오차 × 2.8(양측 5%, 검정력 80%의 근사).
- 비용 축만 쓴 권고(e33): 매출원가율·판관비율 두 축의 합이 0.9단위 이상이고 나머지 축의 합이 0.02단위 미만.
- 매출원가율 비중(e33): 매출원가율 단위 ÷ (매출원가율 + 판관비율 단위).

## 쓰지 않는 것

- 기존 권고로 축별 효과를 직선으로 추정한 예측(세션의 e32)은 이미 아는 값을 맞히지 못해 버렸다. 묶음에 넣지 않았다.

## 출력

`e27_output.txt` ~ `e34_output.txt`는 이 세션에서 실행한 결과다(numpy 2.5.3, pandas 3.0.5, pyarrow 26.0.1).
