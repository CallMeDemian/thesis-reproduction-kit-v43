# V4.3 Experiment Summary

## A Validation-First Evaluation Framework for LLM-Generated Corporate Credit-Rating Recourse

이 문서는 V4.3 실험의 전체 구조와 주요 결과를 빠르게 볼 수 있도록 정리한 technical note입니다.  
재현 명령이나 저장소 구조보다 **무슨 질문을 했고, 어떤 조건을 비교했고, 결과가 어떻게 나왔는지**에 초점을 맞췄습니다.

전체 수치와 세부 검정은 논문 본문, `THESIS_TABLE_CONTRACT.csv`, Stage8/Stage9 evidence에서 확인할 수 있습니다.

---

## 1. 무엇을 연구했나

기업의 현재 재무상태를 보고 LLM이 “향후 1년 동안 어떤 재무행동을 취하면 신용상태 개선에 도움이 될지” 권고한다고 가정합니다.

문제는 그 권고를 실제 기업이 실행한 뒤의 결과를 바로 관찰할 수 없다는 점입니다.  
그래서 이 연구는 먼저 재무행동을 공통 계산환경에서 비교할 수 있는 **평가 하네스**를 만들었습니다.

```text
기업의 현재 재무상태
        ↓
재무행동 제안
        ↓
Simulator
(행동을 적용한 1년 후 재무상태)
        ↓
Oracle α / β / γ
(신용평가 점수)
        ↓
정책가치 비교
```

그 위에서 LLM의 생성절차를 바꿔가며 비교했습니다.

```text
기업 재무정보
   │
   ├─ C4   직접 생성
   ├─ C5   진단 scaffold를 넣은 최초 생성
   │
   └─ C4 raw draft
        ├─ C4R    같은 초안을 reference 없이 재검토
        ├─ C6-E   같은 초안 + 해당 기업의 C3-E reference
        └─ C6-EX  같은 초안 + fixed derangement reference
                         ↓
               동일 Simulator / Oracle
                         ↓
                   정책가치 비교
```

C4R, C6-E, C6-EX는 서로 순차적으로 이어지는 조건이 아니라 **같은 C4 초안에서 각각 분기**합니다.

연구의 중심은 “어떤 LLM이 가장 좋은가”보다, **같은 foundation model이라도 생성방식, 행동공간, 행동예산, 참조정보, generation setting과 실행규칙을 바꾸면 정책가치가 어떻게 달라지는가**에 있습니다.

---

## 2. 연구질문

최종 실험은 RQ1–RQ5의 다섯 축으로 정리했습니다.

| 연구질문 | 핵심 내용 |
|---|---|
| **RQ1. 평가기반** | Oracle 점수와 실제 신용등급의 정렬성, Simulator의 회계적 일관성을 어느 범위까지 정책비교에 사용할 수 있는가 |
| **RQ2. 행동가치·상대순위** | 학습된 RL 참조정책 C3-E와 비교정책의 상대적인 정책가치는 어떠한가 |
| **RQ3. 생성·수정·참조** | 직접생성, 진단 유도, 자기 재검토, 학습참조, 참조의 기업별 연결이 정책가치를 어떻게 바꾸는가 |
| **RQ4. 강건성·행동공간** | Oracle, 행동예산, candidate9/free8에 따라 주요 결과가 어떻게 달라지는가 |
| **RQ5. 정보조건** | 업종코드와 기업명을 추가했을 때 직접생성 결과가 어떻게 달라지는가 |

RQ1은 이후 정책가치를 읽는 기준을 정합니다.  
Oracle과 Simulator는 공통 계산환경에서 행동을 비교하는 도구이고, 실제 기업의 집행효과는 별도의 검증문제로 남습니다.

---

## 3. 행동공간

최종 V4.3의 자유작성 행동은 8차원입니다.

1. CAPEX 축소
2. 총부채 감축
3. 단기차입 리파이낸싱
4. 재고회전 변화
5. 매출채권회전 변화
6. 매입채무회전 변화
7. 매출원가율 변화
8. 판매관리비율 변화

별도로 9개 표준 후보행동을 둡니다.

`A0, DL, RF, CX, WC1, WC2, OE, MX1, MX2`

- **candidate9**: 9개 후보 중 하나를 선택
- **free8**: 8차원 연속 행동을 직접 작성
- **B1**: 정규화된 총 행동강도에 상한 적용
- **BINF**: 각 축의 허용범위는 유지하되 총 행동강도 상한은 없음

이 설계로 “선택지만 주는 방식”과 “행동 자체를 구성하게 하는 방식”을 같은 기업과 평가환경에서 비교할 수 있습니다.

---

## 4. RL 참조정책 C3-E

LLM에 외부 참조를 주기 전에 기업 재무·신용평가 도메인 데이터에서 Offline RL 정책을 학습했습니다.

```text
Stage 3  SSL Encoder
          ↓
Stage 4  Behavior Cloning
          ↓
Stage 5  Implicit Q-Learning
          ↓
28 actor policies
          ↓
C3-E ensemble reference
```

최종 C3-E는 **E2 encoder**와 네 configuration을 사용합니다.

- B27
- M2_S0935
- DT06
- T15_REWARD_S088

각 configuration은 동일한 7개 seed를 갖습니다.

`2, 11, 12, 13, 14, 15, 16`

총 **4 × 7 = 28 actors**입니다.

결합은 두 단계입니다.

1. configuration 내부의 7개 seed actor 확률을 평균
2. 네 configuration을 각각 25%로 다시 평균

### C3-E 결과

Oracle-α 기준 최종 C3-E의 기술통계는 다음과 같습니다.

| 지표 | 값 |
|---|---:|
| 평균 절대점수 | **55.959960** |
| A0 대비 | **+0.923299** |
| OE 대비 | **+0.157936** |

C3-E는 평균적으로 강한 도메인 참조정책으로 사용했습니다. 동시에 evaluator별 차이와 기업별 손실사례가 존재하므로, LLM 실험에서는 “정답”이 아니라 **학습된 외부 행동참조**의 역할을 합니다.

---

## 5. LLM 실험설계

사용한 foundation model은 두 개입니다.

- **GPT-5.4-mini**
- **Gemini 3.1 Flash-Lite**

실험은 **Baseline**을 본실험으로 먼저 수행하고, 이후 같은 기본 행렬을 **High generation regime**으로 한 번 더 실행했습니다.

High에서는 reasoning setting을 높이고 output cap도 4,096에서 32,768로 늘렸습니다.  
따라서 High는 단일 파라미터 하나만 바꾼 실험이라기보다 **더 강한 생성설정 전체를 적용한 추가 캠페인**으로 보는 편이 맞습니다.

한 model, 한 generation regime은 21개 cell입니다.

```text
21 cells × 575 firms = 12,075 requests
```

두 model을 합치면 한 regime에서:

```text
42 cells × 575 firms = 24,150 requests
```

Baseline과 High를 합친 최종 generation inventory는:

```text
24,150 + 24,150 = 48,300 logical requests
```

### 5.1 정책조건

| 조건 | 무엇을 하는가 | 비교하려는 효과 |
|---|---|---|
| **C4** | 기업정보만 보고 처음부터 행동 생성 | 직접생성 기준 |
| **C5** | C4와 같은 최초 생성이지만 진단 scaffold 추가 | 진단 유도의 증분 |
| **C4R** | 같은 run의 C4 원문을 reference 없이 재검토 | 자기 재검토 |
| **C6-E** | 같은 C4 원문 + 해당 기업의 C3-E 참조행동 | 학습참조를 보여준 수정 |
| **C6-EX** | 같은 C4 원문 + C3-E 참조를 기업 간 고정 재배정 | 기업–참조 연결의 역할 |

C6-E와 C6-EX에서는 reference의 출처명, Q값, Oracle 점수 등을 보여주지 않고 **8차원 행동 벡터만** 제시했습니다.

C6-EX는 575개 reference를 새로 무작위 생성하는 조건이 아닙니다.  
C3-E의 전체 reference 분포를 그대로 유지한 채, 어느 기업이 어떤 reference를 받는지만 고정 derangement로 바꿨습니다.

출처 기업은 모두 달라지지만 같은 action ID를 선택한 기업이 있기 때문에 **141/575 기업은 재배정 전후 받은 action ID가 같았습니다.** 이 배정은 공급자와 예산조건에 걸쳐 고정했습니다.

### 5.2 핵심 비교

네 contrast를 중심으로 봅니다.

1. **C5 − C4**: 진단 scaffold를 넣은 최초 생성과 직접생성의 차이
2. **C4R − C4**: 같은 초안을 한 번 더 검토했을 때의 차이
3. **C6-E − C4R**: 같은 C4 초안을 reference 없이 검토한 경우와 C3-E reference를 보며 수정한 경우의 차이
4. **C6-E − C6-EX**: 원래 기업–reference 연결과 fixed derangement 연결의 차이

특히 3번은 “C4R을 수행한 뒤 reference를 추가했다”는 순차효과가 아니라, **동일한 C4 parent에서 분기한 두 수정조건의 차이**입니다.

### 5.3 실행행렬

한 model, 한 generation regime에서:

| 블록 | 조건 | cells |
|---|---|---:|
| Core B1 | free8 / IC-b / B1, C4·C5·C4R·C6-E·C6-EX | 5 |
| Core BINF | free8 / IC-b / BINF, 동일 5조건 | 5 |
| Action-space | candidate9 / IC-b / B1, C4·C4R·C6-E | 3 |
| Information | free8 / B1, IC-a·IC-c의 C4 | 2 |
| Stability | free8 / IC-b / B1·BINF, C4·C4R·C6-E 재실행 | 6 |
| **합계** |  | **21** |

Stability Run2의 C4R과 C6-E도 Run1 C4를 재사용하지 않고 **자기 Run2의 C4에서 분기**합니다.

---

## 6. 정보조건

실제 prompt에 들어간 정보는 다음처럼 구성했습니다.

- **IC-a**: 23개 재무특징 + `fiscal_year` + `log_assets` + `market` = **26 fields**
- **IC-b**: IC-a + `industry_class` = **27 fields**
- **IC-c**: IC-b + `firm_name` = **28 fields**

따라서 정보조건 비교는 다음 두 가지입니다.

- **IC-b − IC-a**: 업종코드 추가
- **IC-c − IC-b**: 기업명 추가

Primary/Main 분석은 IC-b를 사용하고, 정보조건 비교는 Baseline C4/free8/B1에서 따로 봤습니다.

---

## 7. 평가와 통계

정책가치는 575개 동일 기업을 기준으로 paired comparison합니다.

Frozen statistical registry의 primary layer는:

```text
2 models
× 3 Oracles (α, β, γ)
× 2 generation regimes (Baseline, High)
× 2 budgets (B1, BINF)
× 4 contrasts
= 96 rows
```

Primary population은 **Strict ITT / free8 / IC-b**입니다.

불확실성 평가는 10,000회 bootstrap을 사용했고, 각 model·Oracle family에서 네 contrast와 두 budget을 함께 Holm 보정했습니다.

최종 result registry:

| 구분 | rows |
|---|---:|
| Primary | **96** |
| Supplemental | **722** |
| Parent-gate descriptive | **96** |
| **전체** | **914** |

Historical Stage8 canonical evidence는 **96,600 rows**입니다.

논문의 서술 순서는 조금 다릅니다.  
**Baseline Run1을 본실험으로 두고, High·Repaired·PP·Stability를 추가 분석으로 이어가는 구조**입니다. Registry의 96 primary rows는 Baseline과 High의 같은 contrast 체계를 한데 보존한 통계 레이어입니다.

---

# 8. 핵심 결과

## 8.1 네 contrast를 전체 registry에서 한눈에 보면

Baseline과 High를 함께 묶은 96개 primary row를 기술적으로 요약하면:

| Contrast | 양의 점추정 | Holm p < .05 | 전체 패턴 |
|---|---:|---:|---|
| **C4R − C4** | 18 / 24 | 11 / 24 | 자기 재검토는 자주 양수지만 모델·예산에 따라 차이가 큼 |
| **C5 − C4** | 15 / 24 | 5 / 24 | 진단 scaffold의 효과는 가장 혼합적 |
| **C6-E − C4R** | **24 / 24** | 9 / 24 | 네 contrast 중 방향이 가장 일관됨 |
| **C6-E − C6-EX** | **23 / 24** | 2 / 24 | 원래 기업–reference 연결이 대체로 양의 방향 |

이 표는 **Baseline 본실험과 High 추가실험을 한눈에 보는 descriptive summary**입니다.

가장 안정적인 방향성은 C6-E − C4R에서 나타났습니다.  
foundation model, Oracle, generation regime, budget을 바꿔도 24개 cell 모두 점추정이 양수였습니다.

C6-E − C6-EX도 23/24가 양수였습니다. 기업에 맞는 reference 연결이 대체로 더 높은 값을 보였지만, C6-EX가 하나의 fixed derangement라는 점과 141개 기업의 action-ID collision을 함께 봐야 합니다.

---

## 8.2 Baseline의 대표 결과

Baseline / GPT / B1 / Oracle-α에서는:

| Contrast | 평균차 Δα | Holm p |
|---|---:|---:|
| C4R − C4 | **+0.1693** | **0.0016** |
| C5 − C4 | **+0.1960** | **0.0016** |
| C6-E − C4R | +0.0375 | 0.4036 |
| C6-E − C6-EX | **+0.1742** | **0.0020** |

같은 Baseline이라도 모델과 budget에 따라 결과가 달랐습니다.

예를 들어 Gemini / B1에서는 C5 − C4가 +0.1153으로 양수였지만 C4R − C4는 −0.0023에 가까웠고, BINF에서는 오히려 자기 재검토가 +0.1427로 크게 나타났습니다.

즉 진단 유도와 자기 재검토를 하나의 “추론을 더 시킨 효과”로 묶기보다 각각 별도의 생성절차로 보는 것이 결과와 더 잘 맞습니다.

---

## 8.3 High에서 learned reference가 뚜렷했던 조건

High B1 / Oracle-α에서 C6-E − C4R은 두 모델 모두 양수였습니다.

| Model | Contrast | 평균차 Δα | 95% CI | Holm p |
|---|---|---:|---|---:|
| Gemini | C6-E − C4R | **+0.0897** | [+0.0408, +0.1405] | **0.0028** |
| GPT | C6-E − C4R | **+0.1003** | [+0.0368, +0.1631] | **0.0120** |
| GPT | C6-E − C6-EX | **+0.1146** | [+0.0443, +0.1864] | **0.0084** |

GPT High BINF의 C6-E − C4R은 **+0.2142**, 95% CI [+0.1415, +0.2918], Holm p 0.0016이었습니다.

High는 reasoning setting과 output cap을 함께 바꾼 generation regime입니다.  
여기서 중요한 관찰은 **최초 응답의 수용성이 높은 상태에서도 C3-E reference를 보여준 수정이 reference-free review보다 추가가치를 보인 조건이 있었다**는 점입니다.

---

## 8.4 candidate9와 free8

Baseline의 공통 B1에서 candidate9와 free8을 직접 비교했습니다.

논문 본문은 Oracle-α의 6개 비교를 전면에 둡니다.

```text
2 models × 3 policies(C4, C4R, C6-E)
```

6개 모두 **free8 − candidate9 < 0**이었고 모두 Holm 보정 후 차이가 확인됐습니다.

대표적으로 직접생성 C4:

- Gemini: **−0.7028**
- GPT: **−0.3372**

였습니다.

β와 γ까지 포함한 supplemental 확장에서는:

```text
2 models × 3 Oracles × 3 policies = 18 comparisons
```

**18/18 모두 같은 방향이고 18/18 모두 Holm 보정 후 유의**했습니다.

즉 더 넓은 행동공간을 표현할 수 있다는 사실과, 실제로 그 공간 안에서 유효하고 높은 가치의 행동을 구성하는 능력은 별개의 문제였습니다.

---

## 8.5 정보조건

Baseline C4/free8/B1에서 업종코드와 기업명을 하나씩 추가했습니다.

Oracle-α 본문 결과:

| Model | 대비 | 평균차 |
|---|---|---:|
| Gemini | IC-b − IC-a | −0.0175 |
| Gemini | IC-c − IC-b | +0.0228 |
| GPT | IC-b − IC-a | −0.0016 |
| GPT | IC-c − IC-b | +0.0728 |

네 비교 모두 조정 후 뚜렷한 차이는 없었습니다.

β·γ를 포함한 supplemental 확장까지 보면 총 **12개 비교에서 Holm p < .05는 0개**였습니다.

따라서 이 실험에서 업종코드 하나나 기업명 하나를 추가하는 것만으로 평균 정책가치가 일관되게 높아지는 패턴은 나타나지 않았습니다.

---

## 8.6 High generation regime과 action usability

Baseline B1의 free8 C4에서 strict action usability는:

- Gemini: **15.3%**
- GPT: **44.3%**

였습니다.

High B1에서는:

- Gemini: **99.1%**
- GPT: **97.4%**

까지 올라갔습니다.

High에서 가장 큰 변화 중 하나는 **B1 제약 안에서 실제 평가 가능한 행동을 거의 항상 만들어냈다는 점**입니다.

다만 High가 모든 예산과 모든 정책의 점수를 올린 것은 아닙니다.  
예를 들어 Gemini BINF C4는 Baseline보다 High에서 평균 정책가치가 낮았습니다.

그래서 High의 결과는 “추론을 많이 하면 무조건 좋아진다”보다 **generation setting이 수용성·행동강도·후속 수정효과를 함께 바꾼다**는 쪽에 가깝습니다.

---

## 8.7 자기 재검토와 실행규칙

GPT Baseline B1 / Oracle-α에서:

```text
Strict C4R − C4 = +0.169262
```

였습니다.

그런데 같은 원응답에 deterministic Repair를 적용하면:

```text
Repaired C4R − C4 = +0.004748
```

로 크게 줄었습니다.

같은 조건에서 C6-E − C4R은 반대로:

```text
Strict   +0.037527
Repaired +0.193566
```

으로 커졌습니다.

수용상태를 나눠 보면 GPT Baseline B1의 자기 재검토 전체 이득은, 두 응답이 모두 usable이었던 기업의 평균 개선보다 **불용 초안이 후속 검토에서 usable action으로 바뀐 기업들**에서 주로 발생했습니다.

이 결과 때문에 모델 이름이나 prompt만으로 정책을 설명하기보다:

```text
model
+ prompt condition
+ action space
+ budget
+ reference assignment
+ generation setting
+ validation / execution rule
```

을 묶어서 보는 **policy harness** 관점이 중요해집니다.

---

## 8.8 행동예산 B1과 BINF

같은 contrast라도 B1과 BINF에서 크기와 방향이 달라졌습니다.

Baseline / Oracle-α의 단순 B1−BINF 차분에서 눈에 띄는 값은:

- Gemini C5 − C4: **+0.1867**
- Gemini C4R − C4: **−0.1450**
- GPT C6-E − C4R: **−0.1097**
- GPT C5 − C4: **+0.0949**

입니다.

이 결과만 봐도 행동강도 상한은 후처리 옵션 정도가 아니라, **어떤 생성·수정 절차가 유리하게 나타나는지 자체를 바꾸는 실험조건**임을 알 수 있습니다.

---

# 9. 결과를 짧게 정리하면

1. **Oracle–Simulator 하네스는 공통 환경에서 정책을 비교하는 기반을 제공한다.**
2. **C3-E는 Oracle-α에서 A0와 고정 OE보다 높은 평균값을 보인 도메인 학습 reference다.**
3. **C6-E − C4R은 전체 24개 primary cell에서 양의 방향으로 나타나 네 contrast 중 가장 일관됐다.**
4. **C6-E − C6-EX도 23/24가 양수였고, 기업–reference 연결의 정보가 일부 조건에서 실제 차이를 만들었다.**
5. **진단 scaffold와 자기 재검토의 효과는 모델과 예산에 따라 크게 달랐다.**
6. **Baseline B1에서는 candidate9가 free8보다 일관되게 높은 정책가치를 보였다.**
7. **업종코드와 기업명 추가만으로 평균 정책가치가 뚜렷하게 높아지지는 않았다.**
8. **High generation regime은 B1 action usability를 크게 높였고, 높은 수용성 아래에서도 learned reference의 추가가치가 남는 조건이 있었다.**
9. **Strict/Repaired 차이는 응답 검증과 실행규칙도 최종 정책성과를 결정하는 요소임을 보여줬다.**
10. **결국 관측된 정책가치는 foundation model 하나보다 policy harness 전체의 설계에 더 가깝게 연결된다.**

---

## 10. 재현성

이 summary의 실험구조와 수치는 저장소의 frozen evidence에서 다시 확인할 수 있습니다.

### 결과 계약

```text
frozen/evidence/v1.3/
  THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN/
    contracts/THESIS_TABLE_CONTRACT.csv
```

- Primary: 96
- Supplemental: 722
- Parent-gate descriptive: 96
- Total registry: 914

### LLM 실험계약

```text
frozen/evidence/llm/
  final_design.json
  experiment_matrix.csv
  prompt_contract.json
  information_contract.json
  model_contract.json
  analysis_contract.json
```

### RL C3-E

```text
frozen/original_release/rl/
  C3E_E2_7SEED_BALANCED_DFEBAFA6/
```

### Published result reproduction

```powershell
python -m thesis_repro verify
python -m thesis_repro reproduce --run-id published_check
```

이 경로는 LLM API 재호출이나 RL 재학습 없이 published result layer를 다시 계산하고 frozen registry와 대조합니다.

---

## 11. 이 결과를 어떻게 읽으면 되나

이 연구의 결론은 단순히 “LLM이 신용등급을 개선한다”거나 “reference를 주면 무조건 좋아진다”는 이야기는 아닙니다.

더 직접적으로 보면 다음과 같습니다.

> **기업의 재무상태를 행동으로 바꾸는 과정에서, 행동을 어떻게 쓰게 하고, 어떤 초안을 다시 보게 하고, 어떤 reference를 붙이고, 어떤 제약과 실행규칙을 적용하는지가 실제 정책가치를 크게 바꾼다.**

특히 learned reference는 여러 조건에서 추가가치를 보였고, candidate9와 free8의 차이는 행동공간을 넓히는 것만으로 좋은 정책이 자동으로 나오지 않는다는 점을 보여줬습니다. High generation regime과 Repair 결과는 모델의 생성능력뿐 아니라 **usable action으로 연결되는 과정**이 중요하다는 점을 더 분명하게 만들었습니다.

반면 실제 기업이 이 권고를 집행했을 때 같은 신용성과가 발생하는지는 이후 실증자료가 필요한 문제입니다.  
이 연구는 그 이전 단계에서 **어떤 권고 절차를 비교할 가치가 있고, 그 차이가 어디서 발생하는지**를 구조적으로 보여주는 데 초점을 둡니다.

---

## 12. 관련 문서

- [README](../README.md)
- [Reproduction Guide](REPRODUCTION_GUIDE.md)
- [RL Reproduction](RL_REPRODUCTION.md)
- [LLM Reproduction](LLM_REPRODUCTION.md)
- [Evaluation Reproduction](EVALUATION_REPRODUCTION.md)
- [Provenance](../PROVENANCE.md)

이 문서는 전체 논문의 대체물이 아니라, **framework → experiment design → conditions → main results**를 한 번에 볼 수 있도록 압축한 안내문입니다.
