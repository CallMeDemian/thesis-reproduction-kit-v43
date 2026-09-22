# V4.3 Experiment Summary

## A Validation-First Evaluation Framework for LLM-Generated Corporate Credit-Rating Recourse

이 문서는 V4.3 실험을 처음 보는 사람이 전체 구조와 핵심 결과를 빠르게 파악할 수 있도록 정리한 짧은 technical note입니다.  
재현 명령이나 파일 구조보다 **무슨 질문을 했고, 어떤 조건을 비교했고, 결과가 어떻게 나왔는지**에 초점을 맞췄습니다.

전체 수치와 세부 검정은 논문 본문과 `THESIS_TABLE_CONTRACT.csv`, Stage8/Stage9 evidence에서 확인할 수 있습니다.

---

## 1. 무엇을 연구했나

연구의 출발점은 간단합니다.

기업의 현재 재무상태를 보고 LLM이 “향후 1년 동안 어떤 재무행동을 취하면 신용상태 개선에 도움이 될지” 권고한다고 할 때, 그 권고의 실제 결과는 바로 관찰할 수 없습니다.  
부채를 줄이거나, 차입구조를 바꾸거나, 운전자본을 조정하라는 권고가 실제로 실행되었을 때 신용등급이 어떻게 달라질지는 미래의 반사실적 결과이기 때문입니다.

그래서 이 연구는 LLM 자체만 평가하지 않고, 다음과 같은 **평가 하네스**를 먼저 구성했습니다.

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

이 위에 다시 **정책 하네스**를 얹었습니다.

```text
Offline RL reference (C3-E)
              │
              ├─────────────┐
              ↓             ↓
      LLM direct policy   LLM + reference
       C4 / C5 / C4R      C6-E / C6-EX
              │             │
              └──────┬──────┘
                     ↓
           동일 Simulator / Oracle
                     ↓
               정책가치 비교
```

핵심은 “어떤 LLM이 가장 좋은가”보다, **같은 foundation model이라도 생성 방식, 행동공간, 예산, 참조정보, 추론설정과 실행규칙이 달라지면 관측되는 정책가치가 어떻게 달라지는가**를 보는 데 있습니다.

---

## 2. 연구질문

최종 실험은 기존 RQ1–RQ5를 유지하면서, 실제로 식별 가능한 비교를 다음처럼 정리했습니다.

| 연구질문 | 핵심 내용 |
|---|---|
| **RQ1. 평가기반** | Oracle 점수와 실제 신용등급의 정렬성, Simulator의 회계적 일관성을 어느 범위까지 평가에 사용할 수 있는가 |
| **RQ2. 행동가치·상대순위** | 학습된 RL 참조정책 C3-E와 비교정책의 상대적인 정책가치는 어떠한가 |
| **RQ3. 생성·수정·참조** | 직접생성, 진단 유도, 자기 재검토, 학습참조, 잘못 배정된 참조가 권고의 정책가치를 어떻게 바꾸는가 |
| **RQ4. 강건성·행동공간** | Oracle, 행동예산, candidate9/free8 선택에 따라 주요 효과가 유지되는가 |
| **RQ5. 정보조건** | 익명 재무정보, 산업·연도, 기업명까지 정보 범위를 넓혔을 때 직접생성 결과가 달라지는가 |

RQ1은 이후 결과를 해석할 수 있는 범위를 정하는 역할을 합니다.  
Oracle과 Simulator가 유용한 비교기반을 제공하더라도, 여기서 측정한 정책가치를 실제 기업에서의 인과적 실행효과와 동일하게 볼 수는 없습니다. 본문에서는 이 점을 평가기반의 사용범위로 따로 다룹니다.

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

별도로 아래 9개 표준 후보행동을 둡니다.

`A0, DL, RF, CX, WC1, WC2, OE, MX1, MX2`

- **candidate9**: 9개 후보 중 하나를 선택
- **free8**: 8차원 연속 행동을 직접 작성
- **B1**: 정규화된 총 행동강도에 상한을 둠
- **BINF**: 각 축의 허용범위는 유지하되 총 행동강도 상한은 두지 않음

이 구분을 통해 “LLM에게 자유를 더 주면 더 좋은 행동을 만드는가”, “같은 행동예산 안에서 후보선택과 자유작성은 어떻게 다른가”를 따로 볼 수 있습니다.

---

## 4. RL 참조정책 C3-E

LLM에 외부 참조를 주기 전에, 기업 재무·신용평가 도메인 데이터에서 학습한 Offline RL 정책을 만들었습니다.

학습 흐름은 다음과 같습니다.

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

최종 C3-E는 **E2 encoder**를 사용하고 다음 네 configuration으로 구성됩니다.

- B27
- M2_S0935
- DT06
- T15_REWARD_S088

각 configuration마다 동일한 7개 seed를 사용했습니다.

`2, 11, 12, 13, 14, 15, 16`

따라서 총 **4 × 7 = 28 actors**입니다.

앙상블은 먼저 configuration 내부에서 7개 seed의 확률을 평균하고, 이후 네 configuration을 각각 25%로 평균합니다. 특정 seed 하나가 전체 앙상블을 과도하게 좌우하지 않도록 한 구조입니다.

### C3-E 결과

최종 논문에 보고된 Oracle-α 기준 C3-E의 기술통계는 다음과 같습니다.

| 지표 | 값 |
|---|---:|
| 평균 절대점수 | **55.959960** |
| A0 대비 | **+0.923299** |
| OE 대비 | **+0.157936** |

C3-E는 LLM이 참고할 수 있는 학습정책으로 사용했습니다. 다만 evaluator를 바꾸면 상대가치가 달라지고 기업별 손실 사례도 있어, 모든 기업과 모든 평가기준에서 우월한 정책으로 놓지는 않았습니다.

---

## 5. LLM 실험설계

사용한 foundation model은 두 개입니다.

- **GPT-5.4-mini**
- **Gemini 3.1 Flash-Lite**

같은 실험행렬을 두 reasoning regime에서 각각 실행했습니다.

- **BASELINE**
- **HIGH**

한 regime의 Plan-3는 42 cells, 기업은 575개입니다.

```text
42 cells × 575 firms = 24,150 requests
```

BASELINE과 HIGH를 합치면 최종 generation inventory는:

```text
24,150 + 24,150 = 48,300 logical requests
```

### 5.1 정책조건

| 조건 | 무엇을 하는가 | 비교하려는 효과 |
|---|---|---|
| **C4** | 기업정보만 보고 처음부터 행동 생성 | 직접생성 기준 |
| **C5** | C4와 같은 최초 생성이지만 진단 scaffold 추가 | 진단·추론 유도의 증분 |
| **C4R** | 같은 run의 C4 원문을 다시 검토 | 자기 재검토 |
| **C6-E** | 같은 C4 원문 + 해당 기업의 C3-E 참조행동 | 학습참조 추가효과 |
| **C6-EX** | 같은 C4 원문 + 다른 기업에서 고정 derangement한 C3-E 참조행동 | “참조가 있음”과 “기업에 맞는 참조”를 분리 |

C6-E와 C6-EX에서는 참조정책의 출처명, Q값, Oracle 점수 등을 LLM에게 보여주지 않고 **8차원 행동 벡터만** 제시했습니다.

### 5.2 핵심 비교

Primary analysis는 네 contrast를 사전에 고정했습니다.

1. **C5 − C4**: 진단 scaffold의 추가효과
2. **C4R − C4**: 자기 재검토 효과
3. **C6-E − C4R**: 재검토 이후 올바른 학습참조를 추가한 효과
4. **C6-E − C6-EX**: 기업에 맞는 참조와 고정된 잘못 배정된 참조의 차이

### 5.3 실행행렬

한 model, 한 reasoning regime에서:

| 블록 | 조건 | cells |
|---|---|---:|
| Core B1 | free8 / IC-b / B1, C4·C5·C4R·C6-E·C6-EX | 5 |
| Core BINF | free8 / IC-b / BINF, 동일 5조건 | 5 |
| Action-space | candidate9 / IC-b / B1, C4·C4R·C6-E | 3 |
| Information | free8 / B1, IC-a·IC-c의 C4 | 2 |
| Stability | free8 / IC-b / B1·BINF, C4·C4R·C6-E 재실행 | 6 |
| **합계** |  | **21** |

두 model을 합치면 42 cells입니다.

---

## 6. 정보조건

정보범위도 세 단계로 나눴습니다.

- **IC-a**: 익명 재무정보
- **IC-b**: IC-a + 산업 + 회계연도
- **IC-c**: IC-b + 기업명

Primary analysis는 **IC-b**에 고정했고, IC-a와 IC-c는 RQ5의 supplemental comparison에 사용했습니다.

---

## 7. 평가와 통계

정책가치는 575개 동일 기업을 기준으로 paired comparison합니다.

Primary result는 다음 축의 조합입니다.

```text
2 models
× 3 Oracles (α, β, γ)
× 2 reasoning regimes
× 2 budgets (B1, BINF)
× 4 primary contrasts
= 96 primary results
```

Primary population은 **Strict ITT / free8 / IC-b**입니다.

불확실성 평가는 10,000회 bootstrap을 사용했고, primary family에는 Holm 보정을 적용했습니다.

최종 result registry는 다음으로 구성됩니다.

| 구분 | rows |
|---|---:|
| Primary | **96** |
| Supplemental | **722** |
| Parent-gate descriptive | **96** |
| **전체** | **914** |

Historical Stage8 canonical evidence는 **96,600 rows**입니다.

---

# 8. 핵심 결과

## 8.1 네 primary contrast의 전체 패턴

96개 primary result를 네 contrast별로 묶으면 다음과 같습니다.

| Contrast | 양의 점추정 | Holm p < .05 | 요약 |
|---|---:|---:|---|
| **C4R − C4** | 18 / 24 | 11 / 24 | 자기 재검토는 자주 양수였지만 조건 의존성이 큼 |
| **C5 − C4** | 15 / 24 | 5 / 24 | 진단 scaffold의 효과는 비교적 불규칙 |
| **C6-E − C4R** | **24 / 24** | 9 / 24 | 가장 일관된 방향성. 학습참조 추가 후 모든 primary cell에서 점추정이 양수 |
| **C6-E − C6-EX** | **23 / 24** | 2 / 24 | 기업에 맞는 참조가 거의 항상 양의 방향이지만 통계적으로 뚜렷한 cell은 제한적 |

이 표에서 가장 눈에 띄는 결과는 **C6-E − C4R**입니다.  
foundation model, Oracle, reasoning regime, budget을 바꿔도 24개 primary cell 모두에서 점추정이 양수였습니다. 다만 Holm 보정 후 유의한 경우는 9개이므로, 방향의 일관성과 통계적 확실성은 구분해서 보는 편이 맞습니다.

C6-E와 C6-EX의 비교도 23/24가 양수였지만, Holm 보정 후 유의한 결과는 2개였습니다. “참조벡터가 존재하는 것”만으로 설명되지 않는 신호가 보이지만, 기업별 참조의 정보가치를 모든 조건에서 강하게 확정할 정도의 결과는 아니었습니다.

---

## 8.2 대표적인 Alpha 결과

정책가치 차이가 비교적 크게 나타난 Oracle-α 결과 몇 개를 보면:

### Baseline / GPT / B1

| Contrast | 평균차 Δα | Holm p |
|---|---:|---:|
| C4R − C4 | **+0.1693** | **0.0016** |
| C5 − C4 | **+0.1960** | **0.0016** |
| C6-E − C6-EX | **+0.1742** | **0.0020** |

### High / B1의 학습참조 효과

| Model | Contrast | 평균차 Δα | 95% CI | Holm p |
|---|---|---:|---|---:|
| Gemini | C6-E − C4R | **+0.0897** | [+0.0408, +0.1405] | **0.0028** |
| GPT | C6-E − C4R | **+0.1003** | [+0.0368, +0.1631] | **0.0120** |
| GPT | C6-E − C6-EX | **+0.1146** | [+0.0443, +0.1864] | **0.0084** |

High / GPT / BINF에서는 C6-E − C4R이 **+0.2142**로 나타났고, 95% CI는 [+0.1415, +0.2918], Holm p는 0.0016이었습니다.

이 결과만 보면 learned reference가 비교적 안정적으로 도움이 되는 것처럼 보이지만, 아래의 action-space와 execution-rule 결과를 같이 봐야 전체 그림이 잡힙니다.

---

## 8.3 자유작성보다 candidate9가 강했던 B1 비교

RQ4의 행동공간 비교는 candidate9와 free8을 같은 B1 조건에서 맞춰 비교했습니다.

총 18개 비교:

```text
2 models × 3 Oracles × 3 policies(C4, C4R, C6-E)
```

에서 **FREE8 − CANDIDATE9가 18/18 모두 음수였고, 18/18 모두 Holm 보정 후 유의**했습니다.

예를 들면 Oracle-α에서:

- Gemini C4: **−0.7028**
- GPT C4: **−0.3372**

였습니다.

즉 이 실험에서는 “더 자유로운 행동공간이 항상 더 높은 정책가치를 만든다”는 결과가 나오지 않았습니다.  
오히려 동일한 B1 제약 아래에서는 미리 정의된 9개 후보 중 하나를 선택하는 방식이 두 모델과 세 Oracle에 걸쳐 더 높은 값을 보였습니다.

---

## 8.4 정보량을 늘린 효과는 뚜렷하지 않았다

RQ5는 직접생성 C4에서 정보범위를 바꿨습니다.

- IC-b − IC-a: 산업·연도 정보를 추가
- IC-c − IC-b: 기업명을 추가

2 models × 3 Oracles × 2 information contrasts = **12개 결과** 가운데 Holm 보정 후 유의한 결과는 **0개**였습니다.

따라서 이 설정에서는 산업·연도 또는 기업명을 추가한 것만으로 직접생성 정책가치가 뚜렷하게 높아지는 패턴은 확인되지 않았습니다.

---

## 8.5 High reasoning은 “항상 더 높은 정책가치”보다 실행완결성에서 큰 차이가 났다

Baseline B1의 free8 C4에서 strict action acceptance rate는:

- Gemini: **15.3%**
- GPT: **44.3%**

였습니다.

High B1에서는:

- Gemini: **99.1%**
- GPT: **97.4%**

까지 올라갔습니다.

High가 형식 준수와 실행 가능한 action 생성에는 큰 차이를 만들었습니다.  
하지만 정책가치의 모든 contrast가 Baseline보다 커진 것은 아니었습니다.

Reasoning-regime interaction 24개 중 Holm 보정 후 유의한 결과는 3개였고, High 안의 효과가 유의하다는 사실만으로 Baseline 대비 효과크기 자체가 증가했다고 볼 수는 없었습니다.

---

## 8.6 자기 재검토 효과는 실행규칙에 민감했다

Strict 결과만 보면 자기 재검토 C4R − C4는 여러 조건에서 강한 양수였습니다.

대표적으로 Baseline / GPT / B1 / Oracle-α:

```text
Strict C4R − C4 = +0.1693
```

그런데 deterministic Repair를 적용한 민감도 분석에서는 같은 비교가 약 **+0.0047**까지 줄었습니다.

Supplemental execution-sensitivity 48개 가운데 6개는 Holm 보정 후 유의했습니다.  
이 결과는 자기 재검토의 관측효과 중 일부가 “한 번 더 검토했다”는 과정 자체뿐 아니라, **원응답의 형식·행동제약 위반, 후속 분기 기회, 실패 처리규칙**과 함께 결정된다는 점을 보여줍니다.

따라서 이 연구에서 평가단위는 foundation model 하나가 아니라:

```text
model
+ prompt condition
+ action space
+ budget
+ reference assignment
+ reasoning regime
+ execution/failure policy
```

를 묶은 policy harness에 가깝습니다.

---

## 8.7 행동예산도 일부 효과를 바꿨다

B1과 BINF의 효과 차이를 직접 비교한 budget interaction은 8개입니다.

그중 2개가 Holm 보정 후 유의했습니다. 둘 다 Gemini / Baseline / Oracle-α에서 나타났습니다.

- C5 − C4의 B1−BINF 차이: **+0.1867**, Holm p = 0.0048
- C4R − C4의 B1−BINF 차이: **−0.1450**, Holm p = 0.0008

즉 행동강도 제약은 단순한 후처리 옵션이 아니라, 어떤 생성·수정 조건이 유리하게 보이는지 자체를 바꿀 수 있습니다.

---

# 9. 결과를 한 문장씩 정리하면

1. **평가기반은 비교도구로 쓸 수 있지만 실제 실행의 인과효과와 같은 것은 아니다.**
2. **C3-E는 A0와 OE보다 높은 Oracle-α 기술통계를 보인 학습참조정책이다.**
3. **학습참조를 자기 재검토 뒤에 추가한 C6-E − C4R은 24/24 primary cell에서 양의 방향이었다.**
4. **기업에 맞는 참조와 deranged 참조의 차이도 23/24가 양수였지만 통계적 확실성은 더 약했다.**
5. **C5 진단 scaffold와 C4R 자기 재검토 효과는 모델·예산·실행규칙에 따라 달라졌다.**
6. **B1에서는 candidate9가 free8보다 18/18 비교에서 높은 정책가치를 보였다.**
7. **산업·연도 또는 기업명을 추가한 정보조건은 12개 비교에서 유의한 추가효과가 없었다.**
8. **High reasoning은 특히 action acceptance를 크게 높였지만, 모든 정책가치 효과를 일률적으로 확대하지는 않았다.**
9. **Strict/Repaired 차이는 실행규칙도 정책의 일부라는 점을 보여줬다.**

---

## 10. 재현성

이 summary에서 언급한 실험구조와 수치는 저장소의 frozen evidence에서 다시 확인할 수 있습니다.

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

## 11. 해석의 범위

이 실험은 LLM 권고를 실제 기업이 실행한 뒤의 신용등급을 관찰한 연구가 아닙니다.  
대신 동일 기업, 동일 행동계약, 동일 Simulator와 Oracle 위에서 여러 정책 하네스를 비교하는 방식입니다.

그래서 결과는 주로 다음 질문에 답합니다.

> **주어진 평가환경 안에서, 어떤 생성·참조·실행 조건이 더 높은 정책가치와 더 안정적인 실행을 만들어내는가?**

이 범위 안에서는 learned reference, action-space restriction, reasoning regime, failure handling이 모두 중요한 설계변수로 나타났습니다.

반대로 “특정 LLM이 본질적으로 더 좋은 신용의사결정을 한다”거나 “실제 기업이 이 권고를 실행하면 같은 신용등급 개선이 발생한다”는 주장까지 확장하려면 추가적인 현실 데이터와 실행효과 검증이 필요합니다.

---

## 12. 관련 문서

- [README](../README.md)
- [Reproduction Guide](PROFESSOR_REPRODUCTION_GUIDE.md)
- [RL Reproduction](RL_REPRODUCTION.md)
- [LLM Reproduction](LLM_REPRODUCTION.md)
- [Evaluation Reproduction](EVALUATION_REPRODUCTION.md)
- [Provenance](../PROVENANCE.md)

이 문서는 전체 논문의 대체물이 아니라, **framework → experiment design → conditions → main results**를 한 번에 볼 수 있도록 압축한 안내문입니다.
