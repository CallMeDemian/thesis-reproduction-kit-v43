# V4.3 Thesis Reproduction Kit

이 저장소는 제 석사학위논문의 실험 결과를 **다시 확인하고, 어떤 자료와 계산을 거쳐 결과가 만들어졌는지 추적할 수 있도록** 정리한 재현성 패키지입니다.

논문에 들어간 표와 수치를 단순히 모아둔 것이 아니라, 당시 실험에서 사용한 결과 자료와 검증 정보, RL/LLM 실행 기록, 해시와 연결 관계를 함께 보관했습니다.

교수님이나 심사위원께서는 우선 아래의 **“논문 결과 확인”** 부분만 보셔도 됩니다.  
좀 더 자세한 설명은 [Professor Reproduction Guide](docs/PROFESSOR_REPRODUCTION_GUIDE.md)에 정리해 두었습니다.

현재 기준 release는 [v1.0.4](https://github.com/CallMeDemian/thesis-reproduction-kit-v43/releases/tag/thesis-v43-reproduction-kit-v1.0.4)입니다.

---

## 논문 결과를 가장 간단하게 확인하는 방법

Python 3.11 환경에서 아래 세 명령을 실행하면 됩니다.

```powershell
python -m pip install -e ".[full,dev]"
python -m thesis_repro verify
python -m thesis_repro reproduce --run-id professor_check
```

정상적으로 끝나면 논문에 사용한 결과 묶음이 다음과 같이 확인됩니다.

| 구분 | 결과 수 |
|---|---:|
| Primary results | **96** |
| Supplemental results | **722** |
| Parent-gate results | **96** |
| 전체 Result registry | **914** |
| Historical Stage8 observations | **96,600** |

실행 결과는 아래 파일에 남습니다.

```text
runs/professor_check/final/REPRODUCTION_RECEIPT.json
```

이 과정에서는 **LLM API를 새로 호출하지 않고, RL도 다시 학습하지 않습니다.**  
대신 당시 보존한 firm-level evidence에서 논문 결과를 다시 계산하고, 원래 결과 registry와 일치하는지 확인합니다.

즉, “논문에 적힌 숫자를 다시 출력하는 것”보다는 한 단계 더 깊은 검증이지만, raw data부터 모든 실험을 새로 돌리는 방식과는 구분됩니다.

---

## 어디까지 재현할 수 있나요?

재현 수준을 세 가지로 나누면 이해가 쉽습니다.

### 1. 논문 결과 재현

위의 기본 명령으로 가능합니다.

- 논문에 보고한 주요 결과 재계산
- Stage8 96,600개 observation 확인
- 914개 결과 registry 대조
- 결과가 어떤 frozen scientific evidence에서 나왔는지 확인

이 경로에는 private raw data나 GPU, 유료 API가 필요하지 않습니다.

### 2. 당시 실험 자산까지 확인

원래 사용한 Oracle, RL, LLM, evaluation 자산과 그 연결관계까지 보고 싶다면 Git LFS까지 내려받아 확인할 수 있습니다.

```powershell
git clone https://github.com/CallMeDemian/thesis-reproduction-kit-v43.git
cd thesis-reproduction-kit-v43

git lfs install
git lfs pull
git lfs fsck

python -m thesis_repro verify-original
python -m thesis_repro rebuild-c3e --release original
```

여기서는 예를 들어 다음을 확인할 수 있습니다.

- Oracle stage0/stage1 결과
- RL 학습에 사용된 Stage2 compute parent
- 28개 RL actor와 각 parent chain
- C3-E reference policy
- BASELINE/HIGH LLM 실행 기록
- Stage8/Stage9 historical evaluation 자료

GitHub의 단순 Source ZIP은 Git LFS 파일을 모두 포함한 clone과는 다릅니다.

### 3. raw data부터 모든 실험을 새로 실행

전체 fresh pipeline도 코드로 연결되어 있습니다.

```text
raw inputs
  ↓
Oracle
  ↓
Stage2 / Simulator
  ↓
RL training
  ↓
C3-E
  ↓
Stage6
  ↓
LLM
  ↓
Stage8
  ↓
Stage9
  ↓
thesis outputs / comparison / certification
```

실행 명령은 다음과 같습니다.

```powershell
python -m thesis_repro reproduce --full --run-id full_001
```

LLM provider 호출까지 새로 수행하려면:

```powershell
python -m thesis_repro reproduce --full --live-llm --run-id full_001
python -m thesis_repro reproduce --full --live-llm --resume --run-id full_001
```

다만 이 수준의 완전한 fresh 재실험에는 현재 저장소 밖의 자료와 자원이 일부 필요합니다.

- licensed raw financial / nonfinancial data
- original Stage2 producer-input bundle
- fresh RL 학습용 GPU/compute
- OpenAI/Gemini API credentials 및 비용
- provider job completion
- historical RL search catalog / selection-rule artifact

그래서 이 저장소는 **논문 결과의 재현과 당시 실험의 검증은 가능하게 하되**, 외부에 배포할 수 없는 원천자료까지 포함한 완전한 fresh rerun은 별도 조건이 필요하다고 구분하고 있습니다.

---

## 논문의 주요 실험 구성

현재 V4.3의 핵심 설정은 다음과 같습니다.

- 평가 기업: **575개**
- 의사결정 차원: **8개**
- 후보 action: **9개**
  - `A0`, `DL`, `RF`, `CX`, `WC1`, `WC2`, `OE`, `MX1`, `MX2`
- C3-E reference policy:
  - 4 configurations
  - configuration당 7 seeds
  - 총 **28 actors**
- LLM Plan-3:
  - 42 cells × 575 firms × 2 reasoning regimes
  - BASELINE **24,150**
  - HIGH **24,150**
  - 총 **48,300 logical requests**
- Historical Stage8:
  - **96,600 observations**

---

## Stage2와 RL 학습 데이터는 보존되어 있습니다

한 가지 헷갈리기 쉬운 부분이 있어 별도로 적습니다.

RL이 실제로 어떤 Stage2 자료로 학습되었는지는 보존되어 있습니다.

```text
frozen/original_release/rl/stage2/
```

여기에는 당시 RL의 Stage2 compute parent, 학습용 row table, runtime input과 관련 manifest가 들어 있고, 28개 actor의 parent provenance와 연결되어 있습니다.

즉,

- 당시 RL이 어떤 데이터를 부모로 학습되었는지 확인: **가능**
- historical actor의 parent chain 확인: **가능**
- 그 Stage2 자료를 raw data 단계부터 똑같이 새로 만들어내기: **추가 producer input 필요**

입니다.

현재 외부 입력으로 남아 있는 것은 “학습에 사용한 Stage2 자체”가 아니라, **그 Stage2를 처음부터 다시 생성하기 위한 원래 upstream producer input bundle**입니다.

---

## C6-EX는 별도 외부 파일이 필요하지 않습니다

C6-EX의 donor permutation은 certified distribution 안에 exact bytes로 포함되어 있습니다.

Fresh 실행에서는 이 파일을 자동으로 복원한 뒤 SHA-256을 검증합니다.  
따라서 C6-EX permutation은 현재 external input boundary로 보지 않습니다.

---

## Synthetic full-DAG 확인

외부 데이터나 유료 API 없이도 전체 orchestration이 실제로 연결되어 있는지는 deterministic fixture로 확인할 수 있습니다.

```powershell
python -m thesis_repro acceptance-e2e --run-id synthetic_check
python -m thesis_repro trace --run-id synthetic_check
```

정상 결과는 다음과 같습니다.

- `completion_state = PASS`
- `dag_complete = true`
- `certifiable = false`
- `lineage_closed = true`

이 테스트는 **시스템 전체가 제대로 연결되어 있는지 확인하는 용도**이고, 실제 논문 실험을 새로 수행했다는 의미는 아닙니다.

---

## 환경

Reference environment는 Python 3.11입니다.

정확한 package version은 아래 파일에 기록되어 있습니다.

- `contracts/scientific/final_freeze/reproduction_environment_lock.json`
- `contracts/scientific/final_freeze/requirements.reproduction.lock.txt`

---

## 더 자세한 문서

- [Professor Reproduction Guide](docs/PROFESSOR_REPRODUCTION_GUIDE.md)
- [Clean-clone acceptance](docs/CLEAN_CLONE_ACCEPTANCE.md)
- [Oracle reproduction](docs/ORACLE_REPRODUCTION.md)
- [RL reproduction](docs/RL_REPRODUCTION.md)
- [LLM reproduction](docs/LLM_REPRODUCTION.md)
- [Evaluation reproduction](docs/EVALUATION_REPRODUCTION.md)
- [Provenance](PROVENANCE.md)

---

## 정리

이 repository에서 가장 중요하게 확인할 수 있는 것은 두 가지입니다.

1. **논문에 보고한 결과가 보존된 scientific evidence에서 다시 산출되는지**
2. **그 결과가 어떤 Oracle / RL / LLM / evaluation 자산과 연결되어 있는지**

Published-result reproduction과 historical evidence 검증은 현재 저장소 안에서 닫혀 있습니다.  
완전한 fresh numerical replication은 비공개 원천자료와 외부 compute/API가 필요한 부분만 별도로 남겨 두었습니다.
