# V4.3 Thesis Reproduction Kit

> **석사학위논문 V4.3의 실험결과와 근거 자료를 재현·검증하기 위한 패키지입니다.**  
> 논문에 보고된 결과를 단순히 복사해 둔 저장소가 아니라, 동결된 scientific evidence와 결과 registry를 이용해 결과를 다시 산출하고 provenance를 확인할 수 있도록 구성했습니다.

**권장 제출본:** [thesis-v43-reproduction-kit-v1.0.4](https://github.com/CallMeDemian/thesis-reproduction-kit-v43/releases/tag/thesis-v43-reproduction-kit-v1.0.4)

교수님·심사위원용 상세 안내는 [Professor Reproduction Guide](docs/PROFESSOR_REPRODUCTION_GUIDE.md)에 정리되어 있습니다.

---

## 가장 먼저 확인하실 내용

이 저장소의 재현성은 세 수준으로 구분됩니다.

| 수준 | 무엇을 확인하는가 | 추가 외부 자원 |
|---|---|---|
| **A. 논문 결과 재현** | 논문에 보고된 published result layer를 certified evidence에서 다시 산출·검증 | **없음** |
| **B. 원 실험 자산 검증** | Oracle / RL 28 actors / LLM 실행기록 / 평가 자산의 원본 hash·provenance 검증 | Git LFS 필요 |
| **C. 완전한 fresh 재실험** | raw data부터 Oracle → RL → LLM → Stage8/9까지 새로 실행 | 비공개 원천자료, Stage2 producer input, GPU, API 등 필요 |

**논문 심사 목적에는 A가 기본 재현 경로입니다.**  
A 경로는 private raw data, RL 재학습, OpenAI/Gemini API 호출 없이 실행됩니다.

---

## 1. 논문 결과 재현 — 권장 경로

Python 3.11 환경에서:

```powershell
python -m pip install -e ".[full,dev]"
python -m thesis_repro verify
python -m thesis_repro reproduce --run-id professor_check
```

정상 실행 시 다음 결과 contract가 확인됩니다.

| 결과 계층 | 행 수 |
|---|---:|
| Primary results | **96** |
| Supplemental results | **722** |
| Parent-gate results | **96** |
| Result registry | **914** |
| Historical Stage8 evidence | **96,600** |

최종 receipt:

```text
runs/professor_check/final/REPRODUCTION_RECEIPT.json
```

주요 의미:

- `estimate_source = RECOMPUTED_FIRM_LEVEL_PRIMARY_ESTIMATE`
- `bootstrap_ci_source = FROZEN_PRODUCTION_STREAM`
- `api_calls = 0`
- `training_runs = 0`

즉, **논문 표의 숫자를 그대로 복사하는 방식이 아니라 certified firm-level evidence에서 published result layer를 다시 계산하고 검증**합니다. 다만 이 경로는 raw data부터 모든 모델을 새로 학습시키는 fresh experimental rerun과는 구분됩니다.

---

## 2. 논문의 실험 구조

현재 V4.3의 핵심 scientific identity는 다음과 같습니다.

- **8개 managerial action dimensions**
- **9개 candidate actions:** `A0`, `DL`, `RF`, `CX`, `WC1`, `WC2`, `OE`, `MX1`, `MX2`
- **평가 cohort:** 575 firms
- **C3-E reference policy:** 4 configurations × 7 seeds = **28 actors**
- **LLM Plan-3:** 42 cells × 575 firms × 2 reasoning regimes
- **LLM logical requests:** 24,150 BASELINE + 24,150 HIGH = **48,300**
- **Historical Stage8 canonical evidence:** **96,600 rows**

이 저장소는 frozen evidence, fresh computation, 그리고 comparison 결과를 분리하여 관리합니다. Fresh run에서 생성되는 artifact에는 source state, parent hashes, stage manifests, artifact hashes가 기록됩니다.

---

## 3. 원 실험 자산까지 검증하려면

원래 실험에서 사용된 대용량 Oracle/RL/LLM/evaluation 자산까지 확인하려면 Git LFS가 필요합니다.

```powershell
git clone https://github.com/CallMeDemian/thesis-reproduction-kit-v43.git
cd thesis-reproduction-kit-v43

git lfs install
git lfs pull
git lfs fsck

python -m thesis_repro verify-original
python -m thesis_repro rebuild-c3e --release original
```

이 경로에서는 다음과 같은 historical evidence를 검증할 수 있습니다.

- Oracle stage0/stage1 assets
- RL Stage2 compute parent
- 28개 RL actor parent chain
- C3-E release identity
- BASELINE/HIGH LLM execution artifacts
- Stage8/Stage9 historical evaluation assets

> GitHub의 단순 Source ZIP 다운로드는 Git LFS object가 모두 포함된 clone과 동일하지 않습니다.

---

## 4. 완전한 fresh experimental replication

전체 fresh pipeline은 다음 구조입니다.

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
Thesis outputs / comparison / certification
```

실행 명령:

```powershell
python -m thesis_repro reproduce --full --run-id full_001
```

실제 LLM provider 호출까지 포함하려면:

```powershell
python -m thesis_repro reproduce --full --live-llm --run-id full_001
python -m thesis_repro reproduce --full --live-llm --resume --run-id full_001
```

Full fresh execution에는 다음 외부 조건이 필요할 수 있습니다.

- licensed raw financial / nonfinancial inputs
- original Stage2 producer-input pack
- fresh 28-actor 학습을 위한 GPU/compute
- OpenAI/Gemini credentials 및 paid execution
- asynchronous provider completion
- historical RL search catalog / exact selection-rule artifact

반면 **C6-EX donor permutation은 외부 입력이 아닙니다.** Certified distribution에 exact bytes가 포함되어 있으며 SHA-256 검증 후 자동 복원됩니다.

---

## 5. Stage2와 RL 학습 데이터에 대한 주의

RL이 실제로 어떤 Stage2 데이터로 학습되었는지는 보존되어 있습니다.

```text
frozen/original_release/rl/stage2/
```

여기에는 historical Stage2 compute parent와 RL 학습용 row tables 및 관련 runtime inputs가 포함되며, 28개 actor의 parent provenance와 연결되어 있습니다.

현재 외부 boundary로 남아 있는 것은 **당시 Stage2 compute parent 자체가 아니라, 그 Stage2를 raw/upstream 단계에서 다시 생성하기 위한 original producer-input bundle**입니다.

즉:

- historical RL training parent 확인: **가능**
- historical actor provenance 확인: **가능**
- raw data부터 Stage2를 동일하게 fresh 재생성: **추가 external input 필요**

---

## 6. Synthetic full-DAG acceptance

실제 run engine과 canonical adapters의 전체 wiring은 deterministic fixture로 별도 검증할 수 있습니다.

```powershell
python -m thesis_repro acceptance-e2e --run-id synthetic_check
python -m thesis_repro trace --run-id synthetic_check
```

Expected:

- `completion_state = PASS`
- `dag_complete = true`
- `certifiable = false`
- `lineage_closed = true`

이 경로는 **architecture / lineage 검증용**이며 thesis-scale fresh scientific evidence로 사용하지 않습니다.

---

## 7. 재현성 패키지의 범위

이 repository가 주장하는 것은 다음과 같습니다.

### 재현 가능한 것

- published thesis result layer
- frozen Stage8 evidence와 914-row result registry
- historical Oracle/RL/LLM/evaluation provenance
- 28개 RL actor의 parent chain
- C3-E reference identity
- exact C6-EX donor permutation provenance
- LLM experiment design과 historical execution evidence

### 별도 외부 입력이 필요한 것

- 상용/비공개 raw data부터 시작하는 완전한 fresh numerical replication
- original Stage2 producer-input bundle
- fresh RL retraining용 compute
- fresh paid LLM provider execution

따라서 본 저장소는 **published-result reproduction과 historical evidence audit은 독립적으로 수행할 수 있도록 구성**되어 있으며, full fresh replication이 요구하는 비배포 입력과 외부 자원은 별도로 명시합니다.

---

## 8. 환경 및 상세 문서

Reference environment:

- **Python 3.11**
- exact version lock:
  - `contracts/scientific/final_freeze/reproduction_environment_lock.json`
  - `contracts/scientific/final_freeze/requirements.reproduction.lock.txt`

관련 문서:

- [Professor Reproduction Guide](docs/PROFESSOR_REPRODUCTION_GUIDE.md)
- [Clean-clone acceptance](docs/CLEAN_CLONE_ACCEPTANCE.md)
- [Oracle reproduction](docs/ORACLE_REPRODUCTION.md)
- [RL reproduction](docs/RL_REPRODUCTION.md)
- [LLM reproduction](docs/LLM_REPRODUCTION.md)
- [Evaluation reproduction](docs/EVALUATION_REPRODUCTION.md)
- [Provenance](PROVENANCE.md)

---

## 9. Repository status

현재 canonical professor-facing release는 **v1.0.4**입니다.

- Scientific evidence: frozen
- Published-result reproduction: closed
- Historical evidence provenance: closed
- Fresh execution framework: closed with external-input boundaries
- Full fresh numerical experiment: external inputs/resources required

이 저장소의 목적은 **논문 결과의 근거와 재현 절차를 가능한 범위에서 명시적으로 남기고, 재현 가능한 부분과 외부 제약이 있는 부분을 구분하여 검증 가능하게 하는 것**입니다.
