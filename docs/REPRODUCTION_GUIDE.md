# Reproduction Guide

이 문서는 V4.3 결과를 어느 수준까지 다시 확인할 수 있는지 세 단계로 정리한 안내문입니다.

가장 간단한 방법은 논문에 사용한 결과를 보존된 firm-level evidence에서 다시 계산하는 것입니다. 더 깊이 확인하려면 당시 Oracle, RL, LLM, evaluation 자산까지 검증할 수 있고, 필요한 외부 입력과 계산자원이 있다면 raw data부터 전체 실험을 다시 실행할 수도 있습니다.

## Level A — 논문 결과 재현

Python 3.11 환경에서 아래 명령을 실행합니다.

```powershell
python -m pip install -e ".[full,dev]"
python -m thesis_repro verify
python -m thesis_repro reproduce --run-id published_check
```

정상적으로 끝나면 다음 결과가 확인됩니다.

| 구분 | 결과 수 |
|---|---:|
| Primary results | 96 |
| Supplemental results | 722 |
| Parent-gate results | 96 |
| 전체 Result registry | 914 |
| Historical Stage8 observations | 96,600 |

실행 결과는 아래 receipt에 남습니다.

```text
runs/published_check/final/REPRODUCTION_RECEIPT.json
```

이 경로는 RL을 다시 학습하거나 LLM API를 다시 호출하는 방식이 아닙니다. 당시 보존한 firm-level evidence에서 논문 결과를 다시 계산한 뒤 frozen result registry와 대조합니다.

receipt에는 다음 정보가 기록됩니다.

```text
estimate_source = RECOMPUTED_FIRM_LEVEL_PRIMARY_ESTIMATE
bootstrap_ci_source = FROZEN_PRODUCTION_STREAM
```

따라서 논문 표의 숫자를 그대로 복사해 보여주는 방식보다 한 단계 깊은 재현이고, raw data부터 전체 실험을 다시 수행하는 full fresh replication과는 구분됩니다.

## Level B — 원 실험 자산 확인

당시 사용한 Oracle, RL, LLM, evaluation 자산과 provenance까지 확인하려면 Git LFS object를 함께 내려받습니다.

```powershell
git clone https://github.com/CallMeDemian/thesis-reproduction-kit-v43.git
cd thesis-reproduction-kit-v43

git lfs install
git lfs pull
git lfs fsck

python -m thesis_repro verify-original
python -m thesis_repro rebuild-c3e --release original
```

마지막 명령은 보존된 original release에서 C3-E를 다시 구성하는 선택 단계입니다.

이 경로에서는 다음 자료를 확인할 수 있습니다.

- Oracle stage0/stage1 결과
- RL 학습에 사용된 Stage2 compute parent
- 28개 historical RL actor와 parent chain
- C3-E reference policy
- Baseline/High LLM 실행 기록
- Stage8/Stage9 evaluation evidence

GitHub의 Source ZIP에는 Git LFS object가 모두 들어 있지 않을 수 있으므로, 원 실험 자산까지 볼 때는 clone 후 `git lfs pull`을 사용하는 편이 좋습니다.

## Level C — raw data부터 전체 재실행

전체 fresh pipeline은 다음 순서로 연결되어 있습니다.

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

기본 명령:

```powershell
python -m thesis_repro reproduce --full --run-id full_001
```

LLM provider 호출까지 포함할 때:

```powershell
python -m thesis_repro reproduce --full --live-llm --run-id full_001
python -m thesis_repro reproduce --full --live-llm --resume --run-id full_001
```

full fresh 실행에는 저장소 밖의 자료와 자원이 일부 필요합니다.

- licensed raw financial / nonfinancial data
- original Stage2 producer-input bundle
- 28-actor RL 학습을 위한 GPU/compute
- OpenAI/Gemini credentials와 API 비용
- provider batch job completion
- historical RL search catalog와 selection-rule artifact

C6-EX donor permutation은 certified distribution 안에 exact bytes로 포함되어 있어 별도 외부 파일이 필요하지 않습니다.

## 환경

Reference environment는 Python 3.11입니다.

정확한 package version은 아래 파일에 기록되어 있습니다.

```text
contracts/scientific/final_freeze/reproduction_environment_lock.json
contracts/scientific/final_freeze/requirements.reproduction.lock.txt
```

## 무엇부터 보면 되나

연구의 구조와 실험결과를 먼저 보려면 [Experiment Summary](EXPERIMENT_SUMMARY.md)를 보는 편이 빠릅니다.

실제로 결과를 다시 확인하려면 Level A부터 시작하면 됩니다. 원 실험의 actor, LLM 실행기록과 lineage까지 볼 필요가 있을 때 Level B로 내려가고, 외부 원천자료와 계산자원을 갖춘 상태에서 전체 실험을 다시 수행할 때 Level C를 사용합니다.
