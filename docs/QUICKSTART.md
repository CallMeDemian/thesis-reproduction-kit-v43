# Quickstart

Reference environment는 Python 3.11입니다.

```powershell
python -m pip install -e ".[full,dev]"
```

정확한 historical package version은 아래 두 파일에 기록되어 있습니다.

```text
contracts/scientific/final_freeze/reproduction_environment_lock.json
contracts/scientific/final_freeze/requirements.reproduction.lock.txt
```

## 논문 결과 재현

```powershell
python -m thesis_repro verify
python -m thesis_repro reproduce --run-id quickstart_published
```

예상 결과는 primary 96, supplemental 722, parent-gate 96, 전체 result registry 914입니다. 이 경로에서는 LLM API 호출이나 RL 재학습이 없습니다.

전체 재현 범위와 단계별 설명은 [Reproduction Guide](REPRODUCTION_GUIDE.md)를 참고하면 됩니다.

## 원 실험 자산 확인

Oracle, RL actor, LLM 실행기록 등 multi-gigabyte historical asset tree까지 확인하려면 Git LFS를 내려받은 뒤 다음 명령을 실행합니다.

```powershell
git lfs pull
git lfs fsck
python -m thesis_repro verify-original
```

## Full fresh replication

raw data부터 전체 pipeline을 다시 실행하는 경로는 별도 workflow입니다.

```powershell
python -m thesis_repro reproduce --full --run-id full_001
```

외부 입력과 live provider 실행 조건은 [Fresh Replication](FRESH_REPLICATION.md)에 정리되어 있습니다.
