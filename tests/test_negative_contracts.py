import json
from pathlib import Path


def test_live_gate_is_explicit():
    contract = json.loads(Path("contracts/llm/final_as_executed_generation_contract.json").read_text(encoding="utf-8"))
    assert contract["live_gate"] == "THESIS_REPRO_ENABLE_LIVE_LLM=I_APPROVE_FRESH_REPLICATION"
    assert contract["status"] == "SOURCE_SNAPSHOT_FOUND_MIGRATION_PENDING"
