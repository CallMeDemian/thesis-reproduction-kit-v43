import json
from pathlib import Path


def test_contract_files():
    root = Path(__file__).resolve().parents[1]
    with open(root / "contracts/V13_STATISTICAL_CONTRACT.json", encoding="utf-8") as f:
        c = json.load(f)
    assert c["B"] == 10000
    assert c["hash"] == "sha256_first_8_bytes_big_unsigned"

