import hashlib

import pytest

from thesis_repro import stage2

CONTRACT = '{\n  "candidate_ids": ["A0", "DL"],\n  "note": "한글"\n}\n'


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_contract_copy_keeps_bytes(tmp_path, newline):
    source = tmp_path / "contracts/v43_action_contract.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(CONTRACT.replace("\n", newline).encode("utf-8"))
    target = tmp_path / "runs/x/03_stage2/input_source/candidate_action_contract_v4_3.json"
    stage2._copy_contract_bytes(source, target)
    assert target.read_bytes() == source.read_bytes()
    assert _sha256(target) == _sha256(source)
