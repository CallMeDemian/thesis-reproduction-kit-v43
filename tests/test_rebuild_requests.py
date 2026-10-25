from thesis_repro import rebuild_requests as rr

POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 1234\n"


def test_line_ending_hashes_match_either_form():
    lf = b'{\n  "a": 1\n}\n'
    crlf = lf.replace(b"\n", b"\r\n")
    from_lf = rr._line_ending_hashes(lf)
    from_crlf = rr._line_ending_hashes(crlf)
    assert from_lf["lf"] == from_crlf["lf"] == from_lf["checkout"]
    assert from_lf["crlf"] == from_crlf["crlf"] == from_crlf["checkout"]


def test_lfs_pointer_contract_is_input_required(tmp_path):
    contract = tmp_path / rr.FROZEN_ACTION_CONTRACT
    contract.parent.mkdir(parents=True)
    contract.write_bytes(POINTER)
    result = rr.rebuild_requests(tmp_path, "t")
    assert result["status"] == "INPUT_REQUIRED"
    assert "git lfs pull" in result["reason"]


def test_frames_equal_treats_missing_values_alike():
    import pandas as pd

    left = pd.DataFrame({"a": ["x", None], "b": [1, 2]})
    right = pd.DataFrame({"a": ["x", float("nan")], "b": [1, 2]})
    assert rr._frames_equal(left, right)
    assert not rr._frames_equal(left, right.assign(b=[1, 3]))
