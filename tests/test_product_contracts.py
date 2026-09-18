from thesis_repro.contracts import EXPECTED_REQUESTS, contract_report


def test_final_logical_request_count():
    report = contract_report()
    assert report["generated"] == EXPECTED_REQUESTS == 48_300
    assert report["unique_ids"] == 48_300

