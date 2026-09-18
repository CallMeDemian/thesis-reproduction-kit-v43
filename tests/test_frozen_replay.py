from thesis_repro.frozen import verify_frozen


def test_frozen_distribution_and_registry():
    result = verify_frozen()
    assert result["status"] == "PASS"
    assert result["registry_rows"] == 914

