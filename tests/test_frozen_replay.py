from thesis_repro.frozen import verify_frozen
from thesis_repro.release import EXPECTED_RELEASE_ROOT_SHA256, certified_release_root_hash, load_certified_release
from thesis_repro.paths import ROOT, load_json


def test_frozen_distribution_and_registry():
    result = verify_frozen()
    assert result["status"] == "PASS"
    assert result["registry_rows"] == 914


def test_certified_release_root_hash_is_exact():
    payload = load_json(ROOT / "frozen/release/submission/THESIS_V43_SUBMISSION_FINAL_20260917/release.json")
    assert certified_release_root_hash(payload) == EXPECTED_RELEASE_ROOT_SHA256
    loaded = load_certified_release(ROOT)
    assert loaded["computed_root_sha256"] == EXPECTED_RELEASE_ROOT_SHA256

