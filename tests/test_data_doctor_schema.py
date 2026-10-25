"""data doctor on a synthetic fixture (the licensed raw inputs are not available in CI)."""
import json

import pandas as pd

from thesis_repro import data


def _xlsx(path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([["000001"] * len(columns)], columns=columns).to_excel(path, index=False)
    return path


def _inventory(root, groups):
    inventory = {"inputs": [{"target_root": target, "files": [{"relative_path": rel, "size_bytes": (root / target / rel).stat().st_size if (root / target / rel).exists() else 1} for rel in files]} for target, files in groups.items()]}
    path = root / "contracts/scientific/input_inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, ensure_ascii=False), encoding="utf-8")


def test_schema_checked_from_loader_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    _xlsx(tmp_path / "data/raw/raw_all/코스피_재무상태표.xlsx", ["회사명", "거래소코드", "회계년도", "자산총계"])
    _xlsx(tmp_path / "data/raw/raw_all/코스닥_손익계산서.xlsx", ["회사명", "매출액"])
    _xlsx(tmp_path / "data/raw/rating_sample/코스피_신용평가에 관한 사항.xlsx", ["거래소코드", "회계년도", "신용등급"])
    _xlsx(tmp_path / "data/raw/raw_nonfinancial/konex_optional/코넥스_일반사항.xlsx", ["회사명"])
    _inventory(tmp_path, {
        "data/raw/raw_all": ["코스피_재무상태표.xlsx", "코스닥_손익계산서.xlsx"],
        "data/raw/rating_sample": ["코스피_신용평가에 관한 사항.xlsx"],
        "data/raw/raw_nonfinancial": ["konex_optional/코넥스_일반사항.xlsx"],
    })
    report = data.verify_input_contract(write=False)
    rows = {row["path"].split("/")[-1]: row for row in report["files"]}
    assert rows["코스피_재무상태표.xlsx"]["schema_status"] == "SCHEMA_MATCH"
    assert rows["코스닥_손익계산서.xlsx"]["schema_status"] == "SCHEMA_MISMATCH"
    assert set(rows["코스닥_손익계산서.xlsx"]["schema_missing"]) == {"firm_code", "year"}
    assert rows["코스피_신용평가에 관한 사항.xlsx"]["schema_status"] == "SCHEMA_MISMATCH"
    assert "평가일" in rows["코스피_신용평가에 관한 사항.xlsx"]["schema_missing"]
    assert rows["코넥스_일반사항.xlsx"]["schema_status"] == "NOT_CHECKED"
    assert report["status"] == "INPUT_REQUIRED"
    assert report["konex_files"] == {"inventory_entries": 1, "marked_optional": 0, "note": report["konex_files"]["note"]}
    assert "self-referential" in report["comparison_basis"]["hash"]


def test_missing_raw_inputs_stay_input_required(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    _inventory(tmp_path, {"data/raw/raw_all": ["코스피_재무상태표.xlsx"]})
    report = data.verify_input_contract(write=False)
    assert report["status"] == "INPUT_REQUIRED"
    assert report["missing_required_count"] == 1
