"""Single run-scoped owner for the V4.3 LLM preparation boundary.

The scientific prompt and provider contracts live in ``credit_recourse.final_release``.
This module only binds those contracts to a fresh run namespace and provides a
deterministic transport substitute for synthetic acceptance.  A mock receipt is
never accepted as a live scientific execution receipt.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import time
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .paths import ROOT
from credit_recourse.final_release.common import canonical_hash

EXPECTED_REQUESTS = 48_300
FIRM_COUNT = 575
REGIMES = ("baseline", "high")
C6EX_PERMUTATION_SHA256 = "a5385d4c811e73fbf39a46cf299959d72ada8f3931ed3a02a2ba07e4f1c0bd40"
CERTIFIED_DISTRIBUTION_SHA256 = "931631d52541bb6dfdd5ed9fda767e2e48ccb8b7521f70ae771db68df2c1831d"
CERTIFIED_DISTRIBUTION_RELATIVE = Path("frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip")


@dataclass(frozen=True)
class LogicalRequest:
    request_id: str
    generation_regime: str
    cell_id: str
    model: str
    condition: str
    phase: str
    wave: int
    information_condition: str
    budget: str
    replicate_id: int
    firm_ordinal: int
    parent_request_id: str | None
    mode: str
    parent_cell_id: str | None
    c3e_reference: str
    response_contract: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_rows(root: Path = ROOT) -> list[dict[str, str]]:
    path = Path(root) / "frozen/evidence/llm/experiment_matrix.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def iter_logical_requests(run_id: str, *, root: Path = ROOT, firm_count: int = FIRM_COUNT) -> Iterator[LogicalRequest]:
    rows = _matrix_rows(root)
    if len(rows) != 42:
        raise ValueError(f"expected 42 final LLM matrix cells, found {len(rows)}")
    namespace = f"fresh:{run_id}"
    for regime in REGIMES:
        for row in rows:
            for firm in range(1, int(firm_count) + 1):
                parent = row.get("parent_cell_id") or None
                yield LogicalRequest(
                    request_id=f"{namespace}:{regime}:{row['cell_id']}:firm-{firm:04d}",
                    generation_regime=regime,
                    cell_id=row["cell_id"],
                    model=row["model"],
                    condition=row["condition"],
                    phase=row["phase"],
                    wave=int(row["wave"]),
                    information_condition=row["info"],
                    budget=row["budget"],
                    replicate_id=int(row["replicate"]),
                    firm_ordinal=firm,
                    parent_request_id=f"{namespace}:{regime}:{parent}:firm-{firm:04d}" if parent else None,
                    mode=row["mode"],
                    parent_cell_id=parent,
                    c3e_reference="fresh.<run_id>.C3E",
                    response_contract="credit_recourse.final_release",
                )


def _requests(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("request_id", "")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate logical request ids")
    return rows


def prepare_full_llm(paths, *, firm_count: int = FIRM_COUNT) -> dict[str, Any]:
    target = paths.llm_root / "logical_requests.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    ids: set[str] = set()
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for request in iter_logical_requests(paths.run_id, root=paths.root, firm_count=firm_count):
            if request.request_id in ids:
                raise ValueError(f"duplicate fresh request id: {request.request_id}")
            ids.add(request.request_id)
            handle.write(json.dumps(asdict(request), sort_keys=True) + "\n")
            count += 1
    expected = 42 * 2 * int(firm_count)
    report = {
        "schema_version": "v43_final_llm_prepare_v2",
        "status": "PASS" if count == expected else "FAILED",
        "execution_class": "SYNTHETIC_E2E_ACCEPTANCE" if firm_count != FIRM_COUNT else "REAL_COMPUTE",
        "expected_requests": expected,
        "request_count": count,
        "unique_request_ids": len(ids),
        "baseline": 42 * int(firm_count),
        "high": 42 * int(firm_count),
        "c3e_parent": f"runs/{paths.run_id}/07_c3e/release.json",
        "logical_requests_sha256": _sha256(target),
    }
    (paths.llm_root / "prepare_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _mock_response(request: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(str(request["request_id"]).encode()).hexdigest()
    return {
        "request_id": request["request_id"],
        "provider": "mock",
        "transport": "mock",
        "provider_request_id": f"mock:{digest[:16]}",
        "raw_response": {"mode": "candidate_selection", "selected_candidate": "A0", "confidence": "low", "brief_rationale": f"synthetic:{digest[:12]}"},
    }


def execute_full_llm(paths, *, live: bool, firm_count: int = FIRM_COUNT) -> dict[str, Any]:
    request_path = paths.llm_root / "logical_requests.jsonl"
    requests = _requests(request_path)
    if live:
        return execute_real_llm(paths, resume=True)
    response_path = paths.llm_root / "raw_mock_responses.jsonl"
    with response_path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(_mock_response(request), sort_keys=True) + "\n")
    receipt = {"schema_version": "v43_final_llm_mock_receipt_v1", "status": "PASS", "execution_class": "SYNTHETIC_E2E_ACCEPTANCE", "provider_contacted": False, "request_count": len(requests), "raw_response_sha256": _sha256(response_path)}
    (paths.llm_root / "generate_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def materialize_full_stage7(paths, *, expected_count: int | None = None) -> dict[str, Any]:
    request_path = paths.llm_root / "logical_requests.jsonl"
    response_path = paths.llm_root / "raw_mock_responses.jsonl"
    requests = _requests(request_path)
    responses = _requests(response_path) if response_path.is_file() else []
    expected_ids = {str(row["request_id"]) for row in requests}
    observed_ids = [str(row.get("request_id", "")) for row in responses]
    duplicates = sorted({value for value in observed_ids if observed_ids.count(value) > 1})
    missing = sorted(expected_ids - set(observed_ids))
    unexpected = sorted(set(observed_ids) - expected_ids)
    if expected_count is not None and len(expected_ids) != expected_count:
        raise ValueError(f"expected {expected_count} requests, found {len(expected_ids)}")
    if duplicates or missing or unexpected:
        return {"status": "FAILED", "missing_request_ids": missing, "duplicate_request_ids": duplicates, "unexpected_request_ids": unexpected}
    normalized = paths.llm_root / "llm_response_action_layers.parquet"
    normalized.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    request_by_id = {str(req["request_id"]): req for req in requests}
    action_contract_path = paths.root / "contracts/scientific/v43_action_contract.json"
    if not action_contract_path.is_file():
        action_contract_path = ROOT / "contracts/scientific/v43_action_contract.json"
    action_contract = json.loads(action_contract_path.read_text(encoding="utf-8"))
    action_columns = list(action_contract["action_columns"])
    zero_action = {column: 0.0 for column in action_columns}
    records = []
    for row in responses:
        req = request_by_id[str(row["request_id"])]
        condition = str(req["condition"])
        ordinal = int(req["firm_ordinal"]) - 1
        reference_source = "C3-E" if condition == "C6-E" else "C3-EX" if condition == "C6-EX" else "none"
        reference_row_id = ordinal if condition == "C6-E" else ((ordinal + 1) % max(1, int(firm_count_from_requests(requests)))) if condition == "C6-EX" else ordinal
        record = {
            "request_id": str(row["request_id"]), "cell_id": req["cell_id"], "row_id": ordinal,
            "firm_key": f"{int(req['firm_ordinal']):06d}", "model_key": req["model"],
            "generation_regime": req["generation_regime"],
            "phase": req.get("phase", "MAIN"), "info": req["information_condition"],
            "information_condition": req["information_condition"], "budget": req["budget"],
            "replicate": int(req["replicate_id"]), "wave": int(req.get("wave", 1)),
            "parent_cell_id": req.get("parent_cell_id"), "parent_request_id": req.get("parent_request_id"),
            "policy": condition, "mode": req["mode"],
            "candidate_id": "A0", "action_layer": "candidate9" if "candidate9" in str(req["cell_id"]) else "free8",
            "action_application_status": "strict_valid", "reference_source": reference_source,
            "reference_row_id": reference_row_id, "reference_candidate_id": "A0",
            "c6ex_mapping_role": "SYNTHETIC_DERANGEMENT_FIXTURE_ONLY" if condition == "C6-EX" else None,
            "model_response_usable": True, "itt_noop_fallback_applied": False,
            "provider": row.get("provider"), "provider_request_id": row.get("provider_request_id"),
            "reasoning_regime": req["generation_regime"],
            **zero_action,
        }
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_parquet(normalized, index=False)
    per_protocol = frame.copy()
    per_protocol["analysis_population"] = "per_protocol"
    itt = frame.copy()
    itt["analysis_population"] = "itt"
    for name, table in (("llm_stage7_action_table_per_protocol.parquet", per_protocol), ("llm_stage7_action_table_itt.parquet", itt)):
        table.to_parquet(normalized.parent / name, index=False)
        (normalized.parent / "stage7").mkdir(parents=True, exist_ok=True)
        table.to_parquet(normalized.parent / "stage7" / name, index=False)
    failure_columns = ["request_id", "row_id", "policy", "mode", "failure_categories"]
    pd.DataFrame(columns=failure_columns).to_csv(normalized.parent / "llm_stage7_failure_audit.csv", index=False)
    metadata = {"status": "PASS", "scientific_contract_version": "V4.3_FINAL_20260912", "final_action_contract_hash": _sha256(action_contract_path), "backend_is_live": False, "final_paper_run_allowed": False, "baseline": sum(r.get("generation_regime") == "baseline" for r in requests), "high": sum(r.get("generation_regime") == "high" for r in requests), "total": len(requests), "execution_class": "SYNTHETIC_E2E_ACCEPTANCE" if responses and all(r.get("provider") == "mock" for r in responses) else "REAL_COMPUTE"}
    (normalized.parent / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def firm_count_from_requests(requests: list[dict[str, Any]]) -> int:
    return max((int(row.get("firm_ordinal", 0)) for row in requests), default=0)


def contract_report(root: Path = ROOT, run_id: str | None = None) -> dict[str, Any]:
    count = sum(1 for _ in iter_logical_requests(run_id or "report", root=root))
    return {"expected": EXPECTED_REQUESTS, "generated": count, "unique_ids": count, "matrix_cells": 42, "firms": FIRM_COUNT, "regimes": list(REGIMES), "status": "PASS" if count == EXPECTED_REQUESTS else "FAILED"}


def render_requests(*, limit: int | None = None, run_id: str = "report", root: Path = ROOT) -> list[dict[str, Any]]:
    rows = [asdict(item) for item in iter_logical_requests(run_id, root=root)]
    return rows if limit is None else rows[:limit]


def mock_responses(requests: Iterable[dict[str, Any]], destination: Path) -> dict[str, Any]:
    rows = list(requests)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for request in rows:
            handle.write(json.dumps(_mock_response(request), sort_keys=True) + "\n")
    return {"provider": "mock", "request_count": len(rows), "path": str(destination), "execution_class": "mock_only"}


def gate_status() -> dict[str, Any]:
    return {"authorized": False, "authorized_by_cli": False, "credentials": {"openai": bool(os.environ.get("OPENAI_API_KEY")), "gemini": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))}, "note": "live transport is authorized only by reproduce --full --live-llm"}


def prepare_requests(run_root: Path, *, limit: int | None = None) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    from types import SimpleNamespace
    paths = SimpleNamespace(root=ROOT, run_id=run_root.name, run_root=run_root, llm_root=run_root / "09_llm")
    target = paths.llm_root / "logical_requests.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in iter_logical_requests(paths.run_id, root=ROOT):
        rows.append(asdict(item))
        if limit is not None and len(rows) >= limit:
            break
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    report = {"status": "PASS", "request_count": len(rows), "expected_requests": limit if limit is not None else EXPECTED_REQUESTS, "execution_class": "SYNTHETIC_E2E_ACCEPTANCE" if limit is not None else "REAL_COMPUTE"}
    (paths.llm_root / "prepare_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def generate_mock(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root)
    from types import SimpleNamespace
    paths = SimpleNamespace(root=run_root.parent, run_id=run_root.name, run_root=run_root, llm_root=run_root / "09_llm")
    return execute_full_llm(paths, live=False)


def materialize_responses(run_root: Path, *, expected_count: int | None = None) -> dict[str, Any]:
    run_root = Path(run_root)
    from types import SimpleNamespace
    paths = SimpleNamespace(root=run_root.parent, run_id=run_root.name, run_root=run_root, llm_root=run_root / "09_llm")
    return materialize_full_stage7(paths, expected_count=expected_count)


def _real_runtime_environment(paths):
    values = {
        "THESIS_REPRO_LLM_CONFIG_ROOT": str(paths.llm_root / "config"),
        "THESIS_REPRO_LLM_OUTPUT_ROOT": str(paths.llm_root / "final_plan3"),
        "THESIS_REPRO_C3E_ROOT": str(paths.c3e_root),
        "THESIS_REPRO_RUN_ROOT": str(paths.run_root),
    }
    previous = {name: os.environ.get(name) for name in values}

    @contextmanager
    def scoped():
        os.environ.update(values)
        try:
            yield
        finally:
            for name, old in previous.items():
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old

    return scoped()


def _source_label(paths, source: Path | str) -> str:
    value = Path(source)
    try:
        return str(value.relative_to(paths.root)).replace("\\", "/")
    except ValueError:
        return str(value).replace("\\", "/")


def _canonical_industry_firm_universe(paths) -> set[str] | None:
    """Resolve the canonical FY2024 firm-year universe without using results."""
    import pandas as pd

    local_candidates = (
        paths.stage2_root / "input_source/input_splits/canonical_evaluation_row_ids.parquet",
        paths.stage2_root / "input_splits/canonical_evaluation_row_ids.parquet",
    )
    for candidate in local_candidates:
        if not candidate.is_file():
            continue
        try:
            frame = pd.read_parquet(candidate)
            for column in ("canonical_firm_year_id", "firm_key"):
                if column in frame:
                    values = frame[column].astype(str).str.strip()
                    if len(values) == FIRM_COUNT and values.nunique() == FIRM_COUNT:
                        return set(values)
        except Exception:
            continue

    distribution = paths.root / CERTIFIED_DISTRIBUTION_RELATIVE
    if not distribution.is_file() or _sha256(distribution) != CERTIFIED_DISTRIBUTION_SHA256:
        return None
    try:
        with zipfile.ZipFile(distribution) as archive:
            member = next(
                name for name in archive.namelist()
                if name.endswith("stage2_candidate_projection/input_splits/canonical_evaluation_row_ids.parquet")
            )
            frame = pd.read_parquet(io.BytesIO(archive.read(member)))
            values = frame["canonical_firm_year_id"].astype(str).str.strip()
            if len(values) == FIRM_COUNT and values.nunique() == FIRM_COUNT:
                return set(values)
    except Exception:
        return None
    return None


def _industry_candidate_frame(candidate: dict[str, Any], paths):
    """Read one authorized industry candidate and return its frame/provenance."""
    import pandas as pd

    if candidate["kind"] == "path":
        source = Path(candidate["path"])
        if not source.is_file():
            return None, {"valid": False, "source": _source_label(paths, source), "role": candidate["role"], "reason": "SOURCE_MISSING"}
        try:
            raw = source.read_bytes()
            frame = pd.read_parquet(io.BytesIO(raw)).copy()
        except Exception as exc:
            return None, {"valid": False, "source": _source_label(paths, source), "role": candidate["role"], "reason": f"PARQUET_UNREADABLE:{type(exc).__name__}"}
        return frame, {"source": _source_label(paths, source), "role": candidate["role"], "source_sha256": hashlib.sha256(raw).hexdigest()}

    distribution = Path(candidate["zip_path"])
    label = f"{_source_label(paths, distribution)}::{candidate['member']}"
    if not distribution.is_file():
        return None, {"valid": False, "source": label, "role": candidate["role"], "reason": "SOURCE_MISSING"}
    distribution_sha256 = _sha256(distribution)
    if distribution_sha256 != CERTIFIED_DISTRIBUTION_SHA256:
        return None, {"valid": False, "source": label, "role": candidate["role"], "reason": "CERTIFIED_DISTRIBUTION_HASH_MISMATCH", "distribution_zip_sha256": distribution_sha256}
    try:
        with zipfile.ZipFile(distribution) as archive:
            members = [name for name in archive.namelist() if name.endswith(candidate["member_suffix"])]
            if len(members) != 1:
                return None, {"valid": False, "source": label, "role": candidate["role"], "reason": "CERTIFIED_MEMBER_NOT_UNIQUE", "member_count": len(members)}
            member = members[0]
            raw = archive.read(member)
            frame = pd.read_parquet(io.BytesIO(raw)).copy()
            source_manifest = None
            if candidate.get("manifest_suffix"):
                manifests = [name for name in archive.namelist() if name.endswith(candidate["manifest_suffix"])]
                if len(manifests) == 1:
                    source_manifest = json.loads(archive.read(manifests[0]).decode("utf-8"))
                    expected = source_manifest.get("binding_artifact_sha256")
                    if expected and expected != hashlib.sha256(raw).hexdigest():
                        return None, {"valid": False, "source": f"{_source_label(paths, distribution)}::{member}", "role": candidate["role"], "reason": "SOURCE_MANIFEST_HASH_MISMATCH"}
            return frame, {
                "source": f"{_source_label(paths, distribution)}::{member}",
                "role": candidate["role"],
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "distribution_zip": _source_label(paths, distribution),
                "distribution_zip_sha256": distribution_sha256,
                "member": member,
                "source_manifest": source_manifest,
            }
    except Exception as exc:
        return None, {"valid": False, "source": label, "role": candidate["role"], "reason": f"PARQUET_UNREADABLE:{type(exc).__name__}"}


def _evaluate_industry_binding_candidate(candidate: dict[str, Any], paths, canonical_universe: set[str] | None) -> dict[str, Any]:
    """Validate a candidate completely before it can be selected."""
    import pandas as pd

    frame, provenance = _industry_candidate_frame(candidate, paths)
    if frame is None:
        return provenance
    provenance = {**provenance, "priority": candidate["priority"]}
    firm_col = next((name for name in ("canonical_firm_year_id", "firm_key", "firm_id") if name in frame), None)
    year_col = next((name for name in ("fiscal_year", "year") if name in frame), None)
    display_col = next((name for name in ("industry_display_value", "industry_class", "industry") if name in frame), None)
    code_col = next((name for name in ("induty_code", "industry_code", "industry_class") if name in frame), None)
    if not firm_col:
        return {**provenance, "valid": False, "reason": "FIRM_KEY_UNIDENTIFIABLE"}
    first_firm_value = str(frame[firm_col].iloc[0]) if len(frame) else ""
    if not year_col and "::" not in first_firm_value:
        return {**provenance, "valid": False, "reason": "FISCAL_YEAR_UNAVAILABLE"}
    if not display_col or not code_col:
        return {**provenance, "valid": False, "reason": "INDUSTRY_COLUMNS_INCOMPLETE"}

    try:
        if "::" in first_firm_value:
            frame["firm_key"] = frame[firm_col].astype(str).str.strip()
        else:
            years = pd.to_numeric(frame[year_col], errors="raise").astype(int).astype(str)
            frame["firm_key"] = frame[firm_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6) + "::" + years
    except Exception:
        return {**provenance, "valid": False, "reason": "FISCAL_YEAR_UNAVAILABLE"}
    if len(frame) != FIRM_COUNT or frame["firm_key"].nunique() != FIRM_COUNT:
        return {**provenance, "valid": False, "reason": "INDUSTRY_BINDING_COHORT_INCOMPLETE", "rows": len(frame), "unique_firms": int(frame["firm_key"].nunique())}
    if canonical_universe is None:
        return {**provenance, "valid": False, "reason": "CANONICAL_FIRM_UNIVERSE_UNAVAILABLE", "rows": len(frame), "unique_firms": FIRM_COUNT}
    if set(frame["firm_key"]) != canonical_universe:
        return {**provenance, "valid": False, "reason": "CANONICAL_FIRM_UNIVERSE_MISMATCH", "rows": len(frame), "unique_firms": FIRM_COUNT}
    codes = frame[code_col].astype(str).str.strip()
    if codes.eq("").any() or codes.str.lower().isin({"nan", "none", "unknown"}).any():
        return {**provenance, "valid": False, "reason": "INDUSTRY_CODE_INCOMPLETE", "rows": len(frame), "unique_firms": FIRM_COUNT}
    display = frame[display_col].astype(str).str.strip()
    if display.eq("").any() or display.str.lower().isin({"nan", "none", "unknown"}).any():
        return {**provenance, "valid": False, "reason": "INDUSTRY_DISPLAY_INCOMPLETE", "rows": len(frame), "unique_firms": FIRM_COUNT}
    frame["induty_code"] = codes
    frame["industry_display_value"] = display
    return {**provenance, "valid": True, "frame": frame[["firm_key", "induty_code", "industry_display_value"]].reset_index(drop=True), "rows": len(frame), "unique_firms": int(frame["firm_key"].nunique())}


def _industry_binding_candidates(paths) -> list[dict[str, Any]]:
    stage2 = paths.stage2_root
    root = paths.root
    return [
        {"priority": 1, "kind": "path", "path": stage2 / "input_source/FY2024_INDUSTRY_BINDING.parquet", "role": "FRESH_EXOGENOUS_INFORMATION_INPUT"},
        {"priority": 1, "kind": "path", "path": stage2 / "input_source/industry_binding.parquet", "role": "FRESH_EXOGENOUS_INFORMATION_INPUT"},
        {"priority": 1, "kind": "path", "path": stage2 / "input_source/firm_payload_source.parquet", "role": "FRESH_EXOGENOUS_INFORMATION_INPUT"},
        {"priority": 2, "kind": "path", "path": stage2 / "input_source/phase_eval_candidate.parquet", "role": "FRESH_EXOGENOUS_INFORMATION_INPUT"},
        {"priority": 2, "kind": "path", "path": stage2 / "phase_eval_candidate.parquet", "role": "FRESH_EXOGENOUS_INFORMATION_INPUT"},
        {"priority": 3, "kind": "zip", "zip_path": root / CERTIFIED_DISTRIBUTION_RELATIVE, "member_suffix": "archive/DEPLOYED_RELEASE/llm_inputs/fy2024_industry_binding/industry_binding.parquet", "manifest_suffix": "archive/DEPLOYED_RELEASE/llm_inputs/fy2024_industry_binding/FY2024_INDUSTRY_BINDING_MANIFEST.json", "member": "archive/DEPLOYED_RELEASE/llm_inputs/fy2024_industry_binding/industry_binding.parquet", "role": "FROZEN_EXOGENOUS_INFORMATION_INPUT"},
        {"priority": 4, "kind": "path", "path": root / "frozen/original_release/llm/final_plan3/2cf6d6d0e4250e66ce882ab95f9d641f2c73711ffbc6429e9a203dcc2ee680a2/firm_payload_source.parquet", "role": "FROZEN_EXOGENOUS_INFORMATION_INPUT"},
    ]


def _materialize_fresh_industry_binding(paths, config_root: Path) -> dict[str, Any]:
    """Select the first valid authorized public industry source."""
    canonical_universe = _canonical_industry_firm_universe(paths)
    rejected: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate in _industry_binding_candidates(paths):
        result = _evaluate_industry_binding_candidate(candidate, paths, canonical_universe)
        if result.get("valid"):
            selected = result
            break
        rejected.append({key: value for key, value in result.items() if key not in {"frame", "source_manifest"}})
    if selected is None:
        return {"status": "INPUT_REQUIRED", "reason": "INDUSTRY_BINDING_SOURCE_REQUIRED", "rejected_candidates": rejected}

    binding = selected["frame"]
    artifact = config_root / "FY2024_INDUSTRY_BINDING.parquet"
    binding.to_parquet(artifact, index=False)
    selected_source = selected["source"]
    manifest = {
        "schema_version": "fresh_industry_binding_v1",
        "status": "PASS",
        "role": selected["role"],
        "selected_source": selected_source,
        "selected_role": selected["role"],
        "selected_source_sha256": selected["source_sha256"],
        "selection_priority": selected["priority"],
        "rejected_candidates": rejected,
        "source": selected_source,
        "source_sha256": selected["source_sha256"],
        "binding_artifact": "FY2024_INDUSTRY_BINDING.parquet",
        "binding_artifact_sha256": _sha256(artifact),
        "rows": FIRM_COUNT,
        "unique_firm_key": FIRM_COUNT,
        "unresolved_count": 0,
    }
    for key in ("distribution_zip", "distribution_zip_sha256", "member"):
        if key in selected:
            manifest[key] = selected[key]
    manifest_path = config_root / "FY2024_INDUSTRY_BINDING_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    information_path = config_root / "information_contract.json"
    information = json.loads(information_path.read_text(encoding="utf-8"))
    information["industry_binding_manifest"] = "FY2024_INDUSTRY_BINDING_MANIFEST.json"
    information["industry_status"] = "FRESH_BINDING_MATERIALIZED"
    information_path.write_text(json.dumps(information, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "manifest": str(manifest_path.relative_to(paths.root)).replace("\\", "/"), "artifact": str(artifact.relative_to(paths.root)).replace("\\", "/"), "artifact_sha256": manifest["binding_artifact_sha256"], "role": manifest["role"], "selected_source": selected_source, "selected_source_sha256": selected["source_sha256"], "selection_priority": selected["priority"], "rejected_candidates": rejected}


def _write_fresh_design_release(paths, config_root: Path, c6ex_manifest: dict[str, Any]) -> dict[str, Any]:
    """Create a derived release identity bound to same-run C3-E and config."""
    from credit_recourse.contracts.runtime_assets import active_action_contract
    design_path = config_root / "final_design.json"
    release_path = config_root / "design_release.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    historical = json.loads(release_path.read_text(encoding="utf-8"))
    actions_path = paths.c3e_root / "C3E_firm_actions.parquet"
    probabilities_path = paths.c3e_root / "C3E_firm_probabilities.parquet"
    c3e_release = json.loads((paths.c3e_root / "release.json").read_text(encoding="utf-8"))
    c3e_hash = str(c3e_release.get("release_hash") or c3e_release.get("ensemble_release_hash") or "")
    action_contract_path = active_action_contract(paths.root)
    identity = {
        "namespace": "V43_FINAL_PLAN3_FRESH_REPLICATION_V1",
        "parent_design_release_hash": historical.get("design_release_hash"),
        "fresh_c3e_release_hash": c3e_hash,
        "fresh_c3e_actions_sha256": _sha256(actions_path),
        "fresh_c3e_probabilities_sha256": _sha256(probabilities_path),
        "c6ex_permutation_sha256": c6ex_manifest["parent_historical_permutation_sha256"],
        "fresh_c6ex_materialized_sha256": c6ex_manifest["fresh_materialized_sha256"],
        "experiment_matrix_sha256": _sha256(config_root / "experiment_matrix.csv"),
        "action_contract_sha256": _sha256(action_contract_path),
    }
    derived_hash = canonical_hash(identity)
    design["design_release_hash"] = derived_hash
    design["active_reference_policy"]["release_hash"] = c3e_hash
    design["active_reference_policy"]["decision_sha256"] = identity["fresh_c3e_actions_sha256"]
    design["active_reference_policy"]["design_binding_hash"] = derived_hash
    design["C6EX_permutation"] = "C6EX_permutation.parquet"
    design["C6EX_materialized"] = "C6EX_materialized.parquet"
    design["C6EX_manifest"] = "C6EX_manifest.json"
    semantic = json.loads(json.dumps(design))
    semantic.pop("design_release_hash", None)
    semantic.get("active_reference_policy", {}).pop("design_binding_hash", None)
    historical["design_release_hash"] = derived_hash
    historical["c3e_release_hash"] = c3e_hash
    historical["final_design_semantic_hash"] = canonical_hash(semantic)
    historical["parent_design_release_hash"] = identity["parent_design_release_hash"]
    historical["derived_identity"] = identity
    source_paths = {
        "authority_document": paths.root / design["authority_document"],
        "experiment_matrix": config_root / "experiment_matrix.csv",
        "model_contract": config_root / "model_contract.json",
        "prompt_contract": config_root / "prompt_contract.json",
        "feature_dictionary": config_root / "feature_dictionary.csv",
        "information_contract": config_root / "information_contract.json",
        "retry_policy": config_root / "retry_policy.json",
        "action_contract": action_contract_path,
        "c3e_definition": paths.c3e_root / "C3E_definition.json",
        "c3e_probabilities": probabilities_path,
        "c3e_actions": actions_path,
        "c6ex_permutation": config_root / "C6EX_permutation.parquet",
        "c6ex_materialized": config_root / "C6EX_materialized.parquet",
        "c6ex_manifest": config_root / "C6EX_manifest.json",
    }
    for name in historical.get("source_hashes", {}):
        if name in source_paths:
            continue
        relative = {
            "runtime_contract": "src/credit_recourse/final_release/contract.py",
            "runtime_llm_contract": "src/credit_recourse/final_release/llm_contract.py",
            "runtime_parser": "src/credit_recourse/final_release/parsing.py",
            "runtime_prompting": "src/credit_recourse/final_release/prompting.py",
            "runtime_provider_adapters": "src/credit_recourse/final_release/providers.py",
            "runtime_retry_policy": "src/credit_recourse/final_release/retry_policy.py",
            "runtime_ledger": "src/credit_recourse/final_release/ledger.py",
            "runtime_executor": "src/credit_recourse/final_release/executor.py",
            "runtime_collection": "src/credit_recourse/final_release/collection.py",
            "runtime_doctor": "src/credit_recourse/final_release/doctor.py",
            "runtime_failure_coder": "src/credit_recourse/final_release/failure_coder.py",
            "runtime_action_validation": "src/credit_recourse/final_release/action_validation.py",
            "runtime_evaluation_bridge": "src/credit_recourse/final_release/evaluation_bridge.py",
            "runtime_semantic_fixture": "src/credit_recourse/final_release/semantic_fixture.py",
            "runtime_v43_production_bundle": "src/credit_recourse/simulator/v43_production_bundle.py",
            "stage8_evaluator": "src/credit_recourse/eval/final_stage8_llm_multi_oracle_eval/pipeline.py",
            "stage8_failure_enrichment": "src/credit_recourse/eval/final_stage8_llm_multi_oracle_eval/failure_enrichment.py",
            "stage9_evaluator": "src/credit_recourse/eval/final_stage9_llm_rl_comparison/pipeline.py",
            "stage9_revision_metrics": "src/credit_recourse/eval/final_stage9_llm_rl_comparison/revision_metrics.py",
        }
        if name in relative:
            source_paths[name] = paths.root / relative[name]
    historical["source_hashes"] = {name: _sha256(path) for name, path in source_paths.items() if path.is_file()}
    design_path.write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    release_path.write_text(json.dumps(historical, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"design_release_hash": derived_hash, "c3e_release_hash": c3e_hash, "final_design_semantic_hash": historical["final_design_semantic_hash"], "source_hashes": historical["source_hashes"]}


def _resolve_c6ex_permutation(paths, config_root: Path) -> dict[str, Any]:
    """Resolve exact C6-EX bytes from an override or the certified distribution."""
    config_root.mkdir(parents=True, exist_ok=True)
    explicit_permutation = paths.root / "data/design/C6EX_permutation.parquet"
    if explicit_permutation.is_file():
        actual_hash = _sha256(explicit_permutation)
        if actual_hash != C6EX_PERMUTATION_SHA256:
            return {"status": "FAILED", "reason": "C6EX_PERMUTATION_HASH_MISMATCH", "expected_sha256": C6EX_PERMUTATION_SHA256, "actual_sha256": actual_hash, "source": _source_label(paths, explicit_permutation)}
        provenance = {
            "source_role": "EXPLICIT_RESTORED_DESIGN_INPUT",
            "source": _source_label(paths, explicit_permutation),
            "permutation_sha256": actual_hash,
        }
        shutil.copy2(explicit_permutation, config_root / "C6EX_permutation.parquet")
        return {"status": "PASS", "path": config_root / "C6EX_permutation.parquet", "provenance": provenance}

    distribution = paths.root / CERTIFIED_DISTRIBUTION_RELATIVE
    if not distribution.is_file():
        return {"status": "INPUT_REQUIRED", "reason": "C6EX_PERMUTATION_REQUIRED", "expected_sha256": C6EX_PERMUTATION_SHA256}
    distribution_sha256 = _sha256(distribution)
    if distribution_sha256 != CERTIFIED_DISTRIBUTION_SHA256:
        return {"status": "FAILED", "reason": "CERTIFIED_DISTRIBUTION_HASH_MISMATCH", "expected_distribution_sha256": CERTIFIED_DISTRIBUTION_SHA256, "actual_distribution_sha256": distribution_sha256}
    try:
        with zipfile.ZipFile(distribution) as archive:
            members = [name for name in archive.namelist() if name.endswith("configs/current/llm/C6EX_permutation.parquet")]
            if len(members) != 1:
                return {"status": "INPUT_REQUIRED", "reason": "C6EX_PERMUTATION_REQUIRED", "expected_sha256": C6EX_PERMUTATION_SHA256, "member_count": len(members)}
            member = members[0]
            permutation_bytes = archive.read(member)
    except (OSError, zipfile.BadZipFile) as exc:
        return {"status": "FAILED", "reason": "CERTIFIED_DISTRIBUTION_UNREADABLE", "error": type(exc).__name__}
    member_hash = hashlib.sha256(permutation_bytes).hexdigest()
    if member_hash != C6EX_PERMUTATION_SHA256:
        return {"status": "FAILED", "reason": "C6EX_BUNDLED_PERMUTATION_HASH_MISMATCH", "expected_sha256": C6EX_PERMUTATION_SHA256, "actual_sha256": member_hash, "member": member}
    (config_root / "C6EX_permutation.parquet").write_bytes(permutation_bytes)
    return {
        "status": "PASS",
        "path": config_root / "C6EX_permutation.parquet",
        "provenance": {
            "source_role": "CERTIFIED_BUNDLED_DESIGN_INPUT",
            "distribution_zip": _source_label(paths, distribution),
            "distribution_zip_sha256": distribution_sha256,
            "member": member,
            "permutation_sha256": member_hash,
        },
    }


def prepare_real_plan3_config(paths) -> dict[str, Any]:
    """Materialize immutable design inputs and derive a same-run Plan-3 release."""
    config_root = paths.llm_root / "config"
    source_root = paths.root / "frozen/evidence/llm"
    config_root.mkdir(parents=True, exist_ok=True)
    names = ("final_design.json", "design_release.json", "experiment_matrix.csv", "feature_dictionary.csv", "information_contract.json", "model_contract.json", "prompt_contract.json", "retry_policy.json", "analysis_contract.json", "C6EX_manifest.json")
    for name in names:
        source = source_root / name
        if not source.is_file():
            return {"status": "INPUT_REQUIRED", "reason": "FINAL_PLAN3_DESIGN_INPUT_MISSING", "missing": str(source.relative_to(paths.root)).replace("\\", "/")}
        shutil.copy2(source, config_root / name)
    expected_hash = C6EX_PERMUTATION_SHA256
    resolved_permutation = _resolve_c6ex_permutation(paths, config_root)
    if resolved_permutation.get("status") != "PASS":
        return resolved_permutation
    permutation = resolved_permutation["path"]
    permutation_provenance = resolved_permutation["provenance"]
    import pandas as pd
    actions_path = paths.c3e_root / "C3E_firm_actions.parquet"
    if not actions_path.is_file():
        return {"status": "INPUT_REQUIRED", "reason": "fresh C3-E actions are required before LLM preparation"}
    permutation_frame = pd.read_parquet(permutation)
    action_frame = pd.read_parquet(actions_path)
    if len(action_frame) != 575 or len(permutation_frame) != 575:
        return {"status": "FAILED", "reason": "C6-EX requires 575 fresh evaluation firms"}
    from credit_recourse.final_release.llm_contract import build_c6ex_materialization
    try:
        materialized, c6ex_diagnostics = build_c6ex_materialization(
            permutation_frame,
            action_frame,
            expected_rows=FIRM_COUNT,
        )
    except Exception as exc:
        return {"status": "FAILED", "reason": str(exc)}
    materialized.to_parquet(config_root / "C6EX_materialized.parquet", index=False)
    c3e_release_hash = json.loads((paths.c3e_root / "release.json").read_text(encoding="utf-8")).get("release_hash")
    materialized_sha256 = _sha256(config_root / "C6EX_materialized.parquet")
    manifest = {
        "schema_version": "v43_c6ex_fresh_materialization_v1",
        "status": "PASS",
        "c3e_release_hash": c3e_release_hash,
        "permutation_sha256": expected_hash,
        "materialized_sha256": materialized_sha256,
        "parent_historical_permutation_sha256": expected_hash,
        "fresh_c3e_release_hash": c3e_release_hash,
        "fresh_materialized_sha256": materialized_sha256,
        "donor_relation_reused": True,
        "distribution_preserved": c6ex_diagnostics["distribution_preserved"],
        "firm_count": c6ex_diagnostics["firm_count"],
        "self_donor_collision_count": c6ex_diagnostics["self_donor_collision_count"],
        "self_donor_collisions": c6ex_diagnostics["self_donor_collision_count"],
        "action_label_collision_count": c6ex_diagnostics["action_label_collision_count"],
        "rows": c6ex_diagnostics["firm_count"],
        "permutation_provenance": permutation_provenance,
    }
    (config_root / "C6EX_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    industry = _materialize_fresh_industry_binding(paths, config_root)
    if industry.get("status") != "PASS":
        return industry
    release = _write_fresh_design_release(paths, config_root, manifest)
    return {"status": "PASS", "config_root": str(config_root.relative_to(paths.root)).replace("\\", "/"), "c6ex_permutation_sha256": expected_hash, "c6ex_materialized_sha256": manifest["fresh_materialized_sha256"], "c6ex_permutation_provenance": permutation_provenance, "matrix_cells": 42, "industry_binding": industry, "derived_release": release}


def prepare_real_llm(paths) -> dict[str, Any]:
    with _real_runtime_environment(paths):
        prepared = prepare_real_plan3_config(paths)
        if prepared.get("status") != "PASS":
            return prepared
        from credit_recourse.final_release.executor import freeze_release, prepare_wave
        from credit_recourse.high_reasoning.runner import freeze as freeze_high, prepare as prepare_high
        baseline_dir, baseline_release = freeze_release(paths.root)
        baseline_hash = baseline_release["protocol_release_hash"]
        prepare_wave(paths.root, baseline_hash, 1)
        high_dir, high_release = freeze_high(paths.root, baseline_hash)
        high_hash = high_release["protocol_release_hash"]
        prepare_high(paths.root, high_hash, 1, preflight=False)
        manifest = {"status": "PREPARED", "baseline_hash": baseline_hash, "high_hash": high_hash, "config_root": str((paths.llm_root / "config").relative_to(paths.root)).replace("\\", "/")}
        (paths.llm_root / "real_llm_release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest


def execute_real_llm(paths, *, resume: bool) -> dict[str, Any]:
    with _real_runtime_environment(paths):
        if not os.environ.get("OPENAI_API_KEY") or not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            return {"status": "CREDENTIALS_REQUIRED", "reason": "both OpenAI and Gemini provider credentials are required before submission"}
        if os.environ.get("CREDIT_RECOURSE_ENABLE_LLM_V43_FINAL_PLAN3") != "I_APPROVE_LLM_V43_FINAL_PLAN3_BATCH":
            return {"status": "APPROVAL_REQUIRED", "reason": "live LLM sentinel is required"}
        manifest_path = paths.llm_root / "real_llm_release_manifest.json"
        if not manifest_path.is_file():
            prepared = prepare_real_llm(paths)
            if prepared.get("status") != "PREPARED":
                return prepared
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        from credit_recourse.final_release.live_batch import poll_and_download, submit_wave
        from credit_recourse.final_release.retry import prepare_retry
        from credit_recourse.final_release.executor import generation_status, load_release

        def retry_manifest(release_hash: str, wave: int, attempt: int) -> Path:
            directory, _ = load_release(paths.root, release_hash)
            return directory / "manifests" / f"wave{int(wave)}_attempt{int(attempt)}_retry_prepared.json"

        def prepare_retry_after_terminal(release_hash: str, wave: int, attempt: int) -> dict[str, Any]:
            path = retry_manifest(release_hash, wave, attempt)
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
            return prepare_retry(paths.root, release_hash, wave, attempt)

        def no_retry_outcome(release_hash: str, reason: str) -> dict[str, Any]:
            generation = generation_status(paths.root, release_hash)
            pending = any(
                int(generation.get("ledger_states", {}).get(state, 0)) > 0
                for state in ("SUBMITTED", "SUBMITTING")
            )
            return {
                **manifest,
                "status": "EXTERNAL_WAIT" if pending else "FAILED",
                "reason": reason if pending else f"terminal generation is incomplete without an eligible retry: {reason}",
                "generation": generation,
            }

        def arm_completion(release_hash: str, arm: str) -> dict[str, Any] | None:
            generation = generation_status(paths.root, release_hash)
            if generation.get("status") == "GENERATION_COMPLETE":
                return None
            active_states = {"SUBMITTED", "SUBMITTING"}
            has_active_provider_work = any(
                int(generation.get("ledger_states", {}).get(state, 0)) > 0
                for state in active_states
            )
            return {
                **manifest,
                "status": "EXTERNAL_WAIT" if has_active_provider_work else "FAILED",
                "release_hash": release_hash,
                "arm": arm,
                "generation": generation,
                "reason": (
                    f"{arm} generation remains incomplete with active provider work"
                    if has_active_provider_work
                    else f"{arm} generation ledger is terminal but incomplete"
                ),
            }

        completed_generations: dict[str, dict[str, Any]] = {}

        for arm in (("baseline_hash", "baseline"), ("high_hash", "high")):
            release_hash = manifest[arm[0]]
            for wave in (1, 2):
                from credit_recourse.final_release.executor import prepare_wave
                if wave == 2:
                    prepare_wave(paths.root, release_hash, wave) if arm[1] == "baseline" else __import__("credit_recourse.high_reasoning.runner", fromlist=["prepare"]).prepare(paths.root, release_hash, wave, preflight=False)
                for attempt in (1, 2, 3):
                    if attempt > 1:
                        retry = prepare_retry_after_terminal(release_hash, wave, attempt)
                        if retry.get("status") == "NO_RETRY_NEEDED":
                            return no_retry_outcome(release_hash, "prepare_retry returned NO_RETRY_NEEDED")
                        if retry.get("status") != "PREPARED":
                            return {
                                **manifest,
                                "status": "FAILED",
                                "reason": "retry preparation did not produce a PREPARED manifest",
                                "retry": retry,
                            }
                    submit_wave(paths.root, release_hash, wave, approved=True, attempt_index=attempt)
                    polled = poll_and_download(paths.root, release_hash, wave, attempt_index=attempt)
                    if not polled.get("all_terminal_or_downloaded") and "status" not in polled:
                        # A provider job that is still running is an external
                        # wait, not a failed attempt and never a retry trigger.
                        return {**manifest, "status": "EXTERNAL_WAIT", "reason": "provider jobs remain incomplete"}
                    generation = generation_status(paths.root, release_hash)
                    if polled.get("status") in {"PASS", "GENERATION_COMPLETE", "COLLECTED"}:
                        break
                    if polled.get("status") in {"RETRY_REQUIRED", "INCOMPLETE_INFRA"}:
                        if attempt == 3:
                            return {**manifest, "status": "FAILED", "reason": "provider retries exhausted", "generation": generation}
                        # The next iteration may prepare a retry, but only now
                        # that this attempt is terminal.
                        continue
                    if polled.get("status") in {"BLOCKED_RUNTIME", "FAILED"}:
                        if attempt == 3:
                            return {**manifest, "status": "FAILED", "reason": "provider attempt reached a terminal non-success state", "generation": generation}
                    if polled.get("all_terminal_or_downloaded") and "status" not in polled:
                        break
                    return {**manifest, "status": "FAILED", "reason": "provider poll returned an unrecognized terminal state", "poll": polled}
            completed = arm_completion(release_hash, arm[1])
            if completed is not None:
                return completed
            completed_generations[arm[1]] = generation_status(paths.root, release_hash)
        return {
            **manifest,
            "status": "PASS",
            "baseline_generation": completed_generations["baseline"],
            "high_generation": completed_generations["high"],
        }


def materialize_real_stage7(paths) -> dict[str, Any]:
    with _real_runtime_environment(paths):
        from credit_recourse.final_release.evaluation_bridge import materialize_evaluation_inputs
        manifest = json.loads((paths.llm_root / "real_llm_release_manifest.json").read_text(encoding="utf-8"))
        baseline = materialize_evaluation_inputs(paths.root, manifest["baseline_hash"], output_dir=paths.llm_root / "baseline_stage7")
        high = materialize_evaluation_inputs(paths.root, manifest["high_hash"], output_dir=paths.llm_root / "high_stage7")
        import pandas as pd
        frames = []
        for regime, directory in (("BASELINE", paths.llm_root / "baseline_stage7"), ("HIGH", paths.llm_root / "high_stage7")):
            frame = pd.read_parquet(directory / "llm_stage7_action_table_itt.parquet")
            frame["reasoning_regime"] = regime
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        if len(combined) != EXPECTED_REQUESTS:
            raise ValueError("real Stage7 combined ITT cardinality drift")
        combined.to_parquet(paths.llm_root / "llm_stage7_action_table_itt.parquet", index=False)
        metadata = {"status": "PASS", "scientific_contract_version": "V4.3_FINAL_20260912", "backend_is_live": True, "final_paper_run_allowed": True, "baseline_logical_requests": 24150, "high_logical_requests": 24150, "total_logical_requests": 48300, **manifest}
        (paths.llm_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return metadata


def fresh_contract_ready(root: Path = ROOT) -> bool:
    root = Path(root)
    return (root / "contracts/scientific/v43_action_contract.json").is_file() and (root / "frozen/evidence/llm/experiment_matrix.csv").is_file()


def live_runner_ready() -> bool:
    return True


@contextmanager
def scoped_live_gate(enabled: bool):
    name = "CREDIT_RECOURSE_ENABLE_LLM_V43_FINAL_PLAN3"
    previous = os.environ.get(name)
    if enabled:
        os.environ[name] = "I_APPROVE_LLM_V43_FINAL_PLAN3_BATCH"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
