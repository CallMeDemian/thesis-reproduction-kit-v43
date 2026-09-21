from __future__ import annotations

import collections
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .action_validation import CANDIDATES, DIMENSIONS
from credit_recourse.contracts.runtime_assets import active_action_contract, active_stage2_eval_ids, active_stage2_eval_panel
from .common import ContractError, canonical_hash, file_hash, find_repo_root, load_json


MODEL_KEYS = ("openai_gpt54mini", "google_gemini31flashlite")
CONDITIONS = ("C4", "C5", "C4R", "C6-E", "C6-EX")
FINANCIAL_FIELDS = (
    "derived__debt_to_assets", "derived__equity_to_assets", "derived__current_ratio",
    "derived__cash_ratio", "derived__operating_margin", "derived__gross_margin",
    "derived__net_margin", "derived__roa_proxy", "derived__cogs_to_revenue",
    "derived__sga_to_revenue", "derived__financial_cost_to_revenue",
    "derived__capex_to_revenue", "derived__ppe_to_assets",
    "derived__short_debt_to_total_debt", "derived__long_debt_to_total_debt",
    "derived__bond_to_total_debt", "derived__inventory_to_revenue",
    "derived__receivables_to_revenue", "derived__payables_to_revenue",
    "delta_1y__derived__operating_margin", "delta_1y__derived__debt_to_assets",
    "delta_1y__derived__current_ratio", "delta_1y__derived__roa_proxy",
)


@dataclass(frozen=True)
class DesignBundle:
    root: Path
    design: dict[str, Any]
    release: dict[str, Any]
    matrix: list[dict[str, str]]
    models: dict[str, Any]
    prompts: dict[str, Any]
    information: dict[str, Any]
    retry: dict[str, Any]
    action_contract: dict[str, Any]
    design_release_hash: str


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _semantic_design(value: Mapping[str, Any]) -> dict[str, Any]:
    import json
    out = json.loads(json.dumps(dict(value)))
    out.pop("design_release_hash", None)
    out.get("active_reference_policy", {}).pop("design_binding_hash", None)
    return out


def load_design(root: Path | None = None) -> DesignBundle:
    repo = (root or find_repo_root()).resolve()
    configured = os.environ.get("THESIS_REPRO_LLM_CONFIG_ROOT")
    base = Path(configured) if configured and Path(configured).is_absolute() else (repo / configured if configured else repo / "frozen/evidence/llm")
    design = load_json(base / "final_design.json")
    release = load_json(base / "design_release.json")
    matrix = _csv(base / "experiment_matrix.csv")
    models = load_json(base / "model_contract.json")
    prompts = load_json(base / "prompt_contract.json")
    information = load_json(base / "information_contract.json")
    retry = load_json(base / "retry_policy.json")
    action_path = active_action_contract(repo)
    action = load_json(action_path)
    if len(matrix) != 42 or len({row["cell_id"] for row in matrix}) != 42:
        raise ContractError("Final matrix must contain 42 unique cells")
    if collections.Counter(row["model"] for row in matrix) != collections.Counter({k: 21 for k in MODEL_KEYS}):
        raise ContractError("Final matrix must contain 21 cells per model")
    if sum(int(row["expected_requests"]) for row in matrix) != 24_150:
        raise ContractError("Final matrix must contain 24,150 logical requests")
    if set(row["condition"] for row in matrix) != set(CONDITIONS):
        raise ContractError("Final condition inventory drift")
    lookup = {row["cell_id"]: row for row in matrix}
    for row in matrix:
        parent_id = row["parent_cell_id"].strip()
        expected_parent = row["condition"] in {"C4R", "C6-E", "C6-EX"}
        if expected_parent:
            if parent_id not in lookup or lookup[parent_id]["condition"] != "C4":
                raise ContractError(f"Invalid C4 sibling lineage: {row['cell_id']}")
            for key in ("model", "mode", "info", "budget", "replicate"):
                if row[key] != lookup[parent_id][key]:
                    raise ContractError(f"Parent factor drift: {row['cell_id']} / {key}")
        elif parent_id:
            raise ContractError(f"Unexpected parent: {row['cell_id']}")
        if row["phase"] == "STABILITY" and row["replicate"] != "2":
            raise ContractError("Stability run must use replicate 2")
    if tuple(action["candidate_ids"]) != CANDIDATES:
        raise ContractError("Candidate action order drift")
    if tuple(c.removeprefix("action__") for c in action["action_columns"]) != DIMENSIONS:
        raise ContractError("Action dimension order drift")
    if canonical_hash(_semantic_design(design)) != release["final_design_semantic_hash"]:
        raise ContractError("Final design semantic hash drift")
    c3e_root = Path(os.environ.get("THESIS_REPRO_C3E_ROOT", repo / "archive/DEPLOYED_RELEASE/stage5_candidate_iql/C3E_E2_7SEED_BALANCED_DFEBAFA6"))

    def resolve_hash_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo / path

    for name, expected in release["source_hashes"].items():
        rel = {
            "authority_document": design["authority_document"],
            "experiment_matrix": str(base / "experiment_matrix.csv"),
            "model_contract": str(base / "model_contract.json"),
            "prompt_contract": str(base / "prompt_contract.json"),
            "feature_dictionary": str(base / "feature_dictionary.csv"),
            "information_contract": str(base / "information_contract.json"),
            "retry_policy": str(base / "retry_policy.json"),
            "analysis_contract": str(base / "analysis_contract.json"),
            "action_contract": str(active_action_contract(repo)),
            "c3e_definition": str(c3e_root / "C3E_definition.json"),
            "c3e_probabilities": str(c3e_root / "C3E_firm_probabilities.parquet"),
            "c3e_actions": str(c3e_root / "C3E_firm_actions.parquet"),
            "c6ex_permutation": str(base / "C6EX_permutation.parquet"),
            "c6ex_materialized": str(base / "C6EX_materialized.parquet"),
            "c6ex_manifest": str(base / "C6EX_manifest.json"),
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
        }[name]
        if file_hash(resolve_hash_path(rel)) != expected:
            raise ContractError(f"Released source hash drift: {name}")
    return DesignBundle(repo, design, release, matrix, models, prompts, information, retry, action, release["design_release_hash"])


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return 0.0 if number == 0 else float(format(number, ".10g"))


def load_firm_cohort(design: DesignBundle) -> pd.DataFrame:
    ids = pd.read_parquet(active_stage2_eval_ids(design.root))
    panel = pd.read_parquet(active_stage2_eval_panel(design.root))
    ids = ids.sort_values("row_id").reset_index(drop=True)
    panel = panel.copy()
    panel["firm_id"] = panel["firm_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    ids["firm_id"] = ids["firm_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    merged = ids[["row_id", "firm_id", "fiscal_year", "canonical_firm_year_id"]].merge(panel, on=["firm_id", "fiscal_year"], how="left", validate="one_to_one", suffixes=("", "__panel"))
    if len(merged) != 575 or merged["row_id"].tolist() != list(range(575)):
        raise ContractError("Canonical firm cohort order drift")
    missing = sorted(set(FINANCIAL_FIELDS) - set(merged))
    if missing:
        raise ContractError(f"Firm panel missing final prompt fields: {missing}")
    # The original source-name field is the third serving-panel column. It is
    # retained only for IC-c; no firm identifier is exposed in IC-a/IC-b.
    source_name = panel.columns[2]
    name_map = dict(zip(panel["firm_id"], panel[source_name].astype(str)))
    merged["firm_name"] = merged["firm_id"].map(name_map)
    merged["market"] = merged.get("market", pd.Series(["UNKNOWN"] * len(merged))).fillna("UNKNOWN").astype(str)

    # IC-b/IC-c industry values are supplied only by the frozen OpenDART
    # binding. Do not silently fall back to sparse serving-panel metadata.
    config_root = Path(os.environ.get("THESIS_REPRO_LLM_CONFIG_ROOT", design.root / "frozen/evidence/llm"))
    binding_reference = Path(str(design.information["industry_binding_manifest"]))
    binding_manifest = binding_reference if binding_reference.is_absolute() else config_root / binding_reference
    evidence = load_json(binding_manifest) if binding_manifest.is_file() else {}
    configured_root = os.environ.get("THESIS_REPRO_LLM_CONFIG_ROOT")
    role = evidence.get("role")
    historical_authorized = evidence.get("frozen") is True or role == "FROZEN_EXOGENOUS_INFORMATION_INPUT"
    fresh_authorized = bool(configured_root) and role == "FRESH_EXOGENOUS_INFORMATION_INPUT"
    if not (
        (historical_authorized or fresh_authorized)
        and evidence.get("status") == "PASS"
        and evidence.get("unresolved_count") == 0
        and evidence.get("binding_artifact")
    ):
        raise ContractError("Frozen 575-firm industry binding is unavailable")
    binding_reference = Path(str(evidence["binding_artifact"]))
    binding_path = binding_reference if binding_reference.is_absolute() else config_root / binding_reference
    binding = pd.read_parquet(binding_path)
    required_binding = {"firm_key", "induty_code", "industry_display_value"}
    if not required_binding.issubset(binding.columns):
        raise ContractError("Industry binding schema drift")
    if len(binding) != 575 or binding["firm_key"].nunique() != 575:
        raise ContractError("Industry binding cohort cardinality drift")
    if binding["induty_code"].isna().any() or binding["induty_code"].astype(str).str.strip().eq("").any():
        raise ContractError("Industry binding contains missing OpenDART induty_code")
    expected_keys = set(merged["canonical_firm_year_id"].astype(str))
    if set(binding["firm_key"].astype(str)) != expected_keys:
        raise ContractError("Industry binding firm universe drift")
    industry = binding[["firm_key", "industry_display_value"]].copy()
    merged = merged.merge(industry, left_on="canonical_firm_year_id", right_on="firm_key", how="left", validate="one_to_one")
    if merged["industry_display_value"].isna().any() or merged["industry_display_value"].astype(str).str.strip().eq("").any():
        raise ContractError("Industry binding did not supply every canonical firm")
    merged["industry_class"] = merged["industry_display_value"].astype(str)
    for field in (*FINANCIAL_FIELDS, "log_assets"):
        merged[field] = merged[field].map(_number)
    merged["firm_key"] = merged["canonical_firm_year_id"].astype(str)
    return merged


def _json_scalar(value: Any) -> Any:
    """Convert pandas/numpy scalars to strict JSON without imputing values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def state_payload(row: Mapping[str, Any], info: str) -> dict[str, Any]:
    fields = [*FINANCIAL_FIELDS, "fiscal_year", "log_assets", "market"]
    if info in {"IC-b", "IC-c"}:
        fields.append("industry_class")
    if info == "IC-c":
        fields.append("firm_name")
    if info not in {"IC-a", "IC-b", "IC-c"}:
        raise ContractError(f"Unknown information condition: {info}")
    return {field: _json_scalar(row.get(field)) for field in fields}


def candidate_vector(design: DesignBundle, candidate: str) -> dict[str, float]:
    if candidate not in CANDIDATES:
        raise ContractError(f"Unknown candidate: {candidate}")
    raw = design.action_contract["fixed_candidates"][candidate]
    return {name: float(raw[f"action__{name}"]) for name in DIMENSIONS}



