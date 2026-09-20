from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.contracts.stage_paths import stage_dir

from .common import ContractError, sha256_file, utc_now, write_json
from .contract import CONDITIONS, DIMENSIONS, load_design
from credit_recourse.contracts.runtime_assets import active_action_contract
from .executor import generation_status, load_release
from .ledger import connect, ledger_content_hash


ACTION_COLUMNS = tuple(f"action__{name}" for name in DIMENSIONS)


def _reference_fields(condition: str, row_id: int, refs: pd.DataFrame) -> dict[str, Any]:
    if condition == "C6-E":
        row = refs.loc[row_id]
        return {
            "reference_source": "C3-E",
            "reference_row_id": row_id,
            "reference_firm_id": str(row["firm_id"]),
            "reference_candidate_id": str(row["own_C3E_action"]),
        }
    if condition == "C6-EX":
        row = refs.loc[row_id]
        return {
            "reference_source": "C3-EX",
            "reference_row_id": int(row["donor_row_id"]),
            "reference_firm_id": str(row["donor_firm_id"]),
            "reference_candidate_id": str(row["C6EX_reference_action"]),
        }
    return {
        "reference_source": "none",
        "reference_row_id": None,
        "reference_firm_id": None,
        "reference_candidate_id": None,
    }


def _action_record(
    base: dict[str, Any],
    parsed: dict[str, Any],
    *,
    layer: str,
    refs: pd.DataFrame,
) -> dict[str, Any] | None:
    vector = parsed.get(layer)
    if vector is None:
        return None
    if set(vector) != set(DIMENSIONS):
        raise ContractError(f"{layer} action does not have the canonical eight dimensions")
    condition = str(base["condition"])
    row_id = int(base["row_id"])
    candidate = (
        str(parsed.get("candidate_id_req"))
        if base["mode"] == "candidate9" and parsed.get("candidate_id_req")
        else "A0"
        if parsed.get("fallback_applied") and layer == "a_itt"
        else "FREE8"
    )
    rec = {
        **base,
        "policy": condition,
        "information_condition": str(base["info"]),
        "candidate_id": candidate,
        "action_layer": "accepted" if layer == "a_accepted" else "itt",
        "model_response_usable": bool(parsed.get("policy_usable")),
        "itt_noop_fallback_applied": bool(
            layer == "a_itt" and parsed.get("fallback_applied")
        ),
        "action_application_status": str(parsed.get("action_application_status")),
        "action_application_reason": str(parsed.get("action_application_reason")),
        "parser_state": str(parsed.get("parser_state")),
        "completion_kind": str(parsed.get("completion_kind")),
        "action_valid": bool(parsed.get("action_valid")),
        "response_schema_valid": bool(parsed.get("response_schema_valid")),
        "full_response_valid": bool(parsed.get("full_response_valid")),
        **_reference_fields(condition, row_id, refs),
    }
    for name in DIMENSIONS:
        rec[f"action__{name}"] = float(vector[name])
    return rec


def materialize_evaluation_inputs(
    project_root: Path,
    release_hash: str,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    directory, release = load_release(root, release_hash)
    status = generation_status(root, release_hash)
    if status["status"] != "GENERATION_COMPLETE":
        raise ContractError(
            "Evaluation materialization requires all 24,150 logical requests to be DELIVERED_LOCKED"
        )

    design = load_design(root)
    inventory = pd.read_parquet(directory / "logical_requests.parquet")
    if len(inventory) != 24_150 or inventory["request_id"].nunique() != 24_150:
        raise ContractError("Runtime logical-request inventory drift")

    ledger = connect(directory / "generation_ledger.sqlite")
    try:
        rows = ledger.execute(
            "SELECT r.request_id,r.request_state,r.selected_attempt_index,"
            "p.parse_json FROM logical_requests r "
            "LEFT JOIN parsed_responses p USING(request_id) ORDER BY r.request_id"
        ).fetchall()
        ledger_hash = ledger_content_hash(ledger)
    finally:
        ledger.close()
    parsed = pd.DataFrame([dict(row) for row in rows])
    if len(parsed) != 24_150:
        raise ContractError("Ledger logical-request cardinality drift")
    if not parsed["request_state"].eq("DELIVERED_LOCKED").all():
        raise ContractError("Evaluation materialization refuses unlocked requests")
    if parsed["parse_json"].isna().any():
        raise ContractError("Every locked response must have an atomic parse record")

    merged = inventory.merge(
        parsed[["request_id", "selected_attempt_index", "parse_json"]],
        on="request_id",
        how="inner",
        validate="one_to_one",
    )
    config_root = Path(os.environ.get("THESIS_REPRO_LLM_CONFIG_ROOT", root / "frozen/evidence/llm"))
    refs = pd.read_parquet(config_root / "C6EX_materialized.parquet")
    refs = refs.set_index("row_id", drop=False)
    if len(refs) != 575:
        raise ContractError("C6-EX materialized reference table is not 575 rows")

    layers: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    itt: list[dict[str, Any]] = []
    for raw in merged.to_dict("records"):
        parse = json.loads(raw.pop("parse_json"))
        base = {
            key: raw[key]
            for key in (
                "request_id",
                "protocol_release_hash",
                "cell_id",
                "firm_key",
                "row_id",
                "model_key",
                "phase",
                "mode",
                "info",
                "budget",
                "condition",
                "replicate",
                "wave",
                "parent_cell_id",
                "parent_request_id",
            )
        }
        layer_row = {
            **base,
            "selected_attempt_index": int(raw["selected_attempt_index"]),
            "completion_kind": parse.get("completion_kind"),
            "parser_state": parse.get("parser_state"),
            "policy_usable": bool(parse.get("policy_usable")),
            "response_schema_valid": bool(parse.get("response_schema_valid")),
            "fallback_applied": bool(parse.get("fallback_applied")),
            "action_application_status": parse.get("action_application_status"),
            "action_application_reason": parse.get("action_application_reason"),
            "candidate_id_req": parse.get("candidate_id_req"),
            "a_req_json": json.dumps(parse.get("a_req"), sort_keys=True),
            "a_accepted_json": json.dumps(parse.get("a_accepted"), sort_keys=True),
            "a_itt_json": json.dumps(parse.get("a_itt"), sort_keys=True),
            "error_codes_json": json.dumps(parse.get("error_codes") or []),
        }
        layers.append(layer_row)
        pp = _action_record(base, parse, layer="a_accepted", refs=refs)
        if pp is not None:
            pp["analysis_population"] = "per_protocol"
            accepted.append(pp)
        itt_row = _action_record(base, parse, layer="a_itt", refs=refs)
        if itt_row is not None:
            itt_row["analysis_population"] = "itt"
            itt.append(itt_row)

    if len(itt) != 24_150:
        raise ContractError("ITT table must contain one action for every delivered logical request")
    if any(not row["model_response_usable"] for row in accepted):
        raise ContractError("Per-protocol table contains an unusable response")
    if set(merged["condition"].astype(str)) != set(CONDITIONS):
        raise ContractError("Materialized conditions differ from the final Plan-3 contract")

    stage7 = Path(output_dir) if output_dir is not None else stage_dir(root, "stage7")
    stage7.mkdir(parents=True, exist_ok=True)
    layers_df = pd.DataFrame(layers)
    pp_df = pd.DataFrame(accepted)
    itt_df = pd.DataFrame(itt)
    layers_df.to_parquet(stage7 / "llm_response_action_layers.parquet", index=False)
    pp_df.to_parquet(stage7 / "llm_stage7_action_table_per_protocol.parquet", index=False)
    itt_df.to_parquet(stage7 / "llm_stage7_action_table_itt.parquet", index=False)
    failure = itt_df[
        [
            "request_id",
            "cell_id",
            "row_id",
            "firm_key",
            "model_key",
            "policy",
            "mode",
            "information_condition",
            "budget",
            "replicate",
            "model_response_usable",
            "action_application_status",
        ]
    ].copy()
    failure["routed_to_simulator"] = True
    failure["failure_categories"] = failure["model_response_usable"].map(
        lambda usable: "" if bool(usable) else "translational_failure"
    )
    failure.to_csv(
        stage7 / "llm_stage7_failure_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # The unified accepted-action table is used only for revision-behavior
    # analysis. ITT remains a separate explicit population.
    pp_df.to_parquet(stage7 / "llm_stage7_action_table.parquet", index=False)

    metadata = {
        "schema_version": "llm_v43_final_plan3_stage7_materialization_v1",
        "stage": "stage7_llm_action_generation",
        "status": "PASS",
        "created_utc": utc_now(),
        "protocol_release_hash": release_hash,
        "runtime_release": str((directory / "release.json").relative_to(root).as_posix()),
        "generation_ledger": str((directory / "generation_ledger.sqlite").relative_to(root).as_posix()),
        "generation_ledger_content_hash": ledger_hash,
        "backend_is_live": True,
        "final_paper_run_allowed": True,
        "scientific_contract_version": "V4.3_FINAL_20260912",
        "action_semantic_contract_version": design.action_contract[
            "simulator_action_semantic_contract_version"
        ],
        "final_action_contract_hash": sha256_file(active_action_contract(root)),
        "conditions": list(CONDITIONS),
        "action_dimensions": list(DIMENSIONS),
        "logical_requests": 24_150,
        "per_protocol_rows": int(len(pp_df)),
        "itt_rows": int(len(itt_df)),
        "terminal_unusable_itt_a0_rows": int(
            itt_df["itt_noop_fallback_applied"].sum()
        ),
        "unresolved_infrastructure_rows": 0,
        "no_clipping": True,
        "no_projection": True,
        "no_rescale": True,
        "outputs": {
            "response_action_layers": "llm_response_action_layers.parquet",
            "per_protocol": "llm_stage7_action_table_per_protocol.parquet",
            "itt": "llm_stage7_action_table_itt.parquet",
            "revision_actions": "llm_stage7_action_table.parquet",
        },
    }
    write_json(stage7 / "metadata.json", metadata)
    return metadata
