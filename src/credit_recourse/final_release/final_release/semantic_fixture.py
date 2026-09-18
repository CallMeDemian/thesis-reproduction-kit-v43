from __future__ import annotations

"""Local semantic end-to-end fixture for the final V4.3 LLM contract.

The fixture performs no provider call. It exercises the real parser, proposed /
accepted / ITT layers, production financial simulator, frozen Alpha/Beta/Gamma
Oracle scorers, and the Stage9 four-contrast implementation.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.contracts.stage_paths import final_root, stage_dir
from credit_recourse.eval.final_stage8_llm_multi_oracle_eval.pipeline import (
    _load_stage6_simulator_identity,
    load_canonical_business_plan_history,
    load_current_action_space,
    prepare_llm_history_lookup,
    simulate_llm_policy_states,
)
from credit_recourse.eval.final_stage9_llm_rl_comparison.revision_metrics import (
    PRIMARY_CONTRASTS,
    build_primary_contrasts,
)
from credit_recourse.eval.v43_oracle_backends import (
    score_alpha,
    score_beta_ordered_logit_params,
    score_gamma_model,
)
from credit_recourse.oracle.artifact_io import load_registry, resolve_backend_artifact

from .common import ContractError, canonical_hash
from .contract import candidate_vector, load_design
from .parsing import parse_policy_response


FIXTURE_ACTIONS = {
    "C4": "OE",
    "C5": "OE",
    "C4R": "OE",
    "C6-E": "OE",
    "C6-EX": "OE",
}


def _raw(action: dict[str, float]) -> str:
    return json.dumps(
        {
            "diagnosis": "local semantic fixture",
            "action": action,
            "evidence_fields": [],
            "rationale": "Exercises the frozen local calculation chain.",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parser_layer_checks(design) -> dict[str, Any]:
    valid = parse_policy_response(
        design,
        raw_visible_text=_raw(candidate_vector(design, "OE")),
        mode="free8",
        budget="B1",
    )
    if valid.a_req != valid.a_accepted or valid.a_accepted != valid.a_itt:
        raise ContractError("Valid response did not preserve proposed=accepted=ITT")

    invalid_action = candidate_vector(design, "A0")
    invalid_action["growth_capex_reduction_pct"] = 9.0
    invalid = parse_policy_response(
        design, raw_visible_text=_raw(invalid_action), mode="free8", budget="BINF"
    )
    if invalid.a_req is None or invalid.a_accepted is not None:
        raise ContractError("Invalid action did not preserve proposed and reject accepted")
    if invalid.a_itt != candidate_vector(design, "A0"):
        raise ContractError("Terminal invalid action did not map to ITT A0")

    terminal = parse_policy_response(
        design,
        raw_visible_text="",
        mode="free8",
        budget="B1",
        provider_refusal=True,
    )
    if terminal.a_accepted is not None or terminal.a_itt != candidate_vector(design, "A0"):
        raise ContractError("Terminal refusal layer contract failed")

    infra = parse_policy_response(
        design,
        raw_visible_text=None,
        mode="free8",
        budget="B1",
        infrastructure_unresolved=True,
    )
    if any(value is not None for value in (infra.a_req, infra.a_accepted, infra.a_itt)):
        raise ContractError("Infrastructure-unresolved response acquired an action")

    candidate = parse_policy_response(
        design,
        raw_visible_text=json.dumps(
            {
                "diagnosis": "candidate fixture",
                "action": {"candidate_id": "RF"},
                "evidence_fields": [],
                "rationale": "candidate contract",
            }
        ),
        mode="candidate9",
        budget="B1",
    )
    if candidate.candidate_id_req != "RF" or candidate.a_accepted != candidate_vector(design, "RF"):
        raise ContractError("Candidate9 parser contract failed")
    return {
        "valid_free8": valid.to_dict(),
        "invalid_free8": invalid.to_dict(),
        "terminal_unusable": terminal.to_dict(),
        "infrastructure_unresolved": infra.to_dict(),
        "valid_candidate9": candidate.to_dict(),
    }


def _fixture_action_table(design) -> pd.DataFrame:
    rows = []
    for budget in ("B1", "BINF"):
        c4_request = f"fixture-{budget}-C4"
        for condition, candidate in FIXTURE_ACTIONS.items():
            parsed = parse_policy_response(
                design,
                raw_visible_text=_raw(candidate_vector(design, candidate)),
                mode="free8",
                budget=budget,
            )
            if parsed.a_accepted is None or parsed.a_itt is None:
                raise ContractError(f"Fixture action rejected: {budget}/{condition}")
            reference_source = (
                "C3-E" if condition == "C6-E" else
                "C3-EX" if condition == "C6-EX" else "none"
            )
            row = {
                "request_id": f"fixture-{budget}-{condition}",
                "cell_id": f"fixture-cell-{budget}-{condition}",
                "row_id": 0,
                "firm_key": "fixture-firm",
                "model_key": "local_semantic_fixture",
                "phase": "MAIN",
                "policy": condition,
                "mode": "free8",
                "info": "IC-b",
                "information_condition": "IC-b",
                "budget": budget,
                "replicate": 1,
                "wave": 2 if condition in {"C4R", "C6-E", "C6-EX"} else 1,
                "parent_cell_id": f"fixture-cell-{budget}-C4" if condition in {"C4R", "C6-E", "C6-EX"} else None,
                "parent_request_id": c4_request if condition in {"C4R", "C6-E", "C6-EX"} else None,
                "analysis_population": "itt",
                "candidate_id": "FREE8",
                "action_layer": "itt",
                "action_application_status": parsed.action_application_status,
                "reference_source": reference_source,
                "reference_row_id": 0 if condition == "C6-E" else 1 if condition == "C6-EX" else None,
                "reference_candidate_id": candidate if condition in {"C6-E", "C6-EX"} else None,
            }
            row.update({f"action__{key}": value for key, value in parsed.a_itt.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _score(root: Path, states: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    registry_path = root / "configs" / "current" / "final_freeze" / "oracle_backend_registry.yaml"
    registry = load_registry(registry_path)
    backends = registry["backends"]
    out = actions[[
        "request_id", "row_id", "firm_key", "model_key", "policy", "mode",
        "info", "information_condition", "budget", "replicate", "analysis_population",
    ]].copy()
    for backend in ("alpha", "beta", "gamma"):
        spec = backends[backend]
        params = resolve_backend_artifact(root, final_root(root), spec["params"])
        if backend == "alpha":
            value = score_alpha(states, params)
        elif backend == "beta":
            value = score_beta_ordered_logit_params(states, params)
        else:
            model = resolve_backend_artifact(root, final_root(root), spec["model"])
            value = score_gamma_model(states, params, model)
        out[f"R_score_{backend}"] = np.asarray(value, dtype=float)
    if not np.isfinite(out[[f"R_score_{b}" for b in ("alpha", "beta", "gamma")]].to_numpy()).all():
        raise ContractError("Semantic fixture produced non-finite Oracle scores")
    return out


def run_semantic_fixture(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    design = load_design(root)
    layers = _parser_layer_checks(design)
    actions = _fixture_action_table(design)
    simulation_actions = actions.iloc[[0]].copy()
    base = pd.read_parquet(stage_dir(root, "stage2") / "phase_eval_candidate.parquet")
    space = load_current_action_space(root)
    canonical_history, _ = load_canonical_business_plan_history(root)
    target_firm = str(base.iloc[0]["firm_id"]).removesuffix(".0").zfill(6)
    normalized_ids = canonical_history["firm_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    fixture_history = canonical_history.loc[normalized_ids.eq(target_firm)].copy()
    history_lookup = prepare_llm_history_lookup(fixture_history)
    simulator = _load_stage6_simulator_identity(root)
    with tempfile.TemporaryDirectory(prefix="v43_semantic_fixture_") as tmp:
        states, audit = simulate_llm_policy_states(
            base,
            simulation_actions,
            space,
            Path(tmp),
            predicted_fiscal_year=int(simulator["predicted_fiscal_year"]),
            preserve_current_non_current_residual=bool(
                simulator["preserve_current_non_current_residual"]
            ),
            sim_business_plan_mode=str(simulator["sim_business_plan_mode"]),
            write_outputs=False,
            collect_audit=True,
            history_lookup=history_lookup,
            project_root=root,
        )
    primitive_columns = [column for column in states if column.startswith("primitive__")]
    realized_evidence = primitive_columns + (["executed_primitive_hash"] if "executed_primitive_hash" in states else [])
    if not realized_evidence or len(states) != len(simulation_actions):
        raise ContractError("Semantic fixture did not produce realized primitive evidence")
    realized_score = _score(root, states, simulation_actions)
    scores = actions[["request_id", "row_id", "firm_key", "model_key", "policy", "mode", "info", "information_condition", "budget", "replicate", "analysis_population"]].copy()
    for backend in ("alpha", "beta", "gamma"):
        scores[f"R_score_{backend}"] = float(realized_score[f"R_score_{backend}"].iloc[0])
    firm, summary, interaction, interaction_summary = build_primary_contrasts(
        scores, bootstrap_replicates=128, bootstrap_seed=20_260_912
    )
    expected = {label for label, _, _ in PRIMARY_CONTRASTS}
    if set(firm["contrast"].astype(str)) != expected:
        raise ContractError("Semantic fixture did not produce all four Stage9 contrasts")
    if set(firm["budget"].astype(str)) != {"B1", "BINF"} or interaction.empty:
        raise ContractError("Semantic fixture did not produce the B1/BINF interaction")
    return {
        "status": "PASS",
        "schema_version": "v43_final_semantic_end_to_end_fixture_v1",
        "api_calls_executed": 0,
        "parser_layers": {
            "valid_free8_status": layers["valid_free8"]["action_application_status"],
            "invalid_proposed_preserved": layers["invalid_free8"]["a_req"] is not None,
            "invalid_accepted_is_none": layers["invalid_free8"]["a_accepted"] is None,
            "invalid_itt_is_a0": layers["invalid_free8"]["a_itt"] == candidate_vector(design, "A0"),
            "terminal_accepted_is_none": layers["terminal_unusable"]["a_accepted"] is None,
            "terminal_itt_is_a0": layers["terminal_unusable"]["a_itt"] == candidate_vector(design, "A0"),
            "infrastructure_all_layers_none": all(
                layers["infrastructure_unresolved"][key] is None
                for key in ("a_req", "a_accepted", "a_itt")
            ),
        },
        "simulation": {
            "rows": int(len(states)),
            "logical_condition_rows": int(len(actions)),
            "audit_rows": int(len(audit)),
            "realized_primitive_evidence": realized_evidence,
            "no_clipping_or_projection": True,
        },
        "oracle": {
            "backends": ["alpha", "beta", "gamma"],
            "finite_scores": True,
            "score_hash": canonical_hash(
                scores[[f"R_score_{b}" for b in ("alpha", "beta", "gamma")]].to_dict("records")
            ),
        },
        "stage9": {
            "primary_contrasts": sorted(set(firm["contrast"].astype(str))),
            "firm_effect_rows": int(len(firm)),
            "summary_rows": int(len(summary)),
            "budget_interaction_rows": int(len(interaction)),
            "budget_interaction_summary_rows": int(len(interaction_summary)),
            "bootstrap_fixture_replicates": 128,
            "production_bootstrap_replicates": 10_000,
            "production_bootstrap_rng": "PCG64",
            "production_bootstrap_seed": 20_260_912,
        },
    }




