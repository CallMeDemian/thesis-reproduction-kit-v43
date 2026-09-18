from __future__ import annotations

import importlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .action_validation import DIMENSIONS, candidate_action
from .common import ContractError, find_repo_root, load_json, verify_declared_hash
from .contract import load_design, load_firm_cohort
from .llm_contract import industry_binding_status, validate_llm_contract
from .parsing import parse_policy_response
from .providers import build_batch_record
from .rl_reference import validate_final_reference
from .semantic_fixture import run_semantic_fixture


def _provider_adapter_check(root: Path) -> dict[str, Any]:
    design = load_design(root)
    rows = {}
    for key, spec in design.models["models"].items():
        rows[key] = build_batch_record(provider=spec["provider"], exact_model_id=spec["exact_model_id"], request_id="doctor_fixture", system_text="system", user_text="user", max_output_tokens=spec["max_output_tokens"])
    return {"status": "PASS", "models": {k: design.models["models"][k]["exact_model_id"] for k in rows}}


def _parser_check(root: Path) -> dict[str, Any]:
    design = load_design(root)
    free = json.dumps({"diagnosis":"fixture","action":{name:0.0 for name in DIMENSIONS},"evidence_fields":[],"rationale":"fixture"})
    candidate = json.dumps({"diagnosis":"fixture","action":{"candidate_id":"OE"},"evidence_fields":[],"rationale":"fixture"})
    a = parse_policy_response(design, raw_visible_text=free, mode="free8", budget="B1")
    b = parse_policy_response(design, raw_visible_text=candidate, mode="candidate9", budget="B1")
    if not a.policy_usable or not b.policy_usable or b.candidate_id_req != "OE":
        raise ContractError("Final parser fixture failed")
    return {"status":"PASS","free8":a.to_dict(),"candidate9":b.to_dict()}


def _golden_check(root: Path) -> dict[str, Any]:
    import pandas as pd
    golden = load_json(root / "repro/expected/final_c3e_golden.json")
    registry = load_json(root / "configs/current/contract_manifest.json")
    if golden["release_hash"] != registry["rl_release_hash"]:
        raise ContractError("Golden/current release hash mismatch")
    actions = pd.read_parquet(root / registry["rl_reference_actions"]).set_index("row_id")
    for row in golden["sample_rows"]:
        actual = actions.loc[int(row["row_id"])]
        if str(actual["firm_id"]) != str(row["firm_id"]) or str(actual["action_id"]) != str(row["action_id"]):
            raise ContractError("Golden sample action drift")
        for name in ("Alpha","Beta","Gamma"):
            if abs(float(actual[name])-float(row[name])) > 1e-12:
                raise ContractError(f"Golden sample payoff drift: {row['row_id']} {name}")
    return {"status":"PASS","sample_count":len(golden["sample_rows"]),"release_hash":golden["release_hash"]}


def _downstream_reachability() -> dict[str, Any]:
    names = [
        "credit_recourse.simulator.v43_production_bundle",
        "credit_recourse.eval.final_stage8_llm_multi_oracle_eval.pipeline",
        "credit_recourse.eval.final_stage9_llm_rl_comparison.pipeline",
    ]
    for name in names:
        importlib.import_module(name)
    return {"status":"PASS","modules":names}


def run_doctor(root: Path | None = None) -> dict[str, Any]:
    repo = (root or find_repo_root()).resolve()
    checks, errors = [], []
    def check(name: str, fn: Any) -> None:
        try:
            checks.append({"name":name,"status":"PASS","detail":fn()})
        except Exception as exc:
            checks.append({"name":name,"status":"FAIL","detail":str(exc)}); errors.append(f"{name}: {exc}")
    registry = load_json(repo / "configs/current/contract_manifest.json")
    for key in ("action_contract","oracle_registry","simulator_contract","rl_reference_definition","rl_reference_probabilities","rl_reference_actions","llm_design","llm_matrix","llm_design_release","c6ex_permutation","c6ex_materialized","c6ex_manifest","analysis_contract","parser_implementation"):
        expected = registry.get(key+"_sha256")
        if expected:
            check(key+"_hash", lambda k=key,e=expected: verify_declared_hash(repo, registry[k], e))
    check("canonical_rl_reference", lambda: validate_final_reference(repo))
    check("final_llm_contract", lambda: validate_llm_contract(repo))
    check("provider_adapters", lambda: _provider_adapter_check(repo))
    check("strict_parser", lambda: _parser_check(repo))
    check("golden_regression", lambda: _golden_check(repo))
    check("firm_cohort", lambda: {"rows":len(load_firm_cohort(load_design(repo))),"status":"PASS"})
    check("simulator_oracle_reachability", _downstream_reachability)
    check("semantic_end_to_end", lambda: run_semantic_fixture(repo))
    legacy_path = repo / "repro/manifests/runtime_gates/LEGACY_FREE_ASSERTION.json"
    legacy = load_json(legacy_path) if legacy_path.is_file() else {"status":"PENDING","active_legacy_contracts":-1}
    if legacy.get("status") == "PASS": checks.append({"name":"legacy_free","status":"PASS","detail":legacy})
    else: checks.append({"name":"legacy_free","status":"PENDING","detail":legacy})
    industry = industry_binding_status(repo)
    blockers = list(errors)
    if not industry["ready"]: blockers.append(f"industry_binding: {industry['status']}; unresolved={industry['unresolved_count']}")
    if legacy.get("status") != "PASS": blockers.append("legacy removal/post-delete regression not yet sealed")
    live_ready = not blockers
    return {
        "schema_version":"v43_final_release_doctor_v3_semantic_gate", "generated_utc":datetime.now(timezone.utc).isoformat(),
        "repository":str(repo), "python":platform.python_version(),
        "status":"PASS_LIVE_API_READY" if live_ready else ("CONTRACT_FAIL" if errors else "CONTRACT_PASS_LIVE_API_BLOCKED"),
        "contract_integrity_pass":not errors, "live_api_ready":live_ready, "live_api_blockers":blockers,
        "industry_binding":industry, "checks":checks, "scientific_contract":registry["scientific_contract_version"],
        "rl_reference_id":registry["rl_reference_id"], "rl_release_hash":registry["rl_release_hash"],
        "llm_design_release_hash":registry["llm_design_release_hash"], "expected_llm_calls":24150, "api_calls_executed":0,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--json-out",type=Path); p.add_argument("--require-live-ready",action="store_true"); args=p.parse_args(argv)
    try: result=run_doctor()
    except Exception as exc: result={"status":"CONTRACT_FAIL","contract_integrity_pass":False,"live_api_ready":False,"error":str(exc)}
    text=json.dumps(result,ensure_ascii=False,indent=2,default=str); print(text)
    if args.json_out: args.json_out.parent.mkdir(parents=True,exist_ok=True); args.json_out.write_text(text+"\n",encoding="utf-8")
    return 1 if not result.get("contract_integrity_pass") else 2 if args.require_live_ready and not result.get("live_api_ready") else 0


if __name__ == "__main__": raise SystemExit(main())



