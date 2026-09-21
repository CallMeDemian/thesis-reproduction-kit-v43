from __future__ import annotations

import importlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .action_validation import DIMENSIONS, candidate_action
from .common import ContractError, find_repo_root, load_json, published_contract_registry
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
    checks, errors, external_blockers = [], [], []
    def check(name: str, fn: Any) -> None:
        try:
            checks.append({"name":name,"status":"PASS","detail":fn()})
        except Exception as exc:
            checks.append({"name":name,"status":"FAIL","detail":str(exc)}); errors.append(f"{name}: {exc}")
    registry = load_json(published_contract_registry(repo))
    check("canonical_rl_reference", lambda: validate_final_reference(repo))
    fresh_configured = bool(os.environ.get("THESIS_REPRO_LLM_CONFIG_ROOT"))
    if fresh_configured:
        # A run-local derived design release is the only place where adapted
        # runtime source hashes are authoritative.  The published evidence
        # bundle intentionally retains historical hashes and is not used as a
        # fresh live-execution design.
        check("final_llm_contract", lambda: validate_llm_contract(repo))
        check("provider_adapters", lambda: _provider_adapter_check(repo))
        check("strict_parser", lambda: _parser_check(repo))
        check("firm_cohort", lambda: {"rows":len(load_firm_cohort(load_design(repo))),"status":"PASS"})
        check("semantic_end_to_end", lambda: run_semantic_fixture(repo))
    else:
        checks.append({"name":"final_llm_contract","status":"INPUT_REQUIRED","detail":"run-local fresh Plan-3 config is required; published evidence is reference-only"})
        checks.append({"name":"provider_adapters","status":"NOT_APPLICABLE","detail":"validated after a run-local fresh design release is installed"})
        checks.append({"name":"strict_parser","status":"NOT_APPLICABLE","detail":"validated after a run-local fresh design release is installed"})
        checks.append({"name":"firm_cohort","status":"NOT_APPLICABLE","detail":"validated after a run-local fresh design release is installed"})
        checks.append({"name":"semantic_end_to_end","status":"NOT_APPLICABLE","detail":"validated after a run-local fresh design release is installed"})
        external_blockers.append("FRESH_RUN_LOCAL_PLAN3_CONFIG_REQUIRED")
    check("simulator_oracle_reachability", _downstream_reachability)
    legacy_path = repo / "repro/manifests/runtime_gates/LEGACY_FREE_ASSERTION.json"
    legacy = load_json(legacy_path) if legacy_path.is_file() else {"status":"PENDING","active_legacy_contracts":-1}
    if legacy.get("status") == "PASS": checks.append({"name":"legacy_free","status":"PASS","detail":legacy})
    else: checks.append({"name":"legacy_free","status":"PENDING","detail":legacy})
    industry = industry_binding_status(repo)
    blockers = list(errors) + list(external_blockers)
    if not industry["ready"]:
        blockers.append(f"industry_binding: {industry['status']}; unresolved={industry['unresolved_count']}")
    if legacy.get("status") != "PASS":
        checks[-1]["detail"] = {**legacy, "role": "historical_audit_not_required_for_fresh_runtime"}
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



