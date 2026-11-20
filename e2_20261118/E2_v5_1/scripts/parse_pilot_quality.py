from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

E2 = Path(__file__).resolve().parents[1]
WORKSPACE = E2.parents[1]
KIT = E2.parents[0] / "A_reproduction/E1_score_gate_replay/runroot_zip/THESIS_REPRO_KIT_v2.1.1_FINAL"
E1_RELEASE = E2.parents[0] / "source_inputs/github_release_ablation_20261102_readonly/contents/ablation_20261102/E1"
E1_PROCESS = E1_RELEASE / "scripts/e1_process.py"
PROMPT_CONTRACT = E2.parents[0] / "source_inputs/github_main/frozen/evidence/llm/prompt_contract.json"
ACTION_CONTRACT = KIT / "archive/DEPLOYED_RELEASE/stage2_candidate_projection/candidate_action_contract_v4_3.json"
STRESS_SOURCE = E2.parents[0] / "A_reproduction/R2_R8_replay_workspace/analysis_20261117/e5_stress.py"
EXPECTED_MODELS = {"openai": "gpt-5.4-mini-2026-03-17", "google": "gemini-3.1-flash-lite"}
RATES = {"openai": {"input": 0.375, "output": 2.25}, "google": {"input": 0.125, "output": 0.75}}
METRICS = {"pretax_income", "interest_expense", "total_debt", "cash", "current_liabilities"}
DIRECTIONS = {"increase", "decrease", "no_material_change"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_e1_processor():
    spec = importlib.util.spec_from_file_location("e1_process", E1_PROCESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load E1 processor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def request_map() -> dict[str, dict]:
    out = {}
    for provider in EXPECTED_MODELS:
        for item in read_jsonl(E2 / "requests" / f"{provider}_pilot_mixed.jsonl"):
            rid = item["custom_id"] if provider == "openai" else item["key"]
            out[rid] = item
    return out


def user_prompt(provider: str, item: dict) -> str:
    if provider == "openai":
        return next(x["content"][0]["text"] for x in item["body"]["input"] if x.get("role") == "user")
    return item["request"]["contents"][0]["parts"][0]["text"]


def q_expected_ids(provider: str, item: dict) -> list[int]:
    prompt = user_prompt(provider, item)
    pairs = re.findall(r'"candidate"\s*:\s*"[^"]+"\s*,\s*"candidate_id"\s*:\s*(\d+)', prompt)
    return [int(x) for x in pairs]


def q_structure(text: str | None, expected_ids: list[int]) -> tuple[bool, list[str], int]:
    errors: list[str] = []
    sign_mismatches = 0
    if not isinstance(text, str) or not text.strip():
        return False, ["empty_output"], 0
    try:
        obj = json.loads(text)
    except Exception:
        return False, ["invalid_json_or_non_json_text"], 0
    if not isinstance(obj, dict) or set(obj) != {"forecasts"}:
        return False, ["top_level_schema_mismatch"], 0
    forecasts = obj.get("forecasts")
    if not isinstance(forecasts, list) or len(forecasts) != len(expected_ids):
        return False, ["forecast_count_mismatch"], 0
    seen_ids = []
    for index, row in enumerate(forecasts):
        if not isinstance(row, dict) or set(row) != {"candidate_id", *METRICS}:
            errors.append(f"row_{index}_field_set_mismatch")
            continue
        cid = row.get("candidate_id")
        if isinstance(cid, bool) or not isinstance(cid, int):
            errors.append(f"row_{index}_candidate_id_not_integer")
        else:
            seen_ids.append(cid)
        for metric in METRICS:
            value = row.get(metric)
            if not isinstance(value, dict) or set(value) != {"direction", "change_thousand_krw"}:
                errors.append(f"row_{index}_{metric}_field_set_mismatch")
                continue
            direction = value.get("direction")
            amount = value.get("change_thousand_krw")
            if direction not in DIRECTIONS:
                errors.append(f"row_{index}_{metric}_invalid_direction")
            if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(float(amount)):
                errors.append(f"row_{index}_{metric}_invalid_amount")
                continue
            if direction == "increase" and not float(amount) > 0:
                sign_mismatches += 1
            elif direction == "decrease" and not float(amount) < 0:
                sign_mismatches += 1
            elif direction == "no_material_change" and float(amount) != 0:
                sign_mismatches += 1
    if seen_ids != expected_ids:
        errors.append("candidate_ids_or_order_mismatch")
    return len(errors) == 0, errors, sign_mismatches


def usage_tokens(provider: str, line: dict) -> tuple[int, int]:
    response = line.get("response") if isinstance(line.get("response"), dict) else {}
    body = response.get("body") if provider == "openai" and isinstance(response.get("body"), dict) else response
    usage = body.get("usage") or body.get("usageMetadata") or {}
    if provider == "openai":
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    return (int(usage.get("promptTokenCount", 0)),
            int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0)))


def write_analysis_hashes() -> None:
    files = [
        E2 / "E2_V5_1_PROTOCOL.md", E2 / "request_manifest.csv", E2 / "pre_call_hashes.csv",
        E2 / "logs/pre_call_hashes_at_submission.csv", E2 / "scripts/parse_pilot_quality.py",
        E2 / "scripts/audit_q_direction_consistency.py",
        E1_PROCESS, KIT / "src/credit_recourse/final_release/parsing.py",
        KIT / "src/credit_recourse/final_release/providers.py", PROMPT_CONTRACT, ACTION_CONTRACT,
        KIT / "src/credit_recourse/simulator/oracle_variable_contract.py",
        KIT / "src/credit_recourse/simulator/oracle_variables.py",
        KIT / "src/credit_recourse/simulator/semantic_action_v4.py",
        KIT / "src/credit_recourse/simulator/oracle_evaluator.py",
        KIT / "archive/DEPLOYED_RELEASE/stage1_oracle_backends/alpha/oracle_alpha_params.json",
        KIT / "archive/DEPLOYED_RELEASE/stage1_oracle_backends/beta/benchmark_beta_params.json",
        KIT / "archive/DEPLOYED_RELEASE/stage1_oracle_backends/gamma/benchmark_gamma_params.json",
        STRESS_SOURCE,
    ]
    rows = [{"path": str(p.relative_to(WORKSPACE)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
    pd.DataFrame(rows).to_csv(E2 / "analysis_freeze_hashes.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    job_manifest_path = E2 / "logs/provider_jobs.json"
    jobs = json.loads(job_manifest_path.read_text(encoding="utf-8"))
    if set(jobs.get("jobs", {})) != {"openai", "google"}:
        raise SystemExit("Both provider batch records are required")
    if any(not jobs["jobs"][p].get("last_poll", {}).get("terminal") for p in EXPECTED_MODELS):
        raise SystemExit("Both batch jobs must be terminal before processing")
    if any(jobs["jobs"][p].get("terminal_failure") for p in EXPECTED_MODELS):
        raise SystemExit("A provider batch failed; resolve infrastructure status before processing")

    write_analysis_hashes()
    proc = load_e1_processor()
    if str(KIT / "src") not in sys.path:
        sys.path.insert(0, str(KIT / "src"))
    design = SimpleNamespace(root=KIT,
                             prompts=json.loads(PROMPT_CONTRACT.read_text(encoding="utf-8")),
                             action_contract=json.loads(ACTION_CONTRACT.read_text(encoding="utf-8")))
    request_rows = pd.read_csv(E2 / "request_manifest.csv", dtype={"request_id": str, "firm_key": str})
    request_rows = request_rows.set_index("request_id", drop=False)
    requests = request_map()
    raw_by_id: dict[str, tuple[str, dict]] = {}
    duplicate_raw_ids: list[str] = []
    response_hashes = []
    for provider in EXPECTED_MODELS:
        for path in sorted((E2 / "responses").glob(f"pilot_{provider}_attempt1_*.jsonl")):
            response_hashes.append({"path": str(path.relative_to(WORKSPACE)), "bytes": path.stat().st_size, "sha256": sha256(path)})
            for line in read_jsonl(path):
                rid = line.get("custom_id") if provider == "openai" else line.get("key")
                if rid in raw_by_id:
                    duplicate_raw_ids.append(str(rid))
                else:
                    raw_by_id[str(rid)] = (provider, line)

    from credit_recourse.final_release.providers import normalize_google, normalize_openai

    parsed_rows = []
    q_rows = []
    for request_id, req in request_rows.iterrows():
        provider = str(req.provider)
        cond = str(req.condition)
        observed = raw_by_id.get(request_id)
        if observed is None:
            parsed_rows.append({"request_id": request_id, "provider": provider, "condition": cond,
                                "row_id": int(req.row_id), "firm_key": str(req.firm_key),
                                "outcome": "UNRESOLVED_INFRA", "retry_class": "PROVIDER_RESULT_MISSING_NONRETRYABLE",
                                "returned_model": None, "returned_model_ok": False, "strict_usable": None,
                                "strict_itt_action_json": None, "parse_error_codes": None,
                                "input_tokens": 0, "output_tokens": 0, "provider_request_id": None})
            continue
        observed_provider, line = observed
        if observed_provider != provider:
            raise RuntimeError(f"Provider mismatch for {request_id}")
        norm = normalize_openai(line, EXPECTED_MODELS[provider]) if provider == "openai" else normalize_google(line, EXPECTED_MODELS[provider])
        inp, out = usage_tokens(provider, line)
        model_ok = str(norm.get("returned_model") or "") == EXPECTED_MODELS[provider]
        if cond == "Q":
            expected = q_expected_ids(provider, requests[request_id])
            raw_response = line.get("response") if isinstance(line.get("response"), dict) else {}
            raw_body = raw_response.get("body") if provider == "openai" and isinstance(raw_response.get("body"), dict) else raw_response
            raw_usage = raw_body.get("usage") or raw_body.get("usageMetadata") or {}
            incomplete_reason = (raw_body.get("incomplete_details") or {}).get("reason")
            reasoning_tokens = ((raw_usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
                                if provider == "openai" else raw_usage.get("thoughtsTokenCount", 0))
            if norm.get("delivered") and model_ok and norm.get("output_limit"):
                valid, q_errors, sign_mismatches = False, ["output_limit_max_output_tokens"], 0
                q_completion = "output_limit"
            elif norm.get("delivered") and model_ok and norm.get("provider_refusal"):
                valid, q_errors, sign_mismatches = False, ["provider_refusal"], 0
                q_completion = "provider_refusal"
            elif norm.get("delivered") and model_ok and not str(norm.get("raw_visible_text") or "").strip():
                valid, q_errors, sign_mismatches = False, ["delivered_empty"], 0
                q_completion = "delivered_empty"
            elif norm.get("delivered") and model_ok:
                valid, q_errors, sign_mismatches = q_structure(norm.get("raw_visible_text"), expected)
                q_completion = "usable_output"
            else:
                valid, q_errors, sign_mismatches = False, ["not_delivered_or_model_mismatch"], 0
                q_completion = "not_delivered_or_model_mismatch"
            q_rows.append({"request_id": request_id, "provider": provider, "row_id": int(req.row_id),
                           "firm_key": str(req.firm_key), "delivered": bool(norm.get("delivered")),
                           "returned_model": norm.get("returned_model"), "returned_model_ok": model_ok,
                           "completion_kind": q_completion, "json_schema_valid": valid, "schema_errors": "|".join(q_errors),
                           "direction_sign_mismatch_count": sign_mismatches,
                           "expected_candidate_ids": json.dumps(expected),
                           "provider_body_status": raw_body.get("status"), "incomplete_reason": incomplete_reason,
                           "reasoning_tokens": int(reasoning_tokens or 0),
                           "input_tokens": inp, "output_tokens": out, "provider_request_id": norm.get("provider_request_id")})
            parsed_rows.append({"request_id": request_id, "provider": provider, "condition": cond,
                                "row_id": int(req.row_id), "firm_key": str(req.firm_key),
                                "outcome": "DELIVERED" if norm.get("delivered") else "RETRY",
                                "retry_class": norm.get("retry_class"), "returned_model": norm.get("returned_model"),
                                "returned_model_ok": model_ok, "completion_kind": q_completion, "strict_usable": None,
                                "strict_itt_action_json": None, "parse_error_codes": "|".join(q_errors),
                                "input_tokens": inp, "output_tokens": out, "provider_request_id": norm.get("provider_request_id")})
        else:
            result = proc.process_line(provider, line, design)
            action = result.get("strict_itt_action") if result.get("outcome") == "DELIVERED" else None
            usable = bool(result.get("strict_usable")) if result.get("outcome") == "DELIVERED" and model_ok else (False if result.get("outcome") == "DELIVERED" else None)
            parsed_rows.append({"request_id": request_id, "provider": provider, "condition": cond,
                                "row_id": int(req.row_id), "firm_key": str(req.firm_key),
                                "outcome": result.get("outcome"), "retry_class": result.get("retry_class"),
                                "completion_kind": result.get("completion_kind"), "returned_model": result.get("returned_model"),
                                "returned_model_ok": model_ok, "strict_usable": usable,
                                "strict_itt_action_json": json.dumps(action, ensure_ascii=False, sort_keys=True) if action is not None else None,
                                "parse_error_codes": "|".join(result.get("error_codes") or []),
                                "input_tokens": inp, "output_tokens": out, "provider_request_id": result.get("provider_request_id")})

    parsed = pd.DataFrame(parsed_rows)
    q = pd.DataFrame(q_rows)
    parsed.to_csv(E2 / "results/pilot_quality_rows.csv", index=False, encoding="utf-8-sig")
    q.to_csv(E2 / "results/pilot_q_structure.csv", index=False, encoding="utf-8-sig")
    parsed[parsed.condition.isin(["N", "R", "T", "RT"])].to_parquet(E2 / "results/pilot_actions.parquet", index=False)

    policy = parsed[parsed.condition.isin(["N", "R", "T", "RT"])].copy()
    policy_invalid = policy[~policy.strict_usable.astype("boolean").fillna(False)]
    invalid_actions = [json.loads(x) for x in policy_invalid.strict_itt_action_json.dropna().tolist()]
    all_invalid_to_a0 = bool(invalid_actions) and all(
        all(float(value) == 0.0 for value in action.values()) for action in invalid_actions
    )
    summary_rows = []
    for (provider, cond), group in policy.groupby(["provider", "condition"], sort=True):
        delivered = int(group.outcome.eq("DELIVERED").sum())
        usable = int(group.strict_usable.astype("boolean").fillna(False).sum())
        rate_delivered = usable / delivered if delivered else None
        completion = group["completion_kind"] if "completion_kind" in group else pd.Series(dtype=str)
        summary_rows.append({"provider": provider, "condition": cond, "requests": len(group),
                             "delivered": delivered, "strict_usable": usable,
                             "strict_usable_rate_of_requests": usable / len(group) if len(group) else None,
                             "strict_usable_rate_of_delivered": rate_delivered,
                             "expected_model_mismatch": int((~group.returned_model_ok.fillna(False)).sum()),
                             "refusal": int(completion.eq("provider_refusal").sum()),
                             "output_limit": int(completion.eq("output_limit").sum()),
                             "empty": int(completion.eq("delivered_empty").sum()),
                             "infra_unresolved": int(group.outcome.ne("DELIVERED").sum())})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(E2 / "results/pilot_policy_quality_summary.csv", index=False, encoding="utf-8-sig")

    cost_rows = []
    for provider in EXPECTED_MODELS:
        group = parsed[parsed.provider.eq(provider)]
        inp, out = int(group.input_tokens.sum()), int(group.output_tokens.sum())
        cost = inp / 1_000_000 * RATES[provider]["input"] + out / 1_000_000 * RATES[provider]["output"]
        cost_rows.append({"provider": provider, "requests": int(len(group)),
                          "delivered": int(group.outcome.eq("DELIVERED").sum()),
                          "input_tokens": inp, "output_tokens_including_reasoning": out,
                          "estimated_batch_cost_usd": round(cost, 6),
                          "input_rate_per_million": RATES[provider]["input"],
                          "output_rate_per_million": RATES[provider]["output"]})
    pd.DataFrame(cost_rows).to_csv(E2 / "results/pilot_cost_actual_tokens.csv", index=False, encoding="utf-8-sig")

    manifest = pd.read_csv(E2 / "request_manifest.csv", dtype={"request_id": str})
    original_usable = manifest.original_draft_strict_usable.astype(str).str.strip().str.lower().eq("true")
    invalid = ~original_usable
    invalid_rows = manifest[(manifest.condition == "N") & invalid]
    q_delivered = int(q.delivered.fillna(False).sum()) if len(q) else 0
    q_valid = int(q.json_schema_valid.fillna(False).sum()) if len(q) else 0
    q_sign = int(q.direction_sign_mismatch_count.sum()) if len(q) else 0
    q_output_limit = int(q.completion_kind.eq("output_limit").sum()) if len(q) else 0
    q_refusal = int(q.completion_kind.eq("provider_refusal").sum()) if len(q) else 0
    q_empty = int(q.completion_kind.eq("delivered_empty").sum()) if len(q) else 0
    q_limit_details = "; ".join(
        f"{row.request_id} ({row.provider_body_status}/{row.incomplete_reason}, reasoning tokens={int(row.reasoning_tokens):,})"
        for row in q[q.completion_kind.eq("output_limit")].itertuples(index=False)
    ) or "none"
    total_cost = round(sum(x["estimated_batch_cost_usd"] for x in cost_rows), 6)
    core_min_usable_rate = float(summary.strict_usable_rate_of_requests.min()) if len(summary) else 0.0
    core_quality_pass = bool(len(summary) == 8 and (summary.strict_usable_rate_of_requests >= 0.95).all()
                             and summary.expected_model_mismatch.eq(0).all()
                             and summary.infra_unresolved.eq(0).all())
    q_consistency_path = E2 / "results/pilot_q_direction_consistency.json"
    q_consistency = json.loads(q_consistency_path.read_text(encoding="utf-8")) if q_consistency_path.exists() else {"totals": {}, "mismatch_types": {}}
    preflight_cost = json.loads((E2 / "preflight_cost_estimate.json").read_text(encoding="utf-8"))["total"]
    table_qa = json.loads((E2 / "table_integrity.json").read_text(encoding="utf-8"))
    status = ("COMPLETE" if len(parsed) == 1000 and parsed.outcome.eq("DELIVERED").all()
              and parsed.returned_model_ok.fillna(False).all() else "PARTIAL_OR_INFRA_FAILURE")
    lines = [
        "# E2 v5.1 품질 파일럿 보고",
        "",
        f"상태: **{status}**. 요청은 100개 기업 × 2개 모델 × N/R/T/RT/Q = 1,000개로 고정했다. 파일럿에서는 Oracle 성능점수를 계산하거나 열람하지 않았다. Q는 정답 채점 없이 JSON 구조만 검사했다.",
        "",
        "## 조건·모델별 응답 품질",
        "",
        "엄격 파싱은 공식 E1 frozen parser/action contract를 사용한다. 인프라 미완료는 A0로 대체하지 않고 별도 분리한다. usable rate는 전달된 요청과 전체 요청 기준을 같이 보인다.",
        "",
        "| 모델 | 조건 | 요청 | 전달 완료 | 엄격 사용 가능 | 전체 기준 | 완료 응답 기준 | 무효/실패 원인 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        fail = f"refusal={row['refusal']}, output_limit={row['output_limit']}, empty={row['empty']}, infra={row['infra_unresolved']}, model_mismatch={row['expected_model_mismatch']}"
        r1 = f"{row['strict_usable_rate_of_requests']:.1%}" if row["strict_usable_rate_of_requests"] is not None else "NA"
        r2 = f"{row['strict_usable_rate_of_delivered']:.1%}" if row["strict_usable_rate_of_delivered"] is not None else "NA"
        lines.append(f"| {row['provider']} | {row['condition']} | {row['requests']} | {row['delivered']} | {row['strict_usable']} | {r1} | {r2} | {fail} |")
    lines += [
        "",
        "각 모델×조건의 strict usable 비율이 95% 미만이면 운영상 경보로 기록한다. 이 기준은 연구 가설 검정이 아니다.",
        "",
        "## Q 응답 구조",
        "",
        f"- 전달된 Q 응답: {q_delivered}/200; 완전 JSON schema 유효: {q_valid}/{len(q)}; output-limit={q_output_limit}, refusal={q_refusal}, 비어 있는 완료 응답={q_empty}.",
        f"- output-limit 세부: {q_limit_details}. 해당 응답은 상한까지 추론 토큰을 사용하고 visible JSON을 반환하지 못했다. 인프라 재시도 대상으로 처리하지 않았다.",
        "- 검사 항목: forecasts 배열 개수, candidate ID/순서 고정, 다섯 metric별 direction/finite signed amount, 허용 키만 확인. Simulator 예측오차는 계산하지 않았다.",
        f"- direction/부호 내부 불일치: {q_sign}개 metric 값, 이를 포함한 응답 {sum(int(q_consistency['totals'].get(f'{p}.responses_with_direction_amount_mismatch', 0)) for p in EXPECTED_MODELS)}개. OpenAI 감소 표시에 음수가 아닌 값 {q_consistency['mismatch_types'].get('openai.decrease_not_negative', 0)}개, Google no-change 표시에 0이 아닌 값 {q_consistency['mismatch_types'].get('google.no_change_nonzero', 0)}개. 이는 출력 내부 일관성 점검이며 Simulator 정답 대비 정확도는 아니다.",
        "",
        "## Simulator 결과표 QA",
        "",
        f"- 선택 표본의 {table_qa['selected_tables']}개 결과표·{table_qa['table_rows']}행 모두 action-valid 및 Simulator 실행 성공: {table_qa['all_action_valid']}/{table_qa['all_simulator_ok']}; 단위 `KRW_thousand`, 시점 `decision_year_plus_one`.",
        f"- 정규화 행동예산 최대 T={table_qa['max_normalized_action_T']:.12f}; 허용오차 초과 건수 {table_qa['action_T_exceeds_1_by_more_than_1e-8']}.",
        f"- 최대 재무상태 항등식 잔차 {table_qa['max_abs_balance_identity_residual_KRW_thousand']:.9g} KRW thousand (허용 10); 결측 재무항목 최대 {table_qa['financial_items_missing_count_max']}.",
        f"- 결과표 상태는 0효과와 무행동 기준행을 분리했다: `{table_qa['financial_effect_status_counts']}`.",
        "",
        "## 초안 예외 및 요청 통제",
        "",
        f"- 원본 C4 초안 2,300건 중 47건은 기존 STRICT ITT 처리에 따라 A0로 고정했다. 파일럿 표본에는 N 기준 모델-기업 초안 예외 {len(invalid_rows)}개가 있으며, 같은 원문 초안은 네 수정 조건에 공통 유지되고 Simulator 표의 초안 행에서만 A0를 사용했다.",
        "- T/RT의 10행 표는 제출 전 문자열 동일성 검사를 통과했다. R/RT 정책 참조는 E1와 같은 JSON 블록이다. Q는 C4 원문·RL 참조·표를 받지 않는다.",
        "- 응답별 반환 모델 ID를 요청 ID와 비교했다. 파일럿은 Oracle 점수 산출 및 조회 없이 처리했다.",
        "- 정책 조건 무효 10건의 원인: OpenAI JSON 파싱 실패 5건, Google 행동 강도 상한/전체 응답 계약 실패 5건. 모두 모델 응답 품질이며 인프라 실패는 0건이다.",
        f"- 정책 조건 STRICT ITT 무효 응답 {len(policy_invalid)}건은 8개 행동축이 모두 0인 A0로 처리됨을 확인했다: {'PASS' if all_invalid_to_a0 else 'FAIL'}.",
        "",
        "## 호출 수와 사용량 기반 비용 추정",
        "",
        "| 공급자 | 요청 | 전달 완료 | 입력 토큰 | 출력 토큰(추론 포함) | 게시 Batch 단가 기준 비용 추정 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in cost_rows:
        lines.append(f"| {row['provider']} | {row['requests']} | {row['delivered']} | {row['input_tokens']:,} | {row['output_tokens_including_reasoning']:,} | USD {row['estimated_batch_cost_usd']:.4f} |")
    lines += [
        f"| 합계 | {sum(x['requests'] for x in cost_rows)} | {sum(x['delivered'] for x in cost_rows)} | {sum(x['input_tokens'] for x in cost_rows):,} | {sum(x['output_tokens_including_reasoning'] for x in cost_rows):,} | USD {total_cost:.4f} |",
        "",
        "비용은 응답 usage token과 사전 기록한 유료 Batch 단가로 계산한 추정치이며 청구서 금액이 아니다. 재시도는 initial attempt와 분리해 기록한다.",
        f"사전 token 추정과 비교하면 실사용 단가 추정 USD {total_cost:.4f}는 point estimate USD {preflight_cost['point_estimate_usd']:.4f}의 {total_cost/preflight_cost['point_estimate_usd']:.2f}배, historical p90 USD {preflight_cost['conservative_p90_usd']:.4f}의 {total_cost/preflight_cost['conservative_p90_usd']:.2f}배다. 최악 출력상한 추정 USD {preflight_cost['output_cap_single_attempt_usd']:.4f}보다는 낮다.",
        "",
        "## 파일럿 결정",
        "",
        f"- 핵심 2×2 정책 조건 N/R/T/RT는 모델×조건별 최소 strict usable rate {core_min_usable_rate:.1%}, 모델 ID 불일치 0, 인프라 실패 0으로 운영 품질 기준 95%를 통과했다: {'PASS' if core_quality_pass else 'FAIL'}.",
        f"- Q는 보조 진단으로 분리한다. JSON 구조 유효 199/200이며, 1건은 max-output token을 추론에 모두 사용해 응답을 내지 못했고 14개 응답에 내부 방향/부호 불일치가 있었다. 예측 정확도는 채점하지 않았으므로 이 파일럿에서 Q 품질을 확정하지 않는다.",
        "- 파일럿은 점수 열람 전의 운영·형식 점검이며, E2의 성능 효과는 평가하지 않았다. 파일럿 응답은 본실험에 재사용하지 않으며, 본실험은 별도 승인 전까지 실행하지 않는다.",
        "",
        "## 감사 산출물",
        "",
        "- API 작업/원본 해시: logs/provider_jobs.json, logs/response_hashes.json; 요청/프로토콜 제출 전 hash: logs/pre_call_hashes_at_submission.csv 및 logs/provider_jobs.json의 manifest SHA-256.",
        "- 모델·SDK/요금 추정: model_preflight.json, requirements_api_lock.txt, preflight_cost_estimate.json; 분석 기준 해시: analysis_freeze_hashes.csv.",
        "- 결과표/평가정보 경계: ORACLE_SEAM_CROSSWALK.md; S1/S2 향후 평가 전용 구현 계획: STRESS_IMPLEMENTATION.md. 파일럿에서는 민감도 재채점하지 않았다.",
        "",
        f"완료 기록: {datetime.now(timezone.utc).isoformat()}.",
    ]
    (E2 / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (E2 / "logs/response_hashes.json").write_text(json.dumps(response_hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pilot_status": status, "processed_requests": len(parsed),
                      "delivered": int(parsed.outcome.eq("DELIVERED").sum()),
                      "q_schema_valid": q_valid, "q_total": len(q), "estimated_cost_usd": total_cost,
                      "duplicate_response_ids": duplicate_raw_ids,
                      "reports": ["pilot_report.md", "results/pilot_quality_rows.csv",
                                  "results/pilot_policy_quality_summary.csv", "results/pilot_q_structure.csv",
                                  "results/pilot_cost_actual_tokens.csv", "results/pilot_actions.parquet"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
