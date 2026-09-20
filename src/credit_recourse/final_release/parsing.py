from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .action_validation import apply_itt_rule
from .contract import CANDIDATES, DIMENSIONS, DesignBundle, candidate_vector


class DuplicateKey(ValueError):
    pass


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def strict_json_object(text: str) -> dict[str, Any]:
    value = json.loads(
        text,
        object_pairs_hook=_pairs_no_duplicates,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("Top-level JSON must be one object")
    return value


@dataclass(frozen=True)
class ParseResult:
    completion_kind: str
    parser_state: str
    action_valid: bool
    policy_usable: bool
    response_schema_valid: bool
    full_response_valid: bool
    fallback_applied: bool
    candidate_id_req: str | None
    a_req: dict[str, float] | None
    a_accepted: dict[str, float] | None
    a_itt: dict[str, float] | None
    action_application_status: str
    action_application_reason: str
    evidence_fields: list[str] | None
    diagnosis: str | None
    rationale: str | None
    error_codes: list[str]
    intensity: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _zero(design: DesignBundle) -> dict[str, float]:
    return candidate_vector(design, "A0")


def _free8_action(
    design: DesignBundle,
    obj: Mapping[str, Any],
    budget: str,
) -> tuple[dict[str, float] | None, list[str], float | None]:
    """Parse the exact proposed vector while recording semantic violations.

    A finite, correctly keyed proposal is retained in a_req even when it
    violates bounds, mutex, or the B1 cap. It is never clipped, projected, or
    rescaled; strict acceptance and ITT assignment use the canonical rule.
    """
    errors: list[str] = []
    if set(obj) != set(DIMENSIONS):
        return None, ["ACTION_KEY_SET"], None
    action: dict[str, float] = {}
    for name in DIMENSIONS:
        value = obj[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"ACTION_NONNUMERIC:{name}")
            continue
        number = float(value)
        if not math.isfinite(number):
            errors.append(f"ACTION_NONFINITE:{name}")
            continue
        action[name] = number
    if errors:
        return None, errors, None

    bounds = design.prompts["action_contract"]["bounds"]
    for name in DIMENSIONS:
        lower, upper = map(float, bounds[name])
        if action[name] < lower or action[name] > upper:
            errors.append(f"ACTION_BOUNDS:{name}")
    if (
        action["deleveraging_total_debt_pct"] > 0
        and action["refinancing_short_debt_pct"] > 0
    ):
        errors.append("ACTION_DL_RF_MUTEX")
    scales = design.prompts["action_contract"]["scales"]
    intensity = math.fsum(
        abs(action[name]) / float(scales[name]) for name in DIMENSIONS
    )
    cap = design.prompts["budget"]["cap_by_budget"].get(budget)
    if cap is not None and intensity > float(cap) + 1e-9:
        errors.append("ACTION_INTENSITY_CAP")
    return action, errors, intensity


def _candidate_action(
    design: DesignBundle, obj: Mapping[str, Any]
) -> tuple[str | None, dict[str, float] | None, list[str]]:
    if set(obj) != {"candidate_id"} or not isinstance(obj.get("candidate_id"), str):
        return None, None, ["CANDIDATE_ACTION_SCHEMA"]
    candidate_id = str(obj["candidate_id"])
    if candidate_id not in CANDIDATES:
        return candidate_id, None, ["CANDIDATE_ID"]
    return candidate_id, candidate_vector(design, candidate_id), []


def _terminal_empty_result(
    design: DesignBundle,
    *,
    mode: str,
    budget: str,
    provider_refusal: bool,
    output_limit: bool,
) -> ParseResult:
    completion = (
        "provider_refusal"
        if provider_refusal
        else "output_limit"
        if output_limit
        else "delivered_empty"
    )
    code = (
        "PROVIDER_REFUSAL"
        if provider_refusal
        else "OUTPUT_LIMIT"
        if output_limit
        else "DELIVERED_EMPTY"
    )
    applied = apply_itt_rule(
        None,
        budget,
        terminal_completion=True,
        infrastructure_failure=False,
        mode=mode,
        root=design.root,
    )
    return ParseResult(
        completion_kind=completion,
        parser_state="json_invalid",
        action_valid=False,
        policy_usable=False,
        response_schema_valid=False,
        full_response_valid=False,
        fallback_applied=True,
        candidate_id_req=None,
        a_req=None,
        a_accepted=applied.accepted,
        a_itt=applied.itt,
        action_application_status=applied.status,
        action_application_reason=applied.reason,
        evidence_fields=None,
        diagnosis=None,
        rationale=None,
        error_codes=[code],
        intensity=None,
    )


def parse_policy_response(
    design: DesignBundle,
    *,
    raw_visible_text: str | None,
    mode: str,
    budget: str,
    provider_refusal: bool = False,
    output_limit: bool = False,
    infrastructure_unresolved: bool = False,
) -> ParseResult:
    if infrastructure_unresolved:
        return ParseResult(
            completion_kind="no_confirmed_completion",
            parser_state="not_applicable",
            action_valid=False,
            policy_usable=False,
            response_schema_valid=False,
            full_response_valid=False,
            fallback_applied=False,
            candidate_id_req=None,
            a_req=None,
            a_accepted=None,
            a_itt=None,
            action_application_status="UNRESOLVED_INFRA",
            action_application_reason="retry/resume; no ITT action assigned",
            evidence_fields=None,
            diagnosis=None,
            rationale=None,
            error_codes=["UNRESOLVED_INFRA"],
            intensity=None,
        )

    raw = "" if raw_visible_text is None else raw_visible_text
    if len(raw.encode("utf-8")) == 0:
        return _terminal_empty_result(
            design,
            mode=mode,
            budget=budget,
            provider_refusal=provider_refusal,
            output_limit=output_limit,
        )

    parser_state = "json_valid"
    errors: list[str] = []
    try:
        obj = strict_json_object(raw)
    except DuplicateKey as exc:
        obj = None
        parser_state = "duplicate_key"
        errors.append(f"DUPLICATE_KEY:{exc}")
    except (json.JSONDecodeError, ValueError) as exc:
        obj = None
        parser_state = "json_invalid"
        errors.append(f"JSON_INVALID:{type(exc).__name__}")

    action: dict[str, float] | None = None
    candidate_id: str | None = None
    intensity: float | None = None
    action_errors: list[str] = []
    if obj is not None:
        action_obj = obj.get("action")
        if not isinstance(action_obj, dict):
            errors.append("ACTION_OBJECT")
            parser_state = "action_schema_invalid"
        elif mode == "free8":
            action, action_errors, intensity = _free8_action(
                design, action_obj, budget
            )
            errors.extend(action_errors)
            if action_errors:
                parser_state = (
                    "action_semantic_invalid"
                    if any(
                        code.startswith(
                            (
                                "ACTION_BOUNDS",
                                "ACTION_DL_RF_MUTEX",
                                "ACTION_INTENSITY",
                            )
                        )
                        for code in action_errors
                    )
                    else "action_schema_invalid"
                )
        elif mode == "candidate9":
            candidate_id, action, action_errors = _candidate_action(
                design, action_obj
            )
            errors.extend(action_errors)
            if action_errors:
                parser_state = (
                    "action_semantic_invalid"
                    if "CANDIDATE_ID" in action_errors
                    else "action_schema_invalid"
                )
            elif action is not None:
                scales = design.prompts["action_contract"]["scales"]
                intensity = math.fsum(
                    abs(action[name]) / float(scales[name]) for name in DIMENSIONS
                )
        else:
            errors.append(f"MODE:{mode}")
            action_errors.append(f"MODE:{mode}")
            parser_state = "action_schema_invalid"

    action_valid = action is not None and not action_errors
    evidence: list[str] | None = None
    diagnosis: str | None = None
    rationale: str | None = None
    schema_valid = False
    if obj is not None:
        diagnosis = (
            obj.get("diagnosis") if isinstance(obj.get("diagnosis"), str) else None
        )
        rationale = (
            obj.get("rationale") if isinstance(obj.get("rationale"), str) else None
        )
        raw_evidence = obj.get("evidence_fields")
        if isinstance(raw_evidence, list) and all(
            isinstance(item, str) for item in raw_evidence
        ):
            evidence = list(raw_evidence)
        allow = set(
            design.prompts["response_contract"]["evidence_field_allowlist"]
        )
        evidence_valid = (
            evidence is not None
            and len(evidence) <= 4
            and len(evidence) == len(set(evidence))
            and set(evidence).issubset(allow)
        )
        schema_valid = (
            set(obj) == {"diagnosis", "action", "evidence_fields", "rationale"}
            and diagnosis is not None
            and rationale is not None
            and evidence_valid
            and action_valid
        )
        if not schema_valid:
            errors.append("FULL_RESPONSE_SCHEMA")

    terminal_unusable = provider_refusal or output_limit
    if provider_refusal:
        errors.append("PROVIDER_REFUSAL")
    if output_limit:
        errors.append("OUTPUT_LIMIT")
    policy_usable = action_valid and not terminal_unusable

    applied = apply_itt_rule(
        action,
        budget,
        terminal_completion=True,
        infrastructure_failure=False,
        mode=mode,
        candidate_id=candidate_id,
        root=design.root,
    )
    if policy_usable and applied.status != "ACCEPTED":
        raise RuntimeError("Parser/action-validation contract disagreement")
    if not policy_usable and applied.status == "ACCEPTED":
        applied = apply_itt_rule(
            None,
            budget,
            terminal_completion=True,
            infrastructure_failure=False,
            mode=mode,
            root=design.root,
        )

    completion = (
        "provider_refusal"
        if provider_refusal
        else "output_limit"
        if output_limit
        else "usable_output"
        if policy_usable
        else "unusable_output"
    )
    return ParseResult(
        completion_kind=completion,
        parser_state=parser_state,
        action_valid=action_valid,
        policy_usable=policy_usable,
        response_schema_valid=schema_valid,
        full_response_valid=policy_usable and schema_valid,
        fallback_applied=applied.status == "CONFIRMED_UNUSABLE_COMPLETION",
        candidate_id_req=candidate_id,
        a_req=action,
        a_accepted=applied.accepted,
        a_itt=applied.itt,
        action_application_status=applied.status,
        action_application_reason=applied.reason,
        evidence_fields=evidence,
        diagnosis=diagnosis,
        rationale=rationale,
        error_codes=errors,
        intensity=intensity,
    )


def parse_probe_response(raw_visible_text: str | None) -> dict[str, Any]:
    raw = "" if raw_visible_text is None else raw_visible_text
    try:
        obj = strict_json_object(raw)
    except (DuplicateKey, json.JSONDecodeError, ValueError) as exc:
        return {
            "probe_schema_valid": False,
            "parser_state": type(exc).__name__,
            "parsed": None,
        }
    allowed_familiarity = {
        "none",
        "name_only",
        "general_business",
        "specific_financial_knowledge",
    }
    recognized = obj.get("recognized")
    recognized_valid = recognized is None or isinstance(recognized, bool)
    recalled = obj.get("recalled_debt_to_assets_2024")
    recalled_valid = recalled is None or (
        not isinstance(recalled, bool)
        and isinstance(recalled, (int, float))
        and math.isfinite(float(recalled))
    )
    valid = (
        set(obj)
        == {
            "recognized",
            "familiarity",
            "recalled_debt_to_assets_2024",
            "recall_basis",
        }
        and recognized_valid
        and isinstance(obj.get("familiarity"), str)
        and obj.get("familiarity") in allowed_familiarity
        and recalled_valid
        and isinstance(obj.get("recall_basis"), str)
    )
    return {
        "probe_schema_valid": valid,
        "parser_state": "json_valid",
        "parsed": obj,
    }
