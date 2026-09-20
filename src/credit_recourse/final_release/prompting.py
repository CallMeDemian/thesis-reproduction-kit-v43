from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .common import ContractError, canonical_hash
from .contract import DesignBundle, candidate_vector, state_payload


PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_-]+\}\}")


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _visible_dictionary(design: DesignBundle, info: str, currency_unit: str) -> list[dict[str, str]]:
    try:
        rows = design.prompts["information_conditions"][info]["visible_dictionary"]
    except KeyError as exc:
        raise ContractError(f"Unknown information condition: {info}") from exc
    result = []
    for row in rows:
        result.append(
            {
                "field": str(row["field"]),
                "definition": str(row["definition"]).replace(
                    "{{CANONICAL_CURRENCY_UNIT_ENGLISH}}", currency_unit
                ),
            }
        )
    expected = int(design.prompts["information_conditions"][info]["field_count"])
    if len(result) != expected:
        raise ContractError(f"Visible dictionary count drift for {info}")
    return result


def _replace(template: str, bindings: Mapping[str, str]) -> str:
    rendered = template
    for name, value in bindings.items():
        rendered = rendered.replace("{{" + name + "}}", value)
    unresolved = sorted(set(PLACEHOLDER.findall(rendered)))
    if unresolved:
        raise ContractError(f"Unresolved prompt placeholders: {unresolved}")
    return rendered


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str
    visible_payload_sha256: str


def render_policy_prompt(
    design: DesignBundle,
    *,
    row: Mapping[str, Any],
    condition: str,
    mode: str,
    info: str,
    budget: str,
    currency_unit: str,
    raw_parent_text: str | None = None,
    reference_candidate_id: str | None = None,
) -> RenderedPrompt:
    prompt = design.prompts
    if condition not in prompt["conditions"] or mode not in prompt["action_modes"]:
        raise ContractError(f"Unknown prompt factors: condition={condition}, mode={mode}")
    condition_contract = prompt["conditions"][condition]
    requires_parent = condition_contract["parent"] is not None
    if requires_parent and raw_parent_text is None:
        raise ContractError(f"{condition} requires exact parent raw text")
    if not requires_parent and raw_parent_text is not None:
        raise ContractError(f"{condition} must not receive parent text")
    requires_reference = condition_contract["reference_block"] is not None
    if requires_reference and reference_candidate_id is None:
        raise ContractError(f"{condition} requires a frozen reference")
    if not requires_reference and reference_candidate_id is not None:
        raise ContractError(f"{condition} must not receive a reference")

    dictionary = _visible_dictionary(design, info, currency_unit)
    state = state_payload(row, info)
    scales = dict(prompt["action_contract"]["scales"])
    catalog = list(prompt["action_contract"]["catalog"])
    budget_cap = prompt["budget"]["cap_by_budget"].get(budget)
    budget_block = ""
    if budget_cap is not None:
        budget_block = prompt["budget"]["template"].replace("{{INTENSITY_CAP}}", str(int(budget_cap)))
    draft_block = ""
    if requires_parent:
        draft_block = prompt["draft_block"].replace(
            "{{RAW_DRAFT_AS_JSON_STRING}}",
            json.dumps(raw_parent_text, ensure_ascii=False),
        )
    reference_block = ""
    if requires_reference:
        kind = str(condition_contract["reference_block"])
        reference_block = prompt["reference_blocks"][kind].replace(
            "{{REFERENCE_ACTION_JSON}}",
            pretty_json(candidate_vector(design, str(reference_candidate_id))),
        )
    diagnostic = prompt["diagnostic_scaffold"] if condition_contract["diagnostic_scaffold"] else ""
    bindings = {
        "CONDITION_INSTRUCTION": str(condition_contract["instruction"]),
        "DIAGNOSTIC_SCAFFOLD_OR_EMPTY": diagnostic,
        "FROZEN_FEATURE_DICTIONARY_JSON": pretty_json(dictionary),
        "FIRM_STATE_JSON": pretty_json(state),
        "ACTION_DEFINITIONS": str(prompt["action_definitions"]),
        "REFERENCE_SCALES_JSON": pretty_json(scales),
        "NINE_PROGRAM_CATALOG_JSON": pretty_json(catalog),
        "ACTION_MODE_INSTRUCTION": str(prompt["action_modes"][mode]["instruction"]),
        "BUDGET_BLOCK_OR_EMPTY": budget_block,
        "DRAFT_SECTION_OR_EMPTY": draft_block,
        "REFERENCE_SECTION_OR_EMPTY": reference_block,
        "RESPONSE_SCHEMA_TEXT": str(prompt["action_modes"][mode]["response_schema_text"]),
    }
    user = _replace(str(prompt["user_assembly_template"]), bindings)
    system = str(prompt["system"])
    visible_hash = canonical_hash({"system": system, "user": user})
    return RenderedPrompt(system=system, user=user, visible_payload_sha256=visible_hash)


def render_probe_prompt(design: DesignBundle, *, firm_name: str, statement_basis: str) -> RenderedPrompt:
    if statement_basis not in {"consolidated", "separate-company"}:
        raise ContractError(f"Invalid probe statement basis: {statement_basis}")
    if not str(firm_name).strip():
        raise ContractError("Probe requires a nonempty frozen firm name")
    contract = design.prompts["probe"]
    user = _replace(
        str(contract["user_template"]),
        {
            "FIRM_NAME_AS_JSON_STRING": json.dumps(str(firm_name), ensure_ascii=False),
            "STATEMENT_BASIS_ENGLISH": statement_basis,
        },
    )
    system = str(contract["system"])
    return RenderedPrompt(
        system=system,
        user=user,
        visible_payload_sha256=canonical_hash({"system": system, "user": user}),
    )


