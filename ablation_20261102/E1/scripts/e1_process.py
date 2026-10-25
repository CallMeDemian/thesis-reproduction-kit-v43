"""E1 response processing: normalize -> parse -> Strict ITT, using the kit's own functions.

Mirrors ``final_release.collection.collect_downloaded`` for one provider output line. It does not
use ``load_design``, which stops on every checkout (LF or CRLF): some released source hashes were
recorded from CRLF bytes and some from LF bytes, several runtime sources changed after the release,
and the release used the frozen action contract. ``python -m thesis_repro rebuild-requests --verify``
rebuilds and checks the frozen requests instead. This module does not touch the ledger. The design object carries only what the parser and ``apply_itt_rule`` read:
``root``, ``prompts`` (frozen prompt contract) and ``action_contract``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

PROMPT_CONTRACT = "frozen/evidence/llm/prompt_contract.json"
ACTION_CONTRACT = "frozen/original_release/rl/stage2/candidate_action_contract_v4_3.json"
MODE = "free8"
BUDGET = "B1"
EXPECTED_MODEL = {
    "openai": "gpt-5.4-mini-2026-03-17",
    "google": "gemini-3.1-flash-lite",
}


def load_processing_context(kit_root: Path) -> SimpleNamespace:
    root = Path(kit_root).resolve()
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    prompts = json.loads((root / PROMPT_CONTRACT).read_text(encoding="utf-8"))
    action = json.loads((root / ACTION_CONTRACT).read_text(encoding="utf-8"))
    return SimpleNamespace(root=root, prompts=prompts, action_contract=action)


def process_line(provider: str, line: Mapping[str, Any], design: SimpleNamespace) -> dict[str, Any]:
    """Return one processed row.

    ``outcome`` is ``DELIVERED`` (parsed, Strict ITT applied) or ``RETRY`` (no confirmed completion:
    never turned into A0). ``returned_model_ok`` is checked for delivered rows only.
    """
    from credit_recourse.final_release.parsing import parse_policy_response
    from credit_recourse.final_release.providers import normalize_google, normalize_openai

    expected = EXPECTED_MODEL[provider]
    item = normalize_openai(line, expected) if provider == "openai" else normalize_google(line, expected)
    out: dict[str, Any] = {
        "request_id": item.get("request_id"),
        "provider": provider,
        "returned_model": item.get("returned_model"),
        "returned_model_ok": None,
        "provider_request_id": item.get("provider_request_id"),
    }
    if not item["delivered"]:
        out.update(outcome="RETRY", retry_class=item.get("retry_class"), completion_kind=None,
                   strict_usable=None, strict_itt_action=None, action_application_status="UNRESOLVED_INFRA",
                   error_codes=[], raw_visible_text=None)
        return out
    out["returned_model_ok"] = str(item.get("returned_model") or "") == expected
    raw_text = item.get("raw_visible_text")
    if item.get("provider_refusal"):
        completion = "provider_refusal"
    elif item.get("output_limit"):
        completion = "output_limit"
    elif not str(raw_text or "").strip():
        completion = "delivered_empty"
    else:
        completion = "usable_output"
    parsed = parse_policy_response(
        design,
        raw_visible_text=raw_text,
        mode=MODE,
        budget=BUDGET,
        provider_refusal=completion == "provider_refusal",
        output_limit=completion == "output_limit",
    )
    out.update(
        outcome="DELIVERED",
        retry_class=None,
        completion_kind=completion,
        parse_completion_kind=parsed.completion_kind,
        strict_usable=bool(parsed.policy_usable),
        strict_itt_action=parsed.a_itt,
        action_application_status=parsed.action_application_status,
        error_codes=list(parsed.error_codes),
        raw_visible_text=raw_text,
    )
    return out
