from __future__ import annotations

import json
from typing import Any, Mapping

from .common import ContractError, canonical_hash
from .retry_policy import (
    RetryClass,
    classify_google_structured_error,
    classify_http_status,
    classify_openai_structured_error,
)


FORBIDDEN_SAMPLING = {"temperature", "top_p", "top_k", "seed", "candidate_count", "n"}
FORBIDDEN_TOOLS = {"tools", "tool_choice", "web_search", "grounding", "code_execution"}


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        keys.update(str(key) for key in value)
        for nested in value.values():
            keys.update(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_walk_keys(nested))
    return keys


def build_batch_record(
    *,
    provider: str,
    exact_model_id: str,
    request_id: str,
    system_text: str,
    user_text: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    if provider == "openai":
        record = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": exact_model_id,
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
                ],
                "reasoning": {"effort": "none"},
                "max_output_tokens": int(max_output_tokens),
                "store": False,
            },
        }
    elif provider == "google":
        record = {
            "key": request_id,
            "request": {
                "contents": [{"role": "user", "parts": [{"text": user_text}]}],
                "systemInstruction": {"parts": [{"text": system_text}]},
                "generationConfig": {
                    "maxOutputTokens": int(max_output_tokens),
                    "thinkingConfig": {"thinkingLevel": "minimal"},
                },
            },
        }
    else:
        raise ContractError(f"Unsupported provider: {provider}")
    validate_batch_record(provider, record, exact_model_id, max_output_tokens)
    return record


def validate_batch_record(
    provider: str,
    record: Mapping[str, Any],
    exact_model_id: str,
    max_output_tokens: int,
) -> None:
    keys = _walk_keys(record)
    bad_sampling = keys.intersection(FORBIDDEN_SAMPLING)
    if bad_sampling:
        raise ContractError(f"Sampling parameter leaked into {provider} payload: {sorted(bad_sampling)}")
    if keys.intersection(FORBIDDEN_TOOLS):
        raise ContractError(f"Tool configuration leaked into {provider} payload")
    raw = json.dumps(record, ensure_ascii=False, sort_keys=True)
    if provider != "google" and exact_model_id not in raw:
        raise ContractError(f"Exact model ID missing from {provider} request")
    if provider == "openai":
        body = record["body"]
        if record.get("url") != "/v1/responses" or body.get("reasoning") != {"effort": "none"}:
            raise ContractError("OpenAI request must use Responses with reasoning effort none")
        if int(body.get("max_output_tokens", -1)) != int(max_output_tokens):
            raise ContractError("OpenAI output limit drift")
        if "text" in body or "response_format" in body:
            raise ContractError("Native structured output is forbidden")
    elif provider == "google":
        config = record["request"]["generationConfig"]
        if config.get("thinkingConfig") != {"thinkingLevel": "minimal"}:
            raise ContractError("Gemini thinking level must be minimal")
        if "responseMimeType" in config or "responseJsonSchema" in config:
            raise ContractError("Native JSON output is forbidden")


def provider_request_hash(record: Mapping[str, Any]) -> str:
    return canonical_hash(dict(record))


def _join_segments(segments: list[str]) -> tuple[str, list[str]]:
    return "".join(segments), segments


def classify_http_retry(status_code: int) -> str | None:
    code = int(status_code)
    if code == 200:
        return None
    return classify_http_status(code).value


def normalize_openai(
    line: Mapping[str, Any],
    expected_model: str,
    *,
    cancellation_origin: str | None = None,
) -> dict[str, Any]:
    response_value = line.get("response")
    response = response_value if isinstance(response_value, Mapping) else {}
    body_value = response.get("body")
    body = body_value if isinstance(body_value, Mapping) else {}
    explicit_error = line.get("error")
    segments: list[str] = []
    refusal = False
    for item in body.get("output") or []:
        if not isinstance(item, Mapping) or item.get("type") != "message" or item.get("role") != "assistant":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, Mapping):
                continue
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                segments.append(content["text"])
            elif content.get("type") == "refusal":
                refusal = True
                if isinstance(content.get("refusal"), str):
                    segments.append(content["refusal"])
    text, raw_segments = _join_segments(segments)
    incomplete = str(body.get("status") or "").lower() == "incomplete"
    details = body.get("incomplete_details") or {}
    output_limit = incomplete and str(details.get("reason") or "").lower() in {
        "max_output_tokens", "max_tokens"
    }
    try:
        status_code = int(response.get("status_code"))
    except (TypeError, ValueError):
        status_code = 0
    delivered = status_code == 200 and not explicit_error
    if delivered:
        retry_class = None
    elif status_code:
        retry_class = classify_http_status(status_code).value
    else:
        retry_class = classify_openai_structured_error(explicit_error, cancellation_origin=cancellation_origin).value
    return {
        "request_id": line.get("custom_id"),
        "provider": "openai",
        "expected_model": expected_model,
        "returned_model": body.get("model"),
        "provider_request_id": body.get("id") or line.get("id"),
        "delivered": delivered,
        "retry_class": retry_class,
        "http_status_code": status_code or None,
        "provider_refusal": refusal,
        "output_limit": output_limit,
        "raw_visible_text": text if delivered else None,
        "visible_segments": raw_segments,
        "provider_error": explicit_error or (None if delivered else response_value),
    }


def normalize_google(line: Mapping[str, Any], expected_model: str) -> dict[str, Any]:
    response_value = line.get("response")
    response = response_value if isinstance(response_value, Mapping) else {}
    candidates = response.get("candidates") or []
    error = line.get("error")
    if len(candidates) > 1:
        return {
            "request_id": line.get("key"),
            "provider": "google",
            "expected_model": expected_model,
            "returned_model": response.get("modelVersion"),
            "delivered": False,
            "retry_class": RetryClass.ADAPTER_NONRETRYABLE.value,
            "provider_error": {"code": "ADAPTER_MULTIPLE_CANDIDATES", "count": len(candidates)},
            "raw_visible_text": None,
            "visible_segments": [],
            "provider_refusal": False,
            "output_limit": False,
            "provider_request_id": response.get("responseId"),
        }
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    segments = [
        part["text"]
        for part in parts
        if isinstance(part, Mapping)
        and part.get("thought") is not True
        and isinstance(part.get("text"), str)
    ]
    text, raw_segments = _join_segments(segments)
    finish = str(candidates[0].get("finishReason") or "") if candidates else ""
    delivered = isinstance(response_value, Mapping) and not error
    return {
        "request_id": line.get("key"),
        "provider": "google",
        "expected_model": expected_model,
        "returned_model": response.get("modelVersion"),
        "provider_request_id": response.get("responseId"),
        "delivered": delivered,
        "retry_class": None if delivered else classify_google_structured_error(error).value,
        "provider_refusal": finish.upper() in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"},
        "output_limit": finish.upper() in {"MAX_TOKENS", "MAX_OUTPUT_TOKENS"},
        "raw_visible_text": text if delivered else None,
        "visible_segments": raw_segments,
        "provider_error": error,
    }
