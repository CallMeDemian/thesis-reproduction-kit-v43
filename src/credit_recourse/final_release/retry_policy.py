from __future__ import annotations

from enum import Enum
import importlib
from typing import Any, Mapping


class RetryClass(str, Enum):
    TRANSPORT_NO_COMPLETION = "TRANSPORT_NO_COMPLETION"
    HTTP_429 = "HTTP_429"
    HTTP_5XX = "HTTP_5XX"
    PROVIDER_JOB_FAILED = "PROVIDER_JOB_FAILED"
    PROVIDER_JOB_EXPIRED = "PROVIDER_JOB_EXPIRED"
    PROVIDER_REQUEST_TIMEOUT = "PROVIDER_REQUEST_TIMEOUT"
    PROVIDER_JOB_CANCELLED_UNINTENTIONAL = "PROVIDER_JOB_CANCELLED_UNINTENTIONAL"

    HTTP_4XX_NONRETRYABLE = "HTTP_4XX_NONRETRYABLE"
    HTTP_OTHER_NONRETRYABLE = "HTTP_OTHER_NONRETRYABLE"
    PROVIDER_ERROR_UNKNOWN_NONRETRYABLE = "PROVIDER_ERROR_UNKNOWN_NONRETRYABLE"
    PROVIDER_ERROR_CANCELLED_NONRETRYABLE = "PROVIDER_ERROR_CANCELLED_NONRETRYABLE"
    PROVIDER_ERROR_DATA_LOSS_NONRETRYABLE = "PROVIDER_ERROR_DATA_LOSS_NONRETRYABLE"
    PROVIDER_RESULT_MISSING_NONRETRYABLE = "PROVIDER_RESULT_MISSING_NONRETRYABLE"
    PROVIDER_RESPONSE_NONRETRYABLE = "PROVIDER_RESPONSE_NONRETRYABLE"
    ADAPTER_NONRETRYABLE = "ADAPTER_NONRETRYABLE"
    NONRETRYABLE_CANCELLED = "NONRETRYABLE_CANCELLED"
    PROVIDER_JOB_CANCELLED_UNKNOWN_NONRETRYABLE = "PROVIDER_JOB_CANCELLED_UNKNOWN_NONRETRYABLE"


class CancellationOrigin(str, Enum):
    USER_INTENTIONAL = "USER_INTENTIONAL"
    PROVIDER_UNINTENDED = "PROVIDER_UNINTENDED"
    UNKNOWN = "UNKNOWN"


RETRYABLE_TECHNICAL_CLASSES = frozenset(
    {
        RetryClass.TRANSPORT_NO_COMPLETION.value,
        RetryClass.HTTP_429.value,
        RetryClass.HTTP_5XX.value,
        RetryClass.PROVIDER_JOB_FAILED.value,
        RetryClass.PROVIDER_JOB_EXPIRED.value,
        RetryClass.PROVIDER_REQUEST_TIMEOUT.value,
        RetryClass.PROVIDER_JOB_CANCELLED_UNINTENTIONAL.value,
    }
)
KNOWN_RETRY_CLASSES = frozenset(item.value for item in RetryClass)


def is_retryable_class(value: str | RetryClass | None) -> bool:
    raw = value.value if isinstance(value, RetryClass) else value
    return raw in RETRYABLE_TECHNICAL_CLASSES


def is_known_retry_class(value: str | RetryClass | None) -> bool:
    raw = value.value if isinstance(value, RetryClass) else value
    return raw in KNOWN_RETRY_CLASSES


def classify_http_status(status_code: int) -> RetryClass:
    code = int(status_code)
    if code == 429:
        return RetryClass.HTTP_429
    if 500 <= code <= 599:
        return RetryClass.HTTP_5XX
    if 400 <= code <= 499:
        return RetryClass.HTTP_4XX_NONRETRYABLE
    return RetryClass.HTTP_OTHER_NONRETRYABLE


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def classify_openai_structured_error(
    error: Any,
    *,
    cancellation_origin: str | CancellationOrigin | None = None,
) -> RetryClass:
    if not isinstance(error, Mapping) or not error:
        return RetryClass.PROVIDER_ERROR_UNKNOWN_NONRETRYABLE
    for field in ("status_code", "http_status_code", "status"):
        code = _integer(error.get(field))
        if code is not None and 100 <= code <= 599:
            return classify_http_status(code)
    raw_code = error.get("code")
    code = _integer(raw_code)
    if code is not None and 100 <= code <= 599:
        return classify_http_status(code)
    name = str(raw_code or error.get("type") or "").strip().upper()
    if name == "BATCH_EXPIRED":
        return RetryClass.PROVIDER_JOB_EXPIRED
    if name == "REQUEST_TIMEOUT":
        return RetryClass.PROVIDER_REQUEST_TIMEOUT
    if name == "BATCH_CANCELLED":
        return classify_provider_job_terminal("openai", "cancelled", cancellation_origin)
    if name in {"RATE_LIMIT_EXCEEDED", "RATE_LIMIT_ERROR", "TOO_MANY_REQUESTS"}:
        return RetryClass.HTTP_429
    if name in {"SERVER_ERROR", "INTERNAL_ERROR", "INTERNAL_SERVER_ERROR", "BAD_GATEWAY", "SERVICE_UNAVAILABLE", "GATEWAY_TIMEOUT"}:
        return RetryClass.HTTP_5XX
    if name in {"INVALID_REQUEST_ERROR", "INVALID_ARGUMENT", "AUTHENTICATION_ERROR", "PERMISSION_DENIED", "UNAUTHENTICATED", "NOT_FOUND", "CONFLICT"}:
        return RetryClass.HTTP_4XX_NONRETRYABLE
    return RetryClass.PROVIDER_ERROR_UNKNOWN_NONRETRYABLE

def classify_google_structured_error(error: Any) -> RetryClass:
    if not isinstance(error, Mapping) or not error:
        return RetryClass.PROVIDER_ERROR_UNKNOWN_NONRETRYABLE
    status = str(error.get("status") or "").strip().upper()
    code = _integer(error.get("code"))
    if status == "RESOURCE_EXHAUSTED" or code == 8:
        return RetryClass.HTTP_429
    if status in {"INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED", "ABORTED"} or code in {4, 10, 13, 14}:
        return RetryClass.HTTP_5XX
    if status in {
        "INVALID_ARGUMENT", "UNAUTHENTICATED", "PERMISSION_DENIED", "NOT_FOUND",
        "FAILED_PRECONDITION", "ALREADY_EXISTS", "OUT_OF_RANGE",
    } or code in {3, 5, 6, 7, 9, 11, 16}:
        return RetryClass.HTTP_4XX_NONRETRYABLE
    if status == "CANCELLED" or code == 1:
        return RetryClass.PROVIDER_ERROR_CANCELLED_NONRETRYABLE
    if status == "DATA_LOSS" or code == 15:
        return RetryClass.PROVIDER_ERROR_DATA_LOSS_NONRETRYABLE
    if code is not None and 400 <= code <= 599:
        return classify_http_status(code)
    return RetryClass.PROVIDER_ERROR_UNKNOWN_NONRETRYABLE


def is_control_plane_transport_exception(error: BaseException) -> bool:
    # Poll/download failures re-poll the same remote job. Optional imports only
    # inspect SDK exception types and never contact a provider.
    types: list[type[BaseException]] = [TimeoutError, ConnectionError]
    for module_name, names in (
        ("openai", ("APIConnectionError", "APITimeoutError")),
        ("httpx", ("TimeoutException", "NetworkError")),
    ):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            continue
        for name in names:
            candidate = getattr(module, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                types.append(candidate)
    return isinstance(error, tuple(types))


def classify_transport_exception(error: BaseException) -> RetryClass | None:
    # Retained for explicit logical transport evidence. Poll/download callers
    # use is_control_plane_transport_exception and never create attempt2.
    if is_control_plane_transport_exception(error):
        return RetryClass.TRANSPORT_NO_COMPLETION
    return None


def normalized_cancellation_origin(value: str | CancellationOrigin | None) -> CancellationOrigin:
    if isinstance(value, CancellationOrigin):
        return value
    raw = str(value or "").strip().upper()
    try:
        return CancellationOrigin(raw)
    except ValueError:
        return CancellationOrigin.UNKNOWN


def normalize_provider_job_state(provider: str, state: str) -> str:
    raw = str(state or "").strip().upper()
    if str(provider).strip().upper() == "GOOGLE":
        raw = raw.rsplit(".", 1)[-1]
        if raw.startswith("JOB_STATE_"):
            raw = raw[len("JOB_STATE_"):]
    return "CANCELLED" if raw == "CANCELED" else raw


def classify_provider_job_terminal(
    provider: str,
    state: str,
    cancellation_origin: str | CancellationOrigin | None = None,
) -> RetryClass:
    normalized = normalize_provider_job_state(provider, state)
    if normalized == "FAILED":
        return RetryClass.PROVIDER_JOB_FAILED
    if normalized == "EXPIRED":
        return RetryClass.PROVIDER_JOB_EXPIRED
    if normalized == "CANCELLED":
        origin = normalized_cancellation_origin(cancellation_origin)
        if origin is CancellationOrigin.PROVIDER_UNINTENDED:
            return RetryClass.PROVIDER_JOB_CANCELLED_UNINTENTIONAL
        if origin is CancellationOrigin.USER_INTENTIONAL:
            return RetryClass.NONRETRYABLE_CANCELLED
        return RetryClass.PROVIDER_JOB_CANCELLED_UNKNOWN_NONRETRYABLE
    raise ValueError(f"Unsupported provider terminal state: provider={provider}, state={state}")