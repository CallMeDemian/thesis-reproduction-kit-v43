"""Central execution-state contract for fresh scientific runs.

These values are deliberately small and closed.  Stage adapters may add detail
fields, but they must not invent a second terminal-state vocabulary.
"""

from __future__ import annotations

from typing import Final


PASS: Final = "PASS"
PASS_WITH_QUALIFICATION: Final = "PASS_WITH_QUALIFICATION"
FAILED: Final = "FAILED"
INPUT_REQUIRED: Final = "INPUT_REQUIRED"
APPROVAL_REQUIRED: Final = "APPROVAL_REQUIRED"
CREDENTIALS_REQUIRED: Final = "CREDENTIALS_REQUIRED"
RESOURCE_REQUIRED: Final = "RESOURCE_REQUIRED"
NOT_IMPLEMENTED: Final = "NOT_IMPLEMENTED"
NOT_EXECUTED: Final = "NOT_EXECUTED"
EXECUTED_UNVERIFIED: Final = "EXECUTED_UNVERIFIED"
SMOKE_PASS: Final = "SMOKE_PASS"
SMOKE_PASS_WITH_SKIPS: Final = "SMOKE_PASS_WITH_SKIPS"

SCIENTIFIC_ACCEPTED: Final = frozenset({PASS, PASS_WITH_QUALIFICATION})
SMOKE_TERMINAL: Final = frozenset({SMOKE_PASS, SMOKE_PASS_WITH_SKIPS})
BLOCKING: Final = frozenset({
    FAILED,
    INPUT_REQUIRED,
    APPROVAL_REQUIRED,
    CREDENTIALS_REQUIRED,
    RESOURCE_REQUIRED,
    NOT_IMPLEMENTED,
    NOT_EXECUTED,
    EXECUTED_UNVERIFIED,
})


def is_scientific_accepted(status: str) -> bool:
    return status in SCIENTIFIC_ACCEPTED


def is_smoke(status: str) -> bool:
    return status in SMOKE_TERMINAL


def is_terminal_block(status: str) -> bool:
    return status in BLOCKING


def normalize_legacy_status(status: str) -> str:
    """Map pre-contract adapter labels to the single public vocabulary."""
    return {
        "PASS_WITH_SKIPS": SMOKE_PASS_WITH_SKIPS,
        "IMPLEMENTED_UNEXECUTED": NOT_EXECUTED,
        "ORACLE_EXECUTION_FAILED": FAILED,
        "ORACLE_VERIFICATION_FAILED": FAILED,
        "ORACLE_ARTIFACTS_INCOMPLETE": FAILED,
        "HEAVY_EXECUTION_APPROVAL_REQUIRED": APPROVAL_REQUIRED,
        "LIVE_LLM_APPROVAL_REQUIRED": APPROVAL_REQUIRED,
    }.get(status, status)


def aggregate_completion(statuses: list[str], *, profile: str) -> str:
    """Produce a truthful run terminal state; never downgrade failure to smoke."""
    normalized = [normalize_legacy_status(value) for value in statuses]
    if profile == "smoke":
        return SMOKE_PASS if normalized and all(value == SMOKE_PASS for value in normalized) else SMOKE_PASS_WITH_SKIPS
    for candidate in (
        FAILED,
        INPUT_REQUIRED,
        APPROVAL_REQUIRED,
        CREDENTIALS_REQUIRED,
        RESOURCE_REQUIRED,
        NOT_IMPLEMENTED,
        NOT_EXECUTED,
        EXECUTED_UNVERIFIED,
    ):
        if candidate in normalized:
            return candidate
    if normalized and all(value == PASS for value in normalized):
        return PASS
    if normalized and all(value in SCIENTIFIC_ACCEPTED for value in normalized):
        return PASS_WITH_QUALIFICATION
    return FAILED

