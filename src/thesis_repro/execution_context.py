"""Authoritative execution identity for every fresh run."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Iterator

CONTRACT_SMOKE = "CONTRACT_SMOKE"
SYNTHETIC_E2E_ACCEPTANCE = "SYNTHETIC_E2E_ACCEPTANCE"
FRESH_REPLICATION = "FRESH_REPLICATION"

@dataclass(frozen=True)
class ExecutionContext:
    execution_class: str
    profile: str
    run_id: str
    requested_mode: str
    certification_allowed: bool
    scientific_gate_applicable: bool

    @classmethod
    def from_profile(cls, profile: str, run_id: str, requested_mode: str) -> "ExecutionContext":
        normalized = str(profile).strip().lower()
        if normalized == "smoke":
            return cls(CONTRACT_SMOKE, normalized, run_id, requested_mode, False, False)
        if normalized == "synthetic":
            return cls(SYNTHETIC_E2E_ACCEPTANCE, normalized, run_id, requested_mode, False, False)
        if normalized == "full":
            return cls(FRESH_REPLICATION, normalized, run_id, requested_mode, True, True)
        raise ValueError(f"unknown execution profile: {profile!r}")

    def to_dict(self) -> dict[str, object]:
        return {"execution_class": self.execution_class, "execution_profile": self.profile, "run_id": self.run_id, "requested_mode": self.requested_mode, "certification_allowed": self.certification_allowed, "scientific_gate_applicable": self.scientific_gate_applicable}

@contextmanager
def scoped_oracle_compatibility(context: ExecutionContext) -> Iterator[None]:
    name = "THESIS_REPRO_ORACLE_PROFILE"
    previous = os.environ.get(name)
    os.environ[name] = "synthetic" if context.execution_class == SYNTHETIC_E2E_ACCEPTANCE else "production"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def receipt_context(context: ExecutionContext | None) -> dict[str, object]:
    """Return the non-ambient execution identity embedded in every receipt."""
    if context is None:
        return {"execution_class": FRESH_REPLICATION, "execution_profile": "full", "scientific_gate_applicable": True}
    return {"execution_class": context.execution_class, "execution_profile": context.profile, "scientific_gate_applicable": context.scientific_gate_applicable}
