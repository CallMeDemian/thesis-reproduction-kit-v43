"""Explicitly gated batch transport seam for fresh LLM replication.

The mock adapter is deterministic and is used by tests/CI.  Provider adapters
only prepare provider-native JSONL jobs; network submission is deliberately a
separate operation requiring the environment sentinel and credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GATE = "THESIS_REPRO_ENABLE_LIVE_LLM=I_APPROVE_FRESH_REPLICATION"


@dataclass(frozen=True)
class ProviderIdentity:
    name: str
    model: str
    transport: str = "batch"


def gate_status() -> dict[str, Any]:
    return {
        "sentinel": GATE,
        "authorized": os.environ.get("THESIS_REPRO_ENABLE_LIVE_LLM") == "I_APPROVE_FRESH_REPLICATION",
        "credentials": {"openai": bool(os.environ.get("OPENAI_API_KEY")), "gemini": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))},
    }


def render_provider_batch(requests: Iterable[dict[str, Any]], provider: ProviderIdentity, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for request in requests:
            if provider.name in {"openai", "google"}:
                from credit_recourse.final_release.final_release.providers import build_batch_record
                body = build_batch_record(
                    provider=provider.name,
                    exact_model_id=provider.model,
                    request_id=request["request_id"],
                    system_text=str(request.get("system_text", "V4.3 fresh replication system contract")),
                    user_text=json.dumps(request, ensure_ascii=False, sort_keys=True),
                    max_output_tokens=int(request.get("max_output_tokens", 2048)),
                )
            else:
                body = {"custom_id": request["request_id"], "model": provider.model, "metadata": {"cell_id": request["cell_id"], "regime": request["generation_regime"]}, "input": request}
            handle.write(json.dumps(body, ensure_ascii=False) + "\n")
            count += 1
    return {"provider": provider.name, "model": provider.model, "transport": provider.transport, "request_count": count, "path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def mock_responses(requests: Iterable[dict[str, Any]], destination: Path) -> dict[str, Any]:
    """Create deterministic raw responses without claiming provider execution."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for request in requests:
            digest = hashlib.sha256(request["request_id"].encode()).hexdigest()
            response = {"request_id": request["request_id"], "provider": "mock", "transport": "mock", "raw_response": {"mode": "candidate_selection", "selected_candidate": "A0_noop", "reference_response": "not_shown", "confidence": "low", "diagnosis": {"primary_weakness": "mock", "evidence_feature_keys": ["mock"]}, "brief_rationale": f"mock:{digest[:12]}"}}
            handle.write(json.dumps(response, ensure_ascii=False) + "\n")
            count += 1
    return {"provider": "mock", "request_count": count, "path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "execution_class": "mock_only"}


def require_live_gate() -> None:
    if not gate_status()["authorized"]:
        raise PermissionError(f"live LLM is blocked; set {GATE}")


def submit_live_batch(*args, **kwargs):
    """Delegate transport to the production provider runner after the gate.

    Importing the runner is lazy so dry-render and CI never import provider SDKs
    or create network clients.
    """
    require_live_gate()
    from credit_recourse.final_release.final_release.live_batch import submit_wave
    return submit_wave(*args, **kwargs)
