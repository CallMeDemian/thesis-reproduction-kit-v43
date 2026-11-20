from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from google import genai
from openai import OpenAI

E2 = Path(__file__).resolve().parents[1]
E1_PROVIDERS = E2.parents[0] / "source_inputs/github_release_ablation_20261102_readonly/contents/ablation_20261102/E1/scripts/e1_providers.py"
OPENAI_MODEL = "gpt-5.4-mini-2026-03-17"
GOOGLE_MODEL = "gemini-3.1-flash-lite"
JOB_NAMES = {
    "openai": "e2-v5.1-pilot-20261118-openai-attempt1",
    "google": "e2-v5.1-pilot-20261118-google-attempt1",
}


def load_provider_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("e1_providers", E1_PROVIDERS)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main() -> None:
    missing = [k for k in ("OPENAI_API_KEY", "GEMINI_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Required provider credentials are unavailable: {missing}")

    oi = OpenAI().models.retrieve(OPENAI_MODEL)
    google_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    gi = google_client.models.get(model=GOOGLE_MODEL)
    providers = load_provider_module()
    jobs = {}
    for provider, name in JOB_NAMES.items():
        matches = providers.find_jobs(provider, name)
        jobs[provider] = matches
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "metadata-only GET and duplicate batch-name check; no generation request",
        "models": {
            "openai": {"requested_model_id": OPENAI_MODEL, "returned_model_id": str(oi.id), "available": str(oi.id) == OPENAI_MODEL},
            "google": {"requested_model_id": GOOGLE_MODEL, "returned_model_id": str(gi.name).split("/")[-1], "available": str(gi.name).split("/")[-1] == GOOGLE_MODEL},
        },
        "sdk_versions": {
            "openai": version("openai"),
            "google_genai": version("google-genai"),
        },
        "existing_jobs_with_reserved_names": jobs,
    }
    (E2 / "model_preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "metadata_preflight": "PASS" if all(v["available"] for v in result["models"].values()) else "FAIL",
        "models": result["models"],
        "existing_reserved_jobs": {k: len(v) for k, v in jobs.items()},
        "sdk_versions": result["sdk_versions"],
        "generation_calls": 0,
    }, ensure_ascii=False, indent=2))
    if not all(v["available"] for v in result["models"].values()):
        raise SystemExit(2)
    if any(jobs.values()):
        raise SystemExit("Reserved batch names already exist; do not submit duplicates")


if __name__ == "__main__":
    main()
