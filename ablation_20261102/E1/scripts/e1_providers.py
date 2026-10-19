"""Provider transport for E1: keys, synchronous smoke calls, Batch submit / find / poll / download.

Mirrors ``final_release.live_batch`` (``_openai_submit``, ``_google_submit``, ``_find_remote_jobs``,
``_download_openai``, ``_download_google``) but names jobs ``e1-<provider>-shard<k>-attempt<n>`` and
uses the High request bodies as they are in the shard files. ``providers.build_batch_record`` is never
used (it writes Baseline settings).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def load_keys() -> None:
    """Fill OPENAI_API_KEY / GEMINI_API_KEY from the Windows user environment when the process lacks them.

    Values are never printed or logged.
    """
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(name):
            continue
        if sys.platform == "win32":
            import winreg

            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                    value, _ = winreg.QueryValueEx(key, name)
                if value:
                    os.environ[name] = value
            except OSError:
                pass
    missing = [n for n in ("OPENAI_API_KEY", "GEMINI_API_KEY") if not os.environ.get(n)]
    if missing:
        raise SystemExit(f"Missing environment variables: {missing}")


def job_name(provider: str, shard: int, attempt: int) -> str:
    return f"e1-{provider}-shard{int(shard)}-attempt{int(attempt)}"


def _google_client():
    from google import genai

    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


# ---------------------------------------------------------------- synchronous (smoke test only)
def sync_openai(record: dict[str, Any], timeout: float = 1200.0) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(timeout=timeout, max_retries=0)
    resp = client.responses.create(**record["body"])
    return {"custom_id": record["custom_id"], "response": {"status_code": 200, "request_id": getattr(resp, "_request_id", None),
                                                           "body": resp.model_dump(mode="json")}, "error": None}


def sync_google(record: dict[str, Any], model: str) -> dict[str, Any]:
    from google.genai import types

    req = record["request"]
    cfg = req["generationConfig"]
    config = types.GenerateContentConfig(
        system_instruction=req["systemInstruction"]["parts"][0]["text"],
        max_output_tokens=int(cfg["maxOutputTokens"]),
        thinking_config=types.ThinkingConfig(thinking_level=str(cfg["thinkingConfig"]["thinkingLevel"]).upper()),
    )
    client = _google_client()  # keep a reference: a temporary client is closed on garbage collection
    resp = client.models.generate_content(model=model, contents=req["contents"], config=config)
    # by_alias gives the camelCase layout of Batch output lines (modelVersion, usageMetadata, finishReason)
    return {"key": record["key"], "response": resp.model_dump(mode="json", by_alias=True, exclude_none=True)}


# ---------------------------------------------------------------- batch
def find_jobs(provider: str, name: str) -> list[dict[str, Any]]:
    """Existing provider jobs whose display name equals ``name``."""
    out: list[dict[str, Any]] = []
    if provider == "openai":
        from openai import OpenAI

        page = OpenAI().batches.list(limit=100)
        while True:
            for job in page:
                meta = dict(getattr(job, "metadata", None) or {})
                if str(meta.get("e1_job") or "") == name:
                    out.append({"provider": "openai", "batch_job_id": str(job.id), "state": str(job.status),
                                "input_file_id": str(getattr(job, "input_file_id", "") or "")})
            if not getattr(page, "has_next_page", lambda: False)():
                break
            page = page.get_next_page()
    else:
        client = _google_client()
        for job in client.batches.list(config={"page_size": 100}):
            if str(getattr(job, "display_name", "") or "") == name:
                state = str(getattr(getattr(job, "state", None), "name", getattr(job, "state", "")))
                out.append({"provider": "google", "batch_job_id": str(job.name), "state": state, "input_file_id": ""})
    return sorted({j["batch_job_id"]: j for j in out}.values(), key=lambda j: j["batch_job_id"])


def submit_openai(path: Path, name: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    with path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    job = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses", completion_window="24h",
                                metadata={"e1_job": name})
    return {"provider": "openai", "batch_job_id": job.id, "input_file_id": uploaded.id, "state": str(job.status)}


def submit_google(path: Path, model: str, name: str) -> dict[str, Any]:
    from google.genai import types

    client = _google_client()
    uploaded = client.files.upload(file=path, config=types.UploadFileConfig(display_name=name, mime_type="jsonl"))
    job = client.batches.create(model=model, src=uploaded.name, config={"display_name": name})
    return {"provider": "google", "batch_job_id": job.name, "input_file_id": uploaded.name, "state": str(job.state)}


OPENAI_TERMINAL = {"completed", "failed", "expired", "cancelled"}
GOOGLE_TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_EXPIRED", "JOB_STATE_CANCELLED"}


def poll_openai(job_id: str) -> dict[str, Any]:
    from openai import OpenAI

    job = OpenAI().batches.retrieve(job_id)
    counts = getattr(job, "request_counts", None)
    return {"state": str(job.status), "terminal": str(job.status).lower() in OPENAI_TERMINAL,
            "output_file_id": getattr(job, "output_file_id", None), "error_file_id": getattr(job, "error_file_id", None),
            "counts": counts.model_dump() if counts is not None else None}


def poll_google(job_id: str) -> dict[str, Any]:
    client = _google_client()
    job = client.batches.get(name=job_id)
    state = str(getattr(getattr(job, "state", None), "name", getattr(job, "state", "")))
    dest = getattr(job, "dest", None)
    return {"state": state, "terminal": state in GOOGLE_TERMINAL, "dest_file": getattr(dest, "file_name", None), "counts": None}


def download_openai(info: dict[str, Any], target_dir: Path, stem: str) -> list[dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI()
    arts = []
    for kind, fid in (("output", info.get("output_file_id")), ("error", info.get("error_file_id"))):
        if not fid:
            continue
        path = target_dir / f"{stem}_{kind}.jsonl"
        client.files.content(fid).write_to_file(path)
        arts.append({"kind": kind, "file_id": str(fid), "path": path.name})
    return arts


def download_google(info: dict[str, Any], target_dir: Path, stem: str) -> list[dict[str, Any]]:
    name = info.get("dest_file")
    if not name:
        return []
    client = _google_client()
    content = client.files.download(file=name)
    path = target_dir / f"{stem}_output.jsonl"
    path.write_bytes(content if isinstance(content, bytes) else bytes(content))
    return [{"kind": "output", "file_id": str(name), "path": path.name}]
