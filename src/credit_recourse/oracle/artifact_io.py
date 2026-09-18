"""Immutable Oracle artifact readers shared by evaluators."""
from pathlib import Path
from typing import Any
import json
import yaml

def load_registry(path: Path) -> dict[str,Any]:
    if not path.exists(): raise FileNotFoundError(f"Missing oracle backend registry: {path}")
    text=path.read_text(encoding='utf-8')
    try: return json.loads(text)
    except Exception: return yaml.safe_load(text)

def resolve_backend_artifact(root: Path, final: Path, value: Any) -> Path:
    """Resolve backend artifact paths from portable registry entries.

    Supports:
      - existing absolute paths on the local machine,
      - project-root relative paths such as archive/DEPLOYED_RELEASE/..., and
      - stale absolute paths from another checkout by recovering the
        archive/DEPLOYED_RELEASE/... or stage1_oracle_backends/... suffix.
    """
    if value is None or str(value).strip() == "":
        return Path("")
    raw = str(value).strip()
    p = Path(raw)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((root / p).resolve())
        candidates.append((final / p).resolve())

    normalized = raw.replace('\\', '/')
    marker = 'archive/DEPLOYED_RELEASE/'
    if marker in normalized:
        suffix = normalized[normalized.index(marker):]
        candidates.append((root / suffix).resolve())
    marker2 = 'stage1_oracle_backends/'
    if marker2 in normalized:
        suffix = normalized[normalized.index(marker2):]
        candidates.append((final / suffix).resolve())

    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0] if candidates else p
