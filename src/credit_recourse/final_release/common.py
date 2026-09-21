from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class ContractError(RuntimeError):
    """Fail-closed scientific-contract or lineage error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception as exc:  # pragma: no cover - error text is the behavior
        raise ContractError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:
        raise ContractError(f"Cannot read CSV {path}: {exc}") from exc


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            for row in rows:
                payload = canonical_bytes(dict(row)) + b"\n"
                handle.write(payload)
                digest.update(payload)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count, digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractError(f"JSONL row is not an object: {path}:{line_number}")
            yield value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def require_file(root: Path, relative_or_absolute: str | Path) -> Path:
    path = Path(relative_or_absolute)
    if not path.is_absolute():
        path = Path(root) / path
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"Required file is missing: {path}")
    return path


def relative_to_root(root: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(Path(path).resolve())



def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        # The cleaned kit intentionally does not carry the old
        # ``configs/current`` tree.  Root discovery must therefore be based on
        # the repository contract, not on a deleted development registry.
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src").is_dir()
            and (candidate / "contracts").is_dir()
            and (candidate / "frozen").is_dir()
        ):
            return candidate
    raise ContractError("Could not locate repository root")


def published_contract_registry(root: Path) -> Path:
    """Return the immutable published registry, with legacy fallback explicit."""
    repo = Path(root).resolve()
    candidates = (
        repo / "configs/current/contract_manifest.json",
        repo / "frozen/original_release/simulator/configs/current/contract_manifest.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise ContractError("Published contract registry is unavailable")


def sha256_file(path: Path) -> str:
    return file_hash(Path(path))


def verify_declared_hash(root: Path, relative: str, expected: str) -> str:
    actual = file_hash(require_file(root, relative))
    if actual.lower() != str(expected).lower():
        raise ContractError(f"SHA-256 mismatch for {relative}: {actual} != {expected}")
    return actual
