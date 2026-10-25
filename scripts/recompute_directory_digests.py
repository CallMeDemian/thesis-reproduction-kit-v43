#!/usr/bin/env python3
"""Recompute the directory digests that the kit records without a stated algorithm.

Targets (all five are checked; the other ORIGINAL_RELEASE_MANIFEST packs are checked too):
  - frozen/original_release/ORIGINAL_RELEASE_MANIFEST.json `pack_index[].sha256` for
    pack:llm_final_plan3, pack:oracle_stage1_inputs, pack:rl_c3e_e2_final, pack:simulator_release
  - frozen/release/frozen_manifest.json, artifact.v13.frozen_capsule `sha256`

For every digest the script tries a grid of standard constructions and prints each one that
reproduces the recorded value exactly, or "정의 미확인" when none does.

Grid (paths are always sorted; files are read from the kit folder, which must have Git LFS
content pulled; a Git LFS pointer file is hashed by its pointer oid, which is the sha256 of
the real content):
  file set     disk    every file under the directory on disk
               pack    the pack's file list in provenance/ASSET_PACK_MANIFEST.json, hashed from disk
               packrec the pack's file list with the sha256 recorded in ASSET_PACK_MANIFEST.json
               relrec  the ORIGINAL_RELEASE_MANIFEST records whose parent is the pack (recorded sha256)
               git     `git ls-files -z` under the directory (only if the kit folder is a Git checkout)
               gitq    `git ls-files` with Git's default path quoting (core.quotepath=true forced), keeping only names that
                       exist as files (non-ASCII names come out quoted and drop out). This is what
                       `tree_sha256()` in the source release's src/credit_recourse/repro/release.py does.
               srcq    same as gitq, but the names are listed from the source repository (option
                       --orig, see below) at the directory's original location, and read from the kit
               srcblob same names as srcq, hashed from the source repository's blobs (text suffixes
                       CRLF->LF); files whose kit copy differs are printed as notes
  file bytes   raw | lf (CRLF->LF) | crlf (LF->CRLF) | textlf (CRLF->LF only for
               .csv .json .md .py .txt .yaml .yml)       [ignored for recorded-hash file sets]
  path form    dir (relative to the directory) | root (relative to the kit root) |
               named (directory name + "/" + dir-relative path)
  sort         bytes (Python str order = Git byte order for UTF-8) | casefold (Windows Path order)
  serialisation
               tab_nl       "path\\tsha\\n" for every row
               tab_join     "path\\tsha" rows joined by "\\n" (no final newline)
               sums         "sha  path\\n" for every row
               json_pairs   json.dumps([[path, sha], ...], separators=(",", ":"))
               json_map     json.dumps({path: sha}, sort_keys=True, separators=(",", ":"))
               json_map_sp  json.dumps({path: sha}, sort_keys=True)
               json_objs    json.dumps([{"path": p, "sha256": s}, ...], separators=(",", ":"))

Standard library only (pandas is not needed).

Usage: python recompute_directory_digests.py <kit_folder> [--orig <source_repo>] [--quiet]

<source_repo> is a clone of the kit's frozen/original_release/source/REPRO_KIT_semantic_repair.bundle
with its refs fetched, for example:
    git clone --no-checkout <kit>/frozen/original_release/source/REPRO_KIT_semantic_repair.bundle orig
    git -C orig fetch <bundle> "refs/*:refs/remotes/bundle/*"
It is used to list which files the source repository tracked in a directory (set srcq) and to
hash those files from the source blobs (set srcblob).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
LFS = re.compile(rb"^version https://git-lfs\.github\.com/spec/v1\r?\noid sha256:([0-9a-f]{64})\r?\nsize \d+\r?\n?$")
TARGETS = ["pack:llm_final_plan3", "pack:oracle_stage1_inputs", "pack:rl_c3e_e2_final", "pack:simulator_release",
           "artifact.v13.frozen_capsule"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FileHasher:
    """Caches raw/lf/crlf/textlf sha256 per absolute path."""

    def __init__(self) -> None:
        self.cache: dict[Path, dict[str, str]] = {}

    def get(self, path: Path) -> dict[str, str]:
        if path not in self.cache:
            data = path.read_bytes()
            m = LFS.match(data) if len(data) < 400 else None
            if m:  # pointer: oid is the content hash; line-ending variants are not defined
                oid = m.group(1).decode()
                self.cache[path] = {"raw": oid, "lf": oid, "crlf": oid, "textlf": oid}
            elif b"\0" in data[:8000]:
                h = sha(data)
                self.cache[path] = {"raw": h, "lf": h, "crlf": h, "textlf": h}
            else:
                lf = data.replace(b"\r\n", b"\n")
                raw = sha(data)
                lfh = sha(lf)
                self.cache[path] = {"raw": raw, "lf": lfh, "crlf": sha(lf.replace(b"\n", b"\r\n")),
                                    "textlf": lfh if path.suffix.lower() in TEXT_SUFFIXES else raw}
        return self.cache[path]


def serialise(kind: str, rows: list[tuple[str, str]]) -> bytes:
    if kind == "tab_nl":
        return "".join(f"{p}\t{s}\n" for p, s in rows).encode("utf-8")
    if kind == "tab_join":
        return "\n".join(f"{p}\t{s}" for p, s in rows).encode("utf-8")
    if kind == "sums":
        return "".join(f"{s}  {p}\n" for p, s in rows).encode("utf-8")
    if kind == "json_pairs":
        return json.dumps([[p, s] for p, s in rows], separators=(",", ":")).encode("utf-8")
    if kind == "json_map":
        return json.dumps(dict(rows), sort_keys=True, separators=(",", ":")).encode("utf-8")
    if kind == "json_map_sp":
        return json.dumps(dict(rows), sort_keys=True).encode("utf-8")
    if kind == "json_objs":
        return json.dumps([{"path": p, "sha256": s} for p, s in rows], separators=(",", ":")).encode("utf-8")
    raise ValueError(kind)


SERIALISATIONS = ["tab_nl", "tab_join", "sums", "json_pairs", "json_map", "json_map_sp", "json_objs"]


def git_ls_files(kit: Path, rel_dir: str) -> list[str] | None:
    try:
        r = subprocess.run(["git", "-C", str(kit), "ls-files", "-z", "--", rel_dir], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    names = [n for n in r.stdout.decode("utf-8").split("\0") if n]
    return names or None


def git_quoted_existing(kit: Path, rel_dir: str) -> list[str] | None:
    """Emulate tree_sha256(): plain `git ls-files` (default quoting), keep existing files."""
    try:
        r = subprocess.run(["git", "-c", "core.quotepath=true", "-C", str(kit), "ls-files", "--", rel_dir], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    names = [ln.strip() for ln in r.stdout.decode("utf-8").splitlines() if ln.strip()]
    kept = [n for n in names if (kit / n).is_file()]
    return kept or None


def source_quoted_existing(orig: Path, kit: Path, rel_dir: str) -> dict[str, list[str]]:
    """For every ref of the source repo that has a directory with the same basename, list its
    tracked files with default quoting, map them into the kit directory and keep existing files."""
    base = rel_dir.rstrip("/").split("/")[-1]
    out: dict[str, list[str]] = {}
    refs = subprocess.run(["git", "-C", str(orig), "for-each-ref", "--format=%(objectname:short) %(refname)"],
                          capture_output=True, text=True).stdout.split("\n")
    for line in filter(None, refs):
        rev, ref = line.split(" ", 1)
        listing = subprocess.run(["git", "-c", "core.quotepath=true", "-C", str(orig), "ls-tree", "-r", "--name-only", rev],
                                 capture_output=True).stdout.decode("utf-8").splitlines()
        marker = "/" + base + "/"
        hits = [n for n in listing if marker in n or n.startswith(base + "/")]
        if not hits:
            continue
        kept = []
        for n in hits:
            rel = n.split(marker, 1)[1] if marker in n else n[len(base) + 1:]
            p = f"{rel_dir}/{rel}"
            if (kit / p).is_file():  # quoted (non-ASCII) names never exist as files
                kept.append(p)
        out[f"srcq@{rev}"] = kept
    # collapse identical lists
    uniq: dict[str, list[str]] = {}
    for k, v in out.items():
        if v not in uniq.values():
            uniq[k] = v
    return uniq


def source_blobs(orig: Path, rel_dir: str) -> dict[str, list[tuple[str, str]]]:
    """Like source_quoted_existing, but hash the source repository's own blobs (CRLF->LF for text
    suffixes) instead of the kit files. Quoted (non-ASCII) names are skipped, as tree_sha256() did.
    Returns {"srcblob@<rev>": [(kit_path, sha256), ...]} for each distinct listing."""
    base = rel_dir.rstrip("/").split("/")[-1]
    marker = "/" + base + "/"
    out: dict[str, list[tuple[str, str]]] = {}
    refs = subprocess.run(["git", "-C", str(orig), "for-each-ref", "--format=%(objectname:short)"],
                          capture_output=True, text=True).stdout.split()
    for rev in refs:
        listing = subprocess.run(["git", "-c", "core.quotepath=true", "-C", str(orig), "ls-tree", "-r", "--name-only", rev],
                                 capture_output=True).stdout.decode("utf-8").splitlines()
        rows = []
        for n in listing:
            if (marker not in n and not n.startswith(base + "/")) or n.startswith('"'):
                continue
            rel = n.split(marker, 1)[1] if marker in n else n[len(base) + 1:]
            data = subprocess.run(["git", "-C", str(orig), "show", f"{rev}:{n}"], capture_output=True, check=True).stdout
            if Path(n).suffix.lower() in TEXT_SUFFIXES:
                data = data.replace(b"\r\n", b"\n")
            rows.append((f"{rel_dir}/{rel}", sha(data)))
        if rows and rows not in out.values():
            out[f"srcblob@{rev}"] = rows
    return out


def candidates(kit: Path, rel_dir: str, file_sets: dict[str, list[tuple[str, str | None]]], hasher: FileHasher):
    """Yield (label, digest) for every construction.

    file_sets maps a set name to [(root_relative_path, recorded_sha256_or_None)].
    """
    name = rel_dir.rstrip("/").split("/")[-1]
    for set_name, entries in file_sets.items():
        recorded = all(s is not None for _, s in entries)
        byte_modes = ["recorded"] if recorded else ["raw", "lf", "crlf", "textlf"]
        for mode in byte_modes:
            hashed = []
            for rp, s in entries:
                hashed.append((rp, s if recorded else hasher.get(kit / rp)[mode]))
            for pform in ("dir", "root", "named"):
                rows = []
                for rp, h in hashed:
                    dr = rp[len(rel_dir) + 1:] if rp.startswith(rel_dir + "/") else None
                    if pform == "dir":
                        p = dr if dr is not None else rp  # file outside the directory keeps its root path
                    elif pform == "root":
                        p = rp
                    else:
                        p = f"{name}/{dr}" if dr is not None else rp
                    rows.append((p, h))
                for sort in ("bytes", "casefold"):
                    srt = sorted(rows, key=(lambda r: r[0]) if sort == "bytes" else (lambda r: r[0].lower()))
                    for ser in SERIALISATIONS:
                        yield f"set={set_name} bytes={mode} path={pform} sort={sort} ser={ser}", sha(serialise(ser, srt))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    kit = Path(argv[1]).resolve()
    quiet = "--quiet" in argv
    orig = Path(argv[argv.index("--orig") + 1]).resolve() if "--orig" in argv else None
    orm = json.loads((kit / "frozen/original_release/ORIGINAL_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    apm_path = kit / "provenance/ASSET_PACK_MANIFEST.json"
    apm = json.loads(apm_path.read_text(encoding="utf-8")) if apm_path.is_file() else {"packs": []}
    fm = json.loads((kit / "frozen/release/frozen_manifest.json").read_text(encoding="utf-8"))
    packs = {f"pack:{p['pack_id']}": p for p in apm["packs"]}
    hasher = FileHasher()

    jobs = []  # (id, rel_dir, recorded, file_sets)
    for item in orm["pack_index"]:
        if not str(item["logical_id"]).startswith("pack:"):
            continue
        rel_dir = item["path"]
        sets: dict[str, list[tuple[str, str | None]]] = {}
        d = kit / rel_dir
        if d.is_dir():
            sets["disk"] = [(p.relative_to(kit).as_posix(), None) for p in d.rglob("*") if p.is_file()]
        pk = packs.get(item["logical_id"])
        if pk:
            sets["pack"] = [(f["planned_target_path"], None) for f in pk["files"] if (kit / f["planned_target_path"]).is_file()]
            if len(sets["pack"]) != len(pk["files"]):
                del sets["pack"]
            sets["packrec"] = [(f["planned_target_path"], f["sha256"]) for f in pk["files"]]
        rel = [(r["path"], r["sha256"]) for r in orm["records"] if item["logical_id"] in r.get("parents", [])]
        if rel:
            sets["relrec"] = rel
        g = git_ls_files(kit, rel_dir)
        if g:
            sets["git"] = [(n, None) for n in g]
        q = git_quoted_existing(kit, rel_dir)
        if q:
            sets["gitq"] = [(n, None) for n in q]
        jobs.append((item["logical_id"], rel_dir, item["sha256"], sets))
    for art in fm.get("artifacts", []):
        if art.get("logical_id") == "artifact.v13.frozen_capsule":
            rel_dir = art["relative_path"]
            d = kit / rel_dir
            sets = {"disk": [(p.relative_to(kit).as_posix(), None) for p in d.rglob("*") if p.is_file()]}
            g = git_ls_files(kit, rel_dir)
            if g:
                sets["git"] = [(n, None) for n in g]
            q = git_quoted_existing(kit, rel_dir)
            if q:
                sets["gitq"] = [(n, None) for n in q]
            if orig:
                for k, v in source_quoted_existing(orig, kit, rel_dir).items():
                    sets[k] = [(n, None) for n in v]
                for k, v in source_blobs(orig, rel_dir).items():
                    sets[k] = v
                    # files whose kit copy differs from the source blob (expected: provenance/POST_FREEZE_EDITS.csv)
                    for rp, h in v:
                        if not (kit / rp).is_file() or hasher.get(kit / rp)["textlf"] != h:
                            print(f"note: {k} {rp} differs from the kit copy")
            jobs.append((art["logical_id"], rel_dir, art["sha256"], sets))

    jobs.sort(key=lambda j: (j[0] not in TARGETS, j[0]))
    for logical_id, rel_dir, recorded, sets in jobs:
        matches = [label for label, digest in candidates(kit, rel_dir, sets, hasher) if digest == recorded]
        tag = "TARGET" if logical_id in TARGETS else "other"
        sizes = ", ".join(f"{k}={len(v)}" for k, v in sets.items())
        print(f"[{tag}] {logical_id}  ({rel_dir})")
        print(f"    recorded  {recorded}")
        print(f"    file sets {sizes}")
        if matches:
            print(f"    MATCH ({len(matches)} construction(s))")
            for m in (matches if not quiet else matches[:3]):
                print(f"      {m}")
        else:
            print("    정의 미확인")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
