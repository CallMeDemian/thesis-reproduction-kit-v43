"""C2 item 8: which byte form each design_release.json source hash was computed from.

For each key of frozen/evidence/llm/design_release.json `source_hashes`:
  - kit side: the Git blob at <rev> (path as resolved in the kit, from the source hash report
    if given, else from the key map below), hashed raw, LF-normalised and CRLF-normalised;
  - zip side: the member at the original path recorded in the zip's
    configs/current/hash_manifest.json (same keys), hashed the same three ways.
The recorded value is then classified by which form(s) it equals.
A file containing a NUL byte in its first 8000 bytes is reported as binary.

Usage: python c2_design_release_hashes.py <kit_repo> <rev> <zip> [<source_hash_report.json>]
"""
import hashlib, json, subprocess, sys, zipfile

repo, rev, ZIP = sys.argv[1:4]
report = {e["name"]: e for e in json.load(open(sys.argv[4], encoding="utf-8"))} if len(sys.argv) > 4 else {}
h = lambda b: hashlib.sha256(b).hexdigest()
R = "THESIS_REPRO_KIT_v2.1.1_FINAL/"
z = zipfile.ZipFile(ZIP)
zman = json.loads(z.read(R + "configs/current/hash_manifest.json"))["files"]
design = json.loads(subprocess.run(["git", "-C", repo, "show", f"{rev}:frozen/evidence/llm/design_release.json"],
                                   capture_output=True, check=True).stdout)
src = design["source_hashes"]


def forms(b):
    lf = b.replace(b"\r\n", b"\n")
    return {"raw": h(b), "LF": h(lf), "CRLF": h(lf.replace(b"\n", b"\r\n"))}, (b"\0" in b[:8000]), b.count(b"\r\n"), b.count(b"\n")


def kit_blob(path):
    r = subprocess.run(["git", "-C", repo, "cat-file", "blob", f"{rev}:{path}"], capture_output=True)
    if r.returncode:
        return None
    b = r.stdout
    if b.startswith(b"version https://git-lfs.github.com/spec/v1"):
        b = subprocess.run(["git", "-C", repo, "lfs", "smudge"], input=b, capture_output=True, check=True).stdout
    return b


print("key\trecorded12\tkit_path\tkit_match\tzip_path\tzip_match\tzip_crlf_pairs/newlines\tbyte_basis")
for key, rec in sorted(src.items()):
    kp = report.get(key, {}).get("path", "")
    kb = kit_blob(kp) if kp and not kp.startswith("runs/") else None
    km = "-"
    if kb is not None:
        f, binary, _, _ = forms(kb)
        km = ",".join(k for k, v in f.items() if v == rec) or "NONE"
    zp = zman.get(key, {}).get("path", "")
    zm, zinfo, basis = "-", "", ""
    if zp and R + zp in z.namelist():
        f, binary, crlf, nl = forms(z.read(R + zp))
        zm = ",".join(k for k, v in f.items() if v == rec) or "NONE"
        zinfo = f"{crlf}/{nl}"
        if binary:
            basis = "binary"
        elif "LF" in zm.split(",") and "CRLF" in zm.split(","):
            basis = "no line breaks"
        elif zm == "raw,CRLF" or zm == "CRLF" or (km and "CRLF" in km.split(",") and "LF" not in km.split(",")):
            basis = "CRLF"
        elif zm in ("raw,LF", "LF") or (km and "LF" in km.split(",")):
            basis = "LF"
        elif zm == "raw":
            basis = "mixed (raw zip bytes)"
        else:
            basis = "no match"
    if not basis:
        if km != "-" and km != "NONE":
            basis = "CRLF" if "CRLF" in km.split(",") and "LF" not in km.split(",") else ("LF" if "LF" in km.split(",") else km)
        else:
            basis = "not available in kit or zip"
    print(f"{key}\t{rec[:12]}\t{kp}\t{km}\t{zp}\t{zm}\t{zinfo}\t{basis}")
