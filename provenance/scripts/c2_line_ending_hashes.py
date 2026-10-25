"""C2 items 7-8: sha256 of Git blob bytes (as stored, LF for text) and of the CRLF form.

Reads blobs with `git cat-file blob <rev>:<path>`; Git LFS pointers are resolved with
`git lfs smudge` (local object store only, no network: GIT_LFS_SKIP_DOWNLOAD is not
needed because smudge reads the local cache). Prints path, size, raw sha256, CRLF sha256,
LF sha256, and which of these equals an optional expected value.

Usage: python c2_line_ending_hashes.py <repo> <rev> <path>[=<expected_sha256>] ...
"""
import hashlib, subprocess, sys

repo, rev = sys.argv[1:3]
h = lambda b: hashlib.sha256(b).hexdigest()


def blob(path):
    b = subprocess.run(["git", "-C", repo, "cat-file", "blob", f"{rev}:{path}"], capture_output=True, check=True).stdout
    if b.startswith(b"version https://git-lfs.github.com/spec/v1"):
        b = subprocess.run(["git", "-C", repo, "lfs", "smudge"], input=b, capture_output=True, check=True).stdout
    return b


print("path\tbytes\tCR_LF_pairs\tsha256_raw\tsha256_LF\tsha256_CRLF\texpected_matches")
for arg in sys.argv[3:]:
    path, _, exp = arg.partition("=")
    b = blob(path)
    lf = b.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    forms = {"raw": h(b), "LF": h(lf), "CRLF": h(crlf)}
    match = ",".join(k for k, v in forms.items() if v == exp) if exp else ""
    pairs = b.count(b"\r\n")
    print(f"{path}\t{len(b)}\t{pairs}\t{forms['raw']}\t{forms['LF']}\t{forms['CRLF']}\t{match or ('NONE' if exp else '')}")
