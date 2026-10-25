"""C2 item 7: recompute the QA Q12 combined frozen/ hash (ablation_20261102/scripts/90_qa.py, q12).

Q12 definition (copied from 90_qa.py): for every tracked file under frozen/ in
`git ls-files -z -- frozen` order, line = "<sha256(file bytes)>  <posix path>", and
combined = sha256 over (line + "\n") for all lines.

This script reads a checkout made with GIT_LFS_SKIP_SMUDGE=1 (LFS files are pointers on
disk). A Git LFS pointer's oid is the sha256 of the real content, so for LFS files the
oid is used, which equals hashing the pulled content. Run it on two checkouts of the same
commit, one with core.autocrlf=false (LF bytes, as on GitHub) and one with
core.autocrlf=true (Windows default), to get both values.

Usage: python c2_q12_combined_hash.py <checkout_dir> [<other_checkout_dir>]
"""
import hashlib, re, subprocess, sys
from pathlib import Path

PTR = re.compile(rb"^version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize \d+\n?$")


def file_digests(root):
    root = Path(root)
    names = [t for t in subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "frozen"],
                                       capture_output=True, check=True).stdout.decode("utf-8").split("\0") if t]
    out = []
    for n in names:
        b = (root / n).read_bytes()
        m = PTR.match(b.replace(b"\r\n", b"\n")) if len(b) < 300 else None
        out.append((n, m.group(1).decode() if m else hashlib.sha256(b).hexdigest(), bool(m)))
    return out


def combined(digests):
    h = hashlib.sha256()
    for n, d, _ in digests:
        h.update(f"{d}  {n}\n".encode("utf-8"))
    return h.hexdigest()


a = file_digests(sys.argv[1])
print(f"{sys.argv[1]}: files={len(a)} lfs={sum(x[2] for x in a)} combined_sha256={combined(a)}")
if len(sys.argv) > 2:
    b = file_digests(sys.argv[2])
    print(f"{sys.argv[2]}: files={len(b)} lfs={sum(x[2] for x in b)} combined_sha256={combined(b)}")
    diff = [x[0] for x, y in zip(a, b) if x[1] != y[1]]
    assert [x[0] for x in a] == [y[0] for y in b]
    print(f"files whose sha256 differs between the two checkouts: {len(diff)}")
    for n in diff[:5]:
        print("  e.g.", n)
