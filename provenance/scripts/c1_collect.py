"""C1 data collection: compare kit 4a70816 src/credit_recourse files with original 449c942c.

Read-only on both repositories. Blobs are extracted to a temp folder outside the kit
so that `git diff --no-index --numstat` can be run between the two exact blob byte strings.

Usage:
  python c1_collect.py <kit_repo> <orig_repo> <lineage_csv> <tmp_dir> <out_json>
"""
import csv, hashlib, json, os, subprocess, sys

KIT, ORIG, LINEAGE, TMP, OUT = sys.argv[1:6]
KIT_REV, ORIG_REV = "4a70816", "449c942c"


def git(repo, *args, binary=False):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


rows = [r for r in csv.DictReader(open(LINEAGE, encoding="utf-8-sig"))
        if r["status"] == "UNRECORDED_DIFFERENCE"]
assert len(rows) == 57, len(rows)
os.makedirs(TMP, exist_ok=True)
out = []
for i, r in enumerate(rows):
    p = r["path"]
    ob = git(ORIG, "cat-file", "blob", f"{ORIG_REV}:{p}", binary=True)
    cb = git(KIT, "cat-file", "blob", f"{KIT_REV}:{p}", binary=True)
    fo, fc = os.path.join(TMP, f"{i:02d}_orig"), os.path.join(TMP, f"{i:02d}_kit")
    open(fo, "wb").write(ob)
    open(fc, "wb").write(cb)
    ns = subprocess.run(["git", "diff", "--no-index", "--numstat", fo, fc],
                        capture_output=True).stdout.decode().split()
    add, rem = int(ns[0]), int(ns[1])
    log = git(KIT, "log", "--format=%h%x09%ad%x09%s", "--date=short", KIT_REV, "--", p).strip().splitlines()
    bodies = {}
    for line in log:
        h = line.split("\t")[0]
        bodies[h] = git(KIT, "show", "-s", "--format=%B", h).strip()
    out.append(dict(path=p, idx=i,
                    original_sha256=hashlib.sha256(ob).hexdigest(),
                    current_sha256=hashlib.sha256(cb).hexdigest(),
                    added=add, removed=rem, lines_changed=add + rem,
                    commits=[l.split("\t") for l in log], bodies=bodies))
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out), "files;", sum(o["lines_changed"] for o in out), "lines changed in total")
