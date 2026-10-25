"""C2 item 3: recompute configs/current/hash_manifest.json inside the distribution zip.

For each entry: sha256 of the zip member bytes (raw), and of the same bytes with
line endings normalised to LF and to CRLF. Optionally looks the recorded hash up in
the original bundle history (all blobs of every commit) to find which version it matches.

Usage: python c2_zip_hash_manifest.py <zip> [<orig_repo>]
"""
import hashlib, json, subprocess, sys, zipfile

ZIP = sys.argv[1]
ORIG = sys.argv[2] if len(sys.argv) > 2 else None
z = zipfile.ZipFile(ZIP)
R = "THESIS_REPRO_KIT_v2.1.1_FINAL/"
names = set(z.namelist())
m = json.loads(z.read(R + "configs/current/hash_manifest.json"))
h = lambda b: hashlib.sha256(b).hexdigest()

blob_index = {}
if ORIG:
    # sha256 of every blob reachable from any ref, path -> commits
    revs = subprocess.run(["git", "-C", ORIG, "rev-list", "--all"], capture_output=True, text=True).stdout.split()
    wanted = {e["path"] for e in m["files"].values()}
    for rev in revs:
        tree = subprocess.run(["git", "-C", ORIG, "ls-tree", "-r", rev], capture_output=True, text=True).stdout
        for line in tree.splitlines():
            meta, path = line.split("\t", 1)
            if path in wanted:
                oid = meta.split()[2]
                if oid not in blob_index:
                    b = subprocess.run(["git", "-C", ORIG, "cat-file", "blob", oid], capture_output=True).stdout
                    lf = b.replace(b"\r\n", b"\n")
                    blob_index[oid] = (h(b), h(lf.replace(b"\n", b"\r\n")), path, [])
                blob_index[oid][3].append(rev[:8])

rows = []
for key, e in sorted(m["files"].items()):
    member = R + e["path"]
    if member not in names:
        rows.append((key, e["path"], e["sha256"], "NOT_IN_ZIP", "", "", ""))
        continue
    b = z.read(member)
    lf = b.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    status = "MATCH" if h(b) == e["sha256"] else ("MATCH_CRLF" if h(crlf) == e["sha256"] else ("MATCH_LF" if h(lf) == e["sha256"] else "DIFFERENT"))
    hit = ""
    if status == "DIFFERENT" and blob_index:
        for oid, (raw, cr, p, revs_) in blob_index.items():
            if p == e["path"] and e["sha256"] in (raw, cr):
                hit = f"bundle blob {oid[:10]} ({'raw' if e['sha256']==raw else 'CRLF'}) in commits {','.join(sorted(set(revs_)))}"
    rows.append((key, e["path"], e["sha256"], status, h(b), h(crlf), hit))

print("key\tpath\trecorded\tstatus\tzip_raw_sha256\tzip_crlf_sha256\trecorded_found_in")
for r in rows:
    print("\t".join(r))
print("DIFFERENT:", sum(r[3] == "DIFFERENT" for r in rows), "of", len(rows))
