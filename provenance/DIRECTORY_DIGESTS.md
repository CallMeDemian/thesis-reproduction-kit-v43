# Directory digests

## Scope

Five directory digests recorded in the kit had no documented algorithm and had
not been recomputed:

| id | where recorded | recorded value |
|---|---|---|
| `pack:llm_final_plan3` | `frozen/original_release/ORIGINAL_RELEASE_MANIFEST.json` `pack_index` | `b6dd052caa8abd396cd9a714f9de40496817d1f3a880f07f5b7d4bf5786bba2f` |
| `pack:oracle_stage1_inputs` | same | `995b9376c2dd289b91bf22f166fd191f179f1c1701a8d2ec460d25404fb680f6` |
| `pack:rl_c3e_e2_final` | same | `872d9605a1f59906d7168c2e6f2639f77f96305774aead195e92c48b397871ac` |
| `pack:simulator_release` | same | `ab7fa8bddec5b9e5147714ebadd26cb6832618695bdb3876e9b145cfb76c3554` |
| `artifact.v13.frozen_capsule` | `frozen/release/frozen_manifest.json` | `e887648864d55d8e2441647a3b4dc3c5f25d133c5314110f26ceb28315a555a2` |

All five are now reproduced exactly. The definitions were found in the
history of the kit and of the source release; neither is in the current tree.

Script: `scripts/recompute_directory_digests.py` (standard library only). It tries a
grid of constructions for every pack and for the capsule and prints each one
that reproduces the recorded value, or `정의 미확인`.

```
python scripts/recompute_directory_digests.py <kit_folder> --orig <source_repo>
```

`<source_repo>` is a clone of the kit's bundle with refs fetched (commands in
the script header). Without `--orig`, the four packs still match and the
capsule prints `정의 미확인` (reason below).

## How the definitions were found

```
git log --all --format=%h -S<digest>          # in the kit and in the bundle clone
git grep -l -I <digest> 4a70816               # current kit tree
git log --all -S"pack_index"                  # kit
git log --all -- scripts/build_original_release.py
```

- The four pack digests first appear in kit commit `5c1643a` (2026-09-19,
  "release: close original scientific DAG and verifier"), which also added
  `scripts/build_original_release.py`. That script was last changed in
  `fbc0bf7` and deleted in `63e03d3` (2026-09-21). It is readable with
  `git show 63e03d3^:scripts/build_original_release.py`.
- The capsule digest appears in source commit `0cebeb4` (2026-09-17,
  "fix(repro): normalize release-critical tree hashes"), which changed
  `tree_sha256()` in `src/credit_recourse/repro/release.py` (source repository,
  branch `final/thesis-v43-submission-20260917`; not in `449c942c`).

## Pack digests (`ORIGINAL_RELEASE_MANIFEST.json`)

Definition (`build_original_release.py`):

```
rows = [f"{planned_target_path}\t{sha256}" for each file of the pack in provenance/ASSET_PACK_MANIFEST.json]
digest = sha256(("\n".join(sorted(rows)) + "\n").encode())
```

`planned_target_path` is relative to the kit root. Recomputed from
`provenance/ASSET_PACK_MANIFEST.json` at `4a70816`, all 11 pack digests match,
including the four targets. Recomputed from the files on disk with the same
construction (root-relative paths, byte-order sort, `path\tsha256\n` per row),
the four targets also match, provided the file list is the pack's list.

Why a directory walk did not reproduce the four targets:

- `oracle_stage1_inputs`, `rl_c3e_e2_final`, `llm_final_plan3`: these are
  exactly the three packs whose paths mix upper and lower case so that a
  case-insensitive order (the order of `sorted()` on Windows `Path` objects)
  differs from byte order. With byte order they match; with case-folded
  order they do not. The other eight packs give the same order either way.
- `simulator_release`: the pack's file list has 33 files, 9 of which lie
  outside the pack's recorded root
  `frozen/original_release/simulator/src/credit_recourse/simulator`, under
  `frozen/original_release/simulator/configs/current/...`. A walk of the root
  finds 24 files. The recorded `file_count` 33 and `byte_size` 514,234 are
  those of the 33-file list (sum of the 24 files under the root: 237,359).

## v1.3 capsule digest (`frozen_manifest.json`)

Definition (`tree_sha256()` in the source release, after `0cebeb4`):

```
names = `git ls-files -- <dir>` (plain output), mapped to paths; files that do not exist are skipped
file hash = sha256(bytes), with CRLF->LF first for suffixes .csv .json .md .py .txt .yaml .yml
rows = [f"{path relative to <dir>}\t{file hash}" for names in git order]
digest = sha256("\n".join(rows).encode())          # no final newline
```

Applied to the kit capsule
(`frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN`,
37 files), this reproduces `e8876488…` over 34 files: the files tracked in the
source repository at the capsule's original location
(`repro_audit/FINAL_RELEASE/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN`,
36 files at `449c942c`, same blobs as the kit), minus the two files whose
names contain non-ASCII characters
(`manuscript/석사학위논문_조종선_V43_V13_REPRO_FINAL_20260915.docx` and `.pdf`).

The two exclusions follow from the definition itself:

- With Git's default `core.quotepath=true`, plain `git ls-files` prints
  non-ASCII paths quoted and octal-escaped (for example
  `"…/manuscript/\354\204\235…docx"`). The resulting path does not exist, so
  `tree_sha256()` skips it.
- `frozen_inputs/stage8/canonical_itt_observations.parquet` is tracked in the
  kit capsule (added in `af388eb`) but was not tracked there in the source
  repository, so it is not listed.

The kit folder used here has a local `core.quotepath=false`; the script forces
`core.quotepath=true` for this emulation so that the result does not depend
on local Git configuration. Without `--orig` the script cannot know which
files the source repository tracked, so the kit's own listing (35 files after
quoting, including the parquet) does not match and the capsule is reported as
`정의 미확인`.

The digest therefore does not cover the capsule's manuscript `.docx`/`.pdf`
or its Stage8 parquet. Those files are covered elsewhere: all three are listed
with matching hashes in the capsule's `SHA256SUMS.txt`, and the parquet is
also recorded separately (`stage8.canonical.itt`, `aedfadbf…`).

## Result of the run

`python scripts/recompute_directory_digests.py . --orig <clone of frozen/original_release/source/REPRO_KIT_semantic_repair.bundle>` (Windows checkout with Git LFS content)

| id | result | matching construction (byte-order sort) |
|---|---|---|
| `pack:llm_final_plan3` | MATCH | file list (disk, pack list, or recorded list) · root-relative · `path\tsha256\n` |
| `pack:oracle_stage1_inputs` | MATCH | same |
| `pack:rl_c3e_e2_final` | MATCH | same |
| `pack:simulator_release` | MATCH | 33-file pack list only (recorded or hashed from disk) · root-relative · `path\tsha256\n` |
| `artifact.v13.frozen_capsule` | MATCH (with `--orig`) | source-tracked names, quoted names skipped · dir-relative · CRLF→LF for text suffixes · rows joined by `\n` |

The other seven packs (`oracle_stage0_foundation`, `oracle_stage1_backends`,
`stage8_stage9_evaluation`, `rl_stage2_compute_parent`, `rl_stage3_e2_parent`,
`rl_stage4_bc_parent`, `rl_stage6_selector_eval`) also match with the pack
construction. Nothing in `frozen/` was modified.
