# Known discrepancies in frozen and provenance records

## Scope

This record lists recorded values in the kit that do not equal the value
recomputed from the bytes they describe. For each item it gives the recorded
value, the recomputed value, the command, the cause where it is confirmed by
the bytes or the history, and the impact. Nothing in `frozen/` was modified
to produce this record. Item 2 (`provenance/`, outside `frozen/`) was
corrected in the same change set; see section 9.

Revisions used: `main` `4a70816`, `ablation-20261102` `3cab744`, original
release commit `449c942c` (clone of
`frozen/original_release/source/REPRO_KIT_semantic_repair.bundle`, refs
fetched as described in `SCIENTIFIC_CODE_CHANGES.md`). Files are read from Git
objects (`git cat-file blob <rev>:<path>`, Git LFS pointers resolved with
`git lfs smudge` from the local store) unless stated otherwise. The
distribution zip is `frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip`
(member root `THESIS_REPRO_KIT_v2.1.1_FINAL/`). Helper scripts are in
`provenance/scripts/`; `<scratch>` is a working folder outside the repository.

"LF" means the bytes as stored in Git; "CRLF" means the same bytes with every
line ending converted to CRLF.

---

## 1. v1.3 capsule `SHA256SUMS.txt`: two entries

- Path: `frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN/SHA256SUMS.txt`
  (36 entries; 34 match the Git bytes).

| file | recorded | Git bytes (LF) | CRLF |
|---|---|---|---|
| `audit/V13_FINAL_REFREEZE_REPORT.md` | `d3d3b9817d0b…2372` | `d3f1c028e4ed…06ea` (1,163 B) | `50c55dd8c69e…57c` |
| `tests/test_v13_contract.py` | `9fe1aa24786b…1db0` | `84e3d8d9a566…0b65` (314 B) | `221c706e6953…ff3c` |

- Commands:
  ```
  git show 4a70816:<capsule>/SHA256SUMS.txt | grep -E "V13_FINAL_REFREEZE|test_v13_contract"
  git cat-file blob 4a70816:<capsule>/<file> | sha256sum
  python -c "<read zip member THESIS_REPRO_KIT_v2.1.1_FINAL/frozen_replay/v1.3/<file>; sha256; count b'\r\n'>"
  ```
- Cause (confirmed): the recorded values are the hashes of the copies inside
  the distribution zip (`frozen_replay/v1.3/...`), which have mixed line
  endings: the report is 1,174 B with CRLF on 11 of 13 lines (1–9, 12, 13);
  the test file is 316 B with CRLF on lines 10–11 only. Removing the CR bytes
  from the zip copies gives exactly the Git bytes. The Git blobs are the same
  in the kit and in the original release (`449c942c`, blobs `8b6fa649…` and
  `25751d4a…`), and the original release's `SHA256SUMS.txt` is byte-identical
  to the kit's. A uniform CRLF conversion does not reproduce the recorded
  values, because the endings were mixed.
- Impact: a check of the Git checkout against this list reports 2 mismatches.
  Content is identical apart from line endings. No result depends on it.
- `frozen/` is not modified.

## 2. `provenance/RESTORED_SCIENTIFIC_IMPLEMENTATIONS.json`

### 2a. `src/credit_recourse/contracts/v43_alpha_contract.py`

- Recorded `new_sha256`: `d9b943a7d35404907a636c904805bdc08156d37de31f947f779d189994a34066`.
- Actual (blob at `4a70816`): `5a6fa4cd1b5673b80de2ba818558428efcbdfcb341e39862b3eb7426fc582c79`.
- Commands:
  ```
  git log --format="%h %ad %s" --date=short 4a70816 -- src/credit_recourse/contracts/v43_alpha_contract.py
  for c in 4a70816 c82a29b f552b7d 5e28571; do git cat-file blob $c:src/credit_recourse/contracts/v43_alpha_contract.py | sha256sum; done
  git log --format="%h %ad %s" --date=short 4a70816 -- provenance/RESTORED_SCIENTIFIC_IMPLEMENTATIONS.json
  ```
  Results: `5e28571` → `adfcc25d…`, `f552b7d` (2026-09-19) → `d9b943a7…`,
  `c82a29b` (2026-09-20) and `4a70816` → `5a6fa4cd…`. The original
  (`449c942c`) is `09455d50…`, equal to the recorded `source_sha256`.
- Cause (confirmed): the record was last changed in `f552b7d`, the same commit
  that produced `d9b943a7…`. Commit `c82a29b` changed the file again and did
  not update the record. The other four `restored` entries match the current
  files.
- Impact: the record does not describe the current file. The scientific
  constants `EXPECTED_ALPHA_CONTRACT_SHA256` (`7ddc6ebe…`) and
  `EXPECTED_ALPHA_CONTENT_HASH` (`9c8d379e…`) are the same in `449c942c`,
  `f552b7d` and `4a70816`.

### 2b. `run_stage1_oracle_development.py` note "fresh branch stops after backend_gamma for this push"

- Recorded: `intentional_modifications` includes
  "fresh branch stops after backend_gamma for this push".
- Actual: at `4a70816`, `src/thesis_repro/stages/oracle.py` calls
  `stage1_main(["--project-root", ..., "--raw-rating-dir", ..., "--clean"])`
  with no `--end-step`; the default end step is the last entry of
  `STAGE1_STEP_ORDER`, `substrate_validation`.
- Commands:
  ```
  git show 4a70816:src/thesis_repro/stages/oracle.py | sed -n 345,356p
  git grep -n "end-step\|backend_gamma" f552b7d -- src/thesis_repro/stages/oracle.py
  git log --format="%h %s" 4a70816 -S"backend_gamma" -- src/thesis_repro
  ```
- Cause (confirmed): at `f552b7d`, `stages/oracle.py` line 48 passed
  `"--end-step", "backend_gamma", "--pending-ok"`. Commit `c82a29b` removed it.
  The record was not updated.
- Impact: the note describes behaviour that no longer exists.
- The file is outside `frozen/`; see item 9. `frozen/` is not modified.

## 3. Distribution zip: `configs/current/hash_manifest.json`

- Path: zip member `THESIS_REPRO_KIT_v2.1.1_FINAL/configs/current/hash_manifest.json`
  (34 entries; 25 match the zip members).
- Command: `python provenance/scripts/c2_zip_hash_manifest.py <zip> <orig_repo>`

| key | recorded | zip member (raw) | recorded value found in |
|---|---|---|---|
| information_contract | `80c7e965d5a2…` | `d32be9386fc8…` | CRLF form of an earlier blob (`7852ad7a5d`) in source commits e1a81a0…6c42fe4 (2026-09-14/15) |
| runtime_action_validation | `cae6161b4c5e…` | `f03984238d4f…` | earlier blob `b7f8d6ea62` (raw), same commits |
| runtime_contract | `abe6a92e06a8…` | `89adafca3ee9…` | not found in source history |
| runtime_doctor | `4dee5f467b39…` | `9b1a162c99dd…` | not found |
| runtime_executor | `c024260e897e…` | `1e38cfd13668…` | CRLF form of earlier blob `05fe2cbed9`, same commits |
| runtime_llm_contract | `b89f611ad71a…` | `ad07283125c0…` | not found |
| runtime_semantic_fixture | `2a4c72114b06…` | `235f8dc272ee…` | not found |
| stage8_evaluator | `61fa1a6bebf9…` | `8904feb0ed78…` | not found |
| stage9_revision_metrics | `98624db6a734…` | `c17c5d7b5d40…` | earlier blob `42e1d8ffd9` (raw), same commits |

  ("Same commits" = source commits `e1a81a0`, `66d5baa`, `3d69c6c`,
  `e168c0e`, `6c42fe4`.) Eight entries are `.py` files, one is a JSON config.
  For all nine, the zip member hash equals the value in
  `frozen/evidence/llm/design_release.json` `source_hashes`.
- Cause: confirmed for 4 entries: the recorded value is the hash of an earlier
  version of the same file in the source history, so the manifest was not
  regenerated after those files changed. Not confirmed for the other 5 (no
  matching blob, raw or CRLF, in the bundle history).
- Impact: no Python or PowerShell file in the zip or in the kit reads this
  file (`grep -rl hash_manifest` finds only JSON/MD/SHA256SUMS files). It is a
  stale record inside the zip.
- `frozen/` is not modified.

## 4. `CANONICAL_INPUT_MANIFEST.json`: two files not in the kit tree

- Path: `frozen/original_release/rl/stage2/runtime_inputs/business_plan_rate_v4/production/CANONICAL_INPUT_MANIFEST.json` (LFS).

| file | recorded | kit tree | distribution zip |
|---|---|---|---|
| `candidate_575_v4_coverage.csv` | `3256a7d9…a072` | absent (0 commits on any branch) | present, `3256a7d9…a072` |
| `production_lookup_preflight.json` | `c10105ce…0793` | absent (0 commits on any branch) | present, `c10105ce…0793` |
| `production_selected_rate_ledger.parquet` | `a8bf2561…06f8` | present, `a8bf2561…06f8` | present, same |

- Commands:
  ```
  git show 4a70816:<path> | git lfs smudge
  git log --all --oneline -- "*candidate_575_v4_coverage.csv" "*production_lookup_preflight.json"   # empty
  python -c "<sha256 of zip members .../archive/DEPLOYED_RELEASE/stage2_candidate_projection/runtime_inputs/business_plan_rate_v4/production/*>"
  ```
- Cause: not confirmed (why the two files were not copied is not recorded).
  Both are absent from every commit in the bundle history as well.
- Impact: the manifest cannot be fully checked from `frozen/original_release`
  alone; both files are available with the recorded hashes inside the
  distribution zip. `BusinessPlanRateResolverV4.from_frozen_artifacts`
  (`src/credit_recourse/simulator/business_plan_interest_rate_v4.py` line 153)
  requires `production_lookup_preflight.json` next to the production ledger
  under `archive/DEPLOYED_RELEASE/...` when that ledger is present.
- `frozen/` is not modified.

## 5. Distribution zip: fixed PASS outputs

### 5a. SmokeDemo markers

- Path: zip `scripts/v21_runner.py` (sha256 `502f4bb8…a7a1`), `smoke()` lines 387–426.
- Recorded output: a summary dict with `"overall": "PASS"` and eight markers
  (`PARSER_CONTRACT_SMOKE_PASS`, `SIMULATOR_COMPONENT_SMOKE_PASS`,
  `ORACLE_COMPONENT_SMOKE_PASS`, `RL_REFERENCE_ARTIFACT_SMOKE_PASS`,
  `STAGE8_COMPONENT_SMOKE_PASS`, `STAGE9_CONTRAST_SMOKE_PASS`,
  `SEMANTIC_INTEGRATION_SMOKE_PASS`, `SMOKE_DEMO_PASS`).
- Actual: the function runs two checks only, `stage2("smoke")` (line 390) and
  `run_semantic_fixture(ROOT)` (line 394). If both pass, it prints the
  summary dict (line 416, literal values) and the eight markers from a fixed
  tuple (lines 418–423). No marker has its own check.
- Command: `sed -n 387,426p <scratch>/zip/THESIS_REPRO_KIT_v2.1.1_FINAL/scripts/v21_runner.py`
- Impact: the eight markers carry no more information than the two checks.
- `frozen/` is not modified.

### 5b. `REVIEW_THESIS.ps1`

- Path: zip `scripts/REVIEW_THESIS.ps1` (sha256 `690e8674…b489`).
- Actual: after `verify`, `replay`, `analyze` and `build-outputs` each exit 0,
  lines 20–21 write the literal strings
  `{"status":"PASS","profile":"VERIFY"}` to `verification.json` and
  `{"status":"PASS","profile":"REPLAY"}` to `replay.json`, and line 23 prints
  `THESIS_REVIEW_PASS`. The JSON files do not contain any output of the
  commands.
- Command: `cat -n <scratch>/zip/THESIS_REPRO_KIT_v2.1.1_FINAL/scripts/REVIEW_THESIS.ps1`
- Impact: the two JSON files only record that the four exit codes were 0.
- `frozen/` is not modified.

## 6. Distribution zip `tests/`: pytest collects 0 tests

- Path: zip `tests/`, which contains one file,
  `tests/release/test_package_source_isolation.py` (sha256 `73a60d94…f540`);
  it defines `main()` and no `test_` function.
- Commands (zip extracted to `<scratch>/zip`; no network; Python
  `verify_20261025/venv312`, pytest 9.1.1):
  ```
  cd <scratch>/zip/THESIS_REPRO_KIT_v2.1.1_FINAL
  python -m pytest --collect-only -q -p no:cacheprovider tests
  # -> "no tests collected", exit code 5
  cd <scratch>/zip
  python -m pytest --collect-only -q -p no:cacheprovider
  # -> 1 test collected: THESIS_REPRO_KIT_v2.1.1_FINAL/frozen_replay/v1.3/tests/test_v13_contract.py::test_contract_files
  ```
- Actual: 0 tests in `tests/`. Over the whole extraction, the only collected
  test is the v1.3 capsule's own `test_v13_contract.py`.
- Impact: running pytest on the zip's `tests/` checks nothing; the file is a
  script to be run directly.
- `frozen/` is not modified.

## 7. Hashes recorded from CRLF bytes (ablation branch)

All on `3cab744`. Command:

```
python provenance/scripts/c2_line_ending_hashes.py <kit_abl> 3cab744 <path>=<recorded> ...
```

| source file (path under kit) | recorded | LF (Git) | CRLF | recorded equals |
|---|---|---|---|---|
| `frozen/evidence/v1.3/.../canonical_v13/V13_SUPPLEMENTAL_RESULTS.csv` | `62de26f9…c4e8` | `2680befc45…` | `62de26f921…` | CRLF |
| `.../frozen_inputs/stage8/V13_BASELINE_STRICT_PRIMARY_FULLSTREAM.csv` | `deca8883…8de8` | `b0d9ff0314…` | `deca8883eb…` | CRLF |
| `.../frozen_inputs/stage8/V13_HIGH_STRICT_PRIMARY_FULLSTREAM.csv` | `b150f813…85f5` | `c7c0827fb7…` | `b150f8134a…` | CRLF |
| `.../frozen_inputs/stage8/canonical_itt_observations.parquet` | `aedfadbf…8a0d` | `aedfadbf17…` | (binary) | Git bytes |
| `frozen/original_release/evaluation/.../stage9/HIGH_STRICT_ITT/llm_stage9_llm_rl_comparison.csv` | `6c1d94c2…d3b4` | `c76b34e582…` (CRLF→LF) | `6c1d94c2f8…` | Git bytes (the file is stored with CRLF in Git LFS) |
| `ablation_20261102/E1/observations/e1_observations.parquet` | `9afbdd17…364c` | `9afbdd177b…` | (binary) | Git bytes |

- `source_sha256` column in `ablation_20261102/results/R9_candidate_selection.csv`,
  `T1_primary_A1_A4.csv`, `T2_S1.csv`, `T3_REF1.csv`, `T5_E1.csv`: the three
  CSV sources above (R9: `V13_SUPPLEMENTAL_RESULTS.csv`; T1: both
  `*_PRIMARY_FULLSTREAM.csv`) are recorded as CRLF hashes; the parquet and the
  LFS CSV sources are recorded as Git-byte hashes.
- `ablation_20261102/results/_primary_stream_check.json` `sources`: the same
  three values (`aedfadbf…` Git bytes; `deca8883…`, `b150f813…` CRLF).
- QA Q12 combined frozen hash (`90_qa.py` `q12`: sha256 over
  `"<sha256>  <path>\n"` for `git ls-files -- frozen`, 1,101 files):
  ```
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout <kit_abl> <scratch>/q12_lf
  GIT_LFS_SKIP_SMUDGE=1 git -C <scratch>/q12_lf -c core.autocrlf=false checkout 3cab744 -- frozen
  (same into <scratch>/q12_crlf with core.autocrlf=true)
  python provenance/scripts/c2_q12_combined_hash.py <scratch>/q12_lf <scratch>/q12_crlf
  ```
  LF checkout: `5e12a3c803122690c76cd663e80a1b371209ab8888b9c541ef5820374cf7892b`.
  CRLF checkout: `4cfccfadac7c748c66d98d5992ca4f0d2ddb9745c5c179cb4993d6d48740e14b`
  (the value in `QA_REPORT.md` and `_qa_results.json`). 1,007 files are LFS
  (hashed by pointer oid = content sha256); 70 files differ between the two
  checkouts.
- Cause (confirmed): the recorded values were computed on a Windows checkout
  with `core.autocrlf=true`; recomputing on such a checkout reproduces them
  exactly, and recomputing on Git bytes does not.
- Impact: the recorded values cannot be reproduced on an LF checkout (Linux,
  macOS, CI, or Windows with `autocrlf=false`). The data are the same.
- `frozen/` is not modified (Q12 values are recorded under `ablation_20261102/`).

## 8. `frozen/evidence/llm/design_release.json` `source_hashes`: byte basis

- Path: `frozen/evidence/llm/design_release.json` (34 source hashes).
- Command:
  `python provenance/scripts/c2_design_release_hashes.py <kit> 4a70816 <zip> runs/<run-id>/rebuild_requests/source_hash_report.json`
- Result: all 34 recorded values equal the raw bytes of the corresponding
  members of the distribution zip (paths from the zip's
  `configs/current/hash_manifest.json`). By line endings of those bytes:

| basis | keys |
|---|---|
| CRLF (all lines) – 13 | action_contract, analysis_contract, c3e_definition, c6ex_manifest, experiment_matrix, model_contract, prompt_contract, retry_policy, runtime_collection, runtime_ledger, runtime_prompting, runtime_provider_adapters, runtime_retry_policy |
| LF – 9 | feature_dictionary, information_contract, runtime_action_validation, runtime_contract, runtime_executor, runtime_parser, stage8_evaluator, stage8_failure_enrichment, stage9_revision_metrics |
| mixed CRLF/LF – 7 | runtime_doctor, runtime_evaluation_bridge, runtime_failure_coder, runtime_llm_contract, runtime_semantic_fixture, runtime_v43_production_bundle, stage9_evaluator |
| binary – 5 | authority_document (.docx), c3e_actions, c3e_probabilities, c6ex_materialized, c6ex_permutation (.parquet) |

- Against the kit's Git bytes at `4a70816`:
  - CRLF-based keys present in the kit match only after LF→CRLF conversion,
    except `action_contract`, which is stored with CRLF in Git and matches
    as stored.
  - The LF-based `feature_dictionary`, `information_contract`,
    `runtime_parser` and `stage8_failure_enrichment` match as stored.
  - 12 runtime source files do not match. For `runtime_failure_coder`,
    `runtime_semantic_fixture` and `runtime_v43_production_bundle` the
    difference is line endings only (equal after CRLF→LF on both sides);
    the other 9 differ in content (see `SCIENTIFIC_CODE_CHANGES.md`).
  - `authority_document` is not in the kit; the C3-E and C6-EX files are
    read from the zip.
- This agrees with the `matched_bytes` column of
  `runs/<run-id>/rebuild_requests/source_hash_report.json`: 10 `crlf`;
  `checkout` for the rest of the matches; 12 `CONTENT_DIFFERS` (where this
  record further separates the 3 line-ending-only cases); `authority_document`
  `MISSING`.
- Cause (confirmed): the hashes were taken from the files as they are in the
  v2.1.1 distribution (mixed line endings per file), not from normalised
  bytes.
- Impact: `load_design` checks these hashes against files in the kit, so the
  check fails on the kit's Git bytes for the CRLF, mixed and changed files.
- `frozen/` is not modified.

## 9. Can the RESTORED record be corrected outside `frozen/`?

- File: `provenance/RESTORED_SCIENTIFIC_IMPLEMENTATIONS.json`, Git bytes
  sha256 `55c94213d93a4f19a50fe5a47250d5e3df1cf2b307b21c4c0060da4d145d4ddb`,
  CRLF form `202ffddbe92042a109b8b628b49af84ab724c734f3973ccf6cc6de205505d230`,
  blob id `086c3a4c68aa50cb3f3f5c6e71a240385df80d03`.
- Searches (file name `RESTORED_SCIENTIFIC`, both sha256 values, blob id):
  ```
  git grep -n -I -e RESTORED_SCIENTIFIC -e <LF> -e <CRLF> -e <blob> 4a70816      # no hits
  (same for 3cab744 and origin/hb-fy24-final)                                     # no hits
  git log --all -S<LF>; git log --all -S<CRLF>                                    # no hits
  grep -rlaE "RESTORED_SCIENTIFIC|<LF>|<CRLF>" frozen provenance docs *.md        # no hits (LFS content pulled)
  python: zip namelist and every zip member's bytes                               # no hits
  git -C <scratch>/orig grep -c RESTORED_SCIENTIFIC $(git rev-list --all)             # no hits
  ```
  `src/thesis_repro/original_release.py` hashes three other `provenance/`
  files (via `ORIGINAL_RELEASE_MANIFEST.json` records); this file is not
  among them.
- Result: no hash list, manifest, script or test refers to this file by name
  or hash, inside or outside `frozen/`, on any branch, in the zip, or in the
  bundle. The file is not under `frozen/`. It could therefore be corrected
  (entry 2a `new_sha256`, note 2b) without changing any recorded hash. It has
  since been corrected in the same change set (entry 2a `new_sha256`,
  notes 2b; the previous value is kept as `recorded_new_sha256_before_c82a29b`).

## 10. Post-freeze edits in `frozen/`

- List: `provenance/POST_FREEZE_EDITS.csv` (path, previous and new sha256,
  change type). Inside `frozen/` it covers two files of the v1.3 capsule:
  `SHA256SUMS.txt` and the file renamed to `REPRO_KIT_V13_NOTES.md` (its
  previous path is the `old_relative_path` of its `MIGRATION_MANIFEST.json`
  entry).
- Effects on recorded values:
  - The capsule's `SHA256SUMS.txt` lists the new name and sha256, so the
    capsule's own listing still checks against the kit files (apart from the
    pre-existing entries of section 1 and section 7).
  - The capsule digest `e8876488…` (`frozen/release/frozen_manifest.json` and
    the frozen submission release) is unchanged. It reproduces from the source
    repository's blobs, not from the kit copy;
    `scripts/recompute_directory_digests.py --orig` prints the two files above
    as the only differences (`provenance/DIRECTORY_DIGESTS.md`).
  - The distribution zip is unchanged. Its `frozen_replay/v1.3/` copy and its
    `SHA256SUMS.txt`/`RELEASE_MANIFEST.json` still describe the capsule as
    frozen, so the zip copy and the kit capsule differ in these two files.
- Impact: none on results. Neither file is read by any verification or
  reproduction command.
