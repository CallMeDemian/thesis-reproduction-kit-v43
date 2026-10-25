# Scientific code changes not recorded elsewhere

## Scope

This record covers the 57 files under `src/credit_recourse/` whose bytes at kit
commit `4a70816` (branch `main`; `ablation` `3cab744` has identical `src/`)
differ from the original release commit
`449c942cd1c02b17bcc1720f10ee8c96ee0a883a` and whose difference is not listed in
`provenance/MIGRATION_MANIFEST.json` (CURATED entries) or
`provenance/RESTORED_SCIENTIFIC_IMPLEMENTATIONS.json`. The file list is the set
classed `UNRECORDED_DIFFERENCE` in the full-verification lineage table
(`code_lineage.csv`, 313 rows). Every value below was recomputed from Git
objects; no working-tree file was used.

The per-file table is `SCIENTIFIC_CODE_CHANGES.csv` (one row per file).

## How the values were computed

Original source (the bundle's heads are under `refs/remotes/origin/final/*`, so
a plain `git clone <bundle>` gives an empty checkout; the refs were fetched
explicitly):

```
git clone --no-checkout frozen/original_release/source/REPRO_KIT_semantic_repair.bundle <scratch>/orig
git -C <scratch>/orig fetch <bundle> "refs/*:refs/remotes/bundle/*"
git -C <scratch>/orig cat-file -t 449c942c        # -> commit
```

Per file (`provenance/scripts/c1_collect.py`, read-only on both repositories):

```
git -C <scratch>/orig cat-file blob 449c942c:<path>  > NN_orig    # original_sha256 = sha256(bytes)
git -C kit        cat-file blob 4a70816:<path>   > NN_kit     # current_sha256  = sha256(bytes)
git diff --no-index --numstat NN_orig NN_kit                  # lines_added, lines_removed
git -C kit log --format=%h%x09%ad%x09%s --date=short 4a70816 -- <path>   # changed_in_commits
git -C kit show -s --format=%B <commit>                       # full commit message
```

`scripts/c1_write_csv.py` writes the CSV. Cross-check: all 57
`original_sha256` and `current_sha256` values equal the `orig449_sha256` and
`current_sha256` columns of `code_lineage.csv`; total lines changed 1,731.

## Commit messages

The 57 files were touched by 14 kit commits. Every one of them has a subject
line only and an empty body:

| commit | date | subject |
|---|---|---|
| af388eb | 2026-09-19 | chore: initialize V4.3 thesis reproduction product |
| 5e28571 | 2026-09-19 | feat(repro): close fresh v4.3 scientific engine adapters |
| 10a0b86 | 2026-09-19 | fix(oracle): bind fresh stage1 paths to raw contracts |
| 213f00e | 2026-09-20 | fix(oracle): close fresh Stage1 acceptance semantics |
| bf59cd1 | 2026-09-20 | fix(engine): make fresh DAG fail closed |
| c82a29b | 2026-09-20 | feat(oracle): close fresh production engine |
| 94a42e6 | 2026-09-20 | feat(repro): add certified published reproduction lane |
| f45c2c4 | 2026-09-20 | chore(repro): checkpoint certified release and final runtime |
| 129c1ab | 2026-09-21 | feat(repro): complete executable reproduction closure |
| 63e03d3 | 2026-09-21 | feat: close forensic reproduction release |
| b59d2e3 | 2026-09-21 | fix: close final forensic runtime boundaries |
| 1718d84 | 2026-09-21 | fix: close industry binding and generation completion gates |
| 278de06 | 2026-09-21 | fix(provenance): close industry and bundled C6EX resolution |
| c5a579f | 2026-09-21 | fix: close live llm and fresh c6ex contracts |

None of these messages states a reason for a change. `reason_recorded` is
therefore `사유 기록 없음` for all 57 rows. No reason has been inferred.

## Categories

- `path_rebinding`: the change only alters where a file or directory is
  resolved (hard-coded `archive/DEPLOYED_RELEASE/...`, `data/final_freeze/...`
  or `configs/current/...` replaced by run-local, environment-selected or
  call-time paths, plus the imports and labels that go with it). What is read,
  computed and checked is otherwise unchanged.
- `semantic_change`: the change alters computation, an acceptance or
  validation rule, a default (for example a seed), an identifier, an output
  field or status value, or adds/removes an execution mode or function.
- `other`: no behaviour change (docstring, blank lines, help text, UTF-8
  console setup).

A file with both path and behaviour changes is `semantic_change`.

| category | files |
|---|---|
| path_rebinding | 30 |
| semantic_change | 20 |
| other | 7 |

## Diff summaries for `semantic_change` files

Paths are relative to `src/credit_recourse/`. "Before" is `449c942c`, "after"
is `4a70816`.

### `rl/common/actions.py` (+14 −12; 129c1ab)

- The no-op candidate id changes from `A0_noop` to `A0` in all three lookups.
  Before: `noop_idx = cand_names.index("A0_noop") if "A0_noop" in cand_names else 0`.
  After: `noop_idx = cand_names.index("A0") if "A0" in cand_names else 0`.
- Action and candidate sources change from two YAML files under
  `configs/current/final_freeze/` (`final_candidate_library.yaml`,
  `final_action_contract.yaml`) to one JSON file,
  `contracts/scientific/v43_action_contract.json`, used for both. As a result
  `candidate_library_hash` and `final_action_contract_hash` are now the same
  file hash.
- Observation (not a reason): in `449c942c`, `rl/contracts/v43_encoder.py`
  already defines `ACTION_IDS = ("A0", "DL", ...)`.

### `rl/common/candidate_intensity_contract.py` (+5 −5; 129c1ab)

- Same rename: keys of `TARGET_INTENSITY_BY_CANDIDATE` and
  `FAMILY_BY_CANDIDATE`, and the zero-intensity check, use `"A0"` instead of
  `"A0_noop"`.

### `rl/v43_runtime.py` (+12 −10; 129c1ab, f45c2c4)

- Stage6 evaluation rows: before, read from
  `archive/DEPLOYED_RELEASE/stage2_candidate_projection/phase_eval_candidate.parquet`
  with `if len(rows) != 575: raise ValueError("Frozen Stage6 sample changed")`.
  After, read from `<run>/02_data/stage6_rows.parquet` with
  `if rows.empty: raise ValueError("Fresh Stage6 cohort is empty")`. The fixed
  575-row check is gone.
- Checkpoint check: before, the path had to equal one fixed
  `stage_dir(...)/final_epoch.pt`. After, any `final_epoch.pt` inside the
  active run root is accepted.
- Added: the run's `execution.json` `config_hash` must equal the frozen config
  hash.

### `rl/v43_one_pass_contract.py` (+41 −15; 129c1ab, f45c2c4)

- Seed: before `'seed':2`. After, the seed is read from the canonical config's
  stage entries (`stage_seeds = {name: int(value['seed']) ...}`), which must all
  be equal, and `'seed':selected_seed` is written.
- Config source: before `configs/current/final_freeze/v43_rl.json`. After
  `THESIS_REPRO_RL_CONFIG_PATH`, else
  `contracts/scientific/final_freeze/v43_rl.json`, else the old path.
- Checkpoint check in `begin_training`: before, the destination had to equal a
  fixed path. After, `destination.name == "final_epoch.pt"` and inside the
  active run root.

### `contracts/v43_action_contract.py` (+11 −52; 5e28571)

- Before: `load_v43_action_contract` checked the file sha256
  (`f722e189…`), the content hash (`ffa9a065…`), the semantic version string and
  `validate_timing_contract`, and returned `(contract, digest)`.
- After: `load_action_contract` only checks the 9 action ids, that the action
  columns are exactly the 8 `action__*` dimensions, and that no column contains
  `revenue_growth`; it returns the payload. Contract path
  `contracts/scientific/v43_action_contract.json`.

### `eval/final_stage8_llm_multi_oracle_eval/pipeline.py` (+63 −41; 129c1ab, 63e03d3)

- Adds `fixture_mode`. When set: fixed synthetic borrowing rate 0.04 and
  cost 1% of A0 revenue; firm count taken from the data
  (`expected_firms = 575 if not fixture_mode else len(frame) // 9`); target
  year 2025 and registry status checks skipped.
- Always: Stage6 payoff surface must also contain `fiscal_year`; input
  columns not produced by the simulator are copied into each scored row;
  `generation_regime` added to the output key columns; an incoming `row_id`
  column is dropped before indexing.

### `eval/final_stage9_llm_rl_comparison/pipeline.py` (+27 −9; 129c1ab, 63e03d3, b59d2e3)

- Adds `fixture_mode` (firm count from data instead of 575).
- Adds checks: one firm/year per `row_id`, exactly the 9 actions per
  `row_id`, and deployed C3-E identity equal to the Stage6 surface identity.
  The deployed table may also be given as a direct `.parquet` path.

### `eval/final_stage9_llm_rl_comparison/revision_metrics.py` (+12 −4; 129c1ab)

- `cell_id`, `generation_regime` and `phase` added to the grouping/key column
  lists of the revision, identity-contrast and primary-contrast tables.

### `eval/v43_financial_inputs.py` (+9 −56; 5e28571)

- Before: `_row_to_firm_state` built aliases for `raw__`/`sim__`/`avs__`
  prefixed columns, called `load_firm_state_from_registry`, and set
  `rating_num`/`rating_grade`.
- After: it calls `load_firm_state_from_columns(values, firm_id, year, sector)`
  directly; no alias expansion and no rating fields.

### `final_release/contract.py` (+40 −23; f45c2c4, 129c1ab, 63e03d3, 1718d84)

- Config root: `configs/current/llm` → `THESIS_REPRO_LLM_CONFIG_ROOT` or
  `frozen/evidence/llm`; released-source hash paths resolved from there.
- Industry binding acceptance. Before:
  `evidence.get("frozen") and int(evidence.get("unresolved_count", 575)) == 0`.
  After: (`frozen is True` or role `FROZEN_EXOGENOUS_INFORMATION_INPUT`, or a
  configured root with role `FRESH_EXOGENOUS_INFORMATION_INPUT`) and
  `status == "PASS"` and `unresolved_count == 0`.

### `final_release/doctor.py` (+27 −33; f45c2c4, b59d2e3)

- `_golden_check` (comparison with `repro/expected/final_c3e_golden.json`)
  removed, and the per-key registry hash checks removed.
- LLM contract, provider adapter, parser, firm cohort and semantic
  end-to-end checks run only when `THESIS_REPRO_LLM_CONFIG_ROOT` is set;
  otherwise they are reported `INPUT_REQUIRED`/`NOT_APPLICABLE` and a blocker
  `FRESH_RUN_LOCAL_PLAN3_CONFIG_REQUIRED` is added.
- A non-PASS legacy assertion is no longer a blocker.

### `final_release/llm_contract.py` (+178 −19; f45c2c4 … 278de06)

- New `build_c6ex_materialization()` builds C6-EX from the donor permutation
  and fresh C3-E labels.
- `validate_llm_contract` adds checks: manifest fields and
  `schema_version == "v43_c6ex_fresh_materialization_v1"`, unique row/donor
  ids, donor coverage, recomputed action-label collisions, preserved action
  distribution.
- `industry_binding_status` readiness now also requires `status == "PASS"`,
  declared rows/unique firms 575, and valid (non-`nan`/`none`/`unknown`)
  industry codes; the API-key leak scan is reported
  `NOT_APPLICABLE_FRESH_RUNTIME` when a fresh config root is set.

### `oracle/backends/alpha/production_finalize.py` (+14 −2; 213f00e)

- Under `oracle_execution_profile() == "synthetic"`, `fit_source` is rewritten
  and the contract hash recomputed. The checks against
  `EXPECTED_ALPHA_CONTRACT_SHA256` and `EXPECTED_ALPHA_CONTENT_HASH` now run
  only when the profile is `production`.

### `oracle/verification/verify_alpha_contract.py` (+48 −30; c82a29b, 213f00e)

- Paths come from functions (`canonical_alpha_contract_path()` etc.) instead
  of module constants.
- In a non-production profile the fixed-hash and fixed-variable-order checks
  are replaced by self-consistency checks (file present, content hash equals
  `contract_hash(contract)`, variables non-empty). Production checks unchanged.

### `oracle/verification/verify_stage1_substrate_validation.py` (+13 −5; c82a29b, bf59cd1)

- Output adds `scientific_gate_verdict`, `verdict_basis`,
  `verification_execution_status`. Failure status value changes from
  `"FAIL"` to `"FAILED"`.

### `oracle/stage1/stage00_02_ratio_engineering/modules/_task6_compute_ratios.py` (+27 −1; c82a29b)

- When the plain account column is missing, values are taken from the merge
  suffixes (`<code>_x` for current, `<code>_y` for lag).
- When a ratio term resolves to nothing, it is resolved again from
  `numerator_item_code`/`denominator_item_code` (`resolve_item_code`).

### `oracle/stage1/stage00_04_variable_selection/pipeline.py` (+15 −3; c82a29b)

- Spearman screening: inputs coerced to float arrays; returns NaN/flag when
  fewer than 10 rows or either input is constant
  (`np.unique(x).size < 2 or np.unique(y).size < 2`). Before only `len < 10`.
- Report table: `to_markdown` → `to_string`.

### `repro/runtime_helpers/build_bp_rate_v4_production_lookup.py` (+4 −164; 5e28571)

- The original producer (as-of rate selection with PAVA spread estimate,
  ceiling caps, Test3 bitwise-equality check, writing
  `production_selected_rate_ledger.parquet`, `candidate_575_v4_coverage.csv`,
  `production_lookup_preflight.json`) is absent. Two helpers remain:
  `key(firm_id, year)` and `select_missing(rows, lookup)`.

### `analysis/final_v43/runner.py` (+8 −19; 94a42e6, 63e03d3)

- `--profile` choices `["MANUSCRIPT_FROZEN", "CLEAN_REEXECUTION"]` →
  `["MANUSCRIPT_FROZEN"]`; the `CLEAN_REEXECUTION` branch that read
  `runs/<id>/stage9/*` is replaced by
  `raise ValueError("published-result analysis accepts only MANUSCRIPT_FROZEN ...")`.
- `source` labels now name `frozen/evidence/v1.3/...`.

### `analysis/final_v43/selection.py` (+4 −21; 94a42e6, 63e03d3)

- The `CLEAN_REEXECUTION` source branch (RUN_INFO checks, Stage8 parquet
  lookup) is removed; only `MANUSCRIPT_FROZEN` is accepted.

## Notes on selected `path_rebinding` and `other` rows

- `oracle/verification/verify_stage00_04_growth_eligibility_contract.py`: the
  original refers to an undefined name `stage00_04_dir` in the
  empty-growth-selection branch; the kit uses `stage4`.
- `eval/v43_stage6_reporting.py`: the beta/gamma parameter paths are fixed
  run-local paths instead of the registry entries, and the artifact paths
  written into the evaluation contract become absolute strings.
- `rl/v43_one_pass_data.py`: `input_root()` has no default; it raises unless
  `THESIS_REPRO_STAGE2_INPUT_ROOT` is set.
- `contracts/stage_paths.py`: `as_manifest()` and the module CLI are removed;
  stage keys map to numbered run directories.

## Effect on published results

The full verification records show the published results unchanged with this
code: receipts 96/722/96 and registry 914 rows (Stage9 frozen-replay and
reproduce PASS), Stage8 96,600 rows, `verify-original` 1,035/1,035 artifacts,
and `rebuild-c3e` 575/575 actions. The commits that add this record were checked with a
before/after regression of the same commands (pytest, verify, frozen-replay, reproduce,
verify-original, rebuild-c3e, chapter 4 rechecks, ablation run_all, E1 chain); no result
value changed.

## Edits after `4a70816`

Three rows of the CSV reflect a later edit of string literals and comments
(`provenance/POST_FREEZE_EDITS.csv`), so the CSV now has 59 rows:

- `rl/v43_support.py`: one string value in the support metadata; the row's
  `current_sha256` and line counts are those of the edited file (18 lines at
  `4a70816`, now 20).
- `rl/v43_training_math.py` and
  `oracle/stage1/stage00_04_variable_selection/README_PATCH.md`: identical to
  `449c942c` at `4a70816`; one comment or heading line each, category `other`.

All three are 주석·메타데이터 문자열만 변경. No computation, check or numeric output
depends on the edited text; the `v43_support.py` string is written only as the
`simulator_change` label of a fresh run's `action_support_metadata.json`.
