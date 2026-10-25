# Provenance

## Historical source identity

The product is a clean-room reproduction repository for
`CallMeDemian/REPRO_KIT_semantic_repair`. Source commits, the scientific
evidence base, release identity, and the migrated asset inventory are recorded
in `provenance/source_release.json` and the immutable release manifests.

## Migration history

At the initial migration step, some source Git objects were unavailable and
were recorded as migration boundaries. The current `source_release.json`
records the relevant source, evidence-base, and certification objects as
`VERIFIED`; the historical statement is retained here to describe the
migration context, not the current verification state.

## Current closure state

- Historical evidence closure: **CLOSED**.
- Fresh implementation/code closure: **CLOSED WITH EXTERNAL INPUT BOUNDARIES**.
- Published-result reproduction: **CLOSED**.
- Synthetic full-DAG architecture acceptance: **PASS**, execution class
  `SYNTHETIC_E2E_ACCEPTANCE`, `completion_state=PASS`, `dag_complete=true`,
  `certifiable=false`.
- Full fresh numerical execution: **NOT CLAIMED AS COMPLETED**. It requires
  authorized raw inputs, the unrecovered original Stage2 producer-input pack,
  compute, and/or provider credentials as described in the professor guide.

Synthetic acceptance invokes the canonical FullClean DAG, run engine, and
canonical adapters on deterministic fixture inputs. It proves architecture
and lineage, not thesis-scale fresh scientific replication.

## Changes to the scientific code since the original release

`provenance/SCIENTIFIC_CODE_CHANGES.csv` lists the 57 files under
`src/credit_recourse` whose bytes differ from the original release commit
`449c942c` without being recorded elsewhere (original and current sha256,
changed lines, category, kit commits). `provenance/SCIENTIFIC_CODE_CHANGES.md`
summarises the semantic changes with before/after snippets. The kit commits
state no reasons; the record says so instead of inferring one.

## Known discrepancies in frozen records

`provenance/KNOWN_FROZEN_DISCREPANCIES.md` lists recorded values in `frozen/`
and `provenance/` that differ from the value recomputed from the bytes they
describe, with the command, the confirmed cause and the impact. `frozen/` is
not modified.

## Directory digests

The pack digests in `frozen/original_release/ORIGINAL_RELEASE_MANIFEST.json`
are `sha256` of the byte-order-sorted rows `"<root-relative path>\t<sha256>"`,
one per file of the pack in `provenance/ASSET_PACK_MANIFEST.json`, each row
ending with a newline. The v1.3 capsule digest in
`frozen/release/frozen_manifest.json` follows `tree_sha256()` of the source
release (Git file order, CRLF->LF for text suffixes, rows
`"<dir-relative path>\t<sha256>"` joined by newlines without a final newline).
`scripts/recompute_directory_digests.py` recomputes all five; details and
results are in `provenance/DIRECTORY_DIGESTS.md`.
