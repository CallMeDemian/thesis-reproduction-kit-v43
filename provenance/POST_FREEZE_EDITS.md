# Post-freeze edits

`POST_FREEZE_EDITS.csv` lists the files changed after the freeze of the
original release (`449c942c`) and of the kit's frozen records, where the change
is limited to wording, a file name, or a file removal. One row per file:

| column | meaning |
|---|---|
| `path` | current path in the kit (for a renamed file, the new path) |
| `previous_sha256` | sha256 of the bytes before the edit, as stored in Git |
| `new_sha256` | sha256 of the bytes after the edit, as stored in Git |
| `change_type` | `문구 수정(수치·판정 불변)`, `파일 삭제` or `이름 변경` |
| `date` | date of the edit |

No number, verdict, method or recorded decision changed. The exact
differences can be traced in the Git history of each path.

Records updated with these edits: the v1.3 capsule's `SHA256SUMS.txt`,
`MIGRATION_MANIFEST.json` (`new_*` fields of the edited entries; the previous
path of a renamed file stays in its `old_relative_path`),
`SCIENTIFIC_CODE_CHANGES.csv/.md`, `DIRECTORY_DIGESTS.md` and
`KNOWN_FROZEN_DISCREPANCIES.md` (section 10).
