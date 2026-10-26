# Post-freeze edits

`POST_FREEZE_EDITS.csv` lists the files changed after the freeze of the
Human baseline final folder (`human_baseline_final_20261019/`), where the
change is limited to wording, a file name, or a file removal. One row per file:

| column | meaning |
|---|---|
| `path` | current path in the kit (for a renamed file, the new path) |
| `previous_sha256` | sha256 of the bytes before the edit, as stored in Git |
| `new_sha256` | sha256 of the bytes after the edit, as stored in Git |
| `change_type` | `문구 수정(수치·판정 불변)`, `파일 삭제` or `이름 변경` |
| `date` | date of the edit |

No number, verdict, method or recorded decision changed. The exact
differences can be traced in the Git history of each path.

Records updated with these edits: `human_baseline_final_20261019/SHA256SUMS.txt`.
The same two files carry the same edits on the `ablation-20261102` branch.
