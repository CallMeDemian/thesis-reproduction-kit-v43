# QA_REPORT (ablation_20261102)

File status: DRAFT

## Setup (Q01)

| Item | Value |
| --- | --- |
| Base commit | 071848881349c8675bb4e15193a747fa02679986 (origin/hb-fy24-final) |
| Branch | ablation-20261102 |
| HEAD at start of work | 071848881349c8675bb4e15193a747fa02679986 |
| Python | Python 3.12.10 |
| Packages | google-genai==2.31.0, joblib==1.5.3, matplotlib==3.11.0, numpy==2.4.4, openai==3.35.1, openpyxl==3.1.5, pandas==3.0.2, pyarrow==24.0.0, pytest==9.1.1, python-docx==1.2.0, python-pptx==1.0.2, PyYAML==6.0.3, scikit-learn==1.8.0, scipy==1.17.1, statsmodels==0.14.6 |
| LFS state | Cloned with GIT_LFS_SKIP_SMUDGE=1; E1 inputs and stage9 HIGH_STRICT_ITT pulled; other LFS objects not yet fetched |
| thesis_repro verify | PASS (ablation_20261102/E1/verify_output.txt) |
| Deck SHA-256 | 2667921758789af260215270c99344e3c955a9f38a8f6843405f6881d0f79e4b (matches) |
| Thesis SHA-256 | f6257868c52534fc6477ac249ed463731402553de2cf0786b9667df76b0073a1 (matches) |
| git status --porcelain -- frozen | empty |

## E1 pre-registration gates (Q16, Q17)

| Check | Result | Evidence |
| --- | --- | --- |
| Request build | PASS | E1/requests_build_stdout.txt, E1/requests/E1_build_report.json |
| Manifest SHA-256 | 1f12775443db3a929cddf7a619a27882ad49fe5b189da23572f908854064357a, equal to expected; byte-identical | fc /b against the expected manifest |
| Build report vs expected | equal in checks, counts, manifest and shard hashes; matched_as reads crlf (expected) vs raw on this machine | E1/requests/E1_build_report.json |
| e1_analyze.py --selftest | PASS | E1/selftest_output.txt |
