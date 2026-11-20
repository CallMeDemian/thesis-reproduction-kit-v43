from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


E2 = Path(__file__).resolve().parents[1]
TASK = E2.parent
WORKSPACE = TASK.parent
OUT = E2 / "E2_FINAL_ARTIFACT_MANIFEST.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(WORKSPACE))


def main() -> None:
    paths: set[Path] = set()
    explicit = [
        E2 / "E2_FINAL_REPORT.md", E2 / "E2_PAPER_SLIDES_BRIEF.md",
        E2 / "E2_MAIN_REUSE_PROTOCOL.md", E2 / "E2_V5_1_PROTOCOL.md",
        E2 / "logs_main/main_response_hashes.json", E2 / "logs_main/main_response_integrity.json",
        E2 / "logs_main/main_analysis_metadata.json", E2 / "logs_main/provider_jobs_main.json",
        E2 / "logs_main/pre_call_hashes_main.csv", E2 / "logs_main/analysis_script_hashes_pre_scoring.csv",
        E2 / "logs_main/r8_derangement_freeze.json", E2 / "manifests_main/e2_main_execution_manifest.csv",
        E2 / "manifests_main/r8_derangements_seed20261118.csv",
        E2 / "manifests_main/score_action_manifest.csv",
        TASK / "B1_diagnostics/b1_run.py", TASK / "B1_diagnostics/b1_finalize.py",
        TASK / "B1_diagnostics/b1_artifact_sha256.json", TASK / "B1_diagnostics/b1_run_metadata.json",
        TASK / "B1_diagnostics/b1_p5_selected_actions.csv", TASK / "B1_diagnostics/e2_simulator_10row_tables.parquet",
        TASK / "A_reproduction/R2_R8_replay_workspace/logs/replay/r6_restore_manifest.json",
        TASK / "A_reproduction/R2_R8_replay_workspace/logs/replay/e18_an_restore_20261119.txt",
        TASK / "A_reproduction/R2_R8_replay_workspace/logs/replay/e20_restore_20261119.txt",
    ]
    for path in explicit:
        if path.is_file():
            paths.add(path)
    for folder, pattern in [
        (E2 / "scripts", "*.py"), (E2 / "requests_main", "*.jsonl"),
        (E2 / "requests", "*.jsonl"), (E2 / "responses", "*.jsonl"),
        (E2 / "responses_main", "*.jsonl"), (E2 / "results", "*"),
        (E2 / "logs_main", "*"), (E2 / "manifests_main", "*"),
    ]:
        if folder.exists():
            paths.update(p for p in folder.glob(pattern) if p.is_file())

    rows = [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(paths, key=rel)]
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"manifest": rel(OUT), "files": len(rows), "sha256": sha256(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
