from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
POLL = E2 / "scripts/poll_main_jobs.py"
STATE = E2 / "logs_main/provider_jobs_main.json"
LOG = E2 / "logs_main/watch_main_jobs.log"


def write(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def main() -> None:
    while True:
        result = subprocess.run([sys.executable, str(POLL)], capture_output=True, text=True)
        if result.returncode != 0:
            write(f"poll_error exit={result.returncode} stderr={result.stderr[-1000:]}")
        else:
            data = json.loads(STATE.read_text(encoding="utf-8"))
            summary = data.get("last_poll_summary", {})
            jobs = summary.get("jobs", [])
            terminal = sum(bool(x.get("terminal")) for x in jobs)
            counts = []
            for job in jobs:
                if job.get("provider") == "openai" and "-openai-Q-" in str(job.get("name", "")) and not job.get("terminal"):
                    c = job.get("counts") or {}
                    counts.append(f"{job.get('name')}={c.get('completed')}/{c.get('total')}")
            write(f"poll_complete terminal={terminal}/{len(jobs)} pending_q={','.join(counts) or '-'}")
            if len(jobs) == 24 and terminal == 24:
                failures = [x.get("name") for x in jobs if x.get("terminal_failure") or (x.get("counts") or {}).get("failed", 0)]
                write(f"all_terminal failures={failures}")
                break
        time.sleep(60)


if __name__ == "__main__":
    main()
