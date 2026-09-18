from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

from credit_recourse.oracle.backends.alpha.production_finalize import finalize_alpha_backend


def main() -> int:
    # Fresh producer is singular and explicit. The legacy implementation file
    # remains only for historical artifact archaeology. Contract finalization
    # is part of this producer so a clean Stage1 run cannot emit the legacy
    # sparse-bin contract or score rows with a different contract.
    impl = Path(__file__).with_name("_pipeline_impl_runtime.py")
    runpy.run_path(str(impl), run_name="__main__")
    default_output = (
        Path(__file__).resolve().parents[5]
        / "data" / "final_freeze" / "stage1_oracle_backends" / "alpha"
    )
    output_dir = Path(os.environ.get("ORACLE_OUTPUT_DIR", default_output))
    result = finalize_alpha_backend(output_dir)
    print("[Alpha V4.3 production finalization]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
