from __future__ import annotations

import runpy
from pathlib import Path
from credit_recourse.utils.io_contract import configure_utf8_stdio


def main() -> int:
    configure_utf8_stdio()
    # Active fresh producer uses the corrected 10-grade/runtime artifact
    # contract.  Keep _pipeline_impl.py as a historical reference only.
    impl = Path(__file__).with_name("_pipeline_impl_runtime.py")
    runpy.run_path(str(impl), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
