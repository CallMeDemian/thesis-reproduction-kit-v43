from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    # Active fresh producer uses the corrected runtime implementation.  The
    # legacy file remains in place only for historical artifact archaeology.
    impl = Path(__file__).with_name("_pipeline_impl_runtime.py")
    runpy.run_path(str(impl), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
