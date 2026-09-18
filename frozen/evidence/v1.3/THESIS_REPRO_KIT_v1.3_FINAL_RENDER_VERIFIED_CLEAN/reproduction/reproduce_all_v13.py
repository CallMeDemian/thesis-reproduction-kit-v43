from pathlib import Path
import sys
import pandas as pd
import numpy as np
from v13_runtime import run
if __name__ == '__main__':
    raise SystemExit(run(Path(__file__).resolve().parents[1]))
