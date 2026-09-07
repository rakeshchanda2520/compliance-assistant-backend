"""Every suite, one command: `python tests/run_all.py`."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITES = ["test_understanding.py", "test_pipeline.py"]

failed = []
for suite in SUITES:
    print(f"\n{'=' * 70}\n{suite}\n{'=' * 70}")
    result = subprocess.run([sys.executable, str(HERE / suite)],
                            cwd=HERE.parent)
    if result.returncode:
        failed.append(suite)

print(f"\n{'=' * 70}")
print("FAILED: " + ", ".join(failed) if failed else "all suites passed")
sys.exit(1 if failed else 0)
