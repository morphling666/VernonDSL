from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_optimized(script: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": "python", "PYTHONOPTIMIZE": "1"}
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class ProductionChecksUnderOptimizationTests(unittest.TestCase):
    def test_optimization_is_active_in_subprocess(self) -> None:
        result = _run_optimized("import sys; sys.exit(0 if not __debug__ else 2)")
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
