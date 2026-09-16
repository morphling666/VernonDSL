from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from vernon_dsl._versions import COMPILER_CONTRACT_VERSION, PROGRAM_VERSION, RELEASE_VERSION


def test_generated_versions_are_current() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, str(root / "tools/generate_versions.py"), "--check"],
        cwd=root,
        check=True,
    )


def test_public_version_axes_are_valid() -> None:
    assert RELEASE_VERSION
    assert COMPILER_CONTRACT_VERSION > 0
    assert PROGRAM_VERSION > 0
