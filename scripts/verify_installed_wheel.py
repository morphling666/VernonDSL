from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import vernon_dsl as vd
from vernon_dsl._versions import RELEASE_VERSION
from vernon_dsl.runtime_source import cmake_source_dir, version
from wheel_smoke_kernel import add_one


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an installed VernonDSL wheel.")
    parser.add_argument(
        "--skip-package-layout",
        action="store_true",
        help="skip bundled-runtime checks when exercising the script from a source checkout",
    )
    arguments = parser.parse_args()
    if not arguments.skip_package_layout:
        assert cmake_source_dir().is_dir()
        assert version() == RELEASE_VERSION

    vd.init(arch=vd.cpu)
    expected = np.arange(16, dtype=np.float32) + np.float32(1.0)
    output = vd.storage.zeros(dtype=vd.f32, shape=expected.shape)
    add_one(
        output,
        vd.storage.from_numpy(expected - np.float32(1.0)),
        grid=(expected.size, 1, 1),
    )
    np.testing.assert_array_equal(output.to_numpy(), expected)
    with tempfile.TemporaryDirectory(prefix="vernon-wheel-") as directory:
        root = Path(directory)
        source = root / "smoke_shader.py"
        result = root / "smoke.mlir"
        source.write_text(
            "from vernon_dsl import *\n@fragment\ndef main(value: f32) -> f32:\n    return value + 1.0\n",
            encoding="utf-8",
        )
        subprocess.run(
            [sys.executable, "-m", "vernon_dsl.cli", str(source), "-o", str(result)],
            check=True,
        )
        assert result.stat().st_size > 0

        asset_source = root / "smoke_asset.py"
        asset_source.write_text(
            "from typing import Annotated\n"
            "import vernon_dsl as vd\n"
            "@vd.kernel(workgroup_size=(1, 1, 1))\n"
            "def fill(output: vd.TensorView[vd.f32, (vd.dyn,), vd.write], "
            "gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin('global_invocation_id')]) -> None:\n"
            "    output[gid[0]] = 1.0\n"
            "asset = vd.program_asset(id='release/smoke', program=fill)\n",
            encoding="utf-8",
        )
        cooked = root / "cooked"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "vernon_dsl.program_asset_cli",
                f"{asset_source}:asset",
                "--target",
                "cpu",
                "-o",
                str(cooked),
            ],
            check=True,
        )
        assert (cooked / "cooked.program.json").stat().st_size > 0
    print(f"Installed VernonDSL {RELEASE_VERSION} CPU dispatch, frontend, cooker, and package checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
