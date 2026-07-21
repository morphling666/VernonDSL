from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from .diagnostics import CompileError
from .shader_assets import ShaderAssetError, cook_shader_pipeline


def _default_compiler() -> Path:
    executable = "vernon-compile.exe" if os.name == "nt" else "vernon-compile"
    configured = os.environ.get("VERNON_COMPILER")
    if configured:
        return Path(configured).expanduser()
    packaged = Path(__file__).resolve().with_name(executable)
    if packaged.is_file():
        return packaged
    discovered = shutil.which(executable)
    if discovered:
        return Path(discovered)
    raise ShaderAssetError(
        "cannot find vernon-compile; install a native VernonDSL wheel, "
        "pass --compiler, or set VERNON_COMPILER"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vernon-cook-shader",
        description=
        "Cook a shader-pipeline manifest into a schema-2 pipeline asset.",
    )
    parser.add_argument(
        "pipeline",
        type=Path,
        help="legacy JSON manifest or Python source.py:descriptor_name",
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--compiler",
        type=Path,
        help="native compiler (defaults to VERNON_COMPILER or the wheel copy)",
    )
    parser.add_argument("--target", default="opengl")
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        cook_shader_pipeline(
            pipeline_manifest=arguments.pipeline,
            asset_root=arguments.asset_root,
            compiler=arguments.compiler or _default_compiler(),
            output=arguments.output,
            target=arguments.target,
        )
    except (CompileError, ShaderAssetError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
