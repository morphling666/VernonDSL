from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .diagnostics import CompileError
from .shader_assets import ShaderAssetError, cook_shader_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vernon-cook-shader",
        description=
        "Cook a Python pipeline_asset descriptor into a schema-2 pipeline asset.",
    )
    parser.add_argument(
        "pipeline_asset",
        type=str,
        help="Python descriptor reference in source.py:descriptor_name form",
    )
    parser.add_argument(
        "--compiler",
        type=Path,
        help="deprecated compatibility option; ignored (cooking is in-process)",
    )
    parser.add_argument("--target", default="opengl")
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.compiler is not None:
        print("warning: --compiler is deprecated and ignored; "
              "vernon-cook-shader uses vernon_dsl._native",
              file=sys.stderr)
    try:
        cook_shader_pipeline(
            pipeline_asset=arguments.pipeline_asset,
            compiler=arguments.compiler,
            output=arguments.output,
            target=arguments.target,
        )
    except (CompileError, ShaderAssetError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
