from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .compiler import compile_file
from .diagnostics import CompileError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vernon-compile-python",
        description="Compile a restricted Vernon Python DSL module to textual MLIR.",
    )
    parser.add_argument("input", type=Path, help="input Python DSL file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output MLIR file")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        output = compile_file(arguments.input)
        arguments.output.write_text(output, encoding="utf-8", newline="\n")
    except (CompileError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
