"""`python -m vernon_dsl.bindgen generate|check <module.py> [--output-dir DIR] [--pyi PATH]`.

`generate` writes the level 1 outputs; `check` exits 1 when committed outputs differ from what the module generates.
The output directory defaults to the module's own; `--pyi` puts the type stub elsewhere, such as beside the Python
module that re-exports the extension's names.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generator import BindgenError, check, write


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vernon_dsl.bindgen")
    parser.add_argument("command", choices=["generate", "check"])
    parser.add_argument("module", type=Path, help="the vd.library Host module")
    parser.add_argument("--output-dir", type=Path, help="where the outputs live (default: the module's directory)")
    parser.add_argument("--pyi", type=Path, help="where the type stub lives (default: the output directory)")
    arguments = parser.parse_args(argv)
    directory = arguments.output_dir or arguments.module.resolve().parent
    try:
        if arguments.command == "generate":
            for path in write(arguments.module, directory, arguments.pyi):
                print(path)
            return 0
        stale = check(arguments.module, directory, arguments.pyi)
    except BindgenError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for path in stale:
        print(f"{path} is out of date; run python -m vernon_dsl.bindgen generate", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
