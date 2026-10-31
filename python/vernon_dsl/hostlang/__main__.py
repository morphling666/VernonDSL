"""`python -m vernon_dsl.hostlang compile SOURCE... [--entry NAME ...] [...]`: writes a Host object, header, and
reflection; without `--entry`, of every `@host` function and class the listed modules define, in one table.

`python -m vernon_dsl.hostlang include-dir` prints the directory of `vernon_host/reflect.h`, for build systems.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..diagnostics import CompileError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vernon_dsl.hostlang")
    commands = parser.add_subparsers(dest="command", required=True)
    compile_command = commands.add_parser("compile", help="compile a Host module or @host entries ahead of time")
    compile_command.add_argument("source", type=Path, nargs="+")
    compile_command.add_argument(
        "--entry", action="append", help="a @host entry of the one source; repeatable; omit for the modules"
    )
    compile_command.add_argument("--output-dir", type=Path, required=True)
    compile_command.add_argument("--name", help="artifact base name; defaults to the first source file stem")
    compile_command.add_argument("--target", help="LLVM target triple; defaults to this machine")
    compile_command.add_argument("--prefix", default="vernon_host_", help="C symbol prefix of exported entries")
    compile_command.add_argument("--type-prefix", help="prefix of generated C type names")
    compile_command.add_argument("--types-header", help="header that declares every c_type Struct and handle")
    compile_command.add_argument("--cpu", default="")
    compile_command.add_argument("--features", default="")
    compile_command.add_argument("--emit-ir", action="store_true", help="also write the unoptimized LLVM IR")
    commands.add_parser("include-dir", help="print the directory holding vernon_host/reflect.h")
    arguments = parser.parse_args(argv)

    if arguments.command == "include-dir":
        from .reflect import include_dir

        print(include_dir())
        return 0

    from .aot import compile_host

    try:
        artifacts = compile_host(
            arguments.source,
            arguments.entry,
            target=arguments.target,
            prefix=arguments.prefix,
            type_prefix=arguments.type_prefix,
            types_header=arguments.types_header,
            cpu=arguments.cpu,
            features=arguments.features,
        )
    except (CompileError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    name = arguments.name or arguments.source[0].stem
    paths = artifacts.write(arguments.output_dir, name)
    if arguments.emit_ir:
        (arguments.output_dir / f"{name}.ll").write_text(artifacts.ir, encoding="utf-8", newline="\n")
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
