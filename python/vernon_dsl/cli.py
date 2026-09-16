from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .compiler import compile_file
from .diagnostics import CompileError
from .types import SpecializationAssignment, f32, f64, i32, specialization, specialization_assignment, u32
from .types import bool as dsl_bool


def _specialization_assignment(value: str) -> SpecializationAssignment:
    try:
        name_and_type, spelling = value.split("=", 1)
        name, type_name = name_and_type.split(":", 1)
    except ValueError:
        raise argparse.ArgumentTypeError("specialization must be NAME:TYPE=VALUE") from None
    types = {"bool": dsl_bool, "i32": i32, "u32": u32, "f32": f32, "f64": f64}
    scalar_type = types.get(type_name)
    if not name or scalar_type is None:
        raise argparse.ArgumentTypeError("specialization type must be bool, i32, u32, f32, or f64")
    try:
        if scalar_type is dsl_bool:
            if spelling not in {"true", "false"}:
                raise ValueError
            parsed: object = spelling == "true"
        elif scalar_type in {i32, u32}:
            parsed = int(spelling, 10)
        else:
            parsed = float(spelling)
        return specialization_assignment(specialization(name, scalar_type), parsed)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"invalid {type_name} specialization value {spelling!r}") from None


def _specialization_binding(value: str) -> tuple[str, str]:
    try:
        local_name, specialization_name = value.split("=", 1)
    except ValueError:
        raise argparse.ArgumentTypeError("specialization binding must be LOCAL=NAME") from None
    if not local_name.isidentifier() or not specialization_name:
        raise argparse.ArgumentTypeError("specialization binding must contain a Python local name and a public name")
    return local_name, specialization_name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vernon-compile-python",
        description="Compile a restricted Vernon Python DSL module to textual MLIR.",
    )
    parser.add_argument("input", type=Path, help="input Python DSL file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output MLIR file")
    parser.add_argument("--entry", help="compile only the selected entry point")
    parser.add_argument(
        "--specialization",
        type=_specialization_assignment,
        action="append",
        default=[],
        help="bind NAME:TYPE=VALUE for bool, i32, u32, f32, or f64 (repeatable)",
    )
    parser.add_argument(
        "--specialization-binding",
        type=_specialization_binding,
        action="append",
        default=[],
        help="map a non-Boolean source declaration LOCAL to specialization NAME (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        output = compile_file(
            arguments.input,
            entry=arguments.entry,
            specializations=tuple(arguments.specialization),
            specialization_bindings=tuple(arguments.specialization_binding),
        )
        arguments.output.write_text(output, encoding="utf-8", newline="\n")
    except (CompileError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
