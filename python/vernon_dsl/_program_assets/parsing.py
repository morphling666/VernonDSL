"""Static preflight for Program Asset declarations.

This is a **lint**, not a model of what an asset is. Cooking evaluates the asset source and reads the typed
declaration, so nothing here decides a form, builds a transform, or lists stages. Keeping only the checks that are
genuinely static avoids a second, partially-correct answer to questions the typed declaration already answers.

Three things are worth checking without running the source: that the id is a literal, that the variant keys are
canonical and locally declared, and that statically named entries resolve inside the project. The last one earns its
place because the compiler's own answer is late and obscure -- an entry reached through a module alias fails with
"unknown shader entry point" long after cooking has begun.
"""

from __future__ import annotations

import ast
import keyword
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..bundle import ProgramCompileError
from ..diagnostics import CompileError
from ..language.ast_utils import dotted_name
from ..language.stage_registry import ENTRY_DECORATOR_STAGES
from ..module_graph import load_project, resolve_project_entry
from ..types import (
    Specialization,
    SpecializationAssignment,
    f32,
    f64,
    i32,
    specialization_assignment,
    u32,
)
from ..types import (
    bool as bool_type,
)
from .declaration import VARIANT_CAP


@dataclass(frozen=True)
class ProgramAssetLint:
    """What a Program Asset declaration says about itself without being run."""

    id: str
    variants: tuple[tuple[SpecializationAssignment, ...], ...]
    entries: tuple[str, ...]


def program_asset_reference(value: str | Path) -> tuple[Path, str]:
    spelling = str(value)
    marker = spelling.rfind(".py:")
    if marker < 0:
        raise ProgramCompileError(
            "Program Asset input must be a Python descriptor reference in source.py:descriptor_name form"
        )
    source = Path(spelling[: marker + 3]).resolve()
    descriptor_name = spelling[marker + 4 :]
    if not descriptor_name or not descriptor_name.isidentifier() or keyword.iskeyword(descriptor_name):
        raise ProgramCompileError("Program Asset reference requires a valid Python descriptor name after source.py:")
    return source, descriptor_name


def _assigned_values(tree: ast.Module, name: str) -> list[ast.expr]:
    values = []
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if len(targets) == 1 and isinstance(targets[0], ast.Name) and targets[0].id == name:
            if statement.value is not None:
                values.append(statement.value)
    return values


def _specialization_bindings(tree: ast.Module) -> dict[str, Specialization]:
    bindings: dict[str, Specialization] = {}
    scalar_types = {"bool": bool_type, "i32": i32, "u32": u32, "f32": f32, "f64": f64}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if not isinstance(value, ast.Call):
            continue
        kind = (dotted_name(value.func) or "").split(".")[-1]
        if kind not in {"feature", "specialization"}:
            continue
        if (
            value.keywords
            or not value.args
            or not isinstance(value.args[0], ast.Constant)
            or not isinstance(value.args[0].value, str)
        ):
            raise ProgramCompileError("Program Asset specialization declarations require a literal name")
        if kind == "feature":
            if len(value.args) != 1:
                raise ProgramCompileError("feature declarations require one string literal")
            scalar_type = bool_type
        else:
            type_name = (dotted_name(value.args[1]) or "").split(".")[-1] if len(value.args) == 2 else ""
            scalar_type = scalar_types.get(type_name)
            if scalar_type is None:
                raise ProgramCompileError("specialization declarations require bool, i32, u32, f32, or f64")
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            raise ProgramCompileError("Program Asset specialization declarations require a simple name")
        bindings[targets[0].id] = Specialization(value.args[0].value, scalar_type)
    return bindings


def _declaration_call(tree: ast.Module, descriptor_name: str, source_path: Path) -> ast.Call:
    declarations = [
        value
        for value in _assigned_values(tree, descriptor_name)
        if isinstance(value, ast.Call) and (dotted_name(value.func) or "").split(".")[-1] == "program_asset"
    ]
    if not declarations:
        raise ProgramCompileError(f"Program Asset declaration '{descriptor_name}' was not found in {source_path}")
    if len(declarations) != 1:
        raise ProgramCompileError(f"duplicate Program Asset declaration: {descriptor_name}")
    return declarations[0]


def _static_entries(tree: ast.Module, program: ast.expr) -> list[ast.expr]:
    """The entry expressions a reader can see without running the source, if any.

    A directly authored `vd.pipeline(vertex, fragment, ...)` contributes its positional stage entries. Module and
    Module VJP declarations contribute none because their implementation stages only exist after capture.
    """

    if isinstance(program, ast.Name):
        named = [
            value
            for value in _assigned_values(tree, program.id)
            if isinstance(value, ast.Call) and (dotted_name(value.func) or "").split(".")[-1] == "vjp"
        ]
        if len(named) > 1:
            raise ProgramCompileError(f"duplicate VJP program declaration: {program.id}")
        if named:
            program = named[0]
    if isinstance(program, ast.Call) and (dotted_name(program.func) or "").split(".")[-1] == "vjp":
        if not program.args:
            return []
        program = program.args[0]
    if isinstance(program, ast.Call) and (dotted_name(program.func) or "").split(".")[-1] == "pipeline":
        return list(program.args)
    if isinstance(program, ast.Name):
        return [program]
    if isinstance(program, ast.Tuple):
        raise ProgramCompileError("Program Asset graphics must use vd.pipeline(...), not a tuple of entry functions")
    return []


def _resolved_entry(source_path: Path, entry: ast.expr) -> str:
    if not isinstance(entry, ast.Name):
        raise ProgramCompileError("program_asset program entries must use simple imported or local function names")
    try:
        resolution = resolve_project_entry(source_path, entry.id)
    except CompileError as error:
        raise ProgramCompileError(str(error)) from None
    if resolution is None:
        raise ProgramCompileError(f"program_asset program entry '{entry.id}' is not a function")
    stages = {ENTRY_DECORATOR_STAGES[name] for name in resolution.decorators if name in ENTRY_DECORATOR_STAGES}
    if len(stages) != 1:
        raise ProgramCompileError(
            f"program_asset program entry '{entry.id}' must have exactly one entry-stage decorator"
        )
    return entry.id


def _linted_variants(
    tree: ast.Module, keywords: dict[str, ast.expr]
) -> tuple[tuple[SpecializationAssignment, ...], ...]:
    node = keywords.get("variants")
    if node is None:
        return ((),)
    if not isinstance(node, (ast.Tuple, ast.List)):
        raise ProgramCompileError("program_asset variants must be a tuple or list")
    specializations = _specialization_bindings(tree)
    parsed: list[tuple[SpecializationAssignment, ...]] = []
    for row in node.elts:
        if not isinstance(row, ast.Dict):
            raise ProgramCompileError("each Program Asset variant must be a specialization mapping")
        assignments: list[SpecializationAssignment] = []
        for parameter, value in zip(row.keys, row.values, strict=True):
            if not isinstance(parameter, ast.Name) or parameter.id not in specializations:
                raise ProgramCompileError("Program Asset variant keys must reference locally declared specializations")
            try:
                literal = ast.literal_eval(value)
                assignments.append(specialization_assignment(specializations[parameter.id], literal))
            except (ValueError, TypeError, SyntaxError) as error:
                raise ProgramCompileError(str(error)) from None
        key = tuple(sorted(assignments))
        if len({assignment.name for assignment in key}) != len(key):
            raise ProgramCompileError("Program Asset variant contains duplicate specialization names")
        if key in parsed:
            raise ProgramCompileError("duplicate Program Asset variant")
        parsed.append(key)
    if not parsed:
        raise ProgramCompileError("a Program Asset must declare at least one variant")
    if len(parsed) > VARIANT_CAP:
        raise ProgramCompileError(
            f"a Program Asset declares {len(parsed)} variants, exceeding variant cap {VARIANT_CAP}"
        )
    return tuple(parsed)


def lint_python_program_asset(source: str | Path, descriptor_name: str) -> ProgramAssetLint:
    """Check a Program Asset declaration statically. Cooking does not depend on this and never consults it."""

    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise ProgramCompileError(f"Program Asset source does not exist: {source_path}")
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as error:
        raise ProgramCompileError(f"cannot parse Program Asset source {source_path}: {error}") from None

    declaration = _declaration_call(tree, descriptor_name, source_path)
    if declaration.args:
        raise ProgramCompileError("program_asset accepts keyword arguments only")
    keywords = {item.arg: item.value for item in declaration.keywords if item.arg is not None}
    if len(keywords) != len(declaration.keywords):
        raise ProgramCompileError("program_asset does not accept expanded keyword arguments")
    unknown = set(keywords) - {"id", "program", "variants"}
    if unknown:
        raise ProgramCompileError("unknown program_asset argument(s): " + ", ".join(sorted(unknown)))

    identifier: Any = keywords.get("id")
    if identifier is None:
        raise ProgramCompileError("program_asset declaration requires 'id'")
    try:
        identifier = ast.literal_eval(identifier)
    except (ValueError, TypeError, SyntaxError):
        raise ProgramCompileError("program_asset 'id' must be a literal value") from None
    if not isinstance(identifier, str) or not identifier:
        raise ProgramCompileError("Program Asset id must be a non-empty string")

    program = keywords.get("program")
    if program is None:
        raise ProgramCompileError("program_asset declaration requires 'program'")
    entries = tuple(_resolved_entry(source_path, entry) for entry in _static_entries(tree, program))

    variants = _linted_variants(tree, keywords)
    requested = {assignment.name for variant in variants for assignment in variant if assignment.type == "bool"}
    undeclared = requested - set(load_project(source_path).features)
    if undeclared:
        raise ProgramCompileError("variant requests undeclared feature(s): " + ", ".join(sorted(undeclared)))
    return ProgramAssetLint(identifier, variants, entries)


__all__ = ["ProgramAssetLint", "lint_python_program_asset", "program_asset_reference"]
