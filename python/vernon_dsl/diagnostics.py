from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TypeVar

from .language.ast_utils import STATELESS_NODES

_Node = TypeVar("_Node", bound=ast.AST)

# A project combines definitions from several files into one tree, so each node names the file it was parsed from.
_SOURCE_FILE = "vernon_source_file"


@dataclass(frozen=True)
class SourceLocation:
    filename: str
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.filename}:{self.line}:{self.column}"


class CompileError(Exception):
    """A deterministic, source-located DSL compilation error."""

    def __init__(self, message: str, location: SourceLocation):
        self.message = message
        self.location = location
        super().__init__(f"{location}: error: {message}")


class ProgramCompileError(ValueError):
    """A Program cannot be captured, compiled, deployed, or loaded."""


def mark_source_file(tree: ast.AST, filename: str) -> None:
    """Records `filename` as the file of every node of `tree`."""

    for node in ast.walk(tree):
        if not isinstance(node, STATELESS_NODES):
            setattr(node, _SOURCE_FILE, filename)


def copy_location(new_node: _Node, old_node: ast.AST) -> _Node:
    """`ast.copy_location` that also carries the source file."""

    ast.copy_location(new_node, old_node)
    if hasattr(old_node, _SOURCE_FILE):
        setattr(new_node, _SOURCE_FILE, getattr(old_node, _SOURCE_FILE))
    return new_node


def fix_missing_locations(tree: _Node) -> _Node:
    """`ast.fix_missing_locations` that also gives nodes without a source file their parent's."""

    ast.fix_missing_locations(tree)
    pending: list[tuple[ast.AST, str | None]] = [(tree, None)]
    while pending:
        node, inherited = pending.pop()
        filename = getattr(node, _SOURCE_FILE, inherited)
        if filename is not None and not isinstance(node, STATELESS_NODES):
            setattr(node, _SOURCE_FILE, filename)
        pending.extend((child, filename) for child in ast.iter_child_nodes(node))
    return tree


def node_location(node: ast.AST, default_filename: str) -> SourceLocation:
    """Where `node` was written; `default_filename` names the file of nodes no file was recorded for."""

    return SourceLocation(
        getattr(node, _SOURCE_FILE, default_filename),
        getattr(node, "lineno", 1),
        getattr(node, "col_offset", 0) + 1,
    )


__all__ = [
    "CompileError",
    "ProgramCompileError",
    "SourceLocation",
    "copy_location",
    "fix_missing_locations",
    "mark_source_file",
    "node_location",
]
