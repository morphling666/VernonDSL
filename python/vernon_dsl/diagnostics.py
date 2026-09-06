from __future__ import annotations

from dataclasses import dataclass


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


__all__ = ["CompileError", "ProgramCompileError", "SourceLocation"]
