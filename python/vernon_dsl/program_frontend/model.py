"""Immutable program-level IR, separate from kernel function IR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..frontend.model import ConcreteType


@dataclass(frozen=True)
class ProgramType:
    logical: ConcreteType


@dataclass(frozen=True)
class GraphValue:
    name: str
    type: ProgramType
    role: str
    primal: int


@dataclass(frozen=True)
class GraphOperation:
    id: int
    kind: str
    name: str
    operands: tuple[tuple[str, str], ...]
    results: tuple[tuple[str, str], ...]
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ProgramGraph:
    name: str
    direction: str
    values: tuple[GraphValue, ...]
    arguments: tuple[GraphValue, ...]
    results: tuple[GraphValue, ...]
    operations: tuple[GraphOperation, ...]


@dataclass(frozen=True)
class ProgramImplementation:
    callee: str
    entry: str
    kind: str
    mlir: str


@dataclass(frozen=True)
class ParsedProgram:
    """Program-level forward/backward graphs and their canonical MLIR."""

    forward: ProgramGraph
    backward: ProgramGraph | None
    mlir: str
    implementations: tuple[ProgramImplementation, ...] = ()
    provenance: tuple[str, ...] = ()
    vjp_wrt: tuple[str, ...] = ()
    structs: tuple[tuple[str, tuple[tuple[str, ConcreteType], ...]], ...] = ()

    @property
    def identity(self) -> str:
        digest = hashlib.sha256(self.mlir.encode("utf-8"))
        for implementation in self.implementations:
            digest.update(implementation.callee.encode("utf-8"))
            digest.update(implementation.entry.encode("utf-8"))
            digest.update(implementation.kind.encode("utf-8"))
            digest.update(implementation.mlir.encode("utf-8"))
        return digest.hexdigest()


__all__ = [
    "GraphOperation",
    "GraphValue",
    "ParsedProgram",
    "ProgramGraph",
    "ProgramImplementation",
    "ProgramType",
]
