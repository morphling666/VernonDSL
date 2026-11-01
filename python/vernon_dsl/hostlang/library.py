"""Schema identities of `vd.library` modules.

A library's identity is `<name>-v<version>:<hash>`. The hash covers what a binding must agree on to be called
correctly: every extern's symbol, effect, ABI and parameter and result layouts, and the layouts of the module's
Structs, handles and enums (an enum's members and values included). Metadata is not covered; it changes how a
generator reaches C++, not the C functions. The same computation serves the externs' schema when a program loads
and the identity a generated binding publishes, so no declaration spells it by hand.
"""

from __future__ import annotations

import ast
import copy
import threading
from pathlib import Path

from ..frontend.model import ConcreteType
from ..source_identity import source_file_digest
from .model import HostProgram
from .reflect import _Fields, canonical, fnv1a

_lock = threading.Lock()
# Library file -> (the files its identity was computed from with their digests, identity).
_identities: dict[Path, tuple[tuple[tuple[Path, str], ...], str]] = {}


def library_identity(path: str | Path, name: str, version: int) -> str:
    """The identity of the `vd.library(name, version)` declared by `path`."""

    path = Path(path).resolve()
    with _lock:
        cached = _identities.get(path)
        if cached is not None and all(source_file_digest(item) == digest for item, digest in cached[0]):
            return cached[1]
    from .frontend import load_host_library

    program, generic_signatures = load_host_library(path)
    identity = f"{name}-v{version}:{_hash(program, generic_signatures):016x}"
    with _lock:
        _identities[path] = (program.dependencies, identity)
    return identity


def _hash(program: HostProgram, generic_signatures: tuple[str, ...]) -> int:
    fields = _Fields([program])
    lines = [canonical(ConcreteType("struct", name), fields) for name in sorted(program.structs)]
    lines += [f"handle {name}" for name in sorted(program.handles)]
    lines += [canonical(ConcreteType("enum", name), fields) for name in sorted(program.enums)]
    for extern in sorted(program.externs.values(), key=lambda item: item.symbol):
        parameters = ",".join(canonical(value, fields) for _, value in extern.parameters)
        result = "none" if extern.result is None else canonical(extern.result, fields)
        lines.append(f"{extern.symbol}:{extern.effect.spelling}:{extern.abi}({parameters})->{result}")
    lines += sorted(generic_signatures)
    return fnv1a("\n".join(lines))


def generic_signature(node: ast.FunctionDef) -> str:
    """A generic extern's declaration as written, which has no typed form until a call site specializes it."""

    signature = copy.deepcopy(node)
    signature.body = [ast.Expr(ast.Constant(Ellipsis))]
    for decorator in signature.decorator_list:
        if isinstance(decorator, ast.Call):
            decorator.keywords = [keyword for keyword in decorator.keywords if keyword.arg != "meta"]
    return ast.unparse(signature)
