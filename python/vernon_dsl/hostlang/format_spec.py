"""Format specs of Host f-string replacement fields, the subset VernonRuntime's formatter implements as Python's
format() does: `[[fill]align][sign][0][width][.precision][type]`."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from ..frontend.model import ConcreteType

_SPEC = re.compile(
    r"(?:(?P<fill>.)?(?P<align>[<>^=]))?(?P<sign>[-+ ])?(?P<zero>0)?(?P<width>\d+)?(?:\.(?P<precision>\d+))?"
    r"(?P<type>[A-Za-z%]?)",
    re.DOTALL,
)
_INTEGER_TYPES = frozenset({"", "d", "x", "X", "o", "b"})
_FLOAT_TYPES = frozenset({"", "f", "e", "E"})
MAX_PRECISION = 64
MAX_WIDTH = 1 << 20


@dataclass(frozen=True)
class FormatSpec:
    """A parsed spec; empty strings and -1 mean the value's default. `fill` is one code point."""

    fill: str = ""
    align: str = ""
    sign: str = ""
    width: int = 0
    precision: int = -1
    type: str = ""

    @property
    def empty(self) -> bool:
        return self == FormatSpec()


def value_kind(value_type: ConcreteType) -> str | None:
    """How a replacement field formats a value: as "str", "bool", "integer" or "float"; None if it cannot."""

    if value_type.kind == "str":
        return "str"
    if value_type.kind == "handle":
        return "integer"
    if value_type.kind != "scalar":
        return None
    if value_type.name == "bool":
        return "bool"
    return "float" if value_type.is_float else "integer"


def parse_format_spec(text: str, kind: str) -> FormatSpec:
    """The spec `text` for a value formatted as `kind`; raises ValueError with the reason it is not supported."""

    if not text:
        return FormatSpec()
    match = _SPEC.fullmatch(text)
    if match is None:
        raise ValueError(f"format spec '{text}' is not [[fill]align][sign][0][width][.precision][type]")
    fill, align, sign = match["fill"] or "", match["align"] or "", match["sign"] or ""
    width = int(match["width"] or 0)
    precision = -1 if match["precision"] is None else int(match["precision"])
    kind_type = match["type"]
    if match["zero"]:
        # As in Python, a 0 before the width pads with zeros, after the sign unless an alignment is given.
        fill = fill or "0"
        if not align and kind != "str":
            align = "="
    if kind == "bool":
        raise ValueError("a bool formats without a spec; Python formats a bool with one as an integer")
    if kind == "str":
        if sign or align == "=" or precision >= 0 or kind_type not in {"", "s"}:
            raise ValueError(f"format spec '{text}' does not apply to a str; a str takes [[fill]align][width]")
        kind_type = ""
    elif kind == "integer":
        if kind_type not in _INTEGER_TYPES or precision >= 0:
            raise ValueError(f"format spec '{text}' does not apply to an integer; its types are d, x, X, o and b")
    elif kind_type not in _FLOAT_TYPES:
        raise ValueError(f"format spec '{text}' does not apply to a float; its types are f, e and E")
    elif precision >= 0 and not kind_type:
        raise ValueError(f"format spec '{text}' gives a precision without a type; add f, e or E")
    if precision > MAX_PRECISION:
        raise ValueError(f"format precision {precision} exceeds {MAX_PRECISION}")
    if width > MAX_WIDTH:
        raise ValueError(f"format width {width} exceeds {MAX_WIDTH}")
    return FormatSpec(fill, align, sign, width, precision, kind_type)


def spec_text(format_spec: ast.expr | None) -> str | None:
    """The text of a replacement field's spec, "" without one; None when the spec has replacement fields itself."""

    if format_spec is None:
        return ""
    assert isinstance(format_spec, ast.JoinedStr)
    parts = []
    for part in format_spec.values:
        if not (isinstance(part, ast.Constant) and isinstance(part.value, str)):
            return None
        parts.append(part.value)
    return "".join(parts)
