"""The declaration marker every Vernon decorator attaches: `__vernon_dsl__ = (kind, options)`.

`kind` names what the object declares (`struct`, `binding_group`, `compute`, `vertex`, `fragment`, `func`, ...);
`options` holds the decorator's keyword options. Read and write the marker only through this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, TypeVar

_T = TypeVar("_T")
_ATTRIBUTE = "__vernon_dsl__"
_NO_OPTIONS: Mapping[str, Any] = MappingProxyType({})


def mark(value: _T, kind: str, **options: Any) -> _T:
    setattr(value, _ATTRIBUTE, (kind, options))
    return value


def dsl_kind(value: Any) -> str | None:
    marker = getattr(value, _ATTRIBUTE, None)
    return marker[0] if marker is not None else None


def dsl_options(value: Any) -> Mapping[str, Any]:
    marker = getattr(value, _ATTRIBUTE, None)
    return marker[1] if marker is not None else _NO_OPTIONS


def is_struct_class(value: Any) -> bool:
    return isinstance(value, type) and dsl_kind(value) == "struct"


def is_binding_group(value: Any) -> bool:
    return isinstance(value, type) and dsl_kind(value) == "binding_group"


__all__ = ["dsl_kind", "dsl_options", "is_binding_group", "is_struct_class", "mark"]
