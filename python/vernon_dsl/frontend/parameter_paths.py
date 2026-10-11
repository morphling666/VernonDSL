from __future__ import annotations

from collections.abc import Iterable


def split_parameter_path(path: str, parameters: Iterable[str]) -> tuple[str, tuple[str, ...]]:
    """Split a public derivative path into its entry parameter and projection components.

    Binding-group leaves are entry parameters whose names contain dots, so the parameter is the longest name that
    equals the path or is a dotted prefix of it. A path that names no parameter splits at its first dot.
    """
    best: str | None = None
    for name in parameters:
        if (path == name or path.startswith(f"{name}.")) and (best is None or len(name) > len(best)):
            best = name
    if best is None:
        best = path.split(".", 1)[0]
    remainder = path[len(best) + 1 :]
    return best, tuple(remainder.split(".")) if remainder else ()


__all__ = ["split_parameter_path"]
