"""Shared linearized output indexing for generated Program kernels."""

from __future__ import annotations

import re


def view_shape_annotation(rank: int) -> str:
    if rank == 0:
        return "()"
    if rank == 1:
        return "(vd.dyn,)"
    return "(" + ", ".join(["vd.dyn"] * rank) + ")"


def element_token(element: str) -> str:
    if element.startswith("vd.") and "[" not in element:
        return element[3:]
    portable = element.replace(", ", "x")
    return "_".join(part for part in re.split(r"[^A-Za-z0-9]+", portable) if part)


def linear_index_target(rank: int) -> str:
    if rank <= 1:
        return "linear" if rank == 1 else "()"
    return ", ".join(f"index{axis}" for axis in range(rank))


def linear_index_prelude(rank: int) -> list[str]:
    if rank <= 1:
        return ["    linear = gid[0]"] if rank == 1 else []
    names = [f"index{axis}" for axis in range(rank)]
    lines = [
        "    linear = gid[0]",
        "    shape = output.shape",
    ]
    for axis in range(rank - 1, 0, -1):
        lines.append(f"    {names[axis]} = linear % shape[{axis}]")
        lines.append(f"    linear = linear // shape[{axis}]")
    lines.append(f"    {names[0]} = linear")
    return lines


def output_index_names(rank: int) -> list[str]:
    if rank <= 0:
        return []
    if rank == 1:
        return ["linear"]
    return [f"index{axis}" for axis in range(rank)]


__all__ = [
    "element_token",
    "linear_index_prelude",
    "linear_index_target",
    "output_index_names",
    "view_shape_annotation",
]
