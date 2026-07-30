from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarType:
    name: str
    mlir: str
    numpy: str
    category: str
    width: int


SCALAR_TYPES = {
    scalar.name: scalar
    for scalar in (
        ScalarType("bool", "i1", "bool", "bool", 1),
        ScalarType("i32", "i32", "int32", "signed", 32),
        # MLIR arith uses signless integers; unsigned semantics live in ops
        # such as uitofp/divui and in Vernon's ABI dtype metadata.
        ScalarType("u32", "i32", "uint32", "unsigned", 32),
        ScalarType("f16", "f16", "float16", "float", 16),
        ScalarType("f32", "f32", "float32", "float", 32),
        ScalarType("f64", "f64", "float64", "float", 64),
    )
}
SCALAR_ALIASES = {"int": "i32", "float": "f32"}
FLOAT_ORDER = ("f16", "f32", "f64")


def canonical_scalar(name: str) -> str | None:
    canonical = SCALAR_ALIASES.get(name, name)
    return canonical if canonical in SCALAR_TYPES else None


def common_scalar(left: str, right: str, *, true_division: bool = False) -> str | None:
    """Return the deterministic safe common scalar type, if one exists."""

    if left == "bool" or right == "bool":
        return left if left == right and not true_division else None
    if true_division and left in {"i32", "u32"} and right in {"i32", "u32"}:
        return "f32" if left == right else None
    if left == right:
        return left
    left_type = SCALAR_TYPES[left]
    right_type = SCALAR_TYPES[right]
    if left_type.category == "float" and right_type.category == "float":
        return FLOAT_ORDER[max(FLOAT_ORDER.index(left), FLOAT_ORDER.index(right))]
    if left_type.category == "float" and right_type.category in {"signed", "unsigned"}:
        return left
    if right_type.category == "float" and left_type.category in {"signed", "unsigned"}:
        return right
    # i32/u32 have no lossless common representation in language v3.
    return None


def can_implicitly_convert(source: str, target: str) -> bool:
    if source == target:
        return True
    source_type = SCALAR_TYPES[source]
    target_type = SCALAR_TYPES[target]
    if source_type.category in {"signed", "unsigned"} and target_type.category == "float":
        return True
    return source_type.category == "float" and target_type.category == "float" and source_type.width < target_type.width
