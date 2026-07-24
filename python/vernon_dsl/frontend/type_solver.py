from __future__ import annotations

from typing import TypeAlias

from ..language.scalar_types import can_implicitly_convert, common_scalar
from .model import ConcreteType, LiteralType

InferenceType: TypeAlias = ConcreteType | LiteralType


def scalar(name: str) -> ConcreteType:
    return ConcreteType("scalar", name)


def literal(value: bool | int | float) -> InferenceType:
    if isinstance(value, bool):
        return scalar("bool")
    if isinstance(value, int):
        return LiteralType("integer", value)
    return LiteralType("floating", value)


def default_type(value_type: InferenceType) -> ConcreteType:
    if isinstance(value_type, ConcreteType):
        return value_type
    return scalar("i32" if value_type.category == "integer" else "f32")


def contextualize(value_type: InferenceType, expected: ConcreteType | None) -> ConcreteType:
    if isinstance(value_type, ConcreteType):
        return value_type
    if expected is None or expected.kind != "scalar" or expected.name == "bool":
        return default_type(value_type)
    if value_type.category == "floating" and not expected.is_float:
        return default_type(value_type)
    if value_type.category == "integer" and not (expected.is_integer or expected.is_float):
        return default_type(value_type)
    return expected


def element_type(value_type: InferenceType) -> InferenceType:
    if isinstance(value_type, ConcreteType) and value_type.kind == "tensor":
        element = value_type.arguments[0]
        assert isinstance(element, ConcreteType)
        return element
    return value_type


def common_type(
    left: InferenceType,
    right: InferenceType,
    *,
    division: bool = False,
) -> InferenceType | None:
    if isinstance(left, LiteralType) and isinstance(right, LiteralType):
        if division or "floating" in {left.category, right.category}:
            return LiteralType("floating", 0.0)
        return LiteralType("integer", 0)

    if isinstance(left, LiteralType):
        right_element = element_type(right)
        expected = right_element if isinstance(right_element, ConcreteType) else None
        left = contextualize(left, expected)
    if isinstance(right, LiteralType):
        left_element = element_type(left)
        expected = left_element if isinstance(left_element, ConcreteType) else None
        right = contextualize(right, expected)

    assert isinstance(left, ConcreteType)
    assert isinstance(right, ConcreteType)
    if left.kind not in {"scalar", "tensor"} or right.kind not in {"scalar", "tensor"}:
        return left if left == right else None
    if left.kind == "tensor" and right.kind == "tensor" and left.arguments[1:] != right.arguments[1:]:
        return None
    left_element = element_type(left)
    right_element = element_type(right)
    assert isinstance(left_element, ConcreteType)
    assert isinstance(right_element, ConcreteType)
    if left_element.kind != "scalar" or right_element.kind != "scalar":
        return None
    name = common_scalar(left_element.name, right_element.name, true_division=division)
    if name is None:
        return None
    element = scalar(name)
    shape = left.arguments[1:] if left.kind == "tensor" else right.arguments[1:] if right.kind == "tensor" else ()
    return ConcreteType("tensor", "Tensor", (element, *shape)) if shape else element


def can_convert(value_type: InferenceType, target: ConcreteType) -> bool:
    source = contextualize(value_type, target)
    if source == target:
        return True
    if source.kind != target.kind:
        return False
    if source.kind == "tuple":
        return len(source.arguments) == len(target.arguments) and all(
            isinstance(source_element, ConcreteType)
            and isinstance(target_element, ConcreteType)
            and can_convert(source_element, target_element)
            for source_element, target_element in zip(source.arguments, target.arguments, strict=True)
        )
    if source.kind == "tensor" and source.arguments[1:] != target.arguments[1:]:
        return False
    source_element = element_type(source)
    target_element = element_type(target)
    return (
        isinstance(source_element, ConcreteType)
        and isinstance(target_element, ConcreteType)
        and source_element.kind == "scalar"
        and target_element.kind == "scalar"
        and can_implicitly_convert(source_element.name, target_element.name)
    )


def describe(value_type: InferenceType) -> str:
    if isinstance(value_type, ConcreteType):
        return value_type.mlir
    return f"{value_type.category} literal"
