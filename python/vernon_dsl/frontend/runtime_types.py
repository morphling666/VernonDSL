"""Canonical runtime annotations shared by Kernel and Module binding."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from ..host_values import pack_host_value
from ..types import TypeExpr, _Scalar, dyn
from .model import ConcreteType, SemanticCategory, semantic_category


@dataclass(frozen=True)
class _StorageParameter:
    dtype: Any
    shape: tuple[int, ...]
    access: str
    as_view: bool


@dataclass(frozen=True)
class _AnnotatedParameter:
    annotation: Any
    logical: ConcreteType


@dataclass(frozen=True)
class RuntimeParameterDescriptor:
    """Tagged, immutable description of one public runtime parameter."""

    kind: SemanticCategory
    payload: _StorageParameter | _AnnotatedParameter

    def __post_init__(self) -> None:
        if self.kind is SemanticCategory.STORAGE:
            if not isinstance(self.payload, _StorageParameter):
                raise TypeError("storage runtime parameters require storage metadata")
        elif self.kind in {SemanticCategory.VALUE, SemanticCategory.RESOURCE}:
            if not isinstance(self.payload, _AnnotatedParameter):
                raise TypeError("typed runtime parameters require annotation metadata")
        else:
            raise TypeError(f"unsupported runtime parameter category {self.kind!r}")

    @classmethod
    def storage(
        cls,
        dtype: Any,
        shape: tuple[int, ...],
        access: str = "read_write",
        as_view: bool = False,
    ) -> RuntimeParameterDescriptor:
        return cls(
            SemanticCategory.STORAGE,
            _StorageParameter(dtype, tuple(shape), access, as_view),
        )

    @classmethod
    def annotated(
        cls,
        category: SemanticCategory,
        annotation: Any,
        logical: ConcreteType,
    ) -> RuntimeParameterDescriptor:
        return cls(category, _AnnotatedParameter(annotation, logical))

    @property
    def storage_metadata(self) -> _StorageParameter:
        if not isinstance(self.payload, _StorageParameter):
            raise TypeError(f"{self.kind.value} runtime parameter has no storage metadata")
        return self.payload

    @property
    def annotation(self) -> Any:
        if not isinstance(self.payload, _AnnotatedParameter):
            raise TypeError("storage runtime parameter has no canonical runtime annotation")
        return self.payload.annotation

    @property
    def logical(self) -> ConcreteType:
        if isinstance(self.payload, _StorageParameter):
            element = concrete_type_from_annotation(self.payload.dtype)
            return ConcreteType(
                "tensor_view",
                "TensorView",
                (element, self.payload.shape, self.payload.access, "device"),
            )
        return self.payload.logical


def concrete_type_from_annotation(annotation: Any) -> ConcreteType:
    """Translate an evaluated public annotation to its logical DSL type."""

    if annotation is int:
        return ConcreteType("scalar", "i32")
    if annotation is float:
        return ConcreteType("scalar", "f32")
    if annotation is bool:
        return ConcreteType("scalar", "bool")
    if isinstance(annotation, _Scalar):
        return ConcreteType("scalar", annotation.name)
    if isinstance(annotation, type) and getattr(annotation, "__vernon_dsl__", (None, {}))[0] == "struct":
        return ConcreteType("struct", annotation.__name__)
    if not isinstance(annotation, TypeExpr):
        raise TypeError(f"{annotation!r} is not a Vernon runtime annotation")
    if annotation.name == "Sampler":
        return ConcreteType("sampler", "Sampler")
    if annotation.name == "Tuple":
        return ConcreteType(
            "tuple",
            "Tuple",
            tuple(concrete_type_from_annotation(member) for member in annotation.arguments),
        )
    if annotation.name == "Tensor":
        if len(annotation.arguments) != 2 or not isinstance(annotation.arguments[1], tuple):
            raise TypeError("Tensor annotation requires an element type and shape")
        element, shape = annotation.arguments
        return ConcreteType(
            "tensor",
            "Tensor",
            (concrete_type_from_annotation(element), *shape),
        )
    if annotation.name in {"Vector", "Matrix"}:
        rank = 1 if annotation.name == "Vector" else 2
        if len(annotation.arguments) != rank + 1:
            raise TypeError(f"{annotation.name} annotation has an invalid rank")
        return ConcreteType(
            "tensor",
            "Tensor",
            (concrete_type_from_annotation(annotation.arguments[0]), *annotation.arguments[1:]),
        )
    if annotation.name == "TensorView":
        if len(annotation.arguments) != 3 or not isinstance(annotation.arguments[1], tuple):
            raise TypeError("TensorView annotation requires element, shape, and access")
        element, shape, access = annotation.arguments
        return ConcreteType(
            "tensor_view",
            "TensorView",
            (
                concrete_type_from_annotation(element),
                tuple("?" if extent is dyn else extent for extent in shape),
                str(getattr(access, "name", access)),
                "device",
            ),
        )
    if annotation.name == "Texture":
        if len(annotation.arguments) == 2:
            dimension, element = annotation.arguments
            return ConcreteType(
                "texture",
                "Texture",
                (str(dimension), concrete_type_from_annotation(element), "unknown", "sampled"),
            )
        if len(annotation.arguments) == 3:
            dimension, format_name, access = annotation.arguments
            return ConcreteType(
                "texture",
                "Texture",
                (
                    str(dimension),
                    ConcreteType("scalar", "f32"),
                    str(getattr(format_name, "name", format_name)),
                    str(getattr(access, "name", access)),
                ),
            )
        raise TypeError("Texture annotation requires two or three arguments")
    raise TypeError(f"unsupported Vernon runtime annotation {annotation.name!r}")


def runtime_parameter_category(annotation: Any) -> SemanticCategory:
    return runtime_parameter_descriptor(annotation).kind


def runtime_parameter_descriptor(annotation: Any) -> RuntimeParameterDescriptor:
    """Classify an evaluated runtime annotation without retaining a value."""

    logical = concrete_type_from_annotation(annotation)
    category = semantic_category(logical)
    if category is None:
        raise TypeError(f"{annotation!r} is not a runtime parameter type")
    if category is SemanticCategory.STORAGE:
        if not isinstance(annotation, TypeExpr) or annotation.name != "TensorView":
            raise TypeError(f"{annotation!r} has no concrete Module storage descriptor")
        dtype, shape, access = annotation.arguments
        return RuntimeParameterDescriptor.storage(
            dtype,
            tuple(shape),
            str(getattr(access, "name", access)),
            True,
        )
    return RuntimeParameterDescriptor.annotated(category, annotation, logical)


def validate_host_value(annotation: Any, value: Any, name: str) -> None:
    """Apply the canonical host Value conversion without retaining packed bytes."""

    pack_host_value(annotation, value, name)


def resolved_annotations(function: Any) -> dict[str, Any]:
    try:
        return inspect.get_annotations(function, eval_str=True)
    except (NameError, TypeError) as error:
        raise TypeError(f"cannot resolve runtime annotations: {error}") from None


__all__ = [
    "RuntimeParameterDescriptor",
    "concrete_type_from_annotation",
    "resolved_annotations",
    "runtime_parameter_category",
    "runtime_parameter_descriptor",
    "validate_host_value",
]
