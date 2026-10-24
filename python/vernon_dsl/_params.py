"""Host-side `@params` containers and the dotted-leaf view every entry boundary uses."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Annotated, Any, get_args, get_origin

from ._markers import is_params
from .host_values import declared_fields


def _fields(cls: type[Any]) -> dict[str, Any]:
    """A field's interface metadata places its leaf in the shader; the host sees only the value type."""
    return {
        name: get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation
        for name, annotation in declared_fields(cls).items()
    }


def make_params(cls: type[Any]) -> type[Any]:
    if cls.__bases__ != (object,):
        raise TypeError("@params classes do not support inheritance")
    fields = declared_fields(cls)

    def __init__(self: Any, *args: Any, **values: Any) -> None:
        if args:
            raise TypeError(f"{cls.__name__} binds its fields by keyword")
        unexpected = sorted(set(values) - set(fields))
        if unexpected:
            raise TypeError(f"{cls.__name__} has no field(s) named {', '.join(unexpected)}")
        missing = [name for name in fields if name not in values]
        if missing:
            raise TypeError(f"{cls.__name__} missing field value(s): {', '.join(missing)}")
        for name in fields:
            object.__setattr__(self, name, values[name])

    def __setattr__(self: Any, name: str, value: Any) -> None:
        del value
        raise AttributeError(f"{cls.__name__} params are immutable; cannot assign '{name}'")

    def __repr__(self: Any) -> str:
        return f"{cls.__name__}({', '.join(f'{name}={getattr(self, name)!r}' for name in fields)})"

    cls.__init__ = __init__  # type: ignore[method-assign]
    cls.__setattr__ = __setattr__  # type: ignore[method-assign]
    cls.__repr__ = __repr__  # type: ignore[method-assign]
    return cls


def params_leaf_annotations(name: str, annotation: Any) -> Iterator[tuple[str, Any]]:
    """Yields `(dotted name, leaf annotation)` in field order; a non-group parameter is its own leaf."""
    if not is_params(annotation):
        yield name, annotation
        return
    for field, field_annotation in _fields(annotation).items():
        yield from params_leaf_annotations(f"{name}.{field}", field_annotation)


def params_leaf_items(name: str, annotation: Any, value: Any) -> Iterator[tuple[str, Any, Any]]:
    """Yields `(dotted name, leaf annotation, leaf value)` in field order; a non-group argument is its own leaf."""
    if not is_params(annotation):
        yield name, annotation, value
        return
    if type(value) is not annotation:
        raise TypeError(f"argument {name!r} must be a {annotation.__name__} params")
    for field, field_annotation in _fields(annotation).items():
        yield from params_leaf_items(f"{name}.{field}", field_annotation, getattr(value, field))


def params_at(annotations: Mapping[str, Any], path: str) -> Any | None:
    """Returns the params class that dotted `path` names through its root's annotation, if any."""
    root, *components = path.split(".")
    annotation = annotations.get(root)
    for component in components:
        if not is_params(annotation):
            return None
        annotation = _fields(annotation).get(component)
    return annotation if is_params(annotation) else None


def assemble_params(name: str, annotation: Any, leaves: Mapping[str, Any]) -> Any:
    """Rebuilds the argument `name` from its dotted leaves, the inverse of `params_leaf_items`."""
    if not is_params(annotation):
        if name not in leaves:
            raise TypeError(f"params leaf {name!r} has no value")
        return leaves[name]
    return annotation(
        **{
            field: assemble_params(f"{name}.{field}", field_annotation, leaves)
            for field, field_annotation in _fields(annotation).items()
        }
    )


def expand_params_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Expands params containers among keyword arguments into their dotted leaves."""
    expanded: dict[str, Any] = {}
    for name, value in arguments.items():
        annotation = type(value) if is_params(type(value)) else None
        for leaf, _, leaf_value in params_leaf_items(name, annotation, value):
            if leaf in expanded:
                raise TypeError(f"argument {leaf!r} is bound twice")
            expanded[leaf] = leaf_value
    return expanded


__all__ = [
    "assemble_params",
    "params_at",
    "params_leaf_annotations",
    "params_leaf_items",
    "expand_params_arguments",
    "is_params",
    "make_params",
]
