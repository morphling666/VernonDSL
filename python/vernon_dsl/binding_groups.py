"""Host-side `@binding_group` containers and the dotted-leaf view every entry boundary uses."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from ._markers import is_binding_group
from .host_values import declared_fields


def make_binding_group(cls: type[Any]) -> type[Any]:
    if cls.__bases__ != (object,):
        raise TypeError("@binding_group classes do not support inheritance")
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
        raise AttributeError(f"{cls.__name__} binding groups are immutable; cannot assign '{name}'")

    def __repr__(self: Any) -> str:
        return f"{cls.__name__}({', '.join(f'{name}={getattr(self, name)!r}' for name in fields)})"

    cls.__init__ = __init__  # type: ignore[method-assign]
    cls.__setattr__ = __setattr__  # type: ignore[method-assign]
    cls.__repr__ = __repr__  # type: ignore[method-assign]
    return cls


def binding_group_leaf_annotations(name: str, annotation: Any) -> Iterator[tuple[str, Any]]:
    """Yields `(dotted name, leaf annotation)` in field order; a non-group parameter is its own leaf."""
    if not is_binding_group(annotation):
        yield name, annotation
        return
    for field, field_annotation in declared_fields(annotation).items():
        yield from binding_group_leaf_annotations(f"{name}.{field}", field_annotation)


def binding_group_leaf_items(name: str, annotation: Any, value: Any) -> Iterator[tuple[str, Any, Any]]:
    """Yields `(dotted name, leaf annotation, leaf value)` in field order; a non-group argument is its own leaf."""
    if not is_binding_group(annotation):
        yield name, annotation, value
        return
    if type(value) is not annotation:
        raise TypeError(f"argument {name!r} must be a {annotation.__name__} binding group")
    for field, field_annotation in declared_fields(annotation).items():
        yield from binding_group_leaf_items(f"{name}.{field}", field_annotation, getattr(value, field))


def binding_group_at(annotations: Mapping[str, Any], path: str) -> Any | None:
    """Returns the binding-group class that dotted `path` names through its root's annotation, if any."""
    root, *components = path.split(".")
    annotation = annotations.get(root)
    for component in components:
        if not is_binding_group(annotation):
            return None
        annotation = declared_fields(annotation).get(component)
    return annotation if is_binding_group(annotation) else None


def assemble_binding_group(name: str, annotation: Any, leaves: Mapping[str, Any]) -> Any:
    """Rebuilds the argument `name` from its dotted leaves, the inverse of `binding_group_leaf_items`."""
    if not is_binding_group(annotation):
        if name not in leaves:
            raise TypeError(f"binding-group leaf {name!r} has no value")
        return leaves[name]
    return annotation(
        **{
            field: assemble_binding_group(f"{name}.{field}", field_annotation, leaves)
            for field, field_annotation in declared_fields(annotation).items()
        }
    )


def expand_binding_group_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Expands binding-group containers among keyword arguments into their dotted leaves."""
    expanded: dict[str, Any] = {}
    for name, value in arguments.items():
        annotation = type(value) if is_binding_group(type(value)) else None
        for leaf, _, leaf_value in binding_group_leaf_items(name, annotation, value):
            if leaf in expanded:
                raise TypeError(f"argument {leaf!r} is bound twice")
            expanded[leaf] = leaf_value
    return expanded


__all__ = [
    "assemble_binding_group",
    "binding_group_at",
    "binding_group_leaf_annotations",
    "binding_group_leaf_items",
    "expand_binding_group_arguments",
    "is_binding_group",
    "make_binding_group",
]
