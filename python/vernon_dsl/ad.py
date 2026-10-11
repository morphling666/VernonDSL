"""Declarative first-order reverse-mode automatic differentiation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable

from ._markers import dsl_kind
from .bundle import canonical_json

if TYPE_CHECKING:
    from .types import SpecializationAssignment

_PATH = re.compile(r"^[A-Za-z_]\w*(?:\.(?:[A-Za-z_]\w*|\d+))*$", re.ASCII)


def _capability_diagnostic(name: str) -> str:
    capability = import_module("vernon_dsl._native")._program_capability(name)
    return f"{capability['code']}: {capability['diagnostic']}"


def _ordered_paths(values: tuple[str, ...] | list[str], *, label: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError(f"{label} must be a non-empty tuple or list of canonical source paths")
    paths = tuple(values)
    if any(not isinstance(path, str) or not _PATH.fullmatch(path) for path in paths):
        raise ValueError(f"{label} entries must be canonical source paths")
    if len(set(paths)) != len(paths):
        raise ValueError(f"{label} paths must be unique")
    return paths


@dataclass(frozen=True)
class BoundarySelection:
    ordered_paths: tuple[str, ...]

    @classmethod
    def create(cls, values: tuple[str, ...] | list[str], *, label: str) -> BoundarySelection:
        return cls(_ordered_paths(values, label=label))

    @property
    def canonical_key(self) -> tuple[str, ...]:
        return tuple(sorted(self.ordered_paths))

    def expand_binding_groups(self, annotations: Mapping[str, Any], *, role: str) -> BoundarySelection:
        """Replaces each path that names a binding group with that group's leaves for `role`, in field order.

        `wrt` selects every differentiable leaf, `outputs` every writable floating TensorView. Expansion runs before
        the paths become the transform identity, so cotangents and cooked manifests see only leaf paths. Paths that
        do not name a group are kept for the compiler to validate.
        """
        from .binding_groups import binding_group_at

        expanded: dict[str, None] = {}
        for path in self.ordered_paths:
            group = binding_group_at(annotations, path)
            if group is None:
                expanded[path] = None
                continue
            leaves = tuple(_group_vjp_leaves(path, group, role))
            if not leaves:
                kind = "differentiable" if role == "wrt" else "writable floating TensorView"
                raise ValueError(f"VJP {role} path {path!r} names a binding group with no {kind} leaves")
            expanded.update(dict.fromkeys(leaves))
        return BoundarySelection(tuple(expanded))


def _tangent_leaf_paths(path: str, schema: Any) -> Iterator[str]:
    from .host_values import TangentProduct, TangentZero

    if isinstance(schema, TangentZero):
        return
    if isinstance(schema, TangentProduct):
        for name, child in schema.children:
            yield from _tangent_leaf_paths(f"{path}.{name}", child)
    else:
        yield path


def _group_vjp_leaves(path: str, group: type[Any], role: str) -> Iterator[str]:
    from .binding_groups import binding_group_leaf_annotations
    from .frontend.model import SemanticCategory
    from .frontend.runtime_types import runtime_parameter_category, storage_parameter_constraint
    from .host_values import TangentZero, tangent_schema

    for leaf, annotation in binding_group_leaf_annotations(path, group):
        storage = storage_parameter_constraint(annotation)
        if storage is not None:
            writable = storage.access in {"write", "read_write"}
            if (role == "wrt" or writable) and not isinstance(tangent_schema(storage.dtype), TangentZero):
                yield leaf
        elif role == "wrt" and runtime_parameter_category(annotation) is SemanticCategory.VALUE:
            # Struct and Tuple Values expand to their members because a Kernel VJP selects Value leaves.
            yield from _tangent_leaf_paths(leaf, tangent_schema(annotation))


@dataclass(frozen=True)
class ProgramTransformSpec:
    kind: str
    wrt: tuple[str, ...]
    output_cotangents: tuple[str, ...] = ()
    gradient_policy: str = "f16:f32,f32:f32,f64:f64"
    accumulation_policy: str = "fresh"
    tape_policy: str = "bounded"
    planning_policy: str = "min_memory"
    derivative_rules_version: int = 1

    def __post_init__(self) -> None:
        if self.kind != "vjp":
            raise ValueError(f"unsupported program transform {self.kind!r}")
        if self.gradient_policy != "f16:f32,f32:f32,f64:f64":
            raise ValueError("unsupported gradient element-type policy")
        if self.accumulation_policy != "fresh":
            raise ValueError("unsupported gradient accumulation policy")
        if self.tape_policy != "bounded":
            raise ValueError("unsupported autodiff tape policy")
        if self.planning_policy not in {"min_memory", "balanced", "min_runtime"}:
            raise ValueError("autodiff planning policy must be 'min_memory', 'balanced', or 'min_runtime'")
        if self.derivative_rules_version != 1:
            raise ValueError("unsupported derivative rules version")
        object.__setattr__(self, "wrt", BoundarySelection.create(self.wrt, label="wrt").ordered_paths)
        if self.output_cotangents:
            object.__setattr__(
                self,
                "output_cotangents",
                BoundarySelection.create(self.output_cotangents, label="outputs").ordered_paths,
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "wrt": list(self.wrt),
            "output_cotangents": list(self.output_cotangents),
            "gradient_policy": self.gradient_policy,
            "accumulation_policy": self.accumulation_policy,
            "tape_policy": self.tape_policy,
            "planning_policy": self.planning_policy,
            "derivative_rules_version": self.derivative_rules_version,
        }
        return result

    @property
    def identity(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProgramExpression:
    """A VJP over one Kernel or Pipeline, keeping `wrt` and `outputs` as written.

    A path that names a binding group expands per specialization key, because a type specialization decides which
    group, and therefore which leaves, a parameter has. Construction expands against the empty key so a selection
    that no key could satisfy fails at declaration.
    """

    program: Any
    wrt: BoundarySelection
    outputs: BoundarySelection | None
    planning_policy: str

    def __post_init__(self) -> None:
        self.transform  # noqa: B018

    @property
    def transform(self) -> ProgramTransformSpec:
        """The transform for the empty key, which direct execution always uses."""

        return self.transform_for(())

    def transform_for(self, specializations: tuple[SpecializationAssignment, ...]) -> ProgramTransformSpec:
        from ._runtime.pipeline import Pipeline
        from .frontend.runtime_types import resolved_annotations

        # Graphics entries take part in no Program VJP, so only a Kernel's groups expand.
        annotations = (
            {} if isinstance(self.program, Pipeline) else resolved_annotations(self.program._function, specializations)
        )
        return ProgramTransformSpec(
            "vjp",
            self.wrt.expand_binding_groups(annotations, role="wrt").canonical_key,
            output_cotangents=(
                self.outputs.expand_binding_groups(annotations, role="outputs").canonical_key
                if self.outputs is not None
                else ()
            ),
            planning_policy=self.planning_policy,
        )

    def __call__(
        self,
        *arguments: Any,
        grid: tuple[int, int, int] | None = None,
    ) -> tuple[Any, Callable[[Any | None], dict[str, Any]]]:
        from ._runtime.autodiff import execute_direct_vjp

        return execute_direct_vjp(self, arguments, grid)


def vjp(
    program: Any,
    *,
    wrt: tuple[str, ...] | list[str],
    outputs: tuple[str, ...] | list[str] | None = None,
    planning_policy: str = "min_memory",
) -> Any:
    """Describe a VJP transform that may be cooked or directly executed on CPU."""

    from ._runtime.pipeline import Pipeline
    from .module import Module

    if isinstance(program, Module):
        from .program import ModuleVjpExpression

        annotations = type(program)._module_definition.resolved_annotations
        return ModuleVjpExpression(
            program,
            wrt=BoundarySelection.create(wrt, label="wrt").expand_binding_groups(annotations, role="wrt").ordered_paths,
            outputs=(BoundarySelection.create(outputs, label="outputs").ordered_paths if outputs is not None else None),
            planning_policy=planning_policy,
        )
    if isinstance(program, ProgramExpression):
        raise TypeError("higher-order program transforms are not supported")
    if isinstance(program, tuple):
        raise TypeError("graphics programs must use vd.pipeline(...), not a tuple of entry functions")
    if isinstance(program, Pipeline):
        if outputs is not None:
            raise ValueError("graphics VJP does not accept compute Storage outputs")
    else:
        if dsl_kind(program) != "compute":
            raise TypeError("single-entry VJP program must be a compute Kernel")
        if outputs is None:
            raise ValueError("compute VJP requires non-empty writable Storage outputs")
    return ProgramExpression(
        program,
        BoundarySelection.create(wrt, label="wrt"),
        BoundarySelection.create(outputs, label="outputs") if outputs is not None else None,
        planning_policy,
    )


__all__ = ["BoundarySelection", "ProgramExpression", "ProgramTransformSpec", "vjp"]
