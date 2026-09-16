"""Declarative first-order reverse-mode automatic differentiation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

from .bundle import canonical_json

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
    program: object
    transform: ProgramTransformSpec

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

        return ModuleVjpExpression(
            program,
            wrt=BoundarySelection.create(wrt, label="wrt").ordered_paths,
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
        if getattr(program, "__vernon_dsl__", (None,))[0] != "compute":
            raise TypeError("single-entry VJP program must be a compute Kernel")
        if outputs is None:
            raise ValueError("compute VJP requires non-empty writable Storage outputs")
    wrt_selection = BoundarySelection.create(wrt, label="wrt")
    spec = ProgramTransformSpec(
        "vjp",
        wrt_selection.canonical_key,
        output_cotangents=(
            BoundarySelection.create(outputs, label="outputs").canonical_key if outputs is not None else ()
        ),
        planning_policy=planning_policy,
    )
    return ProgramExpression(program, spec)


__all__ = ["BoundarySelection", "ProgramExpression", "ProgramTransformSpec", "vjp"]
