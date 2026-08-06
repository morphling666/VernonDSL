"""Declarative first-order reverse-mode automatic differentiation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .bundle import canonical_json
from .language.stage_registry import GRAPHICS_STAGES, validate_graphics_topology

_PATH = re.compile(r"^[A-Za-z_]\w*(?:\.(?:[A-Za-z_]\w*|\d+))*$")
_RULE_NAMES = ("rasterization", "visibility", "depth", "blend", "texture")


def _canonical_paths(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError("wrt must be a non-empty tuple or list of canonical source paths")
    paths = tuple(values)
    if any(not isinstance(path, str) or not _PATH.fullmatch(path) for path in paths):
        raise ValueError("wrt entries must be canonical source paths")
    if len(set(paths)) != len(paths):
        raise ValueError("wrt paths must be unique")
    return tuple(sorted(paths))


@dataclass(frozen=True)
class RuleSet:
    id: str
    rules: Mapping[str, Callable[..., Any]]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("rule set id must be a non-empty string")
        unknown = set(self.rules) - set(_RULE_NAMES)
        if unknown:
            raise TypeError("unknown VJP rule(s): " + ", ".join(sorted(unknown)))
        if any(not callable(rule) for rule in self.rules.values()):
            raise TypeError("VJP rules must be callable")
        object.__setattr__(self, "rules", MappingProxyType(dict(sorted(self.rules.items()))))

    @property
    def identity(self) -> dict[str, Any]:
        return {"id": self.id, "rules": sorted(self.rules)}

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.identity).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProgramTransformSpec:
    kind: str
    wrt: tuple[str, ...]
    rule_set: str | None = None
    rule_set_identity: str | None = None
    output_cotangents: tuple[str, ...] = ()
    gradient_policy: str = "f16:f32,f32:f32,f64:f64"
    accumulation_policy: str = "fresh"
    tape_policy: str = "bounded"
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
        if self.derivative_rules_version != 1:
            raise ValueError("unsupported derivative rules version")
        if (self.rule_set is None) != (self.rule_set_identity is None):
            raise ValueError("rule set name and identity must be provided together")
        object.__setattr__(self, "wrt", _canonical_paths(self.wrt))
        object.__setattr__(self, "output_cotangents", tuple(sorted(self.output_cotangents)))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "wrt": list(self.wrt),
            "output_cotangents": list(self.output_cotangents),
            "gradient_policy": self.gradient_policy,
            "accumulation_policy": self.accumulation_policy,
            "tape_policy": self.tape_policy,
            "derivative_rules_version": self.derivative_rules_version,
        }
        if self.rule_set is not None:
            result["rule_set"] = self.rule_set
            result["rule_set_identity"] = self.rule_set_identity
        return result

    @property
    def identity(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProgramExpression:
    program: Callable[..., Any] | tuple[Callable[..., Any], ...]
    transform: ProgramTransformSpec
    rules: RuleSet | None = None

    def __call__(
        self,
        *arguments: Any,
        grid: tuple[int, int, int] | None = None,
    ) -> tuple[Any, Callable[[Any | None], dict[str, Any]]]:
        from ._runtime.autodiff import execute_direct_vjp

        return execute_direct_vjp(self, arguments, grid)


def rule_set(*, id: str, **rules: Callable[..., Any]) -> RuleSet:
    """Declare an immutable, versioned graphics VJP rule set."""

    return RuleSet(id, rules)


def vjp(
    program: Callable[..., Any] | tuple[Callable[..., Any], ...],
    *,
    wrt: tuple[str, ...] | list[str],
    rules: RuleSet | None = None,
) -> ProgramExpression:
    """Describe a VJP transform that may be cooked or directly executed on CPU."""

    if isinstance(program, ProgramExpression):
        raise TypeError("higher-order program transforms are not supported")
    if isinstance(program, tuple):
        if not program:
            raise ValueError("graphics program must contain at least one stage")
        kinds = [getattr(value, "__vernon_dsl__", (None,))[0] for value in program]
        if any(kind not in GRAPHICS_STAGES for kind in kinds):
            raise TypeError("graphics VJP program must contain only graphics entry stages")
        validate_graphics_topology(kinds)
        if rules is None:
            raise ValueError("graphics VJP requires a named custom rule set")
    else:
        if getattr(program, "__vernon_dsl__", (None,))[0] != "compute":
            raise TypeError("single-entry VJP program must be a compute Kernel")
        if rules is not None:
            raise ValueError("compute VJP does not accept graphics custom rules")
    if rules is not None and not isinstance(rules, RuleSet):
        raise TypeError("rules must be declared with vd.ad.rule_set")
    spec = ProgramTransformSpec(
        "vjp",
        tuple(wrt),
        rule_set=rules.id if rules is not None else None,
        rule_set_identity=rules.digest if rules is not None else None,
    )
    return ProgramExpression(program, spec, rules)


__all__ = ["ProgramExpression", "ProgramTransformSpec", "RuleSet", "rule_set", "vjp"]
