"""Canonical contracts for compiler-recognized shader-only APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .language.stage_registry import GRAPHICS_STAGES


@dataclass(frozen=True)
class TypeContract:
    kind: str
    name: str
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class BuiltinContract:
    uses: frozenset[tuple[str, str]]
    type: TypeContract

    def supports(self, stage: str, direction: str) -> bool:
        return (stage, direction) in self.uses


@dataclass(frozen=True)
class GeneratedInterfaceContract:
    stage: str
    type: TypeContract
    interface: str
    builtin: str | None


@dataclass(frozen=True)
class TextureSamplingContract:
    explicit_sampler: bool
    has_lod: bool
    stages: frozenset[str]

    @property
    def sampler_mode(self) -> str:
        return "explicit" if self.explicit_sampler else "implicit"


F32 = TypeContract("scalar", "f32")
U32 = TypeContract("scalar", "u32")
BOOL = TypeContract("scalar", "bool")

BUILTIN_CONTRACTS = {
    "position": BuiltinContract(
        frozenset({("vertex", "output")}),
        TypeContract("tensor", "f32", (4,)),
    ),
    "vertex_index": BuiltinContract(frozenset({("vertex", "input")}), U32),
    "instance_index": BuiltinContract(frozenset({("vertex", "input")}), U32),
    "frag_coord": BuiltinContract(
        frozenset({("fragment", "input")}),
        TypeContract("tensor", "f32", (4,)),
    ),
    "front_facing": BuiltinContract(frozenset({("fragment", "input")}), BOOL),
    "global_invocation_id": BuiltinContract(
        frozenset({("compute", "input")}),
        TypeContract("tensor", "u32", (3,)),
    ),
    "local_invocation_id": BuiltinContract(
        frozenset({("compute", "input")}),
        TypeContract("tensor", "u32", (3,)),
    ),
    "workgroup_id": BuiltinContract(
        frozenset({("compute", "input")}),
        TypeContract("tensor", "u32", (3,)),
    ),
}

GENERATED_INTERFACE_CONTRACTS = {
    "resolution": GeneratedInterfaceContract("fragment", TypeContract("tensor", "f32", (2,)), "uniform", None),
    "fragment_coord": GeneratedInterfaceContract(
        "fragment", TypeContract("tensor", "f32", (4,)), "input", "frag_coord"
    ),
    "front_facing": GeneratedInterfaceContract("fragment", BOOL, "input", "front_facing"),
    "vertex_id": GeneratedInterfaceContract("vertex", U32, "input", "vertex_index"),
    "instance_id": GeneratedInterfaceContract("vertex", U32, "input", "instance_index"),
}

ATOMIC_OPERATION_NAMES = frozenset(
    {
        "atomic_add",
        "atomic_exchange",
        "atomic_max",
        "atomic_min",
    }
)

DEVICE_ONLY_TYPE_NAMES = frozenset(
    {
        "Sampler",
        "TensorStorage",
        "TensorView",
        "Texture",
        "builtin",
        "resource",
    }
)
DEVICE_ONLY_OPERATION_NAMES = frozenset(
    {
        *GENERATED_INTERFACE_CONTRACTS,
        *ATOMIC_OPERATION_NAMES,
        "storage_barrier",
        "texture_sample",
        "texture_size",
        "workgroup_array",
        "workgroup_barrier",
    }
)

_FRAGMENT_ONLY = frozenset({"fragment"})
_EXPLICIT_LOD_STAGES = GRAPHICS_STAGES | {"compute"}


def texture_sampling_contract(argument_kinds: Sequence[str]) -> TextureSamplingContract | None:
    """Classify a texture_sample overload from its lowered argument kinds."""
    if not argument_kinds or argument_kinds[0] != "texture":
        return None
    if len(argument_kinds) == 2:
        return TextureSamplingContract(False, False, _FRAGMENT_ONLY)
    if len(argument_kinds) == 3:
        explicit_sampler = argument_kinds[1] == "sampler"
        return TextureSamplingContract(
            explicit_sampler, not explicit_sampler, _FRAGMENT_ONLY if explicit_sampler else GRAPHICS_STAGES
        )
    if len(argument_kinds) == 4 and argument_kinds[1] == "sampler":
        return TextureSamplingContract(True, True, _EXPLICIT_LOD_STAGES)
    return None
