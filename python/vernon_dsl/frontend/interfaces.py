from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from ..language.ast_utils import dotted_name
from ..language.stage_registry import GRAPHICS_STAGES
from ..shader_contracts import GENERATED_INTERFACE_CONTRACTS
from .model import ConcreteType, ResourceEffect, TypedEffect, TypedFunctionInstance, resource_sampler_modes
from .type_parser import AnnotatedType


class InterfaceContext(Protocol):
    def error(self, node: ast.AST, message: str) -> Exception: ...


@dataclass(frozen=True)
class ImplicitSamplerPlan:
    """An implicit sampler argument; its set and binding are assigned by the compiler's graphics binding planner."""

    texture_name: str


@dataclass(frozen=True)
class GeneratedInterfacePlan:
    implicit_samplers: tuple[ImplicitSamplerPlan, ...]
    generated_apis: tuple[str, ...]
    texture_sampler_modes: Mapping[str, str]


def _sampled_textures(argument_types: Mapping[str, ConcreteType]) -> set[str]:
    return {
        name
        for name, argument in argument_types.items()
        if argument.kind == "texture" and argument.arguments[3] == "sampled"
    }


def _read_textures(argument_types: Mapping[str, ConcreteType], effects: tuple[TypedEffect, ...]) -> set[str]:
    """Sampled textures that the entry reaches, directly or through helpers."""
    sampled = _sampled_textures(argument_types)
    return {effect.owner for effect in effects if isinstance(effect, ResourceEffect) and effect.owner in sampled}


def _entry_argument_types(entry: TypedFunctionInstance) -> dict[str, ConcreteType]:
    return {parameter.name: parameter.type for parameter in entry.parameters}


def read_sampled_textures(entry: TypedFunctionInstance) -> frozenset[str]:
    return frozenset(_read_textures(_entry_argument_types(entry), entry.effects))


def unread_sampled_textures(entry: TypedFunctionInstance) -> frozenset[str]:
    argument_types = _entry_argument_types(entry)
    return frozenset(_sampled_textures(argument_types) - _read_textures(argument_types, entry.effects))


def _unpaired_textures(
    argument_types: Mapping[str, ConcreteType],
    effects: tuple[TypedEffect, ...],
    sampler_modes: Mapping[str, str],
    read_elsewhere: frozenset[str],
) -> set[str]:
    """Sampled textures that no explicit sampler pairs with. Those bind as combined image-samplers, so each needs an
    implicit sampler even when it is only fetched or measured. One the entry does not read needs one too, because a
    binding-group leaf stays an endpoint of every Stage of its group, unless another Stage of the pipeline reads it
    and so pairs it there."""
    read = _read_textures(argument_types, effects)
    unread = _sampled_textures(argument_types) - read - read_elsewhere
    return {name for name in read | unread if sampler_modes.get(name) != "explicit"}


def plan_generated_interface(
    context: InterfaceContext,
    node: ast.FunctionDef,
    argument_annotations: list[AnnotatedType],
    stage: str | None,
    effects: tuple[TypedEffect, ...],
    read_elsewhere: frozenset[str] = frozenset(),
) -> GeneratedInterfacePlan:
    argument_types = {
        argument.arg: annotation.type for argument, annotation in zip(node.args.args, argument_annotations, strict=True)
    }
    sampler_modes, _ = resource_sampler_modes(effects)
    implicit_textures = (
        _unpaired_textures(argument_types, effects, sampler_modes, read_elsewhere)
        if stage in GRAPHICS_STAGES
        else set()
    )

    generated_apis: set[str] = set()
    for value in ast.walk(node):
        if not isinstance(value, ast.Call):
            continue
        name = (dotted_name(value.func) or "").split(".")[-1]
        if name in GENERATED_INTERFACE_CONTRACTS:
            contract = GENERATED_INTERFACE_CONTRACTS[name]
            if stage != contract.stage:
                raise context.error(value, f"{name}() is available only in {contract.stage} shaders")
            generated_apis.add(name)

    return GeneratedInterfacePlan(
        tuple(ImplicitSamplerPlan(texture_name) for texture_name in sorted(implicit_textures)),
        tuple(name for name in GENERATED_INTERFACE_CONTRACTS if name in generated_apis),
        sampler_modes,
    )
