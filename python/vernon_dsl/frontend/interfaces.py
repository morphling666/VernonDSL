from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping, Protocol

from ..language.ast_utils import dotted_name
from ..shader_contracts import GENERATED_INTERFACE_CONTRACTS, resource_stage_error, texture_sampling_contract
from .model import ConcreteType
from .type_parser import AnnotatedType


class InterfaceContext(Protocol):
    def error(self, node: ast.AST, message: str) -> Exception: ...


@dataclass(frozen=True)
class ImplicitSamplerPlan:
    texture_name: str
    descriptor_set: int
    binding: int


@dataclass(frozen=True)
class GeneratedInterfacePlan:
    implicit_samplers: tuple[ImplicitSamplerPlan, ...]
    generated_apis: tuple[str, ...]
    texture_sampler_modes: Mapping[str, str]


def plan_generated_interface(
    context: InterfaceContext,
    node: ast.FunctionDef,
    argument_annotations: list[AnnotatedType],
    stage: str | None,
) -> GeneratedInterfacePlan:
    argument_types = {
        argument.arg: annotation.type for argument, annotation in zip(node.args.args, argument_annotations, strict=True)
    }
    argument_metadata = {
        argument.arg: annotation.metadata
        for argument, annotation in zip(node.args.args, argument_annotations, strict=True)
    }
    used_bindings = {
        (int(item.arguments[0]), int(item.arguments[1]))
        for metadata in argument_metadata.values()
        for item in metadata
        if item.kind in {"resource", "uniform"} and len(item.arguments) == 2
    }
    implicit_textures: set[str] = set()
    generated_apis: set[str] = set()
    sampler_modes: dict[str, str] = {}

    for value in ast.walk(node):
        if not isinstance(value, ast.Call):
            continue
        name = (dotted_name(value.func) or "").split(".")[-1]
        if name in GENERATED_INTERFACE_CONTRACTS:
            contract = GENERATED_INTERFACE_CONTRACTS[name]
            if stage != contract.stage:
                raise context.error(value, f"{name}() is available only in {contract.stage} shaders")
            generated_apis.add(name)
        if name != "texture_sample" or not value.args:
            continue
        texture_name = value.args[0].id if isinstance(value.args[0], ast.Name) else None
        if texture_name is None or argument_types.get(texture_name, ConcreteType("void", "void")).kind != "texture":
            continue
        argument_kinds = ["texture"]
        for argument in value.args[1:]:
            argument_type = argument_types.get(argument.id) if isinstance(argument, ast.Name) else None
            argument_kinds.append(argument_type.kind if argument_type is not None else "value")
        sampling = texture_sampling_contract(argument_kinds)
        if sampling is None:
            continue
        if stage is not None and stage not in sampling.stages:
            raise context.error(
                value,
                resource_stage_error("texture_sample", sampling.stages, has_lod=sampling.has_lod),
            )
        previous_mode = sampler_modes.get(texture_name)
        if previous_mode is not None and previous_mode != sampling.sampler_mode:
            raise context.error(
                value,
                "one texture entry parameter cannot be sampled through both "
                "implicit and explicit sampler forms in the same stage",
            )
        sampler_modes[texture_name] = sampling.sampler_mode
        if not sampling.explicit_sampler:
            implicit_textures.add(texture_name)

    sampler_plans: list[ImplicitSamplerPlan] = []
    for texture_name in sorted(implicit_textures):
        resource = next((item for item in argument_metadata[texture_name] if item.kind == "resource"), None)
        descriptor_set = int(resource.arguments[0]) if resource is not None else 0
        binding = 0
        while (descriptor_set, binding) in used_bindings:
            binding += 1
        used_bindings.add((descriptor_set, binding))
        sampler_plans.append(ImplicitSamplerPlan(texture_name, descriptor_set, binding))

    return GeneratedInterfacePlan(
        tuple(sampler_plans),
        tuple(name for name in GENERATED_INTERFACE_CONTRACTS if name in generated_apis),
        sampler_modes,
    )
