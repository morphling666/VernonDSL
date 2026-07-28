from __future__ import annotations

import ast
from typing import Protocol

from ..shader_contracts import texture_sampling_contract
from .interfaces import GeneratedInterfacePlan
from .lowering_types import DslType, ModuleContext, Value


class ResourceEmitter(Protocol):
    context: ModuleContext
    node: ast.FunctionDef
    stage: str | None
    interface_plan: GeneratedInterfacePlan
    implicit_samplers: dict[str, Value]

    def _intrinsic(self, node: ast.AST, name: str, arguments: list[Value], result_type: DslType) -> Value: ...


def lower_texture_sample(
    emitter: ResourceEmitter,
    node: ast.Call,
    arguments: list[Value],
    result_type: DslType,
) -> Value:
    if emitter.node.name in emitter.context.shared_functions:
        raise emitter.context.error(
            node,
            f"shared function '{emitter.node.name}' uses device-only operation 'texture_sample'",
        )
    sampling = texture_sampling_contract(tuple(argument.type.kind for argument in arguments))
    if sampling is None:
        if len(arguments) == 4:
            raise emitter.context.error(node, "the four-argument texture_sample form requires an explicit sampler")
        raise emitter.context.error(
            node,
            "texture_sample requires texture, optional sampler, coordinates, and optional lod",
        )
    texture = arguments[0]
    texture_source = node.args[0].id if isinstance(node.args[0], ast.Name) else None
    planned_mode = emitter.interface_plan.texture_sampler_modes.get(texture_source) if texture_source else None
    if planned_mode is not None and planned_mode != sampling.sampler_mode:
        raise emitter.context.error(node, "texture sampling overload differs from its interface plan")
    if emitter.stage not in sampling.stages:
        requirement = "fragment shaders" if not sampling.has_lod else "graphics stages"
        raise emitter.context.error(
            node,
            f"texture_sample {'without' if not sampling.has_lod else 'with'} lod is supported only in {requirement}",
        )
    if sampling.explicit_sampler:
        sampler = arguments[1]
        coordinates = arguments[2]
        lod = arguments[3] if sampling.has_lod else None
    else:
        sampler = emitter.implicit_samplers.get(texture.name)
        if sampler is None:
            raise emitter.context.error(node.args[0], "implicitly sampled texture must be a texture entry parameter")
        coordinates = arguments[1]
        lod = arguments[2] if sampling.has_lod else None
    if lod is not None and (lod.type.kind != "scalar" or not lod.type.is_float):
        raise emitter.context.error(node.args[-1], "texture_sample lod must be a floating-point scalar")
    element = texture.type.arguments[1]
    assert isinstance(element, DslType)
    dimension = texture.type.arguments[0]
    coordinate_rank = {"2d": 2, "3d": 3, "cube": 3}[dimension]
    if (
        coordinates.type.kind != "tensor"
        or coordinates.type.arguments != (element, coordinate_rank)
        or not coordinates.type.is_float
    ):
        coordinate_source = node.args[2] if sampling.explicit_sampler else node.args[1]
        raise emitter.context.error(
            coordinate_source,
            f"texture_sample coordinates for a {dimension} texture "
            f"must be a {coordinate_rank}-component floating-point vector",
        )
    operands = [texture, sampler, coordinates]
    if lod is not None:
        operands.append(lod)
    return emitter._intrinsic(node, "texture_sample", operands, result_type)


def lower_texture_size(
    emitter: ResourceEmitter,
    node: ast.Call,
    arguments: list[Value],
    result_type: DslType,
) -> Value:
    if emitter.node.name in emitter.context.shared_functions:
        raise emitter.context.error(
            node,
            f"shared function '{emitter.node.name}' uses device-only operation 'texture_size'",
        )
    if len(arguments) not in {1, 2} or arguments[0].type.kind != "texture":
        raise emitter.context.error(node, "texture_size requires texture and optional lod")
    if emitter.stage not in {"vertex", "fragment"}:
        raise emitter.context.error(node, "texture_size is supported only in graphics stages")
    if len(arguments) == 2 and (arguments[1].type.kind != "scalar" or not arguments[1].type.is_integer):
        raise emitter.context.error(node.args[1], "texture_size lod must be an integer scalar")
    return emitter._intrinsic(node, "texture_size", arguments, result_type)
