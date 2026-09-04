from __future__ import annotations

import dataclasses
import hashlib
import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast, get_args, get_origin

from ..bundle import PipelineCompileError, canonical_json, make_target_options
from ..compiler import Compiler, FrontendCompileRequest
from ..frontend.runtime_types import (
    RuntimeParameterDescriptor,
    runtime_parameter_descriptor,
)
from ..language.stage_registry import validate_graphics_topology
from ..operation_graph import ProgramControlDescriptor
from ..render import (
    DrawCommand,
    DynamicState,
    GraphicsPipelineState,
    PrimitiveTopology,
    RenderPass,
    graphics_state,
    lines,
    points,
    triangles,
)
from .resource_common import _session_state
from .tensor import TensorStorage, TensorView


def _runtime_annotation_base(annotation: Any) -> Any:
    base = annotation
    while True:
        if get_origin(base) is Annotated:
            base = get_args(base)[0]
            continue
        if getattr(base, "name", None) != "When":
            return base
        if len(base.arguments) != 2:
            raise TypeError("When runtime annotation requires a feature and a type")
        base = base.arguments[1]


@dataclass(frozen=True)
class _CompiledPipeline:
    identity: str
    invocation: Any
    specialization: Any


class Pipeline:
    """Graphics pipeline compiled and executed as a one-node Program."""

    __vernon_pipeline__ = True

    def __init__(
        self,
        *stages: Any,
        state: GraphicsPipelineState | None = None,
        features: Iterable[str] = (),
    ):
        kinds = cast(
            tuple[str, ...],
            tuple(getattr(stage, "__vernon_dsl__", (None,))[0] for stage in stages),
        )
        validate_graphics_topology(kinds)
        feature_values = tuple(features)
        if any(not isinstance(value, str) or not value for value in feature_values):
            raise TypeError("pipeline features must be non-empty strings")
        self._features = tuple(sorted(set(feature_values)))
        self._graphics_state = graphics_state() if state is None else state
        if not isinstance(self._graphics_state, GraphicsPipelineState):
            raise TypeError("state must be a GraphicsPipelineState")
        self._topology = self._graphics_state.topology
        self._stages = stages
        self._vertex = stages[0]
        self._fragment = stages[-1]
        self._specializations: dict[str, _CompiledPipeline] = {}
        self.compile_count = 0

    def _frontends(self) -> tuple[Any, ...]:
        compiler = Compiler()
        result = []
        for stage in self._stages:
            function = stage.function
            source = Path(inspect.getsourcefile(function) or "").resolve()
            result.append(compiler.compile_request(FrontendCompileRequest(source, function.__name__, self._features)))
        return tuple(result)

    def _parameter_types(
        self,
        arguments: Mapping[str, Any],
        frontends: tuple[Any, ...],
        render_pass: RenderPass,
        draw: DrawCommand | None,
        dynamic_state: DynamicState | None,
    ) -> dict[str, Any]:
        reflected: dict[str, tuple[Any, Any]] = {}
        annotations: dict[str, Any] = {}
        for stage, frontend in zip(self._stages, frontends, strict=True):
            function = stage.function
            entry = next(
                (typed for typed in frontend.typed_functions if typed.source.name == function.__name__),
                None,
            )
            if entry is None:
                raise RuntimeError(f"compiled graphics stage {function.__name__!r} has no typed entry")
            stage_annotations = inspect.get_annotations(function, eval_str=True)
            for parameter in entry.parameters:
                interface = {item.kind for item in parameter.interface}
                if parameter.builtin is not None or "implicit" in interface:
                    continue
                if stage.kind == "fragment" and "varying" in interface:
                    continue
                previous = reflected.get(parameter.name)
                if previous is not None and previous[0].type != parameter.type:
                    raise TypeError(f"graphics stages disagree on parameter {parameter.name!r}")
                reflected[parameter.name] = (parameter, stage)
                annotation = stage_annotations.get(parameter.name)
                if annotation is not None:
                    previous_annotation = annotations.get(parameter.name)
                    if previous_annotation is not None and previous_annotation != annotation:
                        raise TypeError(f"graphics stages disagree on annotation for {parameter.name!r}")
                    annotations[parameter.name] = annotation
        if set(reflected) != set(arguments):
            missing = set(reflected) - set(arguments)
            unexpected = set(arguments) - set(reflected)
            if missing:
                raise TypeError(f"missing pipeline argument(s): {', '.join(sorted(missing))}")
            raise TypeError(f"unexpected pipeline argument(s): {', '.join(sorted(unexpected))}")

        result: dict[str, Any] = {}
        for name, value in arguments.items():
            parameter, stage = reflected[name]
            interface = {item.kind for item in parameter.interface}
            is_storage = parameter.type.kind == "tensor_view" or (
                stage.kind == "vertex" and "attribute" in interface and isinstance(value, (TensorStorage, TensorView))
            )
            if is_storage:
                if not isinstance(value, (TensorStorage, TensorView)):
                    raise TypeError(f"graphics storage argument {name!r} requires TensorStorage or TensorView")
                annotation = annotations.get(name)
                if parameter.type.kind == "tensor_view":
                    if annotation is None:
                        raise TypeError(f"graphics storage argument {name!r} requires a runtime annotation")
                    result[name] = runtime_parameter_descriptor(annotation)
                    continue
                base = _runtime_annotation_base(annotation)
                if getattr(base, "name", None) == "Tensor":
                    dtype, cell_shape = base.arguments
                elif getattr(base, "name", None) in {"Vector", "Matrix"}:
                    dtype, *cell_shape = base.arguments
                    cell_shape = tuple(cell_shape)
                elif isinstance(base, type) and getattr(base, "__vernon_dsl__", (None,))[0] == "struct":
                    dtype, cell_shape = base, ()
                else:
                    raise TypeError(f"vertex storage argument {name!r} has no canonical cell type")
                result[name] = RuntimeParameterDescriptor.storage(
                    dtype,
                    (("?", *cell_shape) if "attribute" in interface else parameter.type.arguments[1]),
                    str(parameter.access.value),
                    True,
                )
                continue
            annotation = annotations.get(name)
            if annotation is None:
                raise TypeError(f"graphics argument {name!r} requires a runtime annotation")
            base = _runtime_annotation_base(annotation)
            result[name] = runtime_parameter_descriptor(base)
        result["__render_pass"] = ProgramControlDescriptor("render_pass", RenderPass, render_pass)
        result["__draw"] = ProgramControlDescriptor("draw", DrawCommand | None, draw)
        result["__dynamic_state"] = ProgramControlDescriptor("dynamic_state", DynamicState | None, dynamic_state)
        return result

    def _identity(
        self,
        render_pass: RenderPass,
        parameter_types: Mapping[str, Any],
    ) -> str:
        target = render_pass.target

        state = _session_state()
        options = make_target_options(
            state._architecture.name,
            (
                {"version": state._interactive_glsl_version()}
                if state._architecture in {state.opengl, state.opengles}
                else {}
            ),
        )
        depth = target._depth_attachment()
        return hashlib.sha256(
            canonical_json(
                {
                    "target": tuple(
                        (location, texture.format.name) for location, texture in target._color_attachments()
                    ),
                    "depth": None if depth is None else depth.format.name,
                    "state": repr(self._graphics_state),
                    "features": self._features,
                    "backend": options.spec,
                    "arguments": tuple(
                        (
                            name,
                            repr(descriptor.logical),
                            repr(getattr(descriptor, "access", None)),
                            bool(getattr(descriptor, "as_view", False)),
                        )
                        for name, descriptor in sorted(parameter_types.items())
                        if not isinstance(descriptor, ProgramControlDescriptor)
                    ),
                }
            ).encode()
        ).hexdigest()

    def _compile(
        self,
        arguments: Mapping[str, Any],
        render_pass: RenderPass,
        draw: DrawCommand | None,
        dynamic_state: DynamicState | None,
    ) -> _CompiledPipeline:
        state = _session_state()
        if state._native is None or state._native_runtime is None:
            raise RuntimeError("graphics requires the native Program runtime")
        if state._architecture == state.cpu:
            raise RuntimeError("CPU graphics pipelines require a software rasterizer, which Vernon does not provide")
        frontends = self._frontends()
        parameter_types = self._parameter_types(arguments, frontends, render_pass, draw, dynamic_state)
        identity = self._identity(render_pass, parameter_types)
        cached = self._specializations.get(identity)
        if cached is not None:
            return cached
        from ..program import _one_node_program

        template, invocation, parsed = _one_node_program(
            self,
            parameter_types,
            lambda capture, inputs: capture.capture_graphics(
                self,
                {name: inputs[name] for name in arguments},
                inputs["__render_pass"],
                inputs["__draw"],
                inputs["__dynamic_state"],
                frontends=frontends,
            ),
        )
        from .program_autodiff import compile_program

        try:
            specialization = compile_program(parsed, template)
        except PipelineCompileError as error:
            raise RuntimeError(str(error)) from None
        compiled = _CompiledPipeline(identity, invocation, specialization)
        self._specializations[identity] = compiled
        self.compile_count += 1
        return compiled

    def _invoke(
        self,
        arguments: Mapping[str, Any],
        render_pass: RenderPass,
        draw: DrawCommand | None,
        dynamic_state: DynamicState | None,
    ) -> None:
        compiled = self._compile(arguments, render_pass, draw, dynamic_state)
        inputs = {
            **arguments,
            "__render_pass": render_pass,
            "__draw": draw,
            "__dynamic_state": dynamic_state,
        }
        compiled.specialization.invoke(dataclasses.replace(compiled.invocation, inputs=inputs))

    def __call__(
        self,
        *,
        render_pass: RenderPass,
        draw: DrawCommand | None = None,
        dynamic_state: DynamicState | None = None,
        **arguments: Any,
    ) -> None:
        if not isinstance(render_pass, RenderPass):
            raise TypeError("render_pass must be a RenderPass")
        if draw is not None and not isinstance(draw, DrawCommand):
            raise TypeError("draw must be a DrawCommand or None")
        if dynamic_state is not None and not isinstance(dynamic_state, DynamicState):
            raise TypeError("dynamic_state must be a DynamicState or None")
        self._invoke(dict(arguments), render_pass, draw, dynamic_state)


def pipeline(
    *stages: Any,
    state: GraphicsPipelineState | None = None,
    features: Iterable[str] = (),
) -> Pipeline:
    return Pipeline(*stages, state=state, features=features)


__all__ = ["Pipeline", "PrimitiveTopology", "lines", "pipeline", "points", "triangles"]
