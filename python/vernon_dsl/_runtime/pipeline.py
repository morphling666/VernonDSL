from __future__ import annotations

import dataclasses
import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast, get_args, get_origin

from ..bundle import canonical_json, make_target_options
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
    GraphicsTargetFormats,
    PrimitiveTopology,
    RenderPass,
    graphics_state,
    lines,
    points,
    triangles,
)
from ..types import (
    Specialization,
    SpecializationAssignment,
    dyn,
    specialization_key,
    specialization_key_data,
)
from .session import _ArtifactCache, _invocation_context, _session_state, _SessionArtifactCache, cpu, opengl, opengles
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


def _fragment_output_count(stages: tuple[Any, ...], frontends: tuple[Any, ...]) -> int:
    for stage, frontend in zip(stages, frontends, strict=True):
        if stage.kind != "fragment":
            continue
        entry = next(
            (function for function in frontend.typed_functions if function.source.name == stage.function.__name__),
            None,
        )
        if entry is None:
            raise RuntimeError(f"compiled graphics stage {stage.function.__name__!r} has no typed entry")
        result = entry.result_type
        if result is None or result.kind == "void":
            return 0
        if result.kind == "struct":
            fields = next((fields for name, fields in frontend.structs if name == result.name), None)
            if fields is None:
                raise RuntimeError(f"fragment result struct {result.name!r} has no typed definition")
            return len(fields)
        if result.kind == "tuple":
            return len(result.arguments)
        return 1
    raise RuntimeError("graphics pipeline has no fragment stage")


def _validate_storage_argument_shape(
    name: str,
    descriptor: RuntimeParameterDescriptor,
    value: TensorStorage | TensorView,
) -> None:
    declared = descriptor.storage_metadata.shape
    actual = tuple(value.shape)
    if len(declared) != len(actual) or any(
        extent != "?" and extent != actual_extent for extent, actual_extent in zip(declared, actual, strict=True)
    ):
        raise ValueError(f"graphics argument {name!r} expects shape {declared}, got {actual}")


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
        targets: GraphicsTargetFormats | None = None,
        specializations: Mapping[Specialization, object] | None = None,
    ):
        kinds = cast(
            tuple[str, ...],
            tuple(getattr(stage, "__vernon_dsl__", (None,))[0] for stage in stages),
        )
        validate_graphics_topology(kinds)
        self._variant = specialization_key(specializations)
        if any(assignment.type != "bool" for assignment in self._variant):
            raise TypeError("graphics pipeline specializations must be Boolean")
        self._graphics_state = graphics_state() if state is None else state
        if not isinstance(self._graphics_state, GraphicsPipelineState):
            raise TypeError("state must be a GraphicsPipelineState")
        if targets is not None and not isinstance(targets, GraphicsTargetFormats):
            raise TypeError("targets must be a GraphicsTargetFormats")
        self._targets = targets
        self._topology = self._graphics_state.topology
        self._stages = stages
        self._vertex = stages[0]
        self._fragment = stages[-1]
        self._cache = _SessionArtifactCache()
        self._frontend_cache = _ArtifactCache()
        self.compile_count = 0

    def _frontends(
        self,
        specializations: tuple[SpecializationAssignment, ...] | None = None,
    ) -> tuple[Any, ...]:
        selected = self._variant if specializations is None else specializations
        return self._frontend_cache.get_or_create(
            selected,
            lambda: self._compile_frontends(selected),
            self._frontends_current,
        )

    def _compile_frontends(
        self,
        specializations: tuple[SpecializationAssignment, ...],
    ) -> tuple[Any, ...]:
        compiler = Compiler()
        result = []
        for stage in self._stages:
            function = stage.function
            source = Path(inspect.getsourcefile(function) or "").resolve()
            result.append(compiler.compile_request(FrontendCompileRequest(source, function.__name__, specializations)))
        return tuple(result)

    @staticmethod
    def _frontends_current(frontends: tuple[Any, ...]) -> bool:
        try:
            return all(
                hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                for frontend in frontends
                for path, digest in frontend.semantic_inputs.get("dependencies", ())
            )
        except OSError:
            return False

    def _parameter_types(
        self,
        arguments: Mapping[str, Any] | None,
        frontends: tuple[Any, ...],
        render_pass: RenderPass | None,
        draw: DrawCommand | None,
        dynamic_state: DynamicState | None,
    ) -> dict[str, Any]:
        """Type this pipeline's parameters from stage reflection, with or without invocation values.

        Cooking has no values and no render target, and passes None for all four. Everything the manifest needs comes
        from the reflected signature and the annotations; the values only let a live call check itself and resolve
        the one case reflection leaves open, an attribute parameter fed something other than a vertex buffer.
        """

        if render_pass is not None:
            fragment_outputs = _fragment_output_count(self._stages, frontends)
            color_attachments = len(render_pass.target._color_attachments())
            if fragment_outputs != color_attachments:
                raise ValueError("compiled fragment outputs must exactly match the color attachments")

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
        if arguments is not None and set(reflected) != set(arguments):
            missing = set(reflected) - set(arguments)
            unexpected = set(arguments) - set(reflected)
            if missing:
                raise TypeError(f"missing pipeline argument(s): {', '.join(sorted(missing))}")
            raise TypeError(f"unexpected pipeline argument(s): {', '.join(sorted(unexpected))}")

        result: dict[str, Any] = {}
        for name in reflected if arguments is None else arguments:
            value = None if arguments is None else arguments[name]
            parameter, stage = reflected[name]
            interface = {item.kind for item in parameter.interface}
            is_storage = parameter.type.kind == "tensor_view" or (
                stage.kind == "vertex"
                and "attribute" in interface
                and (arguments is None or isinstance(value, (TensorStorage, TensorView)))
            )
            if is_storage:
                if arguments is not None and not isinstance(value, (TensorStorage, TensorView)):
                    raise TypeError(f"graphics storage argument {name!r} requires TensorStorage or TensorView")
                annotation = annotations.get(name)
                if parameter.type.kind == "tensor_view":
                    if annotation is None:
                        raise TypeError(f"graphics storage argument {name!r} requires a runtime annotation")
                    descriptor = runtime_parameter_descriptor(annotation)
                    if arguments is not None:
                        _validate_storage_argument_shape(name, descriptor, cast(TensorStorage | TensorView, value))
                    result[name] = descriptor
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
                descriptor = RuntimeParameterDescriptor.storage(
                    dtype,
                    ((dyn, *cell_shape) if "attribute" in interface else parameter.type.arguments[1]),
                    str(parameter.access.value),
                    True,
                )
                if arguments is not None:
                    _validate_storage_argument_shape(name, descriptor, cast(TensorStorage | TensorView, value))
                result[name] = descriptor
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

    def _static_parameter_types(self) -> dict[str, Any]:
        """Return invocation-independent parameter contracts for Module reflection and cooking."""

        return self._parameter_types(None, self._frontends(), None, None, None)

    def _identity(
        self,
        render_pass: RenderPass,
        parameter_types: Mapping[str, Any],
    ) -> str:
        target = render_pass.target

        state = _session_state()
        options = make_target_options(
            state.arch.name,
            ({"version": state.interactive_glsl_version} if state.arch in {opengl, opengles} else {}),
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
                    "specializations": specialization_key_data(self._variant),
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
        if state.native is None:
            raise RuntimeError("graphics requires the native Program runtime")
        if state.arch == cpu:
            raise RuntimeError("CPU graphics pipelines require a software rasterizer, which Vernon does not provide")
        frontends = self._frontends()
        parameter_types = self._parameter_types(arguments, frontends, render_pass, draw, dynamic_state)
        identity = self._identity(render_pass, parameter_types)

        def compile_pipeline() -> _CompiledPipeline:
            compiled = self._compile_uncached(arguments, parameter_types, identity)
            self.compile_count += 1
            return compiled

        return self._cache.get_or_create(state, identity, compile_pipeline)

    def _compile_uncached(
        self,
        arguments: Mapping[str, Any],
        parameter_types: Mapping[str, Any],
        identity: str,
    ) -> _CompiledPipeline:
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
                self._variant,
            ),
        )
        from .program_autodiff import compile_program

        specialization = compile_program(parsed, template)
        compiled = _CompiledPipeline(identity, invocation, specialization)
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
        with _invocation_context():
            self._invoke(dict(arguments), render_pass, draw, dynamic_state)


def pipeline(
    *stages: Any,
    state: GraphicsPipelineState | None = None,
    targets: GraphicsTargetFormats | None = None,
    specializations: Mapping[Specialization, object] | None = None,
) -> Pipeline:
    return Pipeline(*stages, state=state, targets=targets, specializations=specializations)


__all__ = ["Pipeline", "PrimitiveTopology", "lines", "pipeline", "points", "triangles"]
