"""Compilation and invocation of captured Module programs."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from ._runtime.tensor import TensorStorage, TensorView
from .operation_graph import (
    AttachmentProjection,
    GraphAttachmentOutput,
    GraphBuffer,
    GraphControlInput,
    GraphControlResource,
    GraphResourceInput,
    GraphValueInput,
    KernelParameter,
    OperationGraph,
    ProgramControlDescriptor,
    ProgramControlRef,
    ProgramControlSource,
)


def _symbolic_program_inputs(parameter_types: Mapping[str, Any]) -> dict[str, Any]:
    """Create the canonical symbolic boundary values used by every Program entry path."""

    from .frontend.model import SemanticCategory

    inputs: dict[str, Any] = {}
    for name, parameter_type in parameter_types.items():
        if isinstance(parameter_type, ProgramControlDescriptor):
            inputs[name] = GraphControlInput(name, parameter_type.kind, parameter_type.prototype)
        elif parameter_type.kind is SemanticCategory.STORAGE:
            storage = parameter_type.storage_metadata
            inputs[name] = GraphBuffer(
                storage.dtype,
                storage.shape,
                storage.access,
                storage.as_view,
                parameter_type.logical,
            )
        elif parameter_type.kind is SemanticCategory.VALUE:
            inputs[name] = GraphValueInput(name, parameter_type)
        elif parameter_type.kind is SemanticCategory.RESOURCE:
            inputs[name] = GraphResourceInput(name, parameter_type)
        else:
            raise TypeError(f"unknown Program parameter kind {parameter_type.kind!r}")
    return inputs


def _one_node_program(
    root: Any,
    parameter_types: Mapping[str, Any],
    capture_operation: Any,
    output_names: tuple[str, ...] | None = (),
) -> tuple[ProgramTemplate, ProgramInvocation, Any]:
    """Build a one-node Program through the same capture/parser path as Module."""

    inputs = _symbolic_program_inputs(parameter_types)
    capture = ProgramCapture(root, inputs)
    capture_operation(capture, inputs)
    template = ProgramTemplate.compile(capture)
    if output_names is None:
        if len(template.calls) != 1:
            raise RuntimeError("one-node Program capture produced an invalid call count")
        output_names = tuple(
            parameter.name for parameter in template.calls[0].parameters if parameter.access in {"write", "read_write"}
        )
    outputs = {name: inputs[name] for name in output_names}
    invocation = template.bind(capture, outputs)
    from .program_frontend import parse_program

    return template, invocation, parse_program(invocation)


def _graphics_stage_frontend(pipeline: Any, stage: Any) -> Any:
    from .compiler import Compiler, FrontendCompileRequest

    function = stage.function
    source = Path(inspect.getsourcefile(function) or "").resolve()
    return Compiler().compile_request(FrontendCompileRequest(source, function.__name__, tuple(pipeline._features)))


# Attachment logical types are built as "2d" below, so a declared attachment has two extent components.
_ATTACHMENT_RANK = 2


@dataclass(frozen=True)
class _AttachmentDescription:
    """One attachment's compile-time identity: which slot it fills, in what format, at what rank.

    The extent is deliberately absent. It is an invocation fact, and the attachment resource below carries zeros in
    its place so that the same cooked Program serves any target size.
    """

    aspect: str
    location: int | None
    format_name: str
    rank: int


def _attachment_format_name(texture: Any) -> str:
    return getattr(getattr(texture, "format", None), "name", None) or "d32_float"


def _render_pass_attachments(prototype: Any) -> tuple[_AttachmentDescription, ...]:
    """Read the attachment structure off a concrete RenderPass, as a live pipeline or Module invocation can."""

    target = prototype.target
    descriptions = [
        _AttachmentDescription("color", location, _attachment_format_name(texture), len(texture.shape))
        for location, texture in target._color_attachments()
    ]
    depth = target._depth_attachment()
    if depth is not None:
        descriptions.append(_AttachmentDescription("depth", None, _attachment_format_name(depth), len(depth.shape)))
    return tuple(descriptions)


def _declared_attachments(targets: Any) -> tuple[_AttachmentDescription, ...]:
    """Read the same structure off vd.pipeline(..., targets=...), which a cooked asset has instead of a RenderPass."""

    descriptions = [
        _AttachmentDescription("color", location, format.name, _ATTACHMENT_RANK) for location, format in targets.colors
    ]
    if targets.depth is not None:
        descriptions.append(_AttachmentDescription("depth", None, targets.depth.name, _ATTACHMENT_RANK))
    return tuple(descriptions)


def _attachment_slots(descriptions: tuple[_AttachmentDescription, ...]) -> tuple[tuple[str, int | None, str], ...]:
    return tuple((value.aspect, value.location, value.format_name) for value in descriptions)


def _resolve_attachments(pipeline: Any, render_pass: Any) -> tuple[_AttachmentDescription, ...]:
    """Determine a graphics node's attachment structure from whichever source is authoritative.

    A pipeline that declares target formats is authoritative, and a RenderPass bound to it must agree -- the same
    compatibility rule a Vulkan pipeline imposes on the render pass it is used in. A pipeline that declares none can
    still run live by reading the structure off the concrete RenderPass, but it cannot be cooked, because an asset
    has no RenderPass to read.
    """

    declared = getattr(pipeline, "_targets", None)
    prototype = render_pass.prototype if isinstance(render_pass, GraphControlInput) else render_pass
    if declared is None:
        if prototype is None:
            raise TypeError(
                "cooking a graphics Program requires vd.pipeline(..., targets=vd.target_formats(...)); attachment "
                "formats are pipeline state, and no RenderPass exists at cook time to read them from"
            )
        return _render_pass_attachments(prototype)
    descriptions = _declared_attachments(declared)
    if prototype is not None:
        bound = _render_pass_attachments(prototype)
        if _attachment_slots(bound) != _attachment_slots(descriptions):
            raise TypeError(
                "RenderPass attachments do not match the formats this pipeline was built for: "
                f"declared {_attachment_slots(descriptions)}, bound {_attachment_slots(bound)}"
            )
    return descriptions


def _graphics_attachments(
    descriptions: tuple[_AttachmentDescription, ...],
    control: ProgramControlRef,
    cache: dict[AttachmentProjection, GraphControlResource],
) -> tuple[tuple[GraphControlResource, ...], int]:
    from .frontend.model import ConcreteType

    def resource(description: _AttachmentDescription) -> GraphControlResource:
        projection = AttachmentProjection(control, description.aspect, description.location)
        existing = cache.get(projection)
        if existing is not None:
            return existing
        logical = ConcreteType(
            "texture",
            "Texture",
            ("2d", ConcreteType("scalar", "f32"), description.format_name, "read_write"),
        )
        control_name = (
            str(control.identifier)
            if control.source is ProgramControlSource.ARGUMENT
            else f"{control.source.value}.{control.identifier}"
        )
        result = GraphControlResource(
            f"{control_name}.{description.aspect}.{description.location if description.location is not None else 0}",
            tuple(0 for _ in range(description.rank)),
            logical,
            projection,
        )
        cache[projection] = result
        return result

    if not descriptions:
        raise ValueError("graphics Program requires at least one color or depth attachment")
    return tuple(resource(description) for description in descriptions), sum(
        1 for description in descriptions if description.aspect == "color"
    )


def flatten_program_outputs(value: Any, prefix: str = "") -> dict[str, Any]:
    from ._runtime.texture import _TextureResource

    if value is None:
        return {}
    if isinstance(
        value,
        (
            GraphAttachmentOutput,
            GraphBuffer,
            GraphResourceInput,
            TensorStorage,
            TensorView,
            _TextureResource,
        ),
    ):
        return {prefix or "output": value}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Any] = {}
        for field in dataclasses.fields(value):
            path = f"{prefix}.{field.name}" if prefix else field.name
            result.update(flatten_program_outputs(getattr(value, field.name), path))
        return result
    if isinstance(value, Mapping):
        result = {}
        for name, member in value.items():
            if not isinstance(name, str) or not name:
                raise TypeError("Module output mapping keys must be non-empty strings")
            path = f"{prefix}.{name}" if prefix else name
            result.update(flatten_program_outputs(member, path))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, member in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            result.update(flatten_program_outputs(member, path))
        return result
    raise TypeError(
        "Module.forward() outputs must contain only Program buffers, mappings, tuples, lists, or dataclasses"
    )


@dataclass(frozen=True)
class CapturedAlloc:
    kind: str
    result: GraphBuffer
    like: GraphBuffer | None = None
    values: Any = None


@dataclass(frozen=True)
class CapturedKernelCall:
    name: str
    kernel: Any
    arguments: tuple[Any, ...]
    grid: tuple[Any, Any, Any]
    features: tuple[str, ...]
    lowered: Any = None


@dataclass(frozen=True)
class CapturedGraphicsCall:
    name: str
    pipeline: Any
    arguments: Mapping[str, Any]
    render_pass: Any
    draw: Any
    dynamic_state: Any
    render_pass_ref: ProgramControlRef
    draw_ref: ProgramControlRef
    dynamic_state_ref: ProgramControlRef
    frontends: tuple[Any, ...] = ()


class _ControlIdentityRegistry:
    """Capture-local identity table for concrete invocation control objects."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, str], tuple[Any, ProgramControlRef]] = {}

    def ref(self, value: Any, kind: str) -> ProgramControlRef:
        key = (id(value), kind)
        existing = self._entries.get(key)
        if existing is not None and existing[0] is value:
            return existing[1]
        ref = ProgramControlRef(kind, ProgramControlSource.CAPTURE, len(self._entries))
        self._entries[key] = (value, ref)
        return ref


class ProgramCapture:
    def __init__(self, root: Any, inputs: Mapping[str, Any]):
        self.root = root
        self.inputs = dict(inputs)
        self.ops: list[CapturedAlloc | CapturedKernelCall | CapturedGraphicsCall] = []
        self._scopes = [type(root).__name__]
        self._name_counts: dict[str, int] = {}
        self._control_identities = _ControlIdentityRegistry()

    @property
    def calls(self) -> tuple[CapturedKernelCall, ...]:
        return tuple(op for op in self.ops if isinstance(op, CapturedKernelCall))

    @property
    def allocs(self) -> tuple[CapturedAlloc, ...]:
        return tuple(op for op in self.ops if isinstance(op, CapturedAlloc))

    @property
    def graphics_calls(self) -> tuple[CapturedGraphicsCall, ...]:
        return tuple(op for op in self.ops if isinstance(op, CapturedGraphicsCall))

    def control_ref(self, value: Any, kind: str) -> ProgramControlRef:
        if isinstance(value, GraphControlInput):
            if value.kind != kind:
                raise TypeError(f"expected {kind} Program control, got {value.kind}")
            return value.ref
        return self._control_identities.ref(value, kind)

    def enter_module(self, name: str) -> None:
        self._scopes.append(name)

    def leave_module(self) -> None:
        self._scopes.pop()

    def capture_kernel(
        self,
        kernel: Any,
        arguments: tuple[Any, ...],
        grid: tuple[Any, Any, Any] | None,
        features: tuple[str, ...],
        *,
        lowered: Any = None,
    ) -> None:
        dispatch_grid = grid or (1, 1, 1)
        if len(dispatch_grid) != 3:
            raise ValueError("grid must contain three components")
        for value in dispatch_grid:
            if isinstance(value, GraphValueInput):
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("grid components must be positive integers or invocation Values")
        base = ".".join((*self._scopes, kernel.__name__))
        ordinal = self._name_counts.get(base, 0)
        self._name_counts[base] = ordinal + 1
        name = base if ordinal == 0 else f"{base}.{ordinal}"
        self.ops.append(CapturedKernelCall(name, kernel, arguments, dispatch_grid, tuple(features), lowered))

    def capture_graphics(
        self,
        pipeline: Any,
        arguments: Mapping[str, Any],
        render_pass: Any,
        draw: Any,
        dynamic_state: Any,
        *,
        frontends: tuple[Any, ...] = (),
    ) -> None:
        base = ".".join((*self._scopes, "graphics"))
        ordinal = self._name_counts.get(base, 0)
        self._name_counts[base] = ordinal + 1
        name = base if ordinal == 0 else f"{base}.{ordinal}"
        self.ops.append(
            CapturedGraphicsCall(
                name,
                pipeline,
                dict(arguments),
                render_pass,
                draw,
                dynamic_state,
                self.control_ref(render_pass, "render_pass"),
                self.control_ref(draw, "draw"),
                self.control_ref(dynamic_state, "dynamic_state"),
                frontends,
            )
        )

    def capture_allocation(
        self,
        kind: str,
        dtype: Any,
        shape: tuple[int, ...],
        *,
        like: GraphBuffer | None = None,
        values: Any = None,
    ) -> GraphBuffer:
        from .frontend.runtime_types import RuntimeParameterDescriptor

        logical = like.logical if like is not None else RuntimeParameterDescriptor.storage(dtype, tuple(shape)).logical
        result = GraphBuffer(dtype, tuple(shape), "read_write", logical=logical)
        self.ops.append(CapturedAlloc(kind, result, like, values))
        return result


@dataclass(frozen=True)
class KernelCallTemplate:
    name: str
    kernel: Any
    parameter_names: tuple[str, ...]
    parameters: tuple[KernelParameter, ...]
    grid: tuple[Any, Any, Any]
    features: tuple[str, ...]


@dataclass(frozen=True)
class GraphicsCallTemplate:
    name: str
    pipeline: Any
    parameter_names: tuple[str, ...]
    parameters: tuple[KernelParameter, ...]
    features: tuple[str, ...]
    implementations: tuple[tuple[str, str, str], ...]
    structs: tuple[tuple[str, tuple[Any, ...]], ...]


class ProgramTemplate:
    """Typed, reusable structure with no invocation resource references."""

    def __init__(self, calls: tuple[KernelCallTemplate | GraphicsCallTemplate, ...]):
        self.calls = calls

    @classmethod
    def compile(cls, capture: ProgramCapture) -> ProgramTemplate:
        calls: list[KernelCallTemplate | GraphicsCallTemplate] = []
        for captured in (op for op in capture.ops if not isinstance(op, CapturedAlloc)):
            if isinstance(captured, CapturedGraphicsCall):
                parameters: dict[str, KernelParameter] = {}
                implementations: list[tuple[str, str, str]] = []
                structs: dict[str, tuple[Any, ...]] = {}
                for stage_index, stage in enumerate(captured.pipeline._stages):
                    function = stage.function
                    frontend = (
                        captured.frontends[stage_index]
                        if captured.frontends
                        else _graphics_stage_frontend(captured.pipeline, stage)
                    )
                    implementations.append((stage.kind, function.__name__, frontend.mlir))
                    for struct_name, struct_fields in frontend.structs:
                        previous_fields = structs.get(struct_name)
                        if previous_fields is not None and previous_fields != struct_fields:
                            raise TypeError(f"graphics stages disagree on struct {struct_name!r}")
                        structs[struct_name] = struct_fields
                    entry = next(
                        (typed for typed in frontend.typed_functions if typed.source.name == function.__name__),
                        None,
                    )
                    if entry is None:
                        raise RuntimeError(f"compiled graphics stage {function.__name__!r} has no typed entry")
                    for typed in entry.parameters:
                        interface_kinds = {item.kind for item in typed.interface}
                        if typed.builtin is not None or "implicit" in interface_kinds:
                            continue
                        if stage.kind == "fragment" and "varying" in interface_kinds:
                            continue
                        access = str(typed.type.arguments[3]) if typed.type.kind == "texture" else typed.access.value
                        previous = parameters.get(typed.name)
                        parameter = KernelParameter(typed.name, access, False, False)
                        if previous is not None and previous.access != parameter.access:
                            raise TypeError(f"graphics stages disagree on access for parameter {typed.name!r}")
                        parameters[typed.name] = parameter
                parameter_names = tuple(captured.arguments)
                if set(parameter_names) != set(parameters):
                    missing = set(parameters) - set(parameter_names)
                    unexpected = set(parameter_names) - set(parameters)
                    if missing:
                        raise TypeError(f"missing pipeline argument(s): {', '.join(sorted(missing))}")
                    raise TypeError(f"unexpected pipeline argument(s): {', '.join(sorted(unexpected))}")
                calls.append(
                    GraphicsCallTemplate(
                        captured.name,
                        captured.pipeline,
                        parameter_names,
                        tuple(parameters[name] for name in parameter_names),
                        tuple(captured.pipeline._features),
                        tuple(implementations),
                        tuple(sorted(structs.items())),
                    )
                )
                continue
            assert isinstance(captured, CapturedKernelCall)
            lowered = captured.lowered or captured.kernel._lower(captured.features)
            frontend = lowered.frontend
            builtin_names = lowered.builtins
            parameter_names = tuple(
                name for name in inspect.signature(captured.kernel._function).parameters if name not in builtin_names
            )
            if len(parameter_names) != len(captured.arguments):
                raise TypeError(f"{captured.kernel.__name__} expects {len(parameter_names)} launch arguments")
            entry = next(
                (function for function in frontend.typed_functions if function.source.name == captured.kernel._entry),
                None,
            )
            if entry is None or entry.storage_activity is None:
                raise RuntimeError(f"compiled kernel {captured.kernel.__name__!r} has no typed storage activity")
            parameters: list[KernelParameter] = []
            for parameter in entry.parameters:
                if parameter.name not in parameter_names:
                    continue
                is_storage = parameter.type.kind == "tensor_view"
                access = (
                    str(parameter.type.arguments[3]) if parameter.type.kind == "texture" else parameter.access.value
                )
                parameters.append(
                    KernelParameter(
                        parameter.name,
                        access,
                        is_storage and parameter.name in entry.storage_activity.readable_roots,
                        bool(entry.storage_activity.dependencies_for(parameter.name)),
                    )
                )
            calls.append(
                KernelCallTemplate(
                    captured.name,
                    captured.kernel,
                    parameter_names,
                    tuple(parameters),
                    captured.grid,
                    captured.features,
                )
            )
        return cls(tuple(calls))

    def bind(self, capture: ProgramCapture, outputs: Any) -> ProgramInvocation:
        captured_calls = sum(not isinstance(op, CapturedAlloc) for op in capture.ops)
        if captured_calls != len(self.calls):
            raise RuntimeError("Module control flow changed for an existing Program specialization")
        graph = OperationGraph()
        graphics_attachment_cache: dict[AttachmentProjection, GraphControlResource] = {}
        graphics_control_slots: dict[ProgramControlRef, int] = {}
        slots: list[Any] = []
        for name, value in capture.inputs.items():
            graph.import_input(name, value)
        operation_bindings: dict[int, tuple[Any, ...]] = {}

        def resolve_attachment_output(value: Any) -> Any:
            if not isinstance(value, GraphAttachmentOutput):
                return value
            resource = graphics_attachment_cache.get(value.projection)
            if resource is None:
                raise RuntimeError("attachment output must follow a graphics call using the same RenderPass")
            return resource

        call_index = 0
        for captured in capture.ops:
            if isinstance(captured, CapturedAlloc):
                graph.append_alloc(
                    buffer=captured.result,
                    name=captured.kind,
                    like=captured.like,
                    values=captured.values,
                )
                continue
            template = self.calls[call_index]
            call_index += 1
            if isinstance(captured, CapturedGraphicsCall):
                if not isinstance(template, GraphicsCallTemplate) or template.pipeline is not captured.pipeline:
                    raise RuntimeError("Module graphics sequence changed for an existing Program specialization")
                first_slot = len(slots)
                arguments = tuple(
                    resolve_attachment_output(captured.arguments[name]) for name in template.parameter_names
                )
                slots.extend(arguments)
                binding_slots = {name: first_slot + index for index, name in enumerate(template.parameter_names)}
                control_slots = {}
                for kind, value, ref in (
                    ("render_pass", captured.render_pass, captured.render_pass_ref),
                    ("draw", captured.draw, captured.draw_ref),
                    (
                        "dynamic_state",
                        captured.dynamic_state,
                        captured.dynamic_state_ref,
                    ),
                ):
                    slot = graphics_control_slots.get(ref)
                    if slot is None:
                        slot = len(slots)
                        slots.append(value)
                        graphics_control_slots[ref] = slot
                    control_slots[kind] = slot
                attachments, color_count = _graphics_attachments(
                    _resolve_attachments(captured.pipeline, captured.render_pass),
                    captured.render_pass_ref,
                    graphics_attachment_cache,
                )
                for attachment in attachments:
                    graph.import_control_resource(attachment)
                operation = graph.append_graphics(
                    name=template.name,
                    pipeline=template.pipeline,
                    bindings=dict(zip(template.parameter_names, arguments, strict=True)),
                    binding_slots=binding_slots,
                    control_slots=control_slots,
                    parameters=template.parameters,
                    attachments=attachments,
                    color_count=color_count,
                    features=template.features,
                )
                operation_bindings[operation.id] = (
                    *arguments,
                    captured.render_pass,
                    captured.draw,
                    captured.dynamic_state,
                )
                continue
            if not isinstance(template, KernelCallTemplate):
                raise RuntimeError("Module operation sequence changed for an existing Program specialization")
            if (
                template.kernel is not captured.kernel
                or template.grid != captured.grid
                or template.features != captured.features
            ):
                raise RuntimeError("Module kernel sequence changed for an existing Program specialization")
            first_slot = len(slots)
            slots.extend(captured.arguments)
            bindings = dict(zip(template.parameter_names, captured.arguments, strict=True))
            binding_slots = {name: first_slot + index for index, name in enumerate(template.parameter_names)}
            operation = graph.append_kernel(
                name=template.name,
                kernel=template.kernel,
                bindings=bindings,
                binding_slots=binding_slots,
                parameters=template.parameters,
                grid=template.grid,
                features=template.features,
            )
            operation_bindings[operation.id] = captured.arguments
        graph.set_outputs(
            {path: resolve_attachment_output(value) for path, value in flatten_program_outputs(outputs).items()}
        )
        return ProgramInvocation(
            self,
            graph,
            tuple(slots),
            operation_bindings,
            dict(capture.inputs),
            {name: value.annotation for name, value in capture.inputs.items() if isinstance(value, GraphValueInput)},
            outputs,
        )


@dataclass(frozen=True)
class ProgramInvocation:
    template: ProgramTemplate
    graph: OperationGraph
    slots: tuple[Any, ...]
    operation_bindings: Mapping[int, tuple[Any, ...]]
    inputs: Mapping[str, Any]
    input_annotations: Mapping[str, Any]
    outputs: Any

    def slot_value(self, slot: int) -> Any:
        value = self.slots[slot]
        if isinstance(value, GraphControlInput):
            return self.inputs[value.name]
        return value

    @property
    def graphics_controls(self) -> Mapping[int, Mapping[str, tuple[int, Any]]]:
        from types import MappingProxyType

        controls = {}
        for operation in self.graph.operations:
            control_slots = getattr(operation, "control_slots", None)
            if control_slots is None:
                continue
            controls[operation.id] = MappingProxyType(
                {kind: (slot, self.slot_value(slot)) for kind, slot in control_slots.items()}
            )
        return MappingProxyType(controls)


class ModuleVjpExpression:
    def __init__(
        self,
        module: Any,
        *,
        wrt: tuple[str, ...],
        outputs: tuple[str, ...] | None,
        planning_policy: str,
    ):
        self.module = module
        self.wrt = wrt
        self.outputs = outputs
        self.planning_policy = planning_policy

    def __call__(self, *arguments: Any, **keywords: Any) -> tuple[Any, Any]:
        return execute_module_vjp(self, arguments, keywords)


def _capture_module(module: Any, parameter_types: Mapping[str, Any]) -> tuple[ProgramCapture, Any]:
    from .frontend.module_ast import interpret_module_forward

    return interpret_module_forward(module, parameter_types)


def _bind_forward_arguments(module: Any, arguments: tuple[Any, ...], keywords: Mapping[str, Any]) -> dict[str, Any]:
    bound = type(module)._module_definition.signature.bind(*arguments, **keywords)
    bound.apply_defaults()
    return dict(bound.arguments)


def _access_name(access: Any) -> str:
    return str(getattr(access, "name", access))


def _reflect_module_parameter_types(module: Any) -> dict[str, Any]:
    """Infer public resource types from nested typed Program calls."""

    from .module import Module
    from .types import TypeExpr, read_write

    cache: dict[int, dict[str, TypeExpr]] = {}
    active: set[int] = set()

    def merge(
        constraints: dict[str, TypeExpr],
        name: str,
        candidate: TypeExpr,
        owner: Any,
    ) -> None:
        existing = constraints.get(name)
        if existing is None:
            constraints[name] = candidate
            return
        existing_dtype, existing_shape, existing_access = existing.arguments
        dtype, shape, access = candidate.arguments
        if existing_dtype != dtype or existing_shape != shape:
            raise TypeError(
                f"conflicting reflected types for Module parameter {name!r} in {type(owner).__name__}.forward"
            )
        if existing_access != access:
            constraints[name] = TypeExpr("TensorView", (dtype, shape, read_write))

    def resolve(expression: ast.expr, owner: Any, globals_: Mapping[str, Any]) -> Any:
        if isinstance(expression, ast.Name):
            if expression.id == "self":
                return owner
            return globals_.get(expression.id)
        if isinstance(expression, ast.Attribute):
            base = resolve(expression.value, owner, globals_)
            return getattr(base, expression.attr, None) if base is not None else None
        return None

    def target_types(
        target: Any,
    ) -> tuple[inspect.Signature, dict[str, TypeExpr]] | None:
        if isinstance(target, Module):
            return type(target)._module_definition.signature, infer(target)
        function = getattr(target, "_function", None)
        if function is None:
            return None
        try:
            annotations = inspect.get_annotations(function, eval_str=True)
        except (NameError, TypeError):
            return None
        reflected = {
            name: annotation
            for name, annotation in annotations.items()
            if isinstance(annotation, TypeExpr) and annotation.name == "TensorView"
        }
        return inspect.signature(function), reflected

    def infer(owner: Any) -> dict[str, TypeExpr]:
        identity = id(owner)
        if identity in cache:
            return cache[identity]
        if identity in active:
            raise TypeError(f"recursive Module composition is not exportable: {type(owner).__name__}")
        active.add(identity)
        constraints: dict[str, TypeExpr] = {}
        definition = type(owner)._module_definition
        function = definition.original_function
        function_ast = definition.forward_ast
        if function_ast is None:
            active.remove(identity)
            cache[identity] = constraints
            return constraints
        parameters = set(definition.signature.parameters)
        globals_ = function.__globals__
        for call in (node for node in ast.walk(function_ast) if isinstance(node, ast.Call)):
            target = resolve(call.func, owner, globals_)
            reflected = target_types(target)
            if reflected is None:
                continue
            signature, types = reflected
            target_parameters = tuple(signature.parameters)
            bindings: list[tuple[ast.expr, str]] = [
                (argument, target_name) for argument, target_name in zip(call.args, target_parameters, strict=False)
            ]
            bindings.extend(
                (keyword.value, keyword.arg)
                for keyword in call.keywords
                if keyword.arg is not None and keyword.arg in signature.parameters
            )
            for expression, target_name in bindings:
                if not isinstance(expression, ast.Name) or expression.id not in parameters:
                    continue
                candidate = types.get(target_name)
                if candidate is not None:
                    merge(constraints, expression.id, candidate, owner)
        active.remove(identity)
        cache[identity] = constraints
        return constraints

    return infer(module)


def _module_parameter_types(module: Any) -> dict[str, Any]:
    from .frontend.model import SemanticCategory
    from .frontend.runtime_types import (
        RuntimeParameterDescriptor,
        runtime_parameter_descriptor,
    )
    from .operation_graph import ProgramControlDescriptor
    from .types import TypeExpr, dyn

    definition = type(module)._module_definition
    annotations = definition.resolved_annotations
    reflected = _reflect_module_parameter_types(module)
    types: dict[str, Any] = {}
    for parameter in definition.signature.parameters.values():
        annotation = annotations.get(parameter.name)
        control_kind = _program_control_kind(annotation)
        if control_kind is not None:
            types[parameter.name] = ProgramControlDescriptor(control_kind, annotation)
            continue
        if parameter.default is not inspect.Parameter.empty:
            continue
        try:
            descriptor = runtime_parameter_descriptor(annotation)
        except TypeError:
            descriptor = None
        if descriptor is not None and descriptor.kind is not SemanticCategory.STORAGE:
            types[parameter.name] = descriptor
            continue
        inferred = annotation if isinstance(annotation, TypeExpr) and annotation.name == "TensorView" else None
        if inferred is None:
            inferred = reflected.get(parameter.name)
        if inferred is None:
            raise TypeError(
                f"cannot infer Module parameter {parameter.name!r}; annotate it as "
                "TensorView[element, (shape,), access], use it in a typed Program call, "
                "or supply an explicit export argument"
            )
        if len(inferred.arguments) != 3:
            raise TypeError(f"invalid TensorView annotation for Module parameter {parameter.name!r}")
        dtype, shape, access = inferred.arguments
        if not isinstance(shape, tuple) or any(extent is dyn for extent in shape):
            raise TypeError(
                f"cannot infer dynamic shape for Module parameter {parameter.name!r} without an export argument"
            )
        as_view = isinstance(annotation, TypeExpr) and annotation.name == "TensorView"
        types[parameter.name] = RuntimeParameterDescriptor.storage(
            dtype,
            tuple(shape),
            _access_name(access) if as_view else "read_write",
            as_view,
        )
    return types


def _parameter_types_from_values(
    module: Any,
    arguments: tuple[Any, ...],
    keywords: Mapping[str, Any],
) -> dict[str, Any]:
    from ._dtypes import scalar_name
    from .frontend.model import SemanticCategory
    from .frontend.runtime_types import (
        RuntimeParameterDescriptor,
        runtime_parameter_descriptor,
        validate_host_value,
    )
    from .operation_graph import ProgramControlDescriptor
    from .types import TypeExpr

    def tensor_dtype(value: TensorStorage) -> Any:
        if value._element_type is not None:
            return value._element_type
        name = scalar_name(value.dtype)
        if name is None:
            raise TypeError(f"cannot allocate a Module transient for dtype {value.dtype}")
        types = __import__("vernon_dsl.types", fromlist=[name])
        return getattr(types, name)

    def descriptor(annotation: Any, value: Any, name: str) -> Any:
        control_kind = _program_control_kind(annotation)
        if control_kind is not None:
            from .render import DrawCommand, DynamicState, RenderPass

            expected = {
                "render_pass": RenderPass,
                "draw": DrawCommand,
                "dynamic_state": DynamicState,
            }[control_kind]
            if value is not None and not isinstance(value, expected):
                raise TypeError(
                    f"Module.forward() control {name!r} requires {expected.__name__}, got {type(value).__name__}"
                )
            if control_kind == "render_pass" and value is None:
                raise TypeError(f"Module.forward() RenderPass control {name!r} cannot be None")
            return ProgramControlDescriptor(control_kind, annotation, value)
        as_view = isinstance(annotation, TypeExpr) and annotation.name == "TensorView"
        if isinstance(value, TensorView):
            owner = value.owner
            if not isinstance(owner, TensorStorage):
                raise TypeError("Module.forward() TensorView parameters must borrow TensorStorage")
            dtype = value.element_type if value.element_type is not None else tensor_dtype(owner)
            return RuntimeParameterDescriptor.storage(dtype, tuple(value.shape), str(value.access), True)
        if isinstance(value, TensorStorage):
            access = "read_write"
            if as_view and len(annotation.arguments) == 3:
                access = str(getattr(annotation.arguments[2], "name", annotation.arguments[2]))
            return RuntimeParameterDescriptor.storage(tensor_dtype(value), tuple(value.shape), access, as_view)
        if annotation is None:
            raise TypeError(
                f"Module.forward() argument {name!r} is a runtime Value or Resource and requires a DSL annotation"
            )
        result = runtime_parameter_descriptor(annotation)
        if result.kind is SemanticCategory.VALUE:
            validate_host_value(annotation, value, name)
        if result.kind is SemanticCategory.STORAGE:
            raise TypeError(f"Module.forward() argument {name!r} must be TensorStorage or TensorView")
        return result

    annotations = type(module)._module_definition.resolved_annotations
    types = {}
    for name, value in _bind_forward_arguments(module, arguments, keywords).items():
        types[name] = descriptor(annotations.get(name), value, name)
    return types


def _program_control_kind(annotation: Any) -> str | None:
    from .render import DrawCommand, DynamicState, RenderPass

    members = set(get_args(annotation)) or {annotation}
    matches = [
        kind
        for kind, control_type in (
            ("render_pass", RenderPass),
            ("draw", DrawCommand),
            ("dynamic_state", DynamicState),
        )
        if control_type in members
    ]
    return matches[0] if len(matches) == 1 else None


def _parse_module_program(
    module: Any,
    arguments: tuple[Any, ...] = (),
    keywords: Mapping[str, Any] | None = None,
    *,
    vjp_wrt: tuple[str, ...] | None = None,
    vjp_outputs: tuple[str, ...] | None = None,
    autodiff_planning_policy: str = "min_memory",
) -> Any:
    supplied_keywords = {} if keywords is None else dict(keywords)
    parameter_types = (
        _parameter_types_from_values(module, arguments, supplied_keywords)
        if arguments or supplied_keywords
        else _module_parameter_types(module)
    )
    capture, outputs = _capture_module(module, parameter_types)
    if vjp_wrt is not None and capture.graphics_calls:
        raise TypeError("graphics Module programs do not support autodiff")
    template = ProgramTemplate.compile(capture)
    invocation = template.bind(capture, outputs)
    from .program_frontend import parse_program

    selected_outputs = vjp_outputs
    if vjp_wrt is not None:
        selected_outputs = selected_outputs or tuple(invocation.graph.outputs)
        unknown_outputs = tuple(path for path in selected_outputs if path not in invocation.graph.outputs)
        if unknown_outputs:
            raise ValueError(f"Program autodiff outputs do not identify Module output paths: {unknown_outputs}")
    return parse_program(
        invocation,
        vjp_wrt=vjp_wrt,
        vjp_outputs=selected_outputs,
        autodiff_planning_policy=autodiff_planning_policy,
    )


@dataclass(frozen=True)
class _AllocationSpec:
    dtype: Any
    shape: tuple[int, ...]
    initializer: str
    values: Any = None

    def create(self) -> TensorStorage:
        if self.initializer == "from_values":
            return TensorStorage.from_values(self.values, dtype=self.dtype)
        if self.initializer in {"zeros", "zeros_like"}:
            return TensorStorage.zeros(dtype=self.dtype, shape=self.shape)
        return TensorStorage.empty(dtype=self.dtype, shape=self.shape)


@dataclass(frozen=True)
class _ValueRecipe:
    source: str
    key: str | int | tuple[Any, ...] | None = None
    shape: tuple[int, ...] | None = None
    strides: tuple[int, ...] | None = None
    offset: int = 0
    access: str = "read_write"
    dtype: Any = None
    element_type: Any = None
    constant: Any = None
    as_view: bool = False

    def resolve(self, inputs: Mapping[str, Any], allocations: tuple[TensorStorage, ...]) -> Any:
        if self.source == "constant":
            return self.constant
        if self.source == "attachment":
            if not isinstance(self.key, tuple) or len(self.key) != 3:
                raise ValueError("attachment output recipe is incomplete")
            input_name, aspect, location = self.key
            from .render import color_output, depth_output

            render_pass = inputs[input_name]
            return color_output(render_pass, location=location) if aspect == "color" else depth_output(render_pass)
        if self.source == "input":
            if not isinstance(self.key, str):
                raise ValueError("input tree recipe requires a parameter name")
            owner = inputs[self.key]
        else:
            if not isinstance(self.key, int):
                raise ValueError("allocation tree recipe requires an allocation index")
            owner = allocations[self.key]
        if isinstance(owner, TensorView):
            owner = owner.owner
        if self.as_view:
            view = getattr(owner, "view", None)
            if not callable(view):
                raise TypeError("tree recipe view owner does not support TensorView projection")
            return view(access=self.access)
        return owner


@dataclass(frozen=True)
class _TreeRecipe:
    kind: str
    value: Any

    def resolve(self, inputs: Mapping[str, Any], allocations: tuple[TensorStorage, ...]) -> Any:
        if self.kind == "leaf":
            return self.value.resolve(inputs, allocations)
        if self.kind == "mapping":
            return {name: member.resolve(inputs, allocations) for name, member in self.value}
        if self.kind == "tuple":
            return tuple(member.resolve(inputs, allocations) for member in self.value)
        if self.kind == "list":
            return [member.resolve(inputs, allocations) for member in self.value]
        if self.kind == "dataclass":
            cls, fields = self.value
            return cls(**{name: member.resolve(inputs, allocations) for name, member in fields})
        raise RuntimeError(f"unknown Program output recipe {self.kind!r}")


@dataclass(frozen=True)
class _PrimalSpecialization:
    program: Any
    template: ProgramTemplate
    invocation: ProgramInvocation
    signature: inspect.Signature
    allocations: tuple[_AllocationSpec, ...]
    calls: tuple[tuple[Any, tuple[_ValueRecipe, ...], tuple[int, int, int], tuple[str, ...]], ...]
    outputs: _TreeRecipe
    native_program: Any

    def invoke(self, arguments: tuple[Any, ...], keywords: Mapping[str, Any]) -> Any:
        inputs = self.signature.bind(*arguments, **keywords).arguments
        allocations = tuple(spec.create() for spec in self.allocations)
        outputs = self.outputs.resolve(inputs, allocations)
        return self.native_program.invoke(
            dataclasses.replace(
                self.invocation,
                inputs=dict(inputs),
                outputs=outputs,
            )
        )


def _capture_recipes(
    capture: ProgramCapture,
    outputs: Any,
) -> tuple[
    tuple[_AllocationSpec, ...],
    tuple[tuple[Any, tuple[_ValueRecipe, ...], tuple[int, int, int], tuple[str, ...]], ...],
    _TreeRecipe,
]:
    inputs = tuple(capture.inputs.items())
    captured_allocations = tuple(enumerate(capture.allocs))

    def input_name(value: Any) -> str | None:
        return next((name for name, candidate in inputs if candidate is value), None)

    def allocation_index(value: Any) -> int | None:
        return next(
            (index for index, alloc in captured_allocations if alloc.result is value),
            None,
        )

    def recipe(value: Any) -> _ValueRecipe:
        if isinstance(value, GraphAttachmentOutput):
            render_pass_input = input_name(value.render_pass)
            if render_pass_input is None:
                raise TypeError("attachment output RenderPass is not a Module input")
            return _ValueRecipe(
                "attachment",
                (render_pass_input, value.projection.aspect, value.projection.location),
            )
        if isinstance(value, GraphBuffer):
            boundary_name = input_name(value)
            allocation = allocation_index(value)
            if boundary_name is not None:
                source, key = "input", boundary_name
            elif allocation is not None:
                source, key = "allocation", allocation
            else:
                raise TypeError(
                    "Module.forward() captured an unregistered Program buffer; "
                    "use vd.empty(), vd.zeros(), vd.from_values(), or an invocation parameter"
                )
            return _ValueRecipe(source, key, access=value.access, as_view=value.as_view)
        for name, supplied in capture.inputs.items():
            if value is supplied:
                return _ValueRecipe("input", name)
        return _ValueRecipe("constant", constant=value)

    def tree(value: Any) -> _TreeRecipe:
        if value is None:
            return _TreeRecipe("leaf", _ValueRecipe("constant", constant=None))
        if isinstance(value, (GraphAttachmentOutput, GraphBuffer, TensorStorage, TensorView)):
            return _TreeRecipe("leaf", recipe(value))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return _TreeRecipe(
                "dataclass",
                (
                    type(value),
                    tuple((field.name, tree(getattr(value, field.name))) for field in dataclasses.fields(value)),
                ),
            )
        if isinstance(value, Mapping):
            return _TreeRecipe("mapping", tuple((name, tree(member)) for name, member in value.items()))
        if isinstance(value, tuple):
            return _TreeRecipe("tuple", tuple(tree(member) for member in value))
        if isinstance(value, list):
            return _TreeRecipe("list", tuple(tree(member) for member in value))
        raise TypeError("Module output contains a value that cannot be reconstructed from Program IR")

    calls = tuple(
        (
            call.kernel,
            tuple(recipe(value) for value in call.arguments),
            call.grid,
            call.features,
        )
        for call in capture.calls
    )
    allocations = tuple(
        _AllocationSpec(alloc.result.dtype, tuple(alloc.result.shape), alloc.kind, alloc.values)
        for alloc in capture.allocs
    )
    return allocations, calls, tree(outputs)


def _primal_specialization(
    module: Any,
    capture: ProgramCapture,
    outputs: Any,
    invocation: ProgramInvocation,
) -> _PrimalSpecialization:
    allocations, calls, output_recipe = _capture_recipes(capture, outputs)
    from .program_frontend import parse_program

    parsed = parse_program(invocation)
    from ._runtime.program_autodiff import compile_program

    return _PrimalSpecialization(
        parsed,
        invocation.template,
        invocation,
        type(module)._module_definition.signature,
        allocations,
        calls,
        output_recipe,
        compile_program(parsed, invocation.template),
    )


@dataclass(frozen=True)
class _VjpSpecialization:
    compiled: Any
    invocation: ProgramInvocation
    signature: inspect.Signature
    allocations: tuple[_AllocationSpec, ...]
    outputs: _TreeRecipe

    @property
    def template(self) -> ProgramTemplate:
        return self.compiled.template

    @property
    def pipeline(self) -> Any:
        return self.compiled.pipeline

    def invoke(
        self,
        arguments: tuple[Any, ...],
        keywords: Mapping[str, Any],
        *,
        checkpoint_memory_budget: int | None,
        checkpoint_policy: str,
    ) -> tuple[Any, Any]:
        bound = self.signature.bind(*arguments, **keywords)
        bound.apply_defaults()
        inputs = dict(bound.arguments)
        allocations = tuple(spec.create() for spec in self.allocations)
        invocation = dataclasses.replace(
            self.invocation,
            inputs=inputs,
            outputs=self.outputs.resolve(inputs, allocations),
        )
        return self.compiled.invoke(
            invocation,
            checkpoint_memory_budget=checkpoint_memory_budget,
            checkpoint_policy=checkpoint_policy,
        )


def _resolve_module_program_controls(
    module: Any,
    arguments: tuple[Any, ...],
    keywords: Mapping[str, Any],
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    definition = type(module)._module_definition
    bound = definition.signature.bind_partial(*arguments, **keywords)
    for parameter in definition.signature.parameters.values():
        annotation = definition.resolved_annotations.get(parameter.name)
        kind = _program_control_kind(annotation)
        if kind is None:
            continue
        if parameter.name in bound.arguments:
            value = bound.arguments[parameter.name]
        elif parameter.default is not inspect.Parameter.empty:
            value = parameter.default
        else:
            raise TypeError(f"Module invocation must provide {kind} control parameter {parameter.name!r}")
        bound.arguments[parameter.name] = value
    return bound.args, bound.kwargs


def execute_module_primal(module: Any, arguments: tuple[Any, ...], keywords: Mapping[str, Any]) -> Any:
    arguments, keywords = _resolve_module_program_controls(module, arguments, keywords)
    key = module._program_specialization_key(arguments, keywords, variant="primal")
    specialization = module._program_cache.get(key)
    if not isinstance(specialization, _PrimalSpecialization):
        capture, outputs = _capture_module(module, _parameter_types_from_values(module, arguments, keywords))
        template = ProgramTemplate.compile(capture)
        invocation = template.bind(capture, outputs)
        specialization = _primal_specialization(module, capture, outputs, invocation)
        module._program_cache[key] = specialization
    return specialization.invoke(arguments, keywords)


def execute_module_vjp(
    expression: ModuleVjpExpression,
    arguments: tuple[Any, ...],
    keywords: Mapping[str, Any],
) -> tuple[Any, Any]:
    module = expression.module
    key = module._program_specialization_key(
        arguments,
        keywords,
        variant=(
            "vjp",
            expression.wrt,
            expression.outputs,
            expression.planning_policy,
        ),
    )
    specialization = module._program_cache.get(key)
    if not isinstance(specialization, _VjpSpecialization):
        capture, outputs = _capture_module(
            module,
            _parameter_types_from_values(module, arguments, keywords),
        )
        if capture.graphics_calls:
            raise TypeError("graphics Module programs do not support autodiff")
        template = ProgramTemplate.compile(capture)
        invocation = template.bind(capture, outputs)
        selected_outputs = expression.outputs or tuple(invocation.graph.outputs)
        unknown_outputs = tuple(path for path in selected_outputs if path not in invocation.graph.outputs)
        if unknown_outputs:
            raise ValueError(f"Program autodiff outputs do not identify Module output paths: {unknown_outputs}")
        from .program_frontend import parse_program

        parsed_program = parse_program(
            invocation,
            vjp_wrt=expression.wrt,
            vjp_outputs=selected_outputs,
            autodiff_planning_policy=expression.planning_policy,
        )
        from ._runtime.program_autodiff import compile_program_autodiff

        compiled = compile_program_autodiff(parsed_program, template)
        allocations, _, output_recipe = _capture_recipes(capture, outputs)
        specialization = _VjpSpecialization(
            compiled,
            invocation,
            type(module)._module_definition.signature,
            allocations,
            output_recipe,
        )
        module._program_cache[key] = specialization
    budget = getattr(module, "checkpoint_memory_budget", None)
    if not isinstance(budget, int):
        budget = None
    return specialization.invoke(
        arguments,
        keywords,
        checkpoint_memory_budget=budget,
        checkpoint_policy=expression.planning_policy if budget is not None else "",
    )


__all__ = [
    "ModuleVjpExpression",
    "ProgramTemplate",
    "execute_module_primal",
]
