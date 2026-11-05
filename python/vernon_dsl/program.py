"""Compilation and invocation of captured Module programs."""

from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
from collections import ChainMap
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast, get_args

import numpy as np

from ._formats import TEXTURE_FORMATS_BY_NAME
from ._runtime.tensor import (
    TensorStorage,
    TensorView,
    _adopt_device_buffer,
    _ensure_device_residency,
    _owner_controls,
    _PreservedOwners,
)
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
from .types import SpecializationAssignment

if TYPE_CHECKING:
    from .module import ModuleDefinition


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


def _capture_one_node_program(
    root: Any,
    parameter_types: Mapping[str, Any],
    capture_operation: Any,
    output_names: tuple[str, ...] | None = (),
) -> tuple[ProgramCapture, ProgramTemplate, ProgramInvocation, Any]:
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

    return capture, template, invocation, parse_program(invocation)


def _one_node_program(
    root: Any,
    parameter_types: Mapping[str, Any],
    capture_operation: Any,
    output_names: tuple[str, ...] | None = (),
) -> tuple[ProgramTemplate, ProgramInvocation, Any]:
    _, template, invocation, parsed = _capture_one_node_program(
        root,
        parameter_types,
        capture_operation,
        output_names,
    )
    return template, invocation, parsed


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


def _resolve_attachments(
    pipeline: Any, render_pass: Any, specializations: tuple[SpecializationAssignment, ...]
) -> tuple[_AttachmentDescription, ...]:
    """Determine a graphics node's attachment structure from whichever source is authoritative.

    A pipeline that declares target formats is authoritative, and a RenderPass bound to it must agree -- the same
    compatibility rule a Vulkan pipeline imposes on the render pass it is used in. A pipeline that declares none can
    still run live by reading the structure off the concrete RenderPass, but it cannot be cooked, because an asset
    has no RenderPass to read.
    """

    try:
        declared = pipeline._target_formats(specializations)
    except ValueError as error:
        raise TypeError(str(error)) from None
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
        element = ConcreteType("scalar", TEXTURE_FORMATS_BY_NAME[description.format_name].shader_scalar)
        logical = ConcreteType("texture", "Texture", ("2d", element, description.format_name, "attachment"))
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
    specializations: tuple[SpecializationAssignment, ...]


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
    specializations: tuple[SpecializationAssignment, ...]


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
        specializations: tuple[SpecializationAssignment, ...],
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
        self.ops.append(
            CapturedKernelCall(
                name,
                kernel,
                arguments,
                dispatch_grid,
                tuple(specializations),
            )
        )

    def capture_graphics(
        self,
        pipeline: Any,
        arguments: Mapping[str, Any],
        render_pass: Any,
        draw: Any,
        dynamic_state: Any,
        specializations: tuple[SpecializationAssignment, ...],
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
                tuple(specializations),
            )
        )

    def attachment_output(
        self,
        render_pass: GraphControlInput,
        aspect: str,
        location: int | None,
    ) -> GraphAttachmentOutput:
        projection = AttachmentProjection(render_pass.ref, aspect, location)
        if not any(
            isinstance(operation, CapturedGraphicsCall) and operation.render_pass_ref == projection.control
            for operation in self.ops
        ):
            raise TypeError("attachment output must follow a graphics call using the same RenderPass")
        return GraphAttachmentOutput(render_pass, projection)

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
    specializations: tuple[SpecializationAssignment, ...]


@dataclass(frozen=True)
class GraphicsCallTemplate:
    name: str
    pipeline: Any
    parameter_names: tuple[str, ...]
    parameters: tuple[KernelParameter, ...]
    specializations: tuple[SpecializationAssignment, ...]
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
                frontends = captured.pipeline._frontends(captured.specializations)
                for stage_index, stage in enumerate(captured.pipeline._stages):
                    function = stage.function
                    frontend = frontends[stage_index]
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
                        previous = parameters.get(typed.name)
                        parameter = KernelParameter(typed.name, typed.access.value, False, False)
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
                        captured.specializations,
                        tuple(implementations),
                        tuple(sorted(structs.items())),
                    )
                )
                continue
            assert isinstance(captured, CapturedKernelCall)
            lowered = captured.kernel._lower(captured.specializations)
            frontend = lowered.frontend
            parameter_names = captured.kernel._parameter_names(lowered)
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
                parameters.append(
                    KernelParameter(
                        parameter.name,
                        parameter.access.value,
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
                    captured.specializations,
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

        def resolve_resource_reference(value: Any) -> Any:
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
                if (
                    not isinstance(template, GraphicsCallTemplate)
                    or template.pipeline is not captured.pipeline
                    or template.specializations != captured.specializations
                ):
                    raise RuntimeError("Module graphics sequence changed for an existing Program specialization")
                first_slot = len(slots)
                arguments = tuple(captured.arguments[name] for name in template.parameter_names)
                resolved_arguments = tuple(resolve_resource_reference(value) for value in arguments)
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
                    _resolve_attachments(captured.pipeline, captured.render_pass, captured.specializations),
                    captured.render_pass_ref,
                    graphics_attachment_cache,
                )
                for attachment in attachments:
                    graph.import_control_resource(attachment)
                operation = graph.append_graphics(
                    name=template.name,
                    pipeline=template.pipeline,
                    bindings=dict(zip(template.parameter_names, resolved_arguments, strict=True)),
                    binding_slots=binding_slots,
                    control_slots=control_slots,
                    parameters=template.parameters,
                    attachments=attachments,
                    color_count=color_count,
                    specializations=template.specializations,
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
                or template.specializations != captured.specializations
            ):
                raise RuntimeError("Module kernel sequence changed for an existing Program specialization")
            first_slot = len(slots)
            slots.extend(captured.arguments)
            resolved_arguments = tuple(resolve_resource_reference(value) for value in captured.arguments)
            bindings = dict(zip(template.parameter_names, resolved_arguments, strict=True))
            binding_slots = {name: first_slot + index for index, name in enumerate(template.parameter_names)}
            operation = graph.append_kernel(
                name=template.name,
                kernel=template.kernel,
                bindings=bindings,
                binding_slots=binding_slots,
                parameters=template.parameters,
                grid=template.grid,
                specializations=template.specializations,
            )
            operation_bindings[operation.id] = captured.arguments
        graph.set_outputs(
            {path: resolve_resource_reference(value) for path, value in flatten_program_outputs(outputs).items()}
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
        if isinstance(value, GraphAttachmentOutput):
            from .render import color_output, depth_output

            render_pass = self.inputs[value.render_pass.name]
            if value.projection.aspect == "color":
                assert value.projection.location is not None
                return color_output(render_pass, location=value.projection.location)
            return depth_output(render_pass)
        return value

    @property
    def graphics_controls(self) -> Mapping[int, Mapping[str, tuple[int, Any]]]:
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
        from ._runtime.session import _invocation_context

        with _invocation_context():
            return execute_module_vjp(self, arguments, keywords)

    def bind(self, *, gradients: Mapping[str, Any] | None = None, **values: Any) -> BoundModuleVjp:
        """This VJP with the `forward()` inputs `values` bound once, and the gradients of those in `wrt` written into
        `gradients`, or into Storages `bind` allocates (specs/language/contract.md section 8)."""
        return BoundModuleVjp(self, values, gradients)


def _capture_module(module: Any, parameter_types: Mapping[str, Any]) -> tuple[ProgramCapture, Any]:
    from .frontend.module_ast import interpret_module_forward

    return interpret_module_forward(module, parameter_types)


def _bind_program_inputs(
    definition: ModuleDefinition, arguments: tuple[Any, ...], keywords: Mapping[str, Any]
) -> dict[str, Any]:
    return {name: value for name, _, value in definition.bind_leaves(arguments, keywords)}


def _merge_storage_shapes(
    left: tuple[Any, ...],
    right: tuple[Any, ...],
    *,
    parameter: str,
) -> tuple[Any, ...]:
    from .types import dyn

    if len(left) != len(right):
        raise TypeError(f"conflicting ranks for Module parameter {parameter!r}")
    merged = []
    for left_extent, right_extent in zip(left, right, strict=True):
        if left_extent is dyn:
            merged.append(right_extent)
        elif right_extent is dyn:
            merged.append(left_extent)
        elif left_extent == right_extent:
            merged.append(left_extent)
        else:
            raise TypeError(f"conflicting shapes for Module parameter {parameter!r}")
    return tuple(merged)


def _access_capabilities(access: str) -> frozenset[str]:
    if access == "read":
        return frozenset({"read"})
    if access == "write":
        return frozenset({"write"})
    if access == "read_write":
        return frozenset({"read", "write"})
    raise TypeError(f"invalid Storage access {access!r}")


def _join_storage_access(left: str, right: str) -> str:
    capabilities = _access_capabilities(left) | _access_capabilities(right)
    return "read_write" if len(capabilities) == 2 else next(iter(capabilities))


def _module_storage_constraint(annotation: Any) -> Any:
    from .frontend.runtime_types import StorageParameterConstraint, storage_parameter_constraint

    if annotation is TensorStorage:
        return StorageParameterConstraint(None, None, None, False)
    return storage_parameter_constraint(annotation)


def _resolve_storage_parameter(
    name: str,
    annotation: Any,
    inferred: Any,
    *,
    value_dtype: Any | None = None,
    value_shape: tuple[int, ...] | None = None,
    value_access: str | None = None,
) -> Any:
    from .frontend.runtime_types import (
        RuntimeParameterDescriptor,
        StorageParameterConstraint,
        storage_access,
    )
    from .types import TypeExpr

    explicit = _module_storage_constraint(annotation)
    reflected = None
    if inferred is not None:
        if not isinstance(inferred, TypeExpr) or inferred.name != "TensorView" or len(inferred.arguments) != 3:
            raise TypeError(f"invalid reflected Storage type for Module parameter {name!r}")
        dtype, shape, access = inferred.arguments
        reflected = StorageParameterConstraint(dtype, tuple(shape), storage_access(access), False)
    if explicit is None and reflected is None and value_dtype is None:
        return None

    dtype = explicit.dtype if explicit is not None else None
    if reflected is not None:
        if dtype is not None and dtype != reflected.dtype:
            raise TypeError(f"conflicting element types for Module parameter {name!r}")
        dtype = reflected.dtype if dtype is None else dtype
    if value_dtype is not None:
        if dtype is not None and dtype != value_dtype:
            raise TypeError(f"runtime element type conflicts with Module parameter {name!r}")
        dtype = value_dtype if dtype is None else dtype

    shape = explicit.shape if explicit is not None else None
    if reflected is not None and reflected.shape is not None:
        shape = reflected.shape if shape is None else _merge_storage_shapes(shape, reflected.shape, parameter=name)
    if shape is None and value_shape is not None:
        shape = value_shape
    elif shape is not None and value_shape is not None:
        _merge_storage_shapes(shape, value_shape, parameter=name)

    declared_access = explicit.access if explicit is not None else None
    reflected_access = reflected.access if reflected is not None else None
    if declared_access is not None and reflected_access is not None:
        if not _access_capabilities(reflected_access) <= _access_capabilities(declared_access):
            raise TypeError(f"declared access does not cover uses of Module parameter {name!r}")
        access = declared_access
    else:
        access = declared_access or reflected_access or value_access
    if access is not None and value_access is not None:
        if not _access_capabilities(access) <= _access_capabilities(value_access):
            raise TypeError(f"runtime view access is insufficient for Module parameter {name!r}")

    if dtype is None or shape is None or access is None:
        raise TypeError(
            f"cannot infer Module parameter {name!r} Storage contract; "
            "provide TensorStorage[element, (shape,), access], "
            "use it in a typed Program call, or supply an explicit export argument"
        )
    as_view = explicit.as_view if explicit is not None else False
    return RuntimeParameterDescriptor.storage(dtype, tuple(shape), access, as_view)


def _reflect_module_parameter_types(module: Any) -> dict[str, Any]:
    """Infer public resource types from nested typed Program calls."""

    from ._runtime.pipeline import Pipeline
    from .frontend.model import SemanticCategory
    from .module import Module
    from .types import TypeExpr, dyn, read_write

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
        if existing_dtype != dtype:
            raise TypeError(
                f"conflicting reflected types for Module parameter {name!r} in {type(owner).__name__}.forward"
            )
        merged_shape = _merge_storage_shapes(tuple(existing_shape), tuple(shape), parameter=name)
        if existing_access != access:
            constraints[name] = TypeExpr("TensorView", (dtype, merged_shape, read_write))
        else:
            constraints[name] = TypeExpr("TensorView", (dtype, merged_shape, access))

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
            definition = type(target)._module_definition
            inferred = dict(infer(target))
            for parameter in definition.signature.parameters.values():
                annotation = definition.resolved_annotations.get(parameter.name)
                explicit = _module_storage_constraint(annotation)
                if explicit is None:
                    continue
                candidate = inferred.get(parameter.name)
                if candidate is None and (explicit.dtype is None or explicit.shape is None or explicit.access is None):
                    continue
                descriptor = _resolve_storage_parameter(
                    parameter.name,
                    annotation,
                    candidate,
                )
                metadata = descriptor.storage_metadata
                shape = tuple(dyn if extent == "?" else extent for extent in metadata.shape)
                inferred[parameter.name] = TypeExpr(
                    "TensorView",
                    (metadata.dtype, shape, metadata.access),
                )
            return definition.signature, inferred
        if isinstance(target, Pipeline):
            reflected = {}
            for name, descriptor in target._static_parameter_types().items():
                if descriptor.kind is not SemanticCategory.STORAGE:
                    continue
                metadata = descriptor.storage_metadata
                shape = tuple(dyn if extent == "?" else extent for extent in metadata.shape)
                reflected[name] = TypeExpr("TensorView", (metadata.dtype, shape, metadata.access))
            # Binding-group leaves are reflected by dotted path; the call binds their root parameter.
            roots = dict.fromkeys(name.split(".", 1)[0] for name in reflected)
            signature = inspect.Signature(
                inspect.Parameter(root, inspect.Parameter.POSITIONAL_OR_KEYWORD) for root in roots
            )
            return signature, reflected
        function = getattr(target, "_function", None)
        if function is None:
            return None
        from .frontend.runtime_types import resolved_annotations

        try:
            annotations = resolved_annotations(function)
        except TypeError:
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
                for leaf, candidate in types.items():
                    if leaf == target_name:
                        merge(constraints, expression.id, candidate, owner)
                    elif leaf.startswith(target_name + "."):
                        merge(constraints, expression.id + leaf[len(target_name) :], candidate, owner)
        active.remove(identity)
        cache[identity] = constraints
        return constraints

    return infer(module)


def _module_parameter_types(module: Any) -> dict[str, Any]:
    from ._params import params_leaf_annotations
    from .frontend.model import SemanticCategory
    from .frontend.runtime_types import runtime_parameter_descriptor
    from .operation_graph import ProgramControlDescriptor

    definition = type(module)._module_definition
    annotations = definition.resolved_annotations
    reflected = _reflect_module_parameter_types(module)

    def descriptor(name: str, annotation: Any) -> Any:
        inferred = reflected.get(name)
        if _module_storage_constraint(annotation) is not None or inferred is not None:
            return _resolve_storage_parameter(name, annotation, inferred)
        if annotation is None:
            raise TypeError(
                f"cannot infer Module parameter {name!r}; annotate it as "
                "TensorStorage[element, (shape,), access] or TensorView[element, (shape,), access], "
                "use it in a typed Program call, "
                "or supply an explicit export argument"
            )
        result = runtime_parameter_descriptor(annotation)
        if result.kind is SemanticCategory.STORAGE:
            raise RuntimeError("Storage annotation bypassed canonical Module constraint resolution")
        return result

    types: dict[str, Any] = {}
    for parameter in definition.signature.parameters.values():
        annotation = annotations.get(parameter.name)
        control_kind = _program_control_kind(annotation)
        if control_kind is not None:
            types[parameter.name] = ProgramControlDescriptor(control_kind, annotation)
            continue
        if parameter.default is not inspect.Parameter.empty:
            continue
        for name, leaf_annotation in params_leaf_annotations(parameter.name, annotation):
            types[name] = descriptor(name, leaf_annotation)
    return types


def _parameter_types_from_values(
    module: Any,
    arguments: tuple[Any, ...],
    keywords: Mapping[str, Any],
) -> dict[str, Any]:
    from ._dtypes import scalar_name
    from .frontend.model import SemanticCategory
    from .frontend.runtime_types import (
        runtime_parameter_descriptor,
        validate_host_value,
    )
    from .operation_graph import ProgramControlDescriptor

    def tensor_dtype(value: TensorStorage) -> Any:
        if value._element_type is not None:
            return value._element_type
        name = scalar_name(value.dtype)
        if name is None:
            raise TypeError(f"cannot allocate a Module transient for dtype {value.dtype}")
        types = __import__("vernon_dsl.types", fromlist=[name])
        return getattr(types, name)

    reflected = _reflect_module_parameter_types(module)

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
        explicit_storage = _module_storage_constraint(annotation)
        if isinstance(value, TensorView):
            owner = value.owner
            if not isinstance(owner, TensorStorage):
                raise TypeError("Module.forward() TensorView parameters must borrow TensorStorage")
            if explicit_storage is not None and not explicit_storage.as_view:
                raise TypeError(f"Module.forward() owner parameter {name!r} requires TensorStorage, not TensorView")
            dtype = value.element_type if value.element_type is not None else tensor_dtype(owner)
            return _resolve_storage_parameter(
                name,
                annotation,
                reflected.get(name),
                value_dtype=dtype,
                value_shape=tuple(value.shape),
                value_access=str(value.access),
            )
        if isinstance(value, TensorStorage):
            return _resolve_storage_parameter(
                name,
                annotation,
                reflected.get(name),
                value_dtype=tensor_dtype(value),
                value_shape=tuple(value.shape),
                value_access="read_write",
            )
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

    return {
        name: descriptor(annotation, value, name)
        for name, annotation, value in type(module)._module_definition.bind_leaves(arguments, keywords)
    }


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


def _parse_pipeline_program(
    pipeline: Any,
    specializations: tuple[SpecializationAssignment, ...] = (),
) -> Any:
    """Capture a bare vd.pipeline(...) as a one-node Program, the way a Module calling one is captured.

    The render pass, draw, and dynamic state stay symbolic controls with no prototype, so nothing about a particular
    invocation reaches the manifest. Attachment formats come from the pipeline's declared targets instead.
    """

    frontends = pipeline._frontends(specializations)
    parameter_types = pipeline._parameter_types(None, frontends, None, None, None)
    arguments = tuple(name for name in parameter_types if not name.startswith("__"))
    _, _, parsed = _one_node_program(
        pipeline,
        parameter_types,
        lambda capture, inputs: capture.capture_graphics(
            pipeline,
            {name: inputs[name] for name in arguments},
            inputs["__render_pass"],
            inputs["__draw"],
            inputs["__dynamic_state"],
            specializations,
        ),
        output_names=None,
    )
    return parsed


def _parse_kernel_program(
    kernel: Any,
    *,
    specializations: tuple[SpecializationAssignment, ...] = (),
    transform: Any | None = None,
) -> Any:
    """Capture a compute Kernel or Kernel VJP as the same one-node Program used by Modules."""

    return _capture_kernel_program(kernel, specializations=specializations, transform=transform)[3]


def _capture_kernel_program(
    kernel: Any,
    *,
    specializations: tuple[SpecializationAssignment, ...] = (),
    transform: Any | None = None,
) -> tuple[ProgramCapture, ProgramTemplate, ProgramInvocation, Any, tuple[str, ...], Any]:
    """Capture a compute Kernel as an executable canonical one-node Program."""

    from .frontend.runtime_types import runtime_parameter_descriptor
    from .types import u32

    lowered = kernel._lower(specializations)
    annotations = kernel._resolved_annotations(specializations)
    parameter_names = kernel._parameter_names(lowered)
    parameter_types: dict[str, Any] = {}
    for name in parameter_names:
        annotation = annotations.get(name)
        if annotation is None:
            raise TypeError(f"kernel argument {name!r} requires a runtime annotation")
        parameter_types[name] = runtime_parameter_descriptor(annotation)
    for axis in "xyz":
        parameter_types[f"__grid_{axis}"] = runtime_parameter_descriptor(u32)

    capture, template, invocation, parsed = _capture_one_node_program(
        kernel,
        parameter_types,
        lambda capture, inputs: capture.capture_kernel(
            kernel,
            tuple(inputs[name] for name in parameter_names),
            tuple(inputs[f"__grid_{axis}"] for axis in "xyz"),
            specializations,
        ),
        output_names=None,
    )
    if transform is not None:
        from .program_frontend import parse_program

        selected_outputs = transform.output_cotangents or tuple(invocation.graph.outputs)
        parsed = parse_program(
            invocation,
            vjp_wrt=transform.wrt,
            vjp_outputs=selected_outputs,
            autodiff_planning_policy=transform.planning_policy,
        )
    return capture, template, invocation, parsed, parameter_names, lowered.frontend


@dataclass(frozen=True)
class _AllocationSpec:
    dtype: Any
    shape: tuple[int | str, ...]
    initializer: str
    values: Any = None
    like_source: str | None = None
    like_key: str | int | None = None

    def create(self, inputs: Mapping[str, Any], allocations: tuple[TensorStorage, ...]) -> TensorStorage:
        if self.initializer == "from_values":
            return TensorStorage.from_values(self.values, dtype=self.dtype)
        shape = self.shape
        if self.like_source is not None:
            if self.like_source == "input":
                if not isinstance(self.like_key, str):
                    raise RuntimeError("input-relative allocation has no input name")
                owner = inputs[self.like_key]
            else:
                if not isinstance(self.like_key, int):
                    raise RuntimeError("allocation-relative allocation has no source index")
                owner = allocations[self.like_key]
            if not isinstance(owner, (TensorStorage, TensorView)):
                raise TypeError("Module relative allocation source must be TensorStorage or TensorView")
            shape = tuple(owner.shape)
        return self.create_shaped(shape)

    def shape_of(self, inputs: Mapping[str, Any], shapes: list[tuple[int, ...] | None]) -> tuple[int, ...] | None:
        """The allocation's shape given the shapes of the allocations before it, without allocating; None when it is
        not known before the Program runs."""
        if self.initializer == "from_values":
            return None
        if self.like_source == "input":
            owner = inputs.get(self.like_key) if isinstance(self.like_key, str) else None
            if not isinstance(owner, (TensorStorage, TensorView)):
                return None
            return tuple(owner.shape)
        if self.like_source is not None:
            return shapes[self.like_key] if isinstance(self.like_key, int) else None
        if any(not isinstance(extent, int) for extent in self.shape):
            return None
        return cast(tuple[int, ...], self.shape)

    def create_shaped(self, shape: tuple[Any, ...]) -> TensorStorage:
        if self.initializer == "from_values":
            return TensorStorage.from_values(self.values, dtype=self.dtype)
        if any(not isinstance(extent, int) for extent in shape):
            raise RuntimeError("dynamic Module allocation requires an input-relative shape")
        numpy_dtype, layout = self._storage_dtype
        allocate = np.zeros if self.initializer in {"zeros", "zeros_like"} else np.empty
        return TensorStorage._adopt(
            allocate(shape, dtype=numpy_dtype), self.dtype if layout is not None else None, layout
        )

    @functools.cached_property
    def _storage_dtype(self) -> tuple[np.dtype[Any], Any]:
        return TensorStorage._storage_dtype(self.dtype)


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
            if self.as_view:
                return owner
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
        if self.kind in {"dataclass", "params"}:
            cls, fields = self.value
            return cls(**{name: member.resolve(inputs, allocations) for name, member in fields})
        raise RuntimeError(f"unknown Program output recipe {self.kind!r}")

    def allocation_indices(self) -> Iterator[int]:
        """The indices of the allocations the outputs publish."""
        if self.kind == "leaf":
            if self.value.source == "allocation":
                yield self.value.key
            return
        members = self.value[1] if self.kind in {"dataclass", "params"} else self.value
        for member in members:
            yield from (member[1] if self.kind in {"mapping", "dataclass", "params"} else member).allocation_indices()


class _ProgramLink:
    """A Module specialization linked as a schedule (specs/runtime/architecture_v2.md section 16.1): the schedule, the
    boundary slots' paths and directions in ABI order, and which allocations are outputs, which the call creates; the
    others are the schedule's transients."""

    __slots__ = ("attempted", "device_state", "outputs", "schedule", "slots", "zeroed")

    def __init__(self) -> None:
        self.attempted = False
        self.schedule: Any = None
        self.slots: tuple[tuple[str, bool], ...] = ()
        self.outputs: frozenset[int] = frozenset()
        # The outputs that start as zeros, which the schedule zeroes rather than the call uploading them.
        self.zeroed: frozenset[int] = frozenset()
        # The session of a GPU schedule, in which each call's fresh outputs get their device buffers.
        self.device_state: Any = None


@dataclass(frozen=True)
class _PrimalSpecialization:
    program: Any
    template: ProgramTemplate
    invocation: ProgramInvocation
    definition: ModuleDefinition
    allocations: tuple[_AllocationSpec, ...]
    outputs: _TreeRecipe
    native_program: Any
    link: _ProgramLink = dataclasses.field(default_factory=_ProgramLink)

    def invoke(self, arguments: tuple[Any, ...], keywords: Mapping[str, Any]) -> Any:
        return self.invoke_inputs(_bind_program_inputs(self.definition, arguments, keywords))

    def invoke_inputs(self, inputs: dict[str, Any]) -> Any:
        if not self.link.attempted:
            self._link()
        if self.link.schedule is not None:
            linked = self._call_linked(inputs)
            if linked is not None:
                return linked[0]
        allocation_list: list[TensorStorage] = []
        for spec in self.allocations:
            allocation_list.append(spec.create(inputs, tuple(allocation_list)))
        allocations = tuple(allocation_list)
        outputs = self.outputs.resolve(inputs, allocations)
        result = self.native_program.invoke(
            dataclasses.replace(
                self.invocation,
                inputs=dict(inputs),
                outputs=outputs,
            )
        )
        return result

    def _call_linked(self, inputs: Mapping[str, Any]) -> tuple[Any] | None:
        """Runs the linked schedule, creating only the allocations the outputs publish; None when an argument is
        outside what the schedule packs, and the Program then runs instead."""
        link = self.link
        allocations = _create_linked_allocations(self.allocations, inputs, link.outputs, link.zeroed, link.device_state)
        if allocations is None:
            return None
        outputs = self.outputs.resolve(inputs, tuple(allocations))
        published = flatten_program_outputs(outputs)
        state = link.device_state
        arguments = []
        for path, is_input in link.slots:
            source = inputs if is_input else published
            if path not in source:
                return None
            value = source[path]
            # An input only the host holds is uploaded here rather than by a Program call.
            if (
                state is not None
                and is_input
                and isinstance(value, (TensorStorage, TensorView))
                and not _ensure_device_residency(value, state)
            ):
                return None
            arguments.append(value)
        return (outputs,) if link.schedule.call(tuple(arguments), None) else None

    def _link(self) -> None:
        from ._runtime.session import _session_state, cpu

        link = self.link
        link.attempted = True
        state = _session_state()
        linked = _linked_program(state, getattr(self.native_program, "deployment", None))
        if linked is None:
            return
        program, graphs = linked
        graph = graphs.get("forward")
        if graph is None:
            return
        allocated = [row["storage"] for row in graph["inputs"] if row["tag"] == "allocation"]
        if len(allocated) != len(self.allocations):
            return
        slots = _graph_slots(program, "forward")
        published = {row.get("storage_id") for row in slots if row["direction"] == "output"}
        outputs = frozenset(index for index, storage in enumerate(allocated) if storage in published)
        zeroed = _zeroed_transients(self.allocations, allocated, outputs)
        if zeroed is None:
            return
        schedule = _link_graph(state, self.native_program.deployment, graph, zeroed[0], [])
        if schedule is None:
            return
        link.slots = tuple((row["path"], row["direction"] == "input") for row in slots)
        link.outputs = outputs
        link.zeroed = zeroed[1]
        link.device_state = None if state.arch == cpu else state
        link.schedule = schedule


def _linked_program(state: Any, deployment: Any) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """A deployment's variant `program` JSON and its graphs by name, when its session can link schedules and every
    Stage is a compute kernel; None otherwise."""
    import json

    from ._runtime.session import cpu, cuda

    if state.arch not in {cpu, cuda} or deployment is None or not deployment.linked_stages:
        return None
    program = json.loads(deployment.program_json)
    return program, {graph["name"]: graph for graph in program["graphs"]}


def _graph_slots(program: Mapping[str, Any], graph: str) -> list[dict[str, Any]]:
    """The ABI boundary slots of one graph that are schedule slots, in ABI order, which is the order of its schedule's
    first slots. An output that publishes the Storage of an input view is that input's slot, so it has none."""
    values = program["values"]
    inputs: set[int] = set()
    slots = []
    for row in program["abi"]["boundary_slots"]:
        if values[row["value"]]["origin"].get("graph") != graph:
            continue
        storage = row.get("storage_id")
        if row["direction"] == "output" and storage in inputs:
            continue
        if row["direction"] == "input" and storage is not None:
            inputs.add(storage)
        slots.append(row)
    return slots


def _zeroed_transients(
    allocations: tuple[_AllocationSpec, ...], allocated: list[int], exported: frozenset[int]
) -> tuple[list[int], frozenset[int]] | None:
    """The Storages of the allocations a schedule zeroes, and which of the `exported` allocations, those a call
    creates, are among them, so a call does not upload their zeros; None when a transient starts from values, which a
    schedule does not upload."""
    zeroed = []
    created = []
    for index, spec in enumerate(allocations):
        if spec.initializer == "from_values" and index not in exported:
            return None
        if spec.initializer in {"zeros", "zeros_like"}:
            zeroed.append(allocated[index])
            if index in exported:
                created.append(index)
    return zeroed, frozenset(created)


def _link_graph(state: Any, deployment: Any, graph: Mapping[str, Any], zeroed: list[int], exposed: list[int]) -> Any:
    """One graph linked as a schedule with the Stages its nodes run, or None when it uses what schedules do not link."""
    from ._runtime.session import cpu

    used = {node["stage"] for node in graph["nodes"]}
    try:
        return state.native._ScheduleKernel.link_program(
            state.native_runtime,
            deployment.program_json,
            graph["name"],
            [stage for stage in deployment.linked_stages if stage[0] in used],
            zeroed,
            exposed,
            TensorStorage,
            TensorView,
            residency_key=None if state.arch == cpu else state.identity,
        )
    except (RuntimeError, ValueError):
        return None


def _capture_recipes(
    capture: ProgramCapture,
    outputs: Any,
) -> tuple[
    tuple[_AllocationSpec, ...],
    _TreeRecipe,
]:
    from ._params import is_params
    from .host_values import declared_fields

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
        if isinstance(value, (GraphAttachmentOutput, GraphBuffer, GraphResourceInput, TensorStorage, TensorView)):
            return _TreeRecipe("leaf", recipe(value))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return _TreeRecipe(
                "dataclass",
                (
                    type(value),
                    tuple((field.name, tree(getattr(value, field.name))) for field in dataclasses.fields(value)),
                ),
            )
        if is_params(type(value)):
            return _TreeRecipe(
                "params",
                (type(value), tuple((name, tree(getattr(value, name))) for name in declared_fields(type(value)))),
            )
        if isinstance(value, Mapping):
            return _TreeRecipe("mapping", tuple((name, tree(member)) for name, member in value.items()))
        if isinstance(value, tuple):
            return _TreeRecipe("tuple", tuple(tree(member) for member in value))
        if isinstance(value, list):
            return _TreeRecipe("list", tuple(tree(member) for member in value))
        raise TypeError("Module output contains a value that cannot be reconstructed from Program IR")

    allocation_specs = []
    for alloc in capture.allocs:
        like_source = None
        like_key = None
        if alloc.like is not None:
            like_key = input_name(alloc.like)
            if like_key is not None:
                like_source = "input"
            else:
                like_key = allocation_index(alloc.like)
                if like_key is None:
                    raise TypeError("Module relative allocation references an unregistered Program buffer")
                like_source = "allocation"
        allocation_specs.append(
            _AllocationSpec(
                alloc.result.dtype,
                tuple(alloc.result.shape),
                alloc.kind,
                alloc.values,
                like_source,
                like_key,
            )
        )
    allocations = tuple(allocation_specs)
    return allocations, tree(outputs)


def _primal_specialization(
    module: Any,
    capture: ProgramCapture,
    outputs: Any,
    invocation: ProgramInvocation,
) -> _PrimalSpecialization:
    allocations, output_recipe = _capture_recipes(capture, outputs)
    from .program_frontend import parse_program

    parsed = parse_program(invocation)
    from ._runtime.program_autodiff import compile_program

    return _PrimalSpecialization(
        parsed,
        invocation.template,
        invocation,
        type(module)._module_definition,
        allocations,
        output_recipe,
        compile_program(parsed, invocation.template),
    )


class _VjpLink:
    """A Module VJP's linked VJP (specs/runtime/architecture_v2.md section 16.3) and how a call passes it Python values.
    The forward slots are the forward boundary's, each `(is_input, path)`; the backward slots are the cotangents and
    gradients, each `(role, path)`. The backward call reads the forward call's arguments and residuals itself."""

    __slots__ = ("attempted", "backward_slots", "created", "device_state", "forward_slots", "seeds", "vjp", "zeroed")

    def __init__(self) -> None:
        self.attempted = False
        self.vjp: Any = None
        self.forward_slots: tuple[tuple[bool, str], ...] = ()
        self.backward_slots: tuple[tuple[str, str], ...] = ()
        # The allocations a call creates, its outputs; the others are the schedule's transients.
        self.created: frozenset[int] = frozenset()
        # The created allocations that start as zeros. The schedule zeroes its owned Storages, so a call does not
        # upload their zeros.
        self.zeroed: frozenset[int] = frozenset()
        self.device_state: Any = None
        # The ones each omitted cotangent seeds, per output path, shape and element type; no call writes them.
        self.seeds: dict[tuple[Any, ...], TensorStorage] = {}


class _LinkedPullback:
    """The pullback of a linked Module VJP call: the backward schedule over the call's residuals. The residuals point
    into the memory of the call's inputs, which a later write leaves to the pullback. A backward call whose values do
    not pack natively, or that takes a carrier, runs as the general pullback over the same residuals."""

    def __init__(
        self,
        specialization: _VjpSpecialization,
        inputs: Mapping[str, Any],
        published: Mapping[str, Any],
        call: Any,
    ) -> None:
        self._specialization = specialization
        self._inputs = inputs
        self._published = published
        self._call = call
        self._preserved = _PreservedOwners(_owner_controls(inputs.values()))
        from ._runtime.session import _execution_context

        # The general pullback runs in the forward call's context, which is no longer active when it is built.
        self._context = _execution_context()
        self._general: Any = None

    # The residuals' tape telemetry, which the general pullback over them reports.
    _TELEMETRY = frozenset(
        {
            "allocated_tape_bytes",
            "checkpoint_plan",
            "estimated_tape_bytes",
            "logical_residual_bytes",
            "pass_telemetry",
            "peak_runtime_managed_bytes",
            "peak_temporary_bytes",
            "recomputation_factor",
            "resident_tape_bytes",
            "tape_context_limit_bytes",
        }
    )

    def __getattr__(self, name: str) -> Any:
        if name in _LinkedPullback._TELEMETRY:
            return getattr(self._general_pullback(), name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __call__(self, cotangents: Any = None) -> dict[str, Any]:
        return self.apply_with_carrier(cotangents, ())

    def apply_with_carrier(self, cotangents: Any, carrier_shape: tuple[int, ...]) -> dict[str, Any]:
        if not carrier_shape:
            link = self._specialization.link
            arguments, gradients = _backward_arguments(
                link, link.backward_slots, self._inputs, self._published, cotangents
            )
            if arguments is not None:
                try:
                    if self._call.backward(arguments):
                        return gradients
                except ValueError:
                    pass
        return self._general_pullback().apply_with_carrier(cotangents, carrier_shape)

    def _general_pullback(self) -> Any:
        if self._general is None:
            from ._runtime.autodiff import _program_derivative_groups
            from ._runtime.program_autodiff import ProgramNativePullback

            executable = self._specialization.executable
            self._general = ProgramNativePullback(
                self._call.pullback({**self._inputs, **self._published}),
                executable.program_ad_signature,
                self._inputs,
                _program_derivative_groups(executable),
                self._context,
            )
        return self._general


def _backward_arguments(
    link: _VjpLink,
    slots: tuple[tuple[str, str], ...],
    inputs: Mapping[str, Any],
    published: Mapping[str, Any],
    cotangents: Any,
) -> tuple[tuple[Any, ...] | None, dict[str, TensorStorage]]:
    """The values of a backward call's `slots` and the gradients it writes, owner-shaped (language contract section
    8); None for the values when an input's gradient has no Storage the call can write natively."""
    # No cotangent slot is bound, so `slots` holds every one.
    by_path = _cotangents_by_path(slots, cotangents)
    state = link.device_state
    arguments: list[Any] = []
    gradients: dict[str, TensorStorage] = {}
    owners: dict[int, TensorStorage] = {}
    for role, key in slots:
        if role == "cotangent":
            value = _seeded_cotangent(link, key, published[key], by_path[key])
            if state is not None and not _ensure_device_residency(value, state):
                raise RuntimeError("a linked Module pullback argument is held by another session's device")
        else:
            owned = _owner_gradient(inputs[key], state, owners)
            if owned is None:
                return None, gradients
            gradients[key], value = owned
        arguments.append(value)
    return tuple(arguments), gradients


def _seeded_cotangent(link: _VjpLink, path: str, output: Any, supplied: Any) -> Any:
    """The cotangent of `output`; an omitted one seeds ones, one per shape, as the backward graph only reads it."""
    if supplied is not None:
        return _cotangent_storage(output, supplied)
    key = (path, tuple(output.shape), str(output.dtype))
    seed = link.seeds.get(key)
    if seed is None:
        seed = link.seeds[key] = _cotangent_storage(output, None)
    return seed


def _owner_gradient(primal: Any, state: Any, owners: dict[int, TensorStorage]) -> tuple[TensorStorage, Any] | None:
    """A differentiated input's gradient Storage, owner-shaped and shared by the views of one owner in `owners`, and
    what the backward call writes it through: the Storage, or for a view a view of it laid out as the primal. None
    for an owner that is no TensorStorage of scalars."""
    owner = primal._owner if isinstance(primal, TensorView) else primal
    if not isinstance(owner, TensorStorage) or owner.dtype.fields is not None:
        return None
    gradient = owners.get(id(owner))
    if gradient is None:
        gradient = TensorStorage._adopt(np.zeros(tuple(owner.shape), dtype=owner.dtype))
        # The backward schedule zeroes the gradient where the view lies, so only a view's has its zeros uploaded.
        if state is not None and not _adopt_device_buffer(
            gradient, state, upload=isinstance(primal, TensorView), fresh=True
        ):
            raise RuntimeError("a linked Module pullback could not allocate a device gradient")
        owners[id(owner)] = gradient
    if not isinstance(primal, TensorView):
        return gradient, gradient
    return gradient, TensorView(gradient, primal.shape, primal._strides, primal._offset, "read_write")


def _cotangents_by_path(backward_slots: tuple[tuple[str, Any], ...], cotangents: Any) -> Mapping[str, Any]:
    """A pullback's cotangents as one per output path; an omitted one is None, which seeds ones."""
    paths = tuple(key for kind, key in backward_slots if kind == "cotangent")
    if cotangents is None:
        if len(paths) != 1:
            raise ValueError("implicit Program cotangent requires exactly one output")
        return {paths[0]: None}
    if not isinstance(cotangents, Mapping):
        if len(paths) != 1:
            raise TypeError("Program pullback requires a cotangent mapping for multiple outputs")
        return {paths[0]: cotangents}
    if set(cotangents) != set(paths):
        raise ValueError("Program pullback requires exactly one cotangent per output")
    return cotangents


def _cotangent_storage(output: Any, supplied: Any) -> Any:
    if isinstance(supplied, (TensorStorage, TensorView)):
        return supplied
    array = (
        np.ones(tuple(output.shape), dtype=output.dtype)
        if supplied is None
        else np.ascontiguousarray(np.asarray(supplied, dtype=output.dtype))
    )
    return TensorStorage.from_numpy(array.reshape(tuple(output.shape)))


class BoundModuleVjp:
    """A Module VJP with some `forward()` inputs bound once (specs/language/contract.md section 8). A call passes the
    other parameters in order; its pullback writes the bound inputs' gradients into the bound gradient Storages and
    returns those.

    Once a call has linked, calls take `_BoundVjpCall`, which binds both schedules' bound slots natively, so a step's
    Python work does not grow with the number of bound inputs. Any other call runs the VJP with every input and copies
    the gradients into the bound Storages."""

    def __init__(self, expression: ModuleVjpExpression, values: Mapping[str, Any], gradients: Mapping[str, Any] | None):
        definition = type(expression.module)._module_definition
        plan = definition.fast_leaves()
        if plan is None:
            raise TypeError("step.bind requires every forward() parameter to be positional and DSL-annotated")
        names = plan[0]
        unknown = sorted(set(values) - set(names))
        if unknown:
            raise TypeError(f"step.bind names no forward() parameter {', '.join(unknown)}")
        if not values:
            raise TypeError("step.bind binds at least one forward() input")
        supplied = dict(gradients or {})
        stray = sorted(name for name in supplied if name not in values or name not in expression.wrt)
        if stray:
            raise ValueError(f"step.bind gradients name no bound input in wrt: {', '.join(stray)}")
        owners = {id(value.owner if isinstance(value, TensorView) else value) for value in values.values()}
        bound_gradients: dict[str, TensorStorage] = {}
        for name in expression.wrt:
            if name not in values:
                continue
            primal = values[name]
            if not isinstance(primal, (TensorStorage, TensorView)):
                raise TypeError(f"step.bind gradient of {name!r} needs a TensorStorage or TensorView input")
            # A view's gradient is shaped as its owner (language contract section 8).
            shaped = primal.owner if isinstance(primal, TensorView) else primal
            gradient = supplied.get(name)
            if gradient is None:
                gradient = TensorStorage.from_numpy(np.zeros(tuple(shaped.shape), dtype=shaped.dtype))
            elif not isinstance(gradient, TensorStorage):
                raise TypeError(f"step.bind gradient of {name!r} must be a TensorStorage")
            elif tuple(gradient.shape) != tuple(shaped.shape) or gradient.dtype != shaped.dtype:
                raise ValueError(
                    f"step.bind gradient of {name!r} must have the shape and element type of its input's Storage"
                )
            elif id(gradient) in owners:
                raise ValueError(f"step.bind gradient of {name!r} aliases a bound input")
            bound_gradients[name] = gradient
        self._expression = expression
        self._names = names
        self._bound = dict(values)
        self._unbound = tuple(name for name in names if name not in values)
        self.gradients: Mapping[str, TensorStorage] = MappingProxyType(bound_gradients)
        self._fast: _BoundVjpCall | None = None

    def __call__(self, *arguments: Any) -> tuple[Any, Any]:
        from ._runtime.session import _invocation_context, _session_state
        from .module import _configuration_epoch

        if len(arguments) != len(self._unbound):
            raise TypeError(f"a bound VJP takes {len(self._unbound)} forward() argument(s) positionally")
        with _invocation_context():
            fast = self._fast
            if fast is not None and fast.epoch == _configuration_epoch[0] and fast.session is _session_state():
                result = fast.call(arguments)
                if result is not None:
                    return result
            return self._call_unbound(arguments)

    def _call_unbound(self, arguments: tuple[Any, ...]) -> tuple[Any, Any]:
        passed = dict(zip(self._unbound, arguments, strict=True))
        full = tuple(self._bound[name] if name in self._bound else passed[name] for name in self._names)
        outputs, pullback = execute_module_vjp(self._expression, full, {})
        self._fast = _BoundVjpCall.create(self, full)
        return outputs, _CopyingPullback(pullback, self.gradients)


class _CopyingPullback:
    """The pullback of a bound VJP call that ran unbound; it copies the bound inputs' gradients into their Storages."""

    def __init__(self, pullback: Any, gradients: Mapping[str, TensorStorage]) -> None:
        self._pullback = pullback
        self._gradients = gradients

    def __call__(self, cotangents: Any = None) -> Mapping[str, Any]:
        computed = self._pullback(cotangents)
        for name, destination in self._gradients.items():
            destination[...] = computed[name].to_numpy()
        if len(computed) == len(self._gradients):
            return self._gradients
        return {**computed, **self._gradients}


class _BoundVjpCall:
    """The linked calls of a bound VJP: the forward and backward slots its bound inputs and gradients hold, bound once
    natively, and, per shape of the unbound tensor arguments, the allocations a call creates."""

    def __init__(
        self,
        bound: BoundModuleVjp,
        specialization: _VjpSpecialization,
        forward: Any,
        backward: Any,
        forward_slots: tuple[tuple[bool, str], ...],
        backward_slots: tuple[tuple[str, str], ...],
        epoch: int,
        session: Any,
    ) -> None:
        self.bound = bound
        self.specialization = specialization
        self.link = specialization.link
        self.forward = forward
        self.backward = backward
        # The unbound slots of each record, in order, as `_VjpLink`'s.
        self.forward_slots = forward_slots
        self.backward_slots = backward_slots
        self.epoch = epoch
        self.session = session
        self.gradients = bound.gradients
        self.names = bound._unbound
        self.bound_inputs = MappingProxyType(bound._bound)
        self.bound_controls = _owner_controls(bound._bound.values())
        self.template: tuple[Any, ...] = (None,) * len(specialization.allocations)
        self.plans: dict[tuple[Any, ...], tuple[tuple[int, _AllocationSpec, tuple[int, ...], bool], ...]] = {}

    @classmethod
    def create(cls, bound: BoundModuleVjp, full: tuple[Any, ...]) -> _BoundVjpCall | None:
        """The linked calls once `full` has linked the VJP's specialization; None while it has not or cannot."""
        from ._runtime.session import _session_state
        from .module import _configuration_epoch

        expression = bound._expression
        module = expression.module
        if isinstance(getattr(module, "checkpoint_memory_budget", None), int):
            return None
        variant = ("vjp", expression.wrt, expression.outputs, expression.planning_policy)
        key = module._fast_primal_key(full, variant)
        session = _session_state()
        specialization = None if key is None else module._program_cache.lookup(session, key)
        if not isinstance(specialization, _VjpSpecialization):
            return None
        link = specialization.link
        if link.vjp is None:
            return None
        values = bound._bound
        forward_held = {(True, name): value for name, value in values.items()}
        backward_held = {}
        for name, gradient in bound.gradients.items():
            primal = values[name]
            backward_held[("gradient", name)] = (
                TensorView(gradient, primal.shape, primal._strides, primal._offset, "read_write")
                if isinstance(primal, TensorView)
                else gradient
            )

        def bind(backward: bool, slots: tuple[Any, ...], held: Mapping[Any, Any]) -> tuple[Any, tuple[Any, ...]]:
            positions = [index for index, slot in enumerate(slots) if slot in held]
            bound_slots = link.vjp.bind(backward, positions, tuple(held[slots[index]] for index in positions))
            return bound_slots, tuple(slot for slot in slots if slot not in held)

        forward, forward_slots = bind(False, link.forward_slots, forward_held)
        backward, backward_slots = bind(True, link.backward_slots, backward_held)
        return cls(
            bound,
            specialization,
            forward,
            backward,
            forward_slots,
            backward_slots,
            _configuration_epoch[0],
            session,
        )

    def _plan(self, inputs: Mapping[str, Any]) -> tuple[tuple[int, _AllocationSpec, tuple[int, ...], bool], ...] | None:
        link = self.link
        shapes: list[tuple[int, ...] | None] = []
        plan = []
        for index, spec in enumerate(self.specialization.allocations):
            shape = spec.shape_of(inputs, shapes)
            shapes.append(shape)
            if index not in link.created:
                continue
            if shape is None:
                return None
            plan.append(
                (index, spec, shape, index not in link.zeroed and spec.initializer not in {"empty", "empty_like"})
            )
        return tuple(plan)

    def call(self, arguments: tuple[Any, ...]) -> tuple[Any, _BoundPullback] | None:
        inputs = dict(zip(self.names, arguments, strict=True))
        key = tuple(getattr(argument, "shape", None) for argument in arguments)
        plan = self.plans.get(key)
        if plan is None:
            plan = self._plan(ChainMap(inputs, self.bound_inputs))
            if plan is None:
                return None
            self.plans[key] = plan
        state = self.link.device_state
        allocations = list(self.template)
        for index, spec, shape, upload in plan:
            allocation = spec.create_shaped(shape)
            if state is not None and not _adopt_device_buffer(allocation, state, upload=upload, fresh=True):
                return None
            allocations[index] = allocation
        outputs = self.specialization.outputs.resolve(ChainMap(inputs, self.bound_inputs), tuple(allocations))
        published = flatten_program_outputs(outputs)
        passed: list[Any] = []
        for is_input, key in self.forward_slots:
            source = inputs if is_input else published
            if key not in source:
                return None
            value = source[key]
            if (
                state is not None
                and is_input
                and isinstance(value, (TensorStorage, TensorView))
                and not _ensure_device_residency(value, state)
            ):
                return None
            passed.append(value)
        arguments_tuple = tuple(passed)
        vjp = self.link.vjp
        result = vjp.call(arguments_tuple, self.forward)
        if result is None and self.refresh():
            result = vjp.call(arguments_tuple, self.forward)
        if result is None:
            return None
        return outputs, _BoundPullback(self, inputs, published, result)

    def refresh(self) -> bool:
        """Makes every bound buffer current on the device, as a call that found one stale needs; False on the CPU,
        where a stale bound buffer cannot become current here."""
        state = self.link.device_state
        if state is None:
            return False
        for value in (*self.bound_inputs.values(), *self.gradients.values()):
            if isinstance(value, (TensorStorage, TensorView)) and not _ensure_device_residency(value, state):
                return False
        return True


class _BoundPullback:
    """The pullback of a linked bound VJP call, which writes the bound inputs' gradients into their bound Storages. Its
    residuals point into the memory of the forward call's inputs, bound or not, which a later write leaves to it."""

    def __init__(self, call: _BoundVjpCall, inputs: Mapping[str, Any], published: Mapping[str, Any], result: Any):
        self._call = call
        self._inputs = inputs
        self._published = published
        self._result = result
        self._preserved = _PreservedOwners(call.bound_controls + _owner_controls(inputs.values()))

    def __call__(self, cotangents: Any = None) -> Mapping[str, Any]:
        call = self._call
        arguments, gradients = _backward_arguments(
            call.link, call.backward_slots, self._inputs, self._published, cotangents
        )
        called = arguments is not None and self._result.backward(arguments, call.backward)
        if not called and arguments is not None and call.refresh():
            called = self._result.backward(arguments, call.backward)
        if not called:
            raise ValueError("Program pullback cotangents do not match the outputs of the forward call")
        if not gradients:
            return call.gradients
        gradients.update(call.gradients)
        return gradients


def _create_linked_allocations(
    allocations: tuple[_AllocationSpec, ...],
    inputs: Mapping[str, Any],
    created: frozenset[int],
    zeroed: frozenset[int],
    device_state: Any,
) -> list[Any] | None:
    """The allocations a linked call creates, by index, with None for the schedule's transients; on a GPU each gets a
    device buffer, whose contents are not uploaded when the allocation starts empty or the schedule zeroes it. None
    when a shape is not known before the Program runs."""
    shapes: list[tuple[int, ...] | None] = []
    result: list[Any] = []
    for index, spec in enumerate(allocations):
        shape = spec.shape_of(inputs, shapes)
        shapes.append(shape)
        if index not in created:
            result.append(None)
            continue
        if shape is None and spec.initializer != "from_values":
            return None
        allocation = spec.create_shaped(shape or ())
        if device_state is not None and not _adopt_device_buffer(
            allocation,
            device_state,
            upload=index not in zeroed and spec.initializer not in {"empty", "empty_like"},
            fresh=True,
        ):
            return None
        result.append(allocation)
    return result


@dataclass(frozen=True)
class _VjpSpecialization:
    compiled: Any
    invocation: ProgramInvocation
    definition: ModuleDefinition
    allocations: tuple[_AllocationSpec, ...]
    outputs: _TreeRecipe
    link: _VjpLink = dataclasses.field(default_factory=_VjpLink)

    @property
    def template(self) -> ProgramTemplate:
        return self.compiled.template

    @property
    def executable(self) -> Any:
        return self.compiled.executable

    def invoke(
        self,
        arguments: tuple[Any, ...],
        keywords: Mapping[str, Any],
        *,
        checkpoint_memory_budget: int | None,
        checkpoint_policy: str,
    ) -> tuple[Any, Any]:
        inputs = _bind_program_inputs(self.definition, arguments, keywords)
        # A memory budget links the Program's VJP checkpointed, and its pullback reports the checkpoint plan.
        planned = checkpoint_memory_budget is not None or bool(checkpoint_policy)
        if not self.link.attempted and not planned:
            self._link()
        if self.link.vjp is not None and not planned:
            linked = self._call_linked(inputs)
            if linked is not None:
                return linked
        return self._invoke_program(inputs, checkpoint_memory_budget, checkpoint_policy)

    def invoke_fast(self, arguments: tuple[Any, ...]) -> tuple[Any, _LinkedPullback] | None:
        """The linked call for positional arguments that bind one to one to forward()'s parameters; None when the
        call takes the Program path."""
        if self.link.vjp is None:
            return None
        return self._call_linked(dict(zip(self.definition.fast_leaves()[0], arguments, strict=True)))

    def _call_linked(self, inputs: Mapping[str, Any]) -> tuple[Any, _LinkedPullback] | None:
        link = self.link
        allocations = _create_linked_allocations(self.allocations, inputs, link.created, link.zeroed, link.device_state)
        if allocations is None:
            return None
        outputs = self.outputs.resolve(inputs, tuple(allocations))
        published = flatten_program_outputs(outputs)
        state = link.device_state
        arguments = []
        for is_input, key in link.forward_slots:
            source = inputs if is_input else published
            if key not in source:
                return None
            value = source[key]
            # An input only the host holds, such as a fresh batch, is uploaded here rather than by a Program call; the
            # call's fresh outputs already have their device buffers.
            if (
                state is not None
                and is_input
                and isinstance(value, (TensorStorage, TensorView))
                and not _ensure_device_residency(value, state)
            ):
                return None
            arguments.append(value)
        call = link.vjp.call(tuple(arguments))
        if call is None:
            return None
        return outputs, _LinkedPullback(self, inputs, published, call)

    def _link(self) -> None:
        from ._runtime.session import _session_state, cpu

        link = self.link
        link.attempted = True
        compiled = self.compiled
        vjp = compiled._linked_vjp(compiled.executable)
        if vjp is None or not vjp.calls:
            return
        backward_slots = tuple((role, path) for role, path in vjp.backward_slots)
        parameters = set(self.definition.signature.parameters)
        if any("." in path or (role == "gradient" and path not in parameters) for role, path in backward_slots):
            return
        state = _session_state()
        link.forward_slots = tuple((is_input, path) for path, is_input in vjp.forward_slots)
        link.backward_slots = backward_slots
        link.created = frozenset(self.outputs.allocation_indices())
        link.zeroed = frozenset(
            index for index in link.created if self.allocations[index].initializer in {"zeros", "zeros_like"}
        )
        link.device_state = None if state.arch == cpu else state
        link.vjp = vjp

    def _invoke_program(
        self, inputs: dict[str, Any], checkpoint_memory_budget: int | None, checkpoint_policy: str
    ) -> tuple[Any, Any]:
        allocation_list: list[TensorStorage] = []
        for spec in self.allocations:
            allocation_list.append(spec.create(inputs, tuple(allocation_list)))
        allocations = tuple(allocation_list)
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
    from ._runtime.session import _session_state

    runtime_state = _session_state()
    key = None if keywords else module._fast_primal_key(arguments)
    if key is not None:
        cached = module._program_cache.lookup(runtime_state, key)
        if cached is not None:
            names = type(module)._module_definition.fast_leaves()[0]
            return cached.invoke_inputs(dict(zip(names, arguments, strict=True)))
    arguments, keywords = _resolve_module_program_controls(module, arguments, keywords)
    key = module._program_specialization_key(arguments, keywords, variant="primal")

    def compile_primal() -> _PrimalSpecialization:
        capture, outputs = _capture_module(
            module,
            _parameter_types_from_values(module, arguments, keywords),
        )
        template = ProgramTemplate.compile(capture)
        invocation = template.bind(capture, outputs)
        return _primal_specialization(module, capture, outputs, invocation)

    specialization = module._program_cache.get_or_create(runtime_state, key, compile_primal)
    return specialization.invoke(arguments, keywords)


def _compile_module_vjp_specialization(
    expression: ModuleVjpExpression,
    arguments: tuple[Any, ...],
    keywords: Mapping[str, Any],
) -> _VjpSpecialization:
    module = expression.module
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
    allocations, output_recipe = _capture_recipes(capture, outputs)
    return _VjpSpecialization(
        compiled,
        invocation,
        type(module)._module_definition,
        allocations,
        output_recipe,
    )


def execute_module_vjp(
    expression: ModuleVjpExpression,
    arguments: tuple[Any, ...],
    keywords: Mapping[str, Any],
) -> tuple[Any, Any]:
    from ._runtime.session import _session_state

    module = expression.module
    variant = ("vjp", expression.wrt, expression.outputs, expression.planning_policy)
    runtime_state = _session_state()
    budget = getattr(module, "checkpoint_memory_budget", None)
    if not isinstance(budget, int):
        budget = None
    key = None if keywords or budget is not None else module._fast_primal_key(arguments, variant)
    if key is not None:
        cached = module._program_cache.lookup(runtime_state, key)
        if cached is not None:
            linked = cached.invoke_fast(arguments)
            if linked is not None:
                return linked
    key = module._program_specialization_key(arguments, keywords, variant=variant)
    specialization = module._program_cache.get_or_create(
        runtime_state,
        key,
        lambda: _compile_module_vjp_specialization(expression, arguments, keywords),
    )
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
