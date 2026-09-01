from __future__ import annotations

import inspect
import math
from typing import Any, Mapping

import numpy as np

from ..ad import ProgramExpression
from .autodiff import (
    CookedVjpPipeline,
    _compile_direct_vjp,
    _invoke_structured_pipeline,
    _retained_primal_allocation_bytes,
)
from .binding import _NativeBindingCache
from .execution_graph import (
    CompiledExecutionGraph,
    ComputeEncoder,
    ComputePass,
    ExecutionParameter,
    ExecutionResources,
    GraphResource,
    SubmissionState,
)
from .kernel import _session_state
from .tensor import TensorStorage, TensorView

_DerivativeEndpoint = GraphResource | ExecutionParameter


def _root(path: str) -> str:
    return path.split(".", 1)[0]


def _storage_owner(value: TensorStorage | TensorView) -> TensorStorage:
    owner = value.owner if isinstance(value, TensorView) else value
    if not isinstance(owner, TensorStorage):
        raise TypeError("graph VJP requires TensorStorage-backed differentiable resources")
    return owner


def _materialize_recording_value(value: Any) -> None:
    if isinstance(value, (TensorStorage, TensorView)):
        _storage_owner(value)._resident_buffer()
    elif isinstance(value, Mapping):
        for member in value.values():
            _materialize_recording_value(member)
    elif isinstance(value, (tuple, list)):
        for member in value:
            _materialize_recording_value(member)


def _zero_cotangent(primal: Any) -> Any:
    if isinstance(primal, (TensorStorage, TensorView)):
        owner = _storage_owner(primal)
        if owner._element_type is not None:
            return TensorStorage.tangent_zeros(dtype=owner._element_type, shape=owner.shape)
        return np.zeros(primal.shape, dtype=primal.dtype)
    return np.zeros_like(np.asarray(primal))


def _implicit_cotangent(primal: Any) -> Any:
    if isinstance(primal, (TensorStorage, TensorView)):
        owner = _storage_owner(primal)
        if owner._element_type is not None or int(np.prod(primal.shape, dtype=np.int64)) != 1:
            raise ValueError("implicit graph cotangent requires one scalar objective")
        return np.ones(primal.shape, dtype=primal.dtype)
    array = np.asarray(primal)
    if array.size != 1:
        raise ValueError("implicit graph cotangent requires one scalar objective")
    return np.ones_like(array)


class VjpComputePass(ComputePass):
    """A compute pass backed by one structured pipeline VJP."""

    def __init__(
        self,
        name: str,
        program: ProgramExpression | CookedVjpPipeline,
        bindings: Mapping[str, Any],
        *,
        grid: tuple[int, int, int],
    ):
        super().__init__(name)
        if not isinstance(program, (ProgramExpression, CookedVjpPipeline)):
            raise TypeError("VjpComputePass requires a structured VJP program or cooked pipeline")
        if not isinstance(bindings, Mapping):
            raise TypeError("VjpComputePass bindings must be a mapping")
        if len(grid) != 3 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid):
            raise ValueError("grid must contain three positive integers")
        self._program = program
        self._bindings = dict(bindings)
        self._grid = grid
        self._manages_borrows = True
        self._gradient_endpoints: dict[str, _DerivativeEndpoint] = {}
        self._cotangent_resources: dict[str, GraphResource] = {}
        self._binding_cache = _NativeBindingCache()

    def _paths(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if isinstance(self._program, ProgramExpression):
            return self._program.transform.wrt, self._program.transform.output_cotangents
        self._program._load()
        groups = self._program._native.derivative_groups
        gradients = tuple(str(path) for role, path, _ in groups if str(role) == "gradient")
        cotangents = tuple(str(path) for role, path, _ in groups if str(role) == "cotangent")
        return gradients, cotangents

    def _endpoint_for_path(self, path: str) -> _DerivativeEndpoint:
        name = _root(path)
        if name not in self._bindings:
            raise ValueError(f"VJP path {path!r} has no pass binding")
        value = self._bindings[name]
        if isinstance(value, GraphResource):
            assert self._graph is not None
            self._graph._validate_resource(value)
            return value
        if isinstance(value, ExecutionParameter):
            assert self._graph is not None
            self._graph._validate_parameter(value)
            return value
        assert self._graph is not None
        return self._graph.import_resource(value)

    def _program_arguments(self, resources: ExecutionResources | None) -> tuple[Any, ...]:
        assert isinstance(self._program, ProgramExpression)
        kernel = self._program.program
        if not callable(kernel) or not hasattr(kernel, "_function"):
            raise TypeError("graph VJP currently supports one compute Kernel")
        names = tuple(name for name in inspect.signature(kernel._function).parameters if name in self._bindings)
        if set(names) != set(self._bindings):
            unknown = sorted(set(self._bindings) - set(names))
            raise ValueError("VJP pass bindings do not match kernel parameters: " + ", ".join(unknown))
        return tuple(
            resources.resolve(self._bindings[name])
            if resources is not None and isinstance(self._bindings[name], (GraphResource, ExecutionParameter))
            else self._bindings[name].value
            if isinstance(self._bindings[name], GraphResource)
            else self._bindings[name]
            for name in names
        )

    def declare(self) -> None:
        gradients, cotangents = self._paths()
        self._gradient_endpoints = {path: self._endpoint_for_path(path) for path in gradients}
        cotangent_endpoints = {path: self._endpoint_for_path(path) for path in cotangents}
        if any(not isinstance(value, GraphResource) for value in cotangent_endpoints.values()):
            raise TypeError("graph VJP objectives must be graph resources")
        self._cotangent_resources = {
            path: value for path, value in cotangent_endpoints.items() if isinstance(value, GraphResource)
        }
        if isinstance(self._program, ProgramExpression):
            kernel = self._program.program
            kernel._declare_invocation(self._program_arguments(None), (), self)
            return
        self._program._load()
        state = __import__("vernon_dsl._runtime.session", fromlist=["_native"])
        access_names = {
            state._native.ACCESS_READ: self.read,
            state._native.ACCESS_WRITE: self.write,
            state._native.ACCESS_READ_WRITE: self.read_write,
        }
        for parameter in self._program._native.parameters:
            value = self._bindings.get(parameter.name)
            if isinstance(value, GraphResource):
                access_names[parameter.access](value)
            elif isinstance(value, (TensorStorage, TensorView)):
                access_names[parameter.access](value)

    def _native_declare(self) -> None:
        super()._native_declare()
        assert self._native_pass is not None

        def mapping(path: str, endpoint: _DerivativeEndpoint) -> tuple[str, int, int]:
            return (
                path,
                0 if isinstance(endpoint, GraphResource) else 1,
                endpoint.id if isinstance(endpoint, GraphResource) else endpoint._native.id,
            )

        invocation_count = 0
        tape_stride = 0
        replay_cost = 0
        resource_reload_cost = 0
        recomputation_cost = 0
        deterministic_reduction_legal = True
        required_primal_resources: list[tuple[str, int]] = []
        read_footprints: list[tuple[int, list[tuple[int, int]]]] = []
        write_footprints: list[tuple[int, list[tuple[int, int]]]] = []
        retained_primal_bytes = 0
        replay_snapshot_bytes = 0
        workgroup_invocation_count = 0
        if isinstance(self._program, ProgramExpression):
            compiled = _compile_direct_vjp(self._program, self._program_arguments(None))
            workgroup = tuple(int(value) for value in compiled.pipeline.workgroup_size)
            workgroup_invocation_count = math.prod(workgroup)
            invocation_count = math.prod(count * size for count, size in zip(self._grid, workgroup, strict=True))
            tape_stride = compiled.tape_bytes_per_invocation
            replay_cost = compiled.active_operation_count * invocation_count
            resource_reload_cost = compiled.resource_reload_cost * invocation_count
            recomputation_cost = compiled.recomputation_cost * invocation_count
            deterministic_reduction_legal = compiled.deterministic_reduction_legal

            def bound_footprints(reflected: Any, access: str) -> list[tuple[int, list[tuple[int, int]]]]:
                ranges_by_resource: dict[int, list[tuple[int, int]]] = {}
                conservative_resources: set[int] = set()
                for owner_value, whole_view, footprint_indices in reflected:
                    owner = str(owner_value)
                    endpoint = self._endpoint_for_path(owner)
                    if not isinstance(endpoint, GraphResource):
                        continue
                    binding = self._bindings.get(_root(owner))
                    bound = binding.value if isinstance(binding, GraphResource) else binding
                    if bool(whole_view) or not isinstance(bound, (TensorStorage, TensorView)):
                        conservative_resources.add(endpoint.id)
                        continue
                    indices = tuple(int(index) for index in footprint_indices)
                    if len(indices) != len(bound.shape) or any(
                        index < 0 or index >= extent for index, extent in zip(indices, bound.shape, strict=True)
                    ):
                        raise ValueError(f"TensorView {access} footprint for {owner!r} exceeds its bound shape")
                    layout = bound.layout
                    byte_offset = layout.byte_offset + sum(
                        index * stride for index, stride in zip(indices, layout.byte_strides, strict=True)
                    )
                    ranges_by_resource.setdefault(endpoint.id, []).append((byte_offset, int(bound.dtype.itemsize)))
                return [
                    (resource, ranges)
                    for resource, ranges in sorted(ranges_by_resource.items())
                    if resource not in conservative_resources
                ]

            read_footprints = bound_footprints(compiled.pipeline.read_footprints, "read")
            write_footprints = bound_footprints(compiled.pipeline.write_footprints, "write")
            if max(replay_cost, resource_reload_cost, recomputation_cost) > 2**64 - 1:
                raise OverflowError("autodiff checkpoint planning metadata exceeds uint64")
            retained_primal_bindings = {
                name: value.value if isinstance(value, GraphResource) else value
                for name, value in self._bindings.items()
            }
            required_primal_roots = {_root(path.removeprefix("primal.")) for path in compiled.required_primal_paths}
            retained_primal_bytes = _retained_primal_allocation_bytes(
                retained_primal_bindings, compiled.required_primal_paths
            )
            replay_snapshot_bytes = _retained_primal_allocation_bytes(
                retained_primal_bindings, tuple(f"primal.{name}" for name in retained_primal_bindings)
            )
            if tape_stride == 0:
                replay_snapshot_bytes = retained_primal_bytes
            retained_primal_bytes += 8 * sum(
                isinstance(self._bindings.get(root), ExecutionParameter) for root in required_primal_roots
            )
            if retained_primal_bytes > 2**64 - 1:
                raise OverflowError("autodiff retained allocation estimate exceeds uint64")
            for path in compiled.required_primal_paths:
                source_path = str(path).removeprefix("primal.")
                binding = self._bindings.get(_root(source_path))
                if not isinstance(binding, (GraphResource, TensorStorage, TensorView)):
                    continue
                endpoint = self._endpoint_for_path(source_path)
                if isinstance(endpoint, GraphResource):
                    required_primal_resources.append((str(path), endpoint.id))
        self._native_pass.set_autodiff(
            [mapping(path, endpoint) for path, endpoint in self._gradient_endpoints.items()],
            [mapping(path, endpoint) for path, endpoint in self._cotangent_resources.items()],
            required_primal_resources,
            read_footprints,
            write_footprints,
            invocation_count,
            tape_stride,
            workgroup_invocation_count,
            replay_snapshot_bytes,
            replay_cost,
            resource_reload_cost,
            recomputation_cost,
            retained_primal_bytes,
            deterministic_reduction_legal,
            isinstance(self._program, ProgramExpression),
        )

    def _prepare_vjp_recording(self, native_bindings: Any | None) -> None:
        if self._graph is None:
            raise RuntimeError("differentiable pass requires a compiled execution graph")
        state = _session_state()
        if state._architecture == state.cpu:
            return
        resources = ExecutionResources(self._graph, native_bindings)
        for value in self._bindings.values():
            resolved = resources.resolve(value) if isinstance(value, (GraphResource, ExecutionParameter)) else value
            _materialize_recording_value(resolved)

    def _native_vjp_forward(self, native_encoder: Any, native_bindings: Any | None) -> Any:
        return self._native_vjp_command(native_bindings, native_encoder=native_encoder)

    def _native_vjp_plan(self, command_plan: Any, native_bindings: Any | None) -> Any:
        return self._native_vjp_command(native_bindings, command_plan=command_plan)

    def _native_vjp_command(
        self,
        native_bindings: Any | None,
        *,
        native_encoder: Any | None = None,
        command_plan: Any | None = None,
    ) -> Any:
        if self._graph is None:
            raise RuntimeError("differentiable pass requires a compiled execution graph")
        encoder = None if native_encoder is None else ComputeEncoder(native_encoder)
        resources = ExecutionResources(self._graph, native_bindings)
        try:
            state = _session_state()
            encoded = state._architecture != state.cpu
            resolved = {
                name: resources.resolve(value) if isinstance(value, (GraphResource, ExecutionParameter)) else value
                for name, value in self._bindings.items()
            }
            if isinstance(self._program, ProgramExpression):
                compiled = _compile_direct_vjp(self._program, self._program_arguments(resources))
                _, pullback = _invoke_structured_pipeline(
                    compiled.pipeline,
                    resolved,
                    self._grid,
                    compiled.tape_bytes_per_invocation,
                    compiled.active_operation_count,
                    compiled.recomputation_cost,
                    compiled.residual_storage_kind,
                    encoder=encoder if encoded else None,
                    command_plan=command_plan if encoded else None,
                    binding_cache=self._binding_cache if encoded else None,
                )
            else:
                self._program._load()
                _, pullback = _invoke_structured_pipeline(
                    self._program._native,
                    resolved,
                    self._grid,
                    encoder=encoder if encoded else None,
                    command_plan=command_plan if encoded else None,
                    binding_cache=self._binding_cache if encoded else None,
                )
            return pullback
        finally:
            if encoder is not None:
                encoder._native = None

    def _native_graph_cotangent(self, path: str, implicit: bool, native_bindings: Any | None) -> Any:
        if path not in self._cotangent_resources or self._graph is None:
            raise RuntimeError(f"differentiable pass {self.name!r} has no cotangent path {path!r}")
        resources = ExecutionResources(self._graph, native_bindings)
        primal = resources.resolve(self._cotangent_resources[path])
        return _implicit_cotangent(primal) if implicit else _zero_cotangent(primal)

    def execute(self, encoder: ComputeEncoder, resources: ExecutionResources) -> None:
        if isinstance(self._program, ProgramExpression):
            kernel = self._program.program
            if not callable(kernel):
                raise TypeError("graph VJP currently supports one compute Kernel")
            state = _session_state()
            native_kernel: Any = kernel
            native_kernel._invoke_direct(
                self._program_arguments(resources),
                self._grid,
                encoder=None if state._architecture == state.cpu else encoder,
            )
            return
        resolved = {
            name: resources.resolve(value) if isinstance(value, (GraphResource, ExecutionParameter)) else value
            for name, value in self._bindings.items()
        }
        self._program._load()
        parameters = tuple(self._program._native.parameters)
        with self._binding_cache.invocation(self._program._native) as builder:
            for parameter in parameters:
                self._binding_cache.bind_argument(builder, self._program._native, parameter, resolved[parameter.name])
            builder.grid(*self._grid)
            builder.encode(encoder._native)


class _ForwardSubmission:
    def __init__(self, native: Any):
        self._native = native

    @property
    def state(self) -> SubmissionState:
        return SubmissionState(self._native.forward_state)

    def wait(self) -> None:
        self._native.wait_forward()


class GraphBackwardSubmission:
    def __init__(self, native: Any, owner: GraphPullback):
        self._native = native
        self._owner = owner

    @property
    def state(self) -> SubmissionState:
        return SubmissionState(self._native.state)

    @property
    def gradients(self) -> dict[str, Any]:
        return dict(self._native.gradients)

    def wait(self) -> None:
        self._native.wait()


class GraphPullback:
    def __init__(self, native: Any, owner: CompiledExecutionGraph):
        self._native = native
        self._owner = owner
        self._forward_submission = _ForwardSubmission(native)

    @property
    def forward_submission(self) -> _ForwardSubmission:
        return self._forward_submission

    @property
    def estimated_tape_bytes(self) -> int:
        return int(self._native.estimated_tape_bytes)

    @property
    def logical_residual_bytes(self) -> int:
        return int(self._native.logical_residual_bytes)

    @property
    def resident_tape_bytes(self) -> int:
        return int(self._native.resident_tape_bytes)

    @property
    def allocated_tape_bytes(self) -> int:
        return int(self._native.allocated_tape_bytes)

    @property
    def retained_allocation_bytes(self) -> int:
        return int(self._native.retained_allocation_bytes)

    @property
    def checkpoint_bytes(self) -> int:
        return int(self._native.checkpoint_bytes)

    @property
    def peak_runtime_managed_bytes(self) -> int:
        return int(self._native.peak_runtime_managed_bytes)

    @property
    def submission_count(self) -> int:
        return int(self._native.submission_count)

    @property
    def wait_count(self) -> int:
        return int(self._native.wait_count)

    @property
    def readback_count(self) -> int:
        return int(self._native.readback_count)

    @property
    def atomic_publication_count(self) -> int:
        return int(self._native.atomic_publication_count)

    @property
    def temporary_allocation_traffic_bytes(self) -> int:
        return int(self._native.temporary_allocation_traffic_bytes)

    @property
    def device_wait_nanoseconds(self) -> int:
        return int(self._native.device_wait_nanoseconds)

    @property
    def tape_context_limit_bytes(self) -> int:
        return int(self._native.tape_context_limit_bytes)

    @property
    def recomputation_factor(self) -> float:
        return float(self._native.recomputation_factor)

    @property
    def pass_telemetry(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._native.pass_telemetry)

    @property
    def reverse_python_callback_count(self) -> int:
        return int(self._native.reverse_python_callback_count)

    def submit(self, cotangent: Any = None) -> GraphBackwardSubmission:
        return GraphBackwardSubmission(self._native.submit(cotangent), self)

    def __call__(self, cotangent: Any = None) -> dict[str, Any]:
        submission = self.submit(cotangent)
        submission.wait()
        return submission.gradients


__all__ = ["GraphBackwardSubmission", "GraphPullback", "VjpComputePass"]
