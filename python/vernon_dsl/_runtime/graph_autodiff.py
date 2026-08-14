from __future__ import annotations

import inspect
import math
from typing import Any, Mapping

import numpy as np

from ..ad import ProgramExpression
from .autodiff import CookedVjpPipeline, _compile_direct_vjp
from .execution_graph import (
    CompiledExecutionGraph,
    ComputeEncoder,
    ComputePass,
    ExecutionParameter,
    ExecutionResources,
    GraphResource,
    SubmissionState,
)
from .resources import TensorStorage, TensorView

_DerivativeEndpoint = GraphResource | ExecutionParameter


def _root(path: str) -> str:
    return path.split(".", 1)[0]


def _storage_owner(value: TensorStorage | TensorView) -> TensorStorage:
    owner = value.owner if isinstance(value, TensorView) else value
    if not isinstance(owner, TensorStorage):
        raise TypeError("graph VJP requires TensorStorage-backed differentiable resources")
    return owner


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
        if isinstance(self._program, ProgramExpression):
            compiled = _compile_direct_vjp(self._program, self._program_arguments(None))
            workgroup = tuple(int(value) for value in compiled.pipeline.workgroup_size)
            invocation_count = math.prod(count * size for count, size in zip(self._grid, workgroup, strict=True))
            tape_stride = compiled.tape_bytes_per_invocation
            replay_cost = compiled.active_operation_count * invocation_count
            if replay_cost > 2**64 - 1:
                raise OverflowError("autodiff checkpoint planning metadata exceeds uint64")
        self._native_pass.set_autodiff(
            [mapping(path, endpoint) for path, endpoint in self._gradient_endpoints.items()],
            [mapping(path, endpoint) for path, endpoint in self._cotangent_resources.items()],
            invocation_count,
            tape_stride,
            replay_cost,
            isinstance(self._program, ProgramExpression),
        )

    def _native_vjp_forward(self, native_bindings: Any | None) -> Any:
        if self._graph is None:
            raise RuntimeError("differentiable pass requires a compiled execution graph")
        resources = ExecutionResources(self._graph, native_bindings)
        if isinstance(self._program, ProgramExpression):
            _, pullback = self._program(*self._program_arguments(resources), grid=self._grid)
        else:
            resolved = {
                name: resources.resolve(value) if isinstance(value, (GraphResource, ExecutionParameter)) else value
                for name, value in self._bindings.items()
            }
            _, pullback = self._program.vjp(resolved, self._grid)
        return pullback

    def _native_graph_cotangent(self, path: str, implicit: bool, native_bindings: Any | None) -> Any:
        if path not in self._cotangent_resources or self._graph is None:
            raise RuntimeError(f"differentiable pass {self.name!r} has no cotangent path {path!r}")
        resources = ExecutionResources(self._graph, native_bindings)
        primal = resources.resolve(self._cotangent_resources[path])
        return _implicit_cotangent(primal) if implicit else _zero_cotangent(primal)

    def execute(self, encoder: ComputeEncoder, resources: ExecutionResources) -> None:
        del encoder
        if isinstance(self._program, ProgramExpression):
            kernel = self._program.program
            if not callable(kernel):
                raise TypeError("graph VJP currently supports one compute Kernel")
            kernel(*self._program_arguments(resources), grid=self._grid)
            return
        resolved = {
            name: resources.resolve(value) if isinstance(value, (GraphResource, ExecutionParameter)) else value
            for name, value in self._bindings.items()
        }
        self._program.primal(resolved, self._grid)


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
    def checkpoint_bytes(self) -> int:
        return int(self._native.checkpoint_bytes)

    @property
    def peak_runtime_managed_bytes(self) -> int:
        return int(self._native.peak_runtime_managed_bytes)

    @property
    def tape_context_limit_bytes(self) -> int:
        return int(self._native.tape_context_limit_bytes)

    @property
    def recomputation_factor(self) -> float:
        return float(self._native.recomputation_factor)

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
