from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import vernon_dsl as vd

from examples.autodiff_smoke_fluid_kernels import (
    SmokeFluidParameters,
    advect_velocity,
    initialize_pressure,
    jacobi_pressure,
    project_velocity,
    smoke_loss,
    transport_density,
)

_SMOKE_VJPS = {
    advect_velocity: vd.ad.vjp(
        advect_velocity,
        wrt=("state_velocity",),
        outputs=("advected_velocity",),
    ),
    initialize_pressure: vd.ad.vjp(
        initialize_pressure,
        wrt=("advected_velocity",),
        outputs=("divergence",),
    ),
    jacobi_pressure: vd.ad.vjp(
        jacobi_pressure,
        wrt=("divergence", "pressure_input"),
        outputs=("pressure_output",),
    ),
    project_velocity: vd.ad.vjp(
        project_velocity,
        wrt=("advected_velocity", "pressure"),
        outputs=("output_velocity",),
    ),
    transport_density: vd.ad.vjp(
        transport_density,
        wrt=("state_density", "projected_velocity"),
        outputs=("output_density",),
    ),
    smoke_loss: vd.ad.vjp(
        smoke_loss,
        wrt=("output_density",),
        outputs=("output_loss",),
    ),
}


class SmokeDispatch(vd.ComputePass):
    def __init__(self, name: str, invocation: vd.PipelineInvocation):
        super().__init__(name)
        self.invocation = invocation

    def declare(self) -> None:
        self.invocation.declare(self)

    def execute(self, encoder: vd.ComputeEncoder, resources: vd.ExecutionResources) -> None:
        self.invocation.encode(encoder, resources)


def _kernel_invocation(
    kernel: Any,
    *arguments: Any,
    grid: tuple[int, int, int],
) -> vd.PipelineInvocation:
    return kernel.invocation(*arguments, grid=grid)


@dataclass(frozen=True)
class SmokeFluidOutputs:
    density: vd.TensorStorage
    velocity: vd.TensorStorage
    loss: vd.TensorStorage


class SmokeFluidGraph:
    """One DiffTaichi-compatible periodic smoke-fluid forward step."""

    def __init__(
        self,
        *,
        state_density: vd.TensorStorage,
        state_velocity: vd.TensorStorage,
        objective_target_density: vd.TensorStorage,
        parameters: SmokeFluidParameters,
        output_density: vd.TensorStorage | None = None,
        output_velocity: vd.TensorStorage | None = None,
        output_loss: vd.TensorStorage | None = None,
        differentiable: bool = False,
    ):
        width = int(parameters.width)
        height = int(parameters.height)
        pressure_iterations = int(parameters.pressure_iterations)
        if width <= 1 or height <= 1:
            raise ValueError("smoke-fluid width and height must be greater than one")
        if pressure_iterations < 0:
            raise ValueError("pressure_iterations must be non-negative")

        self.parameters = parameters
        self.differentiable = differentiable
        self.state_density = state_density
        self.state_velocity = state_velocity
        self.objective_target_density = objective_target_density
        self.grid = ((width + 15) // 16, (height + 15) // 16, 1)
        self._builder = vd.ExecutionGraph()
        width_value = parameters.width
        height_value = parameters.height

        self.advected_velocity = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(height, width))
        self.divergence = vd.storage.zeros(dtype=vd.f32, shape=(height, width))
        self.pressure_a = vd.storage.zeros(dtype=vd.f32, shape=(height, width))
        self.pressure_b = vd.storage.zeros(dtype=vd.f32, shape=(height, width))
        self.output_density = (
            output_density if output_density is not None else vd.storage.zeros(dtype=vd.f32, shape=(height, width))
        )
        self.output_velocity = (
            output_velocity
            if output_velocity is not None
            else vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(height, width))
        )
        self.output_loss = output_loss if output_loss is not None else vd.storage.zeros(dtype=vd.f32, shape=(1,))

        if differentiable:
            self.state_density_resource = self._builder.differentiable_input("state_density", state_density)
            self.state_velocity_resource = self._builder.differentiable_input("state_velocity", state_velocity)
        else:
            self.state_density_resource = self._builder.import_resource(state_density)
            self.state_velocity_resource = self._builder.import_resource(state_velocity)
        self.objective_target_density_resource = self._builder.import_resource(objective_target_density)
        for value in (
            self.advected_velocity,
            self.divergence,
            self.pressure_a,
            self.pressure_b,
        ):
            self._builder.import_resource(value, exported=False)
        if differentiable:
            self.output_density_resource = self._builder.objective("density", self.output_density)
            self.output_velocity_resource = self._builder.objective("velocity", self.output_velocity)
            self.output_loss_resource = self._builder.objective("loss", self.output_loss)
        else:
            self.output_density_resource = self._builder.import_resource(self.output_density)
            self.output_velocity_resource = self._builder.import_resource(self.output_velocity)
            self.output_loss_resource = self._builder.import_resource(self.output_loss)

        previous = self._add_dispatch(
            "smoke-advect-velocity",
            advect_velocity,
            (
                self.state_velocity,
                self.advected_velocity,
                width_value,
                height_value,
            ),
        )
        previous = self._add_dispatch(
            "smoke-initialize-pressure",
            initialize_pressure,
            (
                self.advected_velocity,
                self.divergence,
                self.pressure_a,
                self.pressure_b,
                width_value,
                height_value,
            ),
            previous,
        )

        pressure_input = self.pressure_a
        pressure_output = self.pressure_b
        for iteration in range(pressure_iterations):
            previous = self._add_dispatch(
                f"smoke-jacobi-{iteration}",
                jacobi_pressure,
                (
                    self.divergence,
                    pressure_input,
                    pressure_output,
                    width_value,
                    height_value,
                ),
                previous,
            )
            pressure_input, pressure_output = pressure_output, pressure_input
        self.final_pressure = pressure_input

        previous = self._add_dispatch(
            "smoke-project-velocity",
            project_velocity,
            (
                self.advected_velocity,
                self.final_pressure,
                self.output_velocity,
                width_value,
                height_value,
            ),
            previous,
        )
        previous = self._add_dispatch(
            "smoke-transport-density",
            transport_density,
            (
                self.state_density,
                self.output_velocity,
                self.output_density,
                width_value,
                height_value,
            ),
            previous,
        )
        self._add_dispatch(
            "smoke-loss",
            smoke_loss,
            (
                self.output_density,
                self.objective_target_density,
                self.output_loss,
                width_value,
                height_value,
            ),
            previous,
            grid=(1, 1, 1),
        )
        self.graph = self._builder.compile()

    def _add_dispatch(
        self,
        name: str,
        kernel: Any,
        arguments: tuple[Any, ...],
        dependency: vd.ExecutionPass | None = None,
        *,
        grid: tuple[int, int, int] | None = None,
    ) -> vd.ExecutionPass:
        dispatch_grid = self.grid if grid is None else grid
        if self.differentiable:
            parameter_names = tuple(name for name in inspect.signature(kernel._function).parameters if name != "gid")
            execution_pass = vd.VjpComputePass(
                name,
                _SMOKE_VJPS[kernel],
                dict(zip(parameter_names, arguments, strict=True)),
                grid=dispatch_grid,
            )
        else:
            execution_pass = SmokeDispatch(
                name,
                _kernel_invocation(kernel, *arguments, grid=dispatch_grid),
            )
        if dependency is not None:
            execution_pass.depends_on(dependency)
        return self._builder.add_pass(execution_pass)

    @property
    def outputs(self) -> SmokeFluidOutputs:
        return SmokeFluidOutputs(self.output_density, self.output_velocity, self.output_loss)

    def execute(self) -> SmokeFluidOutputs:
        self.graph.submit().wait()
        return self.outputs

    def vjp(self) -> vd.GraphPullback:
        if not self.differentiable:
            raise RuntimeError("smoke graph was not built with differentiable=True")
        return self.graph.vjp()


def build_smoke_fluid_graph(
    *,
    state_density: vd.TensorStorage,
    state_velocity: vd.TensorStorage,
    objective_target_density: vd.TensorStorage,
    parameters: SmokeFluidParameters,
    output_density: vd.TensorStorage | None = None,
    output_velocity: vd.TensorStorage | None = None,
    output_loss: vd.TensorStorage | None = None,
    differentiable: bool = False,
) -> SmokeFluidGraph:
    return SmokeFluidGraph(
        state_density=state_density,
        state_velocity=state_velocity,
        objective_target_density=objective_target_density,
        parameters=parameters,
        output_density=output_density,
        output_velocity=output_velocity,
        output_loss=output_loss,
        differentiable=differentiable,
    )


def build_smoke_fluid_sequence_graph(
    *,
    initial_density: vd.TensorStorage,
    initial_velocity: vd.TensorStorage,
    target_density: vd.TensorStorage,
    parameters: SmokeFluidParameters,
    horizon: int,
    checkpoint_memory_budget: int | None = None,
) -> tuple[vd.CompiledExecutionGraph, vd.TensorStorage]:
    """Compile a temporal smoke rollout as one native differentiable graph."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    width = int(parameters.width)
    height = int(parameters.height)
    pressure_iterations = int(parameters.pressure_iterations)
    dispatch_grid = ((width + 15) // 16, (height + 15) // 16, 1)
    builder = vd.ExecutionGraph()
    builder.differentiable_input("state_density", initial_density)
    builder.differentiable_input("state_velocity", initial_velocity)
    builder.import_resource(target_density)
    state_density = initial_density
    state_velocity = initial_velocity
    previous: vd.ExecutionPass | None = None

    def add_dispatch(name: str, kernel: Any, arguments: tuple[Any, ...], *, scalar: bool = False) -> None:
        nonlocal previous
        parameter_names = tuple(name for name in inspect.signature(kernel._function).parameters if name != "gid")
        execution_pass = vd.VjpComputePass(
            name,
            _SMOKE_VJPS[kernel],
            dict(zip(parameter_names, arguments, strict=True)),
            grid=(1, 1, 1) if scalar else dispatch_grid,
        )
        if previous is not None:
            execution_pass.depends_on(previous)
        previous = builder.add_pass(execution_pass)

    final_loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
    for step in range(horizon):
        prefix = f"smoke-step-{step}"
        advected_velocity = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(height, width))
        divergence = vd.storage.zeros(dtype=vd.f32, shape=(height, width))
        pressure_a = vd.storage.zeros(dtype=vd.f32, shape=(height, width))
        pressure_b = vd.storage.zeros(dtype=vd.f32, shape=(height, width))
        output_density = vd.storage.zeros(dtype=vd.f32, shape=(height, width))
        output_velocity = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(height, width))
        for value in (advected_velocity, divergence, pressure_a, pressure_b, output_density, output_velocity):
            builder.import_resource(value, exported=False)
        add_dispatch(
            f"{prefix}-advect-velocity",
            advect_velocity,
            (state_velocity, advected_velocity, parameters.width, parameters.height),
        )
        add_dispatch(
            f"{prefix}-initialize-pressure",
            initialize_pressure,
            (
                advected_velocity,
                divergence,
                pressure_a,
                pressure_b,
                parameters.width,
                parameters.height,
            ),
        )
        pressure_input = pressure_a
        pressure_output = pressure_b
        for iteration in range(pressure_iterations):
            add_dispatch(
                f"{prefix}-jacobi-{iteration}",
                jacobi_pressure,
                (
                    divergence,
                    pressure_input,
                    pressure_output,
                    parameters.width,
                    parameters.height,
                ),
            )
            pressure_input, pressure_output = pressure_output, pressure_input
        add_dispatch(
            f"{prefix}-project-velocity",
            project_velocity,
            (
                advected_velocity,
                pressure_input,
                output_velocity,
                parameters.width,
                parameters.height,
            ),
        )
        add_dispatch(
            f"{prefix}-transport-density",
            transport_density,
            (
                state_density,
                output_velocity,
                output_density,
                parameters.width,
                parameters.height,
            ),
        )
        state_density = output_density
        state_velocity = output_velocity

    builder.objective("loss", final_loss)
    add_dispatch(
        "smoke-sequence-loss",
        smoke_loss,
        (
            state_density,
            target_density,
            final_loss,
            parameters.width,
            parameters.height,
        ),
        scalar=True,
    )
    if checkpoint_memory_budget is not None:
        builder.plan_autodiff_checkpoints(memory_budget=checkpoint_memory_budget)
    return builder.compile(), final_loss
