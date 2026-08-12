from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import vernon_dsl as vd

from examples.autodiff_smoke_fluid_kernels import (
    SmokeFluidParameters,
    advect_velocity,
    apply_forces,
    initialize_pressure,
    jacobi_pressure,
    project_velocity,
    smoke_loss,
    transport_density,
)


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
    """One backend-neutral, seven-phase smoke-fluid forward step."""

    def __init__(
        self,
        *,
        state_density: vd.TensorStorage,
        state_velocity: vd.TensorStorage,
        control_nozzles: vd.TensorStorage,
        objective_target_density: vd.TensorStorage,
        parameters: SmokeFluidParameters,
        output_density: vd.TensorStorage | None = None,
        output_velocity: vd.TensorStorage | None = None,
        output_loss: vd.TensorStorage | None = None,
    ):
        width = int(parameters.width)
        height = int(parameters.height)
        pressure_iterations = int(parameters.pressure_iterations)
        if width <= 1 or height <= 1:
            raise ValueError("smoke-fluid width and height must be greater than one")
        if pressure_iterations < 0:
            raise ValueError("pressure_iterations must be non-negative")

        self.parameters = parameters
        self.grid = ((width + 15) // 16, (height + 15) // 16, 1)
        self.graph = vd.ExecutionGraph()
        width_value = parameters.width
        height_value = parameters.height
        nozzle_count = parameters.nozzle_count
        delta_time = parameters.delta_time

        self.forced_density = vd.storage.zeros(dtype=vd.f32, shape=(height, width))
        self.forced_velocity = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(height, width))
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

        for value in (
            state_density,
            state_velocity,
            control_nozzles,
            objective_target_density,
        ):
            self.graph.import_resource(value)
        for value in (
            self.forced_density,
            self.forced_velocity,
            self.advected_velocity,
            self.divergence,
            self.pressure_a,
            self.pressure_b,
        ):
            self.graph.import_resource(value, exported=False)
        for value in (self.output_density, self.output_velocity, self.output_loss):
            self.graph.import_resource(value)

        previous = self._add_dispatch(
            "smoke-apply-forces",
            _kernel_invocation(
                apply_forces,
                state_density,
                state_velocity,
                control_nozzles,
                self.forced_density,
                self.forced_velocity,
                width_value,
                height_value,
                nozzle_count,
                delta_time,
                grid=self.grid,
            ),
        )
        previous = self._add_dispatch(
            "smoke-advect-velocity",
            _kernel_invocation(
                advect_velocity,
                self.forced_velocity,
                self.advected_velocity,
                width_value,
                height_value,
                delta_time,
                grid=self.grid,
            ),
            previous,
        )
        previous = self._add_dispatch(
            "smoke-initialize-pressure",
            _kernel_invocation(
                initialize_pressure,
                self.advected_velocity,
                self.divergence,
                self.pressure_a,
                self.pressure_b,
                width_value,
                height_value,
                grid=self.grid,
            ),
            previous,
        )

        pressure_input = self.pressure_a
        pressure_output = self.pressure_b
        for iteration in range(pressure_iterations):
            previous = self._add_dispatch(
                f"smoke-jacobi-{iteration}",
                _kernel_invocation(
                    jacobi_pressure,
                    self.divergence,
                    pressure_input,
                    pressure_output,
                    width_value,
                    height_value,
                    grid=self.grid,
                ),
                previous,
            )
            pressure_input, pressure_output = pressure_output, pressure_input
        self.final_pressure = pressure_input

        previous = self._add_dispatch(
            "smoke-project-velocity",
            _kernel_invocation(
                project_velocity,
                self.advected_velocity,
                self.final_pressure,
                self.output_velocity,
                width_value,
                height_value,
                grid=self.grid,
            ),
            previous,
        )
        previous = self._add_dispatch(
            "smoke-transport-density",
            _kernel_invocation(
                transport_density,
                self.forced_density,
                self.output_velocity,
                self.output_density,
                width_value,
                height_value,
                delta_time,
                grid=self.grid,
            ),
            previous,
        )
        self._add_dispatch(
            "smoke-loss",
            _kernel_invocation(
                smoke_loss,
                self.output_density,
                objective_target_density,
                control_nozzles,
                self.output_loss,
                width_value,
                height_value,
                nozzle_count,
                grid=(1, 1, 1),
            ),
            previous,
        )
        self.graph = self.graph.compile()

    def _add_dispatch(
        self,
        name: str,
        invocation: vd.PipelineInvocation,
        dependency: vd.ExecutionPass | None = None,
    ) -> vd.ExecutionPass:
        execution_pass = SmokeDispatch(name, invocation)
        if dependency is not None:
            execution_pass.depends_on(dependency)
        return self.graph.add_pass(execution_pass)

    @property
    def outputs(self) -> SmokeFluidOutputs:
        return SmokeFluidOutputs(self.output_density, self.output_velocity, self.output_loss)

    def execute(self) -> SmokeFluidOutputs:
        self.graph.submit().wait()
        return self.outputs


def build_smoke_fluid_graph(
    *,
    state_density: vd.TensorStorage,
    state_velocity: vd.TensorStorage,
    control_nozzles: vd.TensorStorage,
    objective_target_density: vd.TensorStorage,
    parameters: SmokeFluidParameters,
    output_density: vd.TensorStorage | None = None,
    output_velocity: vd.TensorStorage | None = None,
    output_loss: vd.TensorStorage | None = None,
) -> SmokeFluidGraph:
    return SmokeFluidGraph(
        state_density=state_density,
        state_velocity=state_velocity,
        control_nozzles=control_nozzles,
        objective_target_density=objective_target_density,
        parameters=parameters,
        output_density=output_density,
        output_velocity=output_velocity,
        output_loss=output_loss,
    )
