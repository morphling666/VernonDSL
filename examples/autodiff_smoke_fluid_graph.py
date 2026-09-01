from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


@dataclass(frozen=True)
class SmokeFluidOutputs:
    density: vd.TensorStorage
    velocity: vd.TensorStorage
    loss: vd.TensorStorage


class PressureSolveModule(vd.Module):
    def __init__(
        self,
        *,
        width: int,
        height: int,
        iterations: int,
        grid: tuple[int, int, int],
    ):
        super().__init__()
        if iterations < 0:
            raise ValueError("pressure iterations must be non-negative")
        self.width = width
        self.height = height
        self.width_value = np.int32(width)
        self.height_value = np.int32(height)
        self.iterations = iterations
        self.grid = grid

    def forward(
        self,
        advected_velocity: vd.TensorStorage,
        scalar_template: vd.TensorStorage,
    ) -> vd.TensorStorage:
        divergence = vd.zeros_like(scalar_template)
        pressure_a = vd.zeros_like(scalar_template)
        pressure_b = vd.zeros_like(scalar_template)
        initialize_pressure(
            advected_velocity,
            divergence,
            pressure_a,
            pressure_b,
            self.width_value,
            self.height_value,
            grid=self.grid,
        )
        pressure_input = pressure_a
        pressure_output = pressure_b
        for _ in range(self.iterations):
            jacobi_pressure(
                divergence,
                pressure_input,
                pressure_output,
                self.width_value,
                self.height_value,
                grid=self.grid,
            )
            pressure_input, pressure_output = pressure_output, pressure_input
        return pressure_input


class SmokeFluidModule(vd.Module):
    """One smoke step represented by the shared Program Graph frontend."""

    def __init__(self, parameters: SmokeFluidParameters, *, planning_policy: str = "min_memory"):
        super().__init__()
        width = int(parameters.width)
        height = int(parameters.height)
        if width <= 1 or height <= 1:
            raise ValueError("smoke-fluid width and height must be greater than one")
        self.width = width
        self.height = height
        self.width_value = parameters.width
        self.height_value = parameters.height
        self.grid = ((width + 15) // 16, (height + 15) // 16, 1)
        self.planning_policy = planning_policy
        self.pressure = PressureSolveModule(
            width=width,
            height=height,
            iterations=int(parameters.pressure_iterations),
            grid=self.grid,
        )

    def forward(
        self,
        state_density: vd.TensorStorage,
        state_velocity: vd.TensorStorage,
        target_density: vd.TensorStorage,
    ) -> SmokeFluidOutputs:
        advected_velocity = vd.empty_like(state_velocity)
        advect_velocity(
            state_velocity,
            advected_velocity,
            self.width_value,
            self.height_value,
            grid=self.grid,
        )
        pressure = self.pressure(advected_velocity, state_density)
        output_velocity = vd.empty_like(state_velocity)
        project_velocity(
            advected_velocity,
            pressure,
            output_velocity,
            self.width_value,
            self.height_value,
            grid=self.grid,
        )
        output_density = vd.empty_like(state_density)
        transport_density(
            state_density,
            output_velocity,
            output_density,
            self.width_value,
            self.height_value,
            grid=self.grid,
        )
        output_loss = vd.zeros(dtype=vd.f32, shape=(1,))
        smoke_loss(
            output_density,
            target_density,
            output_loss,
            self.width_value,
            self.height_value,
            grid=(1, 1, 1),
        )
        return SmokeFluidOutputs(output_density, output_velocity, output_loss)


class SmokeFluidRolloutModule(vd.Module):
    def __init__(
        self,
        parameters: SmokeFluidParameters,
        *,
        horizon: int,
        planning_policy: str = "min_memory",
        checkpoint_memory_budget: int | None = None,
    ):
        super().__init__()
        if horizon <= 0:
            raise ValueError("smoke rollout horizon must be positive")
        self.horizon = horizon
        self.checkpoint_memory_budget = checkpoint_memory_budget
        self.step = SmokeFluidModule(parameters, planning_policy=planning_policy)

    def forward(
        self,
        initial_density: vd.TensorStorage,
        initial_velocity: vd.TensorStorage,
        target_density: vd.TensorStorage,
    ) -> vd.TensorStorage | None:
        density = initial_density
        velocity = initial_velocity
        loss: vd.TensorStorage | None = None
        for _ in range(self.horizon):
            outputs = self.step(density, velocity, target_density)
            density = outputs.density
            velocity = outputs.velocity
            loss = outputs.loss
        return loss


__all__ = [
    "PressureSolveModule",
    "SmokeFluidModule",
    "SmokeFluidOutputs",
    "SmokeFluidRolloutModule",
]
