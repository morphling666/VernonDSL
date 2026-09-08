from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import vernon_dsl as vd

from examples.autodiff_smoke_fluid_graph import (
    SmokeFluidModule,
    SmokeFluidRolloutModule,
)
from examples.autodiff_smoke_fluid_kernels import SmokeFluidParameters

GRID = 128
PRESSURE_ITERATIONS = 10
DEFAULT_CHECKPOINT_MEMORY_MIB = 256


@dataclass
class SmokeTimings:
    initialization_seconds: float = 0.0
    simulation_seconds: float = 0.0
    render_seconds: float = 0.0
    simulation_steps: int = 0

    @property
    def average_simulation_seconds(self) -> float:
        return self.simulation_seconds / max(self.simulation_steps, 1)


@dataclass(frozen=True)
class InitialVelocityResult:
    initial_density: np.ndarray
    initial_velocity: np.ndarray
    objective_history: tuple[float, ...]
    gradient_norm_history: tuple[float, ...]


class SmokeFluidSimulation:
    def __init__(
        self,
        *,
        grid: int = GRID,
        pressure_iterations: int = PRESSURE_ITERATIONS,
        differentiable: bool = False,
        planning_policy: str = "min_memory",
    ) -> None:
        self.grid = grid
        self.density = vd.storage.zeros(dtype=vd.f32, shape=(grid, grid))
        self.velocity = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(grid, grid))
        self.target = vd.storage.zeros(dtype=vd.f32, shape=(grid, grid))
        self.output_loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        self.differentiable = differentiable
        self.parameters = cast(Any, SmokeFluidParameters)(
            np.int32(grid),
            np.int32(grid),
            np.int32(pressure_iterations),
        )
        self._module = SmokeFluidModule(
            self.parameters,
            planning_policy=planning_policy,
        )
        self._module_vjp = (
            vd.ad.vjp(
                self._module,
                wrt=("state_density", "state_velocity"),
                outputs=("density", "velocity", "loss"),
                planning_policy=planning_policy,
            )
            if differentiable
            else None
        )
        self._density_states = (self.density,)
        self._velocity_states = (self.velocity,)

    def set_target(self, target: np.ndarray) -> None:
        self.target.copy_from_numpy(np.ascontiguousarray(target, dtype=np.float32))

    def set_state(self, density: np.ndarray, velocity: np.ndarray) -> None:
        density_value = np.ascontiguousarray(density, dtype=np.float32)
        velocity_value = np.ascontiguousarray(velocity, dtype=np.float32)
        if density_value.shape != (self.grid, self.grid) or velocity_value.shape != (self.grid, self.grid, 2):
            raise ValueError("smoke state shape does not match the simulation grid")
        for state in self._density_states:
            state.copy_from_numpy(density_value)
        for state in self._velocity_states:
            state.copy_from_numpy(velocity_value)

    def step(self, target: np.ndarray | None = None) -> None:
        if target is not None:
            self.set_target(target)
        outputs = self._module(self.density, self.velocity, self.target)
        self.density = outputs.density
        self.velocity = outputs.velocity
        self.output_loss = outputs.loss

    def step_vjp(self, target: np.ndarray | None = None) -> Any:
        if not self.differentiable:
            raise RuntimeError("smoke simulation was not created for differentiation")
        if target is not None:
            self.set_target(target)
        assert self._module_vjp is not None
        outputs, pullback = self._module_vjp(self.density, self.velocity, self.target)
        self.density = outputs.density
        self.velocity = outputs.velocity
        self.output_loss = outputs.loss
        return pullback

    def reset(self) -> None:
        zero_density = np.zeros((self.grid, self.grid), dtype=np.float32)
        zero_velocity = np.zeros((self.grid, self.grid, 2), dtype=np.float32)
        for density in self._density_states:
            density.copy_from_numpy(zero_density)
        for velocity in self._velocity_states:
            velocity.copy_from_numpy(zero_velocity)
        self.output_loss.copy_from_numpy(np.zeros((1,), dtype=np.float32))
        self.density = self._density_states[0]
        self.velocity = self._velocity_states[0]

    def restore(self, density: np.ndarray, velocity: np.ndarray) -> None:
        if density.shape != (self.grid, self.grid):
            raise ValueError(f"density checkpoint must have shape ({self.grid}, {self.grid})")
        if velocity.shape != (self.grid, self.grid, 2):
            raise ValueError(f"velocity checkpoint must have shape ({self.grid}, {self.grid}, 2)")
        self.reset()
        self._density_states[0].copy_from_numpy(np.ascontiguousarray(density, dtype=np.float32))
        self._velocity_states[0].copy_from_numpy(np.ascontiguousarray(velocity, dtype=np.float32))

    def density_numpy(self) -> np.ndarray:
        return self.density.to_numpy()

    def velocity_numpy(self) -> np.ndarray:
        return self.velocity.to_numpy()


def v_target(grid: int = GRID) -> np.ndarray:
    v_mask = np.zeros((grid, grid), dtype=np.float32)
    thickness = max(grid // 14, 1)
    for y in range(grid):
        offset = min((y * (grid // 2 - thickness)) // max(grid - 1, 1), grid // 2 - thickness)
        left = offset
        right = grid - 1 - offset
        v_mask[y, max(left - thickness + 1, 0) : min(left + thickness, grid)] = 1.0
        v_mask[y, max(right - thickness + 1, 0) : min(right + thickness, grid)] = 1.0
    coverage = float(np.mean(v_mask))
    background = np.float32((0.5 - coverage) / (1.0 - coverage))
    return np.ascontiguousarray(background + v_mask * (np.float32(1.0) - background))


def smoke_image(
    density: np.ndarray,
    *,
    output_size: int | None = None,
) -> np.ndarray:
    import cv2

    size = density.shape[1] if output_size is None else output_size
    if size <= 0:
        raise ValueError("smoke render size must be positive")

    rendered_density = cv2.resize(density, (size, size), interpolation=cv2.INTER_LINEAR)
    grayscale = (np.clip(rendered_density, 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)
    return np.repeat(grayscale[..., None], 3, axis=2)


def evaluate_initial_velocity(
    simulation: SmokeFluidSimulation,
    *,
    initial_density: np.ndarray,
    initial_velocity: np.ndarray,
    target: np.ndarray,
    horizon: int,
    checkpoint_memory_budget: int | None = None,
) -> tuple[float, np.ndarray]:
    if not simulation.differentiable:
        raise RuntimeError("initial-velocity evaluation requires a differentiable smoke simulation")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    grid_shape = (simulation.grid, simulation.grid)
    if initial_density.shape != grid_shape or target.shape != grid_shape:
        raise ValueError(f"initial_density and target must have shape {grid_shape}")
    if initial_velocity.shape != (*grid_shape, 2):
        raise ValueError(f"initial_velocity must have shape {(*grid_shape, 2)}")
    if checkpoint_memory_budget is not None and checkpoint_memory_budget <= 0:
        raise ValueError("checkpoint_memory_budget must be positive")
    density = vd.storage.from_numpy(np.ascontiguousarray(initial_density, dtype=np.float32))
    velocity = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=grid_shape)
    velocity.copy_from_numpy(np.ascontiguousarray(initial_velocity, dtype=np.float32))
    target_storage = vd.storage.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
    rollout = SmokeFluidRolloutModule(
        simulation.parameters,
        horizon=horizon,
        checkpoint_memory_budget=checkpoint_memory_budget,
    )
    loss, pullback = vd.ad.vjp(
        rollout,
        wrt=("initial_velocity",),
    )(density, velocity, target_storage)
    gradients = pullback(None)
    return float(loss.to_numpy()[0]), cast(Any, gradients["initial_velocity"]).to_numpy()


def checkerboard_smoke(grid: int) -> np.ndarray:
    if grid <= 0:
        raise ValueError("grid must be positive")
    tile = max(grid // 8, 1)
    y, x = np.mgrid[:grid, :grid]
    return np.ascontiguousarray(((y // tile + x // tile) % 2).astype(np.float32))


def optimize_initial_velocity(
    *,
    grid: int = GRID,
    horizon: int = 100,
    iterations: int = 240,
    learning_rate: float | None = None,
    pressure_iterations: int = PRESSURE_ITERATIONS,
    checkpoint_memory_budget: int | None = DEFAULT_CHECKPOINT_MEMORY_MIB * 1024 * 1024,
    initial_density: np.ndarray | None = None,
    initial_velocity: np.ndarray | None = None,
    target: np.ndarray | None = None,
    verbose: bool = True,
) -> InitialVelocityResult:
    if min(grid, horizon, iterations) <= 0:
        raise ValueError("grid, horizon, and iterations must be positive")
    if checkpoint_memory_budget is not None and checkpoint_memory_budget <= 0:
        raise ValueError("checkpoint_memory_budget must be positive")
    effective_learning_rate = 400.0 * (grid / GRID) ** 2 if learning_rate is None else learning_rate
    if not np.isfinite(effective_learning_rate) or effective_learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    vd.init(arch=vd.cpu)
    density = (
        checkerboard_smoke(grid) if initial_density is None else np.ascontiguousarray(initial_density, dtype=np.float32)
    )
    objective = v_target(grid) if target is None else np.ascontiguousarray(target, dtype=np.float32)
    if density.shape != (grid, grid) or objective.shape != (grid, grid):
        raise ValueError(f"initial_density and target must have shape ({grid}, {grid})")
    velocity = (
        np.zeros((grid, grid, 2), dtype=np.float32)
        if initial_velocity is None
        else np.ascontiguousarray(initial_velocity, dtype=np.float32)
    )
    if velocity.shape != (grid, grid, 2):
        raise ValueError(f"initial_velocity must have shape ({grid}, {grid}, 2)")
    simulation = SmokeFluidSimulation(
        grid=grid,
        pressure_iterations=pressure_iterations,
        differentiable=True,
    )
    objective_history: list[float] = []
    gradient_norm_history: list[float] = []
    best_loss = np.inf
    best_velocity = velocity.copy()
    for iteration in range(iterations):
        total_loss, velocity_gradient = evaluate_initial_velocity(
            simulation,
            initial_density=density,
            initial_velocity=velocity,
            target=objective,
            horizon=horizon,
            checkpoint_memory_budget=checkpoint_memory_budget,
        )
        gradient_norm = float(np.linalg.norm(velocity_gradient))
        objective_history.append(total_loss)
        gradient_norm_history.append(gradient_norm)
        if total_loss < best_loss:
            best_loss = total_loss
            best_velocity = velocity.copy()
        velocity -= np.float32(effective_learning_rate) * velocity_gradient
        if verbose:
            print(f"iteration={iteration:03d} objective={total_loss:.6f} gradient_norm={gradient_norm:.6f}")
    return InitialVelocityResult(
        density,
        best_velocity,
        tuple(objective_history),
        tuple(gradient_norm_history),
    )


def simulate(
    *,
    steps: int,
    architecture=vd.cpu,
    verbose: bool = True,
    render_callback=None,
    grid: int = GRID,
    pressure_iterations: int = PRESSURE_ITERATIONS,
    render_size: int | None = None,
    initial_density: np.ndarray | None = None,
    initial_velocity: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float], SmokeTimings]:
    if (initial_density is None) != (initial_velocity is None):
        raise ValueError("initial_density and initial_velocity must be provided together")
    start = time.perf_counter()
    vd.init(arch=architecture)
    simulation = SmokeFluidSimulation(
        grid=grid,
        pressure_iterations=pressure_iterations,
    )
    if initial_density is not None:
        assert initial_velocity is not None
        simulation.restore(initial_density, initial_velocity)
    timings = SmokeTimings(initialization_seconds=time.perf_counter() - start)
    target = v_target(grid)
    simulation.set_target(target)
    losses: list[float] = []

    for step in range(steps):
        simulation_start = time.perf_counter()
        simulation.step()
        timings.simulation_seconds += time.perf_counter() - simulation_start
        timings.simulation_steps += 1
        density = simulation.density_numpy()
        loss = float(simulation.output_loss.to_numpy()[0])
        losses.append(loss)
        if render_callback is not None:
            render_start = time.perf_counter()
            render_callback(
                step,
                smoke_image(
                    density,
                    output_size=render_size,
                ),
                density,
            )
            timings.render_seconds += time.perf_counter() - render_start
        if verbose:
            error = float(np.linalg.norm(density - target))
            print(f"step={step:03d} loss={loss:.6f} density_error={error:.6f}")

    return simulation.density_numpy(), losses, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a smoke velocity field to form a V-shaped density")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--grid", type=int, default=GRID)
    parser.add_argument("--pressure-iterations", type=int, default=PRESSURE_ITERATIONS)
    parser.add_argument("--optimization-iterations", type=int, default=240)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="gradient-descent rate; defaults to 400 scaled by grid area relative to 128x128",
    )
    parser.add_argument(
        "--checkpoint-memory-mib",
        type=int,
        default=DEFAULT_CHECKPOINT_MEMORY_MIB,
        help=f"hard checkpoint-memory limit (default: {DEFAULT_CHECKPOINT_MEMORY_MIB} MiB)",
    )
    parser.add_argument("--render-size", type=int, default=768)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("smoke_initial_velocity.png"))
    parser.add_argument("--animation-output", type=Path, help="write all rendered frames as a looping WebP")
    parser.add_argument("--benchmark-json", type=Path)
    arguments = parser.parse_args()
    if (
        min(
            arguments.steps,
            arguments.fps,
            arguments.grid,
            arguments.render_size,
            arguments.optimization_iterations,
        )
        <= 0
    ):
        parser.error("steps, fps, grid, render size, and optimization iterations must be positive")
    if arguments.checkpoint_memory_mib is not None and arguments.checkpoint_memory_mib <= 0:
        parser.error("checkpoint memory must be positive")
    if arguments.pressure_iterations < 0:
        parser.error("pressure iterations must be non-negative")
    if arguments.animation_output is not None and arguments.animation_output.suffix.lower() != ".webp":
        parser.error("--animation-output must use a .webp extension")

    try:
        import cv2
    except ImportError as error:
        raise SystemExit(
            "Install the project examples dependency to render smoke: pip install '.[examples]'"
        ) from error

    window = "VernonDSL Initial Velocity Optimization"
    last_image: np.ndarray | None = None
    animation_frames: list[np.ndarray] = []

    def render(_step: int, image: np.ndarray, _density: np.ndarray) -> None:
        nonlocal last_image
        last_image = image
        if arguments.animation_output is not None:
            animation_frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2BGRA))
        if not arguments.headless:
            cv2.imshow(window, last_image)
            cv2.waitKey(max(1, round(1000 / arguments.fps)))

    if not arguments.headless:
        cv2.namedWindow(window)
    velocity_result = optimize_initial_velocity(
        grid=arguments.grid,
        horizon=arguments.steps,
        iterations=arguments.optimization_iterations,
        learning_rate=arguments.learning_rate,
        pressure_iterations=arguments.pressure_iterations,
        checkpoint_memory_budget=arguments.checkpoint_memory_mib * 1024 * 1024,
        verbose=not arguments.quiet,
    )
    try:
        density, losses, timings = simulate(
            steps=arguments.steps,
            render_callback=render,
            grid=arguments.grid,
            pressure_iterations=arguments.pressure_iterations,
            render_size=arguments.render_size,
            verbose=not arguments.quiet,
            initial_density=velocity_result.initial_density,
            initial_velocity=velocity_result.initial_velocity,
        )
    finally:
        if not arguments.headless:
            cv2.destroyAllWindows()

    if last_image is None or not cv2.imwrite(str(arguments.output), last_image):
        raise RuntimeError(f"failed to write {arguments.output}")
    if arguments.animation_output is not None:
        from examples.showcase_common import write_animation

        write_animation(arguments.animation_output, animation_frames, arguments.fps)
    np.save(arguments.output.with_name(arguments.output.stem + "_density.npy"), density)
    np.save(
        arguments.output.with_name(arguments.output.stem + "_initial_density.npy"),
        velocity_result.initial_density,
    )
    np.save(
        arguments.output.with_name(arguments.output.stem + "_initial_velocity.npy"),
        velocity_result.initial_velocity,
    )
    benchmark = {
        **asdict(timings),
        "average_simulation_seconds": timings.average_simulation_seconds,
        "final_loss": losses[-1],
        "initial_optimization_objective": velocity_result.objective_history[0],
        "best_optimization_objective": min(velocity_result.objective_history),
    }
    if arguments.benchmark_json is not None:
        arguments.benchmark_json.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n")
    print(json.dumps(benchmark, sort_keys=True))


if __name__ == "__main__":
    main()
