from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import vernon_dsl as vd

from examples.autodiff_smoke_fluid_kernels import SmokeFluidParameters, smoke_fluid_step, smoke_fluid_step_vjp

GRID = 16
NOZZLES = 4
PRESSURE_ITERATIONS = 4
DELTA_TIME = np.float32(0.12)


@dataclass
class SmokeTimings:
    initialization_seconds: float = 0.0
    first_vjp_seconds: float = 0.0
    mpc_seconds: float = 0.0
    simulation_seconds: float = 0.0
    render_seconds: float = 0.0
    mpc_iterations: int = 0
    simulation_steps: int = 0

    @property
    def average_mpc_seconds(self) -> float:
        return self.mpc_seconds / max(self.mpc_iterations, 1)

    @property
    def average_simulation_seconds(self) -> float:
        return self.simulation_seconds / max(self.simulation_steps, 1)


class SmokeFluidSimulation:
    def __init__(self) -> None:
        self.density = vd.storage.zeros(dtype=vd.f32, shape=(GRID, GRID))
        self.velocity = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(GRID, GRID))
        self.controls = vd.storage.zeros(dtype=vd.f32, shape=(NOZZLES,))
        self.target = vd.storage.zeros(dtype=vd.f32, shape=(GRID, GRID))
        self.forced_density = vd.storage.zeros(dtype=vd.f32, shape=(GRID, GRID))
        self.forced_velocity = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(GRID, GRID))
        self.advected_velocity = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(GRID, GRID))
        self.divergence = vd.storage.zeros(dtype=vd.f32, shape=(GRID, GRID))
        self.pressure_a = vd.storage.zeros(dtype=vd.f32, shape=(GRID, GRID))
        self.pressure_b = vd.storage.zeros(dtype=vd.f32, shape=(GRID, GRID))
        self.next_density = vd.storage.zeros(dtype=vd.f32, shape=(GRID, GRID))
        self.next_velocity = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(GRID, GRID))
        self.output_loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        self.parameters = SmokeFluidParameters(
            GRID,
            GRID,
            NOZZLES,
            PRESSURE_ITERATIONS,
            DELTA_TIME,
        )

    def kernel_arguments(self) -> tuple:
        return (
            self.density,
            self.velocity,
            self.controls,
            self.target,
            self.forced_density,
            self.forced_velocity,
            self.advected_velocity,
            self.divergence,
            self.pressure_a,
            self.pressure_b,
            self.next_density,
            self.next_velocity,
            self.output_loss,
            self.parameters,
        )

    def set_objective_inputs(self, control: np.ndarray, target: np.ndarray) -> None:
        self.controls.copy_from_numpy(np.ascontiguousarray(control, dtype=np.float32))
        self.target.copy_from_numpy(np.ascontiguousarray(target, dtype=np.float32))

    def step(self, control: np.ndarray, target: np.ndarray) -> None:
        self.set_objective_inputs(control, target)
        smoke_fluid_step(*self.kernel_arguments(), grid=(1, 1, 1))
        self.density, self.next_density = self.next_density, self.density
        self.velocity, self.next_velocity = self.next_velocity, self.velocity

    def density_numpy(self) -> np.ndarray:
        return self.density.to_numpy()


def v_target(grid: int = GRID) -> np.ndarray:
    target = np.zeros((grid, grid), dtype=np.float32)
    thickness = max(grid // 18, 1)
    for y in range(grid):
        offset = min((y * (grid // 2 - thickness)) // max(grid - 1, 1), grid // 2 - thickness)
        left = offset
        right = grid - 1 - offset
        target[y, max(left - thickness + 1, 0) : min(left + thickness, grid)] = 1.0
        target[y, max(right - thickness + 1, 0) : min(right + thickness, grid)] = 1.0
    return target


def smoke_image(density: np.ndarray, target: np.ndarray, control: np.ndarray) -> np.ndarray:
    normalized = np.clip(density / np.float32(1.3), 0.0, 1.0)
    alpha = 1.0 - np.exp(-normalized * np.float32(2.4))
    blue = np.clip(normalized * 1.4, 0.0, 1.0)
    smoke = np.stack(
        (
            np.clip(alpha * (0.72 + blue * 0.28), 0.0, 1.0),
            np.clip(alpha * (0.78 + blue * 0.22), 0.0, 1.0),
            np.clip(alpha * (0.9 + blue * 0.1), 0.0, 1.0),
        ),
        axis=-1,
    )
    target_overlay = np.zeros_like(smoke)
    target_overlay[..., 1] = target * 0.18
    target_overlay[..., 2] = target * 0.08
    image = np.clip(smoke + target_overlay, 0.0, 1.0)
    nozzle_width = GRID // NOZZLES
    for nozzle, amount in enumerate(control):
        begin = nozzle * nozzle_width
        image[-2:, begin : begin + nozzle_width, 0] = np.maximum(
            image[-2:, begin : begin + nozzle_width, 0], np.float32(amount)
        )
    return (image[..., ::-1] * np.float32(255.0)).astype(np.uint8)


def optimize(
    *,
    steps: int,
    iterations: int,
    learning_rate: float,
    verbose: bool = True,
    render_callback=None,
) -> tuple[np.ndarray, np.ndarray, list[float], SmokeTimings]:
    start = time.perf_counter()
    vd.init(arch=vd.cpu)
    objective = smoke_fluid_step_vjp
    simulation = SmokeFluidSimulation()
    timings = SmokeTimings(initialization_seconds=time.perf_counter() - start)
    target = v_target()
    control = np.full((NOZZLES,), np.float32(0.25), dtype=np.float32)
    applied: list[np.ndarray] = []
    losses: list[float] = []

    for step in range(steps):
        for iteration in range(iterations):
            mpc_start = time.perf_counter()
            simulation.set_objective_inputs(control, target)
            _, pullback = objective(
                *simulation.kernel_arguments(),
                grid=(1, 1, 1),
            )
            if step == 0 and iteration == 0:
                timings.first_vjp_seconds = time.perf_counter() - mpc_start
            gradients = pullback(None)
            control_gradient = gradients["control_nozzles"].to_numpy()
            control = np.clip(
                control - np.float32(learning_rate) * control_gradient,
                0.0,
                1.0,
            ).astype(np.float32)
            timings.mpc_seconds += time.perf_counter() - mpc_start
            timings.mpc_iterations += 1

        simulation_start = time.perf_counter()
        simulation.step(control, target)
        timings.simulation_seconds += time.perf_counter() - simulation_start
        timings.simulation_steps += 1
        density = simulation.density_numpy()
        loss = float(simulation.output_loss.to_numpy()[0])
        error = float(np.linalg.norm(density - target))
        control_norm = float(np.linalg.norm(control))
        losses.append(loss)
        applied.append(control.copy())
        if render_callback is not None:
            render_start = time.perf_counter()
            render_callback(step, smoke_image(density, target, applied[-1]))
            timings.render_seconds += time.perf_counter() - render_start
        if verbose:
            print(f"step={step:03d} loss={loss:.6f} density_error={error:.6f} control_norm={control_norm:.6f}")

    return simulation.density_numpy(), np.stack(applied), losses, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="A compiler-differentiated stable-fluid smoke MPC simulation")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("smoke_mpc.png"))
    parser.add_argument("--benchmark-json", type=Path)
    arguments = parser.parse_args()
    if min(arguments.steps, arguments.iterations, arguments.fps) <= 0 or arguments.learning_rate <= 0:
        parser.error("steps, iterations, fps, and learning-rate must be positive")

    try:
        import cv2
    except ImportError as error:
        raise SystemExit(
            "Install the project examples dependency to render smoke: pip install '.[examples]'"
        ) from error

    window = "VernonDSL Smoke MPC"
    last_image: np.ndarray | None = None

    def render(_step: int, image: np.ndarray) -> None:
        nonlocal last_image
        last_image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_CUBIC)
        if not arguments.headless:
            cv2.imshow(window, last_image)
            cv2.waitKey(max(1, round(1000 / arguments.fps)))

    if not arguments.headless:
        cv2.namedWindow(window)
    try:
        density, controls, losses, timings = optimize(
            steps=arguments.steps,
            iterations=arguments.iterations,
            learning_rate=arguments.learning_rate,
            render_callback=render,
        )
    finally:
        if not arguments.headless:
            cv2.destroyAllWindows()

    if last_image is None or not cv2.imwrite(str(arguments.output), last_image):
        raise RuntimeError(f"failed to write {arguments.output}")
    np.save(arguments.output.with_name(arguments.output.stem + "_density.npy"), density)
    np.save(arguments.output.with_name(arguments.output.stem + "_controls.npy"), controls)
    benchmark = {
        **asdict(timings),
        "average_mpc_seconds": timings.average_mpc_seconds,
        "average_simulation_seconds": timings.average_simulation_seconds,
        "first_frame_seconds": timings.initialization_seconds + timings.first_vjp_seconds,
        "final_loss": losses[-1],
    }
    if arguments.benchmark_json is not None:
        arguments.benchmark_json.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n")
    print(json.dumps(benchmark, sort_keys=True))


if __name__ == "__main__":
    main()
