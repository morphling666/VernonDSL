from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import vernon_dsl as vd

from examples.autodiff_smoke_fluid_graph import build_smoke_fluid_graph
from examples.autodiff_smoke_fluid_kernels import SmokeFluidParameters

GRID = 16
NOZZLES = 4
PRESSURE_ITERATIONS = 4
DELTA_TIME = np.float32(0.12)


@dataclass
class SmokeTimings:
    initialization_seconds: float = 0.0
    simulation_seconds: float = 0.0
    render_seconds: float = 0.0
    simulation_steps: int = 0

    @property
    def average_simulation_seconds(self) -> float:
        return self.simulation_seconds / max(self.simulation_steps, 1)


class SmokeFluidSimulation:
    def __init__(
        self,
        *,
        grid: int = GRID,
        nozzles: int = NOZZLES,
        pressure_iterations: int = PRESSURE_ITERATIONS,
    ) -> None:
        self.grid = grid
        self.nozzles = nozzles
        self.density = vd.storage.zeros(dtype=vd.f32, shape=(grid, grid))
        self.velocity = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(grid, grid))
        self.controls = vd.storage.zeros(dtype=vd.f32, shape=(nozzles,))
        self.target = vd.storage.zeros(dtype=vd.f32, shape=(grid, grid))
        self.output_loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        self.parameters = SmokeFluidParameters(
            width=np.int32(grid),
            height=np.int32(grid),
            nozzle_count=np.int32(nozzles),
            pressure_iterations=np.int32(pressure_iterations),
            delta_time=DELTA_TIME,
        )

    def set_objective_inputs(self, control: np.ndarray, target: np.ndarray) -> None:
        self.set_control(control)
        self.set_target(target)

    def set_control(self, control: np.ndarray) -> None:
        self.controls.copy_from_numpy(np.ascontiguousarray(control, dtype=np.float32))

    def set_target(self, target: np.ndarray) -> None:
        self.target.copy_from_numpy(np.ascontiguousarray(target, dtype=np.float32))

    def step(self, control: np.ndarray, target: np.ndarray | None = None) -> None:
        self.set_control(control)
        if target is not None:
            self.set_target(target)
        graph = build_smoke_fluid_graph(
            state_density=self.density,
            state_velocity=self.velocity,
            control_nozzles=self.controls,
            objective_target_density=self.target,
            parameters=self.parameters,
        )
        outputs = graph.execute()
        self.density = outputs.density
        self.velocity = outputs.velocity
        self.output_loss = outputs.loss

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
    nozzle_width = density.shape[1] // len(control)
    for nozzle, amount in enumerate(control):
        begin = nozzle * nozzle_width
        image[-2:, begin : begin + nozzle_width, 0] = np.maximum(
            image[-2:, begin : begin + nozzle_width, 0],
            np.float32(amount),
        )
    return (image[..., ::-1] * np.float32(255.0)).astype(np.uint8)


def optimize(**_arguments: object) -> None:
    raise RuntimeError("smoke-fluid optimization is deferred until ExecutionGraph VJP is implemented")


def simulate(
    *,
    steps: int,
    control: np.ndarray | None = None,
    architecture: object = vd.cpu,
    verbose: bool = True,
    render_callback=None,
) -> tuple[np.ndarray, np.ndarray, list[float], SmokeTimings]:
    start = time.perf_counter()
    vd.init(arch=architecture)
    simulation = SmokeFluidSimulation()
    timings = SmokeTimings(initialization_seconds=time.perf_counter() - start)
    target = v_target()
    simulation.set_target(target)
    applied_control = (
        np.full((NOZZLES,), np.float32(0.25), dtype=np.float32)
        if control is None
        else np.ascontiguousarray(control, dtype=np.float32)
    )
    applied: list[np.ndarray] = []
    losses: list[float] = []

    for step in range(steps):
        simulation_start = time.perf_counter()
        simulation.step(applied_control)
        timings.simulation_seconds += time.perf_counter() - simulation_start
        timings.simulation_steps += 1
        density = simulation.density_numpy()
        loss = float(simulation.output_loss.to_numpy()[0])
        losses.append(loss)
        applied.append(applied_control.copy())
        if render_callback is not None:
            render_start = time.perf_counter()
            render_callback(step, smoke_image(density, target, applied_control))
            timings.render_seconds += time.perf_counter() - render_start
        if verbose:
            error = float(np.linalg.norm(density - target))
            print(f"step={step:03d} loss={loss:.6f} density_error={error:.6f}")

    return simulation.density_numpy(), np.stack(applied), losses, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="A forward stable-fluid smoke simulation")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("smoke_forward.png"))
    parser.add_argument("--benchmark-json", type=Path)
    arguments = parser.parse_args()
    if min(arguments.steps, arguments.fps) <= 0:
        parser.error("steps and fps must be positive")

    try:
        import cv2
    except ImportError as error:
        raise SystemExit(
            "Install the project examples dependency to render smoke: pip install '.[examples]'"
        ) from error

    window = "VernonDSL Smoke Forward"
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
        density, controls, losses, timings = simulate(
            steps=arguments.steps,
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
        "average_simulation_seconds": timings.average_simulation_seconds,
        "final_loss": losses[-1],
    }
    if arguments.benchmark_json is not None:
        arguments.benchmark_json.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n")
    print(json.dumps(benchmark, sort_keys=True))


if __name__ == "__main__":
    main()
