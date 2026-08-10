from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import vernon_dsl as vd

from examples.autodiff_smoke_fluid_graph import build_smoke_fluid_graph
from examples.autodiff_smoke_fluid_kernels import SmokeFluidParameters

GRID = 64
NOZZLES = 4
PRESSURE_ITERATIONS = 20
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
        self.parameters = cast(Any, SmokeFluidParameters)(
            np.int32(grid),
            np.int32(grid),
            np.int32(nozzles),
            np.int32(pressure_iterations),
            DELTA_TIME,
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


def smoke_image(
    density: np.ndarray,
    target: np.ndarray,
    control: np.ndarray,
    *,
    output_size: int | None = None,
) -> np.ndarray:
    import cv2

    size = density.shape[1] if output_size is None else output_size
    if size <= 0:
        raise ValueError("smoke render size must be positive")

    rendered_density = cv2.resize(density, (size, size), interpolation=cv2.INTER_CUBIC)
    normalized = np.clip(rendered_density / np.float32(1.45), 0.0, 1.0)
    haze = cv2.GaussianBlur(normalized, (0, 0), sigmaX=max(size / 160.0, 0.8))
    glow = cv2.GaussianBlur(normalized, (0, 0), sigmaX=max(size / 64.0, 1.5))
    opacity = 1.0 - np.exp(-(normalized * np.float32(2.4) + haze * np.float32(0.8)))

    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    x = x / np.float32(max(size - 1, 1))
    y = y / np.float32(max(size - 1, 1))
    vignette = np.clip(1.0 - 0.42 * ((x - 0.5) ** 2 + (y - 0.45) ** 2), 0.65, 1.0)
    background = np.stack(
        (
            (0.055 + 0.025 * (1.0 - y)) * vignette,
            (0.025 + 0.018 * (1.0 - y)) * vignette,
            (0.018 + 0.012 * (1.0 - y)) * vignette,
        ),
        axis=-1,
    )

    temperature = np.clip(normalized * 1.5, 0.0, 1.0)[..., None]
    cool_smoke = np.array([0.78, 0.43, 0.28], dtype=np.float32)
    dense_smoke = np.array([1.0, 0.98, 0.94], dtype=np.float32)
    smoke_color = cool_smoke + (dense_smoke - cool_smoke) * temperature
    image = background * (1.0 - opacity[..., None]) + smoke_color * opacity[..., None]
    image += glow[..., None] * np.array([0.12, 0.07, 0.035], dtype=np.float32)

    target_line = cv2.resize(target, (size, size), interpolation=cv2.INTER_CUBIC)
    target_line = np.clip(target_line, 0.0, 1.0).astype(np.float32)
    target_line = cv2.GaussianBlur(target_line, (0, 0), sigmaX=max(size / 512.0, 0.6))
    target_glow = cv2.GaussianBlur(target_line, (0, 0), sigmaX=max(size / 80.0, 1.0))
    image += target_glow[..., None] * np.array([0.08, 0.24, 0.06], dtype=np.float32)
    image += target_line[..., None] * np.array([0.14, 0.42, 0.12], dtype=np.float32)

    nozzle_spacing = size / len(control)
    nozzle_radius = max(round(nozzle_spacing * 0.06), 2)
    for nozzle, amount in enumerate(control):
        center = (round((nozzle + 0.5) * nozzle_spacing), size - nozzle_radius - 2)
        strength = float(np.clip(amount, 0.0, 1.0))
        color = (round(65 + 90 * strength), round(120 + 105 * strength), round(210 + 45 * strength))
        cv2.circle(image, center, nozzle_radius + 2, (0.035, 0.05, 0.09), -1, cv2.LINE_AA)
        cv2.circle(
            image,
            center,
            nozzle_radius,
            tuple(channel / 255.0 for channel in color),
            -1,
            cv2.LINE_AA,
        )

    return (np.clip(image, 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)


def optimize(**_arguments: object) -> None:
    raise RuntimeError("smoke-fluid optimization is deferred until ExecutionGraph VJP is implemented")


def simulate(
    *,
    steps: int,
    control: np.ndarray | None = None,
    architecture=vd.cpu,
    verbose: bool = True,
    render_callback=None,
    grid: int = GRID,
    nozzles: int = NOZZLES,
    pressure_iterations: int = PRESSURE_ITERATIONS,
    render_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float], SmokeTimings]:
    start = time.perf_counter()
    vd.init(arch=architecture)
    simulation = SmokeFluidSimulation(
        grid=grid,
        nozzles=nozzles,
        pressure_iterations=pressure_iterations,
    )
    timings = SmokeTimings(initialization_seconds=time.perf_counter() - start)
    target = v_target(grid)
    simulation.set_target(target)
    applied_control = (
        np.full((nozzles,), np.float32(0.25), dtype=np.float32)
        if control is None
        else np.ascontiguousarray(control, dtype=np.float32)
    )
    if applied_control.shape != (nozzles,):
        raise ValueError(f"control must have shape ({nozzles},)")
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
            render_callback(
                step,
                smoke_image(
                    density,
                    target,
                    applied_control,
                    output_size=render_size,
                ),
            )
            timings.render_seconds += time.perf_counter() - render_start
        if verbose:
            error = float(np.linalg.norm(density - target))
            print(f"step={step:03d} loss={loss:.6f} density_error={error:.6f}")

    return simulation.density_numpy(), np.stack(applied), losses, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="A forward stable-fluid smoke simulation")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--grid", type=int, default=GRID)
    parser.add_argument("--pressure-iterations", type=int, default=PRESSURE_ITERATIONS)
    parser.add_argument("--render-size", type=int, default=768)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("smoke_forward.png"))
    parser.add_argument("--animation-output", type=Path, help="write all rendered frames as a looping WebP")
    parser.add_argument("--benchmark-json", type=Path)
    arguments = parser.parse_args()
    if min(arguments.steps, arguments.fps, arguments.grid, arguments.render_size) <= 0:
        parser.error("steps, fps, grid, and render size must be positive")
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

    window = "VernonDSL Smoke Forward"
    last_image: np.ndarray | None = None
    animation_frames: list[np.ndarray] = []

    def render(_step: int, image: np.ndarray) -> None:
        nonlocal last_image
        last_image = image
        if arguments.animation_output is not None:
            animation_frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2BGRA))
        if not arguments.headless:
            cv2.imshow(window, last_image)
            cv2.waitKey(max(1, round(1000 / arguments.fps)))

    if not arguments.headless:
        cv2.namedWindow(window)
    try:
        density, controls, losses, timings = simulate(
            steps=arguments.steps,
            render_callback=render,
            grid=arguments.grid,
            pressure_iterations=arguments.pressure_iterations,
            render_size=arguments.render_size,
            verbose=not arguments.quiet,
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
