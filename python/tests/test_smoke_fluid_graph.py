from __future__ import annotations

import importlib
import unittest
from typing import Any, cast

import numpy as np
import vernon_dsl as vd

from examples.autodiff_smoke_fluid_graph import build_smoke_fluid_graph
from examples.autodiff_smoke_fluid_kernels import SmokeFluidParameters, smoke_loss


def _bilinear(field: np.ndarray, x: float, y: float, width: int, height: int) -> np.ndarray:
    sample_x = np.clip(np.float32(x), np.float32(0.0), np.float32(width - 1) - np.float32(0.001))
    sample_y = np.clip(np.float32(y), np.float32(0.0), np.float32(height - 1) - np.float32(0.001))
    x0 = int(np.floor(sample_x))
    y0 = int(np.floor(sample_y))
    tx = np.float32(sample_x - np.float32(x0))
    ty = np.float32(sample_y - np.float32(y0))
    lower = field[y0, x0] * (np.float32(1.0) - tx) + field[y0, x0 + 1] * tx
    upper = field[y0 + 1, x0] * (np.float32(1.0) - tx) + field[y0 + 1, x0 + 1] * tx
    return lower * (np.float32(1.0) - ty) + upper * ty


def smoke_reference(
    density: np.ndarray,
    velocity: np.ndarray,
    controls: np.ndarray,
    target: np.ndarray,
    *,
    pressure_iterations: int,
    delta_time: np.float32,
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    height, width = density.shape
    nozzle_count = len(controls)
    forced_density = np.zeros_like(density)
    forced_velocity = np.zeros_like(velocity)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            nozzle = int(np.float32(x) * np.float32(nozzle_count) / np.float32(width))
            center = (np.float32(nozzle) + np.float32(0.5)) * (np.float32(width) / np.float32(nozzle_count))
            horizontal = np.clip(
                np.float32(1.0) - np.abs(np.float32(x) - center) * np.float32(0.25),
                np.float32(0.0),
                np.float32(1.0),
            )
            source_height = np.float32(height) * np.float32(0.125)
            source_bottom = np.float32(height) - source_height - np.float32(2.0)
            vertical = np.clip(
                (np.float32(y) - source_bottom) / source_height,
                np.float32(0.0),
                np.float32(1.0),
            )
            source = controls[nozzle] * horizontal * vertical
            forced_density[y, x] = np.clip(
                density[y, x] * np.float32(0.995) + source * delta_time * np.float32(3.0),
                np.float32(0.0),
                np.float32(2.0),
            )
            forced_velocity[y, x, 0] = velocity[y, x, 0] + (
                np.float32(nozzle) - (np.float32(nozzle_count) - np.float32(1.0)) * np.float32(0.5)
            ) * source * delta_time * np.float32(0.08)
            forced_velocity[y, x, 1] = (
                velocity[y, x, 1] - (density[y, x] * np.float32(0.9) + source * np.float32(1.6)) * delta_time
            )

    advected_velocity = np.zeros_like(velocity)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            local_velocity = forced_velocity[y, x]
            advected_velocity[y, x] = _bilinear(
                forced_velocity,
                np.float32(x) - local_velocity[0] * delta_time,
                np.float32(y) - local_velocity[1] * delta_time,
                width,
                height,
            ) * np.float32(0.998)

    divergence = np.zeros_like(density)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            divergence[y, x] = (
                advected_velocity[y, x + 1, 0]
                - advected_velocity[y, x - 1, 0]
                + advected_velocity[y + 1, x, 1]
                - advected_velocity[y - 1, x, 1]
            ) * np.float32(0.5)

    pressure_input = np.zeros_like(density)
    pressure_output = np.zeros_like(density)
    for _ in range(pressure_iterations):
        pressure_output.fill(np.float32(0.0))
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                pressure_output[y, x] = (
                    pressure_input[y, x - 1]
                    + pressure_input[y, x + 1]
                    + pressure_input[y - 1, x]
                    + pressure_input[y + 1, x]
                    - divergence[y, x]
                ) * np.float32(0.25)
        pressure_input, pressure_output = pressure_output, pressure_input

    projected_velocity = np.zeros_like(velocity)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            projected_velocity[y, x, 0] = advected_velocity[y, x, 0] - (
                pressure_input[y, x + 1] - pressure_input[y, x - 1]
            ) * np.float32(0.5)
            projected_velocity[y, x, 1] = advected_velocity[y, x, 1] - (
                pressure_input[y + 1, x] - pressure_input[y - 1, x]
            ) * np.float32(0.5)

    output_density = np.zeros_like(density)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            local_velocity = projected_velocity[y, x]
            transported = _bilinear(
                forced_density,
                np.float32(x) - local_velocity[0] * delta_time,
                np.float32(y) - local_velocity[1] * delta_time,
                width,
                height,
            )
            laplacian = (
                forced_density[y, x - 1]
                + forced_density[y, x + 1]
                + forced_density[y - 1, x]
                + forced_density[y + 1, x]
                - forced_density[y, x] * np.float32(4.0)
            )
            output_density[y, x] = np.clip(
                (transported + laplacian * np.float32(0.0008)) * np.float32(0.996),
                np.float32(0.0),
                np.float32(2.0),
            )

    difference = output_density - target
    loss = np.sum(difference * difference, dtype=np.float32)
    loss += np.sum(controls * controls * np.float32(0.002), dtype=np.float32)
    return output_density, projected_velocity, np.float32(loss / np.float32(width * height))


class SmokeFluidGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            importlib.import_module("vernon_dsl._native")
        except (ImportError, OSError) as error:
            raise unittest.SkipTest("native Vernon compiler and runtime are unavailable") from error

    @staticmethod
    def _inputs(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        generator = np.random.default_rng(7000 + size)
        density = generator.uniform(0.0, 0.7, size=(size, size)).astype(np.float32)
        velocity = generator.uniform(-0.12, 0.12, size=(size, size, 2)).astype(np.float32)
        controls = np.array((0.2, 0.35, 0.5, 0.65), dtype=np.float32)
        target = generator.uniform(0.0, 1.0, size=(size, size)).astype(np.float32)
        return density, velocity, controls, target

    def _run(self, architecture: object, size: int, pressure_iterations: int = 3):
        vd.init(arch=architecture)  # type: ignore[arg-type]
        density, velocity, controls, target = self._inputs(size)
        velocity_storage = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(size, size))
        velocity_storage.copy_from_numpy(velocity)
        parameters = cast(Any, SmokeFluidParameters)(
            width=np.int32(size),
            height=np.int32(size),
            nozzle_count=np.int32(len(controls)),
            pressure_iterations=np.int32(pressure_iterations),
            delta_time=np.float32(0.12),
        )
        graph = build_smoke_fluid_graph(
            state_density=vd.storage.from_numpy(density),
            state_velocity=velocity_storage,
            control_nozzles=vd.storage.from_numpy(controls),
            objective_target_density=vd.storage.from_numpy(target),
            parameters=parameters,
        )
        outputs = graph.execute()
        return (
            outputs.density.to_numpy(),
            outputs.velocity.to_numpy(),
            outputs.loss.to_numpy(),
            graph,
            smoke_reference(
                density,
                velocity,
                controls,
                target,
                pressure_iterations=pressure_iterations,
                delta_time=np.float32(0.12),
            ),
        )

    def test_cpu_matches_independent_reference_at_dynamic_shapes(self) -> None:
        for size in (4, 6, 16, 17):
            with self.subTest(size=size):
                density, velocity, loss, graph, expected = self._run(vd.cpu, size)
                np.testing.assert_allclose(density, expected[0], rtol=2.0e-5, atol=2.0e-6)
                np.testing.assert_allclose(velocity, expected[1], rtol=2.0e-5, atol=2.0e-6)
                np.testing.assert_allclose(loss[0], expected[2], rtol=2.0e-5, atol=2.0e-6)
                self.assertEqual(graph.grid, ((size + 15) // 16, (size + 15) // 16, 1))
                self.assertEqual(len(graph.graph.schedule), 9)

    def test_available_gpu_backends_match_cpu(self) -> None:
        expected_density, expected_velocity, expected_loss, _, _ = self._run(vd.cpu, 17)
        for architecture in (vd.cuda, vd.vulkan, vd.directx, vd.metal, vd.opengl, vd.opengles):
            try:
                vd.init(arch=architecture)
            except RuntimeError:
                continue
            actual_density, actual_velocity, actual_loss, _, _ = self._run(architecture, 17)
            with self.subTest(backend=architecture.name):
                np.testing.assert_allclose(actual_density, expected_density, rtol=3.0e-5, atol=3.0e-6)
                np.testing.assert_allclose(actual_velocity, expected_velocity, rtol=3.0e-5, atol=3.0e-6)
                np.testing.assert_allclose(actual_loss, expected_loss, rtol=3.0e-5, atol=3.0e-6)

    def test_loss_is_single_invocation_serial_kernel(self) -> None:
        self.assertEqual(cast(Any, smoke_loss).__vernon_dsl__[1]["workgroup_size"], (1, 1, 1))
        _, _, _, graph, _ = self._run(vd.cpu, 6, pressure_iterations=4)
        self.assertIs(graph.final_pressure, graph.pressure_a)
        self.assertEqual(graph.graph.schedule[-1].name, "smoke-loss")
        _, _, _, odd_graph, _ = self._run(vd.cpu, 6, pressure_iterations=3)
        self.assertIs(odd_graph.final_pressure, odd_graph.pressure_b)


if __name__ == "__main__":
    unittest.main()
