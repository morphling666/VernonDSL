from __future__ import annotations

import importlib
import unittest
from typing import Any, cast

import numpy as np
import vernon_dsl as vd

from examples.autodiff_smoke_fluid_graph import (
    SmokeFluidModule,
    SmokeFluidRolloutModule,
)
from examples.autodiff_smoke_fluid_kernels import SmokeFluidParameters, smoke_loss
from examples.autodiff_smoke_mpc import (
    SmokeFluidSimulation,
    evaluate_initial_velocity,
    optimize_initial_velocity,
    v_target,
)


def _bilinear(field: np.ndarray, x: float, y: float, width: int, height: int) -> np.ndarray:
    sample_x = np.float32(x) % np.float32(width)
    sample_y = np.float32(y) % np.float32(height)
    x0 = int(np.floor(sample_x))
    y0 = int(np.floor(sample_y))
    tx = np.float32(sample_x - np.float32(x0))
    ty = np.float32(sample_y - np.float32(y0))
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    lower = field[y0, x0] * (np.float32(1.0) - tx) + field[y0, x1] * tx
    upper = field[y1, x0] * (np.float32(1.0) - tx) + field[y1, x1] * tx
    return lower * (np.float32(1.0) - ty) + upper * ty


def smoke_reference(
    density: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    *,
    pressure_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    height, width = density.shape
    advected_velocity = np.zeros_like(velocity)
    for y in range(height):
        for x in range(width):
            local_velocity = velocity[y, x]
            advected_velocity[y, x] = _bilinear(
                velocity,
                np.float32(x) - local_velocity[0],
                np.float32(y) - local_velocity[1],
                width,
                height,
            )

    divergence = np.zeros_like(density)
    for y in range(height):
        for x in range(width):
            divergence[y, x] = (
                advected_velocity[y, (x + 1) % width, 0]
                - advected_velocity[y, (x - 1) % width, 0]
                + advected_velocity[(y + 1) % height, x, 1]
                - advected_velocity[(y - 1) % height, x, 1]
            ) * np.float32(0.5)

    pressure_input = np.zeros_like(density)
    pressure_output = np.zeros_like(density)
    for _ in range(pressure_iterations):
        pressure_output.fill(np.float32(0.0))
        for y in range(height):
            for x in range(width):
                pressure_output[y, x] = (
                    pressure_input[y, (x - 1) % width]
                    + pressure_input[y, (x + 1) % width]
                    + pressure_input[(y - 1) % height, x]
                    + pressure_input[(y + 1) % height, x]
                    - divergence[y, x]
                ) * np.float32(0.25)
        pressure_input, pressure_output = pressure_output, pressure_input

    projected_velocity = np.zeros_like(velocity)
    for y in range(height):
        for x in range(width):
            projected_velocity[y, x, 0] = advected_velocity[y, x, 0] - (
                pressure_input[y, (x + 1) % width] - pressure_input[y, (x - 1) % width]
            ) * np.float32(0.5)
            projected_velocity[y, x, 1] = advected_velocity[y, x, 1] - (
                pressure_input[(y + 1) % height, x] - pressure_input[(y - 1) % height, x]
            ) * np.float32(0.5)

    output_density = np.zeros_like(density)
    for y in range(height):
        for x in range(width):
            local_velocity = projected_velocity[y, x]
            output_density[y, x] = _bilinear(
                density,
                np.float32(x) - local_velocity[0],
                np.float32(y) - local_velocity[1],
                width,
                height,
            )

    difference = output_density - target
    loss = np.sum(difference * difference, dtype=np.float32)
    return (
        output_density,
        projected_velocity,
        np.float32(loss / np.float32(width * height)),
    )


class SmokeFluidGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            importlib.import_module("vernon_dsl._native")
        except (ImportError, OSError) as error:
            raise unittest.SkipTest("native Vernon compiler and runtime are unavailable") from error

    @staticmethod
    def _inputs(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        generator = np.random.default_rng(7000 + size)
        density = generator.uniform(0.0, 0.7, size=(size, size)).astype(np.float32)
        velocity = generator.uniform(-0.12, 0.12, size=(size, size, 2)).astype(np.float32)
        target = generator.uniform(0.0, 1.0, size=(size, size)).astype(np.float32)
        return density, velocity, target

    def _run(self, architecture: object, size: int, pressure_iterations: int = 3):
        vd.init(arch=architecture)  # type: ignore[arg-type]
        density, velocity, target = self._inputs(size)
        velocity_storage = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(size, size))
        velocity_storage.copy_from_numpy(velocity)
        parameters = cast(Any, SmokeFluidParameters)(
            width=np.int32(size),
            height=np.int32(size),
            pressure_iterations=np.int32(pressure_iterations),
        )
        module = SmokeFluidModule(parameters)
        outputs = module(
            vd.storage.from_numpy(density),
            velocity_storage,
            vd.storage.from_numpy(target),
        )
        return (
            outputs.density.to_numpy(),
            outputs.velocity.to_numpy(),
            outputs.loss.to_numpy(),
            module,
            smoke_reference(
                density,
                velocity,
                target,
                pressure_iterations=pressure_iterations,
            ),
        )

    def test_cpu_matches_independent_reference_at_dynamic_shapes(self) -> None:
        for size in (4, 6, 16, 17):
            with self.subTest(size=size):
                density, velocity, loss, module, expected = self._run(vd.cpu, size)
                np.testing.assert_allclose(density, expected[0], rtol=2.0e-5, atol=2.0e-6)
                np.testing.assert_allclose(velocity, expected[1], rtol=2.0e-5, atol=2.0e-6)
                np.testing.assert_allclose(loss[0], expected[2], rtol=2.0e-5, atol=2.0e-6)
                self.assertEqual(module.grid, ((size + 15) // 16, (size + 15) // 16, 1))

    def test_module_forward_and_backward_use_program_operation_graph(self) -> None:
        vd.init(arch=vd.cpu)
        size = 4
        density, velocity, target = self._inputs(size)
        velocity_storage = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(size, size))
        velocity_storage.copy_from_numpy(velocity)
        parameters = cast(Any, SmokeFluidParameters)(
            width=np.int32(size),
            height=np.int32(size),
            pressure_iterations=np.int32(1),
        )
        module = SmokeFluidModule(parameters)
        outputs, pullback = vd.ad.vjp(
            module,
            wrt=("state_density", "state_velocity"),
            outputs=("density", "velocity", "loss"),
        )(
            vd.storage.from_numpy(density),
            velocity_storage,
            vd.storage.from_numpy(target),
        )
        expected = smoke_reference(density, velocity, target, pressure_iterations=1)
        np.testing.assert_allclose(outputs.density.to_numpy(), expected[0], rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(outputs.velocity.to_numpy(), expected[1], rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(outputs.loss.to_numpy()[0], expected[2], rtol=2.0e-5, atol=2.0e-6)
        gradients = pullback(
            {
                "density": np.zeros((size, size), dtype=np.float32),
                "velocity": vd.storage.tangent_zeros(
                    dtype=vd.Vector[vd.f32, 2],
                    shape=(size, size),
                ),
                "loss": np.ones((1,), dtype=np.float32),
            }
        )
        self.assertEqual(set(gradients), {"state_density", "state_velocity"})
        self.assertTrue(np.isfinite(gradients["state_density"].to_numpy()).all())
        self.assertTrue(np.isfinite(gradients["state_velocity"].to_numpy()).all())

    def test_available_gpu_backends_match_cpu(self) -> None:
        expected_density, expected_velocity, expected_loss, _, _ = self._run(vd.cpu, 17)
        for architecture in (
            vd.cuda,
            vd.vulkan,
            vd.directx,
            vd.metal,
            vd.opengl,
            vd.opengles,
        ):
            try:
                vd.init(arch=architecture)
            except RuntimeError:
                continue
            actual_density, actual_velocity, actual_loss, _, _ = self._run(architecture, 17)
            with self.subTest(backend=architecture.name):
                np.testing.assert_allclose(actual_density, expected_density, rtol=3.0e-5, atol=3.0e-6)
                np.testing.assert_allclose(actual_velocity, expected_velocity, rtol=3.0e-5, atol=3.0e-6)
                np.testing.assert_allclose(actual_loss, expected_loss, rtol=3.0e-5, atol=3.0e-6)

    def test_available_gpu_backends_validate_optimistic_static_tape_hints(self) -> None:
        size = 4
        density, velocity, target = self._inputs(size)

        def gradient(architecture: Any) -> np.ndarray:
            vd.init(arch=architecture)
            simulation = SmokeFluidSimulation(
                grid=size,
                pressure_iterations=1,
                differentiable=True,
                planning_policy="min_runtime",
            )
            simulation.set_state(density, velocity)
            pullback = simulation.step_vjp(target)
            result = pullback(
                {
                    "density": np.zeros((size, size), dtype=np.float32),
                    "velocity": vd.storage.tangent_zeros(
                        dtype=vd.Vector[vd.f32, 2],
                        shape=(size, size),
                    ),
                    "loss": np.ones((1,), dtype=np.float32),
                }
            )
            return result["state_velocity"].to_numpy()

        expected = gradient(vd.cpu)
        for architecture in (
            vd.cuda,
            vd.vulkan,
            vd.directx,
            vd.metal,
            vd.opengl,
            vd.opengles,
        ):
            try:
                vd.init(arch=architecture)
            except RuntimeError:
                continue
            actual = gradient(architecture)
            with self.subTest(backend=architecture.name):
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, rtol=3.0e-4, atol=3.0e-6)

    def test_loss_is_single_invocation_serial_kernel(self) -> None:
        self.assertEqual(cast(Any, smoke_loss).__vernon_dsl__[1]["workgroup_size"], (1, 1, 1))
        even = self._run(vd.cpu, 6, pressure_iterations=4)
        odd = self._run(vd.cpu, 6, pressure_iterations=3)
        self.assertTrue(np.isfinite(even[2]).all())
        self.assertTrue(np.isfinite(odd[2]).all())

    def test_zero_state_is_invariant(self) -> None:
        vd.init(arch=vd.cpu)
        simulation = SmokeFluidSimulation(grid=16, pressure_iterations=4)
        simulation.step(np.zeros((16, 16), dtype=np.float32))
        np.testing.assert_array_equal(simulation.density_numpy(), np.zeros((16, 16), dtype=np.float32))
        np.testing.assert_array_equal(simulation.velocity_numpy(), np.zeros((16, 16, 2), dtype=np.float32))
        np.testing.assert_array_equal(simulation.output_loss.to_numpy(), np.zeros((1,), dtype=np.float32))

    def test_single_step_velocity_gradient_matches_finite_difference(self) -> None:
        vd.init(arch=vd.cpu)
        size = 4
        pressure_iterations = 1
        density, velocity, target = self._inputs(size)
        velocity_storage = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(size, size))
        velocity_storage.copy_from_numpy(velocity)
        parameters = cast(Any, SmokeFluidParameters)(
            width=np.int32(size),
            height=np.int32(size),
            pressure_iterations=np.int32(pressure_iterations),
        )
        _, pullback = vd.ad.vjp(
            SmokeFluidModule(parameters),
            wrt=("state_velocity",),
            outputs=("density", "velocity", "loss"),
        )(
            vd.storage.from_numpy(density),
            velocity_storage,
            vd.storage.from_numpy(target),
        )

        self.assertLessEqual(pullback.logical_residual_bytes, pullback.estimated_tape_bytes)
        self.assertGreaterEqual(pullback.resident_tape_bytes, pullback.logical_residual_bytes)
        self.assertGreaterEqual(pullback.allocated_tape_bytes, pullback.resident_tape_bytes)
        self.assertGreaterEqual(pullback.recomputation_factor, 1.0)
        gradients = pullback(
            {
                "density": np.zeros((size, size), dtype=np.float32),
                "velocity": vd.storage.tangent_zeros(
                    dtype=vd.Vector[vd.f32, 2],
                    shape=(size, size),
                ),
                "loss": np.ones((1,), dtype=np.float32),
            }
        )
        analytic = gradients["state_velocity"].to_numpy()
        epsilon = np.float32(2.0e-3)
        numerical = np.zeros_like(velocity)
        for index in np.ndindex(velocity.shape):
            lower = velocity.copy()
            upper = velocity.copy()
            lower[index] -= epsilon
            upper[index] += epsilon
            lower_loss = smoke_reference(
                density,
                lower,
                target,
                pressure_iterations=pressure_iterations,
            )[2]
            upper_loss = smoke_reference(
                density,
                upper,
                target,
                pressure_iterations=pressure_iterations,
            )[2]
            numerical[index] = (upper_loss - lower_loss) / (np.float32(2.0) * epsilon)

        np.testing.assert_allclose(analytic, numerical, rtol=2.0e-2, atol=2.0e-3)

    def test_phase2_tape_telemetry_for_ci_grids(self) -> None:
        vd.init(arch=vd.cpu)
        for size in (32, 64):
            simulation = SmokeFluidSimulation(grid=size, pressure_iterations=1, differentiable=True)
            pullback = simulation.step_vjp(v_target(size))
            logical = pullback.logical_residual_bytes
            resident = pullback.resident_tape_bytes
            allocated = pullback.allocated_tape_bytes
            self.assertGreaterEqual(logical, 0)
            self.assertGreaterEqual(resident, logical)
            self.assertGreaterEqual(allocated, resident)
            if logical == 0:
                self.assertEqual(resident, 0)
                self.assertEqual(allocated, 0)
            self.assertGreater(pullback.tape_context_limit_bytes, 0)
            self.assertLessEqual(pullback.peak_runtime_managed_bytes, pullback.tape_context_limit_bytes)
            if logical:
                self.assertLessEqual(
                    resident / logical,
                    3.5,
                    "resident/logical regression tolerance is frozen at 3.5 for normal CI grids",
                )
            loss_telemetry = next(item for item in pullback.pass_telemetry if item["pass_name"].endswith(".smoke_loss"))
            self.assertEqual(loss_telemetry["control_history_kind"], "none")
            self.assertEqual(loss_telemetry["logical_residual_bytes"], 0)
            self.assertEqual(loss_telemetry["resident_tape_bytes"], 0)
            self.assertEqual(loss_telemetry["allocated_tape_bytes"], 0)
            pullback(
                {
                    "density": np.zeros((size, size), dtype=np.float32),
                    "velocity": vd.storage.tangent_zeros(
                        dtype=vd.Vector[vd.f32, 2],
                        shape=(size, size),
                    ),
                    "loss": np.ones((1,), dtype=np.float32),
                }
            )
            capture_telemetry = [
                item
                for item in pullback.pass_telemetry
                if item["residual_source_kind"].split("+", 1)[0] in {"static_capture", "dynamic_capture"}
                and item["estimated_tape_bytes"] > 0
            ]
            self.assertTrue(capture_telemetry)
            self.assertTrue(all(item["peak_temporary_tape_bytes"] > 0 for item in capture_telemetry))
            self.assertLessEqual(pullback.peak_runtime_managed_bytes, pullback.tape_context_limit_bytes)

    def test_checkpointed_initial_velocity_gradient_matches_finite_difference(
        self,
    ) -> None:
        vd.init(arch=vd.cpu)
        size = 4
        generator = np.random.default_rng(481)
        density = generator.uniform(0.0, 0.7, size=(size, size)).astype(np.float32)
        velocity = generator.uniform(-0.08, 0.08, size=(size, size, 2)).astype(np.float32)
        target = generator.uniform(0.0, 1.0, size=(size, size)).astype(np.float32)
        targets = np.zeros((2, size, size), dtype=np.float32)
        targets[-1] = target
        simulation = SmokeFluidSimulation(
            grid=size,
            pressure_iterations=1,
            differentiable=True,
        )
        _, velocity_gradient = evaluate_initial_velocity(
            simulation,
            initial_density=density,
            initial_velocity=velocity,
            target=target,
            horizon=2,
        )

        def reference_loss(initial_velocity: np.ndarray) -> np.float32:
            state_density = density
            state_velocity = initial_velocity
            loss = np.float32(0.0)
            for step in range(2):
                state_density, state_velocity, loss = smoke_reference(
                    state_density,
                    state_velocity,
                    targets[step],
                    pressure_iterations=1,
                )
            return loss

        epsilon = np.float32(2.0e-3)
        for index in ((1, 1, 0), (2, 2, 1)):
            lower = velocity.copy()
            upper = velocity.copy()
            lower[index] -= epsilon
            upper[index] += epsilon
            numerical = (reference_loss(upper) - reference_loss(lower)) / (np.float32(2.0) * epsilon)
            np.testing.assert_allclose(
                velocity_gradient[index],
                numerical,
                rtol=4.0e-2,
                atol=3.0e-3,
            )

    def test_checkpoint_plan_tracks_lowered_program_reverse_dag(self) -> None:
        vd.init(arch=vd.cpu)
        size = 4
        density, velocity, target = self._inputs(size)
        velocity_storage = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(size, size))
        velocity_storage.copy_from_numpy(velocity)
        module = SmokeFluidRolloutModule(
            cast(Any, SmokeFluidParameters)(
                width=np.int32(size),
                height=np.int32(size),
                pressure_iterations=np.int32(1),
            ),
            horizon=1,
            checkpoint_memory_budget=512_000,
        )
        _, pullback = vd.ad.vjp(module, wrt=("initial_velocity",))(
            vd.storage.from_numpy(density),
            velocity_storage,
            vd.storage.from_numpy(target),
        )
        self.assertGreaterEqual(pullback.logical_residual_bytes, 0)

    def test_dynamic_checkpoint_runtime_enforces_physical_budget(self) -> None:
        vd.init(arch=vd.cpu)
        size = 4
        density, velocity, target = self._inputs(size)
        velocity_storage = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(size, size))
        velocity_storage.copy_from_numpy(velocity)
        parameters = cast(Any, SmokeFluidParameters)(
            width=np.int32(size),
            height=np.int32(size),
            pressure_iterations=np.int32(1),
        )
        bounded = SmokeFluidRolloutModule(
            parameters,
            horizon=1,
            checkpoint_memory_budget=50_000,
        )
        _, bounded_pullback = vd.ad.vjp(bounded, wrt=("initial_velocity",))(
            vd.storage.from_numpy(density),
            velocity_storage,
            vd.storage.from_numpy(target),
        )
        self.assertEqual(bounded_pullback.logical_residual_bytes, 0)
        self.assertLessEqual(
            bounded_pullback.checkpoint_plan["peak_bytes"],
            bounded_pullback.checkpoint_plan["memory_budget"],
        )
        bounded_gradient = bounded_pullback(None)["initial_velocity"].to_numpy()
        self.assertTrue(np.isfinite(bounded_gradient).all())

        module = SmokeFluidRolloutModule(
            parameters,
            horizon=1,
            checkpoint_memory_budget=1,
        )
        with self.assertRaisesRegex(ValueError, "checkpoint|memory budget"):
            vd.ad.vjp(module, wrt=("initial_velocity",))(
                vd.storage.from_numpy(density),
                velocity_storage,
                vd.storage.from_numpy(target),
            )

    def test_initial_velocity_optimization_reduces_terminal_objective(self) -> None:
        result = optimize_initial_velocity(
            grid=8,
            horizon=4,
            iterations=2,
            pressure_iterations=1,
            verbose=False,
        )
        self.assertLess(result.objective_history[1], result.objective_history[0])
        self.assertEqual(result.initial_velocity.shape, (8, 8, 2))


if __name__ == "__main__":
    unittest.main()
