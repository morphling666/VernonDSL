from __future__ import annotations

import unittest

import numpy as np
import vernon_dsl as vd

from examples.autodiff_smoke_fluid_kernels import smoke_fluid_step, smoke_fluid_step_vjp
from examples.autodiff_smoke_mpc import NOZZLES, SmokeFluidSimulation, optimize, v_target


class AutodiffSmokeMpcTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from vernon_dsl import _native  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("native Vernon compiler and runtime are unavailable")
        vd.init(arch=vd.cpu)

    def test_storage_objective_and_receding_horizon_execute(self) -> None:
        target = v_target()
        simulation = SmokeFluidSimulation()
        simulation.set_objective_inputs(np.zeros((NOZZLES,), dtype=np.float32), target)
        expression = smoke_fluid_step_vjp
        output, pullback = expression(*simulation.kernel_arguments(), grid=(1, 1, 1))
        self.assertIsNone(output)
        gradient = pullback(None)["control_nozzles"]
        self.assertIsInstance(gradient, vd.TensorStorage)
        self.assertEqual(gradient.shape, (NOZZLES,))

        final_density, applied, losses, timings = optimize(
            steps=3,
            iterations=3,
            learning_rate=0.08,
            verbose=False,
        )
        self.assertEqual(applied.shape, (3, NOZZLES))
        self.assertEqual(len(losses), 3)
        self.assertTrue(np.isfinite(losses).all())
        self.assertTrue(np.isfinite(final_density).all())
        self.assertEqual(timings.mpc_iterations, 9)
        self.assertEqual(timings.simulation_steps, 3)
        self.assertGreater(timings.first_vjp_seconds, 0.0)
        self.assertLess(losses[-1], losses[0])

    def test_control_gradient_matches_finite_difference_and_descends(self) -> None:
        target = v_target()
        control = np.array([0.2, 0.35, 0.5, 0.65], dtype=np.float32)
        simulation = SmokeFluidSimulation()
        simulation.set_objective_inputs(control, target)
        expression = smoke_fluid_step_vjp
        _, pullback = expression(*simulation.kernel_arguments(), grid=(1, 1, 1))
        first = pullback(None)["control_nozzles"]
        gradient = first.to_numpy()
        repeated = pullback(None)["control_nozzles"]
        self.assertIsNot(repeated, first)
        np.testing.assert_array_equal(repeated.to_numpy(), gradient)
        np.testing.assert_allclose(
            pullback(np.array([2.0], dtype=np.float32))["control_nozzles"].to_numpy(),
            gradient * np.float32(2.0),
            rtol=1.0e-6,
            atol=1.0e-7,
        )

        def loss(values: np.ndarray) -> float:
            simulation.set_objective_inputs(values, target)
            smoke_fluid_step(*simulation.kernel_arguments(), grid=(1, 1, 1))
            return float(simulation.output_loss.to_numpy()[0])

        epsilon = np.float32(2.0e-3)
        finite_difference = np.empty_like(control)
        for index in range(NOZZLES):
            lower = control.copy()
            upper = control.copy()
            lower[index] -= epsilon
            upper[index] += epsilon
            finite_difference[index] = (loss(upper) - loss(lower)) / (np.float32(2.0) * epsilon)
        np.testing.assert_allclose(gradient, finite_difference, rtol=5.0e-3, atol=3.0e-6)
        self.assertLess(loss(control - gradient), loss(control))


if __name__ == "__main__":
    unittest.main()
