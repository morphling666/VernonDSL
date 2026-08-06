from __future__ import annotations

import unittest

import numpy as np
import vernon_dsl as vd

from examples.autodiff_smoke_mpc import GRID, optimize, smoke_transition, v_target
from examples.autodiff_smoke_mpc_model import SmokeControlHorizon, smoke_rollout


class AutodiffSmokeMpcTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from vernon_dsl import _native  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("native Vernon compiler and runtime are unavailable")
        vd.init(arch=vd.cpu)

    def test_gradient_step_and_receding_horizon_reduce_target_error(self) -> None:
        expression = vd.ad.vjp(
            smoke_rollout,
            wrt=("controls.first", "controls.second", "controls.third"),
        )
        density = np.zeros((GRID, GRID), dtype=np.float32)
        target = v_target()
        vertical_transport, horizontal_diffusion = smoke_transition()
        controls = [np.zeros((GRID, GRID), dtype=np.float32) for _ in range(3)]
        output, pullback = expression(
            density,
            vertical_transport,
            horizontal_diffusion,
            target,
            SmokeControlHorizon(*controls),
            grid=(1, 1, 1),
        )
        self.assertEqual(
            set(output),
            {"output.loss", "output.next_density", "output.final_density"},
        )
        zero_seed = np.zeros((GRID, GRID), dtype=np.float32)
        gradients = pullback(
            {
                "output.loss": np.float32(1.0),
                "output.next_density": zero_seed,
                "output.final_density": zero_seed,
            }
        )
        updated = [
            np.clip(controls[index] - np.float32(0.08) * gradients[f"controls.{name}"], 0.0, 1.0).astype(np.float32)
            for index, name in enumerate(("first", "second", "third"))
        ]
        improved, _ = expression(
            density,
            vertical_transport,
            horizontal_diffusion,
            target,
            SmokeControlHorizon(*updated),
            grid=(1, 1, 1),
        )
        self.assertLess(float(improved["output.loss"]), float(output["output.loss"]))

        final_density, applied, _ = optimize(steps=3, iterations=3, learning_rate=0.08, verbose=False)
        self.assertEqual(applied.shape, (3, GRID, GRID))
        self.assertLess(np.linalg.norm(final_density - target), np.linalg.norm(density - target))


if __name__ == "__main__":
    unittest.main()
