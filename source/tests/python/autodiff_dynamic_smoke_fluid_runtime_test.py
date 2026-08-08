from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    manifest = Path(sys.argv[1])
    sys.path[:0] = [sys.argv[2], sys.argv[3], sys.argv[4]]

    import numpy as np
    import vernon_dsl as vd
    import vernon_dynamic_smoke_fluid_fixture  # noqa: F401

    from examples.autodiff_smoke_fluid_kernels import (
        SmokeFluidParameters,
        smoke_fluid_step_vjp,
    )

    vd.init(arch=vd.cpu)
    cooked = vd.load_cooked_vjp_asset(manifest)

    def arguments(size: int) -> tuple[tuple[object, ...], dict[str, object]]:
        nozzles = 2

        def scalar() -> vd.TensorStorage:
            return vd.storage.zeros(dtype=vd.f32, shape=(size, size))

        def vector() -> vd.TensorStorage:
            return vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(size, size))

        values: tuple[object, ...] = (
            scalar(),
            vector(),
            vd.storage.from_numpy(np.array([0.25, 0.5], dtype=np.float32)),
            vd.storage.from_numpy(np.ones((size, size), dtype=np.float32)),
            scalar(),
            vector(),
            vector(),
            scalar(),
            scalar(),
            scalar(),
            scalar(),
            vector(),
            vd.storage.zeros(dtype=vd.f32, shape=(1,)),
            SmokeFluidParameters(
                np.int32(size),
                np.int32(size),
                np.int32(nozzles),
                np.int32(2),
                np.float32(0.12),
            ),
        )
        names = (
            "state_density",
            "state_velocity",
            "control_nozzles",
            "objective_target_density",
            "scratch_forced_density",
            "scratch_forced_velocity",
            "scratch_advected_velocity",
            "scratch_divergence",
            "scratch_pressure_a",
            "scratch_pressure_b",
            "output_density",
            "output_velocity",
            "output_loss",
            "parameters",
        )
        return values, dict(zip(names, values, strict=True))

    for size in (4, 6):
        direct_arguments, _ = arguments(size)
        _, direct_pullback = smoke_fluid_step_vjp(*direct_arguments, grid=(1, 1, 1))
        direct_gradient = direct_pullback(None)["control_nozzles"].to_numpy()

        _, cooked_bindings = arguments(size)
        _, cooked_pullback = cooked.vjp(cooked_bindings, (1, 1, 1))
        cooked_gradient = cooked_pullback(None)["control_nozzles"].to_numpy()

        np.testing.assert_allclose(cooked_gradient, direct_gradient, rtol=1.0e-6, atol=1.0e-7)
        if not np.any(cooked_gradient):
            raise AssertionError("dynamic smoke-fluid control gradient must be nonzero")
        np.testing.assert_allclose(
            cooked_bindings["output_loss"].to_numpy(),
            direct_arguments[12].to_numpy(),
            rtol=1.0e-6,
            atol=1.0e-7,
        )


if __name__ == "__main__":
    main()
