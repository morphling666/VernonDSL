from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    manifest = Path(sys.argv[1])
    sys.path[:0] = [sys.argv[2], sys.argv[3], sys.argv[4]]

    import numpy as np
    import vernon_dsl as vd
    import vernon_structured_storage_aggregate_autodiff_fixture  # noqa: F401
    from autodiff_structured_storage_asset import Particle, aggregate_objective

    vd.init(arch=vd.cpu)

    def primal_storage() -> vd.TensorStorage:
        particles = vd.storage.zeros(dtype=Particle, shape=(1,))
        values = particles.to_numpy()
        values["velocity"][0] = np.array([2.0, -3.0], dtype=np.float16)
        values["mass"][0] = np.float32(4.0)
        values["tag"][0] = np.int32(7)
        particles.copy_from_numpy(values)
        return particles

    cotangent = vd.storage.tangent_zeros(dtype=Particle, shape=(1,))
    cotangent["velocity"].copy_from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
    cotangent["mass"].copy_from_numpy(np.array([3.0], dtype=np.float32))

    direct_particles = primal_storage()
    direct_output = vd.storage.zeros(dtype=Particle, shape=(1,))
    _, direct_pullback = vd.ad.vjp(
        aggregate_objective,
        wrt=("particles",),
        outputs=("output",),
    )(
        direct_particles,
        direct_output,
        grid=(1, 1, 1),
    )
    direct_gradient = direct_pullback(cotangent)["particles"]

    cooked = vd.load_program(manifest)
    cooked_particles = primal_storage()
    cooked_output = vd.storage.zeros(dtype=Particle, shape=(1,))
    _, cooked_pullback = cooked.vjp(
        {"particles": cooked_particles, "output": cooked_output},
        (1, 1, 1),
    )
    cooked_gradient = cooked_pullback(cotangent)["particles"]

    np.testing.assert_array_equal(cooked_output.to_numpy(), direct_output.to_numpy())
    np.testing.assert_array_equal(
        cooked_gradient["velocity"].to_numpy(),
        direct_gradient["velocity"].to_numpy(),
    )
    np.testing.assert_array_equal(
        cooked_gradient["mass"].to_numpy(),
        direct_gradient["mass"].to_numpy(),
    )

    vd.init(arch=vd.cpu)
    reloaded_cotangent = vd.storage.tangent_zeros(dtype=Particle, shape=(1,))
    reloaded_cotangent["velocity"].copy_from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
    reloaded_cotangent["mass"].copy_from_numpy(np.array([3.0], dtype=np.float32))
    reloaded_particles = primal_storage()
    reloaded_output = vd.storage.zeros(dtype=Particle, shape=(1,))
    _, reloaded_pullback = cooked.vjp(
        {"particles": reloaded_particles, "output": reloaded_output},
        (1, 1, 1),
    )
    reloaded_gradient = reloaded_pullback(reloaded_cotangent)["particles"]
    np.testing.assert_array_equal(
        reloaded_gradient["velocity"].to_numpy(),
        np.array([[1.0, 2.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(reloaded_gradient["mass"].to_numpy(), np.array([24.0], dtype=np.float32))


if __name__ == "__main__":
    main()
