from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    signed_manifest, scatter_manifest, gather_manifest, multi_manifest = map(Path, sys.argv[1:5])
    sys.path[:0] = [sys.argv[5], sys.argv[6], sys.argv[7]]

    import numpy as np
    import vernon_dsl as vd
    import vernon_dynamic_v2_gather_fixture  # noqa: F401
    import vernon_dynamic_v2_multi_output_fixture  # noqa: F401
    import vernon_dynamic_v2_signed_stride_fixture  # noqa: F401
    import vernon_dynamic_v2_strided_scatter_fixture  # noqa: F401
    from autodiff_dynamic_v2_parity_asset import (
        Particle,
        gather_objective,
        multi_output_objective,
        signed_stride_objective,
        strided_scatter_objective,
    )

    vd.init(arch=vd.cpu)

    def particle_storage(shape: tuple[int, ...]) -> vd.TensorStorage:
        storage = vd.storage.zeros(dtype=Particle, shape=shape)
        values = storage.to_numpy()
        values["velocity"][:] = np.array([[2.0, -3.0]] * shape[0], dtype=np.float16)
        values["mass"][:] = np.arange(4.0, 4.0 + shape[0], dtype=np.float32)
        values["tag"][:] = np.arange(7, 7 + shape[0], dtype=np.int32)
        storage.copy_from_numpy(values)
        return storage

    direct_particles = particle_storage((2,))
    direct_output_owner = vd.storage.zeros(dtype=Particle, shape=(3,))
    direct_output = direct_output_owner.view(shape=(2,), strides=(-1,), offset=2, access="write")
    _, direct_pullback = vd.ad.vjp(
        signed_stride_objective,
        wrt=("particles",),
        outputs=("output",),
    )(direct_particles, direct_output, grid=(1, 1, 1))
    signed_cotangent = vd.storage.tangent_zeros(dtype=Particle, shape=(3,))
    signed_cotangent["velocity"].copy_from_numpy(np.array([[9.0, 9.0], [1.0, 2.0], [4.0, 5.0]], dtype=np.float32))
    signed_cotangent["mass"].copy_from_numpy(np.array([9.0, 3.0, 7.0], dtype=np.float32))
    direct_signed_gradient = direct_pullback(signed_cotangent)["particles"]

    cooked_particles = particle_storage((2,))
    cooked_output_owner = vd.storage.zeros(dtype=Particle, shape=(3,))
    cooked_output = cooked_output_owner.view(shape=(2,), strides=(-1,), offset=2, access="write")
    signed_pipeline = vd.load_cooked_vjp_asset(signed_manifest)
    _, cooked_pullback = signed_pipeline.vjp(
        {"particles": cooked_particles, "output": cooked_output},
        (1, 1, 1),
    )
    cooked_signed_gradient = cooked_pullback(signed_cotangent)["particles"]
    np.testing.assert_array_equal(cooked_output_owner.to_numpy(), direct_output_owner.to_numpy())
    np.testing.assert_array_equal(
        cooked_signed_gradient["velocity"].to_numpy(),
        direct_signed_gradient["velocity"].to_numpy(),
    )
    np.testing.assert_array_equal(
        cooked_signed_gradient["mass"].to_numpy(),
        direct_signed_gradient["mass"].to_numpy(),
    )

    def run_scatter(cooked: bool) -> np.ndarray:
        owner = vd.storage.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        values = owner.view(shape=(2,), strides=(2,), offset=1, access="read")
        loss_owner = vd.storage.zeros(dtype=vd.f32, shape=(3,))
        loss = loss_owner.view(shape=(1,), strides=(1,), offset=1, access="write")
        if cooked:
            pipeline = vd.load_cooked_vjp_asset(scatter_manifest)
            _, pullback = pipeline.vjp({"values": values, "loss": loss}, (1, 1, 1))
        else:
            _, pullback = vd.ad.vjp(
                strided_scatter_objective,
                wrt=("values",),
                outputs=("loss",),
            )(values, loss, grid=(1, 1, 1))
        return pullback(None)["values"].to_numpy()

    direct_scatter = run_scatter(False)
    cooked_scatter = run_scatter(True)
    np.testing.assert_array_equal(cooked_scatter, direct_scatter)
    epsilon = 1.0e-3
    base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    def scatter_reference(values: np.ndarray) -> float:
        logical = values[[1, 3]]
        return float(logical[0] * logical[0] + logical[1])

    finite_scatter = np.zeros(4, dtype=np.float32)
    for index in (1, 3):
        positive = base.copy()
        negative = base.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        finite_scatter[index] = (scatter_reference(positive) - scatter_reference(negative)) / (2.0 * epsilon)
    np.testing.assert_allclose(cooked_scatter, finite_scatter, rtol=2.0e-3, atol=2.0e-3)

    def run_gather(cooked: bool) -> np.ndarray:
        values = vd.storage.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        if cooked:
            pipeline = vd.load_cooked_vjp_asset(gather_manifest)
            _, pullback = pipeline.vjp({"values": values, "output": output}, (1, 1, 1))
        else:
            _, pullback = vd.ad.vjp(
                gather_objective,
                wrt=("values",),
                outputs=("output",),
            )(values, output, grid=(1, 1, 1))
        return pullback(None)["values"].to_numpy()

    direct_gather = run_gather(False)
    cooked_gather = run_gather(True)
    np.testing.assert_array_equal(cooked_gather, direct_gather)
    finite_gather = np.empty(4, dtype=np.float32)
    for index in range(4):
        positive = base.copy()
        negative = base.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        finite_gather[index] = (np.dot(positive, positive) - np.dot(negative, negative)) / (2.0 * epsilon)
    np.testing.assert_allclose(cooked_gather, finite_gather, rtol=2.0e-3, atol=2.0e-3)

    def multi_cotangents() -> dict[str, vd.TensorStorage]:
        first = vd.storage.tangent_zeros(dtype=Particle, shape=(1,))
        first["velocity"].copy_from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
        first["mass"].copy_from_numpy(np.array([3.0], dtype=np.float32))
        second = vd.storage.tangent_zeros(dtype=Particle, shape=(1,))
        second["velocity"].copy_from_numpy(np.array([[4.0, 5.0]], dtype=np.float32))
        second["mass"].copy_from_numpy(np.array([7.0], dtype=np.float32))
        return {"first": first, "second": second}

    def run_multi(cooked: bool) -> vd.TensorStorage:
        particles = particle_storage((1,))
        first = vd.storage.zeros(dtype=Particle, shape=(1,))
        second = vd.storage.zeros(dtype=Particle, shape=(1,))
        if cooked:
            pipeline = vd.load_cooked_vjp_asset(multi_manifest)
            _, pullback = pipeline.vjp(
                {"particles": particles, "first": first, "second": second},
                (1, 1, 1),
            )
        else:
            _, pullback = vd.ad.vjp(
                multi_output_objective,
                wrt=("particles",),
                outputs=("first", "second"),
            )(particles, first, second, grid=(1, 1, 1))
        return pullback(multi_cotangents())["particles"]

    direct_multi = run_multi(False)
    cooked_multi = run_multi(True)
    np.testing.assert_array_equal(cooked_multi["velocity"].to_numpy(), direct_multi["velocity"].to_numpy())
    np.testing.assert_array_equal(cooked_multi["mass"].to_numpy(), direct_multi["mass"].to_numpy())
    np.testing.assert_array_equal(cooked_multi["velocity"].to_numpy(), np.array([[5.0, 7.0]], dtype=np.float32))
    np.testing.assert_array_equal(cooked_multi["mass"].to_numpy(), np.array([45.0], dtype=np.float32))


if __name__ == "__main__":
    main()
