from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    (
        signed_manifest,
        field_manifest,
        scatter_manifest,
        gather_manifest,
        multi_manifest,
        partial_manifest,
        mixed_manifest,
    ) = map(Path, sys.argv[1:8])
    sys.path[:0] = [sys.argv[8], sys.argv[9], sys.argv[10]]

    import numpy as np
    import vernon_dsl as vd
    import vernon_dynamic_v2_field_path_fixture  # noqa: F401
    import vernon_dynamic_v2_gather_fixture  # noqa: F401
    import vernon_dynamic_v2_mixed_alias_fixture  # noqa: F401
    import vernon_dynamic_v2_multi_output_fixture  # noqa: F401
    import vernon_dynamic_v2_partially_dynamic_fixture  # noqa: F401
    import vernon_dynamic_v2_signed_stride_fixture  # noqa: F401
    import vernon_dynamic_v2_strided_scatter_fixture  # noqa: F401
    from autodiff_dynamic_v2_parity_asset import (
        Particle,
        field_path_objective,
        gather_objective,
        mixed_alias_objective,
        multi_output_objective,
        partially_dynamic_objective,
        signed_stride_objective,
        strided_scatter_objective,
    )

    vd.init(arch=vd.cpu)

    def direct_pullback(function, wrt, outputs, *arguments, grid=(1, 1, 1)):
        return vd.ad.vjp(function, wrt=wrt, outputs=outputs)(*arguments, grid=grid)[1]

    def cooked_pullback(pipeline, bindings, grid=(1, 1, 1)):
        return pipeline.vjp(bindings, grid)[1]

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
    direct_signed_pullback = direct_pullback(
        signed_stride_objective,
        ("particles",),
        ("output",),
        direct_particles,
        direct_output,
    )
    signed_cotangent = vd.storage.tangent_zeros(dtype=Particle, shape=(3,))
    signed_cotangent["velocity"].copy_from_numpy(np.array([[9.0, 9.0], [1.0, 2.0], [4.0, 5.0]], dtype=np.float32))
    signed_cotangent["mass"].copy_from_numpy(np.array([9.0, 3.0, 7.0], dtype=np.float32))
    direct_signed_gradient = direct_signed_pullback(signed_cotangent)["particles"]

    cooked_particles = particle_storage((2,))
    cooked_output_owner = vd.storage.zeros(dtype=Particle, shape=(3,))
    cooked_output = cooked_output_owner.view(shape=(2,), strides=(-1,), offset=2, access="write")
    signed_pipeline = vd.load_cooked_vjp_asset(signed_manifest)
    cooked_signed_pullback = cooked_pullback(
        signed_pipeline,
        {"particles": cooked_particles, "output": cooked_output},
    )
    cooked_signed_gradient = cooked_signed_pullback(signed_cotangent)["particles"]
    np.testing.assert_array_equal(cooked_output_owner.to_numpy(), direct_output_owner.to_numpy())
    np.testing.assert_array_equal(
        cooked_signed_gradient["velocity"].to_numpy(),
        direct_signed_gradient["velocity"].to_numpy(),
    )
    np.testing.assert_array_equal(
        cooked_signed_gradient["mass"].to_numpy(),
        direct_signed_gradient["mass"].to_numpy(),
    )

    direct_field_particles = particle_storage((2,))
    direct_field_output = vd.storage.zeros(dtype=Particle, shape=(2,))
    field_cotangent = vd.storage.tangent_zeros(dtype=Particle, shape=(2,))
    field_cotangent["velocity"].copy_from_numpy(np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32))
    field_cotangent["mass"].copy_from_numpy(np.array([3.0, 7.0], dtype=np.float32))
    direct_field_pullback = direct_pullback(
        field_path_objective,
        ("particles.velocity",),
        ("output",),
        direct_field_particles,
        direct_field_output,
    )
    direct_field_gradient = direct_field_pullback(field_cotangent)["particles.velocity"]

    cooked_field_particles = particle_storage((2,))
    cooked_field_output = vd.storage.zeros(dtype=Particle, shape=(2,))
    field_pipeline = vd.load_cooked_vjp_asset(field_manifest)
    cooked_field_pullback = cooked_pullback(
        field_pipeline,
        {"particles": cooked_field_particles, "output": cooked_field_output},
    )
    cooked_field_gradient = cooked_field_pullback(field_cotangent)["particles.velocity"]
    np.testing.assert_array_equal(
        cooked_field_gradient["velocity"].to_numpy(),
        direct_field_gradient["velocity"].to_numpy(),
    )
    np.testing.assert_array_equal(
        cooked_field_gradient["velocity"].to_numpy(),
        np.array([[2.0, 4.0], [8.0, 10.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        cooked_field_gradient["mass"].to_numpy(),
        np.zeros((2,), dtype=np.float32),
    )

    def run_scatter(cooked: bool) -> np.ndarray:
        owner = vd.storage.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        values = owner.view(shape=(2,), strides=(2,), offset=1, access="read")
        loss_owner = vd.storage.zeros(dtype=vd.f32, shape=(3,))
        loss = loss_owner.view(shape=(1,), strides=(1,), offset=1, access="write")
        if cooked:
            pipeline = vd.load_cooked_vjp_asset(scatter_manifest)
            pullback = cooked_pullback(pipeline, {"values": values, "loss": loss})
        else:
            pullback = direct_pullback(
                strided_scatter_objective,
                ("values",),
                ("loss",),
                values,
                loss,
            )
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
            pullback = cooked_pullback(
                pipeline,
                {"values": values, "count": np.int32(4), "output": output},
            )
        else:
            pullback = direct_pullback(
                gather_objective,
                ("values",),
                ("output",),
                values,
                np.int32(4),
                output,
            )
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

    empty = vd.storage.zeros(dtype=vd.f32, shape=(0,))
    direct_output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
    direct_empty_pullback = direct_pullback(
        gather_objective,
        ("values",),
        ("output",),
        empty,
        np.int32(0),
        direct_output,
    )
    gather_pipeline = vd.load_cooked_vjp_asset(gather_manifest)
    cooked_output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
    cooked_empty_pullback = cooked_pullback(
        gather_pipeline,
        {"values": empty, "count": np.int32(0), "output": cooked_output},
    )
    np.testing.assert_array_equal(direct_output.to_numpy(), np.zeros((1,), dtype=np.float32))
    np.testing.assert_array_equal(cooked_output.to_numpy(), direct_output.to_numpy())
    np.testing.assert_array_equal(
        cooked_empty_pullback(None)["values"].to_numpy(),
        direct_empty_pullback(None)["values"].to_numpy(),
    )

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
            pullback = cooked_pullback(
                pipeline,
                {"particles": particles, "first": first, "second": second},
            )
        else:
            pullback = direct_pullback(
                multi_output_objective,
                ("particles",),
                ("first", "second"),
                particles,
                first,
                second,
            )
        return pullback(multi_cotangents())["particles"]

    direct_multi = run_multi(False)
    cooked_multi = run_multi(True)
    np.testing.assert_array_equal(cooked_multi["velocity"].to_numpy(), direct_multi["velocity"].to_numpy())
    np.testing.assert_array_equal(cooked_multi["mass"].to_numpy(), direct_multi["mass"].to_numpy())
    np.testing.assert_array_equal(cooked_multi["velocity"].to_numpy(), np.array([[5.0, 7.0]], dtype=np.float32))
    np.testing.assert_array_equal(cooked_multi["mass"].to_numpy(), np.array([45.0], dtype=np.float32))

    mixed_pipeline = vd.load_cooked_vjp_asset(mixed_manifest)

    def run_mixed(cooked: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        owner = vd.storage.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        left = owner.view(shape=(2,), strides=(2,), offset=0, access="read")
        right = owner.view(shape=(2,), strides=(2,), offset=1, access="read")
        loss = vd.storage.zeros(dtype=vd.f32, shape=(2,))
        if cooked:
            pullback = cooked_pullback(
                mixed_pipeline,
                {"left": left, "right": right, "scale": np.float32(2.0), "loss": loss},
                grid=(2, 1, 1),
            )
        else:
            pullback = direct_pullback(
                mixed_alias_objective,
                ("left", "right", "scale"),
                ("loss",),
                left,
                right,
                np.float32(2.0),
                loss,
                grid=(2, 1, 1),
            )
        cotangent = (
            np.array([1.0, 3.0], dtype=np.float32)
            if cooked
            else np.array([[[[1.0, 0.0], [0.0, 3.0]]]], dtype=np.float32)
        )
        first = pullback(cotangent)
        second = pullback(cotangent)
        if first["left"] is not first["right"] or second["left"] is not second["right"]:
            raise AssertionError("aliased wrt views must materialize one shared gradient owner")
        if first["left"] is second["left"]:
            raise AssertionError("repeated pullbacks must return fresh gradient owners")
        np.testing.assert_array_equal(first["left"].to_numpy(), second["left"].to_numpy())
        return first["left"].to_numpy(), np.asarray(first["scale"]), loss.to_numpy()

    direct_owner_gradient, direct_scale_gradient, direct_loss = run_mixed(False)
    cooked_owner_gradient, cooked_scale_gradient, cooked_loss = run_mixed(True)
    np.testing.assert_array_equal(cooked_owner_gradient, direct_owner_gradient)
    np.testing.assert_array_equal(cooked_scale_gradient, direct_scale_gradient)
    np.testing.assert_array_equal(cooked_loss, direct_loss)
    np.testing.assert_array_equal(cooked_owner_gradient, np.array([8.0, 0.0, 0.0, 32.0], dtype=np.float32))
    np.testing.assert_array_equal(cooked_scale_gradient, np.float32(4.0))
    np.testing.assert_array_equal(cooked_loss, np.array([18.0, 18.0], dtype=np.float32))

    partial_pipeline = vd.load_cooked_vjp_asset(partial_manifest)
    for outer, inner in ((3, 5), (7, 9)):
        shape = (outer, 2, 4, inner)
        values_array = np.zeros(shape, dtype=np.float32)
        values_array[0, 1, 3, 0] = 3.0
        values_array[outer - 1, 0, 0, inner - 1] = 4.0

        direct_values = vd.storage.from_numpy(values_array)
        direct_loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        direct_partial_pullback = direct_pullback(
            partially_dynamic_objective,
            ("values",),
            ("loss",),
            direct_values,
            np.int32(outer),
            np.int32(inner),
            direct_loss,
        )

        cooked_values = vd.storage.from_numpy(values_array)
        cooked_loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        cooked_partial_pullback = cooked_pullback(
            partial_pipeline,
            {
                "values": cooked_values,
                "outer": np.int32(outer),
                "inner": np.int32(inner),
                "loss": cooked_loss,
            },
        )
        direct_gradient = direct_partial_pullback(None)["values"].to_numpy()
        cooked_gradient = cooked_partial_pullback(None)["values"].to_numpy()
        np.testing.assert_array_equal(cooked_gradient, direct_gradient)
        expected = np.zeros(shape, dtype=np.float32)
        expected[0, 1, 3, 0] = 6.0
        expected[outer - 1, 0, 0, inner - 1] = 1.0
        np.testing.assert_array_equal(cooked_gradient, expected)

    invalid = vd.storage.zeros(dtype=vd.f32, shape=(3, 3, 4, 5))
    loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
    try:
        partial_pipeline.vjp(
            {"values": invalid, "outer": np.int32(3), "inner": np.int32(5), "loss": loss},
            (1, 1, 1),
        )
    except ValueError as error:
        if "bound shape for Program value 0 conflicts with the declared Program shape" not in str(error):
            raise
    else:
        raise AssertionError("partially dynamic cooked VJP accepted a mismatched static extent")


if __name__ == "__main__":
    main()
