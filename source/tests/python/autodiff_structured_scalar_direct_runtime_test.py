from __future__ import annotations

import sys


def main() -> None:
    sys.path[:0] = [sys.argv[1], sys.argv[2]]

    import numpy as np
    import vernon_dsl as vd  # pyright: ignore[reportMissingImports]
    from autodiff_structured_scalar_asset import (  # pyright: ignore[reportMissingImports]
        asset,
        boundary_division_program,
        boundary_power_program,
        boundary_unary_program,
        control_flow_program,
        dynamic_program,
        half_program,
        triple_nested_program,
    )

    vd.init(arch=vd.cpu)

    def reference(x: float, y: float, z: float) -> float:
        linear = x + y
        difference = x - y
        return float(
            linear * difference / y
            - z
            + np.sin(x)
            + np.cos(y)
            + np.exp(z)
            + np.log(x)
            + np.sqrt(y)
            + np.arccos(z)
            + np.arctan2(y, x)
            + np.abs(y - z)
            + x**y
        )

    def finite_gradients(x: float, y: float, z: float) -> dict[str, np.float32]:
        epsilon = 1.0e-4
        values = [x, y, z]
        gradients: dict[str, np.float32] = {}
        for index, name in enumerate(("x", "y", "z")):
            positive = values.copy()
            negative = values.copy()
            positive[index] += epsilon
            negative[index] -= epsilon
            gradients[name] = np.float32((reference(*positive) - reference(*negative)) / (2.0 * epsilon))
        return gradients

    rng = np.random.default_rng(0xAD2026)
    for x, y, z in ((1.2, 0.7, 0.2), (0.8, 0.3, 0.6)):
        expected = finite_gradients(x, y, z)
        output, pullback = asset.program(
            np.float32(x),
            np.float32(y),
            np.float32(z),
            grid=(1, 1, 1),
        )
        np.testing.assert_allclose(output, np.float32(reference(x, y, z)), rtol=2.0e-5, atol=2.0e-5)

        gradients = pullback()
        assert set(gradients) == {"x", "y", "z"}
        for name in expected:
            np.testing.assert_allclose(gradients[name], expected[name], rtol=3.0e-3, atol=3.0e-3)

        seed = np.float32(rng.uniform(-2.0, 2.0))
        reused = pullback(seed)
        for name in expected:
            np.testing.assert_allclose(
                reused[name],
                seed * expected[name],
                rtol=3.0e-3,
                atol=3.0e-3,
            )

    for _ in range(24):
        x = float(rng.uniform(0.65, 1.35))
        y = float(rng.uniform(0.35, 0.85))
        z = float(rng.uniform(0.1, 0.75))
        seed = np.float32(rng.uniform(-2.0, 2.0))
        expected = finite_gradients(x, y, z)
        _, randomized_pullback = asset.program(
            np.float32(x),
            np.float32(y),
            np.float32(z),
            grid=(1, 1, 1),
        )
        randomized_gradients = randomized_pullback(seed)
        for name in expected:
            np.testing.assert_allclose(
                randomized_gradients[name],
                seed * expected[name],
                rtol=4.0e-3,
                atol=4.0e-3,
            )

    def assert_scalar_classification(actual: object, expected: float) -> None:
        value = float(np.asarray(actual))
        if np.isnan(expected):
            assert np.isnan(value), value
        elif np.isinf(expected):
            assert np.isinf(value), value
            assert np.signbit(value) == np.signbit(expected), value
        else:
            np.testing.assert_allclose(value, expected, rtol=5.0e-5, atol=2.0e-6)

    unary_boundaries = (
        (0.0, 0, -np.inf, np.inf),
        (-1.0, 0, np.nan, -1.0),
        (0.0, 1, 0.0, np.inf),
        (-1.0, 1, np.nan, np.nan),
        (1.0, 2, 0.0, -np.inf),
        (1.01, 2, np.nan, np.nan),
        (0.0, 3, 0.0, 0.0),
        (-0.0, 3, 0.0, 0.0),
    )
    boundary_seed = np.float32(-1.5)
    for x, mode, expected_output, expected_gradient in (
        (1.0e-4, 0, float(np.log(np.float32(1.0e-4))), 1.0e4),
        (1.0e-4, 1, 1.0e-2, 50.0),
        (0.9999, 2, float(np.arccos(np.float32(0.9999))), -1.0 / np.sqrt(1.0 - np.float32(0.9999) ** 2)),
        (1.0e-7, 3, 1.0e-7, 1.0),
        (-1.0e-7, 3, 1.0e-7, -1.0),
    ):
        boundary_output, boundary_pullback = boundary_unary_program(
            np.float32(x),
            np.int32(mode),
            grid=(1, 1, 1),
        )
        assert_scalar_classification(boundary_output, expected_output)
        assert_scalar_classification(boundary_pullback()["x"], float(expected_gradient))
        assert_scalar_classification(
            boundary_pullback(boundary_seed)["x"],
            float(boundary_seed) * float(expected_gradient),
        )

    for x, mode, expected_output, expected_gradient in unary_boundaries:
        boundary_output, boundary_pullback = boundary_unary_program(
            np.float32(x),
            np.int32(mode),
            grid=(1, 1, 1),
        )
        assert_scalar_classification(boundary_output, expected_output)
        assert_scalar_classification(boundary_pullback()["x"], expected_gradient)
        assert_scalar_classification(boundary_pullback(boundary_seed)["x"], float(boundary_seed) * expected_gradient)

    division_output, division_pullback = boundary_division_program(
        np.float32(2.0),
        np.float32(0.0),
        grid=(1, 1, 1),
    )
    assert_scalar_classification(division_output, np.inf)
    assert_scalar_classification(division_pullback()["x"], np.inf)
    assert_scalar_classification(division_pullback()["y"], -np.inf)
    division_seeded = division_pullback(boundary_seed)
    assert_scalar_classification(division_seeded["x"], -np.inf)
    assert_scalar_classification(division_seeded["y"], np.inf)

    for base, exponent, expected_output, expected_base, expected_exponent in (
        (-2.0, 3.0, -8.0, 12.0, np.nan),
        (-2.0, 0.5, np.nan, np.nan, np.nan),
        (0.0, 2.0, 0.0, 0.0, np.nan),
    ):
        power_output, power_pullback = boundary_power_program(
            np.float32(base),
            np.float32(exponent),
            grid=(1, 1, 1),
        )
        assert_scalar_classification(power_output, expected_output)
        power_gradients = power_pullback()
        assert_scalar_classification(power_gradients["base"], expected_base)
        assert_scalar_classification(power_gradients["exponent"], expected_exponent)
        seeded_power_gradients = power_pullback(boundary_seed)
        assert_scalar_classification(seeded_power_gradients["base"], float(boundary_seed) * expected_base)
        assert_scalar_classification(seeded_power_gradients["exponent"], float(boundary_seed) * expected_exponent)

    del pullback
    vd.init(arch=vd.cpu)
    output, _ = asset.program(
        np.float32(1.2),
        np.float32(0.7),
        np.float32(0.2),
        grid=(1, 1, 1),
    )
    np.testing.assert_allclose(output, np.float32(reference(1.2, 0.7, 0.2)), rtol=2.0e-5, atol=2.0e-5)

    expected = finite_gradients(1.2, 0.7, 0.2)
    outputs, pullback = asset.program(
        np.float32(1.2),
        np.float32(0.7),
        np.float32(0.2),
        grid=(2, 1, 1),
    )
    np.testing.assert_allclose(
        outputs,
        np.full((1, 1, 2), reference(1.2, 0.7, 0.2), dtype=np.float32),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    gradients = pullback(np.array([[[1.0, 2.0]]], dtype=np.float32))
    for name in expected:
        np.testing.assert_allclose(gradients[name], np.float32(3.0) * expected[name], rtol=3.0e-3, atol=3.0e-3)

    half_output, half_pullback = half_program(
        np.float16(1.5),
        np.float16(0.25),
        grid=(1, 1, 1),
    )
    np.testing.assert_array_equal(half_output, np.float16(1.875))
    half_gradients = half_pullback()
    np.testing.assert_array_equal(half_gradients["x"], np.float32(1.25))
    np.testing.assert_array_equal(half_gradients["y"], np.float32(1.5))

    for count in (0, 1, 32, 1500):
        dynamic_output, dynamic_pullback = dynamic_program(
            np.float32(1.25),
            np.int32(count),
            grid=(1, 1, 1),
        )
        expected_scale = np.float32(1 + 2 * count)
        np.testing.assert_array_equal(dynamic_output, expected_scale * np.float32(1.25))
        np.testing.assert_array_equal(dynamic_pullback()["x"], expected_scale)
        np.testing.assert_array_equal(dynamic_pullback(np.float32(2))["x"], 2 * expected_scale)
        epsilon = np.float32(1.25e-1)
        positive, _ = dynamic_program(np.float32(1.25) + epsilon, np.int32(count), grid=(1, 1, 1))
        negative, _ = dynamic_program(np.float32(1.25) - epsilon, np.int32(count), grid=(1, 1, 1))
        finite = (positive - negative) / (2 * epsilon)
        np.testing.assert_allclose(dynamic_pullback()["x"], finite, rtol=2.0e-3, atol=2.0e-3)

    mutable_x = np.array(1.25, dtype=np.float32)
    _, stable_pullback = dynamic_program(mutable_x, np.int32(1), grid=(1, 1, 1))
    mutable_x[...] = np.float32(100)
    np.testing.assert_array_equal(stable_pullback()["x"], np.float32(3))

    def control_reference(x: float, y: float, limit: int, mode: int) -> float:
        result = x + y
        index = 0
        while index < limit:
            index += 1
            inner = 0
            while inner < 3:
                inner += 1
                if inner < 2:
                    continue
                result = result * x + y
                if mode == 1 and index == 2:
                    return result
            if mode == 2 and index > 2:
                break
        else:
            result += x * y
        return result

    def control_finite_difference(x: float, y: float, limit: int, mode: int) -> tuple[np.float32, np.float32]:
        epsilon = 2.0e-4
        dx = (control_reference(x + epsilon, y, limit, mode) - control_reference(x - epsilon, y, limit, mode)) / (
            2.0 * epsilon
        )
        dy = (control_reference(x, y + epsilon, limit, mode) - control_reference(x, y - epsilon, limit, mode)) / (
            2.0 * epsilon
        )
        return np.float32(dx), np.float32(dy)

    for x, y, limit, mode in (
        (0.8, 0.3, 0, 0),
        (0.8, 0.3, 1, 0),
        (0.8, 0.3, 4, 0),
        (0.8, 0.3, 5, 1),
        (0.8, 0.3, 8, 2),
    ):
        control_output, control_pullback = control_flow_program(
            np.float32(x),
            np.float32(y),
            np.int32(limit),
            np.int32(mode),
            grid=(1, 1, 1),
        )
        expected_dx, expected_dy = control_finite_difference(x, y, limit, mode)
        np.testing.assert_allclose(
            control_output,
            np.float32(control_reference(x, y, limit, mode)),
            rtol=2.0e-5,
            atol=2.0e-5,
        )
        gradients = control_pullback()
        np.testing.assert_allclose(gradients["x"], expected_dx, rtol=4.0e-3, atol=4.0e-3)
        np.testing.assert_allclose(gradients["y"], expected_dy, rtol=4.0e-3, atol=4.0e-3)
        seed = np.float32(rng.uniform(-2.0, 2.0))
        reused = control_pullback(seed)
        np.testing.assert_allclose(reused["x"], seed * expected_dx, rtol=4.0e-3, atol=4.0e-3)
        np.testing.assert_allclose(reused["y"], seed * expected_dy, rtol=4.0e-3, atol=4.0e-3)

    def triple_reference(
        x: float,
        y: float,
        outer_limit: int,
        middle_limit: int,
        inner_limit: int,
    ) -> float:
        result = x + y
        for _ in range(outer_limit):
            for middle in range(middle_limit):
                for inner in range(inner_limit):
                    if inner < middle:
                        result = result * x + y
                    else:
                        result = result + x * y
        return result

    for limits in ((0, 2, 3), (1, 1, 1), (2, 3, 4)):
        x = 0.7
        y = 0.2
        triple_output, triple_pullback = triple_nested_program(
            np.float32(x),
            np.float32(y),
            *(np.int32(limit) for limit in limits),
            grid=(1, 1, 1),
        )
        epsilon = 2.0e-4
        expected_dx = (triple_reference(x + epsilon, y, *limits) - triple_reference(x - epsilon, y, *limits)) / (
            2.0 * epsilon
        )
        expected_dy = (triple_reference(x, y + epsilon, *limits) - triple_reference(x, y - epsilon, *limits)) / (
            2.0 * epsilon
        )
        np.testing.assert_allclose(triple_output, np.float32(triple_reference(x, y, *limits)), rtol=2.0e-5, atol=2.0e-5)
        triple_gradients = triple_pullback()
        np.testing.assert_allclose(triple_gradients["x"], np.float32(expected_dx), rtol=5.0e-3, atol=5.0e-3)
        np.testing.assert_allclose(triple_gradients["y"], np.float32(expected_dy), rtol=5.0e-3, atol=5.0e-3)
        seed = np.float32(rng.uniform(-2.0, 2.0))
        triple_reused = triple_pullback(seed)
        np.testing.assert_allclose(triple_reused["x"], seed * np.float32(expected_dx), rtol=5.0e-3, atol=5.0e-3)
        np.testing.assert_allclose(triple_reused["y"], seed * np.float32(expected_dy), rtol=5.0e-3, atol=5.0e-3)

    assert not hasattr(asset.program, "_direct_runtime")

    vd.Kernel.clear_cache()
    output, _ = asset.program(
        np.float32(1.2),
        np.float32(0.7),
        np.float32(0.2),
        grid=(1, 1, 1),
    )
    np.testing.assert_allclose(output, np.float32(reference(1.2, 0.7, 0.2)), rtol=2.0e-5, atol=2.0e-5)


if __name__ == "__main__":
    main()
