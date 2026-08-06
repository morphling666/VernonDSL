from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    sys.path[:0] = [sys.argv[1], sys.argv[2]]

    import numpy as np
    import vernon_dsl as vd  # pyright: ignore[reportMissingImports]
    from autodiff_structured_scalar_asset import asset, half_program  # pyright: ignore[reportMissingImports]
    from vernon_dsl.frontend.autodiff_cpu import execute_program_graph  # pyright: ignore[reportMissingImports]
    from vernon_dsl.frontend.compiler import Compiler, FrontendCompileRequest  # pyright: ignore[reportMissingImports]

    vd.init(arch=vd.cpu)
    legacy = Compiler().compile_request(
        FrontendCompileRequest(
            Path(sys.argv[2]) / "autodiff_structured_scalar_asset.py",
            "objective",
            program_transform=asset.program.transform,
        )
    )
    assert legacy.program_graph is not None

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

    for x, y, z in ((1.2, 0.7, 0.2), (0.8, 0.3, 0.6)):
        expected = finite_gradients(x, y, z)
        bindings = {
            "x": np.float32(x),
            "y": np.float32(y),
            "z": np.float32(z),
        }
        legacy_output, legacy_pullback = execute_program_graph(legacy.program_graph, bindings, (1, 1, 1))
        legacy_gradients = legacy_pullback()
        output, pullback = asset.program(
            np.float32(x),
            np.float32(y),
            np.float32(z),
            grid=(1, 1, 1),
        )
        np.testing.assert_allclose(output, np.float32(reference(x, y, z)), rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(output, legacy_output, rtol=2.0e-5, atol=2.0e-5)

        gradients = pullback()
        assert set(gradients) == {"x", "y", "z"}
        for name in expected:
            np.testing.assert_allclose(gradients[name], expected[name], rtol=3.0e-3, atol=3.0e-3)
            np.testing.assert_allclose(gradients[name], legacy_gradients[name], rtol=2.0e-5, atol=2.0e-5)

        seed = np.float32(1.75)
        reused = pullback(seed)
        legacy_reused = legacy_pullback(seed)
        for name in expected:
            np.testing.assert_allclose(
                reused[name],
                seed * expected[name],
                rtol=3.0e-3,
                atol=3.0e-3,
            )
            np.testing.assert_allclose(reused[name], legacy_reused[name], rtol=2.0e-5, atol=2.0e-5)

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
