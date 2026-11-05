from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    manifest = Path(sys.argv[1])
    sys.path[:0] = [sys.argv[2], sys.argv[3], sys.argv[4]]

    import numpy as np
    import vernon_cooked_autodiff_fixture  # noqa: F401
    import vernon_dsl as vd
    from autodiff_cooked_aggregate_asset import objective

    vd.init(arch=vd.cpu)
    cooked = vd.load_program(manifest)

    def output_storage() -> vd.TensorStorage:
        return vd.storage.zeros(dtype=vd.f32, shape=(2,))

    def output_cotangent(values: list[float]) -> vd.TensorStorage:
        cotangent = vd.storage.tangent_zeros(dtype=vd.f32, shape=(2,))
        cotangent.copy_from_numpy(np.array(values, dtype=np.float32))
        return cotangent

    def parameters(factor: float, selector: float) -> dict[str, np.float32]:
        return {
            "factor": np.float32(factor),
            "selector": np.float32(selector),
        }

    def expect_binding_error(bindings: dict[str, object], message: str) -> None:
        try:
            cooked.vjp(bindings, (1, 1, 1))
        except ValueError as error:
            assert message in str(error), error
        else:
            raise AssertionError(f"expected binding error containing {message!r}")

    expect_binding_error(
        {
            "value": np.array([2.0, 3.0], dtype=np.float32),
            "parameters": parameters(1.0, 1.0),
            "count": np.int32(2),
        },
        "do not match Program parameters",
    )
    expect_binding_error(
        {
            "value": np.array([2.0, 3.0], dtype=np.float32),
            "parameters": {"factor": np.float32(1.0)},
            "count": np.int32(2),
            "output": output_storage(),
        },
        "missing field 'selector'",
    )
    expect_binding_error(
        {
            "value": np.array([2.0, 3.0, 4.0], dtype=np.float32),
            "parameters": parameters(1.0, 1.0),
            "count": np.int32(2),
            "output": output_storage(),
        },
        "shape [3] does not match reflection [2]",
    )

    def run_case(
        factor: float,
        selector: float,
        expected_output: list[float],
        expected_value_gradient: list[float],
    ) -> None:
        value = np.array([2.0, 3.0], dtype=np.float32)
        parameter_values = parameters(factor, selector)
        direct_output = output_storage()
        _, direct_pullback = vd.ad.vjp(
            objective,
            wrt=("value", "parameters.factor", "parameters.selector"),
            outputs=("output",),
        )(value, parameter_values, np.int32(2), direct_output, grid=(1, 1, 1))

        cooked_output = output_storage()
        _, cooked_pullback = cooked.vjp(
            {
                "value": value,
                "parameters": parameter_values,
                "count": np.int32(2),
                "output": cooked_output,
            },
            (1, 1, 1),
        )

        np.testing.assert_array_equal(direct_output.to_numpy(), np.array(expected_output, dtype=np.float32))
        np.testing.assert_array_equal(cooked_output.to_numpy(), direct_output.to_numpy())

        cotangent = output_cotangent([1.0, 2.0])
        direct_gradients = direct_pullback(cotangent)
        cooked_gradients = cooked_pullback(cotangent)
        np.testing.assert_array_equal(cooked_gradients["value"], direct_gradients["value"])
        np.testing.assert_array_equal(
            cooked_gradients["value"],
            np.array(expected_value_gradient, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            cooked_gradients["parameters.factor"],
            direct_gradients["parameters.factor"],
        )
        np.testing.assert_array_equal(
            cooked_gradients["parameters.selector"],
            direct_gradients["parameters.selector"],
        )
        np.testing.assert_array_equal(cooked_gradients["parameters.factor"], np.float32(0.0))
        np.testing.assert_array_equal(cooked_gradients["parameters.selector"], np.float32(0.0))

        reused = cooked_pullback(output_cotangent([2.0, 3.0]))
        np.testing.assert_array_equal(reused["value"], np.array(expected_output, dtype=np.float32))
        np.testing.assert_array_equal(reused["parameters.factor"], np.float32(0.0))
        np.testing.assert_array_equal(reused["parameters.selector"], np.float32(0.0))

    # factor=0 proves the untaken division branch is never evaluated.
    run_case(0.0, 1.0, [4.0, 6.0], [2.0, 4.0])
    run_case(2.0, -1.0, [6.0, 9.0], [3.0, 6.0])


if __name__ == "__main__":
    main()
