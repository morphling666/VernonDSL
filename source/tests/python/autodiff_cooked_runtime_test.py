from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    manifest = Path(sys.argv[1])
    sys.path[:0] = [sys.argv[2], sys.argv[3]]

    import numpy as np
    import vernon_cooked_autodiff_fixture  # noqa: F401
    from vernon_dsl import _native

    runtime = _native.Runtime(_native.RuntimeBackend.CPU)
    pipeline = runtime.load_pipeline_asset(manifest.read_bytes(), str(manifest.parent), [])

    def expect_binding_error(bindings: dict[str, object], message: str) -> None:
        try:
            pipeline.vjp(bindings, (1, 1, 1))
        except ValueError as error:
            assert message in str(error), error
        else:
            raise AssertionError(f"expected binding error containing {message!r}")

    expect_binding_error(
        {
            "value": np.array([2.0, 3.0], dtype=np.float32),
            "count": np.int32(2),
        },
        "do not match pipeline parameters",
    )
    expect_binding_error(
        {
            "value": np.array([2.0, 3.0], dtype=np.float32),
            "parameters": {"factor": np.float32(1.0)},
            "count": np.int32(2),
        },
        "missing field 'selector'",
    )
    expect_binding_error(
        {
            "value": np.array([2.0, 3.0, 4.0], dtype=np.float32),
            "parameters": {
                "factor": np.float32(1.0),
                "selector": np.float32(1.0),
            },
            "count": np.int32(2),
        },
        "shape does not match",
    )

    def run_case(
        factor: float,
        selector: float,
        expected_output: list[float],
        expected_value_gradient: list[float],
    ) -> None:
        output, pullback = pipeline.vjp(
            {
                "value": np.array([2.0, 3.0], dtype=np.float32),
                "parameters": {
                    "factor": np.float32(factor),
                    "selector": np.float32(selector),
                },
                "count": np.int32(2),
            },
            (1, 1, 1),
        )
        np.testing.assert_array_equal(output, np.array(expected_output, dtype=np.float32))
        gradients = pullback(np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(gradients["value"], np.array(expected_value_gradient, dtype=np.float32))
        np.testing.assert_array_equal(gradients["parameters.factor"], np.float32(0.0))
        np.testing.assert_array_equal(gradients["parameters.selector"], np.float32(0.0))
        reused = pullback(np.array([2.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(reused["value"], np.array(expected_output, dtype=np.float32))
        np.testing.assert_array_equal(reused["parameters.factor"], np.float32(0.0))
        np.testing.assert_array_equal(reused["parameters.selector"], np.float32(0.0))

    # factor=0 proves the untaken division branch is never evaluated.
    run_case(0.0, 1.0, [4.0, 6.0], [2.0, 4.0])
    run_case(2.0, -1.0, [6.0, 9.0], [3.0, 6.0])


if __name__ == "__main__":
    main()
