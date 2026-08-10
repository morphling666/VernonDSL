from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    manifest = Path(sys.argv[1])
    sys.path[:0] = sys.argv[2:]

    import vernon_dsl as vd
    from vernon_dsl._runtime import session

    vd.init(arch=vd.cpu)
    assert session._native_runtime is not None
    runtime = session._native_runtime
    __import__("vernon_native_autodiff_numeric_cpu_fixture")

    try:
        run_numeric_acceptance(runtime, manifest)
    finally:
        vd.init(arch=vd.cpu)


def run_numeric_acceptance(runtime, manifest: Path) -> None:
    import numpy as np
    import vernon_dsl as vd

    pipeline = runtime.load_pipeline_asset(manifest.read_bytes(), str(manifest.parent), [])

    def invoke(native_inputs):
        storages = {}
        bindings = dict(native_inputs)
        for name in ("values", "auxiliary"):
            source = native_inputs[name]
            storage = storages.setdefault(
                id(source),
                vd.storage.from_numpy(source.reshape((1, 1, 1) + source.shape)),
            )
            bindings[name] = storage
        output_storage = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
        bindings["output"] = output_storage
        output, pullback = pipeline.vjp(bindings, (1, 1, 1))
        assert output is None
        for name in ("values", "auxiliary"):
            np.copyto(native_inputs[name], bindings[name].to_numpy().reshape(native_inputs[name].shape))

        def apply_pullback():
            gradients = pullback(None)
            return {
                name: (
                    value.to_numpy().reshape(native_inputs[name].shape)
                    if isinstance(value, vd.TensorStorage) and name in {"values", "auxiliary"}
                    else value.to_numpy()
                    if isinstance(value, vd.TensorStorage)
                    else value
                )
                for name, value in gradients.items()
            }

        return output_storage.to_numpy()[0, 0, 0], apply_pullback

    differentiable = (
        "left",
        "right",
        "vector",
        "normal",
        "values",
        "auxiliary",
        "angle",
        "ordinate",
        "abscissa",
        "signed",
    )
    base = {
        "left": np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float64,
        ),
        "right": np.array([[[2.0, 0.0], [0.0, 3.0]]], dtype=np.float64),
        "vector": np.array([3.0, 4.0, 1.0], dtype=np.float64),
        "normal": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "values": np.array([1.0, 2.0], dtype=np.float64),
        "auxiliary": np.array([0.5, -0.5], dtype=np.float64),
        "angle": np.array(0.25, dtype=np.float64),
        "ordinate": np.array(0.75, dtype=np.float64),
        "abscissa": np.array(1.25, dtype=np.float64),
        "signed": np.array(-0.5, dtype=np.float64),
    }

    def reference(inputs: dict[str, np.ndarray], count: int, flag: int) -> tuple[float, np.ndarray, np.ndarray]:
        product = np.matmul(inputs["left"], inputs["right"])
        vector = inputs["vector"]
        normal = inputs["normal"]
        unit = vector / np.linalg.norm(vector)
        bounced = vector - 2.0 * np.dot(vector, normal) * normal
        term = (
            product[0, 0, 0]
            + product[1, 1, 1]
            + unit[0]
            + bounced[1]
            + np.arccos(inputs["angle"])
            + np.arctan2(inputs["ordinate"], inputs["abscissa"])
            + np.abs(inputs["signed"])
        )
        values = inputs["values"].copy()
        auxiliary = inputs["auxiliary"].copy()
        if flag > 0:
            values[0] = values[0] + term
            auxiliary[0] = auxiliary[0] + unit[1]
            for _ in range(count):
                values[0] = values[0] + unit[2]
        else:
            values[1] = values[1] * term
            auxiliary[1] = auxiliary[1] + bounced[0]
            for _ in range(count):
                auxiliary[1] = auxiliary[1] + unit[2]
        output = values[0] + values[1] + auxiliary[0] + auxiliary[1] + unit[2] + bounced[2]
        return float(output), values, auxiliary

    def finite_gradients(count: int, flag: int) -> dict[str, np.ndarray]:
        epsilon = 1.0e-4
        gradients: dict[str, np.ndarray] = {}
        for name in differentiable:
            gradient = np.zeros_like(base[name])
            for index in np.ndindex(base[name].shape):
                positive = {key: value.copy() for key, value in base.items()}
                negative = {key: value.copy() for key, value in base.items()}
                positive[name][index] += epsilon
                negative[name][index] -= epsilon
                positive_output, _, _ = reference(positive, count, flag)
                negative_output, _, _ = reference(negative, count, flag)
                gradient[index] = (positive_output - negative_output) / (2.0 * epsilon)
            gradients[name] = gradient
        return gradients

    count = 2
    for flag in (1, -1):
        native_inputs = {name: value.astype(np.float32) for name, value in base.items()}
        native_inputs["count"] = np.int32(count)
        native_inputs["flag"] = np.int32(flag)
        expected_output, expected_values, expected_auxiliary = reference(base, count, flag)
        expected_gradients = finite_gradients(count, flag)

        output, pullback = invoke(native_inputs)
        np.testing.assert_allclose(output, np.float32(expected_output), rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(native_inputs["values"], expected_values.astype(np.float32), rtol=0.0, atol=2.0e-5)
        np.testing.assert_allclose(
            native_inputs["auxiliary"],
            expected_auxiliary.astype(np.float32),
            rtol=0.0,
            atol=2.0e-5,
        )

        gradients = pullback()
        for name in differentiable:
            np.testing.assert_allclose(
                gradients[name],
                expected_gradients[name].astype(np.float32),
                rtol=2.0e-4,
                atol=2.0e-4,
                err_msg=f"gradient mismatch for {name} with flag={flag}",
            )

    atan2_edge_cases = (
        (0.0, -1.0),
        (-0.0, -1.0),
        (0.0, -0.0),
        (-0.0, -0.0),
        (np.inf, np.inf),
        (-np.inf, -np.inf),
    )
    for ordinate, abscissa in atan2_edge_cases:
        edge_inputs = {name: value.astype(np.float32) for name, value in base.items()}
        edge_inputs.update(
            ordinate=np.float32(ordinate),
            abscissa=np.float32(abscissa),
            count=np.int32(0),
            flag=np.int32(1),
        )
        reference_inputs = {name: value.copy() for name, value in base.items()}
        reference_inputs.update(
            ordinate=np.array(ordinate, dtype=np.float64),
            abscissa=np.array(abscissa, dtype=np.float64),
        )
        expected_output, _, _ = reference(reference_inputs, 0, 1)
        output, _ = invoke(edge_inputs)
        np.testing.assert_allclose(
            output,
            np.float32(expected_output),
            rtol=2.0e-5,
            atol=2.0e-5,
            err_msg=f"atan2 edge mismatch for y={ordinate}, x={abscissa}",
        )

    shared = np.array([1.0, 2.0], dtype=np.float32)
    aliased_inputs = {name: value.astype(np.float32) for name, value in base.items()}
    aliased_inputs.update(values=shared, auxiliary=shared, count=np.int32(count), flag=np.int32(1))
    try:
        invoke(aliased_inputs)
    except ValueError as error:
        if "overlap" not in str(error):
            raise
    else:
        raise AssertionError("native autodiff accepted overlapping writable Storage inputs")


if __name__ == "__main__":
    main()
