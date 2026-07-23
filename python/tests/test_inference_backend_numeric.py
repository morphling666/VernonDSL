from __future__ import annotations

import unittest
from typing import Annotated

import numpy as np
import vernon_dsl as vd


@vd.func
def identity(value):
    return value


@vd.func
def ratio(numerator, denominator):
    return numerator / denominator


@vd.func
def transform(value):
    matrix = vd.Matrix([[1.0, 2.0], [3.0, 4.0]])
    return vd.matmul(matrix, value)


@vd.kernel(workgroup_size=(8, 1, 1))
def inferred_numeric(
    output: vd.Tensor[vd.f32, (None,)],
    values: vd.Tensor[vd.f32, (None,)],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    vector = vd.Vector([values[x], 2])
    transformed = transform(vector)
    scalar_specialization = identity(transformed[0])
    integer_specialization = identity(vd.i32(x))
    output[x] = scalar_specialization + vd.f32(integer_specialization) + ratio(vd.i32(x), 2)


@vd.kernel(workgroup_size=(8, 1, 1))
def scale_f16(
    output: vd.Tensor[vd.f16, (None,)],
    values: vd.Tensor[vd.f16, (None,)],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = values[x] * vd.f16(2)


@vd.kernel(workgroup_size=(8, 1, 1))
def scale_f64(
    output: vd.Tensor[vd.f64, (None,)],
    values: vd.Tensor[vd.f64, (None,)],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = values[x] * vd.f64(2)


class InferenceBackendNumericTests(unittest.TestCase):
    @staticmethod
    def _available(architecture: object) -> bool:
        try:
            vd.init(arch=architecture)  # type: ignore[arg-type]
        except RuntimeError:
            vd.init(arch=vd.cpu)
            return False
        return True

    @staticmethod
    def _run(architecture: object, values: np.ndarray) -> np.ndarray:
        vd.init(arch=architecture)  # type: ignore[arg-type]
        output = vd.Tensor.zeros(dtype=vd.f32, shape=values.shape)
        inferred_numeric(
            output,
            vd.Tensor.from_numpy(values),
            grid=(values.size, 1, 1),
        )
        return output.to_numpy()

    def test_inferred_helpers_execute_on_cpu(self) -> None:
        values = np.linspace(-1.0, 2.0, 16, dtype=np.float32)
        indices = np.arange(values.size, dtype=np.float32)
        expected = values + np.float32(4.0) + indices + indices / np.float32(2.0)
        np.testing.assert_allclose(self._run(vd.cpu, values), expected, rtol=0.0, atol=1e-6)

    def test_cpu_cuda_vulkan_numeric_parity(self) -> None:
        values = np.linspace(-1.0, 2.0, 16, dtype=np.float32)
        expected = self._run(vd.cpu, values)
        for architecture in (vd.cuda, vd.vulkan):
            with self.subTest(backend=architecture.name):
                if not self._available(architecture):
                    self.skipTest(f"{architecture.name} runtime is unavailable")
                np.testing.assert_allclose(
                    self._run(architecture, values),
                    expected,
                    rtol=0.0,
                    atol=1e-6,
                )

    def test_specialization_and_artifact_generation_are_deterministic(self) -> None:
        values = vd.Tensor.from_numpy(np.arange(8, dtype=np.float32))
        output = vd.Tensor.zeros(dtype=vd.f32, shape=(8,))
        first_source, first_reflection = inferred_numeric.compile_artifact(output, values, target="cpu")
        second_source, second_reflection = inferred_numeric.compile_artifact(output, values, target="cpu")
        self.assertEqual(first_source, second_source)
        self.assertEqual(first_reflection, second_reflection)
        self.assertTrue(first_source)
        self.assertIn('"target":"cpu"', first_reflection)

    def test_f16_f64_support_is_executable_or_explicitly_rejected(self) -> None:
        cases = (
            (scale_f16, vd.f16, np.float16),
            (scale_f64, vd.f64, np.float64),
        )
        for architecture in (vd.cpu, vd.cuda, vd.vulkan):
            if not self._available(architecture):
                continue
            for kernel, dsl_type, numpy_type in cases:
                with self.subTest(backend=architecture.name, dtype=dsl_type.name):
                    values_array = np.linspace(0.25, 2.0, 8, dtype=numpy_type)
                    output = vd.Tensor.zeros(dtype=dsl_type, shape=values_array.shape)
                    try:
                        kernel(
                            output,
                            vd.Tensor.from_numpy(values_array),
                            grid=(values_array.size, 1, 1),
                        )
                    except RuntimeError as error:
                        message = str(error).lower()
                        self.assertTrue(
                            any(word in message for word in (dsl_type.name, "unsupported", "illegal", "failed")),
                            message,
                        )
                        continue
                    np.testing.assert_allclose(
                        output.to_numpy(),
                        values_array * numpy_type(2),
                        rtol=2e-3 if numpy_type is np.float16 else 1e-12,
                        atol=2e-3 if numpy_type is np.float16 else 1e-12,
                    )


if __name__ == "__main__":
    unittest.main()
