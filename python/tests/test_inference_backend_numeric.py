from __future__ import annotations

import unittest
from typing import Annotated

import numpy as np
import vernon_dsl as vd
from backend_test_matrix import BackendRequirements, BackendRow, backend_matrix_test, expand_backend_matrix_tests

from python.tests.compiler_test_support import compile_kernel_artifact


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
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
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
    output: vd.TensorView[vd.f16, (vd.dyn,), vd.write],
    values: vd.TensorView[vd.f16, (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = values[x] * vd.f16(2)


@vd.kernel(workgroup_size=(8, 1, 1))
def scale_f64(
    output: vd.TensorView[vd.f64, (vd.dyn,), vd.write],
    values: vd.TensorView[vd.f64, (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = values[x] * vd.f64(2)


@expand_backend_matrix_tests
class InferenceBackendNumericTests(unittest.TestCase):
    @staticmethod
    def _run(architecture: object, values: np.ndarray) -> np.ndarray:
        vd.init(arch=architecture)  # type: ignore[arg-type]
        output = vd.storage.zeros(dtype=vd.f32, shape=values.shape)
        inferred_numeric(
            output,
            vd.storage.from_numpy(values),
            grid=((values.size + 7) // 8, 1, 1),
        )
        return output.to_numpy()

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_inferred_helpers_backend_parity(self, backend: BackendRow) -> None:
        values = np.linspace(-1.0, 2.0, 16, dtype=np.float32)
        indices = np.arange(values.size, dtype=np.float32)
        expected = values + np.float32(4.0) + indices + indices / np.float32(2.0)
        np.testing.assert_allclose(self._run(backend.architecture, values), expected, rtol=0.0, atol=1e-6)

    def test_specialization_and_artifact_generation_are_deterministic(self) -> None:
        first_source, first_reflection = compile_kernel_artifact(inferred_numeric, "cpu")
        second_source, second_reflection = compile_kernel_artifact(inferred_numeric, "cpu")
        self.assertEqual(first_source, second_source)
        self.assertEqual(first_reflection, second_reflection)
        self.assertTrue(first_source)
        self.assertIn('"target":"cpu"', first_reflection)

    @staticmethod
    def _assert_scaled(kernel: object, dsl_type: type, numpy_type: type, tolerance: float) -> None:
        values_array = np.linspace(0.25, 2.0, 8, dtype=numpy_type)
        output = vd.storage.zeros(dtype=dsl_type, shape=values_array.shape)
        kernel(
            output,
            vd.storage.from_numpy(values_array),
            grid=((values_array.size + 7) // 8, 1, 1),
        )
        np.testing.assert_allclose(
            output.to_numpy(),
            values_array * numpy_type(2),
            rtol=tolerance,
            atol=tolerance,
        )

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True, f16=True))
    def test_f16_support_executes(self, backend: BackendRow) -> None:
        self._assert_scaled(scale_f16, vd.f16, np.float16, 2e-3)

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True, f64=True))
    def test_f64_support_executes(self, backend: BackendRow) -> None:
        self._assert_scaled(scale_f64, vd.f64, np.float64, 1e-12)


if __name__ == "__main__":
    unittest.main()
