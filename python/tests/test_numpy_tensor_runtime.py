from __future__ import annotations

import unittest

import numpy as np
import vernon_dsl as vd


@vd.kernel(workgroup_size=(1, 1, 1))
def numpy_tensor_semantics(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    broadcast_left: vd.Tensor[vd.f32, (2, 1, 3)],
    broadcast_right: vd.Tensor[vd.f32, (1, 4, 1)],
    matmul_left: vd.Tensor[vd.f32, (4, 4, 4)],
    matmul_right: vd.Tensor[vd.f32, (4, 4, 2)],
) -> None:
    broadcasted = broadcast_left + broadcast_right
    product = vd.matmul(matmul_left, matmul_right)
    output[0] = broadcasted[0, 0, 0]
    output[1] = broadcasted[0, 3, 2]
    output[2] = broadcasted[1, 1, 1]
    output[3] = product[0, 0, 0]
    output[4] = product[1, 2, 1]
    output[5] = product[3, 3, 0]


@vd.kernel(workgroup_size=(1, 1, 1))
def cpu_numpy_tensor_semantics(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
) -> None:
    broadcast_left = vd.Tensor([[[values[0], values[1]]], [[values[2], values[3]]]])
    broadcast_right = vd.Tensor([[[values[4]], [values[5]]]])
    matmul_left = vd.Tensor(
        [
            [[values[6], values[7]], [values[8], values[9]]],
            [[values[10], values[11]], [values[12], values[13]]],
        ]
    )
    matmul_right = vd.Tensor([[[values[14], values[15]], [values[16], values[17]]]])
    broadcasted = broadcast_left + broadcast_right
    product = vd.matmul(matmul_left, matmul_right)
    output[0] = broadcasted[0, 0, 0]
    output[1] = broadcasted[0, 1, 1]
    output[2] = broadcasted[1, 1, 0]
    output[3] = product[0, 0, 0]
    output[4] = product[1, 0, 1]
    output[5] = product[1, 1, 0]


@vd.kernel(workgroup_size=(1, 1, 1))
def cpu_matmul_rank_categories(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
) -> None:
    left_vector = vd.Vector([values[0], values[1]])
    right_vector = vd.Vector([values[2], values[3]])
    left_matrix = vd.Matrix([[values[4], values[5]], [values[6], values[7]]])
    right_matrix = vd.Matrix([[values[8], values[9]], [values[10], values[11]]])
    vector_vector = vd.matmul(left_vector, right_vector)
    vector_matrix = vd.matmul(left_vector, right_matrix)
    matrix_vector = vd.matmul(left_matrix, right_vector)
    matrix_matrix = vd.matmul(left_matrix, right_matrix)
    output[0] = vector_vector
    output[1] = vector_matrix[0]
    output[2] = vector_matrix[1]
    output[3] = matrix_vector[0]
    output[4] = matrix_vector[1]
    output[5] = matrix_matrix[0, 0]
    output[6] = matrix_matrix[0, 1]
    output[7] = matrix_matrix[1, 0]
    output[8] = matrix_matrix[1, 1]


@vd.kernel(workgroup_size=(1, 1, 1))
def rank_three_tensor_arithmetic(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
) -> None:
    left = vd.Tensor(
        [
            [[values[0], values[1], values[2]], [values[3], values[4], values[5]]],
            [[values[6], values[7], values[8]], [values[9], values[10], values[11]]],
        ]
    )
    right = vd.Tensor(
        [
            [[values[12], values[13], values[14]], [values[15], values[16], values[17]]],
            [[values[18], values[19], values[20]], [values[21], values[22], values[23]]],
        ]
    )
    projection = vd.Tensor([[[values[24], values[25]], [values[26], values[27]], [values[28], values[29]]]])
    elementwise = ((left + right) - right) * right / right
    result = vd.matmul(elementwise, projection)
    output[0] = result[0, 0, 0]
    output[1] = result[0, 1, 1]
    output[2] = result[1, 0, 1]
    output[3] = result[1, 1, 0]


@vd.kernel(workgroup_size=(1, 1, 1))
def static_rank_three_tensor_arithmetic(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    left: vd.Tensor[vd.f32, (2, 2, 3)],
    elementwise_right: vd.Tensor[vd.f32, (2, 2, 3)],
    projection: vd.Tensor[vd.f32, (1, 3, 2)],
) -> None:
    elementwise = ((left + elementwise_right) - elementwise_right) * elementwise_right / elementwise_right
    result = vd.matmul(elementwise, projection)
    output[0] = result[0, 0, 0]
    output[1] = result[0, 1, 1]
    output[2] = result[1, 0, 1]
    output[3] = result[1, 1, 0]


class NumpyTensorRuntimeTests(unittest.TestCase):
    @staticmethod
    def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        broadcast_left = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
        broadcast_right = np.linspace(-0.5, 1.0, 4, dtype=np.float32).reshape(1, 4, 1)
        matmul_left = np.arange(64, dtype=np.float32).reshape(4, 4, 4) * np.float32(0.125)
        matmul_right = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(4, 4, 2)
        return broadcast_left, broadcast_right, matmul_left, matmul_right

    @classmethod
    def _expected(cls) -> np.ndarray:
        broadcast_left, broadcast_right, matmul_left, matmul_right = cls._inputs()
        broadcasted = broadcast_left + broadcast_right
        product = np.matmul(matmul_left, matmul_right)
        return np.array(
            (
                broadcasted[0, 0, 0],
                broadcasted[0, 3, 2],
                broadcasted[1, 1, 1],
                product[0, 0, 0],
                product[1, 2, 1],
                product[3, 3, 0],
            ),
            dtype=np.float32,
        )

    @classmethod
    def _run(cls, architecture: object) -> np.ndarray:
        vd.init(arch=architecture)  # type: ignore[arg-type]
        output = vd.storage.zeros(dtype=vd.f32, shape=(6,))
        broadcast_left, broadcast_right, matmul_left, matmul_right = cls._inputs()
        numpy_tensor_semantics(
            output,
            broadcast_left,
            broadcast_right,
            matmul_left,
            matmul_right,
            grid=(1, 1, 1),
        )
        return output.to_numpy()

    def test_cpu_values_match_numpy_across_composite_boundary(self) -> None:
        values = np.linspace(-1.0, 2.0, 18, dtype=np.float32)
        broadcast_left = values[:4].reshape(2, 1, 2)
        broadcast_right = values[4:6].reshape(1, 2, 1)
        matmul_left = values[6:14].reshape(2, 2, 2)
        matmul_right = values[14:18].reshape(1, 2, 2)
        broadcasted = broadcast_left + broadcast_right
        product = np.matmul(matmul_left, matmul_right)
        expected = np.array(
            (
                broadcasted[0, 0, 0],
                broadcasted[0, 1, 1],
                broadcasted[1, 1, 0],
                product[0, 0, 0],
                product[1, 0, 1],
                product[1, 1, 0],
            ),
            dtype=np.float32,
        )
        vd.init(arch=vd.cpu)
        output = vd.storage.zeros(dtype=vd.f32, shape=(6,))
        cpu_numpy_tensor_semantics(
            output,
            vd.storage.from_numpy(values),
            grid=(1, 1, 1),
        )
        np.testing.assert_allclose(output.to_numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_cpu_matmul_rank_categories_match_numpy(self) -> None:
        values = np.linspace(-1.0, 2.0, 12, dtype=np.float32)
        left_vector = values[:2]
        right_vector = values[2:4]
        left_matrix = values[4:8].reshape(2, 2)
        right_matrix = values[8:12].reshape(2, 2)
        expected = np.concatenate(
            (
                np.atleast_1d(np.matmul(left_vector, right_vector)),
                np.matmul(left_vector, right_matrix),
                np.matmul(left_matrix, right_vector),
                np.matmul(left_matrix, right_matrix).reshape(-1),
            )
        )
        vd.init(arch=vd.cpu)
        output = vd.storage.zeros(dtype=vd.f32, shape=(9,))
        cpu_matmul_rank_categories(
            output,
            vd.storage.from_numpy(values),
            grid=(1, 1, 1),
        )
        np.testing.assert_allclose(output.to_numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_available_graphics_compute_backends_match_numpy(self) -> None:
        expected = self._expected()
        for architecture in (vd.opengl, vd.vulkan, vd.directx):
            with self.subTest(backend=architecture.name):
                try:
                    actual = self._run(architecture)
                except RuntimeError as error:
                    if any(word in str(error).lower() for word in ("unavailable", "not available", "unsupported")):
                        continue
                    raise
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_available_backends_match_numpy_matmul_rank_categories(self) -> None:
        values = np.linspace(-1.0, 2.0, 12, dtype=np.float32)
        expected = np.concatenate(
            (
                np.atleast_1d(np.matmul(values[:2], values[2:4])),
                np.matmul(values[:2], values[8:12].reshape(2, 2)),
                np.matmul(values[4:8].reshape(2, 2), values[2:4]),
                np.matmul(values[4:8].reshape(2, 2), values[8:12].reshape(2, 2)).reshape(-1),
            )
        )
        for architecture in (vd.cuda, vd.opengl, vd.vulkan, vd.directx):
            with self.subTest(backend=architecture.name):
                try:
                    vd.init(arch=architecture)
                    output = vd.storage.zeros(dtype=vd.f32, shape=(9,))
                    cpu_matmul_rank_categories(
                        output,
                        vd.storage.from_numpy(values),
                        grid=(1, 1, 1),
                    )
                except RuntimeError as error:
                    if any(word in str(error).lower() for word in ("unavailable", "not available", "unsupported")):
                        continue
                    raise
                np.testing.assert_allclose(output.to_numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_rank_three_add_subtract_multiply_divide_on_all_compute_backends(self) -> None:
        left = np.linspace(-3.0, 4.0, 12, dtype=np.float32)
        right = np.linspace(0.5, 2.0, 12, dtype=np.float32)
        projection = np.array(((1.0, 0.25), (-0.5, 1.0), (0.75, -0.25)), dtype=np.float32).reshape(1, 3, 2)
        values = vd.storage.from_numpy(np.concatenate((left, right, projection.reshape(-1))))
        left_tensor = left.reshape(2, 2, 3)
        right_tensor = right.reshape(2, 2, 3)
        product = np.matmul(left_tensor, projection)
        expected = product[(0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0)]
        completed: set[str] = set()
        for architecture in (vd.cpu, vd.cuda, vd.vulkan, vd.opengl, vd.opengles, vd.directx):
            with self.subTest(backend=architecture.name):
                try:
                    vd.init(arch=architecture)
                    output = vd.storage.zeros(dtype=vd.f32, shape=(4,))
                    if architecture in (vd.cpu, vd.cuda):
                        rank_three_tensor_arithmetic(output, values, grid=(1, 1, 1))
                    else:
                        static_rank_three_tensor_arithmetic(
                            output,
                            left_tensor,
                            right_tensor,
                            projection,
                            grid=(1, 1, 1),
                        )
                except RuntimeError as error:
                    if any(word in str(error).lower() for word in ("unavailable", "not available", "unsupported")):
                        continue
                    raise
                np.testing.assert_allclose(output.to_numpy(), expected, rtol=1e-5, atol=1e-5)
                completed.add(architecture.name)
        self.assertIn(vd.cpu.name, completed)


if __name__ == "__main__":
    unittest.main()
