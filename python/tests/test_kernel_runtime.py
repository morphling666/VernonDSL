from __future__ import annotations

import unittest
from typing import Annotated

import numpy as np

import vernon_dsl as vd
import fractal


@vd.kernel(workgroup_size=(4, 2, 1))
def tensor_operators(
    output: vd.Tensor[vd.f32, (None, None, None)],
    left: vd.Tensor[vd.f32, (None, None, None)],
    right: vd.Tensor[vd.f32, (None, None, None)],
    scale: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3, )],
                   vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    y = gid[1]
    z = gid[2]
    output[z, y, x] = (
        (left[z, y, x] + right[z, y, x]) * scale - right[z, y, x]) / scale


@vd.kernel(workgroup_size=(8, 1, 1))
def vector_while(
    output: vd.Tensor[vd.f32, (None, )],
    phase: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3, )],
                   vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    c = vd.vec2(-0.8, vd.cos(phase) * 0.2)
    z = vd.vec2(vd.f32(x) * 0.01, 0.1)
    iterations = 0
    while vd.norm(z) < 20.0 and iterations < 8:
        z = vd.vec2(
            z[0] * z[0] - z[1] * z[1],
            z[1] * z[0] * 2.0,
        ) + c
        iterations += 1
    output[x] = vd.f32(iterations)


@vd.kernel(workgroup_size=(2, 1, 1))
def matrix_vector(
    output: vd.Tensor[vd.f32, (None, )],
    gid: Annotated[vd.Tensor[vd.u32, (3, )],
                   vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    matrix = vd.mat2(1.0, 2.0, 3.0, 4.0)
    value = vd.matmul(matrix, vd.vec2(5.0, 6.0))
    output[x] = value[x]


class KernelTensorRuntimeTests(unittest.TestCase):

    @staticmethod
    def _run_tensor_operators(arch: object) -> np.ndarray:
        vd.init(arch=arch)  # type: ignore[arg-type]
        shape = (2, 3, 4)
        left_array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        right_array = np.linspace(0.25, 2.5, np.prod(shape),
                                  dtype=np.float32).reshape(shape)
        output = vd.Tensor.zeros(dtype=vd.f32, shape=shape)
        tensor_operators(
            output,
            vd.Tensor.from_numpy(left_array),
            vd.Tensor.from_numpy(right_array),
            2.0,
            grid=(shape[2], shape[1], shape[0]),
        )
        return output.to_numpy()

    @staticmethod
    def _run_vector_while(arch: object) -> np.ndarray:
        vd.init(arch=arch)  # type: ignore[arg-type]
        output = vd.Tensor.zeros(dtype=vd.f32, shape=(16, ))
        vector_while(output, 0.35, grid=(16, 1, 1))
        return output.to_numpy()

    @staticmethod
    def _cuda_available() -> bool:
        try:
            vd.init(arch=vd.cuda)
        except RuntimeError:
            vd.init(arch=vd.cpu)
            return False
        return True

    @staticmethod
    def _vulkan_available() -> bool:
        try:
            vd.init(arch=vd.vulkan)
        except RuntimeError:
            vd.init(arch=vd.cpu)
            return False
        return True

    @staticmethod
    def _runtime_available(arch: object) -> bool:
        try:
            vd.init(arch=arch)  # type: ignore[arg-type]
        except RuntimeError:
            vd.init(arch=vd.cpu)
            return False
        return True

    def test_rank_three_tensor_operators(self) -> None:
        actual = self._run_tensor_operators(vd.cpu)
        shape = (2, 3, 4)
        left = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        right = np.linspace(0.25, 2.5, np.prod(shape),
                            dtype=np.float32).reshape(shape)
        expected = ((left + right) * 2.0 - right) / 2.0
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

        backends = []
        if self._cuda_available():
            backends.append(vd.cuda)
        if self._vulkan_available():
            backends.append(vd.vulkan)
        for architecture in (vd.opengl, vd.opengles):
            if self._runtime_available(architecture):
                backends.append(architecture)
        for backend in backends:
            with self.subTest(backend=backend.name):
                backend_actual = self._run_tensor_operators(backend)
                np.testing.assert_allclose(backend_actual,
                                           expected,
                                           rtol=0.0,
                                           atol=1e-6)

    def test_loop_carried_vector_norm(self) -> None:
        expected = self._run_vector_while(vd.cpu)
        backends = []
        if self._cuda_available():
            backends.append(vd.cuda)
        if self._vulkan_available():
            backends.append(vd.vulkan)
        for backend in backends:
            with self.subTest(backend=backend.name):
                actual = self._run_vector_while(backend)
                np.testing.assert_allclose(actual,
                                           expected,
                                           rtol=0.0,
                                           atol=1e-6)

    def test_matrix_specialization(self) -> None:
        backends = []
        if self._cuda_available():
            backends.append(vd.cuda)
        if self._vulkan_available():
            backends.append(vd.vulkan)
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                output = vd.Tensor.zeros(dtype=vd.f32, shape=(2, ))
                matrix_vector(output, grid=(2, 1, 1))
                np.testing.assert_allclose(output.to_numpy(),
                                           np.array((17.0, 39.0),
                                                    dtype=np.float32),
                                           rtol=0.0,
                                           atol=1e-6)

    def test_cross_compiled_source_generation(self) -> None:
        output = vd.Tensor.zeros(dtype=vd.f32, shape=(16, ))
        cases = {
            "metal": "kernel void vector_while",
            "opengl": "#version 430",
            "opengles": "#version 310 es",
        }
        for target, marker in cases.items():
            with self.subTest(target=target):
                source, reflection = vector_while.compile_artifact(
                    output, 0.35, target=target)
                self.assertIn(marker, source.decode())
                self.assertIn(f'"target":"{target}"', reflection)


@vd.kernel(workgroup_size=(4, 2, 1))
def fill(
    output: vd.Tensor[vd.f32, (None, None)],
    scale: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3, )],
                   vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    y = gid[1]
    if x < 3 and y < 2:
        output[y, x] = vd.f32(x) + vd.f32(y) * scale


class TensorTests(unittest.TestCase):

    def test_numpy_copy_contract(self) -> None:
        source = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = vd.Tensor.from_numpy(source)
        source.fill(0)
        np.testing.assert_array_equal(
            tensor.to_numpy(),
            np.arange(6, dtype=np.float32).reshape(2, 3))
        result = tensor.to_numpy()
        result.fill(0)
        self.assertNotEqual(float(tensor.to_numpy()[1, 2]), 0.0)

    def test_copy_validates_layout(self) -> None:
        tensor = vd.Tensor.zeros(dtype=vd.f32, shape=(2, 3))
        with self.assertRaises(ValueError):
            tensor.copy_from_numpy(np.zeros((3, 2), dtype=np.float32))


class KernelTests(unittest.TestCase):

    def setUp(self) -> None:
        vd.init(arch=vd.cpu)
        fill.compile_count = 0
        fill._cache.clear()

    def test_explicit_grid_and_cache(self) -> None:
        output = vd.Tensor.zeros(dtype=vd.f32, shape=(2, 3))
        fill(output, 10.0, grid=(3, 2, 1))
        np.testing.assert_array_equal(
            output.to_numpy(),
            np.array([[0, 1, 2], [10, 11, 12]], dtype=np.float32),
        )
        self.assertEqual(fill.compile_count, 1)
        fill(output, 20.0, grid=(3, 2, 1))
        self.assertEqual(fill.compile_count, 1)

    def test_specialized_shape_invalidates_cache(self) -> None:
        first = vd.Tensor.zeros(dtype=vd.f32, shape=(2, 3))
        second = vd.Tensor.zeros(dtype=vd.f32, shape=(3, 3))
        fill(first, 1.0, grid=(3, 2, 1))
        fill(second, 1.0, grid=(3, 2, 1))
        self.assertEqual(fill.compile_count, 2)

    def test_grid_is_inferred_and_validated(self) -> None:
        output = vd.Tensor.zeros(dtype=vd.f32, shape=(2, 3))
        fill(output, 1.0)
        np.testing.assert_array_equal(
            output.to_numpy(),
            np.array([[0, 1, 2], [1, 2, 3]], dtype=np.float32),
        )
        with self.assertRaises(ValueError):
            fill(output, 1.0, grid=(3, 0, 1))

    def test_native_tensor_residency_and_lazy_download(self) -> None:
        if not KernelTensorRuntimeTests._runtime_available(vd.opengl):
            self.skipTest("OpenGL runtime unavailable")
        output = vd.Tensor.zeros(dtype=vd.f32, shape=(2, 3))
        fill(output, 2.0)
        fill(output, 3.0)
        self.assertEqual(output._allocation_count, 1)
        self.assertEqual(output._upload_count, 1)
        self.assertEqual(output._download_count, 0)
        output.to_numpy()
        self.assertEqual(output._download_count, 1)

    def test_fractal_matches_vectorized_numpy_reference(self) -> None:
        width, height = 4, 3
        time = 0.2
        output = vd.Tensor.zeros(dtype=vd.f32,
                                 shape=(fractal.HEIGHT, fractal.WIDTH))
        fractal.paint(output, time, grid=(width, height, 1))

        y, x = np.mgrid[:height, :width]
        z = np.stack(
            (
                (x.astype(np.float32) / fractal.HEIGHT - 1.0) * 2.0,
                (y.astype(np.float32) / fractal.HEIGHT - 0.5) * 2.0,
            ),
            axis=-1,
        )
        c = np.array((-0.8, np.cos(time) * 0.2), dtype=np.float32)
        iterations = np.zeros((height, width), dtype=np.int32)
        for _ in range(50):
            active = (np.linalg.norm(z, axis=-1) < 20.0) & (iterations < 50)
            squared = np.stack(
                (
                    z[..., 0] * z[..., 0] - z[..., 1] * z[..., 1],
                    z[..., 1] * z[..., 0] * 2.0,
                ),
                axis=-1,
            )
            z = np.where(active[..., None], squared + c, z)
            iterations += active
        expected = 1.0 - iterations.astype(np.float32) * 0.02
        np.testing.assert_allclose(output.to_numpy()[:height, :width],
                                   expected,
                                   rtol=1e-5,
                                   atol=1e-6)


if __name__ == "__main__":
    unittest.main()
