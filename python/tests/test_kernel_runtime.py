from __future__ import annotations

import unittest
from typing import Annotated

import numpy as np

import vernon_dsl as vd
import fractal


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

    def test_grid_is_required_and_validated(self) -> None:
        output = vd.Tensor.zeros(dtype=vd.f32, shape=(2, 3))
        with self.assertRaises(TypeError):
            fill(output, 1.0)  # type: ignore[call-arg]
        with self.assertRaises(ValueError):
            fill(output, 1.0, grid=(3, 0, 1))

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
