from __future__ import annotations

import unittest

import numpy as np
from vernon_dsl import CompileError, compile_source
from vernon_dsl.frontend.tensor_shapes import broadcast_shape, matmul_shape


class TensorShapeSemanticsTests(unittest.TestCase):
    def test_broadcast_shape_matches_numpy(self) -> None:
        cases = (
            ((), ()),
            ((3,), (1,)),
            ((2, 1, 4), (3, 4)),
            ((5, 1, 3, 1), (1, 4, 1)),
            ((2, 3), (3, 2)),
        )
        for left, right in cases:
            with self.subTest(left=left, right=right):
                try:
                    expected = np.broadcast_shapes(left, right)
                except ValueError:
                    expected = None
                self.assertEqual(broadcast_shape(left, right), expected)

    def test_matmul_shape_matches_numpy(self) -> None:
        valid = (
            ((4,), (4,)),
            ((4,), (4, 3)),
            ((2, 4), (4,)),
            ((2, 4), (4, 3)),
            ((5, 2, 4), (4, 3)),
            ((1, 2, 4), (7, 4, 3)),
            ((6, 1, 2, 4), (5, 4, 3)),
            ((4, 4, 4), (4, 4, 2)),
        )
        for left, right in valid:
            with self.subTest(left=left, right=right):
                expected = np.matmul(np.zeros(left), np.zeros(right)).shape
                self.assertEqual(matmul_shape(left, right), expected)

    def test_matmul_rejects_numpy_incompatible_shapes(self) -> None:
        invalid = (
            ((), (2, 2)),
            ((2, 3), (4, 2)),
            ((2, 3, 4), (5, 4, 2)),
            ((4, 4, 4), (4, 2, 2)),
        )
        for left, right in invalid:
            with self.subTest(left=left, right=right):
                with self.assertRaises(ValueError):
                    np.matmul(np.empty(left), np.empty(right))
                self.assertIsNone(matmul_shape(left, right))

    def test_frontend_normalizes_broadcast_and_batched_matmul(self) -> None:
        source = """
from vernon_dsl import *

@func
def broadcast_values(
    left: Tensor[f32, (2, 1, 4)],
    right: Tensor[f32, (3, 4)],
) -> Tensor[f32, (2, 3, 4)]:
    return left + right

@func
def batched_matmul(
    left: Tensor[f32, (1, 2, 4)],
    right: Tensor[f32, (7, 4, 3)],
) -> Tensor[f32, (7, 2, 3)]:
    return matmul(left, right)
"""
        output = compile_source(source, "tensor_shapes.py")
        self.assertIn('name = "broadcast"', output)
        self.assertIn('name = "matmul"', output)
        self.assertIn("tensor<2x3x4xf32>", output)
        self.assertIn("tensor<7x2x3xf32>", output)

    def test_frontend_rejects_invalid_matmul(self) -> None:
        source = """
from vernon_dsl import *

@func
def invalid(
    left: Tensor[f32, (4, 4, 4)],
    right: Tensor[f32, (4, 2, 2)],
):
    return matmul(left, right)

@kernel(workgroup_size=(1, 1, 1))
def invalid_entry(
    output: TensorView[f32, 1, write],
    left: Tensor[f32, (4, 4, 4)],
    right: Tensor[f32, (4, 2, 2)],
) -> None:
    output[0] = invalid(left, right)[0, 0, 0]
"""
        with self.assertRaisesRegex(CompileError, "incompatible core or batch dimensions"):
            compile_source(source, "invalid_matmul.py")


if __name__ == "__main__":
    unittest.main()
