"""Integer math builtins, float floor division, and bool-Tensor masks agree with numpy on every backend, in compute
kernels and in fragment shaders, which take separate SPIR-V translation paths."""

from __future__ import annotations

import unittest
from typing import Annotated, Any

import numpy as np
import vernon_dsl as vd
from backend_test_matrix import (
    BackendRequirements,
    BackendRow,
    backend_matrix_test,
    backend_row,
    expand_backend_matrix_tests,
    probe_backend,
    require_available,
)
from pipeline_shader import triangle_vertex


@vd.kernel(workgroup_size=(4, 1, 1))
def signed_builtins(
    output: vd.TensorView[vd.i32, (vd.dyn, 8), vd.write],
    values: vd.TensorView[vd.i32, (vd.dyn, 3), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    a = values[x, 0]
    b = values[x, 1]
    c = values[x, 2]
    left = vd.Vector([a, b, c])
    right = vd.Vector([c, a, b])
    output[x, 0] = vd.min(a, b)
    output[x, 1] = vd.max(a, b)
    output[x, 2] = vd.clamp(a, vd.min(b, c), vd.max(b, c))
    output[x, 3] = vd.abs(a)
    output[x, 4] = vd.dot(left, right)
    output[x, 5] = vd.cross(left, right)[1]
    output[x, 6] = vd.matmul(vd.Matrix([[a, b], [c, a]]), vd.Vector([b, c]))[1]
    output[x, 7] = vd.select(left > right, left, right)[2] + vd.max(left, right)[0]


@vd.kernel(workgroup_size=(4, 1, 1))
def unsigned_builtins(
    output: vd.TensorView[vd.u32, (vd.dyn, 4), vd.write],
    values: vd.TensorView[vd.u32, (vd.dyn, 2), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    a = values[x, 0]
    b = values[x, 1]
    output[x, 0] = vd.min(a, b)
    output[x, 1] = vd.max(a, b)
    output[x, 2] = vd.clamp(a, vd.u32(2), b)
    output[x, 3] = vd.abs(a)


@vd.kernel(workgroup_size=(4, 1, 1))
def float_floor_division(
    quotients: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    remainders: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    dividends: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    divisors: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    quotients[x] = dividends[x] // divisors[x]
    remainders[x] = dividends[x] % divisors[x]


@vd.kernel(workgroup_size=(4, 1, 1))
def vector_masks(
    output: vd.TensorView[vd.f32, (vd.dyn, 6), vd.write],
    values: vd.TensorView[vd.f32, (vd.dyn, 6), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    left = vd.Vector([values[x, 0], values[x, 1], values[x, 2]])
    right = vd.Vector([values[x, 3], values[x, 4], values[x, 5]])
    less = left < right
    picked = vd.select(less, left, right)
    output[x, 0] = picked[0]
    output[x, 1] = picked[1]
    output[x, 2] = picked[2]
    output[x, 3] = 1.0 if vd.any(less) else 0.0
    output[x, 4] = 1.0 if vd.all(left == right) else 0.0
    output[x, 5] = vd.select(left[0] < 0.0, right, left)[0]


@vd.fragment
def integer_math_fragment(values: vd.TensorView[vd.i32, (vd.dyn, 4), vd.read]) -> vd.Vector[vd.f32, 4]:
    row = vd.u32(vd.fragment_coord().x)
    a = values[row, 0]
    b = values[row, 1]
    c = values[row, 2]
    d = values[row, 3]
    quotient = 0
    remainder = 0
    for step in range(2):
        quotient += a // (b + step)
        remainder += a % (b + step)
    extremes = vd.clamp(a, vd.min(b, c), vd.max(b, c)) + vd.abs(d)
    left = vd.Vector([a, b, c])
    right = vd.Vector([d, c, b])
    mixed = vd.dot(left, right) + vd.cross(left, right)[2] + vd.matmul(vd.Matrix([[a, b], [c, d]]), right[:2])[1]
    return vd.Vector([vd.f32(quotient), vd.f32(remainder), vd.f32(extremes), vd.f32(mixed)])


@vd.fragment
def float_mask_fragment(values: vd.TensorView[vd.f32, (vd.dyn, 4), vd.read]) -> vd.Vector[vd.f32, 4]:
    row = vd.u32(vd.fragment_coord().x)
    a = values[row, 0]
    b = values[row, 1]
    left = vd.Vector([a, b])
    right = vd.Vector([values[row, 2], values[row, 3]])
    picked = vd.select(left < right, left, right)
    flags = (1.0 if vd.any(left < right) else 0.0) + (2.0 if vd.all(left == right) else 0.0)
    return vd.Vector([a // b, picked[0], picked[1], flags])


def _signed_rows() -> np.ndarray:
    edges = np.array(((7, 2, -3), (-7, 2, 3), (7, -2, 5), (-7, -2, -5), (0, -1, 1), (-1000, 999, 3)), dtype=np.int32)
    generator = np.random.default_rng(20261106)
    rows = generator.integers(-1000, 1000, size=(26, 3), dtype=np.int32)
    return np.concatenate((edges, rows))


def _signed_expected(rows: np.ndarray) -> np.ndarray:
    a, b, c = (rows[:, axis].astype(np.int64) for axis in range(3))
    left = np.stack((a, b, c), axis=1)
    right = np.stack((c, a, b), axis=1)
    columns = (
        np.minimum(a, b),
        np.maximum(a, b),
        np.clip(a, np.minimum(b, c), np.maximum(b, c)),
        np.abs(a),
        (left * right).sum(axis=1),
        np.cross(left, right)[:, 1],
        c * b + a * c,
        np.where(left > right, left, right)[:, 2] + np.maximum(left, right)[:, 0],
    )
    return np.stack(columns, axis=1).astype(np.int32)


def _floor_division_operands() -> tuple[np.ndarray, np.ndarray]:
    # Shading-language fmod is approximate, so inexact divisors pair with small dividends; the floor, sign, and
    # zero-divisor and infinite-operand rules are what must agree on every backend.
    pairs = (
        (7.5, 2.0),
        (-7.5, 2.0),
        (7.5, -2.0),
        (-7.5, -2.0),
        (6.0, 2.0),
        (-6.0, 2.0),
        (0.25, 2.0),
        (-0.25, 2.0),
        (0.0, -2.0),
        (-0.0, 2.0),
        (7.5, 3.0),
        (-7.5, 3.0),
        (7.5, -5.0),
        (1.0e6, 0.5),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 0.0),
        (float("nan"), 1.0),
        (-3.0, 4.0),
        (3.0, -4.0),
        (3.0, float("inf")),
        (-3.0, float("inf")),
        (3.0, float("-inf")),
        (float("inf"), 2.0),
    )
    dividends, divisors = (np.array(axis, dtype=np.float32) for axis in zip(*pairs, strict=True))
    return dividends, divisors


def _mask_rows() -> np.ndarray:
    return np.array(
        (
            (1.0, 5.0, 3.0, 2.0, 4.0, 3.0),
            (-1.0, 2.0, 3.0, -1.0, 2.0, 3.0),
            (4.0, 4.0, 4.0, 1.0, 1.0, 1.0),
            (-2.5, 0.5, 8.0, 9.0, -0.5, 7.0),
        ),
        dtype=np.float32,
    )


class BuiltinMathKernelTests(unittest.TestCase):
    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_signed_builtins_match_numpy(self, backend: BackendRow) -> None:
        rows = _signed_rows()
        output = vd.storage.zeros(dtype=vd.i32, shape=(rows.shape[0], 8))
        signed_builtins(output, vd.storage.from_numpy(rows), grid=(rows.shape[0] // 4, 1, 1))
        np.testing.assert_array_equal(output.to_numpy(), _signed_expected(rows))

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_unsigned_builtins_compare_without_sign(self, backend: BackendRow) -> None:
        rows = np.array(((0xFFFFFFF0, 1), (1, 0xFFFFFFF0), (0x80000000, 0x7FFFFFFF), (5, 9)), dtype=np.uint32)
        output = vd.storage.zeros(dtype=vd.u32, shape=(4, 4))
        unsigned_builtins(output, vd.storage.from_numpy(rows), grid=(1, 1, 1))
        a, b = rows[:, 0], rows[:, 1]
        expected = np.stack((np.minimum(a, b), np.maximum(a, b), np.minimum(np.maximum(a, 2), b), a), axis=1)
        np.testing.assert_array_equal(output.to_numpy(), expected)

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_float_floor_division_matches_numpy(self, backend: BackendRow) -> None:
        dividends, divisors = _floor_division_operands()
        quotients = vd.storage.zeros(dtype=vd.f32, shape=(dividends.size,))
        remainders = vd.storage.zeros(dtype=vd.f32, shape=(dividends.size,))
        float_floor_division(
            quotients,
            remainders,
            vd.storage.from_numpy(dividends),
            vd.storage.from_numpy(divisors),
            grid=(dividends.size // 4, 1, 1),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = (np.floor_divide(dividends, divisors), np.remainder(dividends, divisors))
        for actual, wanted in zip((quotients.to_numpy(), remainders.to_numpy()), expected, strict=True):
            np.testing.assert_array_equal(actual, wanted)
            numbers = ~np.isnan(wanted)
            np.testing.assert_array_equal(np.signbit(actual[numbers]), np.signbit(wanted[numbers]))

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_vector_comparisons_select_elementwise(self, backend: BackendRow) -> None:
        rows = _mask_rows()
        output = vd.storage.zeros(dtype=vd.f32, shape=(rows.shape[0], 6))
        vector_masks(output, vd.storage.from_numpy(rows), grid=(1, 1, 1))
        left, right = rows[:, :3], rows[:, 3:]
        expected = np.concatenate(
            (
                np.where(left < right, left, right),
                (left < right).any(axis=1, keepdims=True),
                (left == right).all(axis=1, keepdims=True),
                np.where(left[:, :1] < 0.0, right[:, :1], left[:, :1]),
            ),
            axis=1,
        ).astype(np.float32)
        np.testing.assert_array_equal(output.to_numpy(), expected)


expand_backend_matrix_tests(BuiltinMathKernelTests)


_FULL_SCREEN = np.array(((-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)), dtype=np.float32)


def _draw_columns(fragment: Any, rows: np.ndarray) -> np.ndarray:
    """One pixel column per input row; returns the fragment's four outputs for each row."""

    target = vd.Texture.zeros(shape=(2, rows.shape[0]), format=vd.rgba32_float)
    vd.pipeline(triangle_vertex, fragment)(
        position=vd.storage.from_numpy(_FULL_SCREEN),
        values=vd.storage.from_numpy(rows),
        render_pass=vd.render_pass(
            vd.RenderTarget.from_attachments(colors={0: target}), color=vd.clear((0.0, 0.0, 0.0, 0.0))
        ),
    )
    return target.to_numpy()[0]


class _BuiltinMathFragmentCases:
    arch: Any
    requirements: BackendRequirements

    def setUp(self) -> None:
        require_available(probe_backend(backend_row(self.arch), self.requirements))

    def test_integer_math_in_a_fragment_loop_matches_numpy(self) -> None:
        rows = np.array(
            ((7, 2, -3, 4), (-7, 2, 3, -4), (7, -3, 5, 1), (-7, -3, -5, -1), (0, 4, 1, 0), (-100, 7, 9, 3)),
            dtype=np.int32,
        )
        a, b, c, d = (rows[:, axis].astype(np.int64) for axis in range(4))
        left = np.stack((a, b, c), axis=1)
        right = np.stack((d, c, b), axis=1)
        expected = np.stack(
            (
                a // b + a // (b + 1),
                a % b + a % (b + 1),
                np.clip(a, np.minimum(b, c), np.maximum(b, c)) + np.abs(d),
                (left * right).sum(axis=1) + np.cross(left, right)[:, 2] + c * d + d * c,
            ),
            axis=1,
        ).astype(np.float32)
        np.testing.assert_array_equal(_draw_columns(integer_math_fragment, rows), expected)

    def test_float_floor_division_and_masks_in_a_fragment_match_numpy(self) -> None:
        rows = np.array(
            (
                (7.5, 2.0, 1.0, 9.0),
                (-7.5, 2.0, -8.0, 2.0),
                (7.5, -2.0, 7.5, -2.0),
                (-0.25, 2.0, 0.0, 3.0),
                (7.5, 3.0, 8.0, 1.0),
                (-7.5, 3.0, -7.5, 3.0),
                (-3.0, np.inf, 1.0, 2.0),
                (3.0, -np.inf, 3.0, -np.inf),
            ),
            dtype=np.float32,
        )
        left, right = rows[:, :2], rows[:, 2:]
        picked = np.where(left < right, left, right)
        flags = (left < right).any(axis=1) + 2.0 * (left == right).all(axis=1)
        expected = np.stack(
            (np.floor_divide(rows[:, 0], rows[:, 1]), picked[:, 0], picked[:, 1], flags), axis=1
        ).astype(np.float32)
        np.testing.assert_array_equal(_draw_columns(float_mask_fragment, rows), expected)


class OpenGLBuiltinMathFragmentTests(_BuiltinMathFragmentCases, unittest.TestCase):
    arch = vd.opengl
    requirements = BackendRequirements(graphics=True, minimum_api_version=(4, 3))


class OpenGLESBuiltinMathFragmentTests(_BuiltinMathFragmentCases, unittest.TestCase):
    arch = vd.opengles
    requirements = BackendRequirements(graphics=True, minimum_api_version=(3, 1))


class VulkanBuiltinMathFragmentTests(_BuiltinMathFragmentCases, unittest.TestCase):
    arch = vd.vulkan
    requirements = BackendRequirements(graphics=True)


class DirectXBuiltinMathFragmentTests(_BuiltinMathFragmentCases, unittest.TestCase):
    arch = vd.directx
    requirements = BackendRequirements(graphics=True)


class MetalBuiltinMathFragmentTests(_BuiltinMathFragmentCases, unittest.TestCase):
    arch = vd.metal
    requirements = BackendRequirements(graphics=True)


if __name__ == "__main__":
    unittest.main()
