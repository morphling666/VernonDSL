from __future__ import annotations

import unittest
from importlib import import_module
from typing import Annotated

import numpy as np
import vernon_dsl as vd
from backend_test_matrix import (
    BackendRequirements,
    BackendRow,
    backend_matrix_test,
    expand_backend_matrix_tests,
)
from vernon_dsl.diagnostics import ProgramCompileError


@vd.struct(shared=True)
class Inner:
    bias: vd.Vector[vd.f32, 2]
    count: vd.i32


@vd.struct(shared=True)
class Params:
    tint: vd.Vector[vd.f32, 4]
    gain: vd.f32
    inner: Inner
    basis: vd.Tensor[vd.f32, (2, 2)]


@vd.struct(shared=True)
class Flags:
    enabled: vd.bool
    flip: vd.bool
    scale: vd.f32
    last: vd.bool


@vd.struct(shared=True)
class Wide:
    value: vd.f64
    gain: vd.f32


@vd.kernel(workgroup_size=(1, 1, 1))
def struct_value(
    params: Params,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = params.tint[x] * params.gain + params.inner.bias[1] + vd.f32(params.inner.count) + params.basis[1, 0]


@vd.kernel(workgroup_size=(1, 1, 1))
def tuple_value(
    pair: vd.Tuple[vd.f32, vd.Vector[vd.f32, 2]],
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = pair[0] + pair[1][1] * vd.f32(x)


@vd.kernel(workgroup_size=(1, 1, 1))
def flags_value(
    flags: Flags,
    table: vd.Tensor[Flags, (2,)],
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    value = flags.scale * vd.f32(x)
    if flags.flip:
        value = -value
    if not flags.enabled:
        value = 0.0
    if flags.last:
        value = value + 100.0
    entry = table[x % 2]
    if entry.flip:
        value = value + entry.scale
    if entry.last:
        value = value + 1000.0
    output[x] = value


@vd.kernel(workgroup_size=(1, 1, 1))
def runtime_indexed_tensors(
    lanes: vd.Tensor[vd.f32, (3,)],
    table: vd.Tensor[vd.f32, (2, 4)],
    wide: vd.Tensor[vd.f32, (20,)],
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = lanes[x % 3] + table[x % 2, x % 4] * 10.0 + wide[19 - x] * 100.0


@vd.kernel(workgroup_size=(1, 1, 1))
def wide_value(
    wide: Wide,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = vd.f32(wide.value) * wide.gain


@vd.kernel(workgroup_size=(1, 1, 1))
def updated_values(
    lanes: vd.Tensor[vd.f32, (4,)],
    table: vd.Tensor[vd.f32, (2, 3)],
    flags: Flags,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    moved = lanes.at[x].set(10.0).at[-1].add(100.0)
    grid = table.at[x % 2, -1].add(1000.0)
    changed = vd.replace(flags, scale=flags.scale * 2.0, flip=x == 1)
    value = moved[x] + moved[3] + grid[x % 2, 2] + lanes[x]
    if changed.flip:
        value = value + changed.scale
    output[x] = value


@vd.func(shared=True)
def slice_mix(table: vd.Tensor[vd.f32, (3, 4)], i: vd.i32) -> vd.f32:
    row = table[1]
    corner = table[0:2, 2:]
    window = table[2, i : i + 2]
    flipped = row[::-1]
    patched = table.at[1:3, i : i + 2].set(corner * 10.0)
    filled = table.at[:, -1].add(100.0)
    stepped = table.at[0, ::2].set(-1.0)
    return (
        row[i % 4]
        + corner[1, 0]
        + window[1] * 10.0
        + flipped[0] * 100.0
        + patched[2, i % 4] * 1000.0
        + patched[1, 0]
        + filled[2, 3]
        + stepped[0, 2]
    )


@vd.kernel(workgroup_size=(1, 1, 1))
def sliced_values(
    table: vd.Tensor[vd.f32, (3, 4)],
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = slice_mix(table, vd.i32(x) - 2)


@vd.kernel(workgroup_size=(4, 1, 1))
def sliced_objective(
    scale: vd.f32,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    i = vd.i32(x)
    lanes = vd.Vector([scale, scale * 2.0, scale * 3.0, scale * 4.0])
    block = lanes[i : i + 2] * scale
    moved = lanes.at[1:3].set(block)
    output[x] = moved[i] + block[1]


sliced_objective_vjp = vd.ad.vjp(sliced_objective, wrt=("scale",), outputs=("output",))


@vd.kernel(workgroup_size=(4, 1, 1))
def updated_objective(
    scale: vd.f32,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    lanes = vd.Vector([scale, scale * 2.0, scale * 3.0, scale * 4.0])
    changed = lanes.at[x].set(scale * scale).at[0].add(scale)
    output[x] = changed[x] + changed[3]


updated_objective_vjp = vd.ad.vjp(updated_objective, wrt=("scale",), outputs=("output",))


def _output() -> vd.TensorStorage:
    return vd.storage.zeros(dtype=vd.f32, shape=(4,))


@expand_backend_matrix_tests
class KernelAggregateValueTests(unittest.TestCase):
    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_nested_struct_value(self, backend: BackendRow) -> None:
        vd.init(arch=backend.architecture)  # type: ignore[arg-type]
        output = _output()
        params = Params(
            np.array((1.0, 2.0, 3.0, 4.0), dtype=np.float32),
            vd.f32(2.0),
            Inner(np.array((0.5, 10.0), dtype=np.float32), vd.i32(100)),
            np.array(((0.0, 0.0), (1000.0, 0.0)), dtype=np.float32),
        )
        struct_value(params, output, grid=(4, 1, 1))
        np.testing.assert_allclose(output.to_numpy(), [1112.0, 1114.0, 1116.0, 1118.0])

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_tuple_value(self, backend: BackendRow) -> None:
        vd.init(arch=backend.architecture)  # type: ignore[arg-type]
        output = _output()
        tuple_value((vd.f32(1.0), np.array((0.0, 2.0), dtype=np.float32)), output, grid=(4, 1, 1))
        np.testing.assert_allclose(output.to_numpy(), [1.0, 3.0, 5.0, 7.0])

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_unaligned_bool_fields_in_struct_and_struct_tensor(self, backend: BackendRow) -> None:
        vd.init(arch=backend.architecture)  # type: ignore[arg-type]
        output = _output()
        table = vd.storage.from_values(
            (Flags(False, True, vd.f32(5.0), False), Flags(True, False, vd.f32(7.0), True)), dtype=Flags
        )
        flags_value(Flags(True, True, vd.f32(3.0), True), table, output, grid=(4, 1, 1))
        np.testing.assert_allclose(output.to_numpy(), [105.0, 1097.0, 99.0, 1091.0])

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_runtime_indexed_register_tensors(self, backend: BackendRow) -> None:
        vd.init(arch=backend.architecture)  # type: ignore[arg-type]
        lanes = np.array((1.0, 2.0, 3.0), dtype=np.float32)
        table = np.arange(8, dtype=np.float32).reshape(2, 4)
        wide = np.arange(20, dtype=np.float32)
        output = vd.storage.zeros(dtype=vd.f32, shape=(20,))
        runtime_indexed_tensors(lanes, table, wide, output, grid=(20, 1, 1))
        x = np.arange(20)
        expected = lanes[x % 3] + table[x % 2, x % 4] * 10.0 + wide[19 - x] * 100.0
        np.testing.assert_allclose(output.to_numpy(), expected)

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_values_update_into_new_values(self, backend: BackendRow) -> None:
        vd.init(arch=backend.architecture)  # type: ignore[arg-type]
        lanes = np.array((1.0, 2.0, 3.0, 4.0), dtype=np.float32)
        table = np.arange(6, dtype=np.float32).reshape(2, 3)
        output = _output()
        updated_values(lanes, table, Flags(True, False, vd.f32(0.5), False), output, grid=(4, 1, 1))
        # The update builds new values: lanes itself still reads its old elements.
        np.testing.assert_allclose(output.to_numpy(), [1117.0, 1122.0, 1119.0, 1229.0])

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_slices_select_and_update_blocks(self, backend: BackendRow) -> None:
        vd.init(arch=backend.architecture)  # type: ignore[arg-type]
        table = np.arange(12, dtype=np.float32).reshape(3, 4)
        output = vd.storage.zeros(dtype=vd.f32, shape=(6,))
        sliced_values(table, output, grid=(6, 1, 1))
        # Python mode runs the same helper, so every start from -2 to 3 (wrapped, clamped) must agree with it.
        expected = [slice_mix(vd.Tensor(table), np.int32(i)) for i in range(-2, 4)]
        np.testing.assert_allclose(output.to_numpy(), np.array(expected, dtype=np.float32))

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True, f64=True))
    def test_f64_struct_field(self, backend: BackendRow) -> None:
        vd.init(arch=backend.architecture)  # type: ignore[arg-type]
        output = _output()
        wide_value(Wide(vd.f64(2.5), vd.f32(2.0)), output, grid=(1, 1, 1))
        np.testing.assert_allclose(output.to_numpy()[0], 5.0)

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_f64_struct_field_is_a_capability_error_without_f64(self, backend: BackendRow) -> None:
        native = import_module("vernon_dsl._native")
        if dict(native.target_capabilities(getattr(native.Target, backend.compiler_target)))["f64"]:
            self.skipTest(f"{backend.name} supports f64")
        vd.init(arch=backend.architecture)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ProgramCompileError, "PROGRAM_SPIRV_F64_UNSUPPORTED"):
            wide_value(Wide(vd.f64(2.5), vd.f32(2.0)), _output(), grid=(1, 1, 1))


class ValueUpdateGradientTests(unittest.TestCase):
    def test_updates_differentiate_through_the_selected_element(self) -> None:
        vd.init(arch=vd.cpu)
        output = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        _, pullback = updated_objective_vjp(np.float32(2.0), output, grid=(1, 1, 1))
        # s^2 + 5s, then s^2 + 4s twice, then 2s^2.
        np.testing.assert_allclose(output.to_numpy(), [14.0, 12.0, 12.0, 8.0])
        cotangent = np.zeros((1, 1, 4, 4), dtype=np.float32)
        cotangent[0, 0, np.arange(4), np.arange(4)] = 1.0
        # 2s + 5, 2s + 4, 2s + 4 and 4s at s = 2.
        np.testing.assert_allclose(pullback(cotangent)["scale"], np.array(33.0, dtype=np.float32))

    def test_slices_differentiate_through_the_selected_block(self) -> None:
        vd.init(arch=vd.cpu)
        output = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        _, pullback = sliced_objective_vjp(np.float32(2.0), output, grid=(1, 1, 1))
        # s + 2s^2, 5s^2, 8s^2, and 4s + 4s^2 where the window clamps to start 2.
        np.testing.assert_allclose(output.to_numpy(), [10.0, 20.0, 32.0, 24.0])
        cotangent = np.zeros((1, 1, 4, 4), dtype=np.float32)
        cotangent[0, 0, np.arange(4), np.arange(4)] = 1.0
        # 1 + 4s, 10s, 16s and 4 + 8s at s = 2.
        np.testing.assert_allclose(pullback(cotangent)["scale"], np.array(81.0, dtype=np.float32))


class ValueWriteDiagnosticsTests(unittest.TestCase):
    def _compile_error(self, body: str) -> str:
        source = (
            "import vernon_dsl as vd\n\n"
            "@vd.kernel(workgroup_size=(1, 1, 1))\n"
            "def k(lanes: vd.Tensor[vd.f32, (4,)], output: vd.TensorView[vd.f32, (vd.dyn,), vd.write]) -> None:\n"
            f"{body}"
            "    output[0] = lanes[0]\n"
        )
        with self.assertRaises(Exception) as raised:
            vd.compile_source(source, "value_writes.py")
        return str(raised.exception)

    def test_tensor_values_are_never_written_into(self) -> None:
        self.assertIn("lanes = lanes.at[1].set(x)", self._compile_error("    lanes[1] = 2.0\n"))
        self.assertIn("never written into", self._compile_error("    lanes[1] += 2.0\n"))

    def test_constant_update_indices_are_checked(self) -> None:
        self.assertIn("outside an axis of extent 4", self._compile_error("    lanes = lanes.at[4].set(1.0)\n"))
        self.assertIn("at most one per axis", self._compile_error("    lanes = lanes.at[0, 1].set(1.0)\n"))

    def test_views_are_not_sliced(self) -> None:
        self.assertIn("slices select from a Tensor value", self._compile_error("    x = output[0:2]\n"))


if __name__ == "__main__":
    unittest.main()
