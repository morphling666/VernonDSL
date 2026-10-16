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


if __name__ == "__main__":
    unittest.main()
