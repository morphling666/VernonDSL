from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType
from typing import Annotated
from unittest import mock

import numpy as np
import vernon_dsl as vd
from aggregate_vertex_shader import (
    AggregateVertexPayload,
    ComplexAggregateVertex,
    copy_complex_aggregate_tensor_view,
    inspect_multidimensional_aggregate_tensor_value,
)
from vernon_dsl.host_values import host_abi_layout


def _load_fractal() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "examples" / "fractal.py"
    spec = importlib.util.spec_from_file_location("vernon_test_fractal", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fractal = _load_fractal()


@vd.kernel(workgroup_size=(4, 2, 1))
def tensor_operators(
    output: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.write],
    left: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read],
    right: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read],
    scale: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    y = gid[1]
    z = gid[2]
    output[z, y, x] = ((left[z, y, x] + right[z, y, x]) * scale - right[z, y, x]) / scale


@vd.kernel(workgroup_size=(8, 1, 1))
def vector_while(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    phase: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    c = vd.Vector([-0.8, vd.cos(phase) * 0.2])
    z = vd.Vector([vd.f32(x) * 0.01, 0.1])
    iterations = 0
    running = vd.norm(z) < 20.0
    while running:
        z = (
            vd.Vector(
                [
                    z[0] * z[0] - z[1] * z[1],
                    z[1] * z[0] * 2.0,
                ]
            )
            + c
        )
        iterations += 1
        running = vd.norm(z) < 20.0
        if iterations >= 8:
            running = False
    output[x] = vd.f32(iterations)


@vd.kernel(workgroup_size=(2, 2, 2))
def storage_texture_round_trip(
    image: vd.Texture["3d", vd.rgba32_float, vd.read_write],  # noqa: F722
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    coordinate = vd.Vector([vd.i32(gid[0]), vd.i32(gid[1]), vd.i32(gid[2])])
    value = vd.texture_load(image, coordinate)
    vd.texture_store(image, coordinate, value + vd.Vector([1.0, 2.0, 3.0, 4.0]))


@vd.kernel(workgroup_size=(2, 1, 1))
def matrix_vector(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    matrix = vd.Matrix([[1.0, 2.0], [3.0, 4.0]])
    value = vd.matmul(matrix, vd.Vector([5.0, 6.0]))
    output[x] = value[x]


@vd.kernel(workgroup_size=(4, 1, 1))
def multi_group_objective(
    scale: vd.f32,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = scale * vd.f32(gid[0])


multi_group_objective_vjp = vd.ad.vjp(multi_group_objective, wrt=("scale",), outputs=("output",))


@vd.kernel(workgroup_size=(4, 1, 1))
def multi_group_storage_objective(
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    output[index] = values[index] * values[index]


multi_group_storage_objective_vjp = vd.ad.vjp(
    multi_group_storage_objective,
    wrt=("values",),
    outputs=("output",),
)


@vd.kernel(workgroup_size=(4, 1, 1))
def cooperative_shared_objective(
    scale: vd.f32,
    carriers: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
    lane_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("local_invocation_id")],
) -> None:
    shared = vd.workgroup_storage(vd.f32, shape=(1,))
    if lane_id[0] == 0:
        shared[0] = scale
    vd.workgroup_barrier()
    output[gid[0]] = shared[0] * carriers[gid[0]]


cooperative_shared_objective_vjp = vd.ad.vjp(
    cooperative_shared_objective,
    wrt=("scale",),
    outputs=("output",),
)


@vd.kernel(workgroup_size=(8, 1, 1))
def floating_power(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    exponent: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = values[x] ** 2.5 + values[x] ** exponent


@vd.kernel(workgroup_size=(1, 1, 1))
def copy_tensor_view(
    output: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read_write],
    source: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[1], gid[0]] = source[gid[1], gid[0]]


@vd.kernel(workgroup_size=(4, 1, 1))
def short_circuit_boolean(
    output: vd.TensorView[vd.i32, (vd.dyn,), vd.write],
    left: vd.TensorView[vd.i32, (vd.dyn,), vd.read],
    right: vd.TensorView[vd.i32, (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    if left[x] > 0 and right[x] > 0:
        output[x] = 1
    else:
        output[x] = 0


@vd.kernel(workgroup_size=(4, 1, 1))
def conditional_select(
    output: vd.TensorView[vd.i32, (vd.dyn,), vd.write],
    left: vd.TensorView[vd.i32, (vd.dyn,), vd.read],
    right: vd.TensorView[vd.i32, (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = left[x] if x % 2 == 0 else right[x]


@vd.func
def sum_odds_until_break(limit: vd.i32) -> vd.i32:
    index = 0
    total = 0
    while index < limit:
        index += 1
        if index % 2 == 0:
            continue
        if index > 7:
            break
        total += index
    return total


@vd.kernel
def loop_control(
    output: vd.TensorView[vd.i32, (vd.dyn,), vd.write],
    limit: vd.i32,
) -> None:
    output[0] = sum_odds_until_break(limit)


@vd.func
def nested_loop_control_value() -> vd.i32:
    outer = 0
    total = 0
    while outer < 3:
        outer += 1
        inner = 0
        while inner < 5:
            inner += 1
            if inner == 2:
                break
            total += 10
        if outer == 2:
            continue
        total += 1
    return total


@vd.kernel
def nested_loop_control(
    output: vd.TensorView[vd.i32, (vd.dyn,), vd.write],
) -> None:
    output[0] = nested_loop_control_value()


@vd.func
def early_return_from_loop(limit: vd.i32) -> vd.i32:
    index = 0
    while index < limit:
        index += 1
        if index == 3:
            return index * 10
    return -1


@vd.kernel
def loop_early_return(
    output: vd.TensorView[vd.i32, (vd.dyn,), vd.write],
    source: vd.TensorView[vd.i32, (vd.dyn,), vd.read],
) -> None:
    output[0] = early_return_from_loop(source[0])


@vd.func
def dynamic_range_sum(start: vd.i32, stop: vd.i32, step: vd.i32) -> vd.i32:
    total = 0
    for index in range(start, stop, step):
        total += index
    return total


@vd.kernel
def dynamic_range(
    output: vd.TensorView[vd.i32, (vd.dyn,), vd.write],
    controls: vd.TensorView[vd.i32, (vd.dyn,), vd.read],
) -> None:
    output[0] = dynamic_range_sum(controls[0], controls[1], controls[2])


@vd.kernel
def range_i32_expressions(
    output: vd.TensorView[vd.i32, (vd.dyn,), vd.write],
    controls: vd.TensorView[vd.i32, (vd.dyn,), vd.read],
) -> None:
    width = controls[0]
    for index in range(width):
        if index % 2 == 0:
            output[index] = index + 1
        else:
            output[index] = width - index


@vd.func
def signed_literal_step_range_sum(
    start: vd.i32,
    stop: vd.i32,
    direction: vd.i32,
) -> vd.i32:
    total = 0
    if direction > 0:
        for index in range(start, stop, 2):
            total += index
    else:
        for index in range(start, stop, -3):
            total += index
    return total


@vd.kernel
def signed_literal_step_range(
    output: vd.TensorView[vd.i32, (vd.dyn,), vd.write],
    controls: vd.TensorView[vd.i32, (vd.dyn,), vd.read],
) -> None:
    output[0] = signed_literal_step_range_sum(controls[0], controls[1], controls[2])


@vd.func
def range_loop_control_value(stop: vd.i32) -> vd.i32:
    total = 0
    for index in range(0, stop):
        value = vd.i32(index)
        if value == 2:
            continue
        if value == 6:
            break
        total += value
    return total


@vd.func
def range_early_return_value(stop: vd.i32) -> vd.i32:
    for index in range(0, stop):
        value = vd.i32(index)
        if value == 3:
            return value * 10
    return -1


@vd.kernel
def range_control_flow(
    output: vd.TensorView[vd.i32, (vd.dyn,), vd.write],
    controls: vd.TensorView[vd.i32, (vd.dyn,), vd.read],
) -> None:
    output[0] = range_loop_control_value(controls[0])
    output[1] = range_early_return_value(controls[0])


@vd.struct(shared=True)
class AggregateRecord:
    vector: vd.Tensor[vd.f32, (2,)]
    pair: vd.Tuple[vd.i32, vd.f32]


@vd.struct(shared=True)
class RuntimeParameters:
    scale: vd.f32
    bias: vd.f32


@vd.struct(shared=True)
class OtherRuntimeParameters:
    scale: vd.f32
    bias: vd.f32


@vd.kernel
def use_runtime_parameters(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    parameters: RuntimeParameters,
) -> None:
    output[0] = parameters.scale + parameters.bias


@vd.struct
class PaddedWorkgroupRecord:
    valid: vd.bool
    payload: AggregateRecord


@vd.kernel(workgroup_size=(4, 1, 1))
def aggregate_workgroup_values(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    lane_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("local_invocation_id")],
    group_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("workgroup_id")],
) -> None:
    shared = vd.workgroup_storage(PaddedWorkgroupRecord, shape=(2, 3))
    lane = lane_id[0]
    group = group_id[0]
    if lane == 0:
        for column in range(0, 3):
            value = vd.i32(group) * 100 + vd.i32(column)
            shared[1, column] = PaddedWorkgroupRecord(
                True,
                AggregateRecord(
                    vd.Vector([vd.f32(value), vd.f32(value) + 0.5]),
                    (value + 10, vd.f32(value) + 0.25),
                ),
            )
    vd.workgroup_barrier()
    column = lane
    if lane == 3:
        column = vd.u32(0)
    loaded = shared[1, column]
    payload = loaded.payload
    offset = (group * 4 + lane) * 4
    if loaded.valid:
        output[offset] = payload.vector[0]
        output[offset + 1] = payload.vector[1]
        output[offset + 2] = vd.f32(payload.pair[0])
        output[offset + 3] = payload.pair[1]


@vd.func
def early_record(flag: vd.bool) -> AggregateRecord:
    if flag:
        return AggregateRecord(vd.Vector([5.0, 6.0]), (7, 2.5))
    return AggregateRecord(vd.Vector([10.0, 11.0]), (9, 4.5))


@vd.func
def early_tuple(flag: vd.bool) -> vd.Tuple[vd.i32, vd.f32]:
    if flag:
        return (3, 4.5)
    return (8, 9.5)


@vd.kernel
def early_return_values(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    flag: vd.i32,
) -> None:
    record = early_record(flag != 0)
    pair = early_tuple(flag != 0)
    output[0] = vd.f32(record.pair[0])
    output[1] = record.pair[1]
    output[2] = vd.f32(pair[0])
    output[3] = pair[1]
    output[4] = record.vector[0]
    output[5] = record.vector[1]


@vd.kernel(workgroup_size=(2, 1, 1))
def copy_tuple_tensor_view(
    output: vd.TensorView[vd.Tuple[vd.i32, vd.f32], (vd.dyn,), vd.write],
    source: vd.TensorView[vd.Tuple[vd.i32, vd.f32], (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = source[gid[0]]


@vd.kernel(workgroup_size=(2, 1, 1))
def copy_value_tensor_view(
    output: vd.TensorView[vd.Tensor[vd.f32, (2,)], (vd.dyn,), vd.write],
    source: vd.TensorView[vd.Tensor[vd.f32, (2,)], (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = source[gid[0]]


@vd.func
def load_vector_element(
    source: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn,), vd.read],
    index: vd.i32,
) -> vd.Vector[vd.f32, 2]:
    return source[index]


@vd.kernel(workgroup_size=(2, 1, 1))
def copy_vector_tensor_view_through_helper(
    output: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn,), vd.write],
    source: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = vd.i32(gid[0])
    output[index] = load_vector_element(source, index)


@vd.kernel(workgroup_size=(64, 1, 1))
def global_atomic_increment(
    values: vd.TensorView[vd.i32, (vd.dyn,), vd.read_write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    vd.atomic_add(values, 0, 1)


@vd.kernel(workgroup_size=(1, 1, 1))
def atomic_fill(
    values: vd.TensorView[vd.i32, (vd.dyn,), vd.read_write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    vd.atomic_add(values, gid[0], 1)


@vd.kernel(workgroup_size=(1, 1, 1))
def global_atomic_operations(values: vd.TensorView[vd.i32, (vd.dyn,), vd.read_write]) -> None:
    vd.atomic_exchange(values, 0, 5)
    vd.atomic_add(values, 1, 3)
    vd.atomic_min(values, 2, 7)
    vd.atomic_max(values, 3, 9)


@vd.kernel(workgroup_size=(4, 1, 1))
def workgroup_atomic_lanes(
    output: vd.TensorView[vd.i32, (vd.dyn,), vd.read_write],
    lane_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("local_invocation_id")],
    group_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("workgroup_id")],
) -> None:
    shared = vd.workgroup_storage(vd.i32, shape=(1,))
    lane = lane_id[0]
    group = group_id[0]
    if lane == 0:
        shared[0] = vd.i32(group) * 100
    vd.workgroup_barrier()
    previous = vd.atomic_add(shared, 0, 1)
    vd.workgroup_barrier()
    if lane == 0:
        vd.atomic_exchange(output, 8 + group, shared[0])
    output[group * 4 + lane] = previous


class KernelTensorRuntimeTests(unittest.TestCase):
    @staticmethod
    def _run_tensor_operators(arch: object) -> np.ndarray:
        vd.init(arch=arch)  # type: ignore[arg-type]
        shape = (2, 4, 4)
        left_array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        right_array = np.linspace(0.25, 2.5, np.prod(shape), dtype=np.float32).reshape(shape)
        output = vd.storage.zeros(dtype=vd.f32, shape=shape)
        tensor_operators(
            output,
            vd.storage.from_numpy(left_array),
            vd.storage.from_numpy(right_array),
            2.0,
            grid=(1, 2, 2),
        )
        return output.to_numpy()

    @staticmethod
    def _run_vector_while(arch: object) -> np.ndarray:
        vd.init(arch=arch)  # type: ignore[arg-type]
        output = vd.storage.zeros(dtype=vd.f32, shape=(16,))
        vector_while(output, 0.35, grid=(2, 1, 1))
        return output.to_numpy()

    @staticmethod
    def _runtime_available(arch: object) -> bool:
        try:
            vd.init(arch=arch)  # type: ignore[arg-type]
        except RuntimeError:
            vd.init(arch=vd.cpu)
            return False
        return True

    def _available_compute_backends(self, *, include_cpu: bool = True) -> list[object]:
        backends: list[object] = [vd.cpu] if include_cpu else []
        for architecture in (vd.cuda, vd.vulkan, vd.directx, vd.metal, vd.opengl, vd.opengles):
            if self._runtime_available(architecture):
                backends.append(architecture)
        return backends

    def test_storage_texture_backend_parity(self) -> None:
        backends = [
            architecture
            for architecture in (vd.vulkan, vd.directx, vd.metal, vd.opengl, vd.opengles)
            if self._runtime_available(architecture)
        ]
        if not backends:
            self.skipTest("no storage texture backend is available")
        source = np.arange(2 * 4 * 4 * 4, dtype=np.float32).reshape(2, 4, 4, 4)
        device_result = source + np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)  # type: ignore[arg-type]
                image = vd.Texture.from_numpy(
                    source,
                    dimension="3d",
                    format=vd.rgba32_float,
                    usage=("storage", "transfer_source", "transfer_destination"),
                )
                storage_texture_round_trip(image, grid=(2, 2, 1))
                replacement = np.full((1, 1, 1, 4), 42.0, dtype=np.float32)
                image.upload(replacement, origin=(0, 1, 1))
                expected = device_result.copy()
                expected[0:1, 1:2, 1:2] = replacement
                np.testing.assert_array_equal(
                    image.download(origin=(0, 1, 1), shape=(1, 2, 2)),
                    expected[0:1, 1:3, 1:3],
                )
                np.testing.assert_array_equal(image.download(), expected)

    def test_three_dimensional_texture_mipmap_backend_parity(self) -> None:
        backends = [architecture for architecture in (vd.vulkan, vd.metal) if self._runtime_available(architecture)]
        if not backends:
            self.skipTest("no 3D texture mipmap backend is available")
        source = np.broadcast_to(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            (2, 4, 4, 4),
        ).copy()
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)  # type: ignore[arg-type]
                image = vd.Texture.from_numpy(
                    source,
                    dimension="3d",
                    format=vd.rgba32_float,
                    mip_levels=2,
                    usage=("sampled", "transfer_source", "transfer_destination"),
                )
                image.generate_mipmaps()
                np.testing.assert_array_equal(
                    image.download(mip_level=1),
                    np.broadcast_to(source[0, 0, 0], (1, 2, 2, 4)),
                )

    def test_global_tensor_view_atomic_backend_parity(self) -> None:
        backends: list[object] = [vd.cpu]
        for architecture in (vd.cuda, vd.vulkan):
            if self._runtime_available(architecture):
                backends.append(architecture)
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)  # type: ignore[arg-type]
                values = vd.storage.zeros(dtype=vd.i32, shape=(1,))
                global_atomic_increment(values, grid=(4, 1, 1))
                self.assertEqual(values.to_numpy()[0], 256)

                inferred = vd.storage.zeros(dtype=vd.i32, shape=(8,))
                atomic_fill(inferred)
                np.testing.assert_array_equal(inferred.to_numpy(), np.ones(8, dtype=np.int32))

                operations = vd.storage.from_numpy(np.array([0, 10, 10, 1], dtype=np.int32))
                global_atomic_operations(operations)
                np.testing.assert_array_equal(operations.to_numpy(), np.array([5, 22, 7, 9], dtype=np.int32))

    def test_workgroup_atomic_lane_backend_parity(self) -> None:
        backends = [
            architecture
            for architecture in (vd.cpu, vd.cuda, vd.vulkan, vd.opengl, vd.directx)
            if self._runtime_available(architecture)
        ]
        if not backends:
            self.skipTest("no workgroup synchronization backend is available")

        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)  # type: ignore[arg-type]
                output = vd.storage.zeros(dtype=vd.i32, shape=(10,))
                workgroup_atomic_lanes(output, grid=(2, 1, 1))
                result = output.to_numpy()
                for group in range(2):
                    base = group * 100
                    previous = np.sort(result[group * 4 : group * 4 + 4])
                    np.testing.assert_array_equal(previous, np.arange(base, base + 4, dtype=np.int32))
                    self.assertEqual(result[8 + group], base + 4)

    def test_aggregate_workgroup_backend_parity(self) -> None:
        backends = [
            architecture
            for architecture in (vd.cpu, vd.cuda, vd.vulkan, vd.opengl, vd.directx)
            if self._runtime_available(architecture)
        ]
        if not backends:
            self.skipTest("no aggregate workgroup backend is available")

        expected = np.empty(32, dtype=np.float32)
        for group in range(2):
            for lane in range(4):
                column = lane if lane < 3 else 0
                value = group * 100 + column
                offset = (group * 4 + lane) * 4
                expected[offset : offset + 4] = (value, value + 0.5, value + 10, value + 0.25)

        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)  # type: ignore[arg-type]
                output = vd.storage.zeros(dtype=vd.f32, shape=(32,))
                aggregate_workgroup_values(output, grid=(2, 1, 1))
                np.testing.assert_allclose(output.to_numpy(), expected, rtol=0.0, atol=0.0)

    def test_rank_three_tensor_operators(self) -> None:
        actual = self._run_tensor_operators(vd.cpu)
        shape = (2, 4, 4)
        left = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        right = np.linspace(0.25, 2.5, np.prod(shape), dtype=np.float32).reshape(shape)
        expected = ((left + right) * 2.0 - right) / 2.0
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

        backends = self._available_compute_backends(include_cpu=False)
        for backend in backends:
            with self.subTest(backend=backend.name):
                backend_actual = self._run_tensor_operators(backend)
                np.testing.assert_allclose(backend_actual, expected, rtol=0.0, atol=1e-6)

    def test_autodiff_grid_counts_multiple_cooperative_workgroups(self) -> None:
        vd.init(arch=vd.cpu)
        output = vd.storage.zeros(dtype=vd.f32, shape=(8,))
        result, pullback = multi_group_objective_vjp(np.float32(2.0), output, grid=(2, 1, 1))
        self.assertIsNone(result)
        np.testing.assert_array_equal(output.to_numpy(), np.arange(8, dtype=np.float32) * 2.0)
        cotangent = np.zeros((1, 1, 8, 8), dtype=np.float32)
        cotangent[0, 0, np.arange(8), np.arange(8)] = np.arange(1, 9, dtype=np.float32)
        gradient = pullback(cotangent)["scale"]
        np.testing.assert_allclose(gradient, np.array(168.0, dtype=np.float32), rtol=0.0, atol=0.0)

    def test_autodiff_storage_gradient_keeps_every_cooperative_lane(self) -> None:
        vd.init(arch=vd.cpu)
        values_array = np.linspace(0.5, 4.0, 8, dtype=np.float32)
        weights = np.arange(1, 9, dtype=np.float32)
        values = vd.storage.from_numpy(values_array)
        output = vd.storage.zeros(dtype=vd.f32, shape=(8,))
        _, pullback = multi_group_storage_objective_vjp(values, output, grid=(2, 1, 1))

        cotangent = np.zeros((1, 1, 8, 8), dtype=np.float32)
        cotangent[0, 0, np.arange(8), np.arange(8)] = weights
        expected = 2.0 * values_array * weights
        first = pullback(cotangent)["values"].to_numpy()
        second = pullback(cotangent * np.float32(-0.25))["values"].to_numpy()
        np.testing.assert_allclose(first, expected, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(second, expected * -0.25, rtol=1.0e-6, atol=1.0e-6)

        epsilon = np.float32(1.0e-3)
        for index in (0, 3, 7):
            positive = values_array.copy()
            negative = values_array.copy()
            positive[index] += epsilon
            negative[index] -= epsilon
            finite_difference = (
                weights[index]
                * (np.float64(positive[index]) ** 2 - np.float64(negative[index]) ** 2)
                / np.float64(positive[index] - negative[index])
            )
            np.testing.assert_allclose(first[index], finite_difference, rtol=2.0e-3, atol=2.0e-3)

    def test_autodiff_joint_reverse_of_workgroup_shared_storage(self) -> None:
        vd.init(arch=vd.cpu)
        carriers_array = np.linspace(0.25, 2.0, 8, dtype=np.float32)
        weights = np.arange(1, 9, dtype=np.float32)
        carriers = vd.storage.from_numpy(carriers_array)
        output = vd.storage.zeros(dtype=vd.f32, shape=(8,))
        _, pullback = cooperative_shared_objective_vjp(
            np.float32(1.5),
            carriers,
            output,
            grid=(2, 1, 1),
        )
        cotangent = np.zeros((1, 1, 8, 8), dtype=np.float32)
        cotangent[0, 0, np.arange(8), np.arange(8)] = weights
        expected = np.sum(carriers_array * weights, dtype=np.float32)
        gradient = pullback(cotangent)["scale"]
        np.testing.assert_allclose(gradient, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_lazy_short_circuit_boolean_backend_parity(self) -> None:
        left_values = np.array((1, 0, -1, 2), dtype=np.int32)
        right_values = np.array((1, 1, 1, -2), dtype=np.int32)
        expected = np.array((1, 0, 0, 0), dtype=np.int32)
        backends = self._available_compute_backends()
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                output = vd.storage.zeros(dtype=vd.i32, shape=(4,))
                short_circuit_boolean(
                    output,
                    vd.storage.from_numpy(left_values),
                    vd.storage.from_numpy(right_values),
                    grid=(1, 1, 1),
                )
                np.testing.assert_array_equal(output.to_numpy(), expected)

    def test_conditional_expression_backend_parity(self) -> None:
        left_values = np.array((10, 20, 30, 40), dtype=np.int32)
        right_values = np.array((1, 2, 3, 4), dtype=np.int32)
        expected = np.array((10, 2, 30, 4), dtype=np.int32)
        backends = self._available_compute_backends()
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                output = vd.storage.zeros(dtype=vd.i32, shape=(4,))
                conditional_select(
                    output,
                    vd.storage.from_numpy(left_values),
                    vd.storage.from_numpy(right_values),
                    grid=(1, 1, 1),
                )
                np.testing.assert_array_equal(output.to_numpy(), expected)

    def test_break_continue_backend_parity(self) -> None:
        backends = self._available_compute_backends()
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                output = vd.storage.zeros(dtype=vd.i32, shape=(1,))
                loop_control(output, 20)
                np.testing.assert_array_equal(output.to_numpy(), np.array((16,), dtype=np.int32))

    def test_nested_loop_control_targets_nearest_loop(self) -> None:
        backends = self._available_compute_backends()
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                output = vd.storage.zeros(dtype=vd.i32, shape=(1,))
                nested_loop_control(output)
                np.testing.assert_array_equal(output.to_numpy(), np.array((32,), dtype=np.int32))

    def test_early_return_aggregate_payload_backend_parity(self) -> None:
        backends = self._available_compute_backends()
        for backend in backends:
            for flag, expected in (
                (True, np.array((7.0, 2.5, 3.0, 4.5, 5.0, 6.0), dtype=np.float32)),
                (False, np.array((9.0, 4.5, 8.0, 9.5, 10.0, 11.0), dtype=np.float32)),
            ):
                with self.subTest(backend=backend.name, flag=flag):
                    vd.init(arch=backend)
                    output = vd.storage.zeros(dtype=vd.f32, shape=(6,))
                    early_return_values(output, 1 if flag else 0, grid=(1, 1, 1))
                    np.testing.assert_array_equal(output.to_numpy(), expected)

    def test_loop_early_return_backend_parity(self) -> None:
        backends = self._available_compute_backends()
        for backend in backends:
            for limit, expected in ((2, -1), (8, 30)):
                with self.subTest(backend=backend.name, limit=limit):
                    vd.init(arch=backend)
                    output = vd.storage.zeros(dtype=vd.i32, shape=(1,))
                    source = vd.storage.from_numpy(np.array((limit,), dtype=np.int32))
                    loop_early_return(output, source)
                    np.testing.assert_array_equal(output.to_numpy(), np.array((expected,), dtype=np.int32))

    def test_dynamic_signed_range_step_cpu_cuda_parity(self) -> None:
        cases = (
            ((0, 10, 2), 20),
            ((10, 0, -3), 22),
            ((5, 5, 1), 0),
            ((0, 5, -1), 0),
            ((2_147_483_646, 2_147_483_647, 2), 2_147_483_646),
            ((-2_147_483_647, -2_147_483_648, -2), -2_147_483_647),
        )
        backends = [vd.cpu]
        if self._runtime_available(vd.cuda):
            backends.append(vd.cuda)
        for backend in backends:
            for controls, expected in cases:
                with self.subTest(backend=backend.name, controls=controls):
                    vd.init(arch=backend)
                    output = vd.storage.zeros(dtype=vd.i32, shape=(1,))
                    control_values = vd.storage.from_numpy(np.array(controls, dtype=np.int32))
                    dynamic_range(output, control_values, grid=(1, 1, 1))
                    np.testing.assert_array_equal(output.to_numpy(), np.array((expected,), dtype=np.int32))

    def test_range_induction_is_i32_until_tensor_view_indexing(self) -> None:
        vd.init(arch=vd.cpu)
        output = vd.storage.zeros(dtype=vd.i32, shape=(6,))
        controls = vd.storage.from_numpy(np.array((6,), dtype=np.int32))
        range_i32_expressions(output, controls, grid=(1, 1, 1))
        np.testing.assert_array_equal(output.to_numpy(), np.array((1, 5, 3, 3, 5, 1), dtype=np.int32))

    def test_shared_struct_value_parameter_executes_on_cpu(self) -> None:
        vd.init(arch=vd.cpu)
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        use_runtime_parameters(
            output,
            RuntimeParameters(np.float32(2.5), np.float32(1.25)),
            grid=(1, 1, 1),
        )
        np.testing.assert_array_equal(output.to_numpy(), np.array((3.75,), dtype=np.float32))
        use_runtime_parameters(
            output,
            {"scale": np.float32(4.0), "bias": np.float32(-1.5)},
            grid=(1, 1, 1),
        )
        np.testing.assert_array_equal(output.to_numpy(), np.array((2.5,), dtype=np.float32))
        with self.assertRaisesRegex(TypeError, "is missing field 'bias'"):
            use_runtime_parameters(
                output,
                {"scale": np.float32(2.5)},
                grid=(1, 1, 1),
            )
        with self.assertRaisesRegex(TypeError, "has type OtherRuntimeParameters, expected RuntimeParameters"):
            use_runtime_parameters(
                output,
                OtherRuntimeParameters(np.float32(2.5), np.float32(1.25)),
                grid=(1, 1, 1),
            )

    def test_dynamic_zero_range_step_is_runtime_contract_violation(self) -> None:
        import os
        import subprocess
        import sys

        project = Path(__file__).resolve().parents[2]
        script = (
            "import sys; "
            "sys.path.insert(0, 'python/tests'); "
            "import numpy as np, test_kernel_runtime as tests, vernon_dsl as vd; "
            "vd.init(arch=vd.cpu); "
            "output = vd.storage.zeros(dtype=vd.i32, shape=(1,)); "
            "controls = vd.storage.from_numpy(np.array((0, 4, 0), dtype=np.int32)); "
            "tests.dynamic_range(output, controls, grid=(1, 1, 1))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project,
            env={**os.environ, "PYTHONPATH": "python"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_dynamic_range_bounds_all_backend_parity(self) -> None:
        cases = (((0, 10, 1), 20), ((10, 0, -1), 22), ((5, 5, 1), 0))
        backends = self._available_compute_backends()
        for backend in backends:
            for controls, expected in cases:
                with self.subTest(backend=backend.name, controls=controls):
                    vd.init(arch=backend)
                    output = vd.storage.zeros(dtype=vd.i32, shape=(1,))
                    control_values = vd.storage.from_numpy(np.array(controls, dtype=np.int32))
                    signed_literal_step_range(output, control_values, grid=(1, 1, 1))
                    np.testing.assert_array_equal(output.to_numpy(), np.array((expected,), dtype=np.int32))

    def test_range_break_continue_and_early_return_backend_parity(self) -> None:
        backends = self._available_compute_backends()
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                output = vd.storage.zeros(dtype=vd.i32, shape=(2,))
                controls = vd.storage.from_numpy(np.array((10,), dtype=np.int32))
                range_control_flow(output, controls, grid=(1, 1, 1))
                np.testing.assert_array_equal(output.to_numpy(), np.array((13, 30), dtype=np.int32))

    def test_loop_carried_vector_norm(self) -> None:
        expected = self._run_vector_while(vd.cpu)
        backends = self._available_compute_backends(include_cpu=False)
        for backend in backends:
            with self.subTest(backend=backend.name):
                actual = self._run_vector_while(backend)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_matrix_specialization(self) -> None:
        backends = self._available_compute_backends(include_cpu=False)
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                output = vd.storage.zeros(dtype=vd.f32, shape=(2,))
                matrix_vector(output, grid=(1, 1, 1))
                np.testing.assert_allclose(
                    output.to_numpy(), np.array((17.0, 39.0), dtype=np.float32), rtol=0.0, atol=1e-6
                )

    def test_literal_and_dynamic_floating_power(self) -> None:
        values = np.linspace(0.25, 2.0, 16, dtype=np.float32)
        exponent = 1.75
        expected = values ** np.float32(2.5) + values ** np.float32(exponent)
        backends = self._available_compute_backends()
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                output = vd.storage.zeros(dtype=vd.f32, shape=values.shape)
                floating_power(
                    output,
                    vd.storage.from_numpy(values),
                    exponent,
                    grid=(2, 1, 1),
                )
                np.testing.assert_allclose(output.to_numpy(), expected, rtol=2e-6, atol=2e-6)

    def test_strided_tensor_view_dispatch(self) -> None:
        backends = self._available_compute_backends()
        expected = np.array([[2.0, 1.0, 0.0], [8.0, 7.0, 6.0]], dtype=np.float32)
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                source_bytes = bytearray(np.arange(12, dtype=np.float32).tobytes())
                output_bytes = bytearray(12 * np.dtype(np.float32).itemsize)
                source = vd.interop.RawBuffer.from_buffer(source_bytes, alignment=4).typed_view(
                    dtype=vd.f32,
                    shape=(2, 3),
                    byte_strides=(24, -4),
                    byte_offset=8,
                    access="read",
                    layout_units="bytes",
                )
                output = vd.interop.RawBuffer.from_buffer(output_bytes, alignment=4).typed_view(
                    dtype=vd.f32,
                    shape=(2, 3),
                    byte_strides=(20, 4),
                    byte_offset=4,
                    access="read_write",
                    layout_units="bytes",
                )

                copy_tensor_view(output, source)
                np.testing.assert_array_equal(output.to_numpy(), expected)

        for target in ("cuda", "vulkan"):
            with self.subTest(target=target):
                artifact, reflection = copy_tensor_view.compile_artifact(output, source, target=target)
                self.assertTrue(artifact)
                self.assertIn('"tensor_views"', reflection)

    def test_dynamic_tensor_view_artifact_reuses_transposed_dispatch(self) -> None:
        backends = self._available_compute_backends()
        for backend in backends:
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                copy_tensor_view.compile_count = 0
                type(copy_tensor_view).clear_cache()

                first_source = vd.storage.from_numpy(np.arange(4, dtype=np.float32).reshape(2, 2))
                first_output = vd.storage.zeros(dtype=vd.f32, shape=(2, 2))
                copy_tensor_view(first_output, first_source)
                np.testing.assert_array_equal(first_output.to_numpy(), first_source.to_numpy())

                owner = vd.storage.from_numpy(np.arange(6, dtype=np.float32).reshape(3, 2))
                transposed_reversed = owner.view(shape=(2, 3), strides=(-1, 2), offset=1, access="read")
                second_output = vd.storage.zeros(dtype=vd.f32, shape=(8,)).view(
                    shape=(2, 3), strides=(4, 1), offset=1, access="read_write"
                )
                copy_tensor_view(second_output, transposed_reversed)
                np.testing.assert_array_equal(second_output.to_numpy(), owner.to_numpy().T[::-1])
                self.assertEqual(copy_tensor_view.compile_count, 1)
                self.assertEqual(len(type(copy_tensor_view)._dispatch_cache), 1)

    def test_cached_tensor_view_dispatch_revalidates_abi(self) -> None:
        vd.init(arch=vd.cpu)
        copy_tensor_view.compile_count = 0
        type(copy_tensor_view).clear_cache()
        output = vd.storage.zeros(dtype=vd.f32, shape=(2, 2))
        source = vd.storage.zeros(dtype=vd.f32, shape=(2, 2))
        copy_tensor_view(output, source)

        invalid_rank = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        with self.assertRaisesRegex(TypeError, "rank 1, expected 2"):
            copy_tensor_view(output, invalid_rank)

        invalid_dtype = vd.storage.zeros(dtype=vd.i32, shape=(2, 2))
        with self.assertRaisesRegex(TypeError, "dtype int32 does not match f32"):
            copy_tensor_view(output, invalid_dtype)

        invalid_access = vd.storage.zeros(dtype=vd.f32, shape=(2, 2)).view(access="write")
        with self.assertRaisesRegex(TypeError, "access 'write' does not satisfy 'read'"):
            copy_tensor_view(output, invalid_access)

        self.assertEqual(copy_tensor_view.compile_count, 1)
        self.assertEqual(len(type(copy_tensor_view)._dispatch_cache), 1)

    def test_aggregate_tensor_view_dispatch(self) -> None:
        values = tuple(
            ComplexAggregateVertex(
                np.array((float(index), float(index) + 0.5), dtype=np.float32),
                AggregateVertexPayload(
                    vd.i32(index * 3),
                    np.array((index + 0.25, index + 0.75), dtype=np.float32),
                    np.array((index + 1.0, index + 2.0), dtype=np.float32),
                ),
                np.array(((index + 3.0, index + 4.0), (index + 5.0, index + 6.0)), dtype=np.float32),
            )
            for index in range(4)
        )
        for backend in self._available_compute_backends():
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                source = vd.storage.from_values(values, dtype=ComplexAggregateVertex).view(
                    shape=(4,), strides=(-1,), offset=3, access="read"
                )
                output_storage = vd.storage.zeros(dtype=ComplexAggregateVertex, shape=(4,))
                self.assertEqual(host_abi_layout(ComplexAggregateVertex).size, 44)
                self.assertEqual(output_storage._native_host_array().nbytes, 4 * 44)
                output = output_storage.view(access="write")
                copy_complex_aggregate_tensor_view(output, source, grid=(1, 1, 1))
                actual = output_storage.to_values()
                for result, expected in zip(actual, reversed(values), strict=True):
                    np.testing.assert_array_equal(result.position, expected.position)
                    self.assertEqual(result.payload.object_id, expected.payload.object_id)
                    np.testing.assert_array_equal(result.payload.uv, expected.payload.uv)
                    np.testing.assert_array_equal(result.payload.weights, expected.payload.weights)
                    np.testing.assert_array_equal(result.basis, expected.basis)

                tuple_type = vd.Tuple[vd.i32, vd.f32]
                tuple_values = ((vd.i32(2), vd.f32(3.5)), (vd.i32(5), vd.f32(7.5)))
                tuple_source = vd.storage.from_values(tuple_values, dtype=tuple_type)
                tuple_output = vd.storage.zeros(dtype=tuple_type, shape=(2,))
                copy_tuple_tensor_view(
                    tuple_output.view(access="write"),
                    tuple_source.view(access="read"),
                    grid=(1, 1, 1),
                )
                self.assertEqual(tuple_output.to_values(), tuple_values)

                tensor_type = vd.Tensor[vd.f32, (2,)]
                tensor_values = (
                    np.array([1.0, 2.0], dtype=np.float32),
                    np.array([3.0, 4.0], dtype=np.float32),
                )
                tensor_source_bytes = bytearray(np.asarray(tensor_values, dtype=np.float32).tobytes())
                tensor_output_bytes = bytearray(4 * np.dtype(np.float32).itemsize)
                tensor_source = vd.interop.RawBuffer.from_buffer(tensor_source_bytes, alignment=4).typed_view(
                    dtype=tensor_type,
                    shape=(2,),
                    byte_strides=(8,),
                    access="read",
                    layout_units="bytes",
                )
                tensor_output = vd.interop.RawBuffer.from_buffer(tensor_output_bytes, alignment=4).typed_view(
                    dtype=tensor_type,
                    shape=(2,),
                    byte_strides=(8,),
                    access="write",
                    layout_units="bytes",
                )
                copy_value_tensor_view(tensor_output, tensor_source, grid=(1, 1, 1))
                tensor_output.owner.synchronize()
                np.testing.assert_array_equal(
                    np.frombuffer(tensor_output_bytes, dtype=np.float32).reshape(2, 2),
                    np.asarray(tensor_values),
                )

                vector_type = vd.Vector[vd.f32, 2]
                vector_source = vd.storage.from_values(tensor_values, dtype=vector_type)
                vector_output = vd.storage.zeros(dtype=vector_type, shape=(2,))
                vector_source.view(access="read_write").copy_from_numpy(
                    np.ascontiguousarray(tensor_values, dtype=np.float32)
                )
                copy_vector_tensor_view_through_helper(vector_output, vector_source, grid=(1, 1, 1))
                np.testing.assert_array_equal(
                    vector_output.to_numpy(),
                    np.asarray(tensor_values),
                )
                for actual, expected in zip(vector_output.to_values(), tensor_values, strict=True):
                    np.testing.assert_array_equal(actual, expected)

    def test_multidimensional_aggregate_tensor_dispatch(self) -> None:
        values = tuple(
            ComplexAggregateVertex(
                np.array((float(index), float(index) + 0.5), dtype=np.float32),
                AggregateVertexPayload(
                    vd.i32(index * 3),
                    np.array((index + 0.25, index + 0.75), dtype=np.float32),
                    np.array((index + 1.0, index + 2.0), dtype=np.float32),
                ),
                np.array(((index + 3.0, index + 4.0), (index + 5.0, index + 6.0)), dtype=np.float32),
            )
            for index in range(2 * 4 * 3)
        )
        expected = np.array((23.0, 23.5, 69.0, 23.75, 24.0, 28.0), dtype=np.float32)

        for backend in self._available_compute_backends():
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                output = vd.storage.zeros(dtype=vd.f32, shape=(6,))
                tensor = vd.storage.from_values(
                    tuple(
                        tuple(tuple(values[plane * 12 + row * 4 + column] for column in range(4)) for row in range(3))
                        for plane in range(2)
                    ),
                    dtype=ComplexAggregateVertex,
                )
                self.assertEqual(tensor._native_host_array().nbytes, 2 * 3 * 4 * 44)
                inspect_multidimensional_aggregate_tensor_value(output, tensor, grid=(1, 1, 1))
                np.testing.assert_array_equal(output.to_numpy(), expected)

    def test_cross_compiled_source_generation(self) -> None:
        output = vd.storage.zeros(dtype=vd.f32, shape=(16,))
        cases = {
            "metal": "kernel void vector_while",
            "opengl": "#version 430",
            "opengles": "#version 310 es",
        }
        for target, marker in cases.items():
            with self.subTest(target=target):
                source, reflection = vector_while.compile_artifact(output, 0.35, target=target)
                self.assertIn(marker, source.decode())
                self.assertIn(f'"target":"{target}"', reflection)


@vd.kernel(workgroup_size=(1, 1, 1))
def fill(
    output: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    scale: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    y = gid[1]
    if x < 3:
        if y < 2:
            output[y, x] = vd.f32(x) + vd.f32(y) * scale


class TensorTests(unittest.TestCase):
    def test_numpy_copy_contract(self) -> None:
        source = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = vd.storage.from_numpy(source)
        source.fill(0)
        np.testing.assert_array_equal(tensor.to_numpy(), np.arange(6, dtype=np.float32).reshape(2, 3))
        result = tensor.to_numpy()
        result.fill(0)
        self.assertNotEqual(float(tensor.to_numpy()[1, 2]), 0.0)

    def test_copy_validates_layout(self) -> None:
        tensor = vd.storage.zeros(dtype=vd.f32, shape=(2, 3))
        with self.assertRaises(ValueError):
            tensor.copy_from_numpy(np.zeros((3, 2), dtype=np.float32))


class KernelTests(unittest.TestCase):
    def setUp(self) -> None:
        vd.init(arch=vd.cpu)
        fill.compile_count = 0
        type(fill).clear_cache()

    def test_explicit_grid_and_cache(self) -> None:
        output = vd.storage.zeros(dtype=vd.f32, shape=(2, 3))
        fill(output, 10.0, grid=(3, 2, 1))
        np.testing.assert_array_equal(
            output.to_numpy(),
            np.array([[0, 1, 2], [10, 11, 12]], dtype=np.float32),
        )
        self.assertEqual(fill.compile_count, 1)
        fill(output, 20.0, grid=(3, 2, 1))
        self.assertEqual(fill.compile_count, 1)

    def test_warm_dispatch_skips_frontend_lowering(self) -> None:
        output = vd.storage.zeros(dtype=vd.f32, shape=(2, 3))
        with mock.patch.object(fill, "_lower", wraps=fill._lower) as lower:
            fill(output, 10.0, grid=(3, 2, 1))
            fill(output, 20.0, grid=(3, 2, 1))
        lower.assert_called_once()

    def test_cpu_kernel_uses_in_process_owning_compiler(self) -> None:
        output = vd.storage.zeros(dtype=vd.f32, shape=(2, 3))
        with mock.patch("subprocess.run", side_effect=AssertionError("subprocess prohibited")):
            fill(output, 1.0)
        np.testing.assert_array_equal(
            output.to_numpy(),
            np.array([[0, 1, 2], [1, 2, 3]], dtype=np.float32),
        )

    def test_compile_artifact_uses_in_process_owning_compiler(self) -> None:
        output = vd.storage.zeros(dtype=vd.f32, shape=(2, 3))
        with mock.patch("subprocess.run", side_effect=AssertionError("subprocess prohibited")):
            artifact, reflection = fill.compile_artifact(output, 1.0, target="vulkan")
        self.assertTrue(artifact)
        self.assertIn('"target":"vulkan"', reflection)

    def test_runtime_reinit_reloads_without_recompiling(self) -> None:
        output = vd.storage.zeros(dtype=vd.f32, shape=(2, 3))
        fill(output, 1.0)
        self.assertEqual(fill.compile_count, 1)
        vd.init(arch=vd.cpu)
        fill(output, 2.0)
        self.assertEqual(fill.compile_count, 1)

    def test_dynamic_tensor_view_shape_reuses_cache(self) -> None:
        first = vd.storage.zeros(dtype=vd.f32, shape=(2, 3))
        second = vd.storage.zeros(dtype=vd.f32, shape=(3, 3))
        fill(first, 1.0, grid=(3, 2, 1))
        fill(second, 1.0, grid=(3, 2, 1))
        self.assertEqual(fill.compile_count, 1)

    def test_grid_is_inferred_and_validated(self) -> None:
        output = vd.storage.zeros(dtype=vd.f32, shape=(2, 3))
        fill(output, 1.0)
        np.testing.assert_array_equal(
            output.to_numpy(),
            np.array([[0, 1, 2], [1, 2, 3]], dtype=np.float32),
        )
        with self.assertRaises(ValueError):
            fill(output, 1.0, grid=(3, 0, 1))

    def test_cpu_host_tensor_avoids_device_residency(self) -> None:
        output = vd.storage.zeros(dtype=vd.f32, shape=(2, 3))
        fill(output, 2.0)
        fill(output, 3.0)
        self.assertEqual(output._allocation_count, 0)
        self.assertEqual(output._upload_count, 0)
        self.assertEqual(output._download_count, 0)
        output.to_numpy()
        self.assertEqual(output._download_count, 0)

    def test_fractal_matches_vectorized_numpy_reference(self) -> None:
        width, height = 4, 3
        time = 0.2
        output = vd.storage.zeros(dtype=vd.f32, shape=(fractal.HEIGHT, fractal.WIDTH))
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
        np.testing.assert_allclose(output.to_numpy()[:height, :width], expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
