from __future__ import annotations

import gc
import tempfile
import unittest
import weakref
from pathlib import Path

import numpy as np
import vernon_dsl as vd
from vernon_dsl import CompileError, Compiler, compile_source
from vernon_dsl._runtime.resources import (
    _dispatch_borrow_scope,
    _logical_collection_shape,
    _validate_dispatch_borrows,
)
from vernon_dsl.compiler import FrontendCompileRequest
from vernon_dsl.frontend.analysis import typed_model_data


@vd.struct(shared=True)
class StorageVertex:
    position: vd.Vector[vd.f32, 3]
    weight: vd.f64
    uv: vd.Vector[vd.f32, 2]


@vd.struct(shared=True)
class NestedRecord:
    vertex: StorageVertex
    pair: vd.Tuple[vd.f32, vd.i32]


class TensorStorageRuntimeTests(unittest.TestCase):
    def test_logical_collection_shape_uses_value_structure(self) -> None:
        tuple_type = vd.Tuple[vd.i32, vd.f32]
        first = (vd.i32(1), vd.f32(2.0))
        second = (vd.i32(3), vd.f32(4.0))

        self.assertEqual(_logical_collection_shape(first, tuple_type), ())
        self.assertEqual(_logical_collection_shape((first, second), tuple_type), (2,))
        self.assertEqual(_logical_collection_shape(((first, second), (second, first)), tuple_type), (2, 2))
        with self.assertRaisesRegex(ValueError, "rectangular"):
            _logical_collection_shape(((first,), (first, second)), tuple_type)

        tensor_type = vd.Tensor[vd.f32, (2,)]
        tensor_value = np.array([1.0, 2.0], dtype=np.float32)
        self.assertEqual(_logical_collection_shape(tensor_value, tensor_type), ())
        self.assertEqual(_logical_collection_shape((tensor_value, tensor_value), tensor_type), (2,))

    def test_dense_owner_reports_element_and_byte_layout(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(2, 3))

        self.assertEqual(storage.shape, (2, 3))
        self.assertEqual(storage.layout.element_strides, (3, 1))
        self.assertEqual(storage.layout.element_offset, 0)
        self.assertEqual(storage.layout.byte_strides, (12, 4))
        self.assertEqual(storage.layout.byte_offset, 0)

    def test_strided_and_negative_views_project_without_copying(self) -> None:
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        storage = vd.TensorStorage.from_numpy(values)
        interleaved = storage.view(shape=(3, 2), strides=(4, 1), offset=1, access="read")
        reversed_row = storage.view(shape=(4,), strides=(-1,), offset=3, access="read")

        np.testing.assert_array_equal(interleaved.to_numpy(), values[:, 1:3])
        np.testing.assert_array_equal(reversed_row.to_numpy(), values[0, ::-1])
        self.assertEqual(interleaved.layout.byte_strides, (16, 4))
        self.assertEqual(interleaved.layout.byte_offset, 4)

    def test_view_validation_rejects_out_of_bounds_and_writable_aliasing(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.i32, shape=(8,))

        with self.assertRaisesRegex(ValueError, "outside its owner"):
            storage.view(shape=(4,), strides=(1,), offset=6)
        with self.assertRaisesRegex(ValueError, "internally injective"):
            storage.view(shape=(2, 2), strides=(1, 1), access="write")

        overlapping_reader = storage.view(shape=(2, 2), strides=(1, 1), access="read")
        np.testing.assert_array_equal(overlapping_reader.to_numpy(), np.zeros((2, 2), dtype=np.int32))

    def test_access_modes_and_owner_lifetime_are_enforced(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(4,))
        owner_ref = weakref.ref(storage)
        read_view = storage.view(shape=(4,), strides=(1,), access="read")
        write_view = storage.view(shape=(4,), strides=(1,), access="write")
        del storage
        gc.collect()

        self.assertIsNotNone(owner_ref())
        with self.assertRaises(PermissionError):
            read_view.copy_from_numpy(np.ones((4,), dtype=np.float32))
        with self.assertRaises(PermissionError):
            write_view.to_numpy()
        write_view.copy_from_numpy(np.arange(4, dtype=np.float32))
        np.testing.assert_array_equal(read_view.to_numpy(), np.arange(4, dtype=np.float32))

    def test_storage_namespace_is_distinct_from_tensor_values(self) -> None:
        storage = vd.storage.zeros(dtype=vd.f32, shape=(2,))
        value = vd.Tensor([1.0, 2.0])

        self.assertIsInstance(storage, vd.TensorStorage)
        self.assertFalse(hasattr(vd.Tensor, "zeros"))
        self.assertFalse(value.flags.writeable)

    def test_dispatch_borrows_allow_disjoint_writes_and_reject_aliases(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(4, 4))
        left = storage.view(shape=(4, 2), strides=(4, 1), offset=0, access="write")
        right = storage.view(shape=(4, 2), strides=(4, 1), offset=2, access="write")
        overlapping = storage.view(shape=(4, 2), strides=(4, 1), offset=1, access="read")

        _validate_dispatch_borrows([("left", left, "write"), ("right", right, "write")])
        with self.assertRaisesRegex(ValueError, "incompatible overlapping borrows"):
            _validate_dispatch_borrows([("left", left, "write"), ("overlapping", overlapping, "read")])

    def test_dispatch_access_cannot_exceed_view_capability(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(4,))
        reader = storage.view(shape=(4,), strides=(1,), access="read")
        writer = storage.view(shape=(4,), strides=(1,), access="write")

        with self.assertRaisesRegex(ValueError, "does not permit writes"):
            _validate_dispatch_borrows([("reader", reader, "write")])
        with self.assertRaisesRegex(ValueError, "does not permit reads"):
            _validate_dispatch_borrows([("writer", writer, "read")])

    def test_dispatch_scope_blocks_host_access_until_completion(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(4,))
        values = np.arange(4, dtype=np.float32)

        with _dispatch_borrow_scope([("input", storage, "read")]):
            np.testing.assert_array_equal(storage.to_numpy(), np.zeros((4,), dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError, "host mutation"):
                storage.copy_from_numpy(values)
        storage.copy_from_numpy(values)

        with _dispatch_borrow_scope([("output", storage, "write")]):
            with self.assertRaisesRegex(RuntimeError, "host reads"):
                storage.to_numpy()
            with self.assertRaisesRegex(RuntimeError, "host mutation"):
                storage.copy_from_numpy(values)
        np.testing.assert_array_equal(storage.to_numpy(), values)

    def test_outstanding_dispatch_borrows_are_region_aware(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(8,))
        left = storage.view(shape=(4,), strides=(1,), offset=0, access="write")
        right = storage.view(shape=(4,), strides=(1,), offset=4, access="write")
        overlap = storage.view(shape=(4,), strides=(1,), offset=2, access="write")

        with _dispatch_borrow_scope([("left", left, "write")]):
            with _dispatch_borrow_scope([("right", right, "write")]):
                pass
            with self.assertRaisesRegex(RuntimeError, "outstanding device borrow"):
                with _dispatch_borrow_scope([("overlap", overlap, "write")]):
                    pass

    def test_raw_buffer_requires_explicit_byte_layout_units(self) -> None:
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        backing = bytearray(values.tobytes())
        raw = vd.interop.RawBuffer.from_buffer(backing, alignment=4)
        view = raw.typed_view(
            dtype=vd.f32,
            shape=(3, 2),
            byte_strides=(16, 4),
            byte_offset=4,
            access="read_write",
            layout_units="bytes",
        )

        self.assertEqual(raw.byte_size, values.nbytes)
        np.testing.assert_array_equal(view.to_numpy(), values[:, 1:3])
        view.copy_from_numpy(np.full((3, 2), 7, dtype=np.float32))
        np.testing.assert_array_equal(np.frombuffer(backing, dtype=np.float32).reshape(3, 4)[:, 1:3], 7)

        with self.assertRaisesRegex(ValueError, "layout_units='bytes'"):
            raw.typed_view(
                dtype=vd.f32,
                shape=(1,),
                byte_strides=(4,),
                access="read",
                layout_units="elements",
            )

    def test_raw_buffer_rejects_ambiguous_alignment_and_readonly_writes(self) -> None:
        raw = vd.interop.RawBuffer.allocate(16, alignment=4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            raw.typed_view(
                dtype=vd.f32,
                shape=(1,),
                byte_strides=(4,),
                byte_offset=1,
                access="read",
                layout_units="bytes",
            )
        readonly = vd.interop.RawBuffer.from_buffer(bytes(16), alignment=4)
        with self.assertRaisesRegex(ValueError, "writable RawBuffer"):
            readonly.typed_view(
                dtype=vd.f32,
                shape=(4,),
                byte_strides=(4,),
                access="write",
                layout_units="bytes",
            )

    def test_raw_buffer_typed_view_accepts_aggregate_value_abi(self) -> None:
        vertex = StorageVertex(vd.Vector([1.0, 2.0, 3.0]), 4.0, vd.Vector([0.25, 0.75]))
        value = NestedRecord(vertex, (vd.f32(5.0), vd.i32(6)))
        packed = vd.TensorStorage.from_values((value,), dtype=NestedRecord)
        backing = bytearray(packed.to_numpy().tobytes())
        view = vd.interop.RawBuffer.from_buffer(backing, alignment=8).typed_view(
            dtype=NestedRecord,
            shape=(1,),
            byte_strides=(packed.dtype.itemsize,),
            access="read",
            layout_units="bytes",
        )

        self.assertIs(view.element_type, NestedRecord)
        self.assertEqual(view.dtype, packed.dtype)
        np.testing.assert_array_equal(view.to_numpy(), packed.to_numpy())

    def test_struct_storage_uses_canonical_aos_layout_and_field_views(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=StorageVertex, shape=(4,))

        self.assertEqual(storage.dtype.itemsize, 32)
        self.assertEqual(storage.dtype.fields["position"][1], 0)
        self.assertEqual(storage.dtype.fields["weight"][1], 16)
        self.assertEqual(storage.dtype.fields["uv"][1], 24)

        position = storage.field("position", access="write")
        weight = storage.field("weight", access="write")
        uv = storage.field("uv", access="write")
        self.assertEqual(position.shape, (4, 3))
        self.assertEqual(position.layout.element_strides, (8, 1))
        self.assertEqual(weight.layout.element_strides, (4,))
        self.assertEqual(weight.layout.element_offset, 2)
        self.assertEqual(uv.layout.element_strides, (8, 1))
        self.assertEqual(uv.layout.element_offset, 6)

        positions = np.arange(12, dtype=np.float32).reshape(4, 3)
        weights = np.arange(4, dtype=np.float64)
        coordinates = np.arange(8, dtype=np.float32).reshape(4, 2)
        position.copy_from_numpy(positions)
        weight.copy_from_numpy(weights)
        uv.copy_from_numpy(coordinates)
        np.testing.assert_array_equal(storage.field("position", access="read").to_numpy(), positions)
        np.testing.assert_array_equal(storage.field("weight", access="read").to_numpy(), weights)
        np.testing.assert_array_equal(storage.field("uv", access="read").to_numpy(), coordinates)

        _validate_dispatch_borrows([("position", position, "write"), ("uv", uv, "write")])

    def test_struct_storage_rejects_non_projectable_or_unknown_fields(self) -> None:
        @vd.struct
        class DeviceRecord:
            pair: vd.Tuple[vd.f32, vd.i32]

        storage = vd.TensorStorage.zeros(dtype=DeviceRecord, shape=(2,))
        with self.assertRaisesRegex(TypeError, "Scalar or Tensor"):
            storage.field("pair")
        with self.assertRaisesRegex(ValueError, "has no field"):
            storage.field("missing")
        aggregate_view = storage.view()
        self.assertIs(aggregate_view.element_type, DeviceRecord)
        with self.assertRaisesRegex(TypeError, "cannot materialize"):
            storage.to_values()

    def test_struct_storage_round_trips_immutable_logical_values(self) -> None:
        values = tuple(
            StorageVertex(
                vd.Vector([float(index), float(index + 1), float(index + 2)]),
                vd.f64(index + 0.5),
                vd.Vector([float(index) / 4.0, 1.0]),
            )
            for index in range(3)
        )
        storage = vd.TensorStorage.from_values(values, dtype=StorageVertex)
        round_trip = storage.to_values()

        self.assertEqual(storage.shape, (3,))
        self.assertTrue(all(isinstance(value, StorageVertex) for value in round_trip))
        for actual, expected in zip(round_trip, values, strict=True):
            np.testing.assert_array_equal(actual.position, expected.position)
            np.testing.assert_array_equal(actual.uv, expected.uv)
            self.assertEqual(actual.weight, expected.weight)
            self.assertFalse(actual.position.flags.writeable)

        storage.field("weight").copy_from_numpy(np.zeros((3,), dtype=np.float64))
        self.assertEqual(round_trip[2].weight, values[2].weight)
        with self.assertRaisesRegex(ValueError, "does not match storage shape"):
            storage.copy_from_values(values[:2])

    def test_nested_struct_and_tuple_storage_round_trip(self) -> None:
        vertex = StorageVertex(vd.Vector([1.0, 2.0, 3.0]), 4.0, vd.Vector([0.25, 0.75]))
        value = NestedRecord(vertex, (vd.f32(5.0), vd.i32(6)))
        storage = vd.TensorStorage.from_values((value,), dtype=NestedRecord)
        actual = storage.to_values()[0]

        self.assertIsInstance(actual, NestedRecord)
        np.testing.assert_array_equal(actual.vertex.position, vertex.position)
        self.assertEqual(actual.pair, value.pair)


class TensorViewFrontendTests(unittest.TestCase):
    SOURCE = (
        "from typing import Annotated\n"
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def copy(\n"
        "    output: TensorView[f32, (dyn,), write],\n"
        "    source: TensorView[f32, (dyn,), read],\n"
        "    gid: Annotated[Tensor[u32, (3,)], builtin('global_invocation_id')],\n"
        ") -> None:\n"
        "    output[gid[0]] = source[gid[0]]\n"
    )

    def test_typed_model_preserves_tensor_view_shape_and_access(self) -> None:
        with self.subTest("lowering"):
            output = compile_source(self.SOURCE, "tensor_view.py")
            self.assertIn(
                '!vernon.tensor_view<f32, [-1], "write", "device">',
                output,
            )
            self.assertIn(
                '!vernon.tensor_view<f32, [-1], "read", "device">',
                output,
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensor_view.py"
            path.write_text(self.SOURCE, encoding="utf-8")
            result = Compiler().compile_request(FrontendCompileRequest(path, "copy"))
        parameters = typed_model_data(result.typed_functions)[0]["parameters"]
        self.assertEqual(
            parameters[0]["type"],
            '!vernon.tensor_view<f32, [-1], "write", "device">',
        )
        self.assertEqual(result.typed_functions[0].parameters[0].type.arguments[1], ("?",))
        self.assertNotIn("storage", parameters[0])
        self.assertEqual(parameters[1]["access"], "read")

    def test_tensor_view_accepts_static_dynamic_and_mixed_shapes(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@kernel\n"
            "def shapes(\n"
            "    static: TensorView[f32, (4, 8), read],\n"
            "    dynamic: TensorView[f32, (dyn,), read],\n"
            "    mixed: TensorView[f32, (dyn, 4), read],\n"
            ") -> None:\n"
            "    pass\n",
            "tensor_view_shapes.py",
        )

        self.assertIn(
            '!vernon.tensor_view<f32, [4, 8], "read", "device">',
            output,
        )
        self.assertFalse(callable(vd.dyn))
        with self.assertRaisesRegex(CompileError, "shape must be a non-empty tuple"):
            compile_source(
                "from vernon_dsl import *\n@kernel\ndef removed(value: TensorView[f32, 1, read]) -> None:\n    pass\n",
                "removed_tensor_view_rank.py",
            )

    def test_storage_diagnostics_are_explicit(self) -> None:
        with self.assertRaisesRegex(CompileError, "host-runtime owner"):
            compile_source(
                "from vernon_dsl import *\n@kernel\ndef bad(value: TensorStorage[f32]) -> None:\n    pass\n",
                "storage_parameter.py",
            )
        with self.assertRaisesRegex(CompileError, "write-only TensorView"):
            compile_source(
                "from vernon_dsl import *\n"
                "@kernel\n"
                "def bad(value: TensorView[f32, (dyn,), write]) -> f32:\n"
                "    return value[0]\n",
                "write_only_load.py",
            )
        output = compile_source(
            "from vernon_dsl import *\n"
            "@kernel\n"
            "def read(value: TensorView[f32, (dyn, dyn), read]) -> f32:\n"
            "    return value[0, 0]\n",
            "rank_two_view.py",
        )
        self.assertIn('"vernon.load"', output)

    def test_tensor_view_layout_is_not_frontend_specialization_data(self) -> None:
        source = (
            "from vernon_dsl import *\n"
            "@kernel\n"
            "def read(output: TensorView[f32, (1,), write], value: TensorView[f32, (2, dyn), read]) -> None:\n"
            "    output[0] = value[1, 2]\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank_two_view.py"
            path.write_text(source, encoding="utf-8")
            result = Compiler().compile_request(FrontendCompileRequest(path, "read"))

        self.assertIn('"vernon.load"', result.mlir)
        self.assertNotIn("vernon.tensor_shape", result.mlir)
        self.assertNotIn("vernon.tensor_strides", result.mlir)
        self.assertNotIn("vernon.tensor_offset", result.mlir)
        self.assertNotIn("tensor_view_layouts", result.semantic_inputs)

    def test_v3_buffer_public_spelling_is_removed(self) -> None:
        self.assertFalse(hasattr(vd, "Buffer"))
        with self.assertRaisesRegex(CompileError, "unknown DSL type constructor 'Buffer'"):
            compile_source(
                "from vernon_dsl import *\n@kernel\ndef bad(value: Buffer[f32]) -> None:\n    pass\n",
                "removed_buffer.py",
            )

    def test_raw_buffer_is_runtime_interop_not_source_type(self) -> None:
        self.assertFalse(hasattr(vd, "RawBuffer"))
        self.assertEqual(vd.interop.RawBuffer.__module__, "vernon_dsl._runtime.resources")
        with self.assertRaisesRegex(CompileError, "unknown DSL type 'RawBuffer'"):
            compile_source(
                "from vernon_dsl import *\n@kernel\ndef bad(value: RawBuffer) -> None:\n    pass\n",
                "runtime_only_raw_buffer.py",
            )


if __name__ == "__main__":
    unittest.main()
