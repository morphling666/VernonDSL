from __future__ import annotations

import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated
from unittest import mock

import numpy as np
import vernon_dsl as vd
import vernon_dsl._native as native
from vernon_dsl import CompileError, Compiler, compile_source
from vernon_dsl._runtime.binding import (
    _dispatch_borrow_scope,
    _DispatchBorrowLease,
    _NativeBindingCache,
)
from vernon_dsl._runtime.session import RuntimeUnavailableError
from vernon_dsl._runtime.tensor import _logical_collection_shape
from vernon_dsl.compiler import FrontendCompileRequest
from vernon_dsl.frontend.analysis import typed_model_data
from vernon_dsl.host_values import host_abi_layout


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
    def test_native_binding_cache_prepares_only_changed_execution_tokens(self) -> None:
        cache = _NativeBindingCache()
        builder = mock.Mock()
        builder.prepare_host_tensor.side_effect = lambda *_: object()
        parameter = SimpleNamespace(
            slot=0,
            name="amount",
            element_leaves=((native.DATA_F32, 1, 0),),
            shape=(),
        )

        cache.bind_argument(builder, object(), parameter, np.float32(2.0), binding_token=11)
        cache.bind_argument(builder, object(), parameter, np.float32(2.0), binding_token=11)
        cache.bind_argument(builder, object(), parameter, np.float32(5.0), binding_token=12)

        self.assertEqual(builder.prepare_host_tensor.call_count, 2)
        self.assertEqual(builder.prepared_argument.call_count, 3)
        cache.clear()
        cache.bind_argument(builder, object(), parameter, np.float32(5.0), binding_token=12)
        self.assertEqual(builder.prepare_host_tensor.call_count, 3)

    def test_nested_host_layout_uses_one_complete_native_plan(self) -> None:
        with mock.patch.object(native, "_plan_value_abi", wraps=native._plan_value_abi) as planner:
            layout = host_abi_layout(NestedRecord)

        self.assertEqual(planner.call_count, 1)
        self.assertEqual(layout.size, 40)

    def test_logical_collection_shape_uses_value_structure(self) -> None:
        tuple_type = vd.Tuple[vd.i32, vd.f32]
        first = (vd.i32(1), vd.f32(2.0))
        second = (vd.i32(3), vd.f32(4.0))

        self.assertEqual(_logical_collection_shape(first, tuple_type), ())
        self.assertEqual(_logical_collection_shape((first, second), tuple_type), (2,))
        self.assertEqual(
            _logical_collection_shape(((first, second), (second, first)), tuple_type),
            (2, 2),
        )
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

    def test_partial_updates_track_only_changed_backing_ranges(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(5, 4))
        storage.update([1, 3], np.full((2, 4), 2.0, dtype=np.float32))

        one_dimensional = vd.TensorStorage.zeros(dtype=vd.f32, shape=(5,))
        one_dimensional.update([1, 3], np.array([2.0, 3.0], dtype=np.float32))

        alternating_rows = storage.view(shape=(2, 4), strides=(8, 1), offset=4)
        alternating_rows.copy_from_numpy(np.full((2, 4), 3.0, dtype=np.float32))

        unchanged = storage.to_numpy()
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            storage.update(np.array([[0]], dtype=np.int32), np.ones((1, 1, 4), dtype=np.float32))
        with self.assertRaisesRegex(TypeError, "must be integers"):
            storage.update([True, False], np.ones((2, 4), dtype=np.float32))
        np.testing.assert_array_equal(storage.to_numpy(), unchanged)

    def test_residency_uploads_only_coalesced_dirty_ranges(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(5, 4))
        uploads: list[tuple[int, int]] = []
        native_buffer = mock.Mock()
        native_buffer.upload_ranges.side_effect = lambda ranges: uploads.extend(
            (offset, len(data)) for offset, data in ranges
        )
        state = mock.Mock()
        state._native_runtime = object()
        state._rhi_host.create_buffer.return_value = native_buffer
        state._runtime_generation = 7

        with mock.patch("vernon_dsl._runtime.tensor._session_state", return_value=state):
            storage._resident_buffer()
            self.assertEqual(uploads, [(0, 80)])
            uploads.clear()
            storage.update([1, 3], np.full((2, 4), 2.0, dtype=np.float32))
            storage._resident_buffer()

        self.assertEqual(uploads, [(16, 16), (48, 16)])

    def test_partial_host_update_does_not_read_back_gpu_written_storage(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(8,))
        uploads: list[tuple[int, int]] = []
        backing = bytearray(8 * np.dtype(np.float32).itemsize)
        native_buffer = mock.Mock()

        def upload_ranges(ranges: list[tuple[int, bytes]]) -> None:
            for offset, data in ranges:
                uploads.append((offset, len(data)))
                backing[offset : offset + len(data)] = data

        native_buffer.upload_ranges.side_effect = upload_ranges
        native_buffer.download.side_effect = lambda: bytes(backing)
        state = mock.Mock()
        state._native_runtime = object()
        state._rhi_host.create_buffer.return_value = native_buffer
        state._runtime_generation = 7

        with mock.patch("vernon_dsl._runtime.tensor._session_state", return_value=state):
            storage._resident_buffer()
            uploads.clear()
            backing[:] = np.full((8,), 2.0, dtype=np.float32).tobytes()
            storage._mark_device_dirty()
            storage.update([1, 6], np.array([3.0, 4.0], dtype=np.float32))
            native_buffer.download.assert_not_called()
            result = storage.to_numpy()

        self.assertEqual(uploads, [(4, 4), (24, 4)])
        np.testing.assert_array_equal(result, np.array([2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0], dtype=np.float32))

    def test_planned_upload_commits_only_after_completion(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(4,))
        native_buffer = mock.Mock()
        state = mock.Mock()
        state._native_runtime = object()
        state._rhi_host.create_buffer.return_value = native_buffer
        state._runtime_generation = 7

        with mock.patch("vernon_dsl._runtime.tensor._session_state", return_value=state):
            buffer, transaction, uploads = storage._begin_planned_upload()
            self.assertIs(buffer, native_buffer)
            self.assertEqual(uploads, [(0, bytes(storage._array.nbytes))])
            self.assertTrue(storage._dirty_ranges)
            self.assertFalse(storage._device_dirty)
            transaction._commit_planned_state()

        self.assertFalse(storage._dirty_ranges)
        self.assertFalse(storage._device_dirty)
        self.assertIs(storage._native_buffer, native_buffer)

    def test_failed_planned_write_invalidates_unpublished_device_state(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(4,))
        native_buffer = mock.Mock()
        state = mock.Mock()
        state._native_runtime = object()
        state._rhi_host.create_buffer.return_value = native_buffer
        state._runtime_generation = 7

        with mock.patch("vernon_dsl._runtime.tensor._session_state", return_value=state):
            _, upload, _ = storage._begin_planned_upload()
            upload._commit_planned_state()
            write = storage._begin_planned_device_write()
            write._rollback_planned_state()

        self.assertIsNone(storage._native_buffer)
        self.assertEqual(storage._native_generation, -1)
        self.assertFalse(storage._device_dirty)
        self.assertEqual(storage._dirty_ranges.ranges, ((0, storage._array.nbytes),))

    def test_pending_planned_write_never_uploads_host_state(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(4,))
        native_buffer = mock.Mock()
        state = mock.Mock()
        state._native_runtime = object()
        state._rhi_host.create_buffer.return_value = native_buffer
        state._runtime_generation = 7

        with mock.patch("vernon_dsl._runtime.tensor._session_state", return_value=state):
            write = storage._begin_planned_device_write()
            self.assertTrue(storage._planned_device_write_pending())
            self.assertIs(storage._resident_buffer(), native_buffer)
            native_buffer.upload_ranges.assert_not_called()
            write._rollback_planned_state()

        self.assertFalse(storage._planned_device_write_pending())

    def test_fragmented_view_update_preserves_gpu_written_gaps(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(1024,))
        backing = bytearray(1024 * np.dtype(np.float32).itemsize)
        native_buffer = mock.Mock()

        def upload_ranges(ranges: list[tuple[int, bytes]]) -> None:
            for offset, data in ranges:
                backing[offset : offset + len(data)] = data

        native_buffer.upload_ranges.side_effect = upload_ranges
        native_buffer.download.side_effect = lambda: bytes(backing)
        state = mock.Mock()
        state._native_runtime = object()
        state._rhi_host.create_buffer.return_value = native_buffer
        state._runtime_generation = 7

        with mock.patch("vernon_dsl._runtime.tensor._session_state", return_value=state):
            storage._resident_buffer()
            backing[:] = np.full((1024,), 2.0, dtype=np.float32).tobytes()
            storage._mark_device_dirty()
            storage.view(shape=(300,), strides=(2,), access="write").copy_from_numpy(
                np.full((300,), 3.0, dtype=np.float32)
            )
            result = storage.to_numpy()

        expected = np.full((1024,), 2.0, dtype=np.float32)
        expected[:600:2] = 3.0
        np.testing.assert_array_equal(result, expected)

    def test_large_fragmented_view_update_promotes_to_bounded_full_upload(self) -> None:
        element_count = 10_000
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(element_count,))
        backing = bytearray(element_count * np.dtype(np.float32).itemsize)
        uploads: list[list[tuple[int, bytes]]] = []
        native_buffer = mock.Mock()

        def upload_ranges(ranges: list[tuple[int, bytes]]) -> None:
            uploads.append(ranges)
            for offset, data in ranges:
                backing[offset : offset + len(data)] = data

        native_buffer.upload_ranges.side_effect = upload_ranges
        native_buffer.download.side_effect = lambda: bytes(backing)
        state = mock.Mock()
        state._native_runtime = object()
        state._rhi_host.create_buffer.return_value = native_buffer
        state._runtime_generation = 7

        with mock.patch("vernon_dsl._runtime.tensor._session_state", return_value=state):
            storage._resident_buffer()
            uploads.clear()
            backing[:] = np.full((element_count,), 2.0, dtype=np.float32).tobytes()
            storage._mark_device_dirty()
            storage.view(shape=(element_count // 2,), strides=(2,), access="write").copy_from_numpy(
                np.full((element_count // 2,), 3.0, dtype=np.float32)
            )
            storage._resident_buffer()
            result = storage.to_numpy()

        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0][0][0], 0)
        self.assertEqual(len(uploads[0][0][1]), len(backing))
        expected = np.full((element_count,), 2.0, dtype=np.float32)
        expected[::2] = 3.0
        np.testing.assert_array_equal(result, expected)

    def test_metal_buffer_accepts_disjoint_partial_uploads(self) -> None:
        try:
            vd.init(arch=vd.metal)
        except RuntimeUnavailableError as error:
            self.skipTest(f"Metal runtime is unavailable: {error}")
        try:
            storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(1024,))
            native_buffer = storage._resident_buffer()
            storage.update([1, 513], np.array([5.0, 7.0], dtype=np.float32))
            storage._resident_buffer()

            resident = np.frombuffer(native_buffer.download(), dtype=np.float32)
            np.testing.assert_array_equal(resident[[1, 513]], np.array([5.0, 7.0], dtype=np.float32))
        finally:
            vd.init(arch=vd.cpu)

    def test_strided_and_negative_views_project_without_copying(self) -> None:
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        storage = vd.TensorStorage.from_numpy(values)
        interleaved = storage.view(shape=(3, 2), strides=(4, 1), offset=1, access="read")
        reversed_row = storage.view(shape=(4,), strides=(-1,), offset=3, access="read")

        np.testing.assert_array_equal(interleaved.to_numpy(), values[:, 1:3])
        np.testing.assert_array_equal(reversed_row.to_numpy(), values[0, ::-1])
        self.assertEqual(interleaved.layout.byte_strides, (16, 4))
        self.assertEqual(interleaved.layout.byte_offset, 4)

    def test_rank_zero_view_round_trips_scalar_storage(self) -> None:
        storage = vd.TensorStorage.from_numpy(np.array(3.0, dtype=np.float32))
        view = storage.view(shape=(), strides=())

        self.assertEqual(view.shape, ())
        self.assertEqual(view.layout.byte_strides, ())
        self.assertEqual(view.to_numpy().shape, ())
        self.assertEqual(view.to_numpy()[()], 3.0)
        self.assertEqual(view[()], 3.0)

        view[()] = 7.0
        self.assertEqual(storage.to_numpy()[()], 7.0)

    def test_view_validation_rejects_out_of_bounds_and_defers_injectivity(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.i32, shape=(8,))

        with self.assertRaisesRegex(ValueError, "outside its owner"):
            storage.view(shape=(4,), strides=(1,), offset=6)
        writable_alias = storage.view(shape=(2, 2), strides=(1, 1), access="write")
        self.assertEqual(writable_alias.layout.element_strides, (1, 1))

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

    def test_dispatch_borrows_defer_same_dispatch_alias_validation(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(4, 4))
        left = storage.view(shape=(4, 2), strides=(4, 1), offset=0, access="write")
        right = storage.view(shape=(4, 2), strides=(4, 1), offset=2, access="write")
        overlapping = storage.view(shape=(4, 2), strides=(4, 1), offset=1, access="read")

        with _DispatchBorrowLease([("left", left, "write"), ("right", right, "write")]):
            pass
        with _DispatchBorrowLease([("left", left, "write"), ("overlapping", overlapping, "read")]):
            pass

    def test_dispatch_access_cannot_exceed_view_capability(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(4,))
        reader = storage.view(shape=(4,), strides=(1,), access="read")
        writer = storage.view(shape=(4,), strides=(1,), access="write")

        with self.assertRaisesRegex(ValueError, "does not permit writes"):
            _DispatchBorrowLease([("reader", reader, "write")])
        with self.assertRaisesRegex(ValueError, "does not permit reads"):
            _DispatchBorrowLease([("writer", writer, "read")])
        with self.assertRaisesRegex(ValueError, "does not permit writes"):
            reader._with_access("write")
        with self.assertRaisesRegex(ValueError, "does not permit reads"):
            writer._with_access("read")

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

    def test_dispatch_scope_protects_texture_host_access(self) -> None:
        texture = vd.Texture.zeros(shape=(2, 2), usage=("storage", "transfer_source", "transfer_destination"))
        values = np.ones((2, 2, 4), dtype=np.uint8)

        with _dispatch_borrow_scope([("input", texture, "read")]):
            np.testing.assert_array_equal(texture.download(), np.zeros_like(values))
            with self.assertRaisesRegex(RuntimeError, "host mutation"):
                texture.upload(values)
            with _dispatch_borrow_scope([("second_input", texture, "read")]):
                pass

        with _dispatch_borrow_scope([("output", texture, "write")]):
            with self.assertRaisesRegex(RuntimeError, "host reads"):
                texture.download()
            with self.assertRaisesRegex(RuntimeError, "host mutation"):
                texture.upload(values)
            with self.assertRaisesRegex(RuntimeError, "outstanding device borrow"):
                with _dispatch_borrow_scope([("conflicting_output", texture, "write")]):
                    pass

        texture.upload(values)
        np.testing.assert_array_equal(texture.download(), values)

    def test_dispatch_lease_retains_borrows_until_release(self) -> None:
        storage = vd.TensorStorage.zeros(dtype=vd.f32, shape=(4,))
        lease = _DispatchBorrowLease([("output", storage, "write")])
        with self.assertRaisesRegex(RuntimeError, "host mutation"):
            storage.copy_from_numpy(np.ones((4,), dtype=np.float32))
        lease.release()
        storage.copy_from_numpy(np.ones((4,), dtype=np.float32))

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

    def test_texture_view_borrows_are_subresource_aware(self) -> None:
        texture = vd.Texture.zeros(shape=(8, 8), mip_levels=2)
        mip0 = texture.view(base_mip_level=0, mip_level_count=1)
        mip1 = texture.view(base_mip_level=1, mip_level_count=1)

        with _dispatch_borrow_scope([("mip0", mip0, "write")]):
            with _dispatch_borrow_scope([("mip1", mip1, "write")]):
                pass
            with self.assertRaisesRegex(RuntimeError, "outstanding device borrow"):
                with _dispatch_borrow_scope([("same_mip", mip0, "read")]):
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

        with _DispatchBorrowLease([("position", position, "write"), ("uv", uv, "write")]):
            pass

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

        replacement = StorageVertex(vd.Vector([9.0, 8.0, 7.0]), vd.f64(6.0), vd.Vector([0.5, 0.25]))
        storage.update_values(1, replacement)
        updated = storage.to_values()[1]
        np.testing.assert_array_equal(updated.position, replacement.position)
        self.assertEqual(updated.weight, replacement.weight)

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

    def test_tensor_view_accepts_rank_zero_static_dynamic_and_mixed_shapes(
        self,
    ) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@kernel\n"
            "def shapes(\n"
            "    scalar: TensorView[f32, (), read_write],\n"
            "    static: TensorView[f32, (4, 8), read],\n"
            "    dynamic: TensorView[f32, (dyn,), read],\n"
            "    mixed: TensorView[f32, (dyn, 4), read],\n"
            ") -> None:\n"
            "    scalar[()] = scalar[()] + 1.0\n",
            "tensor_view_shapes.py",
        )

        self.assertIn(
            '!vernon.tensor_view<f32, [], "read_write", "device">',
            output,
        )
        self.assertIn(
            '!vernon.tensor_view<f32, [4, 8], "read", "device">',
            output,
        )
        self.assertFalse(callable(vd.dyn))
        with self.assertRaisesRegex(CompileError, "shape must be a tuple"):
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

    def test_shape_lowers_to_get_shape_for_tensor_and_tensor_view(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@kernel\n"
            "def extents(\n"
            "    output: TensorView[f32, (dyn, dyn), write],\n"
            "    tile: Tensor[f32, (2, 5)],\n"
            "    vec: Vector[f32, 3],\n"
            "    mat: Matrix[f32, 2, 4],\n"
            ") -> None:\n"
            "    output[output.shape[0], tile.shape[1]] = f32(vec.shape[0]) + f32(mat.shape[1])\n",
            "tensor_and_view_shape.py",
        )
        self.assertEqual(output.count('"vernon.get_shape"'), 4)
        self.assertIn(
            '!vernon.tensor_view<f32, [-1, -1], "write", "device">) -> tensor<2xi32>',
            output,
        )
        self.assertIn("tensor<2x5xf32>) -> tensor<2xi32>", output)
        self.assertIn("tensor<3xf32>) -> tensor<1xi32>", output)
        self.assertIn("tensor<2x4xf32>) -> tensor<2xi32>", output)
        self.assertNotIn("cannot load through a write-only TensorView", output)

    def test_rank_zero_view_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(CompileError, "shape requires rank >= 1"):
            compile_source(
                "from vernon_dsl import *\n"
                "@kernel\n"
                "def bad(value: TensorView[f32, (), read_write]) -> None:\n"
                "    extent = value.shape\n",
                "rank_zero_shape.py",
            )

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
        self.assertEqual(vd.interop.RawBuffer.__module__, "vernon_dsl._runtime.tensor")
        with self.assertRaisesRegex(CompileError, "unknown DSL type 'RawBuffer'"):
            compile_source(
                "from vernon_dsl import *\n@kernel\ndef bad(value: RawBuffer) -> None:\n    pass\n",
                "runtime_only_raw_buffer.py",
            )


@vd.kernel(workgroup_size=(1, 1, 1))
def fill_with_extents(
    output: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    tile: vd.Tensor[vd.f32, (2, 5)],
    vec: vd.Vector[vd.f32, 3],
    mat: vd.Matrix[vd.f32, 2, 4],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[1], gid[0]] = (
        vd.f32(output.shape[0]) * 1000.0
        + vd.f32(tile.shape[1]) * 100.0
        + vd.f32(vec.shape[0]) * 10.0
        + vd.f32(mat.shape[1])
    )


class TensorViewShapeRuntimeTests(unittest.TestCase):
    @staticmethod
    def _runtime_available(architecture: object) -> bool:
        try:
            vd.init(arch=architecture)  # type: ignore[arg-type]
        except RuntimeError:
            vd.init(arch=vd.cpu)
            return False
        return True

    def _available_compute_backends(self) -> list[object]:
        backends: list[object] = [vd.cpu]
        for architecture in (vd.cuda, vd.vulkan, vd.directx, vd.metal, vd.opengl, vd.opengles):
            if self._runtime_available(architecture):
                backends.append(architecture)
        return backends

    def test_tensor_and_tensor_view_shape_read_static_and_descriptor_extents(
        self,
    ) -> None:
        expected = np.full((3, 4), 3534.0, dtype=np.float32)
        tile = np.zeros((2, 5), dtype=np.float32)
        vec = np.zeros(3, dtype=np.float32)
        mat = np.zeros((2, 4), dtype=np.float32)
        for backend in self._available_compute_backends():
            with self.subTest(backend=getattr(backend, "name", backend)):
                vd.init(arch=backend)  # type: ignore[arg-type]
                output = vd.storage.zeros(dtype=vd.f32, shape=(3, 4))
                fill_with_extents(output, tile, vec, mat, grid=(4, 3, 1))
                np.testing.assert_array_equal(output.to_numpy(), expected)
        vd.init(arch=vd.cpu)


if __name__ == "__main__":
    unittest.main()
