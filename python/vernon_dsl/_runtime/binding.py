from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from .._dtypes import NUMPY_DTYPE_BY_SCALAR
from ..host_values import host_abi_layout, pack_host_value
from .resource_common import _session_state
from .sampler import SamplerState
from .tensor import RawBuffer, TensorStorage, TensorView, _borrow_ranges_may_overlap, _checked_access
from .texture import TextureView, _TextureResource


class _PersistentBindingTable:
    """Owns native transactional persistent bindings for one executable."""

    def __init__(self) -> None:
        self._native_pipeline: Any | None = None
        self._instance: Any | None = None
        self._transactions = threading.local()
        self._lock = threading.RLock()

    def _native_instance(self, native_pipeline: Any) -> Any:
        with self._lock:
            if self._instance is None or self._native_pipeline is not native_pipeline:
                self._native_pipeline = native_pipeline
                self._instance = native_pipeline.program_instance()
            return self._instance

    def clear(self) -> None:
        with self._lock:
            self._native_pipeline = None
            self._instance = None

    @contextmanager
    def invocation(self, native_pipeline: Any) -> Iterator[Any]:
        transaction = self._native_instance(native_pipeline).begin_invocation()
        if getattr(self._transactions, "current", None) is not None:
            raise RuntimeError("persistent binding invocations cannot be nested on one thread")
        self._transactions.current = transaction
        try:
            yield transaction.builder
        except BaseException:
            transaction.rollback()
            raise
        else:
            transaction.commit()
        finally:
            self._transactions.current = None

    @property
    def telemetry(self) -> dict[str, int]:
        with self._lock:
            return {} if self._instance is None else dict(self._instance.telemetry)

    def _bind(
        self,
        builder: Any,
        parameter: Any,
        token: tuple[Any, ...],
        prepare: Any,
        *,
        upload_bytes: int = 0,
        upload_ranges: int = 0,
        eager_upload: bool = False,
    ) -> None:
        transaction = getattr(self._transactions, "current", None)
        if transaction is None:
            raise RuntimeError("binding update requires an active invocation transaction")
        transaction.bind(parameter.slot, token, prepare, upload_bytes, upload_ranges, eager_upload)

    def bind_argument(
        self,
        builder: Any,
        native_pipeline: Any,
        parameter: Any,
        value: Any,
        *,
        host_value: bool = False,
        annotation: Any | None = None,
        binding_token: int | None = None,
    ) -> Any:
        state = _session_state()
        if isinstance(value, (TensorStorage, TensorView)):
            if state._architecture == state.cpu or host_value:
                array = value._native_host_array()
                token = (
                    "host-resource",
                    int(array.ctypes.data),
                    array.dtype.str,
                    tuple(array.shape),
                    tuple(array.strides),
                )
                return self._bind(
                    builder,
                    parameter,
                    token,
                    lambda: builder.prepare_host_tensor(parameter.slot, array),
                )
            if state._rhi_host is None:
                raise RuntimeError("device Tensor arguments require a GPU RHI host")
            owner = value.owner if isinstance(value, TensorView) else value
            dirty_tracker = getattr(owner, "_dirty_ranges", None)
            dirty_ranges = tuple(dirty_tracker.ranges) if dirty_tracker is not None else ()
            buffer = value._resident_buffer()
            layout = value.layout
            token = (
                "rhi-tensor",
                id(buffer),
                parameter.access,
                tuple(value.shape),
                tuple(layout.byte_strides),
                layout.byte_offset,
            )
            return self._bind(
                builder,
                parameter,
                token,
                lambda: builder.prepare_rhi_tensor(
                    parameter.slot,
                    buffer,
                    parameter.access,
                    list(value.shape),
                    list(layout.byte_strides),
                    layout.byte_offset,
                ),
                upload_bytes=sum(end - begin for begin, end in dirty_ranges),
                upload_ranges=len(dirty_ranges),
                eager_upload=True,
            )
        if isinstance(value, _TextureResource):
            if state._architecture == state.cpu:
                raise TypeError("CPU kernels do not support Texture arguments")
            if state._rhi_host is None:
                raise RuntimeError("Texture arguments require a GPU RHI host")
            owner = value.owner if isinstance(value, TextureView) else value
            dirty_mips = tuple(owner._host_dirty_mips)
            upload_bytes = sum(owner._mip_arrays[level].nbytes for level in dirty_mips)
            view = value._resident_view()
            return self._bind(
                builder,
                parameter,
                ("rhi-texture", id(view)),
                lambda: builder.prepare_rhi_texture(parameter.slot, view),
                upload_bytes=upload_bytes,
                upload_ranges=len(dirty_mips),
                eager_upload=True,
            )

        execution_token = ("execution-value", binding_token) if binding_token is not None else None
        value_type = type(value)
        struct_type = (
            annotation
            if isinstance(annotation, type) and getattr(annotation, "__vernon_dsl__", (None, {}))[0] == "struct"
            else (value_type if getattr(value_type, "__vernon_dsl__", (None, {}))[0] == "struct" else None)
        )
        canonical_annotation = annotation if annotation is not None else struct_type
        if canonical_annotation is not None:
            layout = host_abi_layout(canonical_annotation)
            host_array = np.empty((), dtype=layout.dtype)
            host_array[()] = pack_host_value(canonical_annotation, value, parameter.name)
        else:
            numpy_dtypes = {
                state._native.DATA_BOOL: NUMPY_DTYPE_BY_SCALAR["bool"],
                state._native.DATA_I32: NUMPY_DTYPE_BY_SCALAR["i32"],
                state._native.DATA_U32: NUMPY_DTYPE_BY_SCALAR["u32"],
                state._native.DATA_F16: NUMPY_DTYPE_BY_SCALAR["f16"],
                state._native.DATA_F32: NUMPY_DTYPE_BY_SCALAR["f32"],
                state._native.DATA_F64: NUMPY_DTYPE_BY_SCALAR["f64"],
            }
            leaves = tuple(parameter.element_leaves)
            dtype = numpy_dtypes.get(leaves[0][0]) if len(leaves) == 1 and leaves[0][2] == 0 else None
            if dtype is None:
                raise TypeError(f"aggregate parameter {parameter.name!r} requires canonical TensorStorage")
            source = np.asarray(value, dtype=dtype)
            whole_value = not parameter.shape and leaves[0][1] > 1
            if whole_value:
                packed = np.ascontiguousarray(source)
                if packed.size != leaves[0][1] or packed.nbytes != parameter.element_byte_size:
                    raise ValueError(
                        f"parameter {parameter.name!r} expects {leaves[0][1]} scalar values, got {packed.size}"
                    )
                host_array = np.empty(
                    (),
                    dtype=np.dtype([("_bytes", np.uint8, (parameter.element_byte_size,))]),
                )
                host_array["_bytes"] = packed.view(np.uint8).reshape(-1)
            else:
                host_array = source
            if not whole_value and tuple(host_array.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"parameter {parameter.name!r} expects shape {tuple(parameter.shape)}, "
                    f"got {tuple(host_array.shape)}"
                )
        token = execution_token or (
            "host-value",
            host_array.dtype.str,
            tuple(host_array.shape),
            host_array.tobytes(),
        )
        return self._bind(
            builder,
            parameter,
            token,
            lambda: builder.prepare_host_tensor(parameter.slot, host_array),
            upload_bytes=int(host_array.nbytes),
            upload_ranges=1,
        )

    def bind_sampler(self, builder: Any, native_pipeline: Any, parameter: Any, sampler: SamplerState) -> None:
        resident = sampler._resident_sampler()
        self._bind(
            builder,
            parameter,
            ("rhi-sampler", id(resident)),
            lambda: builder.prepare_rhi_sampler(parameter.slot, resident),
        )

    def bind_render_pass_control(self, slot: int, token: tuple[Any, ...], control: Any) -> None:
        transaction = getattr(self._transactions, "current", None)
        if transaction is None:
            raise RuntimeError("Program control update requires an active invocation transaction")
        transaction.bind_render_pass(slot, token, control)

    def bind_draw_command_control(self, slot: int, token: tuple[Any, ...], control: Any) -> None:
        transaction = getattr(self._transactions, "current", None)
        if transaction is None:
            raise RuntimeError("Program control update requires an active invocation transaction")
        transaction.bind_draw_command(slot, token, control)

    def bind_dynamic_state_control(self, slot: int, token: tuple[Any, ...], control: Any) -> None:
        transaction = getattr(self._transactions, "current", None)
        if transaction is None:
            raise RuntimeError("Program control update requires an active invocation transaction")
        transaction.bind_dynamic_state(slot, token, control)

    def forward(self) -> None:
        transaction = getattr(self._transactions, "current", None)
        if transaction is None:
            raise RuntimeError("Program forward requires an active invocation transaction")
        transaction.forward()


def _normalize_dispatch_borrows(
    borrows: list[tuple[str, TensorStorage | RawBuffer | TensorView | _TextureResource, str]],
) -> list[tuple[str, RawBuffer | TensorView | _TextureResource, str]]:
    normalized: list[tuple[str, RawBuffer | TensorView | _TextureResource, str]] = []
    for name, value, access in borrows:
        access = _checked_access(access)
        resource = value._full_view("read_write") if isinstance(value, TensorStorage) else value
        if isinstance(resource, TensorView):
            if access in {"read", "read_write"} and resource.access == "write":
                raise ValueError(f"TensorView argument '{name}' does not permit reads")
            if access in {"write", "read_write"} and resource.access == "read":
                raise ValueError(f"TensorView argument '{name}' does not permit writes")
        normalized.append((name, resource, access))
    return normalized


def _dispatch_borrow_owner(
    resource: RawBuffer | TensorView | _TextureResource,
) -> TensorStorage | RawBuffer | _TextureResource:
    return resource.owner if isinstance(resource, (TensorView, TextureView)) else resource


def _dispatch_borrows_overlap(
    left: RawBuffer | TensorView | _TextureResource,
    right: RawBuffer | TensorView | _TextureResource,
) -> bool:
    if isinstance(left, TensorView) and isinstance(right, TensorView):
        return _borrow_ranges_may_overlap(left, right)
    if isinstance(left, TextureView) and isinstance(right, TextureView) and left.owner is right.owner:
        left_mip_end = left._base_mip_level + left._mip_level_count
        right_mip_end = right._base_mip_level + right._mip_level_count
        left_layer_end = left._base_array_layer + left._array_layer_count
        right_layer_end = right._base_array_layer + right._array_layer_count
        return (
            left._base_mip_level < right_mip_end
            and right._base_mip_level < left_mip_end
            and left._base_array_layer < right_layer_end
            and right._base_array_layer < left_layer_end
            and not left._aspects.isdisjoint(right._aspects)
        )
    return True


class _DispatchBorrowLease:
    """An acquired resource borrow released only after dispatch completion."""

    def __init__(
        self,
        borrows: list[tuple[str, TensorStorage | RawBuffer | TensorView | _TextureResource, str]],
    ):
        self._owners: list[TensorStorage | RawBuffer | _TextureResource] = []
        self._token: object | None = None
        normalized = _normalize_dispatch_borrows(borrows)
        self._owners = sorted({_dispatch_borrow_owner(resource) for _, resource, _ in normalized}, key=id)
        self._token = object()
        for owner in self._owners:
            owner._borrow_lock.acquire()
        try:
            for name, resource, access in normalized:
                owner = _dispatch_borrow_owner(resource)
                for _, active_resource, active_access in owner._active_borrows:
                    if access == active_access == "read":
                        continue
                    if isinstance(active_resource, (RawBuffer, TensorView, _TextureResource)) and (
                        _dispatch_borrows_overlap(resource, active_resource)
                    ):
                        raise RuntimeError(f"dispatch argument '{name}' conflicts with an outstanding device borrow")
            for _, resource, access in normalized:
                _dispatch_borrow_owner(resource)._active_borrows.append((self._token, resource, access))
        finally:
            for owner in reversed(self._owners):
                owner._borrow_lock.release()

    def release(self) -> None:
        token = self._token
        if token is None:
            return
        self._token = None
        for owner in self._owners:
            owner._borrow_lock.acquire()
        try:
            for owner in self._owners:
                owner._active_borrows[:] = [active for active in owner._active_borrows if active[0] is not token]
        finally:
            for owner in reversed(self._owners):
                owner._borrow_lock.release()

    def release_writes(self) -> None:
        token = self._token
        if token is None:
            return
        for owner in self._owners:
            owner._borrow_lock.acquire()
        try:
            for owner in self._owners:
                owner._active_borrows[:] = [
                    active for active in owner._active_borrows if active[0] is not token or active[2] == "read"
                ]
        finally:
            for owner in reversed(self._owners):
                owner._borrow_lock.release()

    def __enter__(self) -> _DispatchBorrowLease:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()


@contextmanager
def _dispatch_borrow_scope(
    borrows: list[tuple[str, TensorStorage | RawBuffer | TensorView | _TextureResource, str]],
) -> Iterator[None]:
    with _DispatchBorrowLease(borrows):
        yield
