from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Annotated, Any, get_args, get_origin
from weakref import WeakValueDictionary

import numpy as np

from .. import _native
from .._dtypes import NUMPY_DTYPE_BY_SCALAR
from ..host_values import (
    HostAbiLayout,
    TangentLayout,
    host_abi_layout,
    host_scalar_shape,
    pack_host_value,
    tangent_layout,
    unpack_host_value,
    unpack_tangent_value,
)
from ..types import TypeExpr, _Scalar
from .residency import _BufferRegion, _ResourceControlBlock, _scoped_host_access
from .session import _InvocationContext, cpu

_ACCESS_MODES = frozenset({"read", "write", "read_write"})
_VALUE_FIELD = "__value"


def _storage_array_dtype(layout: HostAbiLayout | TangentLayout) -> np.dtype[Any]:
    if layout.dtype.subdtype is None:
        return layout.dtype
    return np.dtype(
        {
            "names": (_VALUE_FIELD,),
            "formats": (layout.dtype,),
            "offsets": (0,),
            "itemsize": layout.size,
        }
    )


def _is_logical_collection(value: Any) -> bool:
    return isinstance(value, (list, tuple)) or (isinstance(value, np.ndarray) and value.ndim > 0)


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple[int, ...]
    byte_strides: tuple[int, ...]
    byte_offset: int = 0
    components: tuple[int, ...] | None = None
    element_strides: tuple[int, ...] = ()
    element_offset: int = 0


def _checked_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(shape, tuple) or any(
        not isinstance(extent, int) or isinstance(extent, bool) or extent < 0 for extent in shape
    ):
        raise ValueError("TensorView shape must contain non-negative integer extents")
    return shape


def _checked_access(access: str) -> str:
    if access not in _ACCESS_MODES:
        raise ValueError("TensorView access must be 'read', 'write', or 'read_write'")
    return access


def _address_range(shape: tuple[int, ...], strides: tuple[int, ...], offset: int) -> tuple[int, int] | None:
    if any(extent == 0 for extent in shape):
        return None
    low = offset
    high = offset
    for extent, stride in zip(shape, strides, strict=True):
        span = (extent - 1) * stride
        low += min(span, 0)
        high += max(span, 0)
    return low, high


def _matches_logical_value(value: Any, element_type: Any) -> bool:
    if get_origin(element_type) is Annotated:
        element_type = get_args(element_type)[0]
    if element_type is int:
        element_type = _Scalar("i32")
    elif element_type is float:
        element_type = _Scalar("f32")
    if isinstance(element_type, _Scalar):
        return np.isscalar(value) and not isinstance(value, (str, bytes))
    if isinstance(element_type, type):
        return getattr(element_type, "__vernon_dsl__", (None, {}))[0] == "struct" and isinstance(value, element_type)
    if not isinstance(element_type, TypeExpr):
        return False
    if element_type.name == "Tuple":
        return (
            isinstance(value, tuple)
            and len(value) == len(element_type.arguments)
            and all(
                _matches_logical_value(item, item_type)
                for item, item_type in zip(value, element_type.arguments, strict=True)
            )
        )
    if element_type.name == "Tensor" and len(element_type.arguments) == 2:
        item_type, shape = element_type.arguments
    elif element_type.name == "Vector" and len(element_type.arguments) == 2:
        item_type, shape = element_type.arguments[0], (element_type.arguments[1],)
    elif element_type.name == "Matrix" and len(element_type.arguments) == 3:
        item_type, shape = element_type.arguments[0], tuple(element_type.arguments[1:])
    else:
        return False
    if not isinstance(shape, tuple) or any(not isinstance(extent, int) or extent <= 0 for extent in shape):
        return False

    def matches_shaped(current: Any, dimensions: tuple[int, ...]) -> bool:
        if not dimensions:
            return _matches_logical_value(current, item_type)
        if not _is_logical_collection(current):
            return False
        return len(current) == dimensions[0] and all(matches_shaped(item, dimensions[1:]) for item in current)

    return matches_shaped(value, shape)


def _logical_collection_shape(values: Any, element_type: Any) -> tuple[int, ...]:
    if _matches_logical_value(values, element_type):
        return ()
    if not _is_logical_collection(values):
        raise TypeError(f"{values!r} is not a valid logical Value or Value collection")
    if len(values) == 0:
        return (0,)
    child_shapes = tuple(_logical_collection_shape(value, element_type) for value in values)
    if any(shape != child_shapes[0] for shape in child_shapes[1:]):
        raise ValueError("logical Values must form a rectangular storage shape")
    return (len(values), *child_shapes[0])


def _coalesced_byte_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for begin, end in sorted(ranges):
        if begin >= end:
            continue
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((begin, end))
    return result


def _intersect_byte_ranges(
    left: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    right: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> list[tuple[int, int]]:
    left_ranges = _coalesced_byte_ranges(list(left))
    right_ranges = _coalesced_byte_ranges(list(right))
    intersections: list[tuple[int, int]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left_ranges) and right_index < len(right_ranges):
        left_begin, left_end = left_ranges[left_index]
        right_begin, right_end = right_ranges[right_index]
        begin = max(left_begin, right_begin)
        end = min(left_end, right_end)
        if begin < end:
            intersections.append((begin, end))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return intersections


def _subtract_byte_ranges(
    ranges: list[tuple[int, int]],
    removed: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    source = _coalesced_byte_ranges(ranges)
    exclusions = _coalesced_byte_ranges(removed)
    remaining: list[tuple[int, int]] = []
    exclusion_index = 0
    for begin, end in source:
        cursor = begin
        while exclusion_index < len(exclusions) and exclusions[exclusion_index][1] <= cursor:
            exclusion_index += 1
        current = exclusion_index
        while current < len(exclusions) and exclusions[current][0] < end:
            removed_begin, removed_end = exclusions[current]
            if cursor < removed_begin:
                remaining.append((cursor, min(removed_begin, end)))
            cursor = max(cursor, removed_end)
            if cursor >= end:
                break
            current += 1
        if cursor < end:
            remaining.append((cursor, end))
        exclusion_index = current
    return remaining


class _DirtyRangeSet:
    """Python resource adapter for runtime-owned dirty byte tracking."""

    def __init__(self, byte_size: int, *, dirty: bool = False):
        self._native = _native._DirtyRangeSet(byte_size, dirty)

    @property
    def ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._native.ranges)

    def mark(self, ranges: list[tuple[int, int]], *, allow_full: bool) -> None:
        self._native.mark(ranges, allow_full)

    def should_promote_full(self, ranges: list[tuple[int, int]]) -> bool:
        return self._native.should_promote_full(ranges)

    def mark_all(self) -> None:
        self._native.mark_all()

    def clear(self) -> None:
        self._native.clear()

    def discard(self, ranges: list[tuple[int, int]]) -> None:
        remaining = _subtract_byte_ranges(list(self.ranges), ranges)
        self.clear()
        self.mark(remaining, allow_full=False)

    def __bool__(self) -> bool:
        return bool(self._native)


@dataclass
class _BufferCoherence:
    dirty_ranges: _DirtyRangeSet
    device_dirty_ranges: list[tuple[int, int]] = field(default_factory=list)
    device_dirty: bool = False


def _buffer_bytes(owner: Any) -> np.ndarray:
    return owner._array.reshape(-1).view(np.uint8)


def _synchronize_buffer(owner: Any, region: _BufferRegion) -> None:
    control = owner._control
    reservation = None
    with control.lock:
        coherence = control.coherence
        residency = control.residencies.latest()
        if not coherence.device_dirty or residency is None:
            return
        downloads = _intersect_byte_ranges(coherence.device_dirty_ranges, region.ranges)
        if not downloads:
            return
        io_region = _BufferRegion(tuple(downloads))
        reservation = control.reserve_io(io_region, "write")
        handle = residency.handle
    try:
        downloaded = handle.download_ranges(downloads)
        with control.lock:
            bytes_view = _buffer_bytes(owner)
            for begin, data in downloaded:
                bytes_view[begin : begin + len(data)] = np.frombuffer(data, dtype=np.uint8)
            coherence.device_dirty_ranges = _subtract_byte_ranges(coherence.device_dirty_ranges, downloads)
            coherence.device_dirty = bool(coherence.device_dirty_ranges)
            if not coherence.device_dirty:
                control.host_version = control.version
    finally:
        with control.lock:
            control.release_io(reservation)


def _materialize_buffer(
    owner: Any,
    context: _InvocationContext,
    region: _BufferRegion,
    access: str,
) -> Any:
    state = context.session
    if state.rhi_host is None:
        raise RuntimeError("device buffer residency requires a GPU RHI host")
    control = owner._control
    reservation = None
    retry = False
    with control.lock:
        coherence = control.coherence
        residency = control.residencies.get(state)
        if residency is not None and not coherence.device_dirty and (access == "write" or not coherence.dirty_ranges):
            return residency.handle
        latest = control.residencies.latest()
        migration = coherence.device_dirty and (residency is None or residency is not latest)
        create = residency is None
        if create or migration:
            io_region = owner._region
            upload_ranges = list(owner._region.ranges)
            download_ranges = list(coherence.device_dirty_ranges) if migration else []
        else:
            io_region = region
            download_ranges = []
            upload_ranges = (
                _intersect_byte_ranges(coherence.dirty_ranges.ranges, region.ranges)
                if access in {"read", "read_write"}
                else []
            )
        if not create and not migration and not upload_ranges:
            return residency.handle
        expected_state = (
            residency.handle if residency is not None else None,
            latest.handle if latest is not None else None,
            control.version,
            tuple(coherence.device_dirty_ranges),
            tuple(coherence.dirty_ranges.ranges),
        )
        reservation = control.reserve_io(io_region, "write")
        current_residency = control.residencies.get(state)
        current_latest = control.residencies.latest()
        current_state = (
            current_residency.handle if current_residency is not None else None,
            current_latest.handle if current_latest is not None else None,
            control.version,
            tuple(coherence.device_dirty_ranges),
            tuple(coherence.dirty_ranges.ranges),
        )
        retry = current_state != expected_state
        if retry:
            control.release_io(reservation)
            reservation = None
        else:
            generation = control.version
            host_bytes = bytes(_buffer_bytes(owner))
            source_handle = latest.handle if migration and latest is not None else None
            target_handle = residency.handle if residency is not None else None
    if retry:
        return _materialize_buffer(owner, context, region, access)
    try:
        if download_ranges:
            if source_handle is None:
                raise RuntimeError("device-dirty buffer has no authoritative residency")
            merged = bytearray(host_bytes)
            for begin, data in source_handle.download_ranges(download_ranges):
                merged[begin : begin + len(data)] = data
            host_bytes = bytes(merged)
        if target_handle is None:
            target_handle = state.rhi_host.create_buffer(owner._array.nbytes)
        uploads = [(begin, host_bytes[begin:end]) for begin, end in upload_ranges]
        if uploads:
            target_handle.upload_ranges(uploads)
        with control.lock:
            if residency is None:
                residency = control.residencies.replace(state, target_handle)
            if download_ranges:
                bytes_view = _buffer_bytes(owner)
                bytes_view[:] = np.frombuffer(host_bytes, dtype=np.uint8)
                coherence.device_dirty_ranges = _subtract_byte_ranges(
                    coherence.device_dirty_ranges,
                    download_ranges,
                )
                coherence.device_dirty = bool(coherence.device_dirty_ranges)
            coherence.dirty_ranges.discard(upload_ranges)
            residency.version = max(residency.version, generation)
            if not coherence.device_dirty:
                control.host_version = generation
            return residency.handle
    finally:
        with control.lock:
            control.release_io(reservation)


def _array_byte_ranges(array: np.ndarray, allocation: np.ndarray) -> list[tuple[int, int]]:
    if not array.size:
        return []
    base_offset = int(array.ctypes.data) - int(allocation.ctypes.data)
    block_size = array.dtype.itemsize
    split = array.ndim
    for dimension in range(array.ndim - 1, -1, -1):
        if array.strides[dimension] != block_size:
            break
        block_size *= array.shape[dimension]
        split = dimension
    if split == 0:
        return [(base_offset, base_offset + block_size)]
    ranges = []
    for index in np.ndindex(array.shape[:split]):
        offset = base_offset + sum(
            component * stride for component, stride in zip(index, array.strides[:split], strict=True)
        )
        ranges.append((offset, offset + block_size))
    return _coalesced_byte_ranges(ranges)


class TensorStorage:
    """Host owner of a dense row-major allocation with one canonical element layout."""

    def __init__(
        self,
        array: np.ndarray,
        *,
        element_type: Any | None = None,
        element_layout: HostAbiLayout | TangentLayout | None = None,
    ):
        if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
            raise ValueError("TensorStorage requires a contiguous NumPy array")
        if array.dtype not in NUMPY_DTYPE_BY_SCALAR.values() and element_type is None:
            raise TypeError(f"unsupported TensorStorage dtype {array.dtype}")
        if element_layout is not None and array.dtype != _storage_array_dtype(element_layout):
            raise TypeError("TensorStorage array dtype does not match its canonical element ABI")
        self._array = array
        self._element_type = element_type
        self._element_layout = element_layout
        self._element_alignment = element_layout.alignment if element_layout is not None else array.dtype.itemsize
        self._control = _ResourceControlBlock(_BufferCoherence(_DirtyRangeSet(array.nbytes, dirty=True)))
        self._region = _BufferRegion(((0, array.nbytes),))
        self._full_views: WeakValueDictionary[str, TensorView] = WeakValueDictionary()

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr(cls.__name__, arguments)

    @staticmethod
    def _dtype(dtype: _Scalar) -> np.dtype[Any]:
        if not isinstance(dtype, _Scalar) or dtype.name not in NUMPY_DTYPE_BY_SCALAR:
            raise TypeError("dtype must be a Vernon scalar type")
        return NUMPY_DTYPE_BY_SCALAR[dtype.name]

    @staticmethod
    def _storage_dtype(dtype: Any) -> tuple[np.dtype[Any], HostAbiLayout | None]:
        if isinstance(dtype, _Scalar):
            return TensorStorage._dtype(dtype), None
        layout = host_abi_layout(dtype)
        return _storage_array_dtype(layout), layout

    @classmethod
    def zeros_like(cls, value: TensorStorage | TensorView) -> TensorStorage:
        if not isinstance(value, (TensorStorage, TensorView)):
            raise TypeError("zeros_like() requires TensorStorage or TensorView")
        owner = value.owner if isinstance(value, TensorView) else value
        if not isinstance(owner, TensorStorage):
            raise TypeError("Transients require TensorStorage-backed values")
        return cls.zeros(dtype=owner.dtype, shape=tuple(value.shape))

    @classmethod
    def zeros(cls, *, dtype: Any, shape: tuple[int, ...]) -> TensorStorage:
        numpy_dtype, layout = cls._storage_dtype(dtype)
        return cls(
            np.zeros(shape, dtype=numpy_dtype, order="C"),
            element_type=dtype if layout is not None else None,
            element_layout=layout,
        )

    @classmethod
    def empty_like(cls, value: TensorStorage | TensorView) -> TensorStorage:
        if not isinstance(value, (TensorStorage, TensorView)):
            raise TypeError("empty_like() requires TensorStorage or TensorView")
        owner = value.owner if isinstance(value, TensorView) else value
        if not isinstance(owner, TensorStorage):
            raise TypeError("Transients require TensorStorage-backed values")
        return cls.empty(dtype=owner.dtype, shape=tuple(value.shape))

    @classmethod
    def empty(cls, *, dtype: Any, shape: tuple[int, ...]) -> TensorStorage:
        numpy_dtype, layout = cls._storage_dtype(dtype)
        return cls(
            np.empty(shape, dtype=numpy_dtype, order="C"),
            element_type=dtype if layout is not None else None,
            element_layout=layout,
        )

    @classmethod
    def _tangent_zeros(cls, layout: TangentLayout, shape: tuple[int, ...]) -> TensorStorage:
        return cls(
            np.zeros(shape, dtype=_storage_array_dtype(layout), order="C"),
            element_type=layout.primal_element_type,
            element_layout=layout,
        )

    @classmethod
    def tangent_zeros(cls, *, dtype: Any, shape: tuple[int, ...]) -> TensorStorage:
        return cls._tangent_zeros(tangent_layout(dtype), shape)

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> TensorStorage:
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a NumPy ndarray")
        if array.dtype not in NUMPY_DTYPE_BY_SCALAR.values():
            raise TypeError(f"unsupported TensorStorage dtype {array.dtype}")
        return cls(np.array(array, copy=True, order="C"))

    @classmethod
    def from_values(cls, values: Any, *, dtype: Any) -> TensorStorage:
        storage = cls.zeros(dtype=dtype, shape=_logical_collection_shape(values, dtype))
        storage.copy_from_values(values)
        return storage

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._element_layout.dtype if self._element_layout is not None else self._array.dtype

    @property
    def element_layout(self) -> HostAbiLayout | TangentLayout | None:
        return self._element_layout

    @property
    def layout(self) -> TensorLayout:
        element_strides = tuple(stride // self.dtype.itemsize for stride in self._array.strides)
        return TensorLayout(
            self.shape,
            self._array.strides,
            element_strides=element_strides,
        )

    def field(self, name: str, *, access: str = "read_write") -> TensorView:
        if isinstance(self._element_layout, TangentLayout):
            return self[name]
        if self._element_layout is None or self._element_type is None:
            raise TypeError("TensorStorage field projection requires a Struct element type")
        fields = getattr(self._element_type, "__vernon_fields__", {})
        if name not in fields or name not in self._element_layout.field_names:
            raise ValueError(f"Struct '{self._element_type.__name__}' has no field '{name}'")
        field_index = self._element_layout.field_names.index(name)
        scalar, field_shape = host_scalar_shape(fields[name])
        scalar_dtype = self._dtype(scalar)
        byte_offset = self._element_layout.field_offsets[field_index]
        if byte_offset % scalar_dtype.itemsize:
            raise ValueError(f"Struct field '{name}' offset is not aligned to its scalar leaf")
        inner_byte_strides: list[int] = []
        stride = scalar_dtype.itemsize
        for extent in reversed(field_shape):
            inner_byte_strides.append(stride)
            stride *= extent
        inner_byte_strides.reverse()
        byte_strides = (*self._array.strides, *inner_byte_strides)
        if any(value % scalar_dtype.itemsize for value in byte_strides):
            raise ValueError(f"Struct field '{name}' stride is not aligned to its scalar leaf")
        return TensorView(
            self,
            (*self.shape, *field_shape),
            tuple(value // scalar_dtype.itemsize for value in byte_strides),
            byte_offset // scalar_dtype.itemsize,
            access,
            dtype=scalar_dtype,
        )

    def __getitem__(self, path: str) -> TensorView:
        return self._tangent_view(
            path,
            shape=self.shape,
            strides=tuple(stride // self.dtype.itemsize for stride in self._array.strides),
            offset=0,
            access="read_write",
        )

    def _tangent_view(
        self,
        path: str,
        *,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        offset: int,
        access: str,
    ) -> TensorView:
        if not isinstance(self._element_layout, TangentLayout):
            raise TypeError("structural path projection is available only on tangent TensorStorage")
        leaf = self._element_layout.project(path)
        byte_strides = (
            *(stride * self._element_layout.size for stride in strides),
            *leaf.inner_byte_strides,
        )
        byte_offset = offset * self._element_layout.size + leaf.byte_offset
        if any(stride % leaf.dtype.itemsize for stride in byte_strides) or byte_offset % leaf.dtype.itemsize:
            raise RuntimeError(f"tangent path '{path}' is not representable as a scalar TensorView")
        return TensorView(
            self,
            (*shape, *leaf.shape),
            tuple(stride // leaf.dtype.itemsize for stride in byte_strides),
            byte_offset // leaf.dtype.itemsize,
            access,
            dtype=leaf.dtype,
        )

    def view(
        self,
        *,
        shape: tuple[int, ...] | None = None,
        strides: tuple[int, ...] | None = None,
        offset: int = 0,
        access: str = "read_write",
    ) -> TensorView:
        resolved_shape = self.shape if shape is None else shape
        if strides is None:
            strides = tuple(stride // self.dtype.itemsize for stride in self._array.strides)
        return TensorView(
            self,
            resolved_shape,
            strides,
            offset,
            access,
            element_type=self._element_type,
        )

    def _full_view(self, access: str) -> TensorView:
        access = _checked_access(access)
        view = self._full_views.get(access)
        if view is None:
            view = self.view(access=access)
            self._full_views[access] = view
        return view

    def swizzle(self, components: str) -> TensorView:
        if len(self.shape) < 2:
            raise ValueError("Tensor swizzle requires a trailing component axis")
        spelling = "xyzw"
        aliases = "rgba"
        indices = tuple(spelling.find(value) if value in spelling else aliases.find(value) for value in components)
        if not indices or any(index < 0 or index >= self.shape[-1] for index in indices):
            raise ValueError("Tensor swizzle is outside the component axis")
        start = indices[0]
        if indices != tuple(range(start, start + len(indices))):
            raise ValueError("runtime Tensor swizzles must select contiguous components in storage order")
        view = self.view(
            shape=(*self.shape[:-1], len(indices)),
            strides=tuple(stride // self.dtype.itemsize for stride in self._array.strides),
            offset=start,
        )
        view._components = tuple(range(start, start + len(indices)))
        return view

    @_scoped_host_access("read")
    def to_numpy(self) -> np.ndarray:
        self.synchronize()
        return self._native_host_array().copy(order="C")

    def _borrowed_array(self) -> np.ndarray:
        self.synchronize()
        return self._array

    def _native_host_array(self) -> np.ndarray:
        if self._element_layout is not None and self._element_layout.dtype.subdtype is not None:
            return self._array[_VALUE_FIELD]
        return self._array

    @staticmethod
    def _gradient_layout(left: TensorStorage, right: TensorStorage) -> TangentLayout | None:
        if left.shape != right.shape:
            raise ValueError("graph cotangent contributions have incompatible shapes")
        left_layout = left._element_layout
        right_layout = right._element_layout
        if left_layout is None and right_layout is None:
            return None
        if (
            not isinstance(left_layout, TangentLayout)
            or not isinstance(right_layout, TangentLayout)
            or left_layout.layout_hash != right_layout.layout_hash
        ):
            raise ValueError("graph cotangent contributions have incompatible tangent layouts")
        return left_layout

    @staticmethod
    def _empty_gradient_like(value: TensorStorage) -> TensorStorage:
        if isinstance(value._element_layout, TangentLayout):
            return TensorStorage._tangent_zeros(value._element_layout, value.shape)
        if value._element_layout is not None:
            raise ValueError("graph gradients require scalar or tangent TensorStorage")
        return TensorStorage(np.zeros(value.shape, dtype=value.dtype, order="C"))

    @staticmethod
    def _gradient_add_views(
        output: TensorStorage, left: TensorStorage, right: TensorStorage
    ) -> list[
        tuple[
            TensorStorage | TensorView,
            TensorStorage | TensorView,
            TensorStorage | TensorView,
        ]
    ]:
        layout = TensorStorage._gradient_layout(left, right)
        if layout is None:
            return [(output, left, right)]
        return [(output[leaf.path], left[leaf.path], right[leaf.path]) for leaf in layout.leaves]

    @staticmethod
    def _add_gradients(left: TensorStorage, right: TensorStorage | np.ndarray) -> TensorStorage:
        if not isinstance(left, TensorStorage):
            raise TypeError("graph gradient accumulation requires TensorStorage")
        owners = [left] if not isinstance(right, TensorStorage) or right is left else sorted((left, right), key=id)
        with ExitStack() as claims:
            for owner in owners:
                claims.enter_context(owner._control.host_access(owner._region, "read", "TensorStorage"))
            left.synchronize()
            if isinstance(right, TensorStorage):
                right.synchronize()
                layout = TensorStorage._gradient_layout(left, right)
                if layout is not None:
                    result = TensorStorage._empty_gradient_like(left)
                    for output, left_view, right_view in TensorStorage._gradient_add_views(result, left, right):
                        np.add(
                            left_view._native_host_array(),
                            right_view._native_host_array(),
                            out=output._native_host_array(),
                        )
                    return result
                right_array = right._native_host_array()
            else:
                if left._element_layout is not None:
                    raise ValueError("packed tangent cotangents require matching TensorStorage contributions")
                right_array = np.asarray(right)
            left_array = left._native_host_array()
            if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
                raise ValueError("graph cotangent contributions have incompatible shapes or dtypes")
            result = TensorStorage(np.empty_like(left_array, order="C"))
            np.add(left_array, right_array, out=result._native_host_array())
            return result

    def _materialize_gradient(
        self,
        gradient: np.ndarray,
        path: str,
        destination: TensorStorage | None = None,
    ) -> TensorStorage:
        if self._element_type is not None:
            return self._full_view("read")._materialize_gradient(gradient, path, destination)
        gradient = np.asarray(gradient)
        if destination is None:
            return TensorStorage.from_numpy(gradient)
        target = destination._native_host_array()
        if target.shape != gradient.shape or target.dtype != gradient.dtype:
            raise ValueError("shared-owner gradient leaves require matching shapes and dtypes")
        np.add(target, gradient, out=target)
        return destination

    def _gradient_device_view(
        self,
        path: str,
        destination: TensorStorage,
        gradient_shape: tuple[int, ...],
    ) -> TensorView:
        gradient_shape = tuple(gradient_shape)
        return self._full_view("read")._gradient_device_view(path, destination, gradient_shape)

    def _gradient_boundary_layout(
        self,
        destination: TensorStorage,
        element_byte_size: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        return self._full_view("read")._gradient_boundary_layout(destination, element_byte_size)

    @_scoped_host_access("write")
    def copy_from_numpy(self, array: np.ndarray) -> None:
        target = self._native_host_array()
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != target.dtype
            or array.shape != target.shape
            or not array.flags.c_contiguous
        ):
            raise ValueError("upload requires matching dtype, shape, and contiguity")
        np.copyto(target, array)
        self._mark_host_dirty([(0, self._array.nbytes)])

    @_scoped_host_access("write")
    def update(self, indices: int | slice | list[int] | np.ndarray, values: np.ndarray) -> None:
        """Replace selected first-axis elements and dirty only their backing byte ranges."""
        if not self.shape:
            raise ValueError("partial TensorStorage updates require a non-scalar storage shape")
        target = self._native_host_array()
        normalized_indices: np.ndarray | None = None
        if isinstance(indices, (int, np.integer)) and not isinstance(indices, bool):
            normalized = int(indices)
            if normalized < 0:
                normalized += self.shape[0]
            if normalized < 0 or normalized >= self.shape[0]:
                raise IndexError("partial update index is outside TensorStorage")
            selection: int | slice | np.ndarray = normalized
        elif isinstance(indices, slice):
            selection = indices
        else:
            raw_indices = np.asarray(indices)
            if raw_indices.ndim != 1:
                raise ValueError("partial update indices must be one-dimensional")
            if raw_indices.dtype.kind not in {"i", "u"}:
                raise TypeError("partial update indices must be integers")
            normalized_indices = raw_indices.astype(np.int64, copy=False)
            normalized_indices = np.where(
                normalized_indices < 0,
                normalized_indices + self.shape[0],
                normalized_indices,
            )
            if np.any(normalized_indices < 0) or np.any(normalized_indices >= self.shape[0]):
                raise IndexError("partial update index is outside TensorStorage")
            selection = normalized_indices
        selected = target[selection]
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != selected.dtype
            or values.shape != selected.shape
            or not values.flags.c_contiguous
        ):
            raise ValueError("partial update values require matching dtype, shape, and contiguity")
        if isinstance(selection, int):
            ranges = _array_byte_ranges(target[normalized : normalized + 1], self._array)
        elif isinstance(selection, slice):
            ranges = _array_byte_ranges(target[selection], self._array)
        else:
            assert normalized_indices is not None
            ranges = [
                byte_range
                for index in normalized_indices
                for byte_range in _array_byte_ranges(target[int(index) : int(index) + 1], self._array)
            ]
        dirty_ranges = self._prepare_partial_host_write(ranges)
        target[selection] = values
        self._mark_host_dirty(dirty_ranges, host_complete=False)

    @_scoped_host_access("write")
    def update_values(self, indices: int | list[int] | np.ndarray, values: Any) -> None:
        """Replace selected logical struct elements without repacking unchanged elements."""
        if (
            self._element_layout is None
            or isinstance(self._element_layout, TangentLayout)
            or self._element_type is None
        ):
            raise TypeError("update_values requires a Struct TensorStorage")
        if len(self.shape) != 1:
            raise ValueError("update_values currently requires one-dimensional Struct storage")
        scalar_index = isinstance(indices, (int, np.integer)) and not isinstance(indices, bool)
        raw_indices = np.asarray([indices] if scalar_index else indices)
        if raw_indices.ndim != 1:
            raise ValueError("partial value update indices must be one-dimensional")
        if raw_indices.dtype.kind not in {"i", "u"}:
            raise TypeError("partial value update indices must be integers")
        normalized_indices = raw_indices.astype(np.int64, copy=False)
        normalized_indices = np.where(
            normalized_indices < 0,
            normalized_indices + self.shape[0],
            normalized_indices,
        )
        if np.any(normalized_indices < 0) or np.any(normalized_indices >= self.shape[0]):
            raise IndexError("partial value update index is outside TensorStorage")
        logical_values = (values,) if scalar_index else tuple(values)
        if len(logical_values) != len(normalized_indices):
            raise ValueError("partial value update requires one logical Value per index")
        ranges = [
            byte_range
            for index in normalized_indices
            for byte_range in _array_byte_ranges(self._array[int(index) : int(index) + 1], self._array)
        ]
        dirty_ranges = self._prepare_partial_host_write(ranges)
        for index, logical_value in zip(normalized_indices, logical_values, strict=True):
            packed = pack_host_value(self._element_type, logical_value)
            if self._element_layout.dtype.subdtype is None:
                self._array[int(index)] = packed
            else:
                self._array[_VALUE_FIELD][int(index)] = packed
        self._mark_host_dirty(dirty_ranges, host_complete=False)

    @_scoped_host_access("write")
    def copy_from_values(self, values: Any) -> None:
        if (
            self._element_layout is None
            or isinstance(self._element_layout, TangentLayout)
            or self._element_type is None
        ):
            raise TypeError("copy_from_values requires a Struct TensorStorage")
        logical_shape = _logical_collection_shape(values, self._element_type)
        if logical_shape != self.shape:
            raise ValueError(f"logical Value shape {logical_shape} does not match storage shape {self.shape}")
        for index in np.ndindex(self.shape):
            logical_value = values
            for component in index:
                logical_value = logical_value[component]
            packed = pack_host_value(self._element_type, logical_value)
            if self._element_layout.dtype.subdtype is None:
                self._array[index] = packed
            else:
                self._array[_VALUE_FIELD][index] = packed
        self._mark_host_dirty([(0, self._array.nbytes)])

    def _mark_host_dirty(self, ranges: list[tuple[int, int]], *, host_complete: bool = True) -> None:
        coherence = self._control.coherence
        if host_complete:
            coherence.device_dirty_ranges.clear()
        else:
            coherence.device_dirty_ranges = _subtract_byte_ranges(coherence.device_dirty_ranges, ranges)
        coherence.device_dirty = bool(coherence.device_dirty_ranges)
        coherence.dirty_ranges.mark(ranges, allow_full=not coherence.device_dirty)
        self._control.publish_host_write(
            host_complete=not coherence.device_dirty,
            recovers_unknown=host_complete,
        )

    def _prepare_partial_host_write(self, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        coherence = self._control.coherence
        if coherence.device_dirty and coherence.dirty_ranges.should_promote_full(ranges):
            coalesced = _coalesced_byte_ranges(ranges)
            promoted = [(coalesced[0][0], coalesced[-1][1])]
            self.synchronize(_BufferRegion(tuple(promoted)))
            return promoted
        return ranges

    @_scoped_host_access("read")
    def to_values(self) -> Any:
        if self._element_layout is None or self._element_type is None:
            raise TypeError("to_values requires a Struct TensorStorage")
        self.synchronize()
        element_layout = self._element_layout

        def materialize(prefix: tuple[int, ...], dimension: int) -> Any:
            if dimension == len(self.shape):
                value = (
                    self._array[prefix] if element_layout.dtype.subdtype is None else self._array[_VALUE_FIELD][prefix]
                )
                if isinstance(element_layout, TangentLayout):
                    return unpack_tangent_value(element_layout.tangent_schema, value)
                return unpack_host_value(self._element_type, value)
            return tuple(materialize((*prefix, index), dimension + 1) for index in range(self.shape[dimension]))

        return materialize((), 0)

    def synchronize(self, region: _BufferRegion | None = None) -> None:
        _synchronize_buffer(self, self._region if region is None else region)

    def _device_claim_region(self, context: _InvocationContext, region: _BufferRegion, access: str) -> _BufferRegion:
        if context.session.arch == cpu:
            return region
        residency = self._control.residencies.get(context.session)
        latest = self._control.residencies.latest()
        if residency is None or (self._control.coherence.device_dirty and residency is not latest):
            return self._region
        return region

    def _resident_buffer(
        self,
        context: _InvocationContext,
        region: _BufferRegion | None = None,
        access: str = "read_write",
    ) -> Any:
        return _materialize_buffer(self, context, self._region if region is None else region, access)

    def _publish_device_write(
        self,
        context: _InvocationContext,
        regions: tuple[_BufferRegion, ...],
    ) -> None:
        if not regions:
            return
        coherence = self._control.coherence
        if context.session.arch == cpu:
            self._control.publish_host_write(host_complete=True)
            coherence.device_dirty = False
            coherence.dirty_ranges.mark(
                [byte_range for region in regions for byte_range in region.ranges],
                allow_full=True,
            )
            return
        self._control.publish_device_write(context.session)
        coherence.device_dirty = True
        coherence.device_dirty_ranges = _coalesced_byte_ranges(
            [
                *coherence.device_dirty_ranges,
                *(byte_range for region in regions for byte_range in region.ranges),
            ]
        )
        coherence.dirty_ranges.clear()


class RawBuffer:
    """Owned raw bytes with explicit alignment and typed-view construction."""

    def __init__(self, buffer: Any, *, alignment: int):
        if not isinstance(alignment, int) or isinstance(alignment, bool) or alignment <= 0:
            raise ValueError("RawBuffer alignment must be a positive integer")
        try:
            view = memoryview(buffer)
        except TypeError as error:
            raise TypeError("RawBuffer requires a contiguous buffer-protocol object") from error
        if not view.c_contiguous:
            raise ValueError("RawBuffer requires contiguous external bytes")
        self._array = np.frombuffer(view, dtype=np.uint8).copy()
        self._alignment = alignment
        self._control = _ResourceControlBlock(_BufferCoherence(_DirtyRangeSet(self._array.nbytes, dirty=True)))
        self._region = _BufferRegion(((0, self._array.nbytes),))

    @classmethod
    def allocate(cls, byte_size: int, *, alignment: int) -> RawBuffer:
        if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size < 0:
            raise ValueError("RawBuffer byte_size must be a non-negative integer")
        return cls(bytearray(byte_size), alignment=alignment)

    @classmethod
    def from_buffer(cls, buffer: Any, *, alignment: int) -> RawBuffer:
        return cls(buffer, alignment=alignment)

    @property
    def byte_size(self) -> int:
        return self._array.nbytes

    @property
    def alignment(self) -> int:
        return self._alignment

    def typed_view(
        self,
        *,
        dtype: Any,
        shape: tuple[int, ...],
        byte_strides: tuple[int, ...],
        byte_offset: int = 0,
        access: str,
        layout_units: str,
    ) -> TensorView:
        if layout_units != "bytes":
            raise ValueError("RawBuffer typed views require layout_units='bytes'")
        if isinstance(dtype, _Scalar):
            resolved_dtype = TensorStorage._dtype(dtype)
            element_type: Any | None = None
            element_alignment = resolved_dtype.itemsize
        else:
            abi_layout = host_abi_layout(dtype)
            resolved_dtype = abi_layout.dtype
            element_type = dtype
            element_alignment = abi_layout.alignment
        item_size = resolved_dtype.itemsize
        if not isinstance(byte_strides, tuple) or any(
            not isinstance(stride, int) or isinstance(stride, bool) for stride in byte_strides
        ):
            raise ValueError("RawBuffer byte_strides must be signed integers")
        if not isinstance(byte_offset, int) or isinstance(byte_offset, bool):
            raise ValueError("RawBuffer byte_offset must be an integer")
        if any(stride % item_size != 0 for stride in byte_strides) or byte_offset % item_size != 0:
            raise ValueError("RawBuffer byte layout must be divisible by the typed element size")
        if self._alignment < element_alignment or (self._array.ctypes.data + byte_offset) % element_alignment:
            raise ValueError("RawBuffer typed view does not satisfy element alignment")
        return TensorView(
            self,
            shape,
            tuple(stride // item_size for stride in byte_strides),
            byte_offset // item_size,
            access,
            dtype=resolved_dtype,
            element_type=element_type,
        )

    def _mark_host_dirty(self, ranges: list[tuple[int, int]], *, recovers_unknown: bool = False) -> None:
        coherence = self._control.coherence
        coherence.device_dirty_ranges = _subtract_byte_ranges(coherence.device_dirty_ranges, ranges)
        coherence.device_dirty = bool(coherence.device_dirty_ranges)
        coherence.dirty_ranges.mark(ranges, allow_full=not coherence.device_dirty)
        self._control.publish_host_write(
            host_complete=not coherence.device_dirty,
            recovers_unknown=recovers_unknown,
        )

    def synchronize(self, region: _BufferRegion | None = None) -> None:
        _synchronize_buffer(self, self._region if region is None else region)

    def _prepare_partial_host_write(self, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        coherence = self._control.coherence
        if coherence.device_dirty and coherence.dirty_ranges.should_promote_full(ranges):
            coalesced = _coalesced_byte_ranges(ranges)
            promoted = [(coalesced[0][0], coalesced[-1][1])]
            self.synchronize(_BufferRegion(tuple(promoted)))
            return promoted
        return ranges

    def _device_claim_region(self, context: _InvocationContext, region: _BufferRegion, access: str) -> _BufferRegion:
        if context.session.arch == cpu:
            return region
        residency = self._control.residencies.get(context.session)
        latest = self._control.residencies.latest()
        if residency is None or (self._control.coherence.device_dirty and residency is not latest):
            return self._region
        return region

    def _resident_buffer(
        self,
        context: _InvocationContext,
        region: _BufferRegion | None = None,
        access: str = "read_write",
    ) -> Any:
        return _materialize_buffer(self, context, self._region if region is None else region, access)

    def _publish_device_write(
        self,
        context: _InvocationContext,
        regions: tuple[_BufferRegion, ...],
    ) -> None:
        if not regions:
            return
        coherence = self._control.coherence
        if context.session.arch == cpu:
            self._control.publish_host_write(host_complete=True)
            coherence.device_dirty = False
            coherence.dirty_ranges.mark(
                [byte_range for region in regions for byte_range in region.ranges],
                allow_full=True,
            )
            return
        self._control.publish_device_write(context.session)
        coherence.device_dirty = True
        coherence.device_dirty_ranges = _coalesced_byte_ranges(
            [
                *coherence.device_dirty_ranges,
                *(byte_range for region in regions for byte_range in region.ranges),
            ]
        )
        coherence.dirty_ranges.clear()


class TensorView:
    """A validated non-owning shaped and strided borrow of TensorStorage."""

    def __init__(
        self,
        owner: TensorStorage | RawBuffer,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        offset: int = 0,
        access: str = "read_write",
        *,
        dtype: np.dtype[Any] | None = None,
        element_type: Any | None = None,
    ):
        if not isinstance(owner, (TensorStorage, RawBuffer)):
            raise TypeError("TensorView owner must be a TensorStorage or RawBuffer")
        if isinstance(owner, TensorStorage):
            if dtype is not None and owner._element_layout is None and dtype != owner.dtype:
                raise TypeError("TensorStorage view dtype must match its owner")
            dtype = owner.dtype if dtype is None else dtype
            if element_type is None:
                element_type = owner._element_type
        elif dtype is None:
            raise TypeError("RawBuffer TensorView requires an explicit dtype")
        if dtype is None:
            raise TypeError("TensorView requires a resolved element dtype")
        shape = _checked_shape(shape)
        if (
            not isinstance(strides, tuple)
            or len(strides) != len(shape)
            or any(not isinstance(stride, int) or isinstance(stride, bool) for stride in strides)
        ):
            raise ValueError("TensorView requires one signed integer stride per dimension")
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise ValueError("TensorView offset must be an integer element offset")
        access = _checked_access(access)
        address_range = _address_range(shape, strides, offset)
        owner_elements = owner._array.nbytes // dtype.itemsize
        if address_range is not None and (address_range[0] < 0 or address_range[1] >= owner_elements):
            raise ValueError("TensorView layout is outside its owner allocation")
        self._owner = owner
        self._shape = shape
        self._strides = strides
        self._offset = offset
        self._access = access
        self._dtype = dtype
        self._element_type = element_type
        self._components: tuple[int, ...] | None = None
        self._region = _BufferRegion(tuple(_array_byte_ranges(self._native_host_array(), owner._array)))

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr("TensorView", arguments)

    @_scoped_host_access("read")
    def __getitem__(self, index: Any) -> Any:
        if self.access == "write":
            raise PermissionError("cannot read from a write-only TensorView")
        return self._borrowed_array()[index]

    @_scoped_host_access("write")
    def __setitem__(self, index: Any, value: Any) -> None:
        if self.access == "read":
            raise PermissionError("cannot write through a read-only TensorView")
        target = self._native_host_array()
        ranges = _array_byte_ranges(target, self._owner._array)
        dirty_ranges = self._owner._prepare_partial_host_write(ranges)
        target[index] = value
        if isinstance(self._owner, TensorStorage):
            self._owner._mark_host_dirty(dirty_ranges, host_complete=False)
        else:
            self._owner._mark_host_dirty(dirty_ranges)

    @property
    def owner(self) -> TensorStorage | RawBuffer:
        return self._owner

    def _host_access_region(self, access: str) -> _BufferRegion:
        if access != "write":
            return self._region
        coherence = self._owner._control.coherence
        if not coherence.device_dirty or not coherence.dirty_ranges.should_promote_full(list(self._region.ranges)):
            return self._region
        return _BufferRegion(((self._region.ranges[0][0], self._region.ranges[-1][1]),))

    @property
    def access(self) -> str:
        return self._access

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def element_type(self) -> Any | None:
        return self._element_type

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def layout(self) -> TensorLayout:
        return TensorLayout(
            self.shape,
            tuple(stride * self.dtype.itemsize for stride in self._strides),
            self._offset * self.dtype.itemsize,
            self._components,
            self._strides,
            self._offset,
        )

    def _with_access(self, access: str) -> TensorView:
        access = _checked_access(access)
        if access in {"read", "read_write"} and self.access == "write":
            raise ValueError("TensorView does not permit reads")
        if access in {"write", "read_write"} and self.access == "read":
            raise ValueError("TensorView does not permit writes")
        if access == self.access:
            return self
        result = TensorView(
            self.owner,
            self.shape,
            self._strides,
            self._offset,
            access,
            dtype=self.dtype,
            element_type=self.element_type,
        )
        result._components = self._components
        return result

    @_scoped_host_access("read")
    def to_numpy(self) -> np.ndarray:
        if self.access == "write":
            raise PermissionError("cannot read from a write-only TensorView")
        return np.array(self._borrowed_array(), copy=True, order="C")

    def _borrowed_array(self) -> np.ndarray:
        self._owner.synchronize(self._host_access_region("read"))
        return self._native_host_array()

    def _native_host_array(self) -> np.ndarray:
        return np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=self._owner._array,
            offset=self._offset * self.dtype.itemsize,
            strides=tuple(stride * self.dtype.itemsize for stride in self._strides),
        )

    def _materialize_gradient(
        self,
        gradient: np.ndarray,
        path: str,
        destination: TensorStorage | None = None,
    ) -> TensorStorage:
        gradient = np.asarray(gradient)
        if self._element_type is not None and not isinstance(self._element_type, _Scalar):
            layout = tangent_layout(self._element_type)
            root = path.split(".", 1)[0]
            leaf_path = path[len(root) + 1 :] if path != root else ""
            leaf = layout.project(leaf_path)
            trailing_shape = leaf.shape
            if (
                tuple(gradient.shape[: len(self.shape)]) != self.shape
                or tuple(gradient.shape[len(self.shape) :]) != trailing_shape
            ):
                raise ValueError("logical gradient shape does not match its aggregate TensorView leaf")
            if destination is None:
                if not isinstance(self.owner, TensorStorage):
                    raise TypeError("aggregate tangent owners require canonical TensorStorage primals")
                destination = TensorStorage._tangent_zeros(layout, self.owner.shape)
            elif not isinstance(destination._element_layout, TangentLayout) or (
                destination._element_layout.layout_hash != layout.layout_hash
            ):
                raise ValueError("shared-owner aggregate gradients require one tangent layout")
            target = np.ndarray(
                gradient.shape,
                dtype=leaf.dtype,
                buffer=destination._array,
                offset=self._offset * layout.size + leaf.byte_offset,
                strides=(
                    *(stride * layout.size for stride in self._strides),
                    *leaf.inner_byte_strides,
                ),
            )
            np.add(target, gradient, out=target)
            return destination
        trailing_shape = tuple(gradient.shape[len(self.shape) :])
        if tuple(gradient.shape[: len(self.shape)]) != self.shape:
            raise ValueError("logical gradient shape does not match its TensorView descriptor")
        scalar_count = int(np.prod(trailing_shape, dtype=np.int64)) if trailing_shape else 1
        owner_element_count = self.owner._array.nbytes // self.dtype.itemsize
        owner_shape = (
            (*self.owner.shape, *trailing_shape)
            if isinstance(self.owner, TensorStorage)
            else (owner_element_count, *trailing_shape)
        )
        if destination is None:
            owner_gradient = np.zeros(owner_shape, dtype=gradient.dtype, order="C")
            destination = TensorStorage.from_numpy(owner_gradient)
            owner_gradient = destination._native_host_array()
        else:
            owner_gradient = destination._native_host_array()
            if owner_gradient.shape != owner_shape or owner_gradient.dtype != gradient.dtype:
                raise ValueError("shared-owner TensorView gradients require matching owner layouts")
        inner_strides: list[int] = []
        stride = 1
        for extent in reversed(trailing_shape):
            inner_strides.append(stride)
            stride *= extent
        inner_strides.reverse()
        scalar_strides = (
            *(value * scalar_count for value in self._strides),
            *inner_strides,
        )
        target = np.ndarray(
            gradient.shape,
            dtype=gradient.dtype,
            buffer=owner_gradient,
            offset=self._offset * scalar_count * gradient.dtype.itemsize,
            strides=tuple(value * gradient.dtype.itemsize for value in scalar_strides),
        )
        np.add(target, gradient, out=target)
        return destination

    def _gradient_device_view(
        self,
        path: str,
        destination: TensorStorage,
        gradient_shape: tuple[int, ...],
    ) -> TensorView:
        gradient_shape = tuple(gradient_shape)
        if self._element_type is not None and not isinstance(self._element_type, _Scalar):
            layout = tangent_layout(self._element_type)
            root = path.split(".", 1)[0]
            leaf_path = path[len(root) + 1 :] if path != root else ""
            leaf = layout.project(leaf_path)
            if gradient_shape != (*self.shape, *leaf.shape):
                raise ValueError("logical gradient shape does not match its aggregate device leaf")
            return destination._tangent_view(
                leaf_path,
                shape=self.shape,
                strides=self._strides,
                offset=self._offset,
                access="read_write",
            )
        trailing_shape = gradient_shape[len(self.shape) :]
        if gradient_shape[: len(self.shape)] != self.shape:
            raise ValueError("logical gradient shape does not match its device TensorView")
        scalar_count = int(np.prod(trailing_shape, dtype=np.int64)) if trailing_shape else 1
        inner_strides: list[int] = []
        stride = 1
        for extent in reversed(trailing_shape):
            inner_strides.append(stride)
            stride *= extent
        inner_strides.reverse()
        return destination.view(
            shape=gradient_shape,
            strides=(
                *(value * scalar_count for value in self._strides),
                *inner_strides,
            ),
            offset=self._offset * scalar_count,
            access="read_write",
        )

    def _gradient_boundary_layout(
        self,
        destination: TensorStorage,
        element_byte_size: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        if not isinstance(destination, TensorStorage):
            raise TypeError("canonical derivative publication requires TensorStorage backing")
        if element_byte_size <= 0:
            raise ValueError("canonical derivative element size must be positive")
        return (
            self.shape,
            tuple(stride * element_byte_size for stride in self._strides),
            self._offset * element_byte_size,
        )

    @_scoped_host_access("write")
    def copy_from_numpy(self, array: np.ndarray) -> None:
        if self.access == "read":
            raise PermissionError("cannot write through a read-only TensorView")
        target = self._native_host_array()
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != target.dtype
            or array.shape != target.shape
            or not array.flags.c_contiguous
        ):
            raise ValueError("upload requires matching dtype and shape")
        ranges = _array_byte_ranges(target, self._owner._array)
        dirty_ranges = self._owner._prepare_partial_host_write(ranges)
        np.copyto(target, array)
        if isinstance(self._owner, TensorStorage):
            self._owner._mark_host_dirty(dirty_ranges, host_complete=False)
        else:
            self._owner._mark_host_dirty(dirty_ranges, recovers_unknown=self._region == self._owner._region)

    def _resident_buffer(self, context: _InvocationContext) -> Any:
        return self._owner._resident_buffer(context, self._region, self.access)
