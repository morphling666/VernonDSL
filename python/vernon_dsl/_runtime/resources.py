from __future__ import annotations

import importlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, get_args, get_origin

import numpy as np

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


def _session_state() -> Any:
    return importlib.import_module("vernon_dsl._runtime.session")


_NUMPY_DTYPES = {
    "bool": np.dtype(np.bool_),
    "i32": np.dtype(np.int32),
    "u32": np.dtype(np.uint32),
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}

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


def _bind_native_argument(
    builder: Any,
    parameter: Any,
    value: Any,
    *,
    host_value: bool = False,
    annotation: Any | None = None,
) -> Any:
    state = _session_state()
    if isinstance(value, (TensorStorage, TensorView)):
        if state._architecture == state.cpu or host_value:
            return builder.host_tensor(parameter.name, value._native_host_array())
        if state._rhi_host is None:
            raise RuntimeError("device Tensor arguments require a GPU RHI host")
        layout = value.layout
        return builder.rhi_tensor(
            parameter.name,
            value._resident_buffer(),
            parameter.access,
            list(value.shape),
            list(layout.byte_strides),
            layout.byte_offset,
        )

    value_type = type(value)
    struct_type = (
        annotation
        if isinstance(annotation, type) and getattr(annotation, "__vernon_dsl__", (None, {}))[0] == "struct"
        else value_type
        if getattr(value_type, "__vernon_dsl__", (None, {}))[0] == "struct"
        else None
    )
    if struct_type is not None:
        layout = host_abi_layout(struct_type)
        host_array = np.empty((), dtype=layout.dtype)
        host_array[()] = pack_host_value(struct_type, value, parameter.name)
        return builder.host_tensor(parameter.name, host_array)

    numpy_dtypes = {
        state._native.DATA_BOOL: np.dtype(np.bool_),
        state._native.DATA_I32: np.dtype(np.int32),
        state._native.DATA_U32: np.dtype(np.uint32),
        state._native.DATA_F16: np.dtype(np.float16),
        state._native.DATA_F32: np.dtype(np.float32),
        state._native.DATA_F64: np.dtype(np.float64),
    }
    leaves = tuple(parameter.element_leaves)
    dtype = numpy_dtypes.get(leaves[0][0]) if len(leaves) == 1 and leaves[0][1:] == (1, 0) else None
    if dtype is None:
        raise TypeError(f"aggregate parameter {parameter.name!r} requires canonical TensorStorage")
    host_array = np.asarray(value, dtype=dtype)
    if tuple(host_array.shape) != tuple(parameter.shape):
        raise ValueError(
            f"parameter {parameter.name!r} expects shape {tuple(parameter.shape)}, got {tuple(host_array.shape)}"
        )
    return builder.host_tensor(parameter.name, host_array)


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


def _borrow_ranges_may_overlap(left: TensorView, right: TensorView) -> bool:
    if left.owner is not right.owner or any(extent == 0 for extent in left.shape + right.shape):
        return False
    left_range = _address_range(left.shape, left._strides, left._offset)
    right_range = _address_range(right.shape, right._strides, right._offset)
    if left_range is None or right_range is None:
        return False
    return left_range[0] <= right_range[1] and right_range[0] <= left_range[1]


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
        if array.dtype not in _NUMPY_DTYPES.values() and element_type is None:
            raise TypeError(f"unsupported TensorStorage dtype {array.dtype}")
        if element_layout is not None and array.dtype != _storage_array_dtype(element_layout):
            raise TypeError("TensorStorage array dtype does not match its canonical element ABI")
        self._array = array
        self._element_type = element_type
        self._element_layout = element_layout
        self._element_alignment = element_layout.alignment if element_layout is not None else array.dtype.itemsize
        self._native_buffer: Any | None = None
        self._native_generation = -1
        self._host_version = 1
        self._uploaded_version = 0
        self._device_dirty = False
        self._allocation_count = 0
        self._upload_count = 0
        self._download_count = 0
        self._borrow_lock = threading.RLock()
        self._active_borrows: list[tuple[object, TensorView, str]] = []
        self._full_views: dict[str, TensorView] = {}
        _session_state()._runtime_children.add(self)

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr(cls.__name__, arguments)

    @staticmethod
    def _dtype(dtype: _Scalar) -> np.dtype[Any]:
        if not isinstance(dtype, _Scalar) or dtype.name not in _NUMPY_DTYPES:
            raise TypeError("dtype must be a Vernon scalar type")
        return _NUMPY_DTYPES[dtype.name]

    @staticmethod
    def _storage_dtype(dtype: Any) -> tuple[np.dtype[Any], HostAbiLayout | None]:
        if isinstance(dtype, _Scalar):
            return TensorStorage._dtype(dtype), None
        layout = host_abi_layout(dtype)
        return _storage_array_dtype(layout), layout

    @classmethod
    def zeros(cls, *, dtype: Any, shape: tuple[int, ...]) -> TensorStorage:
        numpy_dtype, layout = cls._storage_dtype(dtype)
        return cls(
            np.zeros(shape, dtype=numpy_dtype, order="C"),
            element_type=dtype if layout is not None else None,
            element_layout=layout,
        )

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
        if array.dtype not in _NUMPY_DTYPES.values():
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

    def to_numpy(self) -> np.ndarray:
        self._ensure_host_read_allowed()
        self.synchronize()
        return self._native_host_array().copy(order="C")

    def _borrowed_array(self) -> np.ndarray:
        self.synchronize()
        return self._array

    def _native_host_array(self) -> np.ndarray:
        if self._element_layout is not None and self._element_layout.dtype.subdtype is not None:
            return self._array[_VALUE_FIELD]
        return self._array

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

    def copy_from_numpy(self, array: np.ndarray) -> None:
        self._ensure_host_mutation_allowed()
        target = self._native_host_array()
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != target.dtype
            or array.shape != target.shape
            or not array.flags.c_contiguous
        ):
            raise ValueError("upload requires matching dtype, shape, and contiguity")
        np.copyto(target, array)
        self._host_version += 1
        self._device_dirty = False

    def copy_from_values(self, values: Any) -> None:
        self._ensure_host_mutation_allowed()
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
        self._host_version += 1
        self._device_dirty = False

    def to_values(self) -> Any:
        if self._element_layout is None or self._element_type is None:
            raise TypeError("to_values requires a Struct TensorStorage")
        self._ensure_host_read_allowed()
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

    def synchronize(self) -> None:
        if not self._device_dirty or self._native_buffer is None:
            return
        downloaded = np.frombuffer(self._native_buffer.download(), dtype=self._array.dtype).reshape(self.shape)
        np.copyto(self._array, downloaded)
        self._download_count += 1
        self._device_dirty = False
        self._host_version += 1
        self._uploaded_version = self._host_version

    def _resident_buffer(self) -> Any:
        state = _session_state()
        if state._native_runtime is None or state._rhi_host is None:
            raise RuntimeError("device TensorStorage residency requires a GPU RHI host")
        if self._native_buffer is None or self._native_generation != state._runtime_generation:
            self._native_buffer = state._rhi_host.create_buffer(self._array.nbytes)
            self._native_generation = state._runtime_generation
            self._uploaded_version = 0
            self._device_dirty = False
            self._allocation_count += 1
        if self._uploaded_version != self._host_version:
            self._native_buffer.upload(self._array.tobytes(order="C"))
            self._uploaded_version = self._host_version
            self._upload_count += 1
        return self._native_buffer

    def _mark_device_dirty(self) -> None:
        self._device_dirty = True

    def _ensure_host_mutation_allowed(self) -> None:
        with self._borrow_lock:
            if self._active_borrows:
                raise RuntimeError("host mutation is forbidden while a device dispatch borrows TensorStorage")

    def _ensure_host_read_allowed(self) -> None:
        with self._borrow_lock:
            if any(access != "read" for _, _, access in self._active_borrows):
                raise RuntimeError("host reads are forbidden while a device dispatch writes TensorStorage")


class RawBuffer:
    """Explicitly aligned external bytes with typed-view construction."""

    def __init__(self, buffer: Any, *, alignment: int):
        if not isinstance(alignment, int) or isinstance(alignment, bool) or alignment <= 0:
            raise ValueError("RawBuffer alignment must be a positive integer")
        try:
            view = memoryview(buffer)
        except TypeError as error:
            raise TypeError("RawBuffer requires a contiguous buffer-protocol object") from error
        if not view.c_contiguous:
            raise ValueError("RawBuffer requires contiguous external bytes")
        self._external_owner = buffer
        self._array = np.frombuffer(view, dtype=np.uint8)
        self._alignment = alignment
        self._readonly = view.readonly
        self._native_buffer: Any | None = None
        self._native_generation = -1
        self._host_version = 1
        self._uploaded_version = 0
        self._device_dirty = False
        self._borrow_lock = threading.RLock()
        self._active_borrows: list[tuple[object, TensorView, str]] = []
        _session_state()._runtime_children.add(self)

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
        if access != "read" and self._readonly:
            raise ValueError("writable TensorView requires writable RawBuffer bytes")
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

    def synchronize(self) -> None:
        if not self._device_dirty or self._native_buffer is None:
            return
        downloaded = self._native_buffer.download()
        self._array[:] = np.frombuffer(downloaded, dtype=np.uint8)
        self._device_dirty = False
        self._host_version += 1
        self._uploaded_version = self._host_version

    def _resident_buffer(self) -> Any:
        state = _session_state()
        if state._native_runtime is None or state._rhi_host is None:
            raise RuntimeError("device RawBuffer residency requires a GPU RHI host")
        if self._native_buffer is None or self._native_generation != state._runtime_generation:
            self._native_buffer = state._rhi_host.create_buffer(self.byte_size)
            self._native_generation = state._runtime_generation
            self._uploaded_version = 0
            self._device_dirty = False
        if self._uploaded_version != self._host_version:
            self._native_buffer.upload(self._array.tobytes())
            self._uploaded_version = self._host_version
        return self._native_buffer

    def _mark_device_dirty(self) -> None:
        self._device_dirty = True

    def _ensure_host_mutation_allowed(self) -> None:
        with self._borrow_lock:
            if self._active_borrows:
                raise RuntimeError("host mutation is forbidden while a device dispatch borrows RawBuffer")

    def _ensure_host_read_allowed(self) -> None:
        with self._borrow_lock:
            if any(access != "read" for _, _, access in self._active_borrows):
                raise RuntimeError("host reads are forbidden while a device dispatch writes RawBuffer")


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

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr("TensorView", arguments)

    if TYPE_CHECKING:

        def __getitem__(self, index: Any) -> Any: ...

        def __setitem__(self, index: Any, value: Any) -> None: ...

    @property
    def owner(self) -> TensorStorage | RawBuffer:
        return self._owner

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

    def to_numpy(self) -> np.ndarray:
        if self.access == "write":
            raise PermissionError("cannot read from a write-only TensorView")
        return np.array(self._borrowed_array(), copy=True, order="C")

    def _borrowed_array(self) -> np.ndarray:
        self._owner._ensure_host_read_allowed()
        self._owner.synchronize()
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

    def copy_from_numpy(self, array: np.ndarray) -> None:
        self._owner._ensure_host_mutation_allowed()
        if self.access == "read":
            raise PermissionError("cannot write through a read-only TensorView")
        self._owner.synchronize()
        target = self._native_host_array()
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != target.dtype
            or array.shape != target.shape
            or not array.flags.c_contiguous
        ):
            raise ValueError("upload requires matching dtype and shape")
        np.copyto(target, array)
        self._owner._host_version += 1
        self._owner._device_dirty = False

    def _resident_buffer(self) -> Any:
        return self._owner._resident_buffer()

    def _mark_device_dirty(self) -> None:
        self._owner._mark_device_dirty()


def _normalize_dispatch_borrows(
    borrows: list[tuple[str, TensorStorage | TensorView, str]],
) -> list[tuple[str, TensorView, str]]:
    normalized: list[tuple[str, TensorView, str]] = []
    for name, value, access in borrows:
        access = _checked_access(access)
        view = value._full_view("read_write") if isinstance(value, TensorStorage) else value
        if access in {"read", "read_write"} and view.access == "write":
            raise ValueError(f"TensorView argument '{name}' does not permit reads")
        if access in {"write", "read_write"} and view.access == "read":
            raise ValueError(f"TensorView argument '{name}' does not permit writes")
        normalized.append((name, view, access))
    return normalized


def _validate_dispatch_borrows(
    borrows: list[tuple[str, TensorStorage | TensorView, str]],
) -> None:
    _normalize_dispatch_borrows(borrows)


@contextmanager
def _dispatch_borrow_scope(
    borrows: list[tuple[str, TensorStorage | TensorView, str]],
) -> Iterator[None]:
    """Retain owners and reject incompatible borrows until dispatch completion."""
    normalized = _normalize_dispatch_borrows(borrows)
    owners = sorted({view.owner for _, view, _ in normalized}, key=id)
    token = object()
    for owner in owners:
        owner._borrow_lock.acquire()
    try:
        for name, view, access in normalized:
            for _, active_view, active_access in view.owner._active_borrows:
                if access == active_access == "read":
                    continue
                if _borrow_ranges_may_overlap(view, active_view):
                    raise RuntimeError(f"dispatch argument '{name}' conflicts with an outstanding device borrow")
        for _, view, access in normalized:
            view.owner._active_borrows.append((token, view, access))
    finally:
        for owner in reversed(owners):
            owner._borrow_lock.release()
    try:
        yield
    finally:
        for owner in owners:
            owner._borrow_lock.acquire()
        try:
            for owner in owners:
                owner._active_borrows[:] = [active for active in owner._active_borrows if active[0] is not token]
        finally:
            for owner in reversed(owners):
                owner._borrow_lock.release()


@dataclass(frozen=True)
class _TextureFormat:
    name: str
    _native_name: str
    _depth: bool = False


rgba8 = _TextureFormat("rgba8", "RGBA8_UNORM")
depth32 = _TextureFormat("depth32", "D32_FLOAT", True)


class Texture:
    """Runtime image storage and shader texture annotation constructor."""

    def __init__(
        self,
        array: np.ndarray,
        *,
        format: _TextureFormat | None = None,
        dimension: str = "2d",
    ):
        if not isinstance(array, np.ndarray):
            raise ValueError("Texture storage must be a NumPy array")
        format = format or rgba8
        if dimension not in {"2d", "cube"}:
            raise ValueError("Texture dimension must be '2d' or 'cube'")
        if format._depth:
            valid = dimension == "2d" and array.dtype == np.float32 and array.ndim == 2
            expected = "contiguous float32 (height, width)"
        elif dimension == "cube":
            valid = (
                array.dtype == np.uint8
                and array.ndim == 4
                and array.shape[0] == 6
                and array.shape[1] == array.shape[2]
                and array.shape[3] == 4
            )
            expected = "contiguous uint8 (6, size, size, 4)"
        else:
            valid = array.dtype == np.uint8 and array.ndim == 3 and array.shape[2] == 4
            expected = "contiguous uint8 (height, width, 4)"
        if not isinstance(array, np.ndarray) or not valid or not array.flags.c_contiguous:
            raise ValueError(f"Texture storage must be {expected}")
        self._array = np.array(array, copy=True, order="C")
        self._format = format
        self._dimension = dimension
        self._native_texture: Any | None = None
        self._native_generation = -1
        self._host_dirty = not format._depth
        self._device_dirty = False
        _session_state()._runtime_children.add(self)

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr("Texture", arguments)

    @classmethod
    def zeros(cls, *, shape: tuple[int, int], format: _TextureFormat | None = None) -> Texture:
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape)
        ):
            raise ValueError("Texture shape must contain two positive dimensions")
        if format is not None and format is not depth32:
            raise ValueError("unsupported Texture format")
        if format is depth32:
            return cls(np.zeros(shape, dtype=np.float32), format=format)
        return cls(np.zeros((*shape, 4), dtype=np.uint8))

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> Texture:
        return cls(array)

    @classmethod
    def cube(cls, faces: np.ndarray) -> Texture:
        return cls(faces, dimension="cube")

    @property
    def shape(self) -> tuple[int, int]:
        if self._dimension == "cube":
            return self._array.shape[1:3]
        return self._array.shape[:2]

    def copy_from_numpy(self, array: np.ndarray) -> None:
        if self._format._depth:
            raise RuntimeError("depth Texture storage is produced by rendering and cannot be uploaded from the host")
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.uint8
            or array.shape != self._array.shape
            or not array.flags.c_contiguous
        ):
            raise ValueError("texture upload requires matching uint8 shape and contiguity")
        np.copyto(self._array, array)
        self._host_dirty = True
        self._device_dirty = False

    def to_numpy(self) -> np.ndarray:
        if self._device_dirty:
            if self._native_texture is None:
                raise RuntimeError("device-dirty Texture has no allocation")
            downloaded = np.frombuffer(self._native_texture.download(), dtype=self._array.dtype).reshape(
                self._array.shape
            )
            np.copyto(self._array, downloaded)
            self._device_dirty = False
            self._host_dirty = False
        return self._array.copy(order="C")

    def _resident_texture(self) -> Any:
        state = _session_state()
        if state._native_runtime is None:
            raise RuntimeError("Texture requires an initialized native runtime")
        if self._native_texture is None or self._native_generation != state._runtime_generation:
            height, width = self.shape
            if state._rhi_host is None:
                raise RuntimeError("Texture requires a GPU RHI host")
            native_format = getattr(state._native.TextureFormat, self._format._native_name)
            native_dimension = (
                state._native.TextureDimension.CUBE
                if self._dimension == "cube"
                else state._native.TextureDimension.TEXTURE_2D
            )
            self._native_texture = state._rhi_host.create_image(width, height, native_format, native_dimension)
            self._native_generation = state._runtime_generation
            self._host_dirty = not self._format._depth
            self._device_dirty = False
        if self._host_dirty:
            self._native_texture.upload(self._array.tobytes(order="C"))
            self._host_dirty = False
        return self._native_texture

    def _mark_device_dirty(self) -> None:
        self._device_dirty = True
        self._host_dirty = False


class SamplerState:
    """Immutable runtime sampler bound to a shader ``Sampler`` parameter."""

    def __init__(self, *, address: str = "repeat"):
        if address not in {"repeat", "clamp_to_edge", "mirrored_repeat"}:
            raise ValueError("sampler address must be 'repeat', 'clamp_to_edge', or 'mirrored_repeat'")
        self._address = address
        self._native_sampler: Any | None = None
        self._native_generation = -1
        _session_state()._runtime_children.add(self)

    def _resident_sampler(self) -> Any:
        state = _session_state()
        if state._native_runtime is None or state._rhi_host is None:
            raise RuntimeError("SamplerState requires an initialized GPU RHI runtime")
        if self._native_sampler is None or self._native_generation != state._runtime_generation:
            address = {
                "repeat": state._native.SamplerAddressMode.REPEAT,
                "clamp_to_edge": state._native.SamplerAddressMode.CLAMP_TO_EDGE,
                "mirrored_repeat": state._native.SamplerAddressMode.MIRRORED_REPEAT,
            }[self._address]
            self._native_sampler = state._rhi_host.create_sampler(address)
            self._native_generation = state._runtime_generation
        return self._native_sampler


def sampler(*, address: str = "repeat") -> SamplerState:
    return SamplerState(address=address)


class RenderTarget:
    """Backend-neutral collection of color and render-only depth attachments."""

    def __init__(self, *, shape: tuple[int, int]):
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape)
        ):
            raise ValueError("RenderTarget shape must contain two positive dimensions")
        self._shape = shape
        self._colors: dict[int, Texture] = {}
        self._depth_format: _TextureFormat | None = None
        self._depth_texture: Texture | None = None
        self._native_depth: Any | None = None
        self._native_depth_generation = -1
        _session_state()._runtime_children.add(self)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def attach_color(self, location: int, texture: Texture) -> RenderTarget:
        if not isinstance(location, int) or isinstance(location, bool) or not 0 <= location < 2**32:
            raise ValueError("color attachment location must be a non-negative u32")
        if location in self._colors:
            raise ValueError(f"color attachment location {location} is already occupied")
        if not isinstance(texture, Texture):
            raise TypeError("color attachment must be a Texture")
        if texture._format._depth or texture._dimension != "2d":
            raise ValueError("color attachment must be a two-dimensional color Texture")
        if texture.shape != self.shape:
            raise ValueError("color attachment dimensions must match RenderTarget shape")
        self._colors[location] = texture
        return self

    def attach_depth(
        self,
        *,
        format: _TextureFormat | None = None,
        texture: Texture | None = None,
    ) -> RenderTarget:
        if self._depth_format is not None:
            raise ValueError("RenderTarget already has a depth attachment")
        if (format is None) == (texture is None):
            raise ValueError("depth attachment requires exactly one of format or texture")
        if texture is not None:
            if texture._format is not depth32 or texture._dimension != "2d":
                raise ValueError("depth attachment texture must be a two-dimensional depth32 Texture")
            if texture.shape != self.shape:
                raise ValueError("depth attachment dimensions must match RenderTarget shape")
            self._depth_texture = texture
            self._depth_format = depth32
            return self
        if format is not depth32:
            raise ValueError("depth attachment requires a depth format")
        self._depth_format = format
        return self

    def _color_attachments(self) -> tuple[tuple[int, Texture], ...]:
        return tuple(sorted(self._colors.items()))

    def _resident_depth_attachment(self) -> Any | None:
        if self._depth_format is None:
            return None
        if self._depth_texture is not None:
            return self._depth_texture._resident_texture()
        state = _session_state()
        if state._native_runtime is None or state._rhi_host is None or state._native is None:
            raise RuntimeError("RenderTarget depth attachment requires an initialized GPU RHI runtime")
        if self._native_depth is None or self._native_depth_generation != state._runtime_generation:
            height, width = self.shape
            native_format = getattr(state._native.TextureFormat, self._depth_format._native_name)
            self._native_depth = state._rhi_host.create_attachment_image(
                width,
                height,
                native_format,
                state._native.IMAGE_DEPTH_STENCIL_ATTACHMENT,
            )
            self._native_depth_generation = state._runtime_generation
        return self._native_depth


__all__ = [
    "RawBuffer",
    "RenderTarget",
    "SamplerState",
    "TensorLayout",
    "TensorStorage",
    "TensorView",
    "Texture",
    "depth32",
    "sampler",
]
