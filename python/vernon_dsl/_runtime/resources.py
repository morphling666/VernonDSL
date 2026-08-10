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


class _TextureResource:
    def __init__(self) -> None:
        self._borrow_lock = threading.RLock()
        self._active_borrows: list[tuple[object, _TextureResource, str]] = []

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _resident_texture(self) -> Any:
        raise NotImplementedError

    def _mark_device_dirty(self) -> None:
        raise NotImplementedError

    def _graph_identity(self) -> object:
        return self

    def _ensure_host_mutation_allowed(self) -> None:
        with self._borrow_lock:
            if self._active_borrows:
                raise RuntimeError("host mutation is forbidden while a device dispatch borrows Texture")

    def _ensure_host_read_allowed(self) -> None:
        with self._borrow_lock:
            if any(access != "read" for _, _, access in self._active_borrows):
                raise RuntimeError("host reads are forbidden while a device dispatch writes Texture")


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
    if isinstance(value, _TextureResource):
        if state._architecture == state.cpu:
            raise TypeError("CPU kernels do not support Texture arguments")
        if state._rhi_host is None:
            raise RuntimeError("Texture arguments require a GPU RHI host")
        return builder.rhi_texture(parameter.name, value._resident_texture())

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
        assert self._native_buffer is not None
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
        assert self._native_buffer is not None
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
    borrows: list[tuple[str, TensorStorage | TensorView | _TextureResource, str]],
) -> list[tuple[str, TensorView | _TextureResource, str]]:
    normalized: list[tuple[str, TensorView | _TextureResource, str]] = []
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


def _validate_dispatch_borrows(
    borrows: list[tuple[str, TensorStorage | TensorView | _TextureResource, str]],
) -> None:
    _normalize_dispatch_borrows(borrows)


def _dispatch_borrow_owner(resource: TensorView | _TextureResource) -> TensorStorage | RawBuffer | _TextureResource:
    return resource.owner if isinstance(resource, TensorView) else resource


def _dispatch_borrows_overlap(
    left: TensorView | _TextureResource,
    right: TensorView | _TextureResource,
) -> bool:
    if isinstance(left, TensorView) and isinstance(right, TensorView):
        return _borrow_ranges_may_overlap(left, right)
    return True


@contextmanager
def _dispatch_borrow_scope(
    borrows: list[tuple[str, TensorStorage | TensorView | _TextureResource, str]],
) -> Iterator[None]:
    """Retain owners and reject incompatible borrows until dispatch completion."""
    normalized = _normalize_dispatch_borrows(borrows)
    owners = sorted({_dispatch_borrow_owner(resource) for _, resource, _ in normalized}, key=id)
    token = object()
    for owner in owners:
        owner._borrow_lock.acquire()
    try:
        for name, resource, access in normalized:
            owner = _dispatch_borrow_owner(resource)
            for _, active_resource, active_access in owner._active_borrows:
                if access == active_access == "read":
                    continue
                if _dispatch_borrows_overlap(resource, active_resource):
                    raise RuntimeError(f"dispatch argument '{name}' conflicts with an outstanding device borrow")
        for _, resource, access in normalized:
            _dispatch_borrow_owner(resource)._active_borrows.append((token, resource, access))
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
class TextureFormat:
    name: str
    _native_name: str
    dtype: np.dtype[Any]
    channels: int
    storage: bool = True


rgba8_unorm = TextureFormat("rgba8_unorm", "RGBA8_UNORM", np.dtype(np.uint8), 4)
rgba8_srgb = TextureFormat("rgba8_srgb", "RGBA8_SRGB", np.dtype(np.uint8), 4, False)
rgba16_float = TextureFormat("rgba16_float", "RGBA16_FLOAT", np.dtype(np.float16), 4)
rgba32_float = TextureFormat("rgba32_float", "RGBA32_FLOAT", np.dtype(np.float32), 4)
r8_unorm = TextureFormat("r8_unorm", "R8_UNORM", np.dtype(np.uint8), 1)
r16_float = TextureFormat("r16_float", "R16_FLOAT", np.dtype(np.float16), 1)
r32_float = TextureFormat("r32_float", "R32_FLOAT", np.dtype(np.float32), 1)
rg8_unorm = TextureFormat("rg8_unorm", "RG8_UNORM", np.dtype(np.uint8), 2)
rgb8_unorm = TextureFormat("rgb8_unorm", "RGB8_UNORM", np.dtype(np.uint8), 3, False)
r11g11b10_float = TextureFormat("r11g11b10_float", "R11G11B10_FLOAT", np.dtype(np.uint32), 1, False)

_TEXTURE_FORMATS = (
    rgba8_unorm,
    rgba8_srgb,
    rgba16_float,
    rgba32_float,
    r8_unorm,
    r16_float,
    r32_float,
    rg8_unorm,
    rgb8_unorm,
    r11g11b10_float,
)
_TEXTURE_FORMAT_SET = frozenset(_TEXTURE_FORMATS)
_TEXTURE_USAGES = frozenset({"sampled", "storage", "transfer_source", "transfer_destination", "color_attachment"})


class Texture(_TextureResource):
    """Runtime image storage and shader texture annotation constructor."""

    def __init__(
        self,
        array: np.ndarray,
        *,
        format: TextureFormat = rgba8_unorm,
        dimension: str = "2d",
        mip_levels: int = 1,
        usage: tuple[str, ...] | None = None,
    ):
        if not isinstance(array, np.ndarray):
            raise ValueError("Texture storage must be a NumPy array")
        if format not in _TEXTURE_FORMAT_SET:
            raise ValueError("unsupported Texture format")
        if dimension not in {"2d", "3d", "cube"}:
            raise ValueError("Texture dimension must be '2d', '3d', or 'cube'")
        logical_rank = 3 if dimension in {"3d", "cube"} else 2
        channel_rank = 0 if format.channels == 1 else 1
        valid = array.dtype == format.dtype and array.ndim == logical_rank + channel_rank
        if channel_rank:
            valid = valid and array.shape[-1] == format.channels
        if dimension == "cube":
            valid = valid and array.shape[0] == 6 and array.shape[1] == array.shape[2]
        expected_shape = {
            "2d": "(height, width)",
            "3d": "(depth, height, width)",
            "cube": "(6, size, size)",
        }[dimension]
        if channel_rank:
            expected_shape = f"{expected_shape[:-1]}, {format.channels})"
        expected = f"contiguous {format.dtype.name} {expected_shape}"
        if not isinstance(array, np.ndarray) or not valid or not array.flags.c_contiguous:
            raise ValueError(f"Texture storage must be {expected}")
        if not isinstance(mip_levels, int) or isinstance(mip_levels, bool) or mip_levels <= 0:
            raise ValueError("Texture mip_levels must be a positive integer")
        logical_shape = self._logical_shape(array.shape, dimension, format.channels)
        max_mip_levels = max(logical_shape).bit_length()
        if mip_levels > max_mip_levels:
            raise ValueError(f"Texture mip_levels cannot exceed {max_mip_levels} for shape {logical_shape}")
        if usage is None:
            resolved_usage = {"sampled", "transfer_source", "transfer_destination"}
            if dimension == "2d":
                resolved_usage.add("color_attachment")
        else:
            if not isinstance(usage, tuple) or not usage or any(item not in _TEXTURE_USAGES for item in usage):
                raise ValueError(f"Texture usage must be a non-empty tuple containing {sorted(_TEXTURE_USAGES)}")
            resolved_usage = set(usage)
        if "storage" in resolved_usage and not format.storage:
            raise ValueError(f"Texture format {format.name!r} does not support storage usage")
        if "color_attachment" in resolved_usage and dimension != "2d":
            raise ValueError("color_attachment usage requires a two-dimensional Texture")
        super().__init__()
        self._array = np.array(array, copy=True, order="C")
        self._mip_arrays: dict[int, np.ndarray] = {0: self._array}
        self._format = format
        self._dimension = dimension
        self._mip_levels = mip_levels
        self._usage = frozenset(resolved_usage)
        self._native_texture: Any | None = None
        self._native_generation = -1
        self._host_dirty_mips = {0}
        self._device_dirty_mips: set[int] = set()
        _session_state()._runtime_children.add(self)

    @staticmethod
    def _logical_shape(array_shape: tuple[int, ...], dimension: str, channels: int) -> tuple[int, ...]:
        shape = array_shape[:-1] if channels != 1 else array_shape
        return shape[1:] if dimension == "cube" else shape

    def _mip_shape(self, mip_level: int) -> tuple[int, ...]:
        if not isinstance(mip_level, int) or isinstance(mip_level, bool) or not 0 <= mip_level < self._mip_levels:
            raise ValueError("Texture mip level is out of range")
        return tuple(max(1, extent >> mip_level) for extent in self.shape)

    def _array_shape(self, mip_level: int) -> tuple[int, ...]:
        logical = self._mip_shape(mip_level)
        if self._dimension == "cube":
            logical = (6, *logical)
        return (*logical, self._format.channels) if self._format.channels != 1 else logical

    def _download_device_region(
        self,
        mip_level: int,
        origin: tuple[int, ...],
        shape: tuple[int, ...],
    ) -> None:
        if self._native_texture is None:
            raise RuntimeError("device-dirty Texture has no allocation")
        if self._dimension == "3d":
            offset_z, offset_y, offset_x = origin
            download_depth, download_height, download_width = shape
        else:
            offset_y, offset_x = origin
            offset_z = 0
            download_height, download_width = shape
            download_depth = 1
        downloaded_shape = (6, *shape) if self._dimension == "cube" else shape
        if self._format.channels != 1:
            downloaded_shape = (*downloaded_shape, self._format.channels)
        downloaded = np.frombuffer(
            self._native_texture.download(
                mip_level,
                offset_x,
                offset_y,
                offset_z,
                download_width,
                download_height,
                download_depth,
            ),
            dtype=self._format.dtype,
        ).reshape(downloaded_shape)
        target = self._mip_arrays.setdefault(
            mip_level, np.zeros(self._array_shape(mip_level), dtype=self._format.dtype)
        )
        slices = tuple(slice(start, start + size) for start, size in zip(origin, shape, strict=True))
        if self._dimension == "cube":
            slices = (slice(None), *slices)
        if self._format.channels != 1:
            slices = (*slices, slice(None))
        np.copyto(target[slices], downloaded)
        if origin == (0,) * len(shape) and shape == self._mip_shape(mip_level):
            self._device_dirty_mips.discard(mip_level)

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr("Texture", arguments)

    @classmethod
    def zeros(
        cls,
        *,
        shape: tuple[int, ...],
        format: TextureFormat = rgba8_unorm,
        dimension: str = "2d",
        mip_levels: int = 1,
        usage: tuple[str, ...] | None = None,
    ) -> Texture:
        if dimension not in {"2d", "3d"}:
            raise ValueError("Texture.zeros dimension must be '2d' or '3d'")
        expected_rank = 3 if dimension == "3d" else 2
        if (
            not isinstance(shape, tuple)
            or len(shape) != expected_rank
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape)
        ):
            raise ValueError(f"{dimension} Texture shape must contain {expected_rank} positive dimensions")
        if format not in _TEXTURE_FORMAT_SET:
            raise ValueError("unsupported Texture format")
        array_shape = (*shape, format.channels) if format.channels != 1 else shape
        return cls(
            np.zeros(array_shape, dtype=format.dtype),
            format=format,
            dimension=dimension,
            mip_levels=mip_levels,
            usage=usage,
        )

    @classmethod
    def from_numpy(
        cls,
        array: np.ndarray,
        *,
        format: TextureFormat = rgba8_unorm,
        dimension: str = "2d",
        mip_levels: int = 1,
        usage: tuple[str, ...] | None = None,
    ) -> Texture:
        return cls(array, format=format, dimension=dimension, mip_levels=mip_levels, usage=usage)

    @classmethod
    def cube(
        cls,
        faces: np.ndarray,
        *,
        format: TextureFormat = rgba8_unorm,
        mip_levels: int = 1,
        usage: tuple[str, ...] | None = None,
    ) -> Texture:
        return cls(faces, format=format, dimension="cube", mip_levels=mip_levels, usage=usage)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._logical_shape(self._array.shape, self._dimension, self._format.channels)

    @property
    def format(self) -> TextureFormat:
        return self._format

    @property
    def dimension(self) -> str:
        return self._dimension

    @property
    def mip_levels(self) -> int:
        return self._mip_levels

    @property
    def usage(self) -> frozenset[str]:
        return self._usage

    def copy_from_numpy(self, array: np.ndarray) -> None:
        self.upload(array)

    def upload(
        self,
        array: np.ndarray,
        *,
        mip_level: int = 0,
        origin: tuple[int, ...] | None = None,
    ) -> None:
        self._ensure_host_mutation_allowed()
        if "transfer_destination" not in self._usage:
            raise RuntimeError("Texture was not created with transfer_destination usage")
        mip_shape = self._mip_shape(mip_level)
        if origin is None:
            origin = (0,) * len(mip_shape)
        if (
            not isinstance(origin, tuple)
            or len(origin) != len(mip_shape)
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in origin)
        ):
            raise ValueError(f"Texture upload origin must contain {len(mip_shape)} non-negative integers")
        channel_rank = 0 if self._format.channels == 1 else 1
        layer_rank = 1 if self._dimension == "cube" else 0
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != self._format.dtype
            or array.ndim != len(mip_shape) + channel_rank + layer_rank
            or (layer_rank and array.shape[0] != 6)
            or (channel_rank and array.shape[-1] != self._format.channels)
            or not array.flags.c_contiguous
        ):
            raise ValueError(
                f"Texture upload requires contiguous {self._format.dtype.name} data with "
                f"{self._format.channels} channel(s)"
            )
        region_shape = array.shape[layer_rank : -1 if channel_rank else None]
        if any(
            start >= extent or size > extent - start
            for start, size, extent in zip(origin, region_shape, mip_shape, strict=True)
        ):
            raise ValueError("Texture upload region exceeds the selected mip level")
        full_region = origin == (0,) * len(mip_shape) and region_shape == mip_shape
        if mip_level in self._device_dirty_mips and not full_region:
            self._download_device_region(mip_level, (0,) * len(mip_shape), mip_shape)
        destination = self._mip_arrays.get(mip_level)
        if destination is None:
            destination = np.zeros(self._array_shape(mip_level), dtype=self._format.dtype)
            self._mip_arrays[mip_level] = destination
        slices = tuple(slice(start, start + size) for start, size in zip(origin, region_shape, strict=True))
        if layer_rank:
            slices = (slice(None), *slices)
        if channel_rank:
            slices = (*slices, slice(None))
        np.copyto(destination[slices], array)
        state = _session_state()
        resident = self._native_texture is not None and self._native_generation == state._runtime_generation
        if resident:
            assert self._native_texture is not None
            if self._dimension == "3d":
                offset_z, offset_y, offset_x = origin
                upload_depth, upload_height, upload_width = region_shape
            else:
                offset_y, offset_x = origin
                offset_z = 0
                upload_height, upload_width = region_shape
                upload_depth = 1
            self._native_texture.upload(
                array.tobytes(order="C"),
                mip_level,
                offset_x,
                offset_y,
                offset_z,
                upload_width,
                upload_height,
                upload_depth,
            )
            self._host_dirty_mips.discard(mip_level)
        else:
            self._host_dirty_mips.add(mip_level)
        self._device_dirty_mips.discard(mip_level)

    def download(
        self,
        *,
        mip_level: int = 0,
        origin: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        self._ensure_host_read_allowed()
        if "transfer_source" not in self._usage:
            raise RuntimeError("Texture was not created with transfer_source usage")
        mip_shape = self._mip_shape(mip_level)
        if origin is None:
            origin = (0,) * len(mip_shape)
        if (
            not isinstance(origin, tuple)
            or len(origin) != len(mip_shape)
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in origin)
        ):
            raise ValueError(f"Texture download origin must contain {len(mip_shape)} non-negative integers")
        if shape is None:
            shape = tuple(extent - start for start, extent in zip(origin, mip_shape, strict=True))
        if (
            not isinstance(shape, tuple)
            or len(shape) != len(mip_shape)
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape)
            or any(
                start >= extent or size > extent - start
                for start, size, extent in zip(origin, shape, mip_shape, strict=True)
            )
        ):
            raise ValueError("Texture download region exceeds the selected mip level")
        if mip_level in self._device_dirty_mips:
            self._download_device_region(mip_level, origin, shape)
        array = self._mip_arrays.get(mip_level)
        if array is None:
            raise RuntimeError("Texture mip level has not been initialized")
        slices = tuple(slice(start, start + size) for start, size in zip(origin, shape, strict=True))
        if self._dimension == "cube":
            slices = (slice(None), *slices)
        if self._format.channels != 1:
            slices = (*slices, slice(None))
        return array[slices].copy(order="C")

    def to_numpy(self) -> np.ndarray:
        return self.download()

    def generate_mipmaps(self) -> None:
        self._ensure_host_mutation_allowed()
        if self._mip_levels < 2:
            raise RuntimeError("Texture has no mip chain to generate")
        if not {"transfer_source", "transfer_destination"}.issubset(self._usage):
            raise RuntimeError("mipmap generation requires transfer_source and transfer_destination usage")
        texture = self._resident_texture()
        texture.generate_mipmaps()
        self._mip_arrays = {0: self._array}
        self._host_dirty_mips.clear()
        self._device_dirty_mips.update(range(1, self._mip_levels))

    def _resident_texture(self) -> Any:
        state = _session_state()
        if state._native_runtime is None:
            raise RuntimeError("Texture requires an initialized native runtime")
        if self._native_texture is None or self._native_generation != state._runtime_generation:
            if self._dimension == "3d":
                depth, height, width = self.shape
            else:
                height, width = self.shape
                depth = 1
            if state._rhi_host is None:
                raise RuntimeError("Texture requires a GPU RHI host")
            native_format = getattr(state._native.TextureFormat, self._format._native_name)
            native_dimension = {
                "2d": state._native.TextureDimension.TEXTURE_2D,
                "3d": state._native.TextureDimension.TEXTURE_3D,
                "cube": state._native.TextureDimension.CUBE,
            }[self._dimension]
            self._native_texture = state._rhi_host.create_image(
                width,
                height,
                native_format,
                native_dimension,
                depth,
                self._mip_levels,
                self._native_usage(),
            )
            self._native_generation = state._runtime_generation
            self._host_dirty_mips = set(self._mip_arrays)
            self._device_dirty_mips.clear()
        assert self._native_texture is not None
        for mip_level in sorted(self._host_dirty_mips):
            array = self._mip_arrays[mip_level]
            self._native_texture.upload(array.tobytes(order="C"), mip_level)
        self._host_dirty_mips.clear()
        return self._native_texture

    def _native_usage(self) -> int:
        state = _session_state()
        names = {
            "sampled": "IMAGE_SAMPLED",
            "storage": "IMAGE_STORAGE",
            "transfer_source": "IMAGE_TRANSFER_SOURCE",
            "transfer_destination": "IMAGE_TRANSFER_DESTINATION",
            "color_attachment": "IMAGE_COLOR_ATTACHMENT",
        }
        return sum(int(getattr(state._native, names[item])) for item in self._usage)

    def _mark_device_dirty(self) -> None:
        self._device_dirty_mips.add(0)
        self._host_dirty_mips.discard(0)


class _DepthTexture(_TextureResource):
    """Sampled view of a RenderTarget-owned depth attachment."""

    def __init__(self, owner: RenderTarget):
        super().__init__()
        self._owner = owner

    @property
    def shape(self) -> tuple[int, int]:
        return self._owner.shape

    @property
    def dimension(self) -> str:
        return "2d"

    @property
    def mip_levels(self) -> int:
        return 1

    @property
    def usage(self) -> frozenset[str]:
        return frozenset({"sampled"})

    def upload(self, array: np.ndarray, *, mip_level: int = 0, origin: tuple[int, ...] | None = None) -> None:
        del array, mip_level, origin
        raise RuntimeError("depth attachments cannot be uploaded from the host")

    def copy_from_numpy(self, array: np.ndarray) -> None:
        self.upload(array)

    def download(
        self,
        *,
        mip_level: int = 0,
        origin: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        del mip_level, origin, shape
        raise RuntimeError("depth attachment readback is not exposed as Texture data")

    def to_numpy(self) -> np.ndarray:
        return self.download()

    def _resident_texture(self) -> Any:
        image = self._owner._resident_depth_attachment()
        if image is None:
            raise RuntimeError("RenderTarget has no depth attachment")
        return image

    def _mark_device_dirty(self) -> None:
        pass

    def _graph_identity(self) -> object:
        return self._owner


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

    def __init__(self, *, shape: tuple[int, ...]):
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape)
        ):
            raise ValueError("RenderTarget shape must contain two positive dimensions")
        self._shape = shape
        self._colors: dict[int, Texture] = {}
        self._has_depth = False
        self._depth_texture: _DepthTexture | None = None
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
        if texture._dimension != "2d":
            raise ValueError("color attachment must be a two-dimensional color Texture")
        if "color_attachment" not in texture._usage:
            raise ValueError("color attachment Texture requires color_attachment usage")
        if texture.shape != self.shape:
            raise ValueError("color attachment dimensions must match RenderTarget shape")
        self._colors[location] = texture
        return self

    def attach_depth(self) -> RenderTarget:
        if self._has_depth:
            raise ValueError("RenderTarget already has a depth attachment")
        self._has_depth = True
        self._depth_texture = _DepthTexture(self)
        return self

    @property
    def depth_texture(self) -> _TextureResource:
        if self._depth_texture is None:
            raise RuntimeError("RenderTarget has no depth attachment")
        return self._depth_texture

    def _color_attachments(self) -> tuple[tuple[int, Texture], ...]:
        return tuple(sorted(self._colors.items()))

    def _resident_depth_attachment(self) -> Any | None:
        if not self._has_depth:
            return None
        state = _session_state()
        if state._native_runtime is None or state._rhi_host is None or state._native is None:
            raise RuntimeError("RenderTarget depth attachment requires an initialized GPU RHI runtime")
        if self._native_depth is None or self._native_depth_generation != state._runtime_generation:
            height, width = self.shape
            self._native_depth = state._rhi_host.create_image(
                width,
                height,
                state._native.TextureFormat.D32_FLOAT,
                state._native.TextureDimension.TEXTURE_2D,
                1,
                1,
                int(state._native.IMAGE_DEPTH_STENCIL_ATTACHMENT) + int(state._native.IMAGE_SAMPLED),
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
    "TextureFormat",
    "r11g11b10_float",
    "r16_float",
    "r32_float",
    "r8_unorm",
    "rg8_unorm",
    "rgb8_unorm",
    "rgba16_float",
    "rgba32_float",
    "rgba8_srgb",
    "rgba8_unorm",
    "sampler",
]
