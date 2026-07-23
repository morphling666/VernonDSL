from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

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


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple[int, ...]
    byte_strides: tuple[int, ...]
    byte_offset: int = 0
    components: tuple[int, ...] | None = None


class Tensor:
    """Contiguous row-major runtime Tensor and annotation constructor."""

    def __init__(self, array: np.ndarray):
        if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
            raise ValueError("Tensor storage must be a contiguous NumPy array")
        self._array = array
        self._native_buffer: Any | None = None
        self._native_generation = -1
        self._host_version = 1
        self._uploaded_version = 0
        self._device_dirty = False
        self._allocation_count = 0
        self._upload_count = 0
        self._download_count = 0
        _session_state()._runtime_children.add(self)

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr("Tensor", arguments)

    @staticmethod
    def _dtype(dtype: _Scalar) -> np.dtype[Any]:
        if not isinstance(dtype, _Scalar) or dtype.name not in _NUMPY_DTYPES:
            raise TypeError("dtype must be a Vernon scalar type")
        return _NUMPY_DTYPES[dtype.name]

    @classmethod
    def zeros(cls, *, dtype: _Scalar, shape: tuple[int, ...]) -> Tensor:
        return cls(np.zeros(shape, dtype=cls._dtype(dtype), order="C"))

    @classmethod
    def empty(cls, *, dtype: _Scalar, shape: tuple[int, ...]) -> Tensor:
        return cls(np.empty(shape, dtype=cls._dtype(dtype), order="C"))

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> Tensor:
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a NumPy ndarray")
        if array.dtype not in _NUMPY_DTYPES.values():
            raise TypeError(f"unsupported Tensor dtype {array.dtype}")
        return cls(np.array(array, copy=True, order="C"))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._array.dtype

    @property
    def layout(self) -> TensorLayout:
        return TensorLayout(self.shape, self._array.strides)

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
        return TensorView(self, start, len(indices))

    def to_numpy(self) -> np.ndarray:
        self.synchronize()
        return self._array.copy(order="C")

    def _borrowed_array(self) -> np.ndarray:
        self.synchronize()
        return self._array

    def copy_from_numpy(self, array: np.ndarray) -> None:
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != self.dtype
            or array.shape != self.shape
            or not array.flags.c_contiguous
        ):
            raise ValueError("upload requires matching dtype, shape, and contiguity")
        np.copyto(self._array, array)
        self._host_version += 1
        self._device_dirty = False

    def synchronize(self) -> None:
        if not self._device_dirty or self._native_buffer is None:
            return
        downloaded = np.frombuffer(self._native_buffer.download(), dtype=self.dtype).reshape(self.shape)
        np.copyto(self._array, downloaded)
        self._download_count += 1
        self._device_dirty = False
        self._host_version += 1
        self._uploaded_version = self._host_version

    def _resident_buffer(self) -> Any:
        state = _session_state()
        if state._native_runtime is None:
            raise RuntimeError("native Tensor residency requires a runtime")
        if self._native_buffer is None or self._native_generation != state._runtime_generation:
            self._native_buffer = state._native_runtime.allocate(self._array.nbytes, self.dtype.itemsize)
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


class TensorView:
    """A zero-copy contiguous component selection from a resident Tensor."""

    def __init__(self, owner: Tensor, first_component: int, component_count: int):
        self._owner = owner
        self._first_component = first_component
        self._component_count = component_count

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._owner.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return (*self._owner.shape[:-1], self._component_count)

    @property
    def layout(self) -> TensorLayout:
        offset = self._first_component * self.dtype.itemsize
        return TensorLayout(
            self.shape,
            self._owner._array.strides,
            offset,
            tuple(range(self._first_component, self._first_component + self._component_count)),
        )

    def to_numpy(self) -> np.ndarray:
        self._owner.synchronize()
        stop = self._first_component + self._component_count
        return np.array(self._owner._array[..., self._first_component : stop], copy=True, order="C")

    def _borrowed_array(self) -> np.ndarray:
        self._owner.synchronize()
        stop = self._first_component + self._component_count
        return self._owner._array[..., self._first_component : stop]

    def _resident_buffer(self) -> Any:
        return self._owner._resident_buffer()


class Texture:
    """RGBA8 two-dimensional runtime texture and annotation constructor."""

    def __init__(self, array: np.ndarray):
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.uint8
            or array.ndim != 3
            or array.shape[2] != 4
            or not array.flags.c_contiguous
        ):
            raise ValueError("Texture storage must be contiguous uint8 (height, width, 4)")
        self._array = np.array(array, copy=True, order="C")
        self._native_texture: Any | None = None
        self._native_generation = -1
        self._host_dirty = True
        self._device_dirty = False
        _session_state()._runtime_children.add(self)

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr("Texture", arguments)

    @classmethod
    def zeros(cls, *, shape: tuple[int, int]) -> Texture:
        return cls(np.zeros((*shape, 4), dtype=np.uint8))

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> Texture:
        return cls(array)

    @property
    def shape(self) -> tuple[int, int]:
        return self._array.shape[:2]

    def copy_from_numpy(self, array: np.ndarray) -> None:
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
            downloaded = np.frombuffer(self._native_texture.download(), dtype=np.uint8).reshape(self._array.shape)
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
            self._native_texture = state._native_runtime.create_texture(width, height)
            self._native_generation = state._runtime_generation
            self._host_dirty = True
            self._device_dirty = False
        if self._host_dirty:
            self._native_texture.upload(self._array.tobytes(order="C"))
            self._host_dirty = False
        return self._native_texture

    def _mark_device_dirty(self) -> None:
        self._device_dirty = True
        self._host_dirty = False


__all__ = ["Tensor", "TensorLayout", "TensorView", "Texture"]
