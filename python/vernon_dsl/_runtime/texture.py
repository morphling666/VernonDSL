from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..types import TypeExpr
from .resource_common import _session_state


class _DirtyMipSet:
    """Set-like adapter for runtime-owned texture dirty state."""

    def __init__(self, mip_levels: int, dirty: Any = ()) -> None:
        self._native = _session_state()._native._DirtyIndexSet(mip_levels)
        self.update(dirty)

    def __iter__(self) -> Any:
        return iter(self._native.indices)

    def __contains__(self, mip_level: object) -> bool:
        return isinstance(mip_level, int) and not isinstance(mip_level, bool) and mip_level in self._native

    def __bool__(self) -> bool:
        return bool(self._native)

    def add(self, mip_level: int) -> None:
        self._native.add(mip_level)

    def discard(self, mip_level: int) -> None:
        self._native.discard(mip_level)

    def update(self, mip_levels: Any) -> None:
        self._native.update(list(mip_levels))

    def difference_update(self, mip_levels: Any) -> None:
        self._native.difference_update(list(mip_levels))

    def clear(self) -> None:
        self._native.clear()


class _TextureResource:
    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _resident_texture(self) -> Any:
        raise NotImplementedError

    def _resident_view(self) -> Any:
        raise NotImplementedError

    def _mark_device_dirty(self) -> None:
        raise NotImplementedError

    def _ensure_host_mutation_allowed(self) -> None:
        with self._borrow_lock:
            if self._active_borrows:
                raise RuntimeError("host mutation is forbidden while a device dispatch borrows Texture")

    def _ensure_host_read_allowed(self) -> None:
        with self._borrow_lock:
            if any(access != "read" for _, _, access in self._active_borrows):
                raise RuntimeError("host reads are forbidden while a device dispatch writes Texture")


@dataclass(frozen=True)
class TextureFormat:
    name: str
    _native_name: str
    dtype: np.dtype[Any]
    channels: int
    storage: bool = True
    aspects: frozenset[str] = frozenset({"color"})


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
d32_float = TextureFormat(
    "d32_float",
    "D32_FLOAT",
    np.dtype(np.float32),
    1,
    False,
    frozenset({"depth"}),
)

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
    d32_float,
)
_TEXTURE_FORMAT_SET = frozenset(_TEXTURE_FORMATS)
_TEXTURE_USAGES = frozenset(
    {
        "sampled",
        "storage",
        "transfer_source",
        "transfer_destination",
        "color_attachment",
        "depth_stencil_attachment",
    }
)


def _checked_texture_shape(shape: tuple[int, ...], dimension: str) -> tuple[int, ...]:
    expected_rank = 3 if dimension == "3d" else 2
    if (
        not isinstance(shape, tuple)
        or len(shape) != expected_rank
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape)
    ):
        raise ValueError(f"{dimension} Texture shape must contain {expected_rank} positive dimensions")
    return shape


class Texture(_TextureResource):
    """GPU image resource with optional host-backed contents."""

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
        logical_shape = self._logical_shape(array.shape, dimension, format.channels)
        resolved_usage = self._validate_configuration(logical_shape, format, dimension, mip_levels, usage)
        self._initialize(
            logical_shape,
            format,
            dimension,
            mip_levels,
            resolved_usage,
            np.array(array, copy=True, order="C"),
        )

    @staticmethod
    def _validate_configuration(
        shape: tuple[int, ...],
        format: TextureFormat,
        dimension: str,
        mip_levels: int,
        usage: tuple[str, ...] | None,
    ) -> frozenset[str]:
        if format not in _TEXTURE_FORMAT_SET:
            raise ValueError("unsupported Texture format")
        if dimension not in {"2d", "3d", "cube"}:
            raise ValueError("Texture dimension must be '2d', '3d', or 'cube'")
        expected_rank = 3 if dimension == "3d" else 2
        if len(shape) != expected_rank or any(value <= 0 for value in shape):
            raise ValueError(f"{dimension} Texture shape must contain {expected_rank} positive dimensions")
        if not isinstance(mip_levels, int) or isinstance(mip_levels, bool) or mip_levels <= 0:
            raise ValueError("Texture mip_levels must be a positive integer")
        max_mip_levels = max(shape).bit_length()
        if mip_levels > max_mip_levels:
            raise ValueError(f"Texture mip_levels cannot exceed {max_mip_levels} for shape {shape}")
        if usage is None:
            resolved_usage = {"sampled", "transfer_source", "transfer_destination"}
            if dimension == "2d" and format.aspects == frozenset({"color"}):
                resolved_usage.add("color_attachment")
        else:
            if not isinstance(usage, tuple) or not usage or any(item not in _TEXTURE_USAGES for item in usage):
                raise ValueError(f"Texture usage must be a non-empty tuple containing {sorted(_TEXTURE_USAGES)}")
            resolved_usage = set(usage)
        if "storage" in resolved_usage and not format.storage:
            raise ValueError(f"Texture format {format.name!r} does not support storage usage")
        if "color_attachment" in resolved_usage and (dimension != "2d" or format.aspects != frozenset({"color"})):
            raise ValueError("color_attachment usage requires a two-dimensional color Texture")
        if "depth_stencil_attachment" in resolved_usage and (dimension != "2d" or "depth" not in format.aspects):
            raise ValueError("depth_stencil_attachment usage requires a two-dimensional depth Texture")
        if {"color_attachment", "depth_stencil_attachment"}.issubset(resolved_usage):
            raise ValueError("Texture cannot be both a color and depth/stencil attachment")
        return frozenset(resolved_usage)

    def _initialize(
        self,
        shape: tuple[int, ...],
        format: TextureFormat,
        dimension: str,
        mip_levels: int,
        usage: frozenset[str],
        host_array: np.ndarray | None,
    ) -> None:
        self._borrow_lock = threading.RLock()
        self._active_borrows: list[tuple[object, object, str]] = []
        self._shape = shape
        self._array = host_array
        self._mip_arrays: dict[int, np.ndarray] = {} if host_array is None else {0: host_array}
        self._format = format
        self._dimension = dimension
        self._mip_levels = mip_levels
        self._usage = usage
        self._native_texture: Any | None = None
        self._native_view: Any | None = None
        self._native_generation = -1
        self._host_dirty_mips = _DirtyMipSet(mip_levels, self._mip_arrays)
        self._device_dirty_mips = _DirtyMipSet(mip_levels)
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
        if mip_level == 0:
            self._array = target
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
    def device(
        cls,
        *,
        shape: tuple[int, ...],
        format: TextureFormat = rgba8_unorm,
        dimension: str = "2d",
        mip_levels: int = 1,
        usage: tuple[str, ...] | None = None,
    ) -> Texture:
        """Allocate a device-only Texture without creating a host mirror."""
        if dimension not in {"2d", "3d"}:
            raise ValueError("Texture.device dimension must be '2d' or '3d'")
        shape = _checked_texture_shape(shape, dimension)
        if format not in _TEXTURE_FORMAT_SET:
            raise ValueError("unsupported Texture format")
        if usage is None:
            if dimension != "2d":
                usage = ("sampled",)
            elif "depth" in format.aspects:
                usage = ("sampled", "depth_stencil_attachment")
            else:
                usage = ("sampled", "color_attachment")
        resolved_usage = cls._validate_configuration(shape, format, dimension, mip_levels, usage)
        texture = cls.__new__(cls)
        texture._initialize(shape, format, dimension, mip_levels, resolved_usage, None)
        return texture

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
        shape = _checked_texture_shape(shape, dimension)
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
        return cls(
            array,
            format=format,
            dimension=dimension,
            mip_levels=mip_levels,
            usage=usage,
        )

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
        return self._shape

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

    def view(
        self,
        *,
        format: TextureFormat | None = None,
        dimension: str | None = None,
        base_mip_level: int = 0,
        mip_level_count: int | None = None,
        base_array_layer: int = 0,
        array_layer_count: int | None = None,
        aspects: tuple[str, ...] | None = None,
    ) -> TextureView:
        return TextureView(
            self,
            format=format,
            dimension=dimension,
            base_mip_level=base_mip_level,
            mip_level_count=mip_level_count,
            base_array_layer=base_array_layer,
            array_layer_count=array_layer_count,
            aspects=aspects,
        )

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
            if mip_level == 0:
                self._array = destination
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
        self._mip_arrays = {} if self._array is None else {0: self._array}
        self._host_dirty_mips.clear()
        self._device_dirty_mips.update(range(1, self._mip_levels))

    def _release_runtime_native(self) -> None:
        if "transfer_source" in self._usage:
            for mip_level in sorted(self._device_dirty_mips):
                self._download_device_region(
                    mip_level,
                    (0,) * len(self._mip_shape(mip_level)),
                    self._mip_shape(mip_level),
                )
        else:
            self._device_dirty_mips.clear()
        self._native_texture = None
        self._native_view = None
        self._native_generation = -1

    def _resident_texture(self) -> Any:
        state = _session_state()
        if state._native_runtime is None:
            raise RuntimeError("Texture requires an initialized native runtime")
        if self._native_texture is None or self._native_generation != state._runtime_generation:
            if self._native_texture is not None and "transfer_source" in self._usage:
                for mip_level in sorted(self._device_dirty_mips):
                    self._download_device_region(
                        mip_level,
                        (0,) * len(self._mip_shape(mip_level)),
                        self._mip_shape(mip_level),
                    )
            else:
                self._device_dirty_mips.clear()
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
            self._native_view = None
            self._native_generation = state._runtime_generation
            self._host_dirty_mips.clear()
            self._host_dirty_mips.update(self._mip_arrays)
            self._device_dirty_mips.clear()
        assert self._native_texture is not None
        for mip_level in sorted(self._host_dirty_mips):
            array = self._mip_arrays[mip_level]
            self._native_texture.upload(array.tobytes(order="C"), mip_level)
        self._host_dirty_mips.clear()
        return self._native_texture

    def _resident_view(self) -> Any:
        texture = self._resident_texture()
        if self._native_view is None:
            state = _session_state()
            native_format = getattr(state._native.TextureFormat, self._format._native_name)
            native_dimension = {
                "2d": state._native.TextureDimension.TEXTURE_2D,
                "3d": state._native.TextureDimension.TEXTURE_3D,
                "cube": state._native.TextureDimension.CUBE,
            }[self._dimension]
            aspect_names = {
                "color": "IMAGE_ASPECT_COLOR",
                "depth": "IMAGE_ASPECT_DEPTH",
                "stencil": "IMAGE_ASPECT_STENCIL",
            }
            native_aspects = sum(int(getattr(state._native, aspect_names[value])) for value in self._format.aspects)
            self._native_view = texture.create_view(
                native_format,
                native_dimension,
                0,
                self._mip_levels,
                0,
                6 if self._dimension == "cube" else 1,
                native_aspects,
            )
        return self._native_view

    def _native_usage(self) -> int:
        state = _session_state()
        names = {
            "sampled": "IMAGE_SAMPLED",
            "storage": "IMAGE_STORAGE",
            "transfer_source": "IMAGE_TRANSFER_SOURCE",
            "transfer_destination": "IMAGE_TRANSFER_DESTINATION",
            "color_attachment": "IMAGE_COLOR_ATTACHMENT",
            "depth_stencil_attachment": "IMAGE_DEPTH_STENCIL_ATTACHMENT",
        }
        return sum(int(getattr(state._native, names[item])) for item in self._usage)

    def _mark_device_dirty(self) -> None:
        self._device_dirty_mips.add(0)
        self._host_dirty_mips.discard(0)


class TextureView(_TextureResource):
    """Shader-visible subresource view retaining its Texture owner."""

    def __init__(
        self,
        owner: Texture,
        *,
        format: TextureFormat | None = None,
        dimension: str | None = None,
        base_mip_level: int = 0,
        mip_level_count: int | None = None,
        base_array_layer: int = 0,
        array_layer_count: int | None = None,
        aspects: tuple[str, ...] | None = None,
    ):
        if not isinstance(owner, Texture):
            raise TypeError("TextureView owner must be a Texture")
        format = owner.format if format is None else format
        dimension = owner.dimension if dimension is None else dimension
        if format not in _TEXTURE_FORMAT_SET or dimension not in {"2d", "3d", "cube"}:
            raise ValueError("TextureView format or dimension is unsupported")
        if format != owner.format and {format, owner.format} != {
            rgba8_unorm,
            rgba8_srgb,
        }:
            raise ValueError("TextureView format is incompatible with its owner")
        if dimension != owner.dimension and not (owner.dimension == "cube" and dimension == "2d"):
            raise ValueError("TextureView dimension is incompatible with its owner")
        if (
            not isinstance(base_mip_level, int)
            or isinstance(base_mip_level, bool)
            or not 0 <= base_mip_level < owner.mip_levels
        ):
            raise ValueError("TextureView base_mip_level is out of range")
        mip_level_count = owner.mip_levels - base_mip_level if mip_level_count is None else mip_level_count
        owner_layers = 6 if owner.dimension == "cube" else 1
        if (
            not isinstance(base_array_layer, int)
            or isinstance(base_array_layer, bool)
            or not 0 <= base_array_layer < owner_layers
        ):
            raise ValueError("TextureView base_array_layer is out of range")
        array_layer_count = owner_layers - base_array_layer if array_layer_count is None else array_layer_count
        if (
            not isinstance(mip_level_count, int)
            or isinstance(mip_level_count, bool)
            or mip_level_count <= 0
            or mip_level_count > owner.mip_levels - base_mip_level
            or not isinstance(array_layer_count, int)
            or isinstance(array_layer_count, bool)
            or array_layer_count <= 0
            or array_layer_count > owner_layers - base_array_layer
        ):
            raise ValueError("TextureView subresource range is out of bounds")
        if dimension == "2d" and array_layer_count != 1:
            raise ValueError("2D TextureView must select exactly one array layer")
        if dimension == "3d" and (base_array_layer != 0 or array_layer_count != 1):
            raise ValueError("3D TextureView cannot select array layers")
        if dimension == "cube" and (base_array_layer != 0 or array_layer_count != 6):
            raise ValueError("cube TextureView must select all six faces")
        if aspects is None:
            aspects = tuple(sorted(format.aspects))
        if (
            not isinstance(aspects, tuple)
            or not aspects
            or any(value not in {"color", "depth", "stencil"} for value in aspects)
        ):
            raise ValueError("TextureView aspects must select color, depth, or stencil")
        if not set(aspects).issubset(format.aspects):
            raise ValueError(f"TextureView aspects are incompatible with {format.name}")
        self._owner = owner
        self._format = format
        self._dimension = dimension
        self._base_mip_level = base_mip_level
        self._mip_level_count = mip_level_count
        self._base_array_layer = base_array_layer
        self._array_layer_count = array_layer_count
        self._aspects = frozenset(aspects)
        self._native_view: Any | None = None
        self._native_generation = -1

    @property
    def owner(self) -> Texture:
        return self._owner

    @property
    def shape(self) -> tuple[int, ...]:
        return self._owner._mip_shape(self._base_mip_level)

    @property
    def format(self) -> TextureFormat:
        return self._format

    @property
    def dimension(self) -> str:
        return self._dimension

    @property
    def mip_levels(self) -> int:
        return self._mip_level_count

    @property
    def usage(self) -> frozenset[str]:
        return self._owner.usage

    @property
    def aspects(self) -> frozenset[str]:
        return self._aspects

    def _resident_texture(self) -> Any:
        return self._owner._resident_texture()

    def _resident_view(self) -> Any:
        state = _session_state()
        texture = self._resident_texture()
        if self._native_view is None or self._native_generation != state._runtime_generation:
            native_format = getattr(state._native.TextureFormat, self._format._native_name)
            native_dimension = {
                "2d": state._native.TextureDimension.TEXTURE_2D,
                "3d": state._native.TextureDimension.TEXTURE_3D,
                "cube": state._native.TextureDimension.CUBE,
            }[self._dimension]
            aspect_names = {
                "color": "IMAGE_ASPECT_COLOR",
                "depth": "IMAGE_ASPECT_DEPTH",
                "stencil": "IMAGE_ASPECT_STENCIL",
            }
            native_aspects = sum(int(getattr(state._native, aspect_names[value])) for value in self._aspects)
            self._native_view = texture.create_view(
                native_format,
                native_dimension,
                self._base_mip_level,
                self._mip_level_count,
                self._base_array_layer,
                self._array_layer_count,
                native_aspects,
            )
            self._native_generation = state._runtime_generation
        return self._native_view

    def _mark_device_dirty(self) -> None:
        self._owner._device_dirty_mips.update(range(self._base_mip_level, self._base_mip_level + self._mip_level_count))
        self._owner._host_dirty_mips.difference_update(
            range(self._base_mip_level, self._base_mip_level + self._mip_level_count)
        )

    def _ensure_host_mutation_allowed(self) -> None:
        self._owner._ensure_host_mutation_allowed()

    def _ensure_host_read_allowed(self) -> None:
        self._owner._ensure_host_read_allowed()


class RenderTarget:
    """Immutable attachment-view collection that does not allocate native images."""

    def __init__(
        self,
        *,
        colors: Mapping[int, Texture | TextureView],
        depth: Texture | TextureView | None = None,
    ):
        if not isinstance(colors, Mapping):
            raise TypeError("RenderTarget colors must map locations to Texture attachments")
        normalized_colors: list[tuple[int, TextureView]] = []
        for location, attachment in colors.items():
            if not isinstance(location, int) or isinstance(location, bool) or not 0 <= location < 2**32:
                raise ValueError("color attachment location must be a non-negative u32")
            view = self._view(attachment)
            if view.dimension != "2d" or view.aspects != frozenset({"color"}):
                raise ValueError("color attachment must be a two-dimensional color TextureView")
            if view.mip_levels != 1:
                raise ValueError("color attachment TextureView must select exactly one mip level")
            if "color_attachment" not in view.usage:
                raise ValueError("color attachment Texture requires color_attachment usage")
            normalized_colors.append((location, view))
        normalized_colors.sort(key=lambda item: item[0])
        if len({location for location, _ in normalized_colors}) != len(normalized_colors):
            raise ValueError("color attachment locations must be unique")

        depth_view = None if depth is None else self._view(depth)
        if depth_view is not None:
            if depth_view.dimension != "2d" or "depth" not in depth_view.aspects:
                raise ValueError("depth attachment must be a two-dimensional depth TextureView")
            if depth_view.mip_levels != 1:
                raise ValueError("depth attachment TextureView must select exactly one mip level")
            if "depth_stencil_attachment" not in depth_view.usage:
                raise ValueError("depth attachment Texture requires depth_stencil_attachment usage")

        attachments = [view for _, view in normalized_colors]
        if depth_view is not None:
            attachments.append(depth_view)
        if not attachments:
            raise ValueError("RenderTarget requires at least one color or depth attachment")
        shape = attachments[0].shape
        if any(view.shape != shape for view in attachments[1:]):
            raise ValueError("all RenderTarget attachment dimensions must match")
        self._shape = shape
        self._colors = tuple(normalized_colors)
        self._depth = depth_view

    @staticmethod
    def _view(attachment: Texture | TextureView) -> TextureView:
        if isinstance(attachment, Texture):
            return attachment.view()
        if isinstance(attachment, TextureView):
            return attachment
        raise TypeError("RenderTarget attachments must be Texture or TextureView")

    @classmethod
    def from_attachments(
        cls,
        *,
        colors: Mapping[int, Texture | TextureView],
        depth: Texture | TextureView | None = None,
    ) -> RenderTarget:
        return cls(colors=colors, depth=depth)

    @classmethod
    def create(
        cls,
        *,
        shape: tuple[int, int],
        color_formats: Mapping[int, TextureFormat] | None = None,
        depth_format: TextureFormat | None = None,
    ) -> RenderTarget:
        shape = _checked_texture_shape(shape, "2d")
        formats = {0: rgba8_unorm} if color_formats is None else dict(color_formats)
        colors = {
            location: Texture.device(
                shape=shape,
                format=format,
            )
            for location, format in formats.items()
        }
        depth = (
            None
            if depth_format is None
            else Texture.device(
                shape=shape,
                format=depth_format,
            )
        )
        return cls(colors=colors, depth=depth)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def depth_texture(self) -> Texture:
        if self._depth is None:
            raise RuntimeError("RenderTarget has no depth attachment")
        return self._depth.owner

    @property
    def depth_view(self) -> TextureView:
        if self._depth is None:
            raise RuntimeError("RenderTarget has no depth attachment")
        return self._depth

    def color_texture(self, location: int) -> Texture:
        colors = dict(self._colors)
        if location not in colors:
            raise ValueError(f"color attachment location {location} is not occupied")
        return colors[location].owner

    def color_view(self, location: int) -> TextureView:
        colors = dict(self._colors)
        if location not in colors:
            raise ValueError(f"color attachment location {location} is not occupied")
        return colors[location]

    def _color_attachments(self) -> tuple[tuple[int, TextureView], ...]:
        return self._colors

    def _depth_attachment(self) -> TextureView | None:
        return self._depth
