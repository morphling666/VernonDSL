from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..types import TypeExpr
from .residency import _ImageRegion, _ResidencySet, _resource_transaction_scope, _ResourceControlBlock
from .session import _invocation_context, _InvocationContext

_ImageSubresource = tuple[str, int, int]


@dataclass
class _TextureCoherence:
    host_dirty_subresources: set[_ImageSubresource]
    device_dirty_subresources: set[_ImageSubresource]
    view_residencies: dict[tuple[Any, ...], _ResidencySet] = field(default_factory=dict)


class _TextureResource:
    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _resident_texture(self, context: _InvocationContext) -> Any:
        raise NotImplementedError

    def _resident_view(self, context: _InvocationContext) -> Any:
        raise NotImplementedError


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
_DEPTH_STENCIL_DTYPE = np.dtype(
    {
        "names": ("depth", "stencil"),
        "formats": (np.float32, np.uint8),
        "offsets": (0, 4),
        "itemsize": 8,
    }
)
d32_float_s8_uint = TextureFormat(
    "d32_float_s8_uint",
    "D32_FLOAT_S8_UINT",
    _DEPTH_STENCIL_DTYPE,
    1,
    False,
    frozenset({"depth", "stencil"}),
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
    d32_float_s8_uint,
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
        self._shape = shape
        self._array = host_array
        self._mip_arrays: dict[int, np.ndarray] = {} if host_array is None else {0: host_array}
        self._format = format
        self._dimension = dimension
        self._mip_levels = mip_levels
        self._usage = usage
        layers = 6 if dimension == "cube" else 1
        self._region = _ImageRegion(
            frozenset(
                (aspect, mip_level, layer)
                for aspect in format.aspects
                for mip_level in range(mip_levels)
                for layer in range(layers)
            )
        )
        initial_host_dirty = (
            {subresource for subresource in self._region.subresources if subresource[1] == 0}
            if host_array is not None
            else set()
        )
        self._control = _ResourceControlBlock(_TextureCoherence(initial_host_dirty, set()))
        self._view_key = (
            format.name,
            dimension,
            0,
            mip_levels,
            0,
            layers,
            tuple(sorted(format.aspects)),
        )

    @staticmethod
    def _logical_shape(array_shape: tuple[int, ...], dimension: str, channels: int) -> tuple[int, ...]:
        shape = array_shape[:-1] if channels != 1 else array_shape
        return shape[1:] if dimension == "cube" else shape

    def _mip_shape(self, mip_level: int) -> tuple[int, ...]:
        if not isinstance(mip_level, int) or isinstance(mip_level, bool) or not 0 <= mip_level < self._mip_levels:
            raise ValueError("Texture mip level is out of range")
        return tuple(max(1, extent >> mip_level) for extent in self.shape)

    def _mip_region(self, mip_level: int) -> _ImageRegion:
        layers = 6 if self._dimension == "cube" else 1
        return _ImageRegion(
            frozenset((aspect, mip_level, layer) for aspect in self._format.aspects for layer in range(layers))
        )

    @staticmethod
    def _dirty_mips(subresources: set[_ImageSubresource]) -> tuple[int, ...]:
        return tuple(sorted({mip_level for _, mip_level, _ in subresources}))

    @staticmethod
    def _contiguous_layers(layers: set[int]) -> tuple[tuple[int, int], ...]:
        if not layers:
            return ()
        ordered = sorted(layers)
        groups: list[tuple[int, int]] = []
        begin = previous = ordered[0]
        for layer in ordered[1:]:
            if layer != previous + 1:
                groups.append((begin, previous - begin + 1))
                begin = layer
            previous = layer
        groups.append((begin, previous - begin + 1))
        return tuple(groups)

    @staticmethod
    def _aspect_layer_groups(
        subresources: set[_ImageSubresource],
        mip_level: int,
    ) -> tuple[tuple[frozenset[str], set[int]], ...]:
        aspects_by_layer: dict[int, set[str]] = {}
        for aspect, candidate_mip, layer in subresources:
            if candidate_mip == mip_level:
                aspects_by_layer.setdefault(layer, set()).add(aspect)
        layers_by_aspects: dict[frozenset[str], set[int]] = {}
        for layer, aspects in aspects_by_layer.items():
            layers_by_aspects.setdefault(frozenset(aspects), set()).add(layer)
        return tuple(layers_by_aspects.items())

    @staticmethod
    def _aspect_mask(aspects: frozenset[str]) -> int:
        values = {"color": 1, "depth": 2, "stencil": 4}
        return sum(values[aspect] for aspect in aspects)

    def _mip_is_device_dirty(self, mip_level: int) -> bool:
        return bool(self._control.coherence.device_dirty_subresources & self._mip_region(mip_level).subresources)

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
        reservation = None
        with self._control.lock:
            residency = self._control.residencies.latest()
            if residency is None:
                raise RuntimeError("device-dirty Texture has no allocation")
            reservation = self._control.reserve_io(self._mip_region(mip_level), "write")
            texture = residency.handle
        if self._dimension == "3d":
            offset_z, offset_y, offset_x = origin
            download_depth, download_height, download_width = shape
        else:
            offset_y, offset_x = origin
            offset_z = 0
            download_height, download_width = shape
            download_depth = 1
        try:
            with self._control.lock:
                dirty_subresources = {
                    subresource
                    for subresource in self._control.coherence.device_dirty_subresources
                    if subresource[1] == mip_level
                }
            requests: list[dict[str, object]] = []
            groups: list[tuple[int, int, frozenset[str], tuple[int, ...], np.dtype[Any]]] = []
            for aspects, dirty_layers in self._aspect_layer_groups(dirty_subresources, mip_level):
                layer_groups = self._contiguous_layers(dirty_layers) if self._dimension == "cube" else ((0, 1),)
                for base_layer, layer_count in layer_groups:
                    downloaded_shape = ((layer_count, *shape) if self._dimension == "cube" else shape) + (
                        () if self._format.channels == 1 else (self._format.channels,)
                    )
                    transfer_dtype = (
                        np.dtype(np.float32)
                        if self._format is d32_float_s8_uint and aspects == {"depth"}
                        else np.dtype(np.uint8)
                        if self._format is d32_float_s8_uint and aspects == {"stencil"}
                        else self._format.dtype
                    )
                    requests.append(
                        {
                            "mip_level": mip_level,
                            "offset_x": offset_x,
                            "offset_y": offset_y,
                            "offset_z": offset_z,
                            "width": download_width,
                            "height": download_height,
                            "depth": download_depth,
                            "base_array_layer": base_layer,
                            "array_layer_count": layer_count,
                            "aspects": self._aspect_mask(aspects),
                        }
                    )
                    groups.append((base_layer, layer_count, aspects, downloaded_shape, transfer_dtype))
            downloaded_groups = [
                (base_layer, layer_count, aspects, np.frombuffer(raw, dtype=transfer_dtype).reshape(downloaded_shape))
                for raw, (base_layer, layer_count, aspects, downloaded_shape, transfer_dtype) in zip(
                    texture.download_regions(requests), groups, strict=True
                )
            ]
            with self._control.lock:
                target = self._mip_arrays.setdefault(
                    mip_level,
                    np.zeros(self._array_shape(mip_level), dtype=self._format.dtype),
                )
                if mip_level == 0:
                    self._array = target
                spatial_slices = tuple(slice(start, start + size) for start, size in zip(origin, shape, strict=True))
                for base_layer, layer_count, aspects, downloaded in downloaded_groups:
                    slices = spatial_slices
                    if self._dimension == "cube":
                        slices = (slice(base_layer, base_layer + layer_count), *slices)
                    if self._format.channels != 1:
                        slices = (*slices, slice(None))
                    destination = (
                        target[next(iter(aspects))]
                        if self._format is d32_float_s8_uint and len(aspects) == 1
                        else target
                    )
                    np.copyto(destination[slices], downloaded)
                if origin == (0,) * len(shape) and shape == self._mip_shape(mip_level):
                    self._control.coherence.device_dirty_subresources.difference_update(dirty_subresources)
        finally:
            with self._control.lock:
                self._control.release_io(reservation)

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
        with self._control.host_access(self._mip_region(mip_level), "write", "Texture"):
            coherence = self._control.coherence
            if self._mip_is_device_dirty(mip_level) and not full_region:
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
            mip_subresources = self._mip_region(mip_level).subresources
            coherence.host_dirty_subresources.update(mip_subresources)
            coherence.device_dirty_subresources.difference_update(mip_subresources)
            self._control.publish_host_write(
                host_complete=not coherence.device_dirty_subresources,
                recovers_unknown=self._mip_levels == 1 and full_region,
            )

    def download(
        self,
        *,
        mip_level: int = 0,
        origin: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
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
        with self._control.host_access(self._mip_region(mip_level), "read", "Texture"):
            if self._mip_is_device_dirty(mip_level):
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
        with _invocation_context() as context:
            if self._mip_levels < 2:
                raise RuntimeError("Texture has no mip chain to generate")
            if not {"transfer_source", "transfer_destination"}.issubset(self._usage):
                raise RuntimeError("mipmap generation requires transfer_source and transfer_destination usage")
            written = self.view(base_mip_level=1, mip_level_count=self._mip_levels - 1)
            with _resource_transaction_scope(
                [
                    (1, self, self._mip_region(0), "read"),
                    (0, self, written._region, "write"),
                ],
                context,
            ) as transaction:
                texture = self._resident_texture(context)
                try:
                    texture.generate_mipmaps()
                    with self._control.lock:
                        self._mip_arrays = {} if self._array is None else {0: self._array}
                        self._control.coherence.host_dirty_subresources.clear()
                    transaction.resolve_mutations({0: 2})
                except BaseException:
                    transaction.resolve_mutations({0: 3})
                    raise

    def _resident_texture(
        self,
        context: _InvocationContext,
        region: _ImageRegion | None = None,
        access: str = "read_write",
    ) -> Any:
        state = context.session
        if state.rhi_host is None:
            raise RuntimeError("Texture requires a GPU RHI host")
        reservation = None
        retry = False
        with self._control.lock:
            coherence = self._control.coherence
            residency = self._control.residencies.get(state)
            latest = self._control.residencies.latest()
            migration = bool(coherence.device_dirty_subresources) and (residency is None or residency is not latest)
            if migration and "transfer_source" not in self._usage:
                raise RuntimeError("cannot migrate device-dirty Texture without transfer_source usage")
            requested = self._region if region is None else region
            create = residency is None
            io_region = self._region if create or migration else requested
            expected_state = (
                residency.handle if residency is not None else None,
                latest.handle if latest is not None else None,
                self._control.version,
                frozenset(coherence.device_dirty_subresources),
                frozenset(coherence.host_dirty_subresources),
            )
            reservation = self._control.reserve_io(io_region, "write")
            current_residency = self._control.residencies.get(state)
            current_latest = self._control.residencies.latest()
            current_state = (
                current_residency.handle if current_residency is not None else None,
                current_latest.handle if current_latest is not None else None,
                self._control.version,
                frozenset(coherence.device_dirty_subresources),
                frozenset(coherence.host_dirty_subresources),
            )
            retry = current_state != expected_state
            if retry:
                self._control.release_io(reservation)
                reservation = None
            else:
                generation = self._control.version
                dirty_device = set(coherence.device_dirty_subresources) if migration else set()
                dirty_host = (
                    set(coherence.host_dirty_subresources)
                    if create or migration
                    else set(coherence.host_dirty_subresources & io_region.subresources)
                    if access in {"read", "read_write"}
                    else set()
                )
                source = latest.handle if migration and latest is not None else None
                texture = residency.handle if residency is not None else None
                arrays = {level: array.copy(order="C") for level, array in self._mip_arrays.items()}
        if retry:
            return self._resident_texture(context, region, access)
        try:
            downloaded: dict[int, np.ndarray] = {}
            if source is not None:
                download_requests: list[dict[str, object]] = []
                download_groups: list[tuple[int, int, int, frozenset[str], tuple[int, ...], np.dtype[Any]]] = []
                for mip_level in self._dirty_mips(dirty_device):
                    shape = self._mip_shape(mip_level)
                    target = arrays.get(
                        mip_level,
                        np.zeros(self._array_shape(mip_level), dtype=self._format.dtype),
                    ).copy()
                    for aspects, dirty_layers in self._aspect_layer_groups(dirty_device, mip_level):
                        layer_groups = self._contiguous_layers(dirty_layers) if self._dimension == "cube" else ((0, 1),)
                        for base_layer, layer_count in layer_groups:
                            downloaded_shape = ((layer_count, *shape) if self._dimension == "cube" else shape) + (
                                () if self._format.channels == 1 else (self._format.channels,)
                            )
                            transfer_dtype = (
                                np.dtype(np.float32)
                                if self._format is d32_float_s8_uint and aspects == {"depth"}
                                else np.dtype(np.uint8)
                                if self._format is d32_float_s8_uint and aspects == {"stencil"}
                                else self._format.dtype
                            )
                            offset_x = offset_y = offset_z = 0
                            depth, height, width = shape if self._dimension == "3d" else (1, *shape)
                            download_requests.append(
                                {
                                    "mip_level": mip_level,
                                    "offset_x": offset_x,
                                    "offset_y": offset_y,
                                    "offset_z": offset_z,
                                    "width": width,
                                    "height": height,
                                    "depth": depth,
                                    "base_array_layer": base_layer,
                                    "array_layer_count": layer_count,
                                    "aspects": self._aspect_mask(aspects),
                                }
                            )
                            download_groups.append(
                                (mip_level, base_layer, layer_count, aspects, downloaded_shape, transfer_dtype)
                            )
                    downloaded[mip_level] = target
                for raw, (mip_level, base_layer, layer_count, aspects, downloaded_shape, transfer_dtype) in zip(
                    source.download_regions(download_requests), download_groups, strict=True
                ):
                    values = np.frombuffer(raw, dtype=transfer_dtype).reshape(downloaded_shape)
                    target = downloaded[mip_level]
                    destination = (
                        target[next(iter(aspects))]
                        if self._format is d32_float_s8_uint and len(aspects) == 1
                        else target
                    )
                    if self._dimension == "cube":
                        destination[base_layer : base_layer + layer_count] = values
                    else:
                        destination[...] = values
            arrays.update(downloaded)
            if texture is None:
                if self._dimension == "3d":
                    depth, height, width = self.shape
                else:
                    height, width = self.shape
                    depth = 1
                native_format = getattr(state.native.TextureFormat, self._format._native_name)
                native_dimension = {
                    "2d": state.native.TextureDimension.TEXTURE_2D,
                    "3d": state.native.TextureDimension.TEXTURE_3D,
                    "cube": state.native.TextureDimension.CUBE,
                }[self._dimension]
                texture = state.rhi_host.create_image(
                    width,
                    height,
                    native_format,
                    native_dimension,
                    depth,
                    self._mip_levels,
                    self._native_usage(state),
                )
                dirty_host.update(
                    subresource for mip_level in arrays for subresource in self._mip_region(mip_level).subresources
                )
            elif migration:
                dirty_host.update(
                    subresource for mip_level in arrays for subresource in self._mip_region(mip_level).subresources
                )
            upload_requests: list[dict[str, object]] = []
            for mip_level in self._dirty_mips(dirty_host):
                shape = self._mip_shape(mip_level)
                depth, height, width = shape if self._dimension == "3d" else (1, *shape)
                for aspects, dirty_layers in self._aspect_layer_groups(dirty_host, mip_level):
                    layer_groups = self._contiguous_layers(dirty_layers) if self._dimension == "cube" else ((0, 1),)
                    for base_layer, layer_count in layer_groups:
                        array = arrays[mip_level]
                        if self._dimension == "cube":
                            array = array[base_layer : base_layer + layer_count]
                        if self._format is d32_float_s8_uint and len(aspects) == 1:
                            array = np.ascontiguousarray(array[next(iter(aspects))])
                        upload_requests.append(
                            {
                                "data": array.tobytes(order="C"),
                                "mip_level": mip_level,
                                "offset_x": 0,
                                "offset_y": 0,
                                "offset_z": 0,
                                "width": width,
                                "height": height,
                                "depth": depth,
                                "base_array_layer": base_layer,
                                "array_layer_count": layer_count,
                                "aspects": self._aspect_mask(aspects),
                            }
                        )
            if upload_requests:
                texture.upload_regions(upload_requests)
            with self._control.lock:
                if residency is None:
                    residency = self._control.residencies.replace(state, texture)
                self._mip_arrays.update(downloaded)
                coherence.device_dirty_subresources.difference_update(dirty_device)
                coherence.host_dirty_subresources.difference_update(dirty_host)
                residency.version = max(residency.version, generation)
                if not coherence.device_dirty_subresources:
                    self._control.host_version = generation
                return residency.handle
        finally:
            with self._control.lock:
                self._control.release_io(reservation)

    def _device_claim_region(self, context: _InvocationContext, region: _ImageRegion, access: str) -> _ImageRegion:
        if context.session.rhi_host is None:
            return region
        residency = self._control.residencies.get(context.session)
        latest = self._control.residencies.latest()
        if residency is None or (self._control.coherence.device_dirty_subresources and residency is not latest):
            return self._region
        return region

    def _resident_view(self, context: _InvocationContext) -> Any:
        return self._resident_image_view(
            context,
            self._view_key,
            self._region,
            self._format,
            self._dimension,
            0,
            self._mip_levels,
            0,
            6 if self._dimension == "cube" else 1,
            self._format.aspects,
        )

    def _resident_image_view(
        self,
        context: _InvocationContext,
        view_key: tuple[Any, ...],
        region: _ImageRegion,
        format: TextureFormat,
        dimension: str,
        base_mip_level: int,
        mip_level_count: int,
        base_array_layer: int,
        array_layer_count: int,
        aspects: frozenset[str],
    ) -> Any:
        texture = self._resident_texture(context, region)
        reservation = None
        with self._control.lock:
            residencies = self._control.coherence.view_residencies.get(view_key)
            if residencies is None:
                residencies = _ResidencySet()
                self._control.coherence.view_residencies[view_key] = residencies
            view_residency = residencies.get(context.session)
            if view_residency is not None:
                return view_residency.handle
            reservation = self._control.reserve_io(region, "write")
            view_residency = residencies.get(context.session)
            if view_residency is not None:
                self._control.release_io(reservation)
                return view_residency.handle
        try:
            state = context.session
            native_format = getattr(state.native.TextureFormat, format._native_name)
            native_dimension = {
                "2d": state.native.TextureDimension.TEXTURE_2D,
                "3d": state.native.TextureDimension.TEXTURE_3D,
                "cube": state.native.TextureDimension.CUBE,
            }[dimension]
            aspect_names = {
                "color": "IMAGE_ASPECT_COLOR",
                "depth": "IMAGE_ASPECT_DEPTH",
                "stencil": "IMAGE_ASPECT_STENCIL",
            }
            native_aspects = sum(int(getattr(state.native, aspect_names[value])) for value in aspects)
            handle = texture.create_view(
                native_format,
                native_dimension,
                base_mip_level,
                mip_level_count,
                base_array_layer,
                array_layer_count,
                native_aspects,
            )
            with self._control.lock:
                view_residency = residencies.replace(context.session, handle)
            return view_residency.handle
        finally:
            with self._control.lock:
                self._control.release_io(reservation)

    def _native_usage(self, state: Any) -> int:
        names = {
            "sampled": "IMAGE_SAMPLED",
            "storage": "IMAGE_STORAGE",
            "transfer_source": "IMAGE_TRANSFER_SOURCE",
            "transfer_destination": "IMAGE_TRANSFER_DESTINATION",
            "color_attachment": "IMAGE_COLOR_ATTACHMENT",
            "depth_stencil_attachment": "IMAGE_DEPTH_STENCIL_ATTACHMENT",
        }
        return sum(int(getattr(state.native, names[item])) for item in self._usage)

    def _publish_device_write(
        self,
        context: _InvocationContext,
        regions: tuple[_ImageRegion, ...],
    ) -> None:
        if not regions:
            return
        self._control.publish_device_write(context.session)
        written = {subresource for region in regions for subresource in region.subresources}
        coherence = self._control.coherence
        coherence.device_dirty_subresources.update(written)
        coherence.host_dirty_subresources.difference_update(written)


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
        self._view_key = (
            format.name,
            dimension,
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            tuple(sorted(self._aspects)),
        )
        self._region = _ImageRegion(
            frozenset(
                (aspect, mip_level, layer)
                for aspect in self._aspects
                for mip_level in range(
                    self._base_mip_level,
                    self._base_mip_level + self._mip_level_count,
                )
                for layer in range(
                    self._base_array_layer,
                    self._base_array_layer + self._array_layer_count,
                )
            )
        )

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

    def _resident_texture(self, context: _InvocationContext) -> Any:
        return self._owner._resident_texture(context, self._region)

    def _resident_view(self, context: _InvocationContext) -> Any:
        return self._owner._resident_image_view(
            context,
            self._view_key,
            self._region,
            self._format,
            self._dimension,
            self._base_mip_level,
            self._mip_level_count,
            self._base_array_layer,
            self._array_layer_count,
            self._aspects,
        )


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
