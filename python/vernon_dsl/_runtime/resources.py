"""Compatibility facade for runtime resource types."""

from __future__ import annotations

from .sampler import SamplerState, sampler
from .tensor import (
    RawBuffer,
    TensorLayout,
    TensorStorage,
    TensorView,
)
from .texture import (
    RenderTarget,
    Texture,
    TextureFormat,
    TextureView,
    d32_float,
    r8_unorm,
    r11g11b10_float,
    r16_float,
    r32_float,
    rg8_unorm,
    rgb8_unorm,
    rgba8_srgb,
    rgba8_unorm,
    rgba16_float,
    rgba32_float,
)

__all__ = [
    "RawBuffer",
    "RenderTarget",
    "SamplerState",
    "TensorLayout",
    "TensorStorage",
    "TensorView",
    "Texture",
    "TextureView",
    "TextureFormat",
    "d32_float",
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
