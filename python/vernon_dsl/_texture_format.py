from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

_DEPTH_STENCIL_DTYPE = np.dtype(
    {
        "names": ("depth", "stencil"),
        "formats": (np.float32, np.uint8),
        "offsets": (0, 4),
        "itemsize": 8,
    }
)


@dataclass(frozen=True)
class TextureFormat:
    """One row of ``source/include/VernonFormats.def``; instances live in ``vernon_dsl._formats``."""

    name: str
    _native_name: str
    component_class: str
    element: str
    channels: int
    pixel_bytes: int
    aspects: frozenset[str]
    capabilities: frozenset[str]

    @property
    def storage(self) -> bool:
        return "storage" in self.capabilities

    @property
    def integer(self) -> bool:
        return self.component_class in ("uint", "sint")

    @property
    def shader_scalar(self) -> str:
        """Scalar type a shader reads from and writes to this format."""
        return {"uint": "u32", "sint": "i32"}.get(self.component_class, "f32")

    @cached_property
    def dtype(self) -> np.dtype[Any]:
        if self.aspects == {"depth", "stencil"}:
            return _DEPTH_STENCIL_DTYPE
        return np.dtype(self.element)
