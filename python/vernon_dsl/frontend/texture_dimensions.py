"""Texture dimension rules shared by type parsing, inference, and lowering.

These mirror ``TextureType`` in the Vernon dialect. As in SPIR-V, arrayed coordinates append the layer as
their last component and arrayed sizes append the layer count; cube-array layers count cubes.
"""

from __future__ import annotations

TEXTURE_DIMENSIONS = ("2d", "3d", "cube", "2d_array", "cube_array")
STORAGE_TEXTURE_DIMENSIONS = ("2d", "3d", "2d_array")
SAMPLE_COORDINATE_RANK = {"2d": 2, "3d": 3, "cube": 3, "2d_array": 3, "cube_array": 4}
TEXEL_COORDINATE_RANK = {"2d": 2, "3d": 3, "2d_array": 3}
SIZE_RANK = {"2d": 2, "3d": 3, "cube": 2, "2d_array": 3, "cube_array": 3}


def dimension_list(dimensions: tuple[str, ...]) -> str:
    return ", ".join(repr(dimension) for dimension in dimensions)
