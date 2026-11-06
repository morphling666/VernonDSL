from __future__ import annotations

import functools
import importlib.util
import math
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import vernon_dsl as vd
from backend_test_matrix import BackendRequirements, BackendRow, backend_matrix_test, expand_backend_matrix_tests
from pipeline_shader import (
    colored_fragment,
    depth_fragment,
    depth_vertex,
    f32x1_fragment,
    f32x2_fragment,
    f32x3_fragment,
    float_fetch_fragment,
    i32x1_fragment,
    i32x2_fragment,
    i32x4_fragment,
    sint_fetch_fragment,
    triangle_vertex,
    u32x1_fragment,
    u32x2_fragment,
    u32x4_fragment,
    uint_fetch_fragment,
)
from vernon_dsl._formats import TEXTURE_FORMATS

SIZE = 4
COLOR_FORMATS = tuple(format for format in TEXTURE_FORMATS if format.aspects == {"color"})
DEPTH_FORMATS = tuple(format for format in TEXTURE_FORMATS if "depth" in format.aspects)
STORAGE_FORMATS = tuple(format for format in TEXTURE_FORMATS if format.storage)
GRAPHICS = BackendRequirements(graphics=True)

# Read back through a 32-bit target of the format's result class, so every component a format holds survives.
_FETCH = {
    "uint": (vd.rgba32_uint, uint_fetch_fragment, np.uint32),
    "sint": (vd.rgba32_sint, sint_fetch_fragment, np.int32),
}
_FLOAT_FETCH = (vd.rgba32_float, float_fetch_fragment, np.float32)
# A fragment writes exactly the components its target holds; no integer format has three.
_WRITE = {
    ("float", 1): f32x1_fragment,
    ("float", 2): f32x2_fragment,
    ("float", 3): f32x3_fragment,
    ("float", 4): colored_fragment,
    ("uint", 1): u32x1_fragment,
    ("uint", 2): u32x2_fragment,
    ("uint", 4): u32x4_fragment,
    ("sint", 1): i32x1_fragment,
    ("sint", 2): i32x2_fragment,
    ("sint", 4): i32x4_fragment,
}
_WRITE_DTYPE = {"uint": np.uint32, "sint": np.int32}


def _covering() -> vd.TensorStorage:
    return vd.storage.from_numpy(np.array(((-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)), dtype=np.float32))


def _component_bits(format: vd.TextureFormat) -> tuple[int, ...]:
    if format.integer:
        return format.integer_bits
    if format.name == "rgb10a2_unorm":
        return (10, 10, 10, 2)
    return (8 * np.dtype(format.element).itemsize,) * format.components


def _values(format: vd.TextureFormat) -> np.ndarray:
    """One exactly representable value per component, spanning the format's range."""
    count = format.components
    if format.component_class == "float":
        return np.array((0.25, 0.5, 1.0, 2.0)[:count])
    bits = np.array(_component_bits(format), dtype=np.int64)
    if format.component_class == "unorm":
        top = 2**bits - 1
        return np.array((top[0] // 4, 1, top[2 % count], top[-1] // 2)[:count]) / top
    if format.component_class == "snorm":
        top = 2 ** (bits - 1) - 1
        return np.array((-top[0], 1, top[2 % count], -(top[-1] // 2))[:count]) / top
    if format.component_class == "uint":
        top = 2**bits - 1
        return np.array((top[0], 1, top[2 % count] // 2, min(7, top[-1]))[:count])
    low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return np.array((low[0], high[1 % count], -1, min(5, high[-1]))[:count])


def _small_float(value: float, mantissa_bits: int) -> int:
    exponent = math.floor(math.log2(value))
    mantissa = round((value / 2.0**exponent - 1.0) * 2**mantissa_bits)
    return ((exponent + 15) << mantissa_bits) | mantissa


def _texel(format: vd.TextureFormat, values: np.ndarray) -> np.ndarray:
    """The host bytes of one texel holding `values`."""
    name = format.name
    if name in ("rgb10a2_unorm", "rgb10a2_uint"):
        codes = np.round(values * (1023, 1023, 1023, 3)) if name == "rgb10a2_unorm" else values
        red, green, blue, alpha = (int(code) for code in codes)
        return np.array(red | green << 10 | blue << 20 | alpha << 30, dtype=np.uint32)
    if name == "r11g11b10_float":
        red, green, blue = values
        word = _small_float(red, 6) | _small_float(green, 6) << 11 | _small_float(blue, 5) << 22
        return np.array(word, dtype=np.uint32)
    if name == "rgb9e5_ufloat":
        exponent = math.floor(math.log2(max(values))) + 1
        red, green, blue = (round(value / 2.0 ** (exponent - 9)) for value in values)
        return np.array(red | green << 9 | blue << 18 | (exponent + 15) << 27, dtype=np.uint32)
    if format.component_class in ("unorm", "snorm"):
        bits = np.array(_component_bits(format))
        top = 2**bits - 1 if format.component_class == "unorm" else 2 ** (bits - 1) - 1
        values = np.round(values * top)
    texel = values.astype(format.dtype)
    return texel[[2, 1, 0, 3]] if name.startswith("bgra") else texel


def _texels(format: vd.TextureFormat, values: np.ndarray) -> np.ndarray:
    texel = _texel(format, values).reshape(() if format.channels == 1 else (format.channels,))
    return np.ascontiguousarray(np.broadcast_to(texel, (SIZE, SIZE, *texel.shape)))


def _decoded(format: vd.TextureFormat, values: np.ndarray) -> np.ndarray:
    """What a shader reads from a texel holding `values`: missing components read as 0 and alpha as 1."""
    if format.name.endswith("srgb"):
        color = values[:3]
        values = np.concatenate(
            (np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4), values[3:])
        )
    result = np.array((0.0, 0.0, 0.0, 1.0))
    result[: format.components] = values
    return result


def _fetch(texture: vd.Texture, format: vd.TextureFormat) -> np.ndarray:
    target_format, fragment, _ = _FETCH.get(format.component_class, _FLOAT_FETCH)
    target = vd.Texture.zeros(shape=(SIZE, SIZE), format=target_format)
    vd.pipeline(triangle_vertex, fragment)(
        position=_covering(),
        source=texture,
        render_pass=vd.render_pass(vd.RenderTarget.from_attachments(colors={0: target}), color=vd.clear((0, 0, 0, 0))),
    )
    return target.to_numpy().reshape(SIZE * SIZE, 4)


def _assert_fetched(
    test: unittest.TestCase, format: vd.TextureFormat, fetched: np.ndarray, expected: np.ndarray
) -> None:
    if format.integer:
        np.testing.assert_array_equal(fetched, np.broadcast_to(expected.astype(fetched.dtype), fetched.shape))
        return
    tolerance = 2e-3 if format.name.endswith("srgb") else 1e-6
    np.testing.assert_allclose(fetched, np.broadcast_to(expected, fetched.shape), rtol=0, atol=tolerance)


def _literal(format: vd.TextureFormat, value: float) -> str:
    if not format.integer:
        return repr(float(value))
    # -2**31 has no i32 literal: its magnitude is not an i32.
    if int(value) == -(2**31):
        return "(vd.i32(-2147483647) - vd.i32(1))"
    return f"vd.{format.shader_scalar}({int(value)})"


_KERNEL_DIRECTORY = tempfile.TemporaryDirectory(prefix="vernon_storage_formats_")


@functools.cache
def _storage_kernels() -> types.ModuleType:
    """One kernel per storage format storing `_values` to every texel; a storage format is named in source."""
    sources = ["from typing import Annotated", "", "import vernon_dsl as vd", ""]
    for format in STORAGE_FORMATS:
        values = list(_values(format)) + [0, 0, 1][format.components - 1 :]
        texel = ", ".join(_literal(format, value) for value in values[:4])
        sources += [
            "",
            "@vd.kernel(workgroup_size=(4, 4, 1))",
            f"def store_{format.name}(",
            f'    image: vd.Texture["2d", vd.{format.name}, vd.write],  # noqa: F722',
            '    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],',
            ") -> None:",
            f"    vd.texture_store(image, vd.Vector([vd.i32(gid[0]), vd.i32(gid[1])]), vd.Vector([{texel}]))",
            "",
        ]
    path = Path(_KERNEL_DIRECTORY.name) / "storage_format_kernels.py"
    path.write_text("\n".join(sources), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _supported(format: vd.TextureFormat, *needed: str) -> bool:
    return set(needed) <= vd.current_session().format_capabilities(format)


def _require_fetch_targets(test: unittest.TestCase) -> None:
    for target_format, _, _ in (*_FETCH.values(), _FLOAT_FETCH):
        if not _supported(target_format, "renderable", "copyable"):
            test.skipTest(f"the device does not render {target_format.name}")


@expand_backend_matrix_tests
class TextureFormatRuntimeTests(unittest.TestCase):
    @backend_matrix_test(GRAPHICS)
    def test_every_copyable_color_format_is_read_as_its_components(self, row: BackendRow) -> None:
        _require_fetch_targets(self)
        for format in COLOR_FORMATS:
            if not _supported(format, "copyable"):
                continue
            with self.subTest(format=format.name):
                values = _values(format)
                texture = vd.Texture.from_numpy(_texels(format, values), format=format)
                _assert_fetched(self, format, _fetch(texture, format), _decoded(format, values))

    @backend_matrix_test(GRAPHICS)
    def test_every_renderable_color_format_holds_what_a_fragment_writes(self, row: BackendRow) -> None:
        _require_fetch_targets(self)
        for format in COLOR_FORMATS:
            if not _supported(format, "renderable", "copyable"):
                continue
            with self.subTest(format=format.name):
                values = _values(format)
                result_class = format.component_class if format.integer else "float"
                written = values.astype(_WRITE_DTYPE.get(result_class, np.float32))
                target = vd.Texture.zeros(shape=(SIZE, SIZE), format=format)
                vd.pipeline(triangle_vertex, _WRITE[result_class, format.components])(
                    position=_covering(),
                    color=vd.storage.from_numpy(written.reshape(()) if format.components == 1 else written),
                    render_pass=vd.render_pass(
                        vd.RenderTarget.from_attachments(colors={0: target}), color=vd.clear((0, 0, 0, 0))
                    ),
                )
                expected = np.array((0.0, 0.0, 0.0, 1.0))
                expected[: format.components] = values
                # sRGB targets encode what the fragment writes and fetches decode it, each within a code.
                tolerance = 1.5 / 255 if format.name.endswith("srgb") else None
                fetched = _fetch(target, format)
                if tolerance is None:
                    _assert_fetched(self, format, fetched, expected)
                else:
                    np.testing.assert_allclose(
                        fetched, np.broadcast_to(expected, fetched.shape), rtol=0, atol=tolerance
                    )

    @backend_matrix_test(GRAPHICS)
    def test_every_renderable_integer_format_clears_to_its_range(self, row: BackendRow) -> None:
        # Every vertex at one point draws nothing, so the target holds only its clear.
        nothing = vd.storage.from_numpy(np.zeros((3, 2), dtype=np.float32))
        for format in COLOR_FORMATS:
            if not format.integer or not _supported(format, "renderable", "copyable"):
                continue
            with self.subTest(format=format.name):
                values = _values(format)
                target = vd.Texture.zeros(shape=(SIZE, SIZE), format=format)
                written = values.astype(_WRITE_DTYPE[format.component_class])
                clear = tuple(int(value) for value in (*values, 0, 0, 1)[:4])
                vd.pipeline(triangle_vertex, _WRITE[format.component_class, format.components])(
                    position=nothing,
                    color=vd.storage.from_numpy(written.reshape(()) if format.components == 1 else written),
                    render_pass=vd.render_pass(
                        vd.RenderTarget.from_attachments(colors={0: target}), color=vd.clear(clear)
                    ),
                )
                np.testing.assert_array_equal(target.download(), _texels(format, values))

    @backend_matrix_test(GRAPHICS)
    def test_every_depth_format_keeps_the_nearest_surface(self, row: BackendRow) -> None:
        triangle = ((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75))
        # The nearer, red triangle draws first, so only the depth test keeps the farther, green one out.
        positions = np.array([(*corner, depth, 1.0) for depth in (0.25, 0.8) for corner in triangle], np.float32)
        colors = np.array(((1.0, 0.0, 0.0, 1.0),) * 3 + ((0.0, 1.0, 0.0, 1.0),) * 3, np.float32)
        render = vd.pipeline(
            depth_vertex,
            depth_fragment,
            state=vd.graphics_state(depth_stencil=vd.DepthStencilState(depth_test=True, depth_write=True)),
        )
        for format in DEPTH_FORMATS:
            if not _supported(format, "renderable", "copyable"):
                continue
            with self.subTest(format=format.name):
                color = vd.Texture.zeros(shape=(32, 32))
                depth = vd.Texture.zeros(
                    shape=(32, 32),
                    format=format,
                    usage=("sampled", "transfer_source", "transfer_destination", "depth_stencil_attachment"),
                )
                render(
                    position=vd.storage.from_numpy(positions),
                    color=vd.storage.from_numpy(colors),
                    render_pass=vd.render_pass(
                        vd.RenderTarget.from_attachments(colors={0: color}, depth=depth),
                        color=vd.clear((0.0, 0.0, 0.0, 0.0)),
                        depth=vd.clear_depth(1.0),
                    ),
                )
                np.testing.assert_array_equal(color.to_numpy()[16, 16], (255, 0, 0, 255))
                depths = depth.to_numpy()
                depths = depths["depth"] if depths.dtype.names else depths
                if depths.dtype == np.uint16:
                    depths = depths / 65535.0
                np.testing.assert_allclose(depths[(16, 0), (16, 0)], (0.25, 1.0), rtol=0, atol=1e-4)

    @backend_matrix_test(BackendRequirements(gpu=True, compute=True, storage_texture=True))
    def test_every_storage_format_holds_what_a_kernel_stores(self, row: BackendRow) -> None:
        kernels = _storage_kernels()
        for format in STORAGE_FORMATS:
            if not _supported(format, "storage", "copyable"):
                continue
            with self.subTest(format=format.name):
                values = _values(format)
                image = vd.Texture.zeros(
                    shape=(SIZE, SIZE), format=format, usage=("storage", "transfer_source", "transfer_destination")
                )
                getattr(kernels, f"store_{format.name}")(image, grid=(1, 1, 1))
                np.testing.assert_array_equal(image.download(), _texels(format, values))

    @backend_matrix_test(GRAPHICS)
    def test_device_capabilities_narrow_the_table_baseline(self, row: BackendRow) -> None:
        session = vd.current_session()
        for format in TEXTURE_FORMATS:
            with self.subTest(format=format.name):
                self.assertLessEqual(session.format_capabilities(format), format.capabilities)


if __name__ == "__main__":
    unittest.main()
