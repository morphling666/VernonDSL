from __future__ import annotations

import json
import types
import unittest
from unittest import mock

import numpy as np
import vernon_dsl as vd
import vernon_dsl._runtime.pipeline as pipeline_module
import vernon_dsl._runtime.session as runtime_module
from advanced_pipeline_shader import (
    advanced_fragment,
    advanced_vertex,
)
from aggregate_vertex_shader import (
    AggregateVertexPayload,
    ComplexAggregateVertex,
    SmallAggregateVertex,
    aggregate_triangle_vertex,
    oversized_aggregate_attribute_vertex,
    small_multidimensional_aggregate_attribute_vertex,
)
from pipeline_shader import (
    colored_fragment,
    copy_static_tensor_value,
    cube_direction_fragment,
    depth_fragment,
    depth_vertex,
    divisor_two_attribute_vertex,
    f16_attribute_vertex,
    f32_attribute_vertex,
    f64_attribute_vertex,
    i32_attribute_vertex,
    instanced_tensor_transform_vertex,
    mat2_vertex,
    matrix_elementwise_fragment,
    matrix_vertex,
    mixed_uniform_fragment,
    non_square_attribute_vertex,
    numpy_tensor_fragment,
    numpy_tensor_vertex,
    optional_texture_fragment,
    oversized_tensor_attribute_vertex,
    rank_three_tensor_attribute_vertex,
    solid_fragment,
    static_tensor_fragment,
    translate_vertices,
    translated_vertex,
    triangle_vertex,
    u32_attribute_vertex,
    volume_coordinate_fragment,
)


def render_target(texture: vd.Texture) -> vd.RenderTarget:
    return vd.RenderTarget(shape=texture.shape).attach_color(0, texture)


def mrt_target(color: vd.Texture, object_id: vd.Texture) -> vd.RenderTarget:
    return vd.RenderTarget(shape=color.shape).attach_color(0, color).attach_color(1, object_id)


class RenderTargetTests(unittest.TestCase):
    def test_attachment_validation(self) -> None:
        color = vd.Texture.zeros(shape=(16, 16))
        with self.assertRaisesRegex(ValueError, "two positive dimensions"):
            vd.RenderTarget(shape=(16, 0))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            vd.RenderTarget(shape=(16, 16)).attach_color(0, vd.Texture.zeros(shape=(8, 16)))

        target = vd.RenderTarget(shape=(16, 16)).attach_color(0, color)
        with self.assertRaisesRegex(ValueError, "already occupied"):
            target.attach_color(0, color)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            target.attach_color(-1, color)

        target.attach_depth()
        with self.assertRaisesRegex(ValueError, "already has"):
            target.attach_depth()

    def test_render_target_owned_depth_and_cube_texture_validation(self) -> None:
        target = vd.RenderTarget(shape=(16, 16)).attach_depth()
        depth = target.depth_texture
        self.assertEqual(target.shape, (16, 16))
        self.assertEqual(depth.shape, (16, 16))
        with self.assertRaisesRegex(RuntimeError, "cannot be uploaded"):
            depth.copy_from_numpy(np.zeros((16, 16), dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "readback is not exposed"):
            depth.to_numpy()

        faces = np.zeros((6, 8, 8, 4), dtype=np.uint8)
        cube = vd.Texture.cube(faces)
        self.assertEqual(cube.shape, (8, 8))
        np.testing.assert_array_equal(cube.to_numpy(), faces)

        with self.assertRaisesRegex(RuntimeError, "no depth attachment"):
            _ = vd.RenderTarget(shape=(16, 16)).depth_texture
        with self.assertRaisesRegex(ValueError, "6"):
            vd.Texture.cube(np.zeros((5, 8, 8, 4), dtype=np.uint8))

    def test_three_dimensional_texture_storage(self) -> None:
        source = np.arange(3 * 4 * 5 * 4, dtype=np.uint8).reshape(3, 4, 5, 4)
        texture = vd.Texture.from_numpy(source, dimension="3d")
        self.assertEqual(texture.shape, (3, 4, 5))
        np.testing.assert_array_equal(texture.to_numpy(), source)

        replacement = np.full_like(source, 23)
        texture.copy_from_numpy(replacement)
        np.testing.assert_array_equal(texture.to_numpy(), replacement)
        self.assertEqual(vd.Texture.zeros(shape=(2, 3, 4), dimension="3d").shape, (2, 3, 4))

        with self.assertRaisesRegex(ValueError, "3 positive dimensions"):
            vd.Texture.zeros(shape=(3, 4), dimension="3d")
        with self.assertRaisesRegex(ValueError, "two positive dimensions"):
            vd.RenderTarget(shape=texture.shape)

    def test_texture_formats_mips_and_subregions(self) -> None:
        texture = vd.Texture.zeros(
            shape=(4, 6, 8),
            dimension="3d",
            format=vd.rgba32_float,
            mip_levels=4,
            usage=("sampled", "storage", "transfer_source", "transfer_destination"),
        )
        self.assertEqual(texture.format, vd.rgba32_float)
        self.assertEqual(texture.dimension, "3d")
        self.assertEqual(texture.mip_levels, 4)
        self.assertIn("storage", texture.usage)

        region = np.full((1, 2, 3, 4), 7.0, dtype=np.float32)
        texture.upload(region, mip_level=0, origin=(2, 1, 4))
        expected = np.zeros((4, 6, 8, 4), dtype=np.float32)
        expected[2:3, 1:3, 4:7] = 7.0
        np.testing.assert_array_equal(texture.download(), expected)

        mip = np.full((2, 3, 4, 4), 2.0, dtype=np.float32)
        texture.upload(mip, mip_level=1)
        np.testing.assert_array_equal(texture.download(mip_level=1), mip)

        cube = vd.Texture.cube(np.zeros((6, 4, 4, 4), dtype=np.uint8), mip_levels=2)
        cube.upload(np.full((6, 2, 2, 4), 9, dtype=np.uint8), mip_level=1)
        np.testing.assert_array_equal(cube.download(mip_level=1), np.full((6, 2, 2, 4), 9, dtype=np.uint8))

        storage_only = vd.Texture.zeros(shape=(2, 2), usage=("storage",))
        with self.assertRaisesRegex(RuntimeError, "transfer_source"):
            storage_only.download()


def assert_depth_attachment_selects_nearest(test: unittest.TestCase) -> None:
    triangle = ((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75))
    positions = vd.storage.from_numpy(
        np.array(
            [(*position, depth, 1.0) for depth in (0.2, 0.8) for position in triangle],
            dtype=np.float32,
        )
    )
    colors = vd.storage.from_numpy(
        np.array(((1.0, 0.0, 0.0, 1.0),) * 3 + ((0.0, 1.0, 0.0, 1.0),) * 3, dtype=np.float32)
    )
    target = vd.Texture.zeros(shape=(32, 32))
    attachments = render_target(target).attach_depth()

    vd.pipeline(depth_vertex, depth_fragment)(position=positions, color=colors, target=attachments)

    pixel = target.to_numpy()[16, 16]
    test.assertGreater(int(pixel[0]), 240)
    test.assertLess(int(pixel[1]), 10)


def assert_rank_three_tensor_attribute_renders(test: unittest.TestCase) -> None:
    values = np.zeros((3, 2, 2, 3), dtype=np.float32)
    values[:, 0, 0, :2] = np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32)
    values[:, 1, 1, :] = np.array((17.0, 23.0, 29.0), dtype=np.float32)
    interleaved = np.zeros((3, 14), dtype=np.float32)
    interleaved[:, :12] = values.reshape(3, 12)
    storage = vd.storage.from_numpy(interleaved)
    view = storage.view(shape=(3, 2, 2, 3), strides=(14, 6, 3, 1), access="read")
    projection = np.zeros((3, 1, 3, 2), dtype=np.float32)
    projection[:, 0, 0, 0] = 1.0
    projection[:, 0, 1, 1] = 1.0
    target = vd.Texture.zeros(shape=(32, 32))
    vd.pipeline(rank_three_tensor_attribute_vertex, solid_fragment)(
        value=view,
        projection=vd.storage.from_numpy(projection),
        target=render_target(target),
    )
    test.assertGreater(int(target.to_numpy()[16, 16, 0]), 240)


def assert_aggregate_attribute_renders(test: unittest.TestCase) -> None:
    def vertex(position: tuple[float, float], object_id: int) -> ComplexAggregateVertex:
        return ComplexAggregateVertex(
            np.array(position, dtype=np.float32),
            AggregateVertexPayload(
                vd.i32(object_id),
                np.array((0.25, 0.5), dtype=np.float32),
                np.array((0.75, 1.0), dtype=np.float32),
            ),
            np.eye(2, dtype=np.float32),
        )

    vertices = vd.storage.from_values(
        (
            vertex((-0.75, -0.75), 11),
            vertex((0.75, -0.75), 12),
            vertex((0.0, 0.75), 13),
        ),
        dtype=ComplexAggregateVertex,
    )
    aggregate_values = tuple(
        ComplexAggregateVertex(
            np.array((float(index), float(index) + 0.5), dtype=np.float32),
            AggregateVertexPayload(
                vd.i32(index * 3),
                np.array((index + 0.25, index + 0.75), dtype=np.float32),
                np.array((index + 1.0, index + 2.0), dtype=np.float32),
            ),
            np.array(((index + 3.0, index + 4.0), (index + 5.0, index + 6.0)), dtype=np.float32),
        )
        for index in range(2 * 3 * 4)
    )
    aggregate = vd.storage.from_values(
        tuple(
            tuple(tuple(aggregate_values[plane * 12 + row * 4 + column] for column in range(4)) for row in range(3))
            for plane in range(2)
        ),
        dtype=ComplexAggregateVertex,
    )
    target = vd.Texture.zeros(shape=(32, 32))
    vd.pipeline(aggregate_triangle_vertex, solid_fragment)(
        vertex=vertices, aggregate=aggregate, target=render_target(target)
    )
    test.assertGreater(int(target.to_numpy()[16, 16, 0]), 240)


def assert_small_multidimensional_aggregate_attribute_renders(test: unittest.TestCase) -> None:
    def tensor_value(position: tuple[float, float]) -> tuple[tuple[tuple[SmallAggregateVertex, ...], ...], ...]:
        filler = SmallAggregateVertex(np.zeros(2, dtype=np.float32), vd.f32(0.0))
        selected = SmallAggregateVertex(np.asarray(position, dtype=np.float32), vd.f32(2.0))
        values = (filler,) * 7 + (selected,)
        return tuple(
            tuple(tuple(values[plane * 4 + row * 2 + column] for column in range(2)) for row in range(2))
            for plane in range(2)
        )

    values = vd.storage.from_values(
        (
            tensor_value((-0.75, -0.75)),
            tensor_value((0.75, -0.75)),
            tensor_value((0.0, 0.75)),
        ),
        dtype=SmallAggregateVertex,
    )
    target = vd.Texture.zeros(shape=(32, 32))
    vd.pipeline(small_multidimensional_aggregate_attribute_vertex, solid_fragment)(
        values=values, target=render_target(target)
    )
    test.assertGreater(int(target.to_numpy()[16, 16, 0]), 240)


def assert_oversized_aggregate_attribute_reports_location_limit(test: unittest.TestCase) -> None:
    value = ComplexAggregateVertex(
        np.zeros(2, dtype=np.float32),
        AggregateVertexPayload(vd.i32(0), np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)),
        np.zeros((2, 2), dtype=np.float32),
    )
    values = vd.storage.from_values(
        tuple(tuple(tuple(tuple(value for _ in range(2)) for _ in range(2)) for _ in range(2)) for _ in range(3)),
        dtype=ComplexAggregateVertex,
    )
    target = vd.Texture.zeros(shape=(8, 8))
    with test.assertRaisesRegex(RuntimeError, "location or format capabilities"):
        vd.pipeline(oversized_aggregate_attribute_vertex, solid_fragment)(values=values, target=render_target(target))


def assert_instanced_mat4_tensor_attribute_renders(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
    transforms = np.eye(4, dtype=np.float32)[None, ...]
    transforms[0, 0, 3] = 0.6
    target = vd.Texture.zeros(shape=(32, 32))
    vd.pipeline(instanced_tensor_transform_vertex, solid_fragment)(
        position=positions,
        transform=vd.storage.from_numpy(transforms),
        target=render_target(target),
    )
    pixels = target.to_numpy()
    test.assertGreater(int(pixels[16, 25, 0]), 240)
    test.assertEqual(tuple(pixels[16, 16]), (0, 0, 0, 0))


def assert_formal_attribute_formats_render(
    test: unittest.TestCase,
    *,
    unsupported: frozenset[str] = frozenset(),
) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
    cases = (
        ("i32", i32_attribute_vertex, np.int32),
        ("u32", u32_attribute_vertex, np.uint32),
        ("f16", f16_attribute_vertex, np.float16),
        ("f32", f32_attribute_vertex, np.float32),
        ("f64", f64_attribute_vertex, np.float64),
    )
    for name, vertex, dtype in cases:
        with test.subTest(dtype=name):
            values = vd.storage.from_numpy(np.zeros((3, 2), dtype=dtype))
            target = vd.Texture.zeros(shape=(32, 32))
            render = vd.pipeline(vertex, solid_fragment)
            if name in unsupported:
                with test.assertRaisesRegex((vd.CompileError, RuntimeError), "support|capabilit|format"):
                    render(position=positions, value=values, target=render_target(target))
                continue
            render(position=positions, value=values, target=render_target(target))
            test.assertGreater(int(target.to_numpy()[16, 16, 0]), 240)


def assert_non_square_and_divisor_two_attributes_render(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.25, -0.25), (0.25, -0.25), (0.0, 0.25)), dtype=np.float32))
    non_square = vd.storage.from_numpy(np.zeros((3, 2, 3), dtype=np.float32))
    target = vd.Texture.zeros(shape=(32, 32))
    vd.pipeline(non_square_attribute_vertex, solid_fragment)(
        position=positions,
        value=non_square,
        target=render_target(target),
    )
    test.assertGreater(int(target.to_numpy()[16, 16, 0]), 240)

    offsets = vd.storage.from_numpy(np.array(((-0.5, 0.0), (0.5, 0.0)), dtype=np.float32))
    target = vd.Texture.zeros(shape=(32, 32))
    vd.pipeline(divisor_two_attribute_vertex, solid_fragment)(
        position=positions,
        offset=offsets,
        target=render_target(target),
    )
    pixels = target.to_numpy()
    test.assertGreater(int(pixels[16, 8, 0]), 240)
    test.assertGreater(int(pixels[16, 24, 0]), 240)


def assert_matrix_uniform_transforms_vertices(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = 0.65
    target = vd.Texture.zeros(shape=(64, 64))

    vd.pipeline(matrix_vertex, solid_fragment)(
        position=positions,
        transform=transform,
        target=render_target(target),
    )

    pixels = target.to_numpy()
    test.assertGreater(int(pixels[32, 48, 0]), 240)
    test.assertEqual(tuple(pixels[32, 16]), (0, 0, 0, 0))


def assert_mixed_uniform_layout_renders(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
    target = vd.Texture.zeros(shape=(32, 32))
    render = vd.pipeline(triangle_vertex, mixed_uniform_fragment)

    render(
        position=positions,
        first=np.array((0.05, 0.1, 0.15), dtype=np.float32),
        second=np.array((0.2, 0.25, 0.3), dtype=np.float32),
        scale=np.float32(0.35),
        bias=np.float32(0.4),
        uv=np.array((0.45, 0.5), dtype=np.float32),
        texel=np.array((0.55, 0.6), dtype=np.float32),
        target=render_target(target),
    )

    np.testing.assert_allclose(
        target.to_numpy()[16, 16],
        np.array((64, 115, 140, 255), dtype=np.uint8),
        atol=1,
    )

    render(
        position=positions,
        first=np.array((0.1, 0.2, 0.3), dtype=np.float32),
        second=np.array((0.4, 0.5, 0.6), dtype=np.float32),
        scale=np.float32(0.1),
        bias=np.float32(0.2),
        uv=np.array((0.2, 0.3), dtype=np.float32),
        texel=np.array((0.4, 0.5), dtype=np.float32),
        target=render_target(target),
    )
    np.testing.assert_allclose(
        target.to_numpy()[16, 16],
        np.array((128, 77, 128, 153), dtype=np.uint8),
        atol=1,
    )


def assert_cube_faces_remain_distinct(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
    colors = np.array(
        (
            (255, 0, 0, 255),
            (0, 255, 0, 255),
            (0, 0, 255, 255),
            (255, 255, 0, 255),
            (255, 0, 255, 255),
            (0, 255, 255, 255),
        ),
        dtype=np.uint8,
    )
    faces = np.broadcast_to(colors[:, None, None, :], (6, 2, 2, 4)).copy()
    image = vd.Texture.cube(faces)
    sampler = vd.sampler(address="clamp_to_edge")
    render = vd.pipeline(triangle_vertex, cube_direction_fragment)
    directions = np.concatenate((np.eye(3, dtype=np.float32), -np.eye(3, dtype=np.float32)))
    face_order = (0, 2, 4, 1, 3, 5)

    for direction, face in zip(directions, face_order, strict=True):
        target = vd.Texture.zeros(shape=(8, 8))
        render(
            position=positions,
            direction=np.ascontiguousarray(direction),
            image=image,
            sampler=sampler,
            target=render_target(target),
        )
        np.testing.assert_array_equal(target.to_numpy()[4, 4], colors[face])

    target = vd.Texture.zeros(shape=(8, 8))
    render(
        position=positions,
        direction=np.array((1.0, 0.0, 0.99), dtype=np.float32),
        image=image,
        sampler=sampler,
        target=render_target(target),
    )
    seam_pixel = target.to_numpy()[4, 4]
    test.assertGreater(int(seam_pixel[0]), 240)
    test.assertLess(int(seam_pixel[1]), 16)
    test.assertGreater(int(seam_pixel[2]), 80)
    test.assertLess(int(seam_pixel[2]), 180)


def assert_three_dimensional_texture_samples(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
    colors = np.array(((255, 32, 16, 255), (24, 64, 255, 255)), dtype=np.uint8)
    volume = np.broadcast_to(colors[:, None, None, :], (2, 2, 2, 4)).copy()
    image = vd.Texture.from_numpy(volume, dimension="3d")
    sampler = vd.sampler(address="clamp_to_edge")
    render = vd.pipeline(triangle_vertex, volume_coordinate_fragment)

    for layer, color in enumerate(colors):
        target = vd.Texture.zeros(shape=(8, 8))
        render(
            position=positions,
            coordinate=np.array((0.5, 0.5, (layer + 0.5) / len(colors)), dtype=np.float32),
            image=image,
            sampler=sampler,
            target=render_target(target),
        )
        np.testing.assert_allclose(target.to_numpy()[4, 4], color, atol=1)


def assert_inactive_texture_binding_renders(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
    base_image = vd.Texture.from_numpy(np.full((2, 2, 4), (0, 255, 0, 255), dtype=np.uint8))
    optional_image = vd.Texture.from_numpy(np.full((2, 2, 4), (255, 0, 0, 255), dtype=np.uint8))
    target = vd.Texture.zeros(shape=(8, 8))

    vd.pipeline(triangle_vertex, optional_texture_fragment)(
        position=positions,
        base_image=base_image,
        base_sampler=vd.sampler(),
        optional_image=optional_image,
        optional_sampler=vd.sampler(),
        target=render_target(target),
    )

    np.testing.assert_array_equal(target.to_numpy()[4, 4], np.array((0, 255, 0, 255), dtype=np.uint8))

    enabled_target = vd.Texture.zeros(shape=(8, 8))
    vd.pipeline(triangle_vertex, optional_texture_fragment, features={"OPTIONAL_IMAGE"})(
        position=positions,
        base_image=base_image,
        base_sampler=vd.sampler(),
        optional_image=optional_image,
        optional_sampler=vd.sampler(),
        target=render_target(enabled_target),
    )
    np.testing.assert_array_equal(enabled_target.to_numpy()[4, 4], np.array((255, 0, 0, 255), dtype=np.uint8))


def assert_mat2_uniform_transforms_vertices(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
    transform = np.array(((0.5, 0.0), (0.0, 1.0)), dtype=np.float32)
    target = vd.Texture.zeros(shape=(64, 64))

    vd.pipeline(mat2_vertex, solid_fragment)(position=positions, transform=transform, target=render_target(target))

    pixels = target.to_numpy()
    test.assertGreater(int(pixels[32, 32, 0]), 240)
    test.assertEqual(tuple(pixels[32, 42]), (0, 0, 0, 0))


def assert_rank_three_uniform_renders(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
    weights = np.zeros((2, 2, 2), dtype=np.float32)
    weights[1, 0, 1] = 0.75
    target = vd.Texture.zeros(shape=(32, 32))

    vd.pipeline(triangle_vertex, static_tensor_fragment)(
        position=positions, weights=weights, target=render_target(target)
    )

    pixel = target.to_numpy()[16, 16]
    test.assertAlmostEqual(int(pixel[0]), 191, delta=2)
    test.assertEqual(int(pixel[1]), 0)
    test.assertEqual(int(pixel[2]), 0)
    test.assertGreater(int(pixel[3]), 250)


def assert_matrix_elementwise_multiply_renders(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
    left = np.array(((0.2, 0.4), (0.6, 0.8)), dtype=np.float32)
    right = np.array(((0.5, 0.5), (0.5, 1.0)), dtype=np.float32)
    target = vd.Texture.zeros(shape=(32, 32))

    vd.pipeline(triangle_vertex, matrix_elementwise_fragment)(
        position=positions,
        left=left,
        right=right,
        target=render_target(target),
    )

    np.testing.assert_allclose(
        target.to_numpy()[16, 16],
        np.rint((left * right).reshape(4) * 255.0).astype(np.uint8),
        rtol=0.0,
        atol=1,
    )


def assert_numpy_tensor_vertex_renders(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.25, -0.25), (0.25, -0.25), (0.0, 0.25)), dtype=np.float32))
    offset_left = np.zeros((8, 8, 4), dtype=np.float32)
    offset_left[7, 7, 0] = 0.25
    offset_right = np.zeros((1, 8, 1), dtype=np.float32)
    offset_right[0, 7, 0] = 0.25
    transform = np.zeros((8, 8, 4), dtype=np.float32)
    transform[7, 0, 0] = 1.0
    transform[7, 1, 1] = 1.0
    target = vd.Texture.zeros(shape=(64, 64))

    vd.pipeline(numpy_tensor_vertex, solid_fragment)(
        position=positions,
        offset_left=offset_left,
        offset_right=offset_right,
        transform=transform,
        target=render_target(target),
    )

    pixels = target.to_numpy()
    test.assertGreater(int(pixels[32, 48, 0]), 240)
    test.assertEqual(tuple(pixels[32, 24]), (0, 0, 0, 0))


def assert_numpy_tensor_fragment_renders(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
    broadcast_left = np.linspace(0.05, 0.3, 8 * 8 * 4, dtype=np.float32).reshape(8, 8, 4)
    broadcast_right = np.linspace(0.01, 0.08, 8, dtype=np.float32).reshape(1, 8, 1)
    matmul_left = np.linspace(0.001, 0.03, 8 * 8 * 4, dtype=np.float32).reshape(8, 8, 4)
    matmul_right = np.linspace(0.1, 0.4, 8, dtype=np.float32).reshape(1, 4, 2)
    target = vd.Texture.zeros(shape=(32, 32))

    vd.pipeline(triangle_vertex, numpy_tensor_fragment)(
        position=positions,
        broadcast_left=broadcast_left,
        broadcast_right=broadcast_right,
        matmul_left=matmul_left,
        matmul_right=matmul_right,
        target=render_target(target),
    )

    broadcasted = broadcast_left + broadcast_right
    product = np.matmul(matmul_left, matmul_right)
    expected = np.array(
        (broadcasted[0, 0, 0], broadcasted[7, 7, 3], product[0, 0, 1], product[7, 7, 0]),
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        target.to_numpy()[16, 16],
        np.rint(expected * 255.0).astype(np.uint8),
        rtol=0.0,
        atol=1,
    )


def assert_static_tensor_compute_argument(test: unittest.TestCase) -> None:
    output = vd.storage.from_numpy(np.full(12, -99.0, dtype=np.float32))
    singleton = np.array((0.125,), dtype=np.float32)
    quad = np.array((1.5, -2.0, 3.25, 4.75), dtype=np.float32)
    weights = np.arange(8, dtype=np.float32).reshape((2, 2, 2)) * 0.25
    large = np.arange(30, dtype=np.float32).reshape((2, 3, 5)) * -0.5
    matrix_left = np.array(((1.5, -2.0), (3.25, 4.75)), dtype=np.float32)
    matrix_right = np.array(((2.0, 0.5), (-1.0, 3.0)), dtype=np.float32)

    copy_static_tensor_value(
        output,
        singleton,
        quad,
        weights,
        large,
        matrix_left,
        matrix_right,
        grid=(3, 1, 1),
    )

    np.testing.assert_allclose(
        output.to_numpy(),
        np.array(
            (
                singleton[0],
                weights[0, 0, 0],
                weights[1, 0, 1],
                weights[1, 1, 1],
                weights[1, 1, 0] * 2.0,
                large[1, 2, 4],
                quad[3],
                *(matrix_left * matrix_right).reshape(4),
                -99.0,
            ),
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-6,
    )


@vd.func
def shader_helper(value: vd.f32) -> vd.f32:
    return value


@vd.struct
class InterleavedVertex:
    ignored: vd.f32
    position: vd.Vector[vd.f32, 2]


class PipelineContractTests(unittest.TestCase):
    def test_declarative_functions_reject_host_calls(self) -> None:
        with self.assertRaisesRegex(TypeError, "shader-only"):
            triangle_vertex(None)
        with self.assertRaisesRegex(TypeError, "shader-only"):
            solid_fragment()
        with self.assertRaisesRegex(TypeError, "shader-only"):
            shader_helper(1.0)

    def test_stage_order_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "pipeline stages"):
            vd.pipeline(triangle_vertex)
        with self.assertRaisesRegex(ValueError, "pipeline stages"):
            vd.pipeline(solid_fragment, triangle_vertex)

    def test_immediate_binding_failure_releases_dispatch_lease(self) -> None:
        pipeline = object.__new__(pipeline_module.Pipeline)
        lease = mock.Mock()
        plan = types.SimpleNamespace(compiled=types.SimpleNamespace(native=object()), lease=lease)
        binding_cache = mock.MagicMock()
        binding_cache.invocation.return_value.__enter__.return_value = object()
        with (
            mock.patch.object(pipeline, "_prepare_graphics_invocation", return_value=plan),
            mock.patch.object(pipeline, "_bind_graphics_arguments", side_effect=RuntimeError("binding failed")),
            self.assertRaisesRegex(RuntimeError, "binding failed"),
        ):
            pipeline._invoke_direct({}, None, binding_cache)
        lease.release.assert_called_once_with()

    def test_tensor_layout_and_contiguous_swizzle(self) -> None:
        tensor = vd.storage.from_numpy(np.zeros((3, 4), dtype=np.float32))
        self.assertEqual(tensor.layout.shape, (3, 4))
        self.assertEqual(tensor.layout.byte_strides, (16, 4))
        view = tensor.swizzle("yz")
        self.assertEqual(view.shape, (3, 2))
        self.assertEqual(view.layout.byte_offset, 4)
        self.assertEqual(view.layout.components, (1, 2))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            tensor.swizzle("zx")

    def test_owned_gl_context_is_selected_and_retained(self) -> None:
        created: list[object] = []
        runtime_calls: list[tuple[object, ...]] = []

        class FakeContext:
            def __init__(self, backend: str, major: int, minor: int):
                self.request = (backend, major, minor)
                self.user_data = 11
                self.make_current = 12
                self.get_proc_address = 13
                created.append(self)

        class FakeRhiHost:
            def create_runtime(self) -> object:
                return object()

            @staticmethod
            def create_external_opengl(*arguments: object) -> object:
                runtime_calls.append(arguments)
                return FakeRhiHost()

        fake_native = types.SimpleNamespace(
            RhiHost=FakeRhiHost,
            RhiBackend=types.SimpleNamespace(OPENGL=3, OPENGL_ES=4),
            RuntimeBackend=types.SimpleNamespace(CPU=0, CUDA=1, VULKAN=2, OPENGL=3, OPENGL_ES=4),
        )
        helper = types.SimpleNamespace(Context=FakeContext)
        with (
            mock.patch.object(runtime_module, "_native", fake_native),
            mock.patch.object(runtime_module, "_gl_context", helper),
        ):
            vd.init(arch=vd.opengl)
            self.assertEqual(created[-1].request, ("opengl", 4, 3))
            self.assertIs(runtime_module._owned_opengl_context, created[-1])
            self.assertEqual(runtime_calls[-1], (3, 11, 12, 13, 4, 3))
            vd.init(arch=vd.opengles)
            self.assertEqual(created[-1].request, ("opengles", 3, 1))
            self.assertEqual(runtime_calls[-1], (4, 11, 12, 13, 3, 1))
            runtime_module._release_runtime()

    def test_registered_context_takes_priority_over_owned_factory(self) -> None:
        class FakeRhiHost:
            calls: list[tuple[object, ...]] = []

            def create_runtime(self) -> object:
                return object()

            @staticmethod
            def create_external_opengl(*arguments: object) -> object:
                FakeRhiHost.calls.append(arguments)
                return FakeRhiHost()

        fake_native = types.SimpleNamespace(
            RhiHost=FakeRhiHost,
            RhiBackend=types.SimpleNamespace(OPENGL=3, OPENGL_ES=4),
            RuntimeBackend=types.SimpleNamespace(CPU=0, CUDA=1, VULKAN=2, OPENGL=3, OPENGL_ES=4),
        )
        helper = mock.Mock()
        previous = runtime_module._external_opengl_contexts.pop(vd.opengl, None)
        try:
            vd.register_external_opengl_context(
                arch=vd.opengl,
                user_data=21,
                make_current=22,
                get_proc_address=23,
                api_version=(3, 3),
            )
            with (
                mock.patch.object(runtime_module, "_native", fake_native),
                mock.patch.object(runtime_module, "_gl_context", helper),
            ):
                vd.init(arch=vd.opengl)
                helper.Context.assert_not_called()
                self.assertEqual(FakeRhiHost.calls[-1], (3, 21, 22, 23, 3, 3))
                runtime_module._release_runtime()
        finally:
            runtime_module._external_opengl_contexts.pop(vd.opengl, None)
            if previous is not None:
                runtime_module._external_opengl_contexts[vd.opengl] = previous


class OpenGLPipelineTests(unittest.TestCase):
    def test_aggregate_attribute_renders(self) -> None:
        assert_aggregate_attribute_renders(self)

    def test_small_multidimensional_aggregate_attribute_renders(self) -> None:
        assert_small_multidimensional_aggregate_attribute_renders(self)

    def test_oversized_aggregate_attribute_reports_device_location_limit(self) -> None:
        assert_oversized_aggregate_attribute_reports_location_limit(self)

    def test_formal_attribute_numeric_formats_render_or_reject(self) -> None:
        assert_formal_attribute_formats_render(self, unsupported=frozenset({"f16", "f64"}))

    def test_non_square_and_divisor_two_attributes_render(self) -> None:
        assert_non_square_and_divisor_two_attributes_render(self)

    def test_instanced_mat4_tensor_attribute_renders(self) -> None:
        assert_instanced_mat4_tensor_attribute_renders(self)

    def test_rank_three_tensor_attribute_renders(self) -> None:
        assert_rank_three_tensor_attribute_renders(self)

    def test_oversized_tensor_attribute_reports_device_location_limit(self) -> None:
        values = vd.storage.from_numpy(np.zeros((3, 8, 8, 4), dtype=np.float32))
        target = vd.Texture.zeros(shape=(8, 8))
        with self.assertRaisesRegex(RuntimeError, "location or format capabilities"):
            vd.pipeline(oversized_tensor_attribute_vertex, solid_fragment)(value=values, target=render_target(target))

    def setUp(self) -> None:
        try:
            vd.init(arch=vd.opengl, api_version=(4, 3))
        except RuntimeError:
            self.skipTest("OpenGL runtime unavailable")

    def test_triangle_renders_to_rgba8_texture(self) -> None:
        render = vd.pipeline(triangle_vertex, solid_fragment)
        positions = vd.storage.from_numpy(
            np.array(
                [
                    (-0.75, -0.75),
                    (0.75, -0.75),
                    (0.0, 0.75),
                ],
                dtype=np.float32,
            )
        )
        target = vd.Texture.zeros(shape=(64, 64))

        attachments = render_target(target)
        render(position=positions, target=attachments)
        pixels = target.to_numpy()

        self.assertEqual(tuple(pixels[0, 0]), (0, 0, 0, 0))
        center = pixels[32, 32]
        self.assertGreater(int(center[0]), 240)
        self.assertGreater(int(center[1]), 40)
        self.assertEqual(int(center[2]), 0)
        self.assertGreater(int(center[3]), 240)
        render(position=positions, target=attachments)
        self.assertEqual(render.compile_count, 1)

    def test_depth_attachment_selects_nearest_fragment(self) -> None:
        assert_depth_attachment_selects_nearest(self)

    def test_matrix_uniform_transforms_vertices(self) -> None:
        assert_matrix_uniform_transforms_vertices(self)

    def test_mixed_uniform_layout_renders(self) -> None:
        assert_mixed_uniform_layout_renders(self)

    def test_cube_faces_remain_distinct(self) -> None:
        assert_cube_faces_remain_distinct(self)

    def test_three_dimensional_texture_samples(self) -> None:
        assert_three_dimensional_texture_samples(self)

    def test_inactive_texture_binding_renders(self) -> None:
        assert_inactive_texture_binding_renders(self)

    def test_mat2_uniform_transforms_vertices(self) -> None:
        assert_mat2_uniform_transforms_vertices(self)

    def test_rank_three_uniform_renders(self) -> None:
        assert_rank_three_uniform_renders(self)

    def test_matrix_elementwise_multiply_renders(self) -> None:
        assert_matrix_elementwise_multiply_renders(self)

    def test_numpy_tensor_vertex_renders(self) -> None:
        assert_numpy_tensor_vertex_renders(self)

    def test_numpy_tensor_fragment_renders(self) -> None:
        assert_numpy_tensor_fragment_renders(self)

    def test_static_tensor_compute_argument(self) -> None:
        assert_static_tensor_compute_argument(self)

    def test_compute_stage_cannot_join_graphics_pipeline(self) -> None:
        with self.assertRaisesRegex(ValueError, "vertex, fragment"):
            vd.pipeline(translate_vertices, triangle_vertex, solid_fragment)

    def test_uniform_tensor_is_shared_draw_state(self) -> None:
        render = vd.pipeline(translated_vertex, solid_fragment)
        positions = vd.storage.from_numpy(
            np.array(
                [
                    (-0.75, -0.75),
                    (0.75, -0.75),
                    (0.0, 0.75),
                ],
                dtype=np.float32,
            )
        )
        offset = vd.storage.from_numpy(np.array((2.0, 0.0), dtype=np.float32))
        target = vd.Texture.zeros(shape=(64, 64))

        render(position=positions, offset=offset, target=render_target(target))

        self.assertEqual(tuple(target.to_numpy()[32, 32]), (0, 0, 0, 0))

    def test_opengl_33_accepts_graphics_only(self) -> None:
        try:
            vd.init(arch=vd.opengl, api_version=(3, 3))
        except RuntimeError:
            self.skipTest("OpenGL 3.3 context unavailable")
        render = vd.pipeline(triangle_vertex, solid_fragment)
        positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
        target = vd.Texture.zeros(shape=(16, 16))
        render(position=positions, target=render_target(target))
        self.assertGreater(int(target.to_numpy()[8, 8, 0]), 240)

    @staticmethod
    def _advanced_inputs() -> tuple[vd.Tensor, vd.Tensor, vd.Tensor]:
        positions = vd.storage.from_numpy(
            np.array(
                ((-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)),
                dtype=np.float32,
            )
        )
        offsets = vd.storage.from_numpy(np.array(((-0.4, 0.0), (0.4, 0.0)), dtype=np.float32))
        indices = vd.storage.from_numpy(np.array((0, 1, 2, 0, 2, 3), dtype=np.uint32))
        return positions, offsets, indices

    def test_indexed_instanced_mrt_variant_and_residency(self) -> None:
        render = vd.pipeline(advanced_vertex, advanced_fragment, features={"PICKING"})
        positions, offsets, indices = self._advanced_inputs()
        color = vd.Texture.zeros(shape=(64, 64))
        object_id = vd.Texture.zeros(shape=(64, 64))
        arguments = {
            "position": positions,
            "offset": offsets,
            "indices": indices,
            "target": mrt_target(color, object_id),
        }
        render(**arguments)
        color_pixels = color.to_numpy()
        id_pixels = object_id.to_numpy()
        self.assertGreater(int(color_pixels[32, 19, 0]), 100)
        self.assertLess(int(color_pixels[32, 19, 0]), 160)
        self.assertGreater(int(color_pixels[32, 19, 1]), 100)
        self.assertLess(int(color_pixels[32, 19, 1]), 160)
        self.assertGreater(int(color_pixels[32, 19, 2]), 240)
        self.assertGreater(int(id_pixels[32, 19, 0]), 240)
        self.assertGreater(int(id_pixels[32, 19, 1]), 40)
        render(**arguments)
        self.assertEqual(render.compile_count, 1)

    def test_feature_and_advanced_draw_validation(self) -> None:
        positions, offsets, indices = self._advanced_inputs()
        color = vd.Texture.zeros(shape=(32, 32))
        object_id = vd.Texture.zeros(shape=(32, 32))
        render = vd.pipeline(advanced_vertex, advanced_fragment, features=("PICKING", "PICKING"))
        with self.assertRaisesRegex(ValueError, "exactly match"):
            render(position=positions, offset=offsets, indices=indices, target=render_target(color))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            vd.RenderTarget(shape=color.shape).attach_color(0, color).attach_color(1, vd.Texture.zeros(shape=(16, 16)))
        with self.assertRaisesRegex(RuntimeError, "shape"):
            render(
                position=positions,
                offset=vd.storage.zeros(dtype=vd.f32, shape=(3, 3)),
                indices=indices,
                target=mrt_target(color, object_id),
            )
        unknown = vd.pipeline(advanced_vertex, advanced_fragment, features={"UNKNOWN"})
        with self.assertRaisesRegex(vd.CompileError, "undeclared feature"):
            unknown(
                position=positions,
                offset=offsets,
                indices=indices,
                target=mrt_target(color, object_id),
            )

    def test_layout_view_and_topologies(self) -> None:
        interleaved = vd.storage.from_numpy(
            np.array(
                ((9.0, -0.75, -0.75, 1.0), (9.0, 0.75, -0.75, 1.0), (9.0, 0.0, 0.75, 1.0)),
                dtype=np.float32,
            )
        )
        target = vd.Texture.zeros(shape=(32, 32))
        vd.pipeline(triangle_vertex, solid_fragment)(position=interleaved.swizzle("yz"), target=render_target(target))
        self.assertGreater(int(target.to_numpy()[16, 16, 0]), 240)

        vertices = vd.TensorStorage.zeros(dtype=InterleavedVertex, shape=(3,))
        vertices.field("position").copy_from_numpy(
            np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32)
        )
        target = vd.Texture.zeros(shape=(32, 32))
        vd.pipeline(triangle_vertex, solid_fragment)(position=vertices.field("position"), target=render_target(target))
        self.assertGreater(int(target.to_numpy()[16, 16, 0]), 240)

        line_positions = vd.storage.from_numpy(np.array(((-0.5, 0.0), (0.5, 0.0)), dtype=np.float32))
        vd.pipeline(triangle_vertex, solid_fragment)(
            position=line_positions, target=render_target(target), topology=vd.lines
        )
        point_positions = vd.storage.from_numpy(np.array(((0.0, 0.0),), dtype=np.float32))
        vd.pipeline(triangle_vertex, solid_fragment)(
            position=point_positions, target=render_target(target), topology=vd.points
        )


class OpenGLESPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            vd.init(arch=vd.opengles, api_version=(3, 1))
        except RuntimeError:
            self.skipTest("OpenGL ES runtime unavailable")

    def test_rank_three_tensor_attribute_arithmetic_renders(self) -> None:
        assert_rank_three_tensor_attribute_renders(self)

    def test_aggregate_attribute_renders(self) -> None:
        assert_aggregate_attribute_renders(self)

    def test_small_multidimensional_aggregate_attribute_renders(self) -> None:
        assert_small_multidimensional_aggregate_attribute_renders(self)

    def test_formal_attribute_numeric_formats_render_or_reject(self) -> None:
        assert_formal_attribute_formats_render(self, unsupported=frozenset({"f16", "f64"}))

    def test_non_square_and_divisor_two_attributes_render(self) -> None:
        assert_non_square_and_divisor_two_attributes_render(self)


class VulkanPipelineTests(unittest.TestCase):
    def test_aggregate_attribute_renders(self) -> None:
        assert_aggregate_attribute_renders(self)

    def test_small_multidimensional_aggregate_attribute_renders(self) -> None:
        assert_small_multidimensional_aggregate_attribute_renders(self)

    def test_formal_attribute_numeric_formats_render_or_reject(self) -> None:
        assert_formal_attribute_formats_render(self, unsupported=frozenset({"f16", "f64"}))

    def test_non_square_and_divisor_two_attributes_render(self) -> None:
        assert_non_square_and_divisor_two_attributes_render(self)

    def test_instanced_mat4_tensor_attribute_renders(self) -> None:
        assert_instanced_mat4_tensor_attribute_renders(self)

    def test_rank_three_tensor_attribute_renders(self) -> None:
        assert_rank_three_tensor_attribute_renders(self)

    def setUp(self) -> None:
        try:
            vd.init(arch=vd.vulkan)
        except RuntimeError:
            self.skipTest("Vulkan runtime unavailable")

    @staticmethod
    def _triangle() -> vd.Tensor:
        return vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))

    def test_triangle_graphics_pipeline(self) -> None:
        positions = self._triangle()
        target = vd.Texture.zeros(shape=(64, 64))
        render = vd.pipeline(triangle_vertex, solid_fragment)
        render(position=positions, target=render_target(target))
        pixels = target.to_numpy()
        self.assertGreater(int(pixels[32, 32, 0]), 240)
        np.testing.assert_allclose(positions.to_numpy()[:, 0], np.array((-0.75, 0.75, 0.0), dtype=np.float32))
        self.assertEqual(render.compile_count, 1)

    def test_depth_attachment_selects_nearest_fragment(self) -> None:
        assert_depth_attachment_selects_nearest(self)

    def test_matrix_uniform_transforms_vertices(self) -> None:
        assert_matrix_uniform_transforms_vertices(self)

    def test_mixed_uniform_layout_renders(self) -> None:
        assert_mixed_uniform_layout_renders(self)

    def test_cube_faces_remain_distinct(self) -> None:
        assert_cube_faces_remain_distinct(self)

    def test_three_dimensional_texture_samples(self) -> None:
        assert_three_dimensional_texture_samples(self)

    def test_inactive_texture_binding_renders(self) -> None:
        assert_inactive_texture_binding_renders(self)

    def test_mat2_uniform_transforms_vertices(self) -> None:
        assert_mat2_uniform_transforms_vertices(self)

    def test_rank_three_uniform_renders(self) -> None:
        assert_rank_three_uniform_renders(self)

    def test_matrix_elementwise_multiply_renders(self) -> None:
        assert_matrix_elementwise_multiply_renders(self)

    def test_numpy_tensor_vertex_renders(self) -> None:
        assert_numpy_tensor_vertex_renders(self)

    def test_numpy_tensor_fragment_renders(self) -> None:
        assert_numpy_tensor_fragment_renders(self)

    def test_static_tensor_compute_argument(self) -> None:
        assert_static_tensor_compute_argument(self)

    def test_stage_uniforms_do_not_overlap_and_y_matches_opengl(self) -> None:
        target = vd.Texture.zeros(shape=(64, 64))
        offset = vd.storage.from_numpy(np.zeros(2, dtype=np.float32))
        color = vd.storage.from_numpy(np.array((0.8, 0.7, 0.2, 1.0), dtype=np.float32))

        vd.pipeline(translated_vertex, colored_fragment)(
            position=self._triangle(), offset=offset, color=color, target=render_target(target)
        )
        pixels = target.to_numpy()

        np.testing.assert_allclose(pixels[32, 32], np.array((204, 178, 51, 255)), atol=1)
        self.assertEqual(tuple(pixels[16, 16]), (0, 0, 0, 0))
        self.assertGreater(int(pixels[48, 16, 0]), 190)

    def test_pipeline_uses_owning_compiler_and_shared_planner(self) -> None:
        render = vd.pipeline(triangle_vertex, solid_fragment)
        positions = self._triangle()
        target = vd.Texture.zeros(shape=(16, 16))
        with (
            mock.patch(
                "subprocess.run",
                side_effect=AssertionError("subprocess prohibited"),
            ),
            mock.patch(
                "vernon_dsl._runtime.pipeline.build_bundle_plan",
                wraps=pipeline_module.build_bundle_plan,
            ) as planner,
        ):
            render(position=positions, target=render_target(target))
        planner.assert_called_once()

    def test_bundle_parameter_output_layout_and_reinit_cache(self) -> None:
        render = vd.pipeline(triangle_vertex, solid_fragment)
        positions = self._triangle()
        target = vd.Texture.zeros(shape=(16, 16))
        attachments = render_target(target)
        render(position=positions, target=attachments)
        compiled = render._compiled
        self.assertIsNotNone(compiled)
        assert compiled is not None
        self.assertTrue(callable(compiled.native.submit))
        self.assertEqual(
            [(parameter.name, parameter.slot, tuple(parameter.shape)) for parameter in compiled.native.parameters],
            [("position", 0, (2,))],
        )
        self.assertEqual(
            [(output.name, output.location) for output in compiled.native.outputs],
            [("output_0", 0)],
        )
        with self.assertRaisesRegex(ValueError, "different reflected kind"):
            compiled.native.invocation_builder().rhi_texture(0, target._resident_view())
        bundle = json.loads(compiled.bundle)
        variant = bundle["variants"][0]
        self.assertEqual(
            [(row["name"], row["slot"], row["kind"]) for row in variant["parameters"]],
            [("position", 0, "tensor")],
        )
        self.assertEqual(
            [(row["name"], row["location"]) for row in variant["outputs"]],
            [("output_0", 0)],
        )
        self.assertNotIn("type", variant["outputs"][0])
        self.assertEqual(render.compile_count, 1)
        vd.init(arch=vd.vulkan)
        render(position=positions, target=attachments)
        self.assertEqual(render.compile_count, 1)

    def test_indexed_instanced_mrt(self) -> None:
        render = vd.pipeline(advanced_vertex, advanced_fragment, features={"PICKING"})
        positions, offsets, indices = OpenGLPipelineTests._advanced_inputs()
        color = vd.Texture.zeros(shape=(64, 64))
        object_id = vd.Texture.zeros(shape=(64, 64))
        render(
            position=positions,
            offset=offsets,
            indices=indices,
            target=mrt_target(color, object_id),
        )
        self.assertGreater(int(color.to_numpy()[32, 19, 2]), 240)
        self.assertGreater(int(object_id.to_numpy()[32, 19, 0]), 240)

    def test_cpu_graphics_has_explicit_error(self) -> None:
        vd.init(arch=vd.cpu)
        with self.assertRaisesRegex(RuntimeError, "software rasterizer"):
            vd.pipeline(triangle_vertex, solid_fragment)(
                position=self._triangle(),
                target=render_target(vd.Texture.zeros(shape=(8, 8))),
            )


class DirectXPipelineTests(unittest.TestCase):
    def test_aggregate_attribute_renders(self) -> None:
        assert_aggregate_attribute_renders(self)

    def test_small_multidimensional_aggregate_attribute_renders(self) -> None:
        assert_small_multidimensional_aggregate_attribute_renders(self)

    def test_formal_attribute_numeric_formats_render_or_reject(self) -> None:
        assert_formal_attribute_formats_render(self, unsupported=frozenset({"f16", "f64"}))

    def test_non_square_and_divisor_two_attributes_render(self) -> None:
        assert_non_square_and_divisor_two_attributes_render(self)

    def test_instanced_mat4_tensor_attribute_renders(self) -> None:
        assert_instanced_mat4_tensor_attribute_renders(self)

    def test_rank_three_tensor_attribute_renders(self) -> None:
        assert_rank_three_tensor_attribute_renders(self)

    def setUp(self) -> None:
        try:
            vd.init(arch=vd.directx)
        except RuntimeError:
            self.skipTest("DirectX 12 runtime unavailable")

    def test_matrix_uniform_transforms_vertices(self) -> None:
        assert_matrix_uniform_transforms_vertices(self)

    def test_mixed_uniform_layout_renders(self) -> None:
        assert_mixed_uniform_layout_renders(self)

    def test_cube_faces_remain_distinct(self) -> None:
        assert_cube_faces_remain_distinct(self)

    def test_three_dimensional_texture_samples(self) -> None:
        assert_three_dimensional_texture_samples(self)

    def test_inactive_texture_binding_renders(self) -> None:
        assert_inactive_texture_binding_renders(self)

    def test_mat2_uniform_transforms_vertices(self) -> None:
        assert_mat2_uniform_transforms_vertices(self)

    def test_rank_three_uniform_renders(self) -> None:
        assert_rank_three_uniform_renders(self)

    def test_matrix_elementwise_multiply_renders(self) -> None:
        assert_matrix_elementwise_multiply_renders(self)

    def test_numpy_tensor_vertex_renders(self) -> None:
        assert_numpy_tensor_vertex_renders(self)

    def test_numpy_tensor_fragment_renders(self) -> None:
        assert_numpy_tensor_fragment_renders(self)

    def test_static_tensor_compute_argument(self) -> None:
        assert_static_tensor_compute_argument(self)


if __name__ == "__main__":
    unittest.main()
