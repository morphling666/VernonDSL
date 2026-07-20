from __future__ import annotations

import unittest

import numpy as np

import vernon_dsl as vd
from python.tests.advanced_pipeline_shader import (  # type: ignore[import-not-found]
    advanced_fragment, advanced_vertex, feature_compute,
)
from python.tests.pipeline_shader import (  # type: ignore[import-not-found]
    solid_fragment, translate_vertices, triangle_vertex, translated_vertex,
)


@vd.func
def shader_helper(value: vd.f32) -> vd.f32:
    return value


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

    def test_tensor_layout_and_contiguous_swizzle(self) -> None:
        tensor = vd.Tensor.from_numpy(np.zeros((3, 4), dtype=np.float32))
        self.assertEqual(tensor.layout.shape, (3, 4))
        self.assertEqual(tensor.layout.byte_strides, (16, 4))
        view = tensor.swizzle("yz")
        self.assertEqual(view.shape, (3, 2))
        self.assertEqual(view.layout.byte_offset, 4)
        self.assertEqual(view.layout.components, (1, 2))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            tensor.swizzle("zx")

    def test_external_gl_backends_require_registration(self) -> None:
        for architecture in (vd.opengl, vd.opengles):
            with self.subTest(architecture=architecture.name):
                with self.assertRaisesRegex(
                        RuntimeError,
                        "requires a registered host-owned external context"):
                    vd.init(arch=architecture)
        vd.init(arch=vd.cpu)


class OpenGLPipelineTests(unittest.TestCase):

    def setUp(self) -> None:
        try:
            vd.init(arch=vd.opengl, api_version=(4, 3))
        except RuntimeError:
            self.skipTest("OpenGL runtime unavailable")

    def test_triangle_renders_to_rgba8_texture(self) -> None:
        render = vd.pipeline(triangle_vertex, solid_fragment)
        positions = vd.Tensor.from_numpy(
            np.array(
                [
                    (-0.75, -0.75),
                    (0.75, -0.75),
                    (0.0, 0.75),
                ],
                dtype=np.float32,
            ))
        target = vd.Texture.zeros(shape=(64, 64))

        render(position=positions, target=target)
        pixels = target.to_numpy()

        self.assertEqual(tuple(pixels[0, 0]), (0, 0, 0, 0))
        center = pixels[32, 32]
        self.assertGreater(int(center[0]), 240)
        self.assertGreater(int(center[1]), 40)
        self.assertEqual(int(center[2]), 0)
        self.assertGreater(int(center[3]), 240)
        render(position=positions, target=target)
        self.assertEqual(render.compile_count, 1)

    def test_compute_stage_runs_before_graphics(self) -> None:
        render = vd.pipeline(translate_vertices, triangle_vertex,
                             solid_fragment)
        positions = vd.Tensor.from_numpy(
            np.array(
                [
                    (-0.75, -0.75),
                    (0.75, -0.75),
                    (0.0, 0.75),
                ],
                dtype=np.float32,
            ))
        target = vd.Texture.zeros(shape=(64, 64))

        render(position=positions, offset=2.0, target=target)

        self.assertEqual(tuple(target.to_numpy()[32, 32]), (0, 0, 0, 0))

    def test_uniform_tensor_is_shared_draw_state(self) -> None:
        render = vd.pipeline(translated_vertex, solid_fragment)
        positions = vd.Tensor.from_numpy(
            np.array(
                [
                    (-0.75, -0.75),
                    (0.75, -0.75),
                    (0.0, 0.75),
                ],
                dtype=np.float32,
            ))
        offset = vd.Tensor.from_numpy(np.array((2.0, 0.0), dtype=np.float32))
        target = vd.Texture.zeros(shape=(64, 64))

        render(position=positions, offset=offset, target=target)

        self.assertEqual(tuple(target.to_numpy()[32, 32]), (0, 0, 0, 0))

    def test_opengl_33_rejects_compute_composition(self) -> None:
        try:
            vd.init(arch=vd.opengl, api_version=(3, 3))
        except RuntimeError:
            self.skipTest("OpenGL 3.3 context unavailable")
        render = vd.pipeline(translate_vertices, triangle_vertex,
                             solid_fragment)
        positions = vd.Tensor.from_numpy(np.zeros((3, 2), dtype=np.float32))
        target = vd.Texture.zeros(shape=(8, 8))
        with self.assertRaisesRegex(RuntimeError, "OpenGL 4.3"):
            render(position=positions, offset=0.0, target=target)

    def test_opengl_33_accepts_graphics_only(self) -> None:
        try:
            vd.init(arch=vd.opengl, api_version=(3, 3))
        except RuntimeError:
            self.skipTest("OpenGL 3.3 context unavailable")
        render = vd.pipeline(triangle_vertex, solid_fragment)
        positions = vd.Tensor.from_numpy(
            np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)),
                     dtype=np.float32))
        target = vd.Texture.zeros(shape=(16, 16))
        render(position=positions, target=target)
        self.assertGreater(int(target.to_numpy()[8, 8, 0]), 240)

    @staticmethod
    def _advanced_inputs() -> tuple[vd.Tensor, vd.Tensor, vd.Tensor]:
        positions = vd.Tensor.from_numpy(
            np.array(
                ((-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)),
                dtype=np.float32,
            ))
        offsets = vd.Tensor.from_numpy(
            np.array(((-0.4, 0.0), (0.4, 0.0)), dtype=np.float32))
        indices = vd.Tensor.from_numpy(
            np.array((0, 1, 2, 0, 2, 3), dtype=np.uint32))
        return positions, offsets, indices

    def test_indexed_instanced_mrt_variant_and_residency(self) -> None:
        render = vd.pipeline(advanced_vertex,
                             advanced_fragment,
                             features={"PICKING"})
        positions, offsets, indices = self._advanced_inputs()
        color = vd.Texture.zeros(shape=(64, 64))
        object_id = vd.Texture.zeros(shape=(64, 64))
        arguments = {
            "position": positions,
            "offset": offsets,
            "indices": indices,
            "targets": {
                "object_id": object_id,
                "color": color,
            },
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
        self.assertEqual(indices._allocation_count, 1)
        render(**arguments)
        self.assertEqual(indices._allocation_count, 1)
        self.assertEqual(render.compile_count, 1)

    def test_compute_stage_uses_pipeline_feature_set(self) -> None:
        render = vd.pipeline(feature_compute,
                             advanced_vertex,
                             advanced_fragment,
                             features={"PICKING"})
        positions, offsets, indices = self._advanced_inputs()
        render(position=positions,
               offset=offsets,
               indices=indices,
               targets={
                   "color": vd.Texture.zeros(shape=(16, 16)),
                   "object_id": vd.Texture.zeros(shape=(16, 16)),
               })
        self.assertEqual(feature_compute.compile_count, 1)

    def test_feature_and_advanced_draw_validation(self) -> None:
        positions, offsets, indices = self._advanced_inputs()
        color = vd.Texture.zeros(shape=(32, 32))
        object_id = vd.Texture.zeros(shape=(32, 32))
        render = vd.pipeline(advanced_vertex,
                             advanced_fragment,
                             features=("PICKING", "PICKING"))
        with self.assertRaisesRegex(ValueError, "exactly match"):
            render(position=positions,
                   offset=offsets,
                   indices=indices,
                   targets={"color": color})
        with self.assertRaisesRegex(ValueError, "same extent"):
            render(position=positions,
                   offset=offsets,
                   indices=indices,
                   targets={
                       "color": color,
                       "object_id": vd.Texture.zeros(shape=(16, 16)),
                   })
        with self.assertRaisesRegex(ValueError, "incompatible shape"):
            render(position=positions,
                   offset=vd.Tensor.zeros(dtype=vd.f32, shape=(3, 3)),
                   indices=indices,
                   targets={
                       "color": color,
                       "object_id": object_id,
                   })
        invalid_indices = vd.Tensor.from_numpy(
            np.array((0, 1, 9), dtype=np.uint32))
        with self.assertRaisesRegex(ValueError, "missing vertex"):
            render(position=positions,
                   offset=offsets,
                   indices=invalid_indices,
                   targets={
                       "color": color,
                       "object_id": object_id,
                   })
        unknown = vd.pipeline(advanced_vertex,
                              advanced_fragment,
                              features={"UNKNOWN"})
        with self.assertRaisesRegex(vd.CompileError, "undeclared feature"):
            unknown(position=positions,
                    offset=offsets,
                    indices=indices,
                    targets={
                        "color": color,
                        "object_id": object_id,
                    })

    def test_layout_view_and_topologies(self) -> None:
        interleaved = vd.Tensor.from_numpy(
            np.array(
                ((9.0, -0.75, -0.75, 1.0), (9.0, 0.75, -0.75, 1.0),
                 (9.0, 0.0, 0.75, 1.0)),
                dtype=np.float32,
            ))
        target = vd.Texture.zeros(shape=(32, 32))
        vd.pipeline(triangle_vertex,
                    solid_fragment)(position=interleaved.swizzle("yz"),
                                    target=target)
        self.assertGreater(int(target.to_numpy()[16, 16, 0]), 240)

        line_positions = vd.Tensor.from_numpy(
            np.array(((-0.5, 0.0), (0.5, 0.0)), dtype=np.float32))
        vd.pipeline(triangle_vertex, solid_fragment)(position=line_positions,
                                                     target=target,
                                                     topology=vd.lines)
        point_positions = vd.Tensor.from_numpy(
            np.array(((0.0, 0.0), ), dtype=np.float32))
        vd.pipeline(triangle_vertex, solid_fragment)(position=point_positions,
                                                     target=target,
                                                     topology=vd.points)


class VulkanPipelineTests(unittest.TestCase):

    def setUp(self) -> None:
        try:
            vd.init(arch=vd.vulkan)
        except RuntimeError:
            self.skipTest("Vulkan runtime unavailable")

    @staticmethod
    def _triangle() -> vd.Tensor:
        return vd.Tensor.from_numpy(
            np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)),
                     dtype=np.float32))

    def test_triangle_and_compute_graphics_pipeline(self) -> None:
        positions = self._triangle()
        target = vd.Texture.zeros(shape=(64, 64))
        render = vd.pipeline(translate_vertices, triangle_vertex,
                             solid_fragment)
        render(position=positions, offset=np.float32(0.25), target=target)
        pixels = target.to_numpy()
        self.assertGreater(int(pixels[32, 40, 0]), 240)
        np.testing.assert_allclose(
            positions.to_numpy()[:, 0],
            np.array((-0.5, 1.0, 0.25), dtype=np.float32),
        )
        self.assertEqual(render.compile_count, 1)

    def test_indexed_instanced_mrt(self) -> None:
        render = vd.pipeline(advanced_vertex,
                             advanced_fragment,
                             features={"PICKING"})
        positions, offsets, indices = OpenGLPipelineTests._advanced_inputs()
        color = vd.Texture.zeros(shape=(64, 64))
        object_id = vd.Texture.zeros(shape=(64, 64))
        render(position=positions,
               offset=offsets,
               indices=indices,
               targets={
                   "color": color,
                   "object_id": object_id,
               })
        self.assertGreater(int(color.to_numpy()[32, 19, 2]), 240)
        self.assertGreater(int(object_id.to_numpy()[32, 19, 0]), 240)

    def test_cpu_graphics_has_explicit_error(self) -> None:
        vd.init(arch=vd.cpu)
        with self.assertRaisesRegex(RuntimeError, "software rasterizer"):
            vd.pipeline(triangle_vertex,
                        solid_fragment)(position=self._triangle(),
                                        target=vd.Texture.zeros(shape=(8, 8)))


if __name__ == "__main__":
    unittest.main()
