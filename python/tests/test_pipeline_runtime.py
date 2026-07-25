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
from pipeline_shader import (
    colored_fragment,
    solid_fragment,
    translate_vertices,
    translated_vertex,
    triangle_vertex,
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

        class FakeRuntime:
            @staticmethod
            def create_external_opengl(*arguments: object) -> object:
                runtime_calls.append(arguments)
                return object()

        fake_native = types.SimpleNamespace(
            Runtime=FakeRuntime,
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
        class FakeRuntime:
            calls: list[tuple[object, ...]] = []

            @staticmethod
            def create_external_opengl(*arguments: object) -> object:
                FakeRuntime.calls.append(arguments)
                return object()

        fake_native = types.SimpleNamespace(
            Runtime=FakeRuntime,
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
                self.assertEqual(FakeRuntime.calls[-1], (3, 21, 22, 23, 3, 3))
                runtime_module._release_runtime()
        finally:
            runtime_module._external_opengl_contexts.pop(vd.opengl, None)
            if previous is not None:
                runtime_module._external_opengl_contexts[vd.opengl] = previous


class OpenGLPipelineTests(unittest.TestCase):
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

        render(position=positions, offset=offset, target=target)

        self.assertEqual(tuple(target.to_numpy()[32, 32]), (0, 0, 0, 0))

    def test_opengl_33_accepts_graphics_only(self) -> None:
        try:
            vd.init(arch=vd.opengl, api_version=(3, 3))
        except RuntimeError:
            self.skipTest("OpenGL 3.3 context unavailable")
        render = vd.pipeline(triangle_vertex, solid_fragment)
        positions = vd.storage.from_numpy(np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32))
        target = vd.Texture.zeros(shape=(16, 16))
        render(position=positions, target=target)
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

    def test_feature_and_advanced_draw_validation(self) -> None:
        positions, offsets, indices = self._advanced_inputs()
        color = vd.Texture.zeros(shape=(32, 32))
        object_id = vd.Texture.zeros(shape=(32, 32))
        render = vd.pipeline(advanced_vertex, advanced_fragment, features=("PICKING", "PICKING"))
        with self.assertRaisesRegex(ValueError, "exactly match"):
            render(position=positions, offset=offsets, indices=indices, targets={"color": color})
        with self.assertRaisesRegex(RuntimeError, "extents differ"):
            render(
                position=positions,
                offset=offsets,
                indices=indices,
                targets={
                    "color": color,
                    "object_id": vd.Texture.zeros(shape=(16, 16)),
                },
            )
        with self.assertRaisesRegex(RuntimeError, "shape"):
            render(
                position=positions,
                offset=vd.storage.zeros(dtype=vd.f32, shape=(3, 3)),
                indices=indices,
                targets={
                    "color": color,
                    "object_id": object_id,
                },
            )
        unknown = vd.pipeline(advanced_vertex, advanced_fragment, features={"UNKNOWN"})
        with self.assertRaisesRegex(vd.CompileError, "undeclared feature"):
            unknown(
                position=positions,
                offset=offsets,
                indices=indices,
                targets={
                    "color": color,
                    "object_id": object_id,
                },
            )

    def test_layout_view_and_topologies(self) -> None:
        interleaved = vd.storage.from_numpy(
            np.array(
                ((9.0, -0.75, -0.75, 1.0), (9.0, 0.75, -0.75, 1.0), (9.0, 0.0, 0.75, 1.0)),
                dtype=np.float32,
            )
        )
        target = vd.Texture.zeros(shape=(32, 32))
        vd.pipeline(triangle_vertex, solid_fragment)(position=interleaved.swizzle("yz"), target=target)
        self.assertGreater(int(target.to_numpy()[16, 16, 0]), 240)

        vertices = vd.TensorStorage.zeros(dtype=InterleavedVertex, shape=(3,))
        vertices.field("position").copy_from_numpy(
            np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32)
        )
        target = vd.Texture.zeros(shape=(32, 32))
        vd.pipeline(triangle_vertex, solid_fragment)(position=vertices.field("position"), target=target)
        self.assertGreater(int(target.to_numpy()[16, 16, 0]), 240)

        line_positions = vd.storage.from_numpy(np.array(((-0.5, 0.0), (0.5, 0.0)), dtype=np.float32))
        vd.pipeline(triangle_vertex, solid_fragment)(position=line_positions, target=target, topology=vd.lines)
        point_positions = vd.storage.from_numpy(np.array(((0.0, 0.0),), dtype=np.float32))
        vd.pipeline(triangle_vertex, solid_fragment)(position=point_positions, target=target, topology=vd.points)


class VulkanPipelineTests(unittest.TestCase):
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
        render(position=positions, target=target)
        pixels = target.to_numpy()
        self.assertGreater(int(pixels[32, 32, 0]), 240)
        np.testing.assert_allclose(positions.to_numpy()[:, 0], np.array((-0.75, 0.75, 0.0), dtype=np.float32))
        self.assertEqual(render.compile_count, 1)

    def test_stage_uniforms_do_not_overlap_and_y_matches_opengl(self) -> None:
        target = vd.Texture.zeros(shape=(64, 64))
        offset = vd.storage.from_numpy(np.zeros(2, dtype=np.float32))
        color = vd.storage.from_numpy(np.array((0.8, 0.7, 0.2, 1.0), dtype=np.float32))

        vd.pipeline(translated_vertex, colored_fragment)(
            position=self._triangle(), offset=offset, color=color, target=target
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
            render(position=positions, target=target)
        planner.assert_called_once()

    def test_bundle_parameter_output_layout_and_reinit_cache(self) -> None:
        render = vd.pipeline(triangle_vertex, solid_fragment)
        positions = self._triangle()
        target = vd.Texture.zeros(shape=(16, 16))
        render(position=positions, target=target)
        compiled = render._compiled
        self.assertIsNotNone(compiled)
        assert compiled is not None
        self.assertTrue(callable(compiled.native.invoke))
        self.assertEqual(
            [(parameter.name, parameter.slot, tuple(parameter.shape)) for parameter in compiled.native.parameters],
            [("position", 0, (2,))],
        )
        self.assertEqual(
            [(output.name, output.location) for output in compiled.native.outputs],
            [("output_0", 0)],
        )
        with self.assertRaisesRegex(ValueError, "different reflected kind"):
            compiled.native.invocation_builder().rhi_texture(0, target._resident_texture())
        bundle = json.loads(compiled.bundle)
        variant = bundle["variants"][0]
        self.assertEqual(
            [(row["name"], row["slot"], row["kind"]) for row in variant["parameters"]],
            [("position", 0, "tensor")],
        )
        self.assertEqual(
            [(row["name"], row["location"], row["type"]) for row in variant["outputs"]],
            [("output_0", 0, "tensor<4xf32>")],
        )
        self.assertEqual(render.compile_count, 1)
        vd.init(arch=vd.vulkan)
        render(position=positions, target=target)
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
            targets={
                "color": color,
                "object_id": object_id,
            },
        )
        self.assertGreater(int(color.to_numpy()[32, 19, 2]), 240)
        self.assertGreater(int(object_id.to_numpy()[32, 19, 0]), 240)

    def test_cpu_graphics_has_explicit_error(self) -> None:
        vd.init(arch=vd.cpu)
        with self.assertRaisesRegex(RuntimeError, "software rasterizer"):
            vd.pipeline(triangle_vertex, solid_fragment)(
                position=self._triangle(), target=vd.Texture.zeros(shape=(8, 8))
            )


if __name__ == "__main__":
    unittest.main()
