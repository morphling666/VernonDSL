from __future__ import annotations

import unittest
from importlib import import_module
from typing import Annotated, Any, cast
from unittest import mock

import numpy as np
import vernon_dsl as vd
from vernon_dsl._runtime.session import RuntimeUnavailableError
from vernon_dsl.frontend.module_ast import interpret_module_forward
from vernon_dsl.frontend.runtime_types import RuntimeParameterDescriptor, runtime_parameter_descriptor
from vernon_dsl.operation_graph import GraphicsCallOp, ProgramControlDescriptor
from vernon_dsl.program import ProgramTemplate
from vernon_dsl.program_frontend import parse_program


@vd.vertex
def managed_vertex(
    vertices: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([vertices, 0.0, 1.0])


@vd.fragment
def managed_fragment() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 1.0, 1.0, 1.0])


@vd.fragment
def managed_blend_fragment() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 0.0, 0.0, 0.5])


managed_pipeline = vd.pipeline(managed_vertex, managed_fragment)


@vd.fragment
def shadow_fragment() -> None:
    return


@vd.vertex
def shadow_vertex(
    vertices: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([vertices, 0.5, 1.0])


@vd.fragment
def pbr_fragment(
    shadow_map: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=0)],  # noqa: F722
    shadow_sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
) -> vd.Vector[vd.f32, 4]:
    depth = vd.texture_sample(shadow_map, shadow_sampler, vd.Vector([0.5, 0.5]))
    return vd.Vector([depth.x, depth.x, depth.x, 1.0])


shadow_pipeline = vd.pipeline(
    shadow_vertex,
    shadow_fragment,
    state=vd.graphics_state(depth_stencil=vd.DepthStencilState(depth_test=True, depth_write=True)),
)
pbr_pipeline = vd.pipeline(managed_vertex, pbr_fragment)


@vd.kernel(workgroup_size=(1, 1, 1))
def generate_vertices(
    vertices: vd.TensorView[vd.f32, (3, 2), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    vertices[index, 0] = vd.f32(index) - 1.0
    vertices[index, 1] = -0.5 if index < 2 else 0.5


culled_pipeline = vd.pipeline(
    managed_vertex,
    managed_fragment,
    state=vd.graphics_state(rasterization=vd.RasterizationState(cull_mode=vd.CullMode.FRONT)),
)
blended_pipeline = vd.pipeline(
    managed_vertex,
    managed_blend_fragment,
    state=vd.graphics_state(
        color_blends={
            0: vd.ColorBlendState(
                enabled=True,
                source_color=vd.BlendFactor.SOURCE_ALPHA,
                destination_color=vd.BlendFactor.ONE_MINUS_SOURCE_ALPHA,
            )
        }
    ),
)


class ManagedGraphics(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> None:
        managed_pipeline(
            vertices=vertices,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )


class ManagedGraphicsTwice(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        first_pass: vd.RenderPass,
        first_draw: vd.DrawCommand,
        first_dynamic: vd.DynamicState,
        second_pass: vd.RenderPass,
        second_draw: vd.DrawCommand,
        second_dynamic: vd.DynamicState,
    ) -> None:
        managed_pipeline(
            vertices=vertices,
            render_pass=first_pass,
            draw=first_draw,
            dynamic_state=first_dynamic,
        )
        managed_pipeline(
            vertices=vertices,
            render_pass=second_pass,
            draw=second_draw,
            dynamic_state=second_dynamic,
        )


class ManagedGraphicsOutput(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        managed_pipeline(
            vertices=vertices,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )
        return vd.color_output(render_pass)


class ManagedGraphicsSamePass(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> None:
        managed_pipeline(
            vertices=vertices,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )
        managed_pipeline(
            vertices=vertices,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )


class ShadowPbrProgram(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        shadow_pass: vd.RenderPass,
        shadow_draw: vd.DrawCommand,
        shadow_dynamic: vd.DynamicState,
        pbr_pass: vd.RenderPass,
        pbr_draw: vd.DrawCommand,
        pbr_dynamic: vd.DynamicState,
        shadow_sampler: vd.Sampler,
    ) -> Any:
        shadow_pipeline(
            vertices=vertices,
            render_pass=shadow_pass,
            draw=shadow_draw,
            dynamic_state=shadow_dynamic,
        )
        shadow = vd.depth_output(shadow_pass)
        pbr_pipeline(
            vertices=vertices,
            shadow_map=shadow,
            shadow_sampler=shadow_sampler,
            render_pass=pbr_pass,
            draw=pbr_draw,
            dynamic_state=pbr_dynamic,
        )
        return vd.color_output(pbr_pass)


class ComputeGeneratedVertices(vd.Module):
    def forward(
        self,
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        vertices = vd.empty(dtype=vd.f32, shape=(3, 2))
        generate_vertices(vertices, grid=(3, 1, 1))
        managed_pipeline(
            vertices=vertices,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )
        return vd.color_output(render_pass)


class CulledManagedGraphics(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> None:
        culled_pipeline(
            vertices=vertices,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )


class DefaultedManagedGraphics(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand | None = None,
        dynamic_state: vd.DynamicState | None = None,
    ) -> None:
        managed_pipeline(
            vertices=vertices,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )


class BlendedManagedGraphics(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        render_pass: vd.RenderPass,
    ) -> None:
        blended_pipeline(vertices=vertices, render_pass=render_pass)


class _Format:
    name = "rgba8_unorm"


class _Texture:
    shape = (16, 16)
    format = _Format()


class _Target:
    def _color_attachments(self):
        return ((0, _Texture()),)

    def _depth_attachment(self):
        return None


class _DepthFormat:
    name = "d32_float"


class _DepthTexture:
    shape = (16, 16)
    format = _DepthFormat()


class _DepthTarget:
    def _color_attachments(self):
        return ()

    def _depth_attachment(self):
        return _DepthTexture()


class ModuleGraphicsControlTests(unittest.TestCase):
    @staticmethod
    def _cook(parsed: Any, pipeline_id: str) -> dict[str, Any]:
        from vernon_dsl._program_assets.cooking import _compile_program_bundle_plan, _native_target
        from vernon_dsl.bundle import make_target_options

        native = import_module("vernon_dsl._native")
        plan = _compile_program_bundle_plan(
            parsed,
            pipeline_id=pipeline_id,
            variant=(),
            target=make_target_options("opengl", {"version": 410}),
            compiler=native.Compiler(),
            native=native,
            native_target=_native_target(native, "opengl"),
        )
        canonical = plan.variants[0].canonical_program
        assert canonical is not None
        return canonical

    def test_every_managed_invocation_requires_render_pass_control(self) -> None:
        vertices = vd.storage.from_numpy(np.zeros((3, 2), dtype=np.float32)).view(access="read")
        with self.assertRaisesRegex(TypeError, "Module invocation must provide render_pass"):
            ManagedGraphics()(vertices=vertices)

    def test_graphics_call_has_independent_stable_control_slots(self) -> None:
        render_pass = vd.RenderPass(cast(vd.RenderTarget, _Target()), ((0, vd.preserve()),))
        parameter_types = {
            "vertices": RuntimeParameterDescriptor.storage(vd.f32, (3, 2), "read", True),
            "render_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, render_pass),
            "draw": ProgramControlDescriptor("draw", vd.DrawCommand, vd.draw(vertex_count=3)),
            "dynamic_state": ProgramControlDescriptor(
                "dynamic_state",
                vd.DynamicState,
                vd.dynamic_state(viewport=(0, 0, 16, 16)),
            ),
        }
        capture, outputs = interpret_module_forward(ManagedGraphics(), parameter_types)
        invocation = ProgramTemplate.compile(capture).bind(capture, outputs)

        operation = invocation.graph.operations[0]
        self.assertIsInstance(operation, GraphicsCallOp)
        assert isinstance(operation, GraphicsCallOp)
        self.assertEqual(tuple(operation.control_slots), ("render_pass", "draw", "dynamic_state"))
        self.assertEqual(len(set(operation.control_slots.values())), 3)
        self.assertEqual(
            tuple(operation.control_slots.values()),
            tuple(range(operation.control_slots["render_pass"], operation.control_slots["render_pass"] + 3)),
        )

        parsed = parse_program(invocation)
        self.assertIn('"vernon_program.graphics"', parsed.mlir)
        self.assertIn("vernon_program.control_slots", parsed.mlir)
        self.assertIn("vernon_program.graphics_state", parsed.mlir)
        self.assertEqual(parsed.implementations[0].kind, "graphics")
        self.assertEqual(
            tuple(role for role, _, _ in parsed.implementations[0].graphics_stages),
            ("vertex", "fragment"),
        )

    def test_graphics_module_autodiff_is_diagnosed(self) -> None:
        from vernon_dsl.program import _parse_module_program

        with self.assertRaisesRegex(TypeError, "graphics Module programs do not support autodiff"):
            with mock.patch(
                "vernon_dsl.program._module_parameter_types",
                return_value={
                    "vertices": RuntimeParameterDescriptor.storage(vd.f32, (3, 2), "read", True),
                    "render_pass": ProgramControlDescriptor(
                        "render_pass",
                        vd.RenderPass,
                        vd.RenderPass(cast(vd.RenderTarget, _Target()), ((0, vd.preserve()),)),
                    ),
                    "draw": ProgramControlDescriptor("draw", vd.DrawCommand, vd.draw(vertex_count=3)),
                    "dynamic_state": ProgramControlDescriptor(
                        "dynamic_state",
                        vd.DynamicState,
                        vd.dynamic_state(),
                    ),
                },
            ):
                _parse_module_program(ManagedGraphics(), vjp_wrt=("vertices",))

    def test_two_graphics_nodes_keep_distinct_control_slots(self) -> None:
        render_pass = vd.RenderPass(cast(vd.RenderTarget, _Target()), ((0, vd.preserve()),))
        parameter_types = {
            "vertices": RuntimeParameterDescriptor.storage(vd.f32, (3, 2), "read", True),
            "first_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, render_pass),
            "first_draw": ProgramControlDescriptor("draw", vd.DrawCommand, vd.draw(vertex_count=3)),
            "first_dynamic": ProgramControlDescriptor("dynamic_state", vd.DynamicState, vd.dynamic_state()),
            "second_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, render_pass),
            "second_draw": ProgramControlDescriptor("draw", vd.DrawCommand, vd.draw(vertex_count=3)),
            "second_dynamic": ProgramControlDescriptor("dynamic_state", vd.DynamicState, vd.dynamic_state()),
        }
        capture, outputs = interpret_module_forward(ManagedGraphicsTwice(), parameter_types)
        invocation = ProgramTemplate.compile(capture).bind(capture, outputs)

        first, second = invocation.graph.operations
        self.assertIsInstance(first, GraphicsCallOp)
        self.assertIsInstance(second, GraphicsCallOp)
        assert isinstance(first, GraphicsCallOp) and isinstance(second, GraphicsCallOp)
        self.assertTrue(set(first.control_slots.values()).isdisjoint(second.control_slots.values()))
        parsed = parse_program(invocation)
        self.assertEqual(parsed.mlir.count("vernon_program.control_slots"), 2)

    def test_attachment_output_projects_the_written_resource_version(self) -> None:
        render_pass = vd.RenderPass(cast(vd.RenderTarget, _Target()), ((0, vd.preserve()),))
        parameter_types = {
            "vertices": RuntimeParameterDescriptor.storage(vd.f32, (3, 2), "read", True),
            "render_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, render_pass),
            "draw": ProgramControlDescriptor("draw", vd.DrawCommand, vd.draw(vertex_count=3)),
            "dynamic_state": ProgramControlDescriptor("dynamic_state", vd.DynamicState, vd.dynamic_state()),
        }
        capture, outputs = interpret_module_forward(ManagedGraphicsOutput(), parameter_types)
        invocation = ProgramTemplate.compile(capture).bind(capture, outputs)
        operation = invocation.graph.operations[0]
        output_id = invocation.graph.outputs["output"]
        self.assertEqual(output_id, operation.outputs["render_pass.color.0"])
        parsed = parse_program(invocation)
        planned = import_module("vernon_dsl._native").Compiler().plan_program_result(parsed.mlir)
        self.assertTrue(planned.ok, planned.diagnostics)

    def test_reused_render_pass_forms_an_attachment_version_chain(self) -> None:
        render_pass = vd.RenderPass(cast(vd.RenderTarget, _Target()), ((0, vd.preserve()),))
        parameter_types = {
            "vertices": RuntimeParameterDescriptor.storage(vd.f32, (3, 2), "read", True),
            "render_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, render_pass),
            "draw": ProgramControlDescriptor("draw", vd.DrawCommand, vd.draw(vertex_count=3)),
            "dynamic_state": ProgramControlDescriptor("dynamic_state", vd.DynamicState, vd.dynamic_state()),
        }
        capture, outputs = interpret_module_forward(ManagedGraphicsSamePass(), parameter_types)
        invocation = ProgramTemplate.compile(capture).bind(capture, outputs)
        first, second = invocation.graph.operations
        self.assertEqual(
            first.outputs["render_pass.color.0"],
            second.inputs["render_pass.color.0"],
        )

    def test_shadow_depth_output_is_the_pbr_sampled_input(self) -> None:
        shadow_pass = vd.RenderPass(cast(vd.RenderTarget, _DepthTarget()), (), vd.preserve())
        pbr_pass = vd.RenderPass(cast(vd.RenderTarget, _Target()), ((0, vd.preserve()),))
        parameter_types = {
            "vertices": RuntimeParameterDescriptor.storage(vd.f32, (3, 2), "read", True),
            "shadow_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, shadow_pass),
            "shadow_draw": ProgramControlDescriptor("draw", vd.DrawCommand, vd.draw(vertex_count=3)),
            "shadow_dynamic": ProgramControlDescriptor("dynamic_state", vd.DynamicState, vd.dynamic_state()),
            "pbr_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, pbr_pass),
            "pbr_draw": ProgramControlDescriptor("draw", vd.DrawCommand, vd.draw(vertex_count=3)),
            "pbr_dynamic": ProgramControlDescriptor("dynamic_state", vd.DynamicState, vd.dynamic_state()),
            "shadow_sampler": runtime_parameter_descriptor(vd.Sampler),
        }
        capture, outputs = interpret_module_forward(ShadowPbrProgram(), parameter_types)
        invocation = ProgramTemplate.compile(capture).bind(capture, outputs)
        shadow, pbr = invocation.graph.operations
        shadow_depth = shadow.outputs["shadow_pass.depth.0"]
        self.assertEqual(shadow_depth, pbr.inputs["shadow_map"])
        self.assertEqual(invocation.graph.outputs["output"], pbr.outputs["pbr_pass.color.0"])

        parsed = parse_program(invocation)
        planned = import_module("vernon_dsl._native").Compiler().plan_program_result(parsed.mlir)
        self.assertTrue(planned.ok, planned.diagnostics)
        canonical = self._cook(parsed, "tests/shadow-pbr")
        shadow_node, pbr_node = canonical["graphs"][0]["nodes"]
        shadow_write = next(access for access in shadow_node["accesses"] if access["tag"] == "attachment")
        pbr_read = next(
            access
            for access in pbr_node["accesses"]
            if access["tag"] == "read" and access["storage"] == shadow_write["storage"]
        )
        self.assertEqual(shadow_write["after"], pbr_read["value"])
        depth_storage = next(
            storage for storage in canonical["storages"] if storage["descriptor"].get("aspects") == ["depth"]
        )
        self.assertEqual(
            set(depth_storage["descriptor"]["usage"]),
            {"depth_stencil_attachment", "sampled"},
        )

    def test_compute_written_vertices_feed_the_graphics_node(self) -> None:
        render_pass = vd.RenderPass(cast(vd.RenderTarget, _Target()), ((0, vd.preserve()),))
        parameter_types = {
            "render_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, render_pass),
            "draw": ProgramControlDescriptor("draw", vd.DrawCommand, vd.draw(vertex_count=3)),
            "dynamic_state": ProgramControlDescriptor("dynamic_state", vd.DynamicState, vd.dynamic_state()),
        }
        capture, outputs = interpret_module_forward(ComputeGeneratedVertices(), parameter_types)
        invocation = ProgramTemplate.compile(capture).bind(capture, outputs)
        compute, graphics = invocation.graph.operations
        generated = next(iter(compute.outputs.values()))
        self.assertIn(generated, graphics.inputs.values())
        self.assertEqual(invocation.graph.outputs["output"], graphics.outputs["render_pass.color.0"])

        parsed = parse_program(invocation)
        planned = import_module("vernon_dsl._native").Compiler().plan_program_result(parsed.mlir)
        self.assertTrue(planned.ok, planned.diagnostics)
        canonical = self._cook(parsed, "tests/compute-generated-vertices")
        compute_node, graphics_node = canonical["graphs"][0]["nodes"]
        compute_write = next(access for access in compute_node["accesses"] if access["tag"] == "write")
        graphics_read = next(access for access in graphics_node["accesses"] if access["tag"] == "read")
        self.assertEqual(compute_write["after"], graphics_read["value"])
        vertex_storage = canonical["storages"][compute_write["storage"]]
        self.assertEqual(set(vertex_storage["descriptor"]["usage"]), {"storage", "vertex"})

    def test_managed_graphics_cooking_packages_both_shader_stages(self) -> None:
        from vernon_dsl._program_assets.cooking import _compile_program_bundle_plan, _native_target
        from vernon_dsl.bundle import make_target_options
        from vernon_dsl.program import _parse_module_program

        render_pass = vd.RenderPass(cast(vd.RenderTarget, _Target()), ((0, vd.preserve()),))
        parsed = _parse_module_program(
            ManagedGraphics(),
            (
                vd.storage.from_numpy(np.zeros((3, 2), dtype=np.float32)).view(access="read"),
                render_pass,
                vd.draw(vertex_count=3),
                vd.dynamic_state(),
            ),
        )
        native = import_module("vernon_dsl._native")
        target = make_target_options("opengl", {"version": 410})
        plan = _compile_program_bundle_plan(
            parsed,
            pipeline_id="tests/managed-graphics",
            variant=(),
            target=target,
            compiler=native.Compiler(),
            native=native,
            native_target=_native_target(native, "opengl"),
        )

        self.assertEqual(len(plan.stages), 1)
        self.assertEqual(plan.stages[0].stage, "graphics")
        self.assertEqual(
            tuple(module["role"] for module in plan.stages[0].metadata["graphics_modules"]),
            ("vertex", "fragment"),
        )
        canonical_program = plan.variants[0].canonical_program
        assert canonical_program is not None
        operation = canonical_program["graphs"][0]["nodes"][0]["operation"]
        self.assertEqual(operation["render_pass"]["control"], 1)
        self.assertEqual(operation["draw"]["control"], 2)
        self.assertEqual(operation["dynamic_state"]["control"], 3)

    def test_real_opengl_managed_module_uses_distinct_node_controls(self) -> None:
        try:
            vd.init(arch=vd.opengl, api_version=(3, 3))
        except RuntimeUnavailableError as error:
            self.skipTest(f"OpenGL runtime unavailable: {error}")
        try:
            vertices = vd.storage.from_numpy(np.array(((-0.8, -0.8), (0.8, -0.8), (0.0, 0.8)), dtype=np.float32)).view(
                access="read"
            )
            first_texture = vd.Texture.zeros(shape=(16, 16))
            second_texture = vd.Texture.zeros(shape=(16, 16))
            first_target = vd.RenderTarget.from_attachments(colors={0: first_texture})
            second_target = vd.RenderTarget.from_attachments(colors={0: second_texture})
            module = ManagedGraphicsTwice()
            module(
                vertices,
                vd.render_pass(first_target, color=vd.clear((0.0, 0.0, 0.0, 1.0))),
                vd.draw(vertex_count=3),
                vd.dynamic_state(viewport=(0, 0, 16, 16)),
                vd.render_pass(second_target, color=vd.clear((0.0, 0.0, 0.0, 1.0))),
                vd.draw(vertex_count=3),
                vd.dynamic_state(viewport=(0, 0, 16, 16)),
            )

            specialization = next(iter(module._program_cache.values()))
            controls = specialization.invocation.graphics_controls
            self.assertEqual(len(controls), 2)
            self.assertNotEqual(controls[0]["render_pass"][0], controls[1]["render_pass"][0])
            first_pixels = first_texture.to_numpy()
            second_pixels = second_texture.to_numpy()
            self.assertGreater(first_pixels[..., :3].sum(), 0)
            self.assertGreater(second_pixels[..., :3].sum(), 0)
        finally:
            vd.init(arch=vd.cpu)

    def test_real_metal_executes_shadow_depth_sampled_by_pbr(self) -> None:
        try:
            vd.init(arch=vd.metal)
        except RuntimeUnavailableError as error:
            self.skipTest(f"Metal runtime unavailable: {error}")
        try:
            vertices = vd.storage.from_numpy(np.array(((-0.8, -0.8), (0.8, -0.8), (0.0, 0.8)), dtype=np.float32)).view(
                access="read"
            )
            depth = vd.Texture.device(
                shape=(16, 16),
                format=vd.d32_float,
                usage=("depth_stencil_attachment", "sampled"),
            )
            shadow_target = vd.RenderTarget.from_attachments(colors={}, depth=depth)
            pbr_texture = vd.Texture.zeros(shape=(16, 16))
            pbr_target = vd.RenderTarget.from_attachments(colors={0: pbr_texture})
            controls = (vd.draw(vertex_count=3), vd.dynamic_state(viewport=(0, 0, 16, 16)))
            ShadowPbrProgram()(
                vertices,
                vd.render_pass(shadow_target, depth=vd.clear_depth(1.0)),
                *controls,
                vd.render_pass(pbr_target, color=vd.clear((0.0, 0.0, 0.0, 1.0))),
                *controls,
                vd.sampler(),
            )
            self.assertGreater(pbr_texture.to_numpy()[..., :3].sum(), 0)
        finally:
            vd.init(arch=vd.cpu)

    def test_real_metal_executes_compute_generated_vertices(self) -> None:
        try:
            vd.init(arch=vd.metal)
        except RuntimeUnavailableError as error:
            self.skipTest(f"Metal runtime unavailable: {error}")
        try:
            controls = (vd.draw(vertex_count=3), vd.dynamic_state(viewport=(0, 0, 16, 16)))
            generated_texture = vd.Texture.zeros(shape=(16, 16))
            ComputeGeneratedVertices()(
                vd.render_pass(
                    vd.RenderTarget.from_attachments(colors={0: generated_texture}),
                    color=vd.clear((0.0, 0.0, 0.0, 1.0)),
                ),
                *controls,
            )
            self.assertGreater(generated_texture.to_numpy()[..., :3].sum(), 0)
        finally:
            vd.init(arch=vd.cpu)

    def test_real_opengl_managed_module_uses_static_pipeline_state(self) -> None:
        try:
            vd.init(arch=vd.opengl, api_version=(3, 3))
        except RuntimeUnavailableError as error:
            self.skipTest(f"OpenGL runtime unavailable: {error}")
        try:
            vertices = vd.storage.from_numpy(np.array(((-0.8, -0.8), (0.8, -0.8), (0.0, 0.8)), dtype=np.float32)).view(
                access="read"
            )
            visible = vd.Texture.zeros(shape=(16, 16))
            culled = vd.Texture.zeros(shape=(16, 16))
            controls = (vd.draw(vertex_count=3), vd.dynamic_state(viewport=(0, 0, 16, 16)))
            ManagedGraphics()(
                vertices,
                vd.render_pass(
                    vd.RenderTarget.from_attachments(colors={0: visible}),
                    color=vd.clear((0.0, 0.0, 0.0, 1.0)),
                ),
                *controls,
            )
            CulledManagedGraphics()(
                vertices,
                vd.render_pass(
                    vd.RenderTarget.from_attachments(colors={0: culled}),
                    color=vd.clear((0.0, 0.0, 0.0, 1.0)),
                ),
                *controls,
            )
            self.assertGreater(visible.to_numpy()[..., :3].sum(), 0)
            self.assertEqual(culled.to_numpy()[..., :3].sum(), 0)
        finally:
            vd.init(arch=vd.cpu)

    def test_real_opengl_managed_module_infers_optional_controls(self) -> None:
        try:
            vd.init(arch=vd.opengl, api_version=(3, 3))
        except RuntimeUnavailableError as error:
            self.skipTest(f"OpenGL runtime unavailable: {error}")
        try:
            vertices = vd.storage.from_numpy(np.array(((-0.8, -0.8), (0.8, -0.8), (0.0, 0.8)), dtype=np.float32)).view(
                access="read"
            )
            texture = vd.Texture.zeros(shape=(16, 16))
            DefaultedManagedGraphics()(
                vertices,
                vd.render_pass(
                    vd.RenderTarget.from_attachments(colors={0: texture}),
                    color=vd.clear((0.0, 0.0, 0.0, 1.0)),
                ),
            )
            self.assertGreater(texture.to_numpy()[..., :3].sum(), 0)
        finally:
            vd.init(arch=vd.cpu)

    def test_real_opengl_managed_module_uses_static_blend_state(self) -> None:
        try:
            vd.init(arch=vd.opengl, api_version=(3, 3))
        except RuntimeUnavailableError as error:
            self.skipTest(f"OpenGL runtime unavailable: {error}")
        try:
            vertices = vd.storage.from_numpy(np.array(((-0.8, -0.8), (0.8, -0.8), (0.0, 0.8)), dtype=np.float32)).view(
                access="read"
            )
            texture = vd.Texture.zeros(shape=(16, 16))
            BlendedManagedGraphics()(
                vertices,
                vd.render_pass(
                    vd.RenderTarget.from_attachments(colors={0: texture}),
                    color=vd.clear((0.0, 0.0, 1.0, 1.0)),
                ),
            )
            center = texture.to_numpy()[8, 8]
            self.assertGreater(int(center[0]), 100)
            self.assertGreater(int(center[2]), 100)
        finally:
            vd.init(arch=vd.cpu)

    def test_real_opengl_managed_controls_are_explicit_per_invocation(self) -> None:
        try:
            vd.init(arch=vd.opengl, api_version=(3, 3))
        except RuntimeUnavailableError as error:
            self.skipTest(f"OpenGL runtime unavailable: {error}")
        try:
            vertices = vd.storage.from_numpy(np.array(((-0.8, -0.8), (0.8, -0.8), (0.0, 0.8)), dtype=np.float32)).view(
                access="read"
            )
            texture = vd.Texture.zeros(shape=(16, 16))
            target = vd.RenderTarget.from_attachments(colors={0: texture})
            module = ManagedGraphics()
            render_pass = vd.render_pass(target, color=vd.clear((0.0, 0.0, 0.0, 1.0)))
            draw = vd.draw(vertex_count=3)
            dynamic_state = vd.dynamic_state(viewport=(0, 0, 16, 16))
            module(vertices, render_pass, draw, dynamic_state)
            texture.upload(np.zeros((16, 16, 4), dtype=np.uint8))
            module(vertices=vertices, render_pass=render_pass, draw=draw, dynamic_state=dynamic_state)
            self.assertGreater(texture.to_numpy()[..., :3].sum(), 0)

            invalid = vd.RenderTarget.from_attachments(
                colors={0: vd.Texture.zeros(shape=(8, 8)), 1: vd.Texture.zeros(shape=(8, 8))}
            )
            with self.assertRaisesRegex((ValueError, RuntimeError), "color|exactly match"):
                module(
                    vertices=vertices,
                    render_pass=vd.render_pass(
                        invalid,
                        colors={
                            0: vd.clear((0.0, 0.0, 0.0, 1.0)),
                            1: vd.clear((0.0, 0.0, 0.0, 1.0)),
                        },
                    ),
                    draw=draw,
                    dynamic_state=dynamic_state,
                )
            texture.upload(np.zeros((16, 16, 4), dtype=np.uint8))
            module(vertices=vertices, render_pass=render_pass, draw=draw, dynamic_state=dynamic_state)
            self.assertGreater(texture.to_numpy()[..., :3].sum(), 0)
        finally:
            vd.init(arch=vd.cpu)

    def test_real_metal_managed_module_executes_typed_controls(self) -> None:
        try:
            vd.init(arch=vd.metal)
        except RuntimeUnavailableError as error:
            self.skipTest(f"Metal runtime unavailable: {error}")
        try:
            vertices = vd.storage.from_numpy(np.array(((-0.8, -0.8), (0.8, -0.8), (0.0, 0.8)), dtype=np.float32)).view(
                access="read"
            )
            texture = vd.Texture.zeros(shape=(16, 16))
            ManagedGraphics()(
                vertices,
                vd.render_pass(
                    vd.RenderTarget.from_attachments(colors={0: texture}),
                    color=vd.clear((0.0, 0.0, 0.0, 1.0)),
                ),
                vd.draw(vertex_count=3),
                vd.dynamic_state(viewport=(0, 0, 16, 16)),
            )
            self.assertGreater(texture.to_numpy()[..., :3].sum(), 0)
        finally:
            vd.init(arch=vd.cpu)


if __name__ == "__main__":
    unittest.main()
