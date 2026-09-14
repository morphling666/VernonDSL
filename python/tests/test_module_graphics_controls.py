from __future__ import annotations

import gc
import sys
import unittest
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, cast
from unittest.mock import patch

import numpy as np
import vernon_dsl as vd
from backend_test_matrix import BackendRequirements, BackendRow, backend_matrix_test, expand_backend_matrix_tests
from vernon_dsl.frontend.module_ast import interpret_module_forward
from vernon_dsl.frontend.runtime_types import RuntimeParameterDescriptor, runtime_parameter_descriptor
from vernon_dsl.operation_graph import GraphicsCallOp, ProgramControlDescriptor
from vernon_dsl.program import ProgramTemplate
from vernon_dsl.program_assets import ProgramCompileError, cook_program_asset
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


@vd.kernel(workgroup_size=(1, 1, 1))
def touch_color_attachment(
    image: vd.Texture["2d", vd.rgba8_unorm, vd.read_write],  # noqa: F722
    marker: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    coordinate = vd.Vector([vd.i32(0), vd.i32(0)])
    value = vd.texture_load(image, coordinate)
    marker[0] = value.x
    vd.texture_store(image, coordinate, value)


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
    ) -> Any:
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
        return vd.color_output(render_pass)


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


@dataclass
class MixedGraphicsOutputs:
    color: Any
    marker: vd.TensorStorage


class GraphicsComputeGraphics(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> MixedGraphicsOutputs:
        managed_pipeline(
            vertices=vertices,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )
        color = vd.color_output(render_pass)
        marker = vd.zeros(dtype=vd.f32, shape=(1,))
        touch_color_attachment(color, marker, grid=(1, 1, 1))
        managed_pipeline(
            vertices=vertices,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )
        return MixedGraphicsOutputs(vd.color_output(render_pass), marker)


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


@expand_backend_matrix_tests
class ModuleGraphicsControlTests(unittest.TestCase):
    @staticmethod
    def _cook(parsed: Any, pipeline_id: str) -> dict[str, Any]:
        from vernon_dsl._program_assets.compile_orchestration import _compile_program_variant, _native_target
        from vernon_dsl.bundle import make_target_options

        native = import_module("vernon_dsl._native")
        _, _, _, program = _compile_program_variant(
            parsed,
            program_id=pipeline_id,
            variant=(),
            target=make_target_options("opengl", {"version": 410}),
            compiler=native.Compiler(),
            native=native,
            native_target=_native_target(native, "opengl"),
        )
        return dict(program)

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

    def test_graphics_module_vjp_cook_is_rejected_before_deployment(self) -> None:
        fixture = Path(__file__).parents[2] / "source/tests/fixtures/module_graphics_program_asset.py"
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProgramCompileError, "PROGRAM_GRAPHICS_VJP_UNSUPPORTED"):
                cook_program_asset(
                    program_asset=f"{fixture}:graphics_vjp_asset",
                    output=Path(directory) / "bundle",
                    target="metal",
                )
            self.assertFalse((Path(directory) / "bundle").exists())

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
        self.assertEqual(invocation.graph.outputs["output"], second.outputs["render_pass.color.0"])
        parsed = parse_program(invocation)
        signature = next(line for line in parsed.mlir.splitlines() if "func.func @forward" in line)
        self.assertIn(f"%v{first.inputs['render_pass.color.0']}:", signature)
        self.assertNotIn(f"%v{second.inputs['render_pass.color.0']}:", signature)
        canonical = self._cook(parsed, "tests/reused-render-pass")
        accesses = [
            access
            for node in canonical["graphs"][0]["nodes"]
            for access in node["accesses"]
            if access["tag"] == "attachment"
        ]
        self.assertEqual(len(accesses), 2)
        self.assertEqual(accesses[0]["after"], accesses[1]["before"])

    def test_attachment_versions_flow_through_compute_and_back_to_graphics(self) -> None:
        render_pass = vd.RenderPass(cast(vd.RenderTarget, _Target()), ((0, vd.preserve()),))
        parameter_types = {
            "vertices": RuntimeParameterDescriptor.storage(vd.f32, (3, 2), "read", True),
            "render_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, render_pass),
            "draw": ProgramControlDescriptor("draw", vd.DrawCommand, vd.draw(vertex_count=3)),
            "dynamic_state": ProgramControlDescriptor("dynamic_state", vd.DynamicState, vd.dynamic_state()),
        }
        capture, outputs = interpret_module_forward(GraphicsComputeGraphics(), parameter_types)
        invocation = ProgramTemplate.compile(capture).bind(capture, outputs)
        first_draw, compute, second_draw = invocation.graph.operations
        attachment = "render_pass.color.0"

        self.assertEqual(first_draw.outputs[attachment], compute.inputs["image"])
        self.assertEqual(compute.outputs["image"], second_draw.inputs[attachment])
        self.assertEqual(invocation.graph.outputs["color"], second_draw.outputs[attachment])

        parsed = parse_program(invocation)
        signature = next(line for line in parsed.mlir.splitlines() if "func.func @forward" in line)
        self.assertIn(f"%v{first_draw.inputs[attachment]}:", signature)
        self.assertNotIn(f"%v{first_draw.outputs[attachment]}:", signature)
        self.assertNotIn(f"%v{compute.outputs['image']}:", signature)
        canonical = self._cook(parsed, "tests/graphics-compute-graphics")
        nodes = canonical["graphs"][0]["nodes"]
        first_attachment = next(access for access in nodes[0]["accesses"] if access["tag"] == "attachment")
        compute_write = next(
            access
            for access in nodes[1]["accesses"]
            if access["tag"] == "write" and access["before"] == first_attachment["after"]
        )
        second_attachment = next(access for access in nodes[2]["accesses"] if access["tag"] == "attachment")
        self.assertEqual(first_attachment["after"], compute_write["before"])
        self.assertEqual(compute_write["after"], second_attachment["before"])

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
        from vernon_dsl._program_assets.compile_orchestration import _compile_program_variant, _native_target
        from vernon_dsl.bundle import build_program_plan, make_target_options
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
        reflected_target, key, stages, program = _compile_program_variant(
            parsed,
            program_id="tests/managed-graphics",
            variant=(),
            target=target,
            compiler=native.Compiler(),
            native=native,
            native_target=_native_target(native, "opengl"),
        )
        plan = build_program_plan("tests/managed-graphics", reflected_target, ((key, stages, program),))

        self.assertEqual(len(plan.compiled_stages), 1)
        self.assertEqual(plan.compiled_stages[0].stage, "graphics")
        self.assertEqual(
            tuple(module["role"] for module in plan.compiled_stages[0].metadata["graphics_modules"]),
            ("vertex", "fragment"),
        )
        canonical_program = plan.variants[0].program
        operation = canonical_program["graphs"][0]["nodes"][0]["operation"]
        self.assertEqual(operation["render_pass"]["control"], 1)
        self.assertEqual(operation["draw"]["control"], 2)
        self.assertEqual(operation["dynamic_state"]["control"], 3)

    @staticmethod
    def _triangle_vertices() -> Any:
        return vd.storage.from_numpy(np.array(((-0.8, -0.8), (0.8, -0.8), (0.0, 0.8)), dtype=np.float32)).view(
            access="read"
        )

    @backend_matrix_test(BackendRequirements(gpu=True, graphics=True))
    def test_managed_module_uses_distinct_node_controls(self, backend: BackendRow) -> None:
        vertices = self._triangle_vertices()
        textures = (vd.Texture.zeros(shape=(16, 16)), vd.Texture.zeros(shape=(16, 16)))
        module = ManagedGraphicsTwice()
        controls = tuple(
            control
            for texture in textures
            for control in (
                vd.render_pass(
                    vd.RenderTarget.from_attachments(colors={0: texture}),
                    color=vd.clear((0.0, 0.0, 0.0, 1.0)),
                ),
                vd.draw(vertex_count=3),
                vd.dynamic_state(viewport=(0, 0, 16, 16)),
            )
        )
        module(vertices, *controls)
        invocation_controls = next(
            iter(module._program_cache._partition(vd.current_session()).snapshot.values())
        ).invocation.graphics_controls
        self.assertEqual(len(invocation_controls), 2)
        self.assertNotEqual(invocation_controls[0]["render_pass"][0], invocation_controls[1]["render_pass"][0])
        for texture in textures:
            self.assertGreater(texture.to_numpy()[..., :3].sum(), 0)

    @backend_matrix_test(BackendRequirements(gpu=True, graphics=True, texture_sampler_operations=True))
    def test_shadow_depth_is_sampled_by_pbr(self, backend: BackendRow) -> None:
        depth = vd.Texture.device(shape=(16, 16), format=vd.d32_float, usage=("depth_stencil_attachment", "sampled"))
        pbr_texture = vd.Texture.zeros(shape=(16, 16))
        controls = (vd.draw(vertex_count=3), vd.dynamic_state(viewport=(0, 0, 16, 16)))
        ShadowPbrProgram()(
            self._triangle_vertices(),
            vd.render_pass(vd.RenderTarget.from_attachments(colors={}, depth=depth), depth=vd.clear_depth(1.0)),
            *controls,
            vd.render_pass(
                vd.RenderTarget.from_attachments(colors={0: pbr_texture}),
                color=vd.clear((0.0, 0.0, 0.0, 1.0)),
            ),
            *controls,
            vd.sampler(),
        )
        self.assertGreater(pbr_texture.to_numpy()[..., :3].sum(), 0)

    @backend_matrix_test(BackendRequirements(gpu=True, compute=True, graphics=True, storage_buffers=True))
    def test_compute_generated_vertices(self, backend: BackendRow) -> None:
        texture = vd.Texture.zeros(shape=(16, 16))
        ComputeGeneratedVertices()(
            vd.render_pass(
                vd.RenderTarget.from_attachments(colors={0: texture}),
                color=vd.clear((0.0, 0.0, 0.0, 1.0)),
            ),
            vd.draw(vertex_count=3),
            vd.dynamic_state(viewport=(0, 0, 16, 16)),
        )
        self.assertGreater(texture.to_numpy()[..., :3].sum(), 0)

    @backend_matrix_test(
        BackendRequirements(gpu=True, compute=True, graphics=True, storage_buffers=True, storage_texture=True)
    )
    def test_attachment_flows_graphics_compute_graphics(self, backend: BackendRow) -> None:
        texture = vd.Texture.zeros(
            shape=(16, 16),
            usage=("color_attachment", "storage", "transfer_source", "transfer_destination"),
        )
        outputs = GraphicsComputeGraphics()(
            self._triangle_vertices(),
            vd.render_pass(
                vd.RenderTarget.from_attachments(colors={0: texture}),
                color=vd.clear((0.0, 0.0, 0.0, 1.0)),
            ),
            vd.draw(vertex_count=3),
            vd.dynamic_state(viewport=(0, 0, 16, 16)),
        )
        self.assertGreater(texture.to_numpy()[..., :3].sum(), 0)
        self.assertEqual(outputs.marker.to_numpy().shape, (1,))

    @backend_matrix_test(BackendRequirements(gpu=True, graphics=True))
    def test_managed_module_uses_static_pipeline_state(self, backend: BackendRow) -> None:
        vertices = self._triangle_vertices()
        visible, culled = vd.Texture.zeros(shape=(16, 16)), vd.Texture.zeros(shape=(16, 16))
        controls = (vd.draw(vertex_count=3), vd.dynamic_state(viewport=(0, 0, 16, 16)))
        for module, texture in ((ManagedGraphics(), visible), (CulledManagedGraphics(), culled)):
            module(
                vertices,
                vd.render_pass(
                    vd.RenderTarget.from_attachments(colors={0: texture}),
                    color=vd.clear((0.0, 0.0, 0.0, 1.0)),
                ),
                *controls,
            )
        self.assertGreater(visible.to_numpy()[..., :3].sum(), 0)
        self.assertEqual(culled.to_numpy()[..., :3].sum(), 0)

    @backend_matrix_test(BackendRequirements(gpu=True, graphics=True))
    def test_managed_module_infers_optional_controls(self, backend: BackendRow) -> None:
        texture = vd.Texture.zeros(shape=(16, 16))
        DefaultedManagedGraphics()(
            self._triangle_vertices(),
            vd.render_pass(
                vd.RenderTarget.from_attachments(colors={0: texture}),
                color=vd.clear((0.0, 0.0, 0.0, 1.0)),
            ),
        )
        self.assertGreater(texture.to_numpy()[..., :3].sum(), 0)

    @backend_matrix_test(BackendRequirements(gpu=True, graphics=True))
    def test_managed_module_uses_static_blend_state(self, backend: BackendRow) -> None:
        texture = vd.Texture.zeros(shape=(16, 16))
        BlendedManagedGraphics()(
            self._triangle_vertices(),
            vd.render_pass(
                vd.RenderTarget.from_attachments(colors={0: texture}),
                color=vd.clear((0.0, 0.0, 1.0, 1.0)),
            ),
        )
        center = texture.to_numpy()[8, 8]
        self.assertGreater(int(center[0]), 100)
        self.assertGreater(int(center[2]), 100)

    @backend_matrix_test(BackendRequirements(gpu=True, graphics=True))
    def test_managed_controls_are_explicit_per_invocation(self, backend: BackendRow) -> None:
        vertices = self._triangle_vertices()
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

    @backend_matrix_test(BackendRequirements(gpu=True, graphics=True))
    def test_managed_module_executes_typed_controls(self, backend: BackendRow) -> None:
        texture = vd.Texture.zeros(shape=(16, 16))
        ManagedGraphics()(
            self._triangle_vertices(),
            vd.render_pass(
                vd.RenderTarget.from_attachments(colors={0: texture}),
                color=vd.clear((0.0, 0.0, 0.0, 1.0)),
            ),
            vd.draw(vertex_count=3),
            vd.dynamic_state(viewport=(0, 0, 16, 16)),
        )
        self.assertGreater(texture.to_numpy()[..., :3].sum(), 0)

    @backend_matrix_test(BackendRequirements(gpu=True, graphics=True))
    def test_program_instance_retains_native_attachment_owner(self, backend: BackendRow) -> None:
        texture = vd.Texture.zeros(shape=(16, 16))
        render_pass = vd.render_pass(
            vd.RenderTarget.from_attachments(colors={0: texture}),
            color=vd.clear((0.0, 0.0, 0.0, 1.0)),
        )
        attachment_view = render_pass.target._color_attachments()[0][1]
        view_type = type(attachment_view)
        resident_view = view_type._resident_view
        captured: list[Any] = []

        def capture_view(view: Any, context: Any) -> Any:
            handle = resident_view(view, context)
            if not captured:
                captured.append(handle)
            return handle

        module = ManagedGraphics()
        with patch.object(view_type, "_resident_view", capture_view):
            module(
                self._triangle_vertices(),
                render_pass,
                vd.draw(vertex_count=3),
                vd.dynamic_state(viewport=(0, 0, 16, 16)),
            )
        native_view = captured[0]
        retained_references = sys.getrefcount(native_view)
        del module
        gc.collect()
        self.assertEqual(sys.getrefcount(native_view), retained_references - 1)

    @backend_matrix_test(BackendRequirements(gpu=True, graphics=True))
    def test_program_reuses_across_attachment_extents_and_dynamic_states(self, backend: BackendRow) -> None:
        module = ManagedGraphics()
        executable_identity = None
        manifest_snapshot = None
        for size, stencil in ((8, 3), (16, 9)):
            texture = vd.Texture.zeros(shape=(size, size))
            module(
                self._triangle_vertices(),
                vd.render_pass(
                    vd.RenderTarget.from_attachments(colors={0: texture}),
                    color=vd.clear((0.0, 0.0, 0.0, 1.0)),
                ),
                vd.draw(vertex_count=3),
                vd.dynamic_state(viewport=(0, 0, size, size), stencil_reference=stencil),
            )
            self.assertGreater(texture.to_numpy()[..., :3].sum(), 0)
            session_cache = module._program_cache._partition(vd.current_session()).snapshot
            self.assertEqual(len(session_cache), 1)
            specialization = next(iter(session_cache.values()))
            if executable_identity is None:
                executable_identity = id(specialization.native_program)
                manifest_snapshot = repr(specialization.invocation.graph)
            else:
                self.assertEqual(id(specialization.native_program), executable_identity)
                self.assertEqual(repr(specialization.invocation.graph), manifest_snapshot)


declared_pipeline = vd.pipeline(
    managed_vertex,
    managed_fragment,
    targets=vd.target_formats(colors={0: vd.rgba16_float}),
)


class DeclaredTargetGraphics(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        render_pass: vd.RenderPass,
    ) -> None:
        declared_pipeline(vertices=vertices, render_pass=render_pass)


class UndeclaredTargetGraphics(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        render_pass: vd.RenderPass,
    ) -> None:
        managed_pipeline(vertices=vertices, render_pass=render_pass)


class DeclaredTargetFormatTests(unittest.TestCase):
    """A graphics Program can be captured from declared target formats, with no concrete RenderPass.

    This is what makes graphics assets cookable: attachment formats are pipeline state, so they have to be known
    before any render target exists. The extent stays out of it and is resolved per invocation.
    """

    @staticmethod
    def _capture(module: Any, render_pass: Any = None) -> Any:
        parameter_types: dict[str, Any] = {
            "vertices": RuntimeParameterDescriptor.storage(vd.f32, (3, 2), "read", True),
            "render_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, render_pass),
        }
        capture, outputs = interpret_module_forward(module, parameter_types)
        return ProgramTemplate.compile(capture).bind(capture, outputs)

    def test_declared_formats_capture_without_a_render_pass(self) -> None:
        invocation = self._capture(DeclaredTargetGraphics())
        operation = invocation.graph.operations[0]
        self.assertIsInstance(operation, GraphicsCallOp)
        self.assertEqual(len(operation.attachment_names), 1)
        self.assertEqual(operation.color_count, 1)

    def test_declared_format_reaches_the_cooked_attachment(self) -> None:
        from vernon_dsl.program import _parse_module_program

        canonical = ModuleGraphicsControlTests._cook(
            _parse_module_program(DeclaredTargetGraphics()), "declared/graphics"
        )
        operation = canonical["graphs"][0]["nodes"][0]["operation"]
        self.assertEqual(operation["render_pass"]["colors"][0]["formats"], ["rgba16_float"])
        images = [
            storage["descriptor"] for storage in canonical["storages"] if storage["descriptor"].get("tag") == "image"
        ]
        self.assertEqual(images[0]["format"], "rgba16_float")
        self.assertNotIn("extent", images[0], "borrowed attachment extent comes from each invocation")

    def test_capture_without_declared_formats_or_a_render_pass_is_refused(self) -> None:
        with self.assertRaisesRegex(TypeError, "requires vd.pipeline\\(\\.\\.\\., targets="):
            self._capture(UndeclaredTargetGraphics())

    def test_bound_render_pass_must_match_the_declared_formats(self) -> None:
        color = vd.Texture.zeros(shape=(16, 16))
        render_pass = vd.RenderPass(vd.RenderTarget.from_attachments(colors={0: color}), ((0, vd.preserve()),))
        with self.assertRaisesRegex(TypeError, "do not match the formats this pipeline was built for"):
            self._capture(DeclaredTargetGraphics(), render_pass)


if __name__ == "__main__":
    unittest.main()
