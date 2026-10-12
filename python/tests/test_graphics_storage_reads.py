from __future__ import annotations

import dataclasses
import json
import unittest
from importlib import import_module
from typing import Annotated, Any, Callable, Iterator, Mapping

import numpy as np
import vernon_dsl as vd
from backend_test_matrix import BackendRequirements, backend_row, probe_backend, require_available
from pipeline_shader import solid_fragment, triangle_vertex
from vernon_dsl.bundle import canonical_json, with_content_hash
from vernon_dsl.diagnostics import ProgramCompileError
from vernon_dsl.frontend.model import SemanticCategory
from vernon_dsl.frontend.module_ast import interpret_module_forward
from vernon_dsl.frontend.runtime_types import RuntimeParameterDescriptor, concrete_type_from_annotation
from vernon_dsl.operation_graph import ProgramControlDescriptor
from vernon_dsl.program import ProgramTemplate
from vernon_dsl.program_frontend import parse_program


@vd.vertex
def pull_vertex(
    positions: vd.TensorView[vd.f32, (vd.dyn, 2), vd.read],
    vertex_index: Annotated[vd.u32, vd.builtin("vertex_index")],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([positions[vertex_index, 0], positions[vertex_index, 1], 0.0, 1.0])


@vd.fragment
def table_fragment(
    colors: vd.TensorView[vd.f32, (vd.dyn, 4), vd.read],
) -> vd.Vector[vd.f32, 4]:
    return vd.Vector([colors[1, 0], colors[1, 1], colors[1, 2], colors[1, 3]])


pull_pipeline = vd.pipeline(pull_vertex, table_fragment, targets=vd.target_formats(colors={0: vd.rgba8_unorm}))


class PullProgram(vd.Module):
    def forward(
        self,
        positions: vd.TensorView[vd.f32, (vd.dyn, 2), vd.read],
        colors: vd.TensorView[vd.f32, (vd.dyn, 4), vd.read],
        render_pass: vd.RenderPass,
    ) -> None:
        pull_pipeline(positions=positions, colors=colors, render_pass=render_pass)


@vd.fragment
def textured_fragment(image: vd.Texture["2d", vd.f32]) -> vd.Vector[vd.f32, 4]:  # noqa: F722
    return vd.texture_sample(image, vd.Vector([0.5, 0.5]))


textured_pull_pipeline = vd.pipeline(
    pull_vertex, textured_fragment, targets=vd.target_formats(colors={0: vd.rgba8_unorm})
)


class TexturedPullProgram(vd.Module):
    def forward(
        self,
        positions: vd.TensorView[vd.f32, (vd.dyn, 2), vd.read],
        image: vd.Texture["2d", vd.f32],  # noqa: F722
        render_pass: vd.RenderPass,
    ) -> None:
        textured_pull_pipeline(positions=positions, image=image, render_pass=render_pass)


@vd.vertex
def scaled_vertex(
    scale: Annotated[vd.f32, vd.uniform(set=0, binding=0)],
    vertex_index: Annotated[vd.u32, vd.builtin("vertex_index")],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([scale, 0.0, 0.0, 1.0])


@vd.fragment
def tinted_fragment(
    tint: Annotated[vd.Vector[vd.f32, 4], vd.uniform(set=0, binding=0)],
) -> vd.Vector[vd.f32, 4]:
    return tint


colliding_pipeline = vd.pipeline(scaled_vertex, tinted_fragment, targets=vd.target_formats(colors={0: vd.rgba8_unorm}))


class CollidingProgram(vd.Module):
    def forward(self, scale: vd.f32, tint: vd.Vector[vd.f32, 4], render_pass: vd.RenderPass) -> None:
        colliding_pipeline(scale=scale, tint=tint, render_pass=render_pass)


def _cook(module: Any, parameter_types: dict[str, Any], target: str, version: int = 450) -> Mapping[str, Any]:
    from vernon_dsl._program_assets.compile_orchestration import _compile_program_variant, _native_target
    from vernon_dsl.bundle import make_target_options

    native = import_module("vernon_dsl._native")
    capture, outputs = interpret_module_forward(module, parameter_types)
    parsed = parse_program(ProgramTemplate.compile(capture).bind(capture, outputs))
    _, _, stages, _ = _compile_program_variant(
        parsed,
        program_id="tests/graphics-storage-reads",
        variant=(),
        target=make_target_options(target, {"version": version} if target == "opengl" else {}),
        compiler=native.Compiler(),
        native=native,
        native_target=_native_target(native, target),
    )
    (stage,) = {stage.id: stage for stage in stages.values()}.values()
    return stage.metadata


def _pull_parameter_types() -> dict[str, Any]:
    return {
        "positions": RuntimeParameterDescriptor.storage(vd.f32, (vd.dyn, 2), "read", True),
        "colors": RuntimeParameterDescriptor.storage(vd.f32, (vd.dyn, 4), "read", True),
        "render_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, None),
    }


def _required_features(node: Any) -> Iterator[list[str]]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key == "required_features":
                yield list(value)
            else:
                yield from _required_features(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _required_features(value)


def _descriptor_bindings(implementation: Mapping[str, Any]) -> dict[tuple[str, int], list[int]]:
    bindings: dict[tuple[str, Any], list[int]] = {}
    for endpoint in implementation["endpoints"]:
        leaves = endpoint.get("storage_leaves")
        if leaves:
            bindings[(endpoint["module"], endpoint["index"])] = [leaf["binding"] for leaf in leaves]
    for role, carrier in implementation.get("metadata_carriers", {}).items():
        bindings[(role, "metadata")] = [carrier["binding"]]
    return bindings


class GraphicsStorageReadCookTests(unittest.TestCase):
    def _assert_per_stage_storage(self, metadata: Mapping[str, Any]) -> None:
        (contract,) = metadata["program_contracts"].values()
        (implementation,) = metadata["program_implementations"].values()
        reflection = contract["reflection"]
        self.assertIn("tensor_views", reflection["required_features"])
        self.assertEqual(sorted(reflection["metadata_carriers"]), ["fragment", "vertex"])
        self.assertEqual(sorted(implementation["metadata_carriers"]), ["fragment", "vertex"])
        for carrier in implementation["metadata_carriers"].values():
            self.assertEqual(carrier["profile"], "portable_shader_metadata_i32")
            self.assertEqual(carrier["set"], 0)
        storage_endpoints = [endpoint for endpoint in reflection["endpoints"] if endpoint.get("role") == "storage"]
        self.assertEqual(sorted(endpoint["module"] for endpoint in storage_endpoints), ["fragment", "vertex"])
        for endpoint in storage_endpoints:
            self.assertEqual(endpoint["tag"], "resource")
            self.assertTrue(endpoint["layout"]["descriptor"])
            self.assertEqual(endpoint["access"], "read")
        bindings = _descriptor_bindings(implementation)
        self.assertEqual(
            bindings,
            {
                ("vertex", 0): [0],
                ("vertex", "metadata"): [1],
                ("fragment", 0): [2],
                ("fragment", "metadata"): [3],
            },
        )

    def test_vertex_and_fragment_views_cook_for_opengl(self) -> None:
        self._assert_per_stage_storage(_cook(PullProgram(), _pull_parameter_types(), "opengl"))

    def test_vertex_and_fragment_views_cook_for_vulkan(self) -> None:
        self._assert_per_stage_storage(_cook(PullProgram(), _pull_parameter_types(), "vulkan"))

    def test_stage_features_merge_into_a_canonical_set(self) -> None:
        texture = vd.Texture["2d", vd.f32]
        parameter_types = {
            "positions": RuntimeParameterDescriptor.storage(vd.f32, (vd.dyn, 2), "read", True),
            "image": RuntimeParameterDescriptor.annotated(
                SemanticCategory.RESOURCE, texture, concrete_type_from_annotation(texture)
            ),
            "render_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, None),
        }
        for target in ("opengl", "vulkan"):
            with self.subTest(target=target):
                metadata = _cook(TexturedPullProgram(), parameter_types, target)
                feature_sets = list(_required_features(metadata))
                self.assertTrue(feature_sets)
                for features in feature_sets:
                    self.assertEqual(features, sorted(set(features)))
                self.assertIn(["samplers", "tensor_views", "textures"], feature_sets)

    def test_opengl_below_430_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProgramCompileError, "PROGRAM_GRAPHICS_STORAGE_UNSUPPORTED.*GLSL 430"):
            _cook(PullProgram(), _pull_parameter_types(), "opengl", version=410)

    def test_write_access_is_rejected(self) -> None:
        source = """
from vernon_dsl import *
@fragment
def main(table: TensorView[f32, (dyn, 4), read_write]) -> Vector[f32, 4]:
    return Vector([table[0, 0], table[0, 1], table[0, 2], table[0, 3]])
"""
        result = _compile_entry_source(source, "graphics_write_view.py")
        self.assertFalse(result.ok)
        self.assertIn("graphics TensorView must have read access", str(result.diagnostics))


class GraphicsBindingPlanTests(unittest.TestCase):
    def test_cross_stage_explicit_binding_collision_is_rejected(self) -> None:
        parameter_types = {
            name: RuntimeParameterDescriptor.annotated(
                SemanticCategory.VALUE, annotation, concrete_type_from_annotation(annotation)
            )
            for name, annotation in (("scale", vd.f32), ("tint", vd.Vector[vd.f32, 4]))
        } | {
            "render_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, None),
        }
        with self.assertRaisesRegex(
            ProgramCompileError,
            r"graphics endpoints vertex 'scale' and fragment 'tint' both use descriptor \(set 0, binding 0\)",
        ):
            _cook(CollidingProgram(), parameter_types, "vulkan")

    def test_implicit_sampler_binding_skips_declared_resources(self) -> None:
        source = """
from vernon_dsl import *
@fragment
def main(
    image: Annotated[Texture["2d", f32], resource(set=0, binding=2)],
    occupied: Annotated[f32, uniform(set=0, binding=0)],
    uv: Vector[f32, 2],
) -> Vector[f32, 4]:
    return texture_sample(image, uv)
"""
        bindings = _reflected_bindings(_compile_entry_source(source, "sampler_binding_collision.py"))
        self.assertEqual(bindings["__vernon_implicit_sampler_image"], (0, 1))

    def test_link_shares_bindings_of_one_parameter_across_stages(self) -> None:
        vertex = vd.compile_source(_SHARED_VERTEX_SOURCE, "shared_vertex.py")
        fragment = vd.compile_source(_SHARED_FRAGMENT_SOURCE, "shared_fragment.py")
        vertex_layout, fragment_layout = _link([vertex, fragment])
        self.assertEqual(
            _layout_bindings(vertex_layout),
            {("table", "storage"): [0], ("", "metadata"): [1]},
        )
        self.assertEqual(
            _layout_bindings(fragment_layout),
            {
                ("table", "storage"): [0],
                ("image", "texture"): [2],
                ("__vernon_implicit_sampler_image", "sampler"): [3],
                ("", "metadata"): [4],
            },
        )
        bindings = _reflected_bindings(
            _compile_entry_source(_SHARED_FRAGMENT_SOURCE, "shared_fragment.py", fragment_layout)
        )
        self.assertEqual(bindings["table"], (0, 0))
        self.assertEqual(bindings["image"], (0, 2))
        self.assertEqual(bindings["__vernon_implicit_sampler_image"], (0, 3))

    def test_layout_of_another_module_is_rejected(self) -> None:
        vertex = vd.compile_source(_SHARED_VERTEX_SOURCE, "shared_vertex.py")
        fragment = vd.compile_source(_SHARED_FRAGMENT_SOURCE, "shared_fragment.py")
        vertex_layout, _ = _link([vertex, fragment])
        result = _compile_entry_source(_SHARED_FRAGMENT_SOURCE, "shared_fragment.py", vertex_layout)
        self.assertFalse(result.ok)
        self.assertIn("graphics binding layout does not match", str(result.diagnostics))

    def test_shared_parameter_cooks_one_descriptor(self) -> None:
        metadata = _cook(
            SharedTableProgram(),
            {
                "table": RuntimeParameterDescriptor.storage(vd.f32, (vd.dyn, 4), "read", True),
                "render_pass": ProgramControlDescriptor("render_pass", vd.RenderPass, None),
            },
            "vulkan",
        )
        (implementation,) = metadata["program_implementations"].values()
        self.assertEqual(
            _descriptor_bindings(implementation),
            {
                ("vertex", 0): [0],
                ("vertex", "metadata"): [1],
                ("fragment", 0): [0],
                ("fragment", "metadata"): [2],
            },
        )


_SHARED_VERTEX_SOURCE = """
from vernon_dsl import *
@vertex
def main(
    table: TensorView[f32, (dyn, 4), read],
    vertex_index: Annotated[u32, builtin("vertex_index")],
) -> Annotated[Vector[f32, 4], builtin("position")]:
    return Vector([table[vertex_index, 0], table[vertex_index, 1], 0.0, 1.0])
"""

_SHARED_FRAGMENT_SOURCE = """
from vernon_dsl import *
@fragment
def main(
    table: TensorView[f32, (dyn, 4), read],
    image: Texture["2d", f32],
    uv: Vector[f32, 2],
) -> Vector[f32, 4]:
    color = texture_sample(image, uv)
    return Vector([color.x * table[0, 0], color.y, color.z, color.w])
"""


@vd.vertex
def shared_vertex(
    table: vd.TensorView[vd.f32, (vd.dyn, 4), vd.read],
    vertex_index: Annotated[vd.u32, vd.builtin("vertex_index")],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([table[vertex_index, 0], table[vertex_index, 1], 0.0, 1.0])


@vd.fragment
def shared_fragment(
    table: vd.TensorView[vd.f32, (vd.dyn, 4), vd.read],
) -> vd.Vector[vd.f32, 4]:
    return vd.Vector([table[0, 0], table[0, 1], table[0, 2], table[0, 3]])


shared_pipeline = vd.pipeline(shared_vertex, shared_fragment, targets=vd.target_formats(colors={0: vd.rgba8_unorm}))


class SharedTableProgram(vd.Module):
    def forward(self, table: vd.TensorView[vd.f32, (vd.dyn, 4), vd.read], render_pass: vd.RenderPass) -> None:
        shared_pipeline(table=table, render_pass=render_pass)


@vd.vertex
def pull_triangle_vertex(
    positions: vd.TensorView[vd.f32, (vd.dyn, 2), vd.read],
    vertex_index: Annotated[vd.u32, vd.builtin("vertex_index")],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([positions[vertex_index, 0], positions[vertex_index, 1], 0.0, 1.0])


@vd.struct(shared=True)
class MaterialRow:
    color: vd.Vector[vd.f32, 4]
    gain: vd.f32
    layer: vd.i32


@vd.fragment
def material_fragment(
    rows: vd.TensorView[MaterialRow, (vd.dyn,), vd.read],
) -> vd.Vector[vd.f32, 4]:
    row = rows[2]
    return vd.Vector([row.color.x * row.gain, row.color.y * row.gain, row.color.z, vd.f32(row.layer) * 0.25])


@vd.fragment
def shared_columns_fragment(
    table: vd.TensorView[vd.f32, (vd.dyn, 4), vd.read],
) -> vd.Vector[vd.f32, 4]:
    return vd.Vector([table[0, 2], table[1, 2], table[2, 2], table[0, 3]])


_FULL_SCREEN = np.array(((-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)), dtype=np.float32)
_TRIANGLE = np.array(((-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)), dtype=np.float32)


def _render_pass(texture: vd.Texture) -> vd.RenderPass:
    return vd.render_pass(vd.RenderTarget.from_attachments(colors={0: texture}), color=vd.clear((0.0, 0.0, 0.0, 0.0)))


def _unorm8(color: tuple[float, ...]) -> np.ndarray:
    return np.round(np.asarray(color, dtype=np.float64) * 255.0)


class _GraphicsStorageRuntimeCases:
    arch: Any
    requirements: BackendRequirements
    assertEqual: Callable[..., None]
    assertRaisesRegex: Callable[..., Any]

    def setUp(self) -> None:
        require_available(probe_backend(backend_row(self.arch), self.requirements))

    def _assert_pixel(self, pixel: np.ndarray, color: tuple[float, ...]) -> None:
        np.testing.assert_allclose(pixel.astype(np.float64), _unorm8(color), atol=1.0)

    def test_vertex_pulling_matches_attribute_draw(self) -> None:
        pulled = vd.Texture.zeros(shape=(32, 32))
        attributed = vd.Texture.zeros(shape=(32, 32))
        vd.pipeline(pull_triangle_vertex, solid_fragment)(
            positions=vd.storage.from_numpy(_TRIANGLE),
            render_pass=_render_pass(pulled),
            draw=vd.draw(vertex_count=3),
        )
        vd.pipeline(triangle_vertex, solid_fragment)(
            position=vd.storage.from_numpy(_TRIANGLE), render_pass=_render_pass(attributed)
        )
        pixels = pulled.to_numpy()
        np.testing.assert_array_equal(pixels, attributed.to_numpy())
        self.assertEqual(tuple(pixels[0, 0]), (0, 0, 0, 0))
        self._assert_pixel(pixels[16, 16], (1.0, 0.25, 0.0, 1.0))

    def test_vertex_and_fragment_views_render(self) -> None:
        target = vd.Texture.zeros(shape=(8, 8))
        pull_pipeline(
            positions=vd.storage.from_numpy(_FULL_SCREEN),
            colors=vd.storage.from_numpy(np.array(((0, 0, 0, 0), (1.0, 0.5, 0.25, 1.0)), dtype=np.float32)),
            render_pass=_render_pass(target),
            draw=vd.draw(vertex_count=3),
        )
        self._assert_pixel(target.to_numpy()[4, 4], (1.0, 0.5, 0.25, 1.0))

    def test_fragment_reads_struct_table_row(self) -> None:
        def row(color: tuple[float, ...], gain: float, layer: int) -> MaterialRow:
            return MaterialRow(np.array(color, dtype=np.float32), vd.f32(gain), vd.i32(layer))

        rows = vd.storage.from_values(
            (
                row((0.0, 0.0, 0.0, 0.0), 0.0, 0),
                row((1.0, 1.0, 1.0, 1.0), 2.0, 1),
                row((0.5, 0.25, 0.75, 0.0), 1.5, 4),
                row((0.0, 1.0, 0.0, 1.0), 3.0, 2),
            ),
            dtype=MaterialRow,
        )
        target = vd.Texture.zeros(shape=(8, 8))
        vd.pipeline(triangle_vertex, material_fragment)(
            position=vd.storage.from_numpy(_FULL_SCREEN), rows=rows, render_pass=_render_pass(target)
        )
        self._assert_pixel(target.to_numpy()[4, 4], (0.75, 0.375, 0.75, 1.0))

    def test_one_value_read_by_both_stages(self) -> None:
        table = vd.storage.from_numpy(
            np.array(((-1.0, -1.0, 0.25, 1.0), (3.0, -1.0, 0.5, 0.0), (-1.0, 3.0, 0.75, 0.0)), dtype=np.float32)
        )
        target = vd.Texture.zeros(shape=(8, 8))
        vd.pipeline(shared_vertex, shared_columns_fragment)(
            table=table, render_pass=_render_pass(target), draw=vd.draw(vertex_count=3)
        )
        self._assert_pixel(target.to_numpy()[4, 4], (0.25, 0.5, 0.75, 1.0))

    def test_non_canonical_subviews_read_selected_records(self) -> None:
        position_owner = np.full((6, 2), 9.0, dtype=np.float32)
        position_owner[1::2] = _FULL_SCREEN
        color_owner = np.full((4, 4), 0.125, dtype=np.float32)
        color_owner[3] = (0.25, 1.0, 0.5, 1.0)
        positions = vd.storage.from_numpy(position_owner).view(shape=(3, 2), strides=(4, 1), offset=2, access="read")
        colors = vd.storage.from_numpy(color_owner).view(shape=(2, 4), strides=(8, 1), offset=4, access="read")
        target = vd.Texture.zeros(shape=(8, 8))
        pull_pipeline(
            positions=positions, colors=colors, render_pass=_render_pass(target), draw=vd.draw(vertex_count=3)
        )
        self._assert_pixel(target.to_numpy()[4, 4], (0.25, 1.0, 0.5, 1.0))

    def test_cooked_manifest_round_trips_runtime_parsing(self) -> None:
        from vernon_dsl._runtime.program_autodiff import _compile_program

        capture, outputs = interpret_module_forward(PullProgram(), _pull_parameter_types())
        deployment = _compile_program(parse_program(ProgramTemplate.compile(capture).bind(capture, outputs)))
        deployment.load()
        manifest = json.loads(deployment.manifest)

        def load_mutated(mutate: Callable[[dict[str, Any]], None]) -> None:
            mutated = json.loads(json.dumps(manifest))
            (artifact,) = mutated["variants"][0]["artifact_system"]["artifacts"].values()
            mutate(artifact["implementation"])
            payload = canonical_json(with_content_hash(mutated)).encode()
            dataclasses.replace(deployment, manifest=payload).load()

        with self.assertRaisesRegex(RuntimeError, "PROGRAM_REFLECTION_MISMATCH.*metadata_carriers"):
            load_mutated(lambda implementation: implementation.pop("metadata_carriers"))
        with self.assertRaisesRegex(
            RuntimeError, "PROGRAM_REFLECTION_MISMATCH.*must match the reflected module carriers"
        ):
            load_mutated(lambda implementation: implementation["metadata_carriers"].pop("vertex"))
        with self.assertRaisesRegex(RuntimeError, "PROGRAM_REFLECTION_MISMATCH.*no compiled native storage leaves"):
            load_mutated(lambda implementation: implementation["endpoints"][0].pop("storage_leaves"))

        def widen_leaf(implementation: dict[str, Any]) -> None:
            implementation["endpoints"][0]["storage_leaves"][0]["element_size"] += 4

        with self.assertRaisesRegex(RuntimeError, "PROGRAM_REFLECTION_MISMATCH.*storage leaves disagree"):
            load_mutated(widen_leaf)


class OpenGLGraphicsStorageRuntimeTests(_GraphicsStorageRuntimeCases, unittest.TestCase):
    arch = vd.opengl
    requirements = BackendRequirements(graphics=True, minimum_api_version=(4, 3))


class OpenGLESGraphicsStorageRuntimeTests(_GraphicsStorageRuntimeCases, unittest.TestCase):
    arch = vd.opengles
    requirements = BackendRequirements(graphics=True, minimum_api_version=(3, 1))


class VulkanGraphicsStorageRuntimeTests(_GraphicsStorageRuntimeCases, unittest.TestCase):
    arch = vd.vulkan
    requirements = BackendRequirements(graphics=True)


class DirectXGraphicsStorageRuntimeTests(_GraphicsStorageRuntimeCases, unittest.TestCase):
    arch = vd.directx
    requirements = BackendRequirements(graphics=True)


def _link(modules: list[str]) -> list[str]:
    native = import_module("vernon_dsl._native")
    linked = native.Compiler().link_graphics_bindings_result(modules)
    if not linked.ok:
        raise AssertionError(str(linked.diagnostics))
    return [json.dumps(layout) for layout in json.loads(linked.reflection)["layouts"]]


def _layout_bindings(layout: str) -> dict[tuple[str, str], list[int]]:
    return {(row["source_name"], row["kind"]): row["bindings"] for row in json.loads(layout)["assignments"]}


def _compile_entry_source(source: str, filename: str, layout: str = "") -> Any:
    native = import_module("vernon_dsl._native")
    return native.Compiler().compile_program_result(
        vd.compile_source(source, filename), native.Target.VULKAN, graphics_binding_layout=layout
    )


def _reflected_entry(result: Any) -> Mapping[str, Any]:
    if not result.ok:
        raise AssertionError(str(result.diagnostics))
    (entry,) = json.loads(result.reflection)["entries"]
    return entry


def _reflected_bindings(result: Any) -> dict[str, tuple[int, int]]:
    return {
        row["vernon.source_name"]: (row["vernon.set"], row["vernon.binding"])
        for row in _reflected_entry(result)["arguments"]
        if "vernon.binding" in row
    }


if __name__ == "__main__":
    unittest.main()
