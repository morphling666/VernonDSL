from __future__ import annotations

import base64
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import vernon_dsl as vd
from vernon_dsl._program_assets.artifact_io import artifact_extension
from vernon_dsl._program_assets.capture import CapturedProgram, capture_program
from vernon_dsl._versions import COMPILER_CONTRACT_VERSION, PROGRAM_VERSION
from vernon_dsl.bundle import (
    CompiledArtifact,
    CompiledStage,
    build_program_manifest,
    build_program_plan,
    canonical_json,
    make_target_options,
    with_content_hash,
)
from vernon_dsl.module_graph import load_project, resolve_project_entry
from vernon_dsl.program_assets import (
    ProgramCompileError,
    cook_program_asset,
    encode_runtime_stage,
    lint_python_program_asset,
)


def _fake_native(compile_program_result: object) -> Any:
    from vernon_dsl._program_assets.compile_orchestration import _native_module

    native = _native_module()

    class FakeCompiler:
        def __init__(self) -> None:
            self._real = native.Compiler()

        def compile_program_result(self, *args: object, **kwargs: object) -> object:
            return cast(Any, compile_program_result)(*args, **kwargs)

        def __getattr__(self, name: str) -> object:
            return getattr(self._real, name)

    class Native:
        Compiler = FakeCompiler
        Target = SimpleNamespace(
            CPU="cpu",
            CUDA="cuda",
            VULKAN="vulkan",
            METAL="metal",
            DIRECTX="directx",
            OPENGL="opengl",
            OPENGL_ES="opengles",
        )

        def __getattr__(self, name: str) -> object:
            return getattr(native, name)

    return Native()


def _native_available() -> bool:
    try:
        from vernon_dsl._program_assets.compile_orchestration import _native_module

        _native_module()
        return True
    except ProgramCompileError:
        return False


def _deployed_modules(document: Any) -> list[dict[str, Any]]:
    """Describe every module in a canonical Program deployment."""

    variant = document["variants"][0]
    system = variant["artifact_system"]
    blobs = document["blobs"]
    return [
        {
            "role": module["role"],
            "entry": module["entry_point"],
            "format": module["format"],
            "path": blobs[module["blob"]]["location"]["uri"],
            "sha256": module["blob"],
            "implementation": artifact.get("implementation", {}),
        }
        for artifact in system["artifacts"].values()
        for module in artifact["modules"]
    ]


def _assert_exact_boundary_slot(
    test: unittest.TestCase,
    program: dict[str, Any],
    slot: dict[str, Any],
) -> None:
    value = program["values"][slot["value"]]
    input_direction = slot["role"] in {"input", "cotangent"}
    expected: dict[str, Any] = {
        "id": slot["id"],
        "path": slot["path"],
        "value": slot["value"],
        "role": slot["role"],
        "direction": "input" if input_direction else "output",
        "category": "value",
        "access": "read" if input_direction else "write",
        "logical_type": value["type"],
        "outer_shape": value.get("shape", []),
        "alias_owner": f"value:{value['id']}",
    }
    if "value_layout" in value:
        expected["value_layout"] = value["value_layout"]
    if "storage" in value:
        storage_id = value["storage"]
        storage = program["storages"][storage_id]
        tag = storage["descriptor"]["tag"]
        expected["category"] = {
            "buffer": "storage_view",
            "image": "texture",
            "opaque": "sampler",
        }[tag]
        expected["alias_owner"] = f"storage:{storage_id}"
        expected["storage_id"] = storage_id
        expected["storage_descriptor"] = storage["descriptor"]
        if input_direction:
            reads = False
            writes = False
            for graph in program["graphs"]:
                if graph["direction"] != "forward":
                    continue
                for node in graph["nodes"]:
                    for access in node["accesses"]:
                        if access["storage"] != storage_id:
                            continue
                        reads |= access["tag"] in {"read", "attachment"} or (
                            access["tag"] == "write" and access["access"] == "read_write"
                        )
                        writes |= access["tag"] in {"initialize", "write", "attachment"}
            expected["access"] = "read_write" if reads and writes else "write" if writes else "read"
    if not input_direction:
        expected["publication"] = "commit_after_success"
    test.assertEqual(slot, expected)


class ShaderAssetManifestTests(unittest.TestCase):
    def test_program_bundle_composes_logical_stages_with_kernel_artifacts(self) -> None:
        target = make_target_options("vulkan")
        stage = CompiledStage(
            "python/square",
            "square",
            "compute",
            target,
            {},
            {
                "arguments": [
                    {
                        "index": 0,
                        "kind": "tensor",
                        "type": '!vernon.tensor_view<f32, [4], "read", "device">',
                        "vernon.source_name": "source",
                        "vernon.interface": "input",
                        "vernon.set": 0,
                        "vernon.binding": 0,
                        "access": "read",
                        "address_space": "device",
                        "source_shape": [4],
                        "element_layout": {
                            "logical_type": "f32",
                            "layout_hash": "f32-layout",
                            "byte_size": 4,
                            "alignment": 4,
                            "leaves": [],
                        },
                    },
                    {
                        "index": 1,
                        "kind": "tensor",
                        "type": '!vernon.tensor_view<f32, [4], "write", "device">',
                        "vernon.source_name": "output",
                        "vernon.interface": "output",
                        "vernon.set": 0,
                        "vernon.binding": 1,
                        "access": "write",
                        "address_space": "device",
                        "source_shape": [4],
                        "element_layout": {
                            "logical_type": "f32",
                            "layout_hash": "f32-layout",
                            "byte_size": 4,
                            "alignment": 4,
                            "leaves": [],
                        },
                    },
                ]
            },
            CompiledArtifact("spirv", b"\x03\x02\x23\x07"),
        )
        plan = build_program_plan(
            "modules/square",
            target,
            (
                (
                    (),
                    {"Module.square": stage},
                    {
                        "stages": {"Module.square": {"operation": "compute", "contract_hash": "0" * 64}},
                        "parameters": [],
                        "storages": [],
                        "values": [],
                        "graphs": [
                            {
                                "direction": "forward",
                                "nodes": [
                                    {
                                        "operation": {"tag": "compute", "workgroups": [4, 1, 1]},
                                    }
                                ],
                            }
                        ],
                        "abi": {},
                    },
                ),
            ),
        )

        self.assertEqual(dict(plan.variants[0].stage_implementations), {"Module.square": stage.id})
        canonical = plan.variants[0].program
        self.assertEqual(set(canonical["stages"]), {"Module.square"})
        self.assertNotIn("dependencies", canonical["graphs"][0]["nodes"][0])
        self.assertEqual(canonical["graphs"][0]["nodes"][0]["operation"]["workgroups"], [4, 1, 1])

    def test_pipeline_asset_promotes_an_imported_entry_without_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "shaders"
            package.mkdir()
            (package / "fullscreen.py").write_text(
                """
import vernon_dsl as vd

@vd.vertex
def vertex(value: vd.f32) -> vd.f32:
    return value
""",
                encoding="utf-8",
            )
            source = package / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd
from .fullscreen import vertex as fullscreen_vertex

@vd.fragment
def fragment_main() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 0.0, 0.0, 1.0])

asset = vd.program_asset(
    id="pipelines/imported",
    program=vd.pipeline(fullscreen_vertex, fragment_main),
)
""",
                encoding="utf-8",
            )

            lint = lint_python_program_asset(source, "asset")
            self.assertIn("fullscreen_vertex", lint.entries)
            resolution = resolve_project_entry(source, "fullscreen_vertex")
            self.assertIsNotNone(resolution)
            assert resolution is not None
            self.assertIn("vertex", resolution.decorators)
            self.assertEqual(resolution.name, "fullscreen_vertex")
            self.assertEqual(resolution.source_path, (package / "fullscreen.py").resolve())
            promoted = load_project(source, entry="fullscreen_vertex")
            self.assertIn("@vd.vertex", promoted.source)
            self.assertIn("def fullscreen_vertex(", promoted.source)
            self.assertIn("fullscreen.py", dict(promoted.dependencies))

    def test_pipeline_asset_rejects_unsupported_import_entry_forms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fullscreen.py").write_text(
                """
import vernon_dsl as vd

@vd.vertex
def fullscreen_vertex(value: vd.f32) -> vd.f32:
    return value
""",
                encoding="utf-8",
            )
            qualified = root / "qualified.py"
            qualified.write_text(
                """
import vernon_dsl as vd
import fullscreen

@vd.fragment
def fragment_main() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 0.0, 0.0, 1.0])

asset = vd.program_asset(
    id="pipelines/qualified",
    program=vd.pipeline(fullscreen.fullscreen_vertex, fragment_main),
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "simple imported or local function names"):
                lint_python_program_asset(qualified, "asset")

            (root / "exports.py").write_text(
                "from fullscreen import fullscreen_vertex\n",
                encoding="utf-8",
            )
            reexported = root / "reexported.py"
            reexported.write_text(
                """
import vernon_dsl as vd
from exports import fullscreen_vertex

@vd.fragment
def fragment_main() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 0.0, 0.0, 1.0])

asset = vd.program_asset(
    id="pipelines/reexported",
    program=vd.pipeline(fullscreen_vertex, fragment_main),
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "has no symbol 'fullscreen_vertex'"):
                lint_python_program_asset(reexported, "asset")

    def test_the_lint_reads_a_declaration_without_executing_it(self) -> None:
        """Cooking evaluates the source; the lint is the part that still promises not to."""

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd
raise RuntimeError("the lint must not execute this module")
FEATURE = vd.feature("FEATURE")

@vd.vertex
def vertex_main(value: vd.f32) -> vd.f32:
    return value

@vd.fragment
def fragment_main(value: vd.f32) -> vd.f32:
    return value

asset = vd.program_asset(
    id="pipelines/static",
    program=vd.pipeline(vertex_main, fragment_main),
    variants=((), (FEATURE,)),
)
""",
                encoding="utf-8",
            )
            lint = lint_python_program_asset(source, "asset")
            self.assertEqual(lint.id, "pipelines/static")
            self.assertEqual(lint.variants, ((), ("FEATURE",)))
            self.assertEqual(set(lint.entries), {"vertex_main", "fragment_main"})

    def test_python_pipeline_asset_rejects_legacy_stage_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def compute_main() -> None:
    pass

asset = vd.program_asset(
    id="pipelines/legacy",
    compute=compute_main,
    variants=((),),
    targets={"cpu": {}},
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "unknown program_asset argument"):
                lint_python_program_asset(source, "asset")

    def test_python_pipeline_asset_rejects_graphics_stage_order(self) -> None:
        """Topology is checked by program_asset itself, so it is checked here where it is enforced."""

        from vernon_dsl._program_assets.source import load_program_asset_declaration

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.vertex
def vertex_main() -> None:
    pass

@vd.fragment
def fragment_main() -> None:
    pass

asset = vd.program_asset(
    id="pipelines/reversed",
    program=vd.pipeline(fragment_main, vertex_main),
    variants=((),),
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "topology order"):
                load_program_asset_declaration(source, "asset")

    def test_python_pipeline_asset_rejects_noncanonical_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd
ZED = vd.feature("ZED")
ALPHA = vd.feature("ALPHA")

@vd.kernel
def compute_main() -> None:
    pass

asset = vd.program_asset(
    id="pipelines/bad",
    program=compute_main,
    variants=((ZED, ALPHA),),
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "not canonical"):
                lint_python_program_asset(source, "asset")

    def test_python_pipeline_asset_enforces_variant_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "asset.py"
            features = "\n".join(f'F{index} = vd.feature("F{index:02}")' for index in range(17))
            variants = ", ".join(f"(F{index}, )" for index in range(17))
            source.write_text(
                f"""
import vernon_dsl as vd
{features}

@vd.kernel
def compute_main() -> None:
    pass

asset = vd.program_asset(
    id="pipelines/too_many",
    program=compute_main,
    variants=({variants},),
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "variant cap 16"):
                lint_python_program_asset(source, "asset")

    def test_example_pipeline_asset_has_canonical_variants(self) -> None:
        root = Path(__file__).parents[2]
        pipeline = lint_python_program_asset(root / "examples" / "variant_mesh.py", "mesh_asset")
        self.assertEqual(pipeline.id, "shaders/variant_mesh")
        self.assertEqual(
            pipeline.variants,
            ((), ("INSTANCE",), ("SKIN",), ("INSTANCE", "SKIN")),
        )


class CanonicalProgramCaptureTests(unittest.TestCase):
    _SOURCE = """
import vernon_dsl as vd

@vd.kernel(workgroup_size=(1, 1, 1))
def square(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0]

@vd.vertex
def vertex_main() -> None:
    pass

@vd.fragment
def fragment_main() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 1.0, 1.0, 1.0])

class Square(vd.Module):
    def forward(
        self,
        source: vd.TensorView[vd.f32, (1,), vd.read],
    ) -> vd.TensorStorage:
        output = vd.empty_like(source)
        square(source, output)
        return output

kernel_asset = vd.program_asset(id="capture/kernel", program=square)
pipeline_asset = vd.program_asset(
    id="capture/pipeline",
    program=vd.pipeline(
        vertex_main,
        fragment_main,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
)
kernel_vjp_asset = vd.program_asset(
    id="capture/kernel_vjp",
    program=vd.ad.vjp(square, wrt=("source",), outputs=("output",)),
)
module_asset = vd.program_asset(id="capture/module", program=Square())
module_vjp_asset = vd.program_asset(
    id="capture/module_vjp",
    program=vd.ad.vjp(Square(), wrt=("source",), outputs=("output",)),
)
"""

    def test_all_five_authored_forms_produce_only_captured_program(self) -> None:
        from vernon_dsl._program_assets.source import load_program_asset_declaration

        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "capture_forms.py"
            source.write_text(self._SOURCE, encoding="utf-8")
            captures = tuple(
                capture_program(load_program_asset_declaration(source, name))
                for name in (
                    "kernel_asset",
                    "pipeline_asset",
                    "kernel_vjp_asset",
                    "module_asset",
                    "module_vjp_asset",
                )
            )

        self.assertTrue(all(type(captured) is CapturedProgram for captured in captures))
        self.assertEqual(tuple(captured.variant_keys for captured in captures), (((),),) * 5)
        self.assertEqual(captures[2].variants[0].ir.vjp_wrt, ("source",))
        self.assertEqual(captures[4].variants[0].ir.vjp_wrt, ("source",))

    def test_tuple_graphics_asset_is_rejected_without_a_legacy_capture_path(self) -> None:
        @vd.vertex
        def vertex_main() -> None:
            pass

        @vd.fragment
        def fragment_main() -> None:
            pass

        with self.assertRaisesRegex(TypeError, r"must use vd\.pipeline"):
            vd.program_asset(id="capture/tuple", program=cast(Any, (vertex_main, fragment_main)))

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tuple_asset.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.vertex
def vertex_main() -> None:
    pass

@vd.fragment
def fragment_main() -> None:
    pass

asset = vd.program_asset(id="capture/tuple", program=(vertex_main, fragment_main))
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, r"must use vd\.pipeline"):
                lint_python_program_asset(source, "asset")


class GraphicsTargetFormatTests(unittest.TestCase):
    """Attachment formats and sample count are pipeline state, so they are declared on vd.pipeline(...).

    Only the attachment extent, the bound texture, and the dynamic state belong to an invocation. Every backend
    bakes formats into the pipeline object, so a cooked graphics asset has to name them; a live pipeline reads
    them from the RenderPass it is called with, so there they stay optional.
    """

    @staticmethod
    def _stages() -> tuple[Any, Any]:
        @vd.vertex
        def vertex_main() -> None:
            pass

        @vd.fragment
        def fragment_main() -> None:
            pass

        return vertex_main, fragment_main

    def test_cooking_a_graphics_pipeline_requires_declared_target_formats(self) -> None:
        vertex_main, fragment_main = self._stages()
        with self.assertRaisesRegex(ValueError, "requires vd.pipeline\\(\\.\\.\\., targets="):
            vd.program_asset(id="graphics", program=vd.pipeline(vertex_main, fragment_main))

    def test_declared_target_formats_are_carried_on_the_pipeline(self) -> None:
        from vernon_dsl._runtime.pipeline import Pipeline

        vertex_main, fragment_main = self._stages()
        asset = vd.program_asset(
            id="graphics",
            program=vd.pipeline(
                vertex_main,
                fragment_main,
                targets=vd.target_formats(colors={0: vd.rgba8_unorm}, depth=vd.d32_float, samples=4),
            ),
        )
        assert isinstance(asset.program, Pipeline)
        targets = asset.program._targets
        assert targets is not None
        self.assertEqual(targets.colors, ((0, vd.rgba8_unorm),))
        self.assertEqual(targets.depth, vd.d32_float)
        self.assertEqual(targets.samples, 4)

    def test_live_graphics_pipelines_do_not_require_target_formats(self) -> None:
        vertex_main, fragment_main = self._stages()
        self.assertIsNone(vd.pipeline(vertex_main, fragment_main)._targets)

    def test_target_formats_reject_incoherent_attachments(self) -> None:
        for arguments, message in (
            ({"colors": {0: vd.d32_float}}, "is not a color format"),
            ({"depth": vd.rgba8_unorm}, "is not a depth format"),
            ({"colors": {-1: vd.rgba8_unorm}}, "must be non-negative"),
            ({}, "at least one color or depth target"),
            ({"colors": {0: vd.rgba8_unorm}, "samples": 0}, "positive integer"),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    vd.target_formats(**cast(dict[str, Any], arguments))

    def test_compute_assets_are_unaffected_by_the_graphics_requirement(self) -> None:
        @vd.kernel(workgroup_size=(1, 1, 1))
        def compute(out: vd.TensorView[vd.f32, (4,), vd.write]) -> None:
            out[0] = 1.0

        self.assertEqual(vd.program_asset(id="compute", program=compute).id, "compute")

    def test_bare_kernel_cooks_symbolic_dispatch_controls(self) -> None:
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        source = Path(__file__).with_name("program_asset_fixture.py")
        with tempfile.TemporaryDirectory() as directory:
            manifest = cook_program_asset(
                program_asset=f"{source}:scale_asset",
                output=directory,
                target="cpu",
            )
            document = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(document["type"], "program")
        program = document["variants"][0]["program"]
        for index, slot in enumerate(program["abi"]["boundary_slots"]):
            self.assertEqual(slot["id"], index)
            _assert_exact_boundary_slot(self, program, slot)
        boundaries = {slot["path"]: slot for slot in program["abi"]["boundary_slots"]}
        self.assertEqual(boundaries["values"]["outer_shape"], [-1])
        values = {value["name"]: value for value in program["values"]}
        for axis in "xyz":
            boundary = boundaries[f"__grid_{axis}"]
            self.assertEqual(boundary["category"], "value")
            self.assertEqual(boundary["value"], values[f"input.__grid_{axis}"]["id"])
        workgroups = program["graphs"][0]["nodes"][0]["operation"]["workgroups"]
        self.assertEqual(
            workgroups,
            [
                {"control": {"value": values["input.__grid_x"]["id"]}},
                {"control": {"value": values["input.__grid_y"]["id"]}},
                {"control": {"value": values["input.__grid_z"]["id"]}},
            ],
        )


_BARE_PIPELINE_ASSET = """
from typing import Annotated

import vernon_dsl as vd


@vd.vertex
def vertex_main(
    vertices: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([vertices, 0.0, 1.0])


@vd.fragment
def fragment_main() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 0.5, 0.25, 1.0])


asset = vd.program_asset(
    id="shaders/bare",
    program=vd.pipeline(
        vertex_main,
        fragment_main,
        {extra}targets=vd.target_formats(colors={{0: vd.{format}}}),
    ),
)
"""


class BarePipelineAssetCookTests(unittest.TestCase):
    """A bare vd.pipeline(...) is a cookable asset form, which it previously was not.

    It parses as an ast.Call, so the old AST kind guess routed it to the Module branch and cooking failed with
    'declared a host Program that is not a Vernon Module'. Dispatch now happens on the evaluated declaration.
    """

    def _cook(self, directory: str, name: str, *, format: str = "rgba16_float", extra: str = "") -> dict[str, Any]:
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        source = Path(directory) / f"{name}.py"
        source.write_text(_BARE_PIPELINE_ASSET.format(format=format, extra=extra), encoding="utf-8")
        manifest = cook_program_asset(
            program_asset=f"{source}:asset",
            output=Path(directory) / name,
            target="vulkan",
        )
        return cast(dict[str, Any], json.loads(manifest.read_text(encoding="utf-8")))

    def test_a_bare_pipeline_asset_cooks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = self._cook(directory, "bare")
        self.assertEqual(document["id"], "shaders/bare")
        self.assertEqual([variant["key"] for variant in document["variants"]], [[]])
        program = document["variants"][0]["program"]
        self.assertEqual(len(program["graphs"][0]["nodes"]), 1)

    def test_the_declared_format_is_what_the_cooked_attachment_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            half = self._cook(directory, "half", format="rgba16_float")
            byte = self._cook(directory, "byte", format="rgba8_unorm")

        def attachment(document: dict[str, Any]) -> tuple[Any, Any]:
            program = document["variants"][0]["program"]
            colors = program["graphs"][0]["nodes"][0]["operation"]["render_pass"]["colors"]
            images = [
                storage["descriptor"] for storage in program["storages"] if storage["descriptor"].get("tag") == "image"
            ]
            return colors[0]["formats"], images[0]

        half_formats, half_image = attachment(half)
        byte_formats, byte_image = attachment(byte)
        self.assertEqual(half_formats, ["rgba16_float"])
        self.assertEqual(byte_formats, ["rgba8_unorm"])
        self.assertEqual(half_image["format"], "rgba16_float")
        self.assertEqual(byte_image["format"], "rgba8_unorm")

    def test_no_invocation_fact_reaches_the_cooked_graphics_asset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            program = self._cook(directory, "symbolic")["variants"][0]["program"]
        storages = {
            storage["descriptor"]["tag"]: storage["descriptor"]
            for storage in program["storages"]
            if "tag" in storage["descriptor"]
        }
        self.assertNotIn("extent", storages["image"], "borrowed attachment extent comes from each bound image view")
        self.assertEqual(storages["buffer"]["byte_length"], 0, "the vertex count is chosen per invocation")
        self.assertIn("vertex", storages["buffer"]["usage"])

    def test_a_cooked_pipeline_may_not_carry_its_own_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProgramCompileError, "must not carry its own features"):
                self._cook(directory, "featured", extra="features=('SKIN',), ")

    def test_every_variant_of_a_canonical_asset_is_deployed(self) -> None:
        """Canonical deployment used to refuse more than one variant, and its caller split the plan to get around it."""

        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "multi.py"
            source.write_text(
                """
from typing import Annotated

import vernon_dsl as vd

TINTED = vd.feature("TINTED")


@vd.vertex
def vertex_main(
    vertices: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([vertices, 0.0, 1.0])


@vd.fragment
def fragment_main() -> vd.Vector[vd.f32, 4]:
    if TINTED:
        return vd.Vector([1.0, 0.0, 0.0, 1.0])
    return vd.Vector([1.0, 1.0, 1.0, 1.0])


asset = vd.program_asset(
    id="shaders/multi",
    program=vd.pipeline(
        vertex_main,
        fragment_main,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
    variants=((), (TINTED,)),
)
""",
                encoding="utf-8",
            )
            manifest = cook_program_asset(
                program_asset=f"{source}:asset",
                output=Path(directory) / "out",
                target="vulkan",
            )
            document = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual([variant["key"] for variant in document["variants"]], [["TINTED"], []])
        deployed = [
            canonical_json(
                {
                    stage: [module["blob"] for module in artifact["modules"]]
                    for stage, artifact in variant["artifact_system"]["artifacts"].items()
                }
            )
            for variant in document["variants"]
        ]
        self.assertEqual(len(set(deployed)), 2, "each variant deploys the binaries compiled for its own feature key")

    def test_a_graphics_vjp_asset_is_refused_as_an_unsupported_capability(self) -> None:
        """One diagnostic, not the contradictory demand for a rule set that no graphics VJP could ever use."""

        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "graphics_vjp.py"
            source.write_text(
                """
from typing import Annotated

import vernon_dsl as vd


@vd.vertex
def vertex_main(
    vertices: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([vertices, 0.0, 1.0])


@vd.fragment
def fragment_main() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 1.0, 1.0, 1.0])


asset = vd.program_asset(
    id="shaders/graphics_vjp",
    program=vd.ad.vjp(
        vd.pipeline(
            vertex_main,
            fragment_main,
            targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
        ),
        wrt=("vertices",),
    ),
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "PROGRAM_GRAPHICS_VJP_UNSUPPORTED"):
                cook_program_asset(
                    program_asset=f"{source}:asset",
                    output=Path(directory) / "out",
                    target="vulkan",
                )


class ShaderAssetCookTests(unittest.TestCase):
    def test_wasm_relocatable_object_preserves_compound_suffix(self) -> None:
        self.assertEqual(
            artifact_extension("relocatable_object", "compute", "module.wasm.o"),
            ".wasm.o",
        )

    def test_cooker_rejects_non_python_pipeline_asset_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ProgramCompileError, "source.py:descriptor_name"):
                cook_program_asset(
                    program_asset=root / "asset.json",
                    output=root / "output",
                )
            with self.assertRaisesRegex(ProgramCompileError, "valid Python descriptor name"):
                cook_program_asset(
                    program_asset=f"{root / 'asset.py'}:",
                    output=root / "output",
                )

    def test_interactive_glsl_and_spirv_descriptors_include_integrity(self) -> None:
        glsl = encode_runtime_stage(
            {
                "format": "glsl",
                "stage": "vertex",
            },
            b"void main() {}\n",
        )
        self.assertEqual(glsl["artifact"]["encoding"], "utf8")
        self.assertEqual(glsl["artifact"]["data"], "void main() {}\n")
        self.assertEqual(glsl["artifact"]["size"], 15)

        spirv_bytes = b"\x03\x02\x23\x07"
        spirv = encode_runtime_stage(
            {
                "format": "spirv",
                "stage": "compute",
            },
            spirv_bytes,
        )
        self.assertEqual(spirv["artifact"]["encoding"], "base64")
        self.assertEqual(
            base64.b64decode(spirv["artifact"]["data"], validate=True),
            spirv_bytes,
        )
        self.assertEqual(spirv["artifact"]["size"], len(spirv_bytes))

    def test_interactive_cuda_stage_uses_inline_ptx_descriptor(self) -> None:
        reflection = {
            "compiler_contract_version": COMPILER_CONTRACT_VERSION,
            "program_version": PROGRAM_VERSION,
            "entries": [{"name": "scale", "stage": "compute"}],
        }
        record = {
            "entry": "scale",
            "stage": "compute",
            "target": "cuda",
            "format": "ptx",
            "reflection": reflection,
        }
        encoded = encode_runtime_stage(record, b".version 8.0\n")
        artifact = encoded["artifact"]
        self.assertEqual(artifact["storage"], "inline")
        self.assertEqual(artifact["format"], "ptx")
        self.assertEqual(artifact["encoding"], "utf8")
        self.assertEqual(artifact["data"], ".version 8.0\n")
        self.assertEqual(artifact["size"], 13)
        self.assertEqual(
            artifact["sha256"],
            hashlib.sha256(b".version 8.0\n").hexdigest(),
        )
        self.assertEqual(encoded["reflection"], reflection)

    def test_module_single_compute_cooks_canonical_program(self) -> None:
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "module_compute.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel(workgroup_size=(1, 1, 1))
def square(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0]

class Square(vd.Module):
    def forward(
        self,
        source: vd.TensorView[vd.f32, (1,), vd.read],
    ) -> vd.TensorStorage:
        output = vd.empty_like(source)
        square(source, output)
        return output

asset = vd.program_asset(id="module/square", program=Square())
""",
                encoding="utf-8",
            )

            manifest = cook_program_asset(
                program_asset=f"{source}:asset",
                output=root / "cooked",
                target="cpu",
            )
            bundle = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(bundle["type"], "program")
            program = bundle["variants"][0]["program"]
            self.assertEqual(
                set(program),
                {
                    "stages",
                    "parameters",
                    "storages",
                    "values",
                    "graphs",
                    "abi",
                },
            )
            self.assertEqual(len(program["graphs"]), 1)
            self.assertEqual(len(program["graphs"][0]["nodes"]), 1)
            node = program["graphs"][0]["nodes"][0]
            self.assertNotIn("dependencies", node)
            self.assertEqual(node["operation"], {"tag": "compute", "workgroups": [1, 1, 1]})
            self.assertEqual(
                [value["origin"]["tag"] for value in program["values"]],
                ["argument", "allocation", "node_result"],
            )
            self.assertEqual([storage["ownership"] for storage in program["storages"]], ["borrowed", "owned"])

            from vernon_dsl._program_assets.artifact_io import write_external_artifact
            from vernon_dsl._program_assets.capture import capture_program
            from vernon_dsl._program_assets.compile_orchestration import (
                _native_module,
                compile_captured_program,
            )
            from vernon_dsl._program_assets.source import load_program_asset_declaration

            native = _native_module()
            declaration = load_program_asset_declaration(source, "asset")
            captured = capture_program(declaration)
            for target_name in ("cpu", "metal", "vulkan"):
                with self.subTest(target=target_name):
                    target = make_target_options(target_name)
                    retained: list[tuple[CompiledStage, object]] = []
                    plan = compile_captured_program(
                        captured,
                        target,
                        retained,
                    )
                    runtime_root = root / f"runtime-{target_name}"
                    runtime_root.mkdir()
                    descriptors = {
                        stage.id: write_external_artifact(
                            runtime_root,
                            stage.artifact.data,
                            stage.artifact.format,
                            stage.stage,
                            stage.artifact.filename,
                        )
                        for stage in plan.compiled_stages
                    }
                    deployed = build_program_manifest(plan, descriptors)
                    artifact_system = deployed["variants"][0]["artifact_system"]
                    host = None
                    if target_name == "cpu":
                        runtime = native.Runtime(native.RuntimeBackend.CPU)
                        compiled_stages = [
                            (stage.metadata["symbol"], stage.entry, result) for stage, result in retained
                        ]
                        # CPU objects are link-time inputs. Interactive ORC
                        # produces the registered entry addresses above; the
                        # runtime must not read the original object files.
                        for descriptor in descriptors.values():
                            (runtime_root / descriptor["path"]).unlink()
                    else:
                        runtime_backend = getattr(native.RuntimeBackend, target_name.upper())
                        if not native.runtime_available(runtime_backend):
                            continue
                        backend = getattr(native.RhiBackend, target_name.upper())
                        host = native.RhiHost(backend)
                        runtime = host.create_runtime()
                        compiled_stages = []
                    loaded = runtime.load_canonical_program(
                        canonical_json(deployed).encode(),
                        str(runtime_root),
                        compiled_stages,
                    )
                    source_array = np.array([3.0], dtype=np.float32)
                    output_array = np.zeros(1, dtype=np.float32)
                    output_buffer = None
                    invocation = loaded.program_instance().begin_invocation()
                    builder = invocation.builder
                    if target_name == "cpu":
                        for parameter, array in zip(loaded.parameters, (source_array, output_array), strict=True):
                            prepared = builder.prepare_host_tensor(parameter.slot, array)
                            invocation.bind(
                                parameter.slot,
                                ("test", parameter.slot),
                                lambda prepared=prepared: prepared,
                            )
                    else:
                        assert host is not None
                        source_buffer = host.create_buffer(source_array.nbytes)
                        output_buffer = host.create_buffer(output_array.nbytes)
                        source_buffer.upload(source_array.tobytes())
                        output_buffer.upload(output_array.tobytes())
                        for parameter, buffer in zip(loaded.parameters, (source_buffer, output_buffer), strict=True):
                            prepared = builder.prepare_rhi_tensor(
                                parameter.slot,
                                buffer,
                                parameter.access,
                                [1],
                                [4],
                            )
                            invocation.bind(
                                parameter.slot,
                                ("test", parameter.slot),
                                lambda prepared=prepared: prepared,
                            )
                    invocation.forward()
                    invocation.commit()
                    if target_name != "cpu":
                        assert output_buffer is not None
                        output_array = np.frombuffer(output_buffer.download(), dtype=np.float32)
                    np.testing.assert_array_equal(output_array, np.array([9.0], dtype=np.float32))
                    if target_name == "cpu":
                        legacy_deployment = copy.deepcopy(deployed)
                        legacy_deployment["variants"][0]["stage_bindings"] = {
                            stage: stage for stage in artifact_system["artifacts"]
                        }
                        with self.assertRaisesRegex(RuntimeError, "invalid|unsupported"):
                            runtime.load_canonical_program(
                                canonical_json(with_content_hash(legacy_deployment)).encode(),
                                str(runtime_root),
                                compiled_stages,
                            )
                        artifact_id = next(iter(artifact_system["artifacts"]))
                        bad_deployment = copy.deepcopy(deployed)
                        bad_program = bad_deployment["variants"][0]["program"]
                        bad_artifacts = bad_deployment["variants"][0]["artifact_system"]
                        bad_reflection = bad_artifacts["artifacts"][artifact_id]["reflection"]
                        bad_reflection["endpoints"][1]["abi"]["bindings"][0]["carrier"]["slot"] = 0
                        contract = {"operation": "compute", "reflection": bad_reflection}
                        contract_hash = hashlib.sha256(
                            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest()
                        bad_artifacts["artifacts"][artifact_id]["contract_hash"] = contract_hash
                        bad_program["stages"][next(iter(bad_program["stages"]))]["contract_hash"] = contract_hash
                        with self.assertRaisesRegex(RuntimeError, "portable ABI slots|resource ABI slots"):
                            runtime.load_canonical_program(
                                canonical_json(with_content_hash(bad_deployment)).encode(),
                                str(runtime_root),
                                compiled_stages,
                            )

                    else:
                        artifact_path = runtime_root / descriptors[next(iter(descriptors))]["path"]
                        original_bytes = artifact_path.read_bytes()
                        artifact_path.write_bytes(b"broken")
                        try:
                            with self.assertRaisesRegex(RuntimeError, "PROGRAM_BLOB_AUTHENTICATION"):
                                runtime.load_canonical_program(
                                    canonical_json(deployed).encode(),
                                    str(runtime_root),
                                    compiled_stages,
                                )
                        finally:
                            artifact_path.write_bytes(original_bytes)

    def test_cpu_graphics_pipeline_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "graphics.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.vertex
def vertex_main() -> None:
    pass

@vd.fragment
def fragment_main() -> None:
    pass

asset = vd.program_asset(
    id="graphics",
    program=vd.pipeline(
        vertex_main,
        fragment_main,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
    variants=((),),
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "CPU Program Assets do not support graphics"):
                cook_program_asset(
                    program_asset=f"{source}:asset",
                    output=root / "output",
                    target="cpu",
                )

    def test_gpu_no_tape_vjp_cooks_canonical_program(self) -> None:
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        source = Path(__file__).parents[2] / "source" / "tests" / "fixtures" / "autodiff_gpu_no_tape_asset.py"
        with tempfile.TemporaryDirectory() as directory:
            manifest = cook_program_asset(
                program_asset=f"{source}:asset",
                output=directory,
                target="vulkan",
            )
            bundle = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(bundle["target"]["kind"], "vulkan")
            self.assertEqual(bundle["type"], "program")
            self.assertNotIn("autodiff", bundle)
            variant = bundle["variants"][0]
            self.assertNotIn("execution", variant)
            program = variant["program"]
            self.assertEqual([graph["direction"] for graph in program["graphs"]], ["forward", "backward"])
            self.assertEqual(set(program["stages"]), set(variant["artifact_system"]["artifacts"]))

    def test_gpu_captured_tape_vjp_cooks_canonical_programs(self) -> None:
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        from vernon_dsl._program_assets.compile_orchestration import _native_module

        native = _native_module()
        source = Path(__file__).parents[2] / "source" / "tests" / "fixtures" / "autodiff_gpu_tape_asset.py"
        targets = {
            "cuda": native.Target.CUDA,
            "vulkan": native.Target.VULKAN,
            "directx": native.Target.DIRECTX,
            "metal": native.Target.METAL,
            "opengl": native.Target.OPENGL,
        }
        for descriptor in ("static_asset", "dynamic_asset"):
            for target, native_target in targets.items():
                with (
                    self.subTest(descriptor=descriptor, target=target),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    if not native.target_available(native_target):
                        continue
                    manifest = cook_program_asset(
                        program_asset=f"{source}:{descriptor}",
                        output=directory,
                        target=target,
                    )
                    bundle = json.loads(manifest.read_text(encoding="utf-8"))
                    self.assertEqual(bundle["target"]["kind"], target)
                    self.assertEqual(bundle["type"], "program")
                    self.assertNotIn("autodiff", bundle)
                    variant = bundle["variants"][0]
                    self.assertNotIn("execution", variant)
                    program = variant["program"]
                    self.assertTrue(program["abi"]["tape_plans"])
                    self.assertEqual(
                        [graph["direction"] for graph in program["graphs"]],
                        ["forward", "backward"],
                    )
                    self.assertEqual(set(program["stages"]), set(variant["artifact_system"]["artifacts"]))

    def test_four_graphics_variants_deploy_canonical_programs(self) -> None:
        root = Path(__file__).parents[2]
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        with tempfile.TemporaryDirectory() as directory:
            manifest = cook_program_asset(
                program_asset=f"{root / 'examples' / 'variant_mesh.py'}:mesh_asset",
                output=directory,
            )
            bundle = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertNotIn("features", bundle)
            self.assertEqual(bundle["program_version"], PROGRAM_VERSION)
            self.assertEqual(bundle["type"], "program")
            self.assertEqual(bundle["id"], "shaders/variant_mesh")
            self.assertEqual(len(bundle["variants"]), 4)
            self.assertNotIn("stage_artifacts", bundle)
            for variant in bundle["variants"]:
                self.assertEqual(set(variant["program"]["stages"]), {"forward:0"})
                artifact = variant["artifact_system"]["artifacts"]["forward:0"]
                self.assertEqual([module["role"] for module in artifact["modules"]], ["vertex", "fragment"])
            combined = next(variant for variant in bundle["variants"] if variant["key"] == ["INSTANCE", "SKIN"])
            endpoints = combined["artifact_system"]["artifacts"]["forward:0"]["implementation"]["endpoints"]
            self.assertEqual(
                [endpoint["index"] for endpoint in endpoints if endpoint["module"] == "vertex"], [0, 1, 2, 3]
            )
            self.assertTrue(
                all(
                    blob["location"]["tag"] == "external" and "uri" in blob["location"]
                    for blob in bundle["blobs"].values()
                )
            )

    def test_python_descriptors_cook_all_runtime_backends(self) -> None:
        root = Path(__file__).parents[2]
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        source = root / "python" / "tests" / "program_asset_fixture.py"
        cases = {
            "opengl": ("triangle_asset", "pipelines/triangle", {"vertex", "fragment"}),
            "cuda": ("scale_asset", "pipelines/scale", {"compute"}),
            "cpu": ("scale_asset", "pipelines/scale", {"compute"}),
        }
        with tempfile.TemporaryDirectory() as directory:
            for target, (name, asset_id, stages) in cases.items():
                output = Path(directory) / target
                manifest = cook_program_asset(
                    program_asset=f"{source}:{name}",
                    output=output,
                    target=target,
                )
                document = json.loads(manifest.read_text(encoding="utf-8"))
                self.assertEqual(document["id"], asset_id)
                self.assertEqual(document["target"]["kind"], target)
                if target == "cpu":
                    self.assertTrue(document["target"]["options"]["triple"])
                else:
                    self.assertEqual(document["target"]["options"], {})
                self.assertEqual({module["role"] for module in _deployed_modules(document)}, stages)
                if target in {"cuda", "cpu"}:
                    self.assertEqual(document["type"], "program")
                    program = document["variants"][0]["program"]
                    for index, slot in enumerate(program["abi"]["boundary_slots"]):
                        self.assertEqual(slot["id"], index)
                        _assert_exact_boundary_slot(self, program, slot)
                    boundaries = {
                        boundary["path"]: boundary
                        for boundary in program["abi"]["boundary_slots"]
                        if boundary["role"] == "input"
                    }
                    self.assertEqual(boundaries["values"]["outer_shape"], [-1])
                if target == "opengl":
                    self.assertEqual(document["type"], "program")
                    self.assertNotIn("stage_artifacts", document)
                    self.assertEqual(len(document["variants"]), 2)
                    repeated = Path(directory) / "opengl_repeated"
                    repeated_manifest = cook_program_asset(
                        program_asset=f"{source}:{name}",
                        output=repeated,
                        target=target,
                    )
                    repeated_document = json.loads(repeated_manifest.read_text(encoding="utf-8"))
                    self.assertEqual(document, repeated_document)
                for module in _deployed_modules(document):
                    data = (output / module["path"]).read_bytes()
                    self.assertTrue(data)
                    self.assertEqual(hashlib.sha256(data).hexdigest(), module["sha256"])

    def test_cook_only_source_targets_support_graphics_and_compute(self) -> None:
        root = Path(__file__).parents[2]
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        from vernon_dsl._program_assets.compile_orchestration import _native_module

        native = _native_module()
        assets = (
            (root / "python" / "tests" / "cube_map_shader.py", "cube_map_asset", {"vertex", "fragment"}),
            (root / "python" / "tests" / "program_asset_fixture.py", "scale_asset", {"compute"}),
        )
        targets = {
            "metal": ("msl", {"platform": "macos"}),
            "directx": ("dxil", {"shader_model": 60}),
        }
        with tempfile.TemporaryDirectory() as directory:
            for source, name, stages in assets:
                for target, (artifact_format, target_options) in targets.items():
                    with self.subTest(asset=name, target=target):
                        if target == "directx" and not native.target_available(native.Target.DIRECTX):
                            continue
                        output = Path(directory) / f"{name}_{target}"
                        manifest = cook_program_asset(
                            program_asset=f"{source}:{name}",
                            output=output,
                            target=make_target_options(target, target_options),
                        )
                        document = json.loads(manifest.read_text(encoding="utf-8"))
                        self.assertEqual(document["target"], {"kind": target, "options": target_options})
                        modules = _deployed_modules(document)
                        self.assertEqual({module["role"] for module in modules}, stages)
                        if name == "cube_map_asset":
                            # The three matrices cross the boundary as whole 4x4 values rather than as one
                            # endpoint per leaf member, which is what a Program Value guarantees.
                            matrices = [
                                endpoint
                                for endpoint in document["variants"][0]["artifact_system"]["artifacts"]["forward:0"][
                                    "reflection"
                                ]["endpoints"]
                                if endpoint.get("type") == "tensor<4x4xf32>"
                            ]
                            self.assertEqual(len(matrices), 3)
                            self.assertTrue(all(endpoint["transport"] == "by_value" for endpoint in matrices))
                        for module in modules:
                            if target == "metal":
                                slots = module["implementation"]["metadata"]["resource_slots"]
                                self.assertTrue(slots)
                                self.assertTrue(
                                    any(slot["entry_point"] == module["entry"] for slot in slots),
                                )
                            self.assertEqual(module["format"], artifact_format)
                            data = (output / module["path"]).read_bytes()
                            self.assertTrue(data)
                            if artifact_format != "dxil":
                                data.decode("utf-8")
                            else:
                                self.assertEqual(data[:4], b"DXBC")

    def test_vulkan_pipeline_bundle_embeds_verified_spirv(self) -> None:
        root = Path(__file__).parents[2]
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        with tempfile.TemporaryDirectory() as directory:
            manifest = cook_program_asset(
                program_asset=f"{root / 'examples' / 'variant_mesh.py'}:mesh_asset",
                output=directory,
                target="vulkan",
            )
            runtime_bundle = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(runtime_bundle["type"], "program")
            for module in _deployed_modules(runtime_bundle):
                self.assertEqual(module["format"], "spirv")
                decoded = (manifest.parent / module["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(decoded).hexdigest(), module["sha256"])


if __name__ == "__main__":
    unittest.main()
