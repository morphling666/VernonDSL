from __future__ import annotations

import base64
import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import numpy as np
import vernon_dsl as vd
from vernon_dsl._program_assets.artifact_io import artifact_extension
from vernon_dsl._versions import COMPILER_CONTRACT_VERSION, PIPELINE_VERSION
from vernon_dsl.bundle import (
    CompiledArtifact,
    CompiledStage,
    CpuTargetOptions,
    build_bundle_plan,
    build_program_bundle_plan,
    make_target_options,
)
from vernon_dsl.module_graph import load_project, resolve_project_entry
from vernon_dsl.program_assets import (
    ProgramCompileError,
    cook_program_asset,
    encode_runtime_stage,
    parse_python_program_asset,
)


def _fake_native(compile_program_result: object) -> SimpleNamespace:
    targets = SimpleNamespace(
        CPU="cpu",
        CUDA="cuda",
        VULKAN="vulkan",
        METAL="metal",
        DIRECTX="directx",
        OPENGL="opengl",
        OPENGL_ES="opengles",
    )
    return SimpleNamespace(
        Target=targets,
        Compiler=lambda: SimpleNamespace(compile_program_result=compile_program_result),
    )


def _native_available() -> bool:
    try:
        from vernon_dsl._program_assets.cooking import _native_module

        _native_module()
        return True
    except ProgramCompileError:
        return False


class ShaderAssetManifestTests(unittest.TestCase):
    def test_program_bundle_composes_logical_stages_with_kernel_artifacts(self) -> None:
        target = make_target_options("vulkan")
        stage = CompiledStage(
            "python/square",
            "{}",
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
        plan = build_program_bundle_plan(
            "modules/square",
            target,
            (),
            {"Module.square": stage},
            {
                "stages": {"Module.square": {"operation": "compute", "contract_hash": "0" * 64}},
                "graphs": [
                    {
                        "nodes": [
                            {
                                "operation": {"tag": "compute", "workgroups": [4, 1, 1]},
                            }
                        ]
                    }
                ],
            },
        )

        self.assertEqual(dict(plan.variants[0].program), {"Module.square": stage.id})
        canonical = plan.variants[0].canonical_program
        assert canonical is not None
        self.assertEqual(set(canonical["stages"]), {"Module.square"})
        self.assertNotIn("dependencies", canonical["graphs"][0]["nodes"][0])
        self.assertEqual(canonical["graphs"][0]["nodes"][0]["operation"]["workgroups"], [4, 1, 1])
        self.assertEqual(plan.variants[0].parameters, ())

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
    program=(fullscreen_vertex, fragment_main),
)
""",
                encoding="utf-8",
            )

            descriptor = parse_python_program_asset(source, "asset")
            self.assertEqual(descriptor.stages["vertex"].entry, "fullscreen_vertex")
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
    program=(fullscreen.fullscreen_vertex, fragment_main),
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "simple imported or local function names"):
                parse_python_program_asset(qualified, "asset")

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
    program=(fullscreen_vertex, fragment_main),
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "has no symbol 'fullscreen_vertex'"):
                parse_python_program_asset(reexported, "asset")

    def test_python_pipeline_asset_is_parsed_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd
raise RuntimeError("the cooker must not execute this module")
FEATURE = vd.feature("FEATURE")

@vd.vertex
def vertex_main(value: vd.f32) -> vd.f32:
    return value

@vd.fragment
def fragment_main(value: vd.f32) -> vd.f32:
    return value

asset = vd.program_asset(
    id="pipelines/static",
    program=(vertex_main, fragment_main),
    variants=((), (FEATURE,)),
)
""",
                encoding="utf-8",
            )
            descriptor = parse_python_program_asset(source, "asset")
            self.assertEqual(descriptor.id, "pipelines/static")
            self.assertEqual(descriptor.variants, ((), ("FEATURE",)))
            self.assertEqual(set(descriptor.stages), {"vertex", "fragment"})
            self.assertNotIn("targets", json.loads(descriptor.canonical_manifest))

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
                parse_python_program_asset(source, "asset")

    def test_python_pipeline_asset_rejects_graphics_stage_order(self) -> None:
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
    program=(fragment_main, vertex_main),
    variants=((),),
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "topology order"):
                parse_python_program_asset(source, "asset")

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
                parse_python_program_asset(source, "asset")

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
                parse_python_program_asset(source, "asset")

    def test_example_pipeline_asset_has_canonical_variants(self) -> None:
        root = Path(__file__).parents[2]
        pipeline = parse_python_program_asset(root / "examples" / "variant_mesh.py", "mesh_asset")
        self.assertEqual(pipeline.id, "shaders/variant_mesh")
        self.assertEqual(
            pipeline.variants,
            ((), ("INSTANCE",), ("SKIN",), ("INSTANCE", "SKIN")),
        )


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
        vertex_main, fragment_main = self._stages()
        asset = vd.program_asset(
            id="graphics",
            program=vd.pipeline(
                vertex_main,
                fragment_main,
                targets=vd.target_formats(colors={0: vd.rgba8_unorm}, depth=vd.d32_float, samples=4),
            ),
        )
        targets = asset.program._targets
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
                    vd.target_formats(**arguments)

    def test_compute_assets_are_unaffected_by_the_graphics_requirement(self) -> None:
        @vd.kernel(workgroup_size=(1, 1, 1))
        def compute(out: vd.TensorView[vd.f32, (4,), vd.write]) -> None:
            out[0] = 1.0

        self.assertEqual(vd.program_asset(id="compute", program=compute).id, "compute")


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
        self.assertEqual(storages["image"]["extent"], [0, 0, 1], "the attachment extent is chosen per invocation")
        self.assertEqual(storages["buffer"]["byte_length"], 0, "the vertex count is chosen per invocation")
        self.assertIn("vertex", storages["buffer"]["usage"])

    def test_a_cooked_pipeline_may_not_carry_its_own_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProgramCompileError, "must not carry its own features"):
                self._cook(directory, "featured", extra="features=('SKIN',), ")

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


def rasterization() -> None:
    return


def visibility() -> None:
    return


def depth() -> None:
    return


def blend() -> None:
    return


def texture() -> None:
    return


rules = vd.ad.rule_set(
    id="render/v1",
    rasterization=rasterization,
    visibility=visibility,
    depth=depth,
    blend=blend,
    texture=texture,
)

asset = vd.program_asset(
    id="shaders/graphics_vjp",
    program=vd.ad.vjp((vertex_main, fragment_main), wrt=("vertices",), rules=rules),
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
            "pipeline_version": PIPELINE_VERSION,
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

    def test_cpu_pipeline_copies_content_addressed_object(self) -> None:
        relocatable_object = b"mock relocatable object"
        digest = hashlib.sha256(relocatable_object).hexdigest()
        reflection = {
            "compiler_contract_version": COMPILER_CONTRACT_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "module_hash": "module",
            "dependencies": [],
            "entries": [
                {
                    "name": "scale",
                    "symbol": "__vernon_cpu_module_scale",
                    "stage": "compute",
                    "arguments": [],
                    "results": [],
                    "workgroup_size": [1, 1, 1],
                }
            ],
            "artifacts": [
                {
                    "entry_point": "scale",
                    "stage": "compute",
                    "format": "relocatable_object",
                    "filename": "module.obj",
                }
            ],
            "target": {
                "kind": "cpu",
                "options": {
                    "triple": "x86_64-pc-windows-msvc",
                    "processor": "generic",
                    "features": ["+sse2"],
                },
            },
        }
        calls: list[tuple[object, dict[str, object]]] = []
        plans: list[object] = []

        def capture_plan(*args: object, **kwargs: object) -> object:
            plan = build_bundle_plan(*args, **kwargs)
            plans.append(plan)
            return plan

        def compile_program_result(_mlir: str, native_target: object, **options: object) -> SimpleNamespace:
            calls.append((native_target, options))
            return SimpleNamespace(
                ok=True,
                diagnostics="",
                reflection=json.dumps(reflection),
                artifacts=[("module.obj", relocatable_object)],
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "kernel.py"
            source.write_text(
                """
import vernon_dsl as vd

UNUSED = vd.feature("UNUSED")

@vd.kernel
def scale() -> None:
    pass

asset = vd.program_asset(
    id="pipelines/cpu",
    program=scale,
    variants=((),),
)
""",
                encoding="utf-8",
            )
            output = root / "cooked"
            with (
                mock.patch("vernon_dsl._program_assets.cooking.compile_file", return_value="module {}"),
                mock.patch(
                    "vernon_dsl._program_assets.cooking.load_project", return_value=SimpleNamespace(features={"UNUSED"})
                ),
                mock.patch(
                    "vernon_dsl._program_assets.cooking._native_module",
                    return_value=_fake_native(compile_program_result),
                ),
                mock.patch("subprocess.run", side_effect=AssertionError("cooker invoked subprocess")),
                mock.patch("vernon_dsl._program_assets.cooking.build_bundle_plan", side_effect=capture_plan),
            ):
                manifest_path = cook_program_asset(
                    program_asset=f"{source}:asset",
                    output=output,
                    target=CpuTargetOptions(processor="generic", features=("+sse2",)),
                )

            self.assertEqual(
                calls,
                [
                    (
                        "cpu",
                        {
                            "options": {
                                "processor": "generic",
                                "features": "+sse2",
                            },
                        },
                    )
                ],
            )
            bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertNotIn("features", bundle)
            self.assertNotIn("execution", bundle["variants"][0])
            self.assertEqual(
                bundle["target"],
                {
                    "kind": "cpu",
                    "options": {
                        "triple": "x86_64-pc-windows-msvc",
                        "processor": "generic",
                        "features": ["+sse2"],
                    },
                },
            )
            self.assertEqual(
                bundle["runtime_requirements"],
                {
                    "backend": "cpu",
                    "features": [],
                    "target_triple": "x86_64-pc-windows-msvc",
                    "object_format": "coff",
                },
            )
            logical = json.loads(json.dumps(bundle))
            logical.pop("content_hash")
            for record in logical["stage_artifacts"].values():
                record.pop("artifact")
                record.pop("symbol")
            self.assertEqual(logical, plans[0].logical_dict())
            stage = next(iter(bundle["stage_artifacts"].values()))
            self.assertEqual(stage["symbol"], "__vernon_cpu_module_scale")
            self.assertEqual(
                set(stage),
                {"stage", "entry", "reflection", "artifact", "symbol"},
            )
            self.assertEqual(stage["artifact"]["sha256"], digest)
            self.assertEqual(stage["artifact"]["size"], len(relocatable_object))
            self.assertEqual(stage["artifact"]["storage"], "external")
            self.assertEqual(stage["artifact"]["format"], "relocatable_object")
            self.assertEqual(stage["artifact"]["path"], f"artifacts/{digest}.obj")
            self.assertEqual((output / stage["artifact"]["path"]).read_bytes(), relocatable_object)
            registration_sources = list(output.glob("vernon_cpu_registration_*.c"))
            self.assertEqual(len(registration_sources), 1)
            registration_source = registration_sources[0].read_text(encoding="utf-8")
            self.assertIn("vernonRuntimeRegisterStaticCpuEntry", registration_source)
            self.assertIn("&__vernon_cpu_module_scale", registration_source)
            self.assertTrue(registration_sources[0].with_suffix(".h").is_file())
            self.assertEqual(manifest_path.name, "cooked.program.json")
            self.assertFalse((output / "pipeline.bundle").exists())

    def test_mocked_gpu_targets_emit_external_deduplicated_artifacts(self) -> None:
        cases = {
            "cuda": (("compute",), "ptx", (".ptx",)),
            "opengl": (("vertex", "fragment"), "glsl", (".vert.glsl", ".frag.glsl")),
            "opengles": (("vertex", "fragment"), "gles", (".vert.gles", ".frag.gles")),
            "vulkan": (("vertex", "fragment"), "spirv", (".spv", ".spv")),
            "metal": (("vertex", "fragment"), "msl", (".vert.metal", ".frag.metal")),
            "directx": (("vertex", "fragment"), "dxil", (".dxil", ".dxil")),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target, (stages, artifact_format, extensions) in cases.items():
                with self.subTest(target=target):
                    case_root = root / target
                    case_root.mkdir()
                    source = case_root / "shader.py"
                    stage_definitions = "\n\n".join(
                        f"@vd.{'kernel' if stage == 'compute' else stage}\ndef {stage}_main() -> None:\n    pass"
                        for stage in stages
                    )
                    program_expression = (
                        f"{stages[0]}_main"
                        if stages == ("compute",)
                        else "(" + ", ".join(f"{stage}_main" for stage in stages) + ")"
                    )
                    source.write_text(
                        f"""
import vernon_dsl as vd

FEATURE = vd.feature("FEATURE")

{stage_definitions}

asset = vd.program_asset(
    id="pipelines/{target}",
    program={program_expression},
    variants=((), (FEATURE,)),
)
""",
                        encoding="utf-8",
                    )
                    compile_index = 0

                    def compile_program_result(
                        _mlir: str,
                        native_target: object,
                        *,
                        _target: str = target,
                        _stages: tuple[str, ...] = stages,
                        _artifact_format: str = artifact_format,
                        **options: object,
                    ) -> SimpleNamespace:
                        nonlocal compile_index
                        self.assertEqual(native_target, _target)
                        expected_native_options = (
                            {"options": {"shader_model": 60}}
                            if _target == "directx"
                            else {"options": {"platform": "macos"}}
                            if _target == "metal"
                            else {"options": {}}
                        )
                        self.assertEqual(options, expected_native_options)
                        expected_options = (
                            {"shader_model": 60}
                            if _target == "directx"
                            else {"platform": "macos"}
                            if _target == "metal"
                            else {}
                        )
                        stage = _stages[compile_index % len(_stages)]
                        compile_index += 1
                        extension = {
                            "glsl": ".glsl",
                            "gles": ".gles",
                            "ptx": ".ptx",
                            "spirv": ".spv",
                            "msl": ".metal",
                            "dxil": ".dxil",
                        }[_artifact_format]
                        filename = f"{stage}{extension}"
                        artifact = {
                            "glsl": b"#version 450\nvoid main() {}\n",
                            "gles": b"#version 310 es\nvoid main() {}\n",
                            "ptx": b".version 8.0\n.target sm_50\n.address_size 64\n",
                            "spirv": struct.pack("<II", 0x07230203, 0x00010300),
                            "msl": b"// metal artifact\n",
                            "dxil": b"DXBC",
                        }[_artifact_format]
                        artifact += stage.encode("ascii")
                        reflection = {
                            "module_hash": "module",
                            "dependencies": [],
                            "target": {"kind": _target, "options": expected_options},
                            "implementation": {"target": _target, "metadata": {}},
                            "entries": [
                                {
                                    "name": f"{stage}_main",
                                    "stage": stage,
                                    "arguments": [],
                                    "results": [],
                                }
                            ],
                            "artifacts": [
                                {
                                    "entry_point": f"{stage}_main",
                                    "stage": stage,
                                    "format": _artifact_format,
                                    "filename": filename,
                                }
                            ],
                        }
                        if _target == "metal":
                            reflection["target"]["output"] = {
                                "language": "msl",
                                "version": [2, 4],
                                "minimum_os_version": [11, 0],
                            }
                            reflection["implementation"]["metadata"]["resource_slots"] = [
                                {
                                    "entry_point": f"{stage}_main",
                                    "stage": stage,
                                    "kind": "uniform_buffer",
                                    "name": f"{stage}_uniforms",
                                    "set": 0,
                                    "binding": 0,
                                    "argument_buffer_index": 0,
                                    "member_id": 0,
                                    "direct_buffer_index": 4294967295,
                                    "count": 1,
                                }
                            ]
                        return SimpleNamespace(
                            ok=True,
                            diagnostics="",
                            reflection=json.dumps(reflection),
                            artifacts=[(filename, artifact)],
                        )

                    output = case_root / f"{target}_asset"
                    with (
                        mock.patch("vernon_dsl._program_assets.cooking.compile_file", return_value="module {}"),
                        mock.patch(
                            "vernon_dsl._program_assets.cooking.load_project",
                            return_value=SimpleNamespace(features={"FEATURE"}),
                        ),
                        mock.patch(
                            "vernon_dsl._program_assets.cooking._native_module",
                            return_value=_fake_native(compile_program_result),
                        ),
                    ):
                        manifest_path = cook_program_asset(
                            program_asset=f"{source}:asset",
                            output=output,
                            target=target,
                        )
                        first_manifest = manifest_path.read_bytes()
                        repeated_path = cook_program_asset(
                            program_asset=f"{source}:asset",
                            output=output,
                            target=target,
                        )

                    self.assertEqual(manifest_path.name, f"{target}_asset.program.json")
                    self.assertEqual(repeated_path, manifest_path)
                    self.assertEqual(manifest_path.read_bytes(), first_manifest)
                    document = json.loads(manifest_path.read_text(encoding="utf-8"))
                    self.assertEqual(document["pipeline_version"], PIPELINE_VERSION)
                    self.assertEqual(document["type"], "pipeline")
                    expected_target_options = (
                        {"shader_model": 60}
                        if target == "directx"
                        else {"platform": "macos"}
                        if target == "metal"
                        else {}
                    )
                    self.assertEqual(document["target"], {"kind": target, "options": expected_target_options})
                    self.assertEqual(
                        document["target"]["options"],
                        expected_target_options,
                    )
                    expected_requirements = {
                        "cuda": {
                            "backend": "cuda",
                            "features": [],
                            "ptx_version": [8, 0],
                            "minimum_compute_capability": [5, 0],
                            "address_size": 64,
                        },
                        "opengl": {
                            "backend": "opengl",
                            "features": [],
                            "glsl_version": 450,
                            "profile": "core",
                            "api_version": [4, 5],
                        },
                        "opengles": {
                            "backend": "opengles",
                            "features": [],
                            "glsl_version": 310,
                            "profile": "es",
                            "api_version": [3, 1],
                        },
                        "vulkan": {
                            "backend": "vulkan",
                            "features": [],
                            "api_version": [1, 1],
                            "spirv_version": [1, 3],
                        },
                        "directx": {
                            "backend": "directx",
                            "features": [],
                            "api_version": [12, 0],
                            "minimum_feature_level": [11, 0],
                            "shader_model": [6, 0],
                            "root_signature_version": [1, 0],
                        },
                        "metal": {
                            "backend": "metal",
                            "features": [],
                            "apple_platform": "macos",
                            "msl_version": [2, 4],
                            "minimum_os_version": [11, 0],
                        },
                    }.get(target)
                    if expected_requirements is None:
                        self.assertNotIn("runtime_requirements", document)
                    else:
                        self.assertEqual(document["runtime_requirements"], expected_requirements)
                    self.assertFalse((output / "pipeline.bundle").exists())
                    descriptors = [value["artifact"] for value in document["stage_artifacts"].values()]
                    if target == "metal":
                        for stage_record in document["stage_artifacts"].values():
                            slots = stage_record["reflection"]["implementation"]["metadata"]["resource_slots"]
                            self.assertEqual(len(slots), 1)
                            self.assertEqual(slots[0]["argument_buffer_index"], 0)
                            self.assertEqual(slots[0]["member_id"], 0)
                    self.assertEqual(len(descriptors), len(stages))
                    self.assertEqual(
                        len(list((output / "artifacts").iterdir())),
                        len(stages),
                    )
                    self.assertCountEqual(
                        [Path(row["path"]).name[64:] for row in descriptors],
                        extensions,
                    )
                    for descriptor in descriptors:
                        self.assertEqual(descriptor["storage"], "external")
                        self.assertEqual(descriptor["format"], artifact_format)
                        self.assertNotIn("data", descriptor)
                        artifact = (output / descriptor["path"]).read_bytes()
                        self.assertEqual(descriptor["size"], len(artifact))
                        self.assertEqual(
                            descriptor["sha256"],
                            hashlib.sha256(artifact).hexdigest(),
                        )
                    unhashed = dict(document)
                    content_hash = unhashed.pop("content_hash")
                    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                        "utf-8"
                    )
                    self.assertEqual(content_hash, hashlib.sha256(canonical).hexdigest())
                    self.assertEqual(
                        manifest_path.read_bytes(),
                        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n",
                    )

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
            self.assertEqual(bundle["type"], "program_bundle")
            program = bundle["variants"][0]["program"]
            self.assertEqual(
                set(program),
                {
                    "stages",
                    "parameters",
                    "storages",
                    "values",
                    "shape_symbols",
                    "shape_constraints",
                    "alias_preconditions",
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
            from vernon_dsl._program_assets.cooking import (
                _canonical_deployment,
                _compile_module_bundle_plan,
                _load_program_asset_declaration,
                _native_module,
                _native_target,
            )

            native = _native_module()
            pipeline = parse_python_program_asset(source, "asset")
            declaration = _load_program_asset_declaration(source, "asset")
            for target_name in ("cpu", "metal", "vulkan"):
                with self.subTest(target=target_name):
                    target = make_target_options(target_name)
                    retained: list[tuple[CompiledStage, object]] = []
                    plan = _compile_module_bundle_plan(
                        declaration,
                        pipeline,
                        target,
                        native,
                        _native_target(native, target_name),
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
                        for stage in plan.stages
                    }
                    canonical_program, artifact_system, stage_bindings = _canonical_deployment(plan, descriptors)
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
                        json.dumps(canonical_program, sort_keys=True, separators=(",", ":")).encode(),
                        json.dumps(artifact_system, sort_keys=True, separators=(",", ":")).encode(),
                        str(runtime_root),
                        stage_bindings,
                        compiled_stages,
                    )
                    source_array = np.array([3.0], dtype=np.float32)
                    output_array = np.zeros(1, dtype=np.float32)
                    builder = loaded.invocation_builder()
                    if target_name == "cpu":
                        for parameter, array in zip(loaded.parameters, (source_array, output_array), strict=True):
                            builder.host_tensor(parameter.slot, array)
                    else:
                        assert host is not None
                        source_buffer = host.create_buffer(source_array.nbytes)
                        output_buffer = host.create_buffer(output_array.nbytes)
                        source_buffer.upload(source_array.tobytes())
                        output_buffer.upload(output_array.tobytes())
                        for parameter, buffer in zip(loaded.parameters, (source_buffer, output_buffer), strict=True):
                            builder.rhi_tensor(parameter.slot, buffer, parameter.access, [1], [4])
                    builder.grid(1, 1, 1).submit().wait()
                    if target_name != "cpu":
                        output_array = np.frombuffer(output_buffer.download(), dtype=np.float32)
                    np.testing.assert_array_equal(output_array, np.array([9.0], dtype=np.float32))
                    if target_name == "cpu":
                        artifact_id = next(iter(artifact_system["artifacts"]))
                        bad_program = copy.deepcopy(canonical_program)
                        bad_artifacts = copy.deepcopy(artifact_system)
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
                                json.dumps(bad_program, sort_keys=True, separators=(",", ":")).encode(),
                                json.dumps(bad_artifacts, sort_keys=True, separators=(",", ":")).encode(),
                                str(runtime_root),
                                stage_bindings,
                                compiled_stages,
                            )

                    else:
                        artifact_path = runtime_root / descriptors[next(iter(descriptors))]["path"]
                        original_bytes = artifact_path.read_bytes()
                        artifact_path.write_bytes(b"broken")
                        try:
                            with self.assertRaisesRegex(RuntimeError, "PROGRAM_BLOB_AUTHENTICATION"):
                                runtime.load_canonical_program(
                                    json.dumps(canonical_program, sort_keys=True, separators=(",", ":")).encode(),
                                    json.dumps(artifact_system, sort_keys=True, separators=(",", ":")).encode(),
                                    str(runtime_root),
                                    stage_bindings,
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
    program=(vertex_main, fragment_main),
    variants=((),),
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "CPU pipeline bundles support"):
                cook_program_asset(
                    program_asset=f"{source}:asset",
                    output=root / "output",
                    target="cpu",
                )

    def test_gpu_no_tape_vjp_cooks_compute_profiles(self) -> None:
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
            self.assertEqual(bundle["autodiff"]["profiles"][0]["residual_storage"], "none")
            self.assertNotIn("execution", bundle["variants"][0])

    def test_gpu_captured_tape_vjp_cooks_static_and_dynamic_profiles(self) -> None:
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        from vernon_dsl._program_assets.cooking import _native_module

        native = _native_module()
        source = Path(__file__).parents[2] / "source" / "tests" / "fixtures" / "autodiff_gpu_tape_asset.py"
        targets = {
            "cuda": native.Target.CUDA,
            "vulkan": native.Target.VULKAN,
            "directx": native.Target.DIRECTX,
            "metal": native.Target.METAL,
            "opengl": native.Target.OPENGL,
        }
        for descriptor, expected_storage in (
            ("static_asset", "static"),
            ("dynamic_asset", "dynamic"),
        ):
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
                    profile = bundle["autodiff"]["profiles"][0]
                    self.assertEqual(profile["residual_storage"], expected_storage)
                    self.assertGreater(profile["static_tape_bytes_hint"], 0)
                    self.assertNotIn("execution", bundle["variants"][0])

    def test_four_variants_share_unchanged_fragment(self) -> None:
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
            self.assertEqual(len(bundle["variants"]), 4)
            self.assertEqual(len(bundle["stage_artifacts"]), 5)
            fragment_ids = {variant["program"]["fragment"] for variant in bundle["variants"]}
            vertex_ids = {variant["program"]["vertex"] for variant in bundle["variants"]}
            self.assertEqual(len(fragment_ids), 1)
            self.assertEqual(len(vertex_ids), 4)
            combined = next(variant for variant in bundle["variants"] if variant["key"] == ["INSTANCE", "SKIN"])
            vertex_record = bundle["stage_artifacts"][combined["program"]["vertex"]]
            interface = next(
                entry for entry in vertex_record["reflection"]["entries"] if entry["name"] == vertex_record["entry"]
            )["arguments"]
            locations = [value["vernon.location"] for value in interface if "vernon.location" in value]
            self.assertEqual(locations, [0, 1, 5, 6])
            runtime_bundle = bundle
            self.assertEqual(runtime_bundle["pipeline_version"], PIPELINE_VERSION)
            self.assertEqual(runtime_bundle["type"], "pipeline")
            self.assertEqual(runtime_bundle["id"], "shaders/variant_mesh")
            self.assertEqual(len(runtime_bundle["variants"]), 4)
            combined_runtime = next(
                variant for variant in runtime_bundle["variants"] if variant["key"] == ["INSTANCE", "SKIN"]
            )
            self.assertEqual(set(combined_runtime["program"]), {"vertex", "fragment"})
            slots = {parameter["name"]: parameter["slot"] for parameter in combined_runtime["parameters"]}
            self.assertEqual(sorted(slots.values()), list(range(len(slots))))
            self.assertTrue(
                all(
                    stage["artifact"]["storage"] == "external" and "data" not in stage["artifact"]
                    for stage in runtime_bundle["stage_artifacts"].values()
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
                self.assertEqual(
                    {stage["stage"] for stage in document["stage_artifacts"].values()},
                    stages,
                )
                if target in {"cuda", "cpu"}:
                    parameters = {parameter["name"]: parameter for parameter in document["variants"][0]["parameters"]}
                    self.assertEqual(parameters["values"]["shape"], [0])
                if target == "opengl":
                    self.assertEqual(len({variant["program"]["vertex"] for variant in document["variants"]}), 2)
                    self.assertEqual(len({variant["program"]["fragment"] for variant in document["variants"]}), 1)
                    repeated = Path(directory) / "opengl_repeated"
                    repeated_manifest = cook_program_asset(
                        program_asset=f"{source}:{name}",
                        output=repeated,
                        target=target,
                    )
                    repeated_document = json.loads(repeated_manifest.read_text(encoding="utf-8"))
                    self.assertEqual(document, repeated_document)
                for stage in document["stage_artifacts"].values():
                    artifact = stage["artifact"]
                    self.assertEqual(artifact["storage"], "external")
                    data = (output / artifact["path"]).read_bytes()
                    self.assertEqual(len(data), artifact["size"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), artifact["sha256"])

    def test_cook_only_source_targets_support_graphics_and_compute(self) -> None:
        root = Path(__file__).parents[2]
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        from vernon_dsl._program_assets.cooking import _native_module

        native = _native_module()
        assets = (
            (root / "python" / "tests" / "cube_map_shader.py", "cube_map_asset", {"vertex", "fragment"}),
            (root / "python" / "tests" / "program_asset_fixture.py", "scale_asset", {"compute"}),
        )
        targets = {
            "metal": ("msl", ".metal", {"platform": "macos"}),
            "directx": ("dxil", ".dxil", {"shader_model": 60}),
        }
        with tempfile.TemporaryDirectory() as directory:
            for source, name, stages in assets:
                for target, (artifact_format, extension, target_options) in targets.items():
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
                        self.assertEqual(
                            {stage["stage"] for stage in document["stage_artifacts"].values()},
                            stages,
                        )
                        if name == "cube_map_asset":
                            uniform_parameters = [
                                (parameter["name"], use)
                                for parameter in document["variants"][0]["parameters"]
                                for use in parameter["uses"]
                                if use["interface"] == "uniform"
                            ]
                            self.assertEqual(
                                {name for name, _ in uniform_parameters},
                                {"projection", "view", "model"},
                            )
                            self.assertTrue(all(use["uniform_name"] for _, use in uniform_parameters))
                            self.assertTrue(all(not name.endswith("._m0") for name, _ in uniform_parameters))
                        for stage in document["stage_artifacts"].values():
                            if target == "metal":
                                slots = stage["reflection"]["implementation"]["metadata"]["resource_slots"]
                                self.assertTrue(slots)
                                self.assertTrue(all(slot["entry_point"] == stage["entry"] for slot in slots))
                            artifact = stage["artifact"]
                            self.assertEqual(artifact["format"], artifact_format)
                            self.assertTrue(artifact["path"].endswith(extension))
                            data = (output / artifact["path"]).read_bytes()
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
            for stage in runtime_bundle["stage_artifacts"].values():
                self.assertNotIn("source", stage)
                artifact = stage["artifact"]
                self.assertEqual(artifact["format"], "spirv")
                self.assertEqual(artifact["storage"], "external")
                self.assertNotIn("encoding", artifact)
                self.assertNotIn("data", artifact)
                decoded = (manifest.parent / artifact["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(decoded).hexdigest(), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
