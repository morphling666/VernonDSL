from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from vernon_dsl.shader_assets import (ShaderAssetError, cook_shader_pipeline,
                                      encode_runtime_stage,
                                      parse_python_pipeline_asset,
                                      parse_shader_pipeline_manifest)


def _find_native_compiler(root: Path) -> Path | None:
    executable = "vernon-compile.exe" if os.name == "nt" else "vernon-compile"
    configured = os.environ.get("VERNON_COMPILER")
    candidates = [Path(configured).expanduser()] if configured else []
    source_build = root / "build" / "source"
    candidates.extend(source_build / configuration / executable
                      for configuration in ("Release", "Debug",
                                            "RelWithDebInfo", "MinSizeRel"))
    candidates.append(source_build / executable)
    discovered = shutil.which(executable)
    if discovered:
        candidates.append(Path(discovered))
    return next((path.resolve() for path in candidates if path.is_file()),
                None)


class ShaderAssetManifestTests(unittest.TestCase):

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

asset = vd.pipeline_asset(
    id="pipelines/static",
    vertex=vertex_main,
    fragment=fragment_main,
    variants=((), (FEATURE,)),
    targets={"opengl": {"glsl_version": 330}},
)
""",
                encoding="utf-8",
            )
            descriptor = parse_python_pipeline_asset(source, "asset")
            self.assertEqual(descriptor.id, "pipelines/static")
            self.assertEqual(descriptor.variants, ((), ("FEATURE", )))
            self.assertEqual(set(descriptor.stages), {"vertex", "fragment"})
            self.assertEqual(descriptor.targets["opengl"]["glsl_version"], 330)

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

asset = vd.pipeline_asset(
    id="pipelines/bad",
    compute=compute_main,
    variants=((ZED, ALPHA),),
    targets={"cpu": {}},
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ShaderAssetError, "not canonical"):
                parse_python_pipeline_asset(source, "asset")

    def test_python_pipeline_asset_enforces_variant_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "asset.py"
            features = "\n".join(
                f'F{index} = vd.feature("F{index:02}")'
                for index in range(17))
            variants = ", ".join(f"(F{index}, )" for index in range(17))
            source.write_text(
                f"""
import vernon_dsl as vd
{features}

@vd.kernel
def compute_main() -> None:
    pass

asset = vd.pipeline_asset(
    id="pipelines/too_many",
    compute=compute_main,
    variants=({variants},),
    targets={{"cpu": {{}}}},
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ShaderAssetError, "variant cap 16"):
                parse_python_pipeline_asset(source, "asset")

    def test_pipeline_manifest_has_canonical_variants(self) -> None:
        root = Path(__file__).parents[2]
        pipeline = parse_shader_pipeline_manifest(
            root / "examples" / "variant_mesh.shader-pipeline.json")
        self.assertEqual(pipeline.id, "shaders/variant_mesh")
        self.assertEqual(
            pipeline.variants,
            ((), ("INSTANCE", ), ("SKIN", ), ("INSTANCE", "SKIN")),
        )

    def test_variant_cap_and_duplicate_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.shader-pipeline.json"
            path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "type": "shader_pipeline",
                    "id": "bad",
                    "stages": {
                        "compute": {
                            "module": "module",
                            "entry": "main"
                        }
                    },
                    "variants": {
                        "include": [[], []],
                        "max_variants": 1
                    },
                    "targets": {
                        "opengl": {}
                    },
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ShaderAssetError):
                parse_shader_pipeline_manifest(path)


class ShaderAssetCookTests(unittest.TestCase):

    def test_interactive_glsl_and_spirv_descriptors_include_integrity(
            self) -> None:
        glsl = encode_runtime_stage({
            "format": "glsl",
            "stage": "vertex",
        }, b"void main() {}\n")
        self.assertEqual(glsl["artifact"]["encoding"], "utf8")
        self.assertEqual(glsl["artifact"]["data"], "void main() {}\n")
        self.assertEqual(glsl["artifact"]["size"], 15)

        spirv_bytes = b"\x03\x02\x23\x07"
        spirv = encode_runtime_stage({
            "format": "spirv",
            "stage": "compute",
        }, spirv_bytes)
        self.assertEqual(spirv["artifact"]["encoding"], "base64")
        self.assertEqual(
            base64.b64decode(spirv["artifact"]["data"], validate=True),
            spirv_bytes,
        )
        self.assertEqual(spirv["artifact"]["size"], len(spirv_bytes))

    def test_interactive_cuda_stage_uses_inline_ptx_descriptor(self) -> None:
        reflection = {
            "gpu_launch_abi_version": 1,
            "entries": [{
                "name": "scale",
                "stage": "compute"
            }],
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

    def test_cpu_pipeline_copies_content_addressed_native_sidecar(
            self) -> None:
        native_library = b"mock native library"
        digest = hashlib.sha256(native_library).hexdigest()
        reflection = {
            "schema_version":
            2,
            "module_hash":
            "module",
            "dependencies": [],
            "entries": [{
                "name": "scale",
                "symbol": "__vernon_cpu_scale",
                "stage": "compute",
                "arguments": [],
                "results": [],
                "workgroup_size": [1, 1, 1],
            }],
        }
        commands: list[list[str]] = []

        def run_compiler(command: list[str], **_: object) -> SimpleNamespace:
            commands.append(command)
            bundle = Path(command[command.index("--compute-bundle") + 1])
            bundle.mkdir(parents=True)
            (bundle / "compute.dll").write_bytes(native_library)
            (bundle / "compute.json").write_text(
                json.dumps({
                    "schema_version": 2,
                    "compiler_version": "0.1.0",
                    "cpu_invocation_abi_version": 1,
                    "target": "cpu",
                    "operating_system": "windows",
                    "architecture": "x86_64",
                    "entry": "scale",
                    "symbol": "__vernon_cpu_scale",
                    "artifact": "compute.dll",
                    "artifact_format": "native_library",
                    "artifact_size": len(native_library),
                    "artifact_sha256": digest,
                    "reflection": reflection,
                }),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "kernel.py"
            source.write_text("def scale():\n    pass\n", encoding="utf-8")
            (root / "kernel.shader-module.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "type": "shader_module",
                    "id": "module",
                    "source": "kernel.py",
                }),
                encoding="utf-8",
            )
            pipeline = root / "kernel.shader-pipeline.json"
            pipeline.write_text(
                json.dumps({
                    "schema_version": 1,
                    "type": "shader_pipeline",
                    "id": "pipelines/cpu",
                    "stages": {
                        "compute": {
                            "module": "module",
                            "entry": "scale",
                        }
                    },
                    "variants": {
                        "include": [[]],
                        "max_variants": 1,
                    },
                    "targets": {
                        "cpu": {}
                    },
                }),
                encoding="utf-8",
            )
            output = root / "cooked"
            with mock.patch(
                    "vernon_dsl.shader_assets.compile_file",
                    return_value="module {}"), mock.patch(
                        "vernon_dsl.shader_assets.load_project",
                        return_value=SimpleNamespace(
                            features=set())), mock.patch(
                                "vernon_dsl.shader_assets.subprocess.run",
                                side_effect=run_compiler):
                manifest_path = cook_shader_pipeline(
                    pipeline_manifest=pipeline,
                    asset_root=root,
                    compiler=Path(__file__),
                    output=output,
                    target="cpu",
                )

            self.assertIn("--compute-bundle", commands[0])
            self.assertNotIn("--output-dir", commands[0])
            bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
            stage = next(iter(bundle["stage_artifacts"].values()))
            self.assertEqual(stage["format"], "native_library")
            self.assertEqual(stage["symbol"], "__vernon_cpu_scale")
            self.assertEqual(stage["operating_system"], "windows")
            self.assertEqual(stage["architecture"], "x86_64")
            self.assertEqual(stage["cpu_invocation_abi_version"], 1)
            self.assertEqual(stage["artifact"]["sha256"], digest)
            self.assertEqual(stage["artifact"]["size"], len(native_library))
            self.assertEqual(stage["artifact"]["storage"], "external")
            self.assertEqual(stage["artifact"]["format"], "native_library")
            self.assertEqual(stage["artifact"]["path"],
                             f"artifacts/{digest}.dll")
            self.assertEqual((output / stage["artifact"]["path"]).read_bytes(),
                             native_library)
            self.assertEqual(manifest_path.name, "cooked.pipeline.json")
            self.assertFalse((output / "pipeline.bundle").exists())
            self.assertFalse((output / "shader.json").exists())

    def test_mocked_gpu_targets_emit_external_deduplicated_artifacts(
            self) -> None:
        cases = {
            "cuda": (("compute", ), "ptx", (".ptx", )),
            "opengl": (("vertex", "fragment"), "glsl",
                       (".vert.glsl", ".frag.glsl")),
            "opengles": (("vertex", "fragment"), "gles",
                         (".vert.gles", ".frag.gles")),
            "vulkan": (("vertex", "fragment"), "spirv", (".spv", ".spv")),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target, (stages, artifact_format,
                         extensions) in cases.items():
                with self.subTest(target=target):
                    case_root = root / target
                    case_root.mkdir()
                    source = case_root / "shader.py"
                    source.write_text("def main():\n    pass\n",
                                      encoding="utf-8")
                    (case_root / "shader.shader-module.json").write_text(
                        json.dumps({
                            "schema_version": 1,
                            "type": "shader_module",
                            "id": "module",
                            "source": "shader.py",
                        }),
                        encoding="utf-8",
                    )
                    stage_manifest = {
                        stage: {
                            "module": "module",
                            "entry": f"{stage}_main",
                        }
                        for stage in stages
                    }
                    pipeline = case_root / "asset.shader-pipeline.json"
                    pipeline.write_text(
                        json.dumps({
                            "schema_version": 1,
                            "type": "shader_pipeline",
                            "id": f"pipelines/{target}",
                            "stages": stage_manifest,
                            "variants": {
                                "include": [[], ["FEATURE"]],
                                "max_variants": 2,
                            },
                            "targets": {
                                target: {}
                            },
                        }),
                        encoding="utf-8",
                    )
                    compile_index = 0

                    def run_compiler(command: list[str],
                                     **_: object) -> SimpleNamespace:
                        nonlocal compile_index
                        stage = stages[compile_index % len(stages)]
                        compile_index += 1
                        artifact_directory = Path(
                            command[command.index("--output-dir") + 1])
                        artifact_directory.mkdir(parents=True)
                        extension = {
                            "glsl": ".glsl",
                            "gles": ".gles",
                            "ptx": ".ptx",
                            "spirv": ".spv",
                        }[artifact_format]
                        filename = f"{stage}{extension}"
                        artifact = ({
                            "vertex": b"vertex artifact\n",
                            "fragment": b"fragment artifact\n",
                            "compute": b"compute artifact\n",
                        }[stage] if artifact_format != "spirv" else
                                    b"\x03\x02\x23\x07" +
                                    stage.encode("ascii"))
                        (artifact_directory / filename).write_bytes(artifact)
                        (artifact_directory / "reflection.json").write_text(
                            json.dumps({
                                "module_hash":
                                "module",
                                "dependencies": [],
                                "entries": [{
                                    "name": f"{stage}_main",
                                    "stage": stage,
                                    "arguments": [],
                                    "results": [],
                                }],
                                "artifacts": [{
                                    "entry_point": f"{stage}_main",
                                    "stage": stage,
                                    "format": artifact_format,
                                    "filename": filename,
                                }],
                            }),
                            encoding="utf-8",
                        )
                        return SimpleNamespace(returncode=0,
                                               stdout="",
                                               stderr="")

                    output = case_root / f"{target}_asset"
                    with mock.patch(
                            "vernon_dsl.shader_assets.compile_file",
                            return_value="module {}"), mock.patch(
                                "vernon_dsl.shader_assets.load_project",
                                return_value=SimpleNamespace(
                                    features={"FEATURE"})), mock.patch(
                                        "vernon_dsl.shader_assets.subprocess.run",
                                        side_effect=run_compiler):
                        manifest_path = cook_shader_pipeline(
                            pipeline_manifest=pipeline,
                            asset_root=case_root,
                            compiler=Path(__file__),
                            output=output,
                            target=target,
                        )
                        first_manifest = manifest_path.read_bytes()
                        repeated_path = cook_shader_pipeline(
                            pipeline_manifest=pipeline,
                            asset_root=case_root,
                            compiler=Path(__file__),
                            output=output,
                            target=target,
                        )

                    self.assertEqual(manifest_path.name,
                                     f"{target}_asset.pipeline.json")
                    self.assertEqual(repeated_path, manifest_path)
                    self.assertEqual(manifest_path.read_bytes(),
                                     first_manifest)
                    document = json.loads(
                        manifest_path.read_text(encoding="utf-8"))
                    self.assertEqual(document["schema_version"], 2)
                    self.assertEqual(document["type"], "pipeline")
                    self.assertEqual(document["invocation_abi_version"], 1)
                    self.assertFalse((output / "pipeline.bundle").exists())
                    descriptors = [
                        value["artifact"]
                        for value in document["stage_artifacts"].values()
                    ]
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
                        self.assertEqual(descriptor["format"],
                                         artifact_format)
                        self.assertNotIn("data", descriptor)
                        artifact = (output /
                                    descriptor["path"]).read_bytes()
                        self.assertEqual(descriptor["size"], len(artifact))
                        self.assertEqual(
                            descriptor["sha256"],
                            hashlib.sha256(artifact).hexdigest(),
                        )
                    unhashed = dict(document)
                    content_hash = unhashed.pop("content_hash")
                    canonical = json.dumps(unhashed,
                                           sort_keys=True,
                                           separators=(",", ":"),
                                           ensure_ascii=False).encode("utf-8")
                    self.assertEqual(content_hash,
                                     hashlib.sha256(canonical).hexdigest())
                    self.assertEqual(
                        manifest_path.read_bytes(),
                        json.dumps(document,
                                   indent=2,
                                   sort_keys=True,
                                   ensure_ascii=False).encode("utf-8") + b"\n",
                    )
                    self.assertFalse((output / "shader.json").exists())

    def test_cpu_graphics_pipeline_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = root / "graphics.shader-pipeline.json"
            pipeline.write_text(
                json.dumps({
                    "schema_version": 1,
                    "type": "shader_pipeline",
                    "id": "graphics",
                    "stages": {
                        "vertex": {
                            "module": "missing",
                            "entry": "vertex"
                        },
                        "fragment": {
                            "module": "missing",
                            "entry": "fragment"
                        },
                    },
                    "variants": {
                        "include": [[]],
                        "max_variants": 1
                    },
                    "targets": {
                        "cpu": {}
                    },
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ShaderAssetError,
                                        "CPU pipeline bundles support"):
                cook_shader_pipeline(
                    pipeline_manifest=pipeline,
                    asset_root=root,
                    compiler=Path(__file__),
                    output=root / "output",
                    target="cpu",
                )

    def test_four_variants_share_unchanged_fragment(self) -> None:
        root = Path(__file__).parents[2]
        compiler = _find_native_compiler(root)
        if compiler is None:
            self.skipTest("native Vernon compiler is not built")
        with tempfile.TemporaryDirectory() as directory:
            manifest = cook_shader_pipeline(
                pipeline_manifest=root / "examples" /
                "variant_mesh.shader-pipeline.json",
                asset_root=root / "examples",
                compiler=compiler,
                output=directory,
            )
            bundle = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(bundle["features"], ["INSTANCE", "SKIN"])
            self.assertEqual(len(bundle["variants"]), 4)
            self.assertEqual(len(bundle["stage_artifacts"]), 5)
            fragment_ids = {
                variant["steps"][0]["fragment"]
                for variant in bundle["variants"]
            }
            vertex_ids = {
                variant["steps"][0]["vertex"]
                for variant in bundle["variants"]
            }
            self.assertEqual(len(fragment_ids), 1)
            self.assertEqual(len(vertex_ids), 4)
            combined = next(variant for variant in bundle["variants"]
                            if variant["key"] == ["INSTANCE", "SKIN"])
            interface = bundle["stage_artifacts"][
                combined["steps"][0]["vertex"]]["interface"]["arguments"]
            locations = [
                value["vernon.location"] for value in interface
                if "vernon.location" in value
            ]
            self.assertEqual(locations, [0, 1, 5, 6])
            runtime_bundle = bundle
            self.assertEqual(runtime_bundle["schema_version"], 2)
            self.assertEqual(runtime_bundle["type"], "pipeline")
            self.assertEqual(runtime_bundle["invocation_abi_version"], 1)
            self.assertEqual(runtime_bundle["id"], "shaders/variant_mesh")
            self.assertEqual(len(runtime_bundle["variants"]), 4)
            combined_runtime = next(variant
                                    for variant in runtime_bundle["variants"]
                                    if variant["key"] == ["INSTANCE", "SKIN"])
            self.assertEqual(combined_runtime["steps"][0]["kind"], "draw")
            slots = {
                parameter["name"]: parameter["slot"]
                for parameter in combined_runtime["parameters"]
            }
            self.assertEqual(sorted(slots.values()), list(range(len(slots))))
            self.assertTrue(
                all(stage["artifact"]["storage"] == "external"
                    and "data" not in stage["artifact"]
                    for stage in runtime_bundle["stage_artifacts"].values()))

    def test_python_descriptors_cook_all_runtime_backends(self) -> None:
        root = Path(__file__).parents[2]
        compiler = _find_native_compiler(root)
        if compiler is None:
            self.skipTest("native Vernon compiler is not built")
        source = root / "python" / "tests" / "pipeline_asset_fixture.py"
        cases = {
            "opengl": ("triangle_asset", "pipelines/triangle", {
                "vertex", "fragment"
            }),
            "cuda": ("scale_asset", "pipelines/scale", {
                "compute"
            }),
            "cpu": ("scale_asset", "pipelines/scale", {
                "compute"
            }),
        }
        with tempfile.TemporaryDirectory() as directory:
            for target, (name, asset_id, stages) in cases.items():
                output = Path(directory) / target
                manifest = cook_shader_pipeline(
                    pipeline_manifest=f"{source}:{name}",
                    asset_root=source.parent,
                    compiler=compiler,
                    output=output,
                    target=target,
                )
                document = json.loads(manifest.read_text(encoding="utf-8"))
                self.assertEqual(document["id"], asset_id)
                self.assertEqual(document["target"], target)
                self.assertEqual(
                    {
                        stage["stage"]
                        for stage in document["stage_artifacts"].values()
                    },
                    stages,
                )
                if target == "opengl":
                    self.assertEqual(
                        len({
                            variant["steps"][0]["vertex"]
                            for variant in document["variants"]
                        }), 2)
                    self.assertEqual(
                        len({
                            variant["steps"][0]["fragment"]
                            for variant in document["variants"]
                        }), 1)
                    repeated = Path(directory) / "opengl_repeated"
                    repeated_manifest = cook_shader_pipeline(
                        pipeline_manifest=f"{source}:{name}",
                        asset_root=source.parent,
                        compiler=compiler,
                        output=repeated,
                        target=target,
                    )
                    repeated_document = json.loads(
                        repeated_manifest.read_text(encoding="utf-8"))
                    self.assertEqual(document, repeated_document)
                for stage in document["stage_artifacts"].values():
                    artifact = stage["artifact"]
                    self.assertEqual(artifact["storage"], "external")
                    data = (output / artifact["path"]).read_bytes()
                    self.assertEqual(len(data), artifact["size"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(),
                                     artifact["sha256"])

    def test_vulkan_pipeline_bundle_embeds_verified_spirv(self) -> None:
        root = Path(__file__).parents[2]
        compiler = _find_native_compiler(root)
        if compiler is None:
            self.skipTest("native Vernon compiler is not built")
        with tempfile.TemporaryDirectory() as directory:
            manifest = cook_shader_pipeline(
                pipeline_manifest=root / "examples" /
                "variant_mesh.shader-pipeline.json",
                asset_root=root / "examples",
                compiler=compiler,
                output=directory,
                target="vulkan",
            )
            runtime_bundle = json.loads(
                manifest.read_text(encoding="utf-8"))
            for stage in runtime_bundle["stage_artifacts"].values():
                self.assertNotIn("source", stage)
                artifact = stage["artifact"]
                self.assertEqual(artifact["format"], "spirv")
                self.assertEqual(artifact["storage"], "external")
                self.assertNotIn("encoding", artifact)
                self.assertNotIn("data", artifact)
                decoded = (manifest.parent / artifact["path"]).read_bytes()
                self.assertEqual(
                    hashlib.sha256(decoded).hexdigest(), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
