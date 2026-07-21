from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from vernon_dsl.shader_assets import (ShaderAssetError, cook_shader_pipeline,
                                      encode_runtime_stage,
                                      parse_shader_pipeline_manifest)


class ShaderAssetManifestTests(unittest.TestCase):

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

    def test_cuda_pipeline_stage_embeds_ptx_and_reflection(self) -> None:
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
        self.assertEqual(encoded["source"], ".version 8.0\n")
        self.assertEqual(encoded["reflection"], reflection)
        self.assertNotIn("artifact", encoded)

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
            self.assertEqual(stage["artifact"]["path"], f"stages/{digest}.dll")
            self.assertEqual((output / stage["artifact"]["path"]).read_bytes(),
                             native_library)
            runtime_bundle = json.loads(
                (output / "pipeline.bundle").read_text(encoding="utf-8"))
            runtime_stage = next(
                iter(runtime_bundle["stage_artifacts"].values()))
            self.assertEqual(runtime_stage["artifact"], stage["artifact"])
            self.assertNotIn("source", runtime_stage)

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
        compiler = root / "build" / "source" / "Release" / "vernon-compile.exe"
        if not compiler.is_file():
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
                variant["stages"]["fragment"]
                for variant in bundle["variants"]
            }
            vertex_ids = {
                variant["stages"]["vertex"]
                for variant in bundle["variants"]
            }
            self.assertEqual(len(fragment_ids), 1)
            self.assertEqual(len(vertex_ids), 4)
            combined = next(variant for variant in bundle["variants"]
                            if variant["key"] == ["INSTANCE", "SKIN"])
            interface = bundle["stage_artifacts"][
                combined["stages"]["vertex"]]["interface"]["arguments"]
            locations = [
                value["vernon.location"] for value in interface
                if "vernon.location" in value
            ]
            self.assertEqual(locations, [0, 1, 5, 6])
            runtime_bundle = json.loads(
                (Path(directory) /
                 "pipeline.bundle").read_text(encoding="utf-8"))
            self.assertEqual(runtime_bundle["pipeline_bundle_schema_version"],
                             1)
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
                all("source" in stage
                    for stage in runtime_bundle["stage_artifacts"].values()))

    def test_vulkan_pipeline_bundle_embeds_verified_spirv(self) -> None:
        root = Path(__file__).parents[2]
        compiler = root / "build" / "source" / "Release" / "vernon-compile.exe"
        if not compiler.is_file():
            self.skipTest("native Vernon compiler is not built")
        with tempfile.TemporaryDirectory() as directory:
            cook_shader_pipeline(
                pipeline_manifest=root / "examples" /
                "variant_mesh.shader-pipeline.json",
                asset_root=root / "examples",
                compiler=compiler,
                output=directory,
                target="vulkan",
            )
            runtime_bundle = json.loads(
                (Path(directory) /
                 "pipeline.bundle").read_text(encoding="utf-8"))
            for stage in runtime_bundle["stage_artifacts"].values():
                self.assertNotIn("source", stage)
                artifact = stage["artifact"]
                self.assertEqual(artifact["format"], "spirv")
                self.assertEqual(artifact["encoding"], "base64")
                decoded = base64.b64decode(artifact["data"], validate=True)
                self.assertEqual(
                    hashlib.sha256(decoded).hexdigest(), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
