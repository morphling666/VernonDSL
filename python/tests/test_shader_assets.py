from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vernon_dsl.shader_assets import (ShaderAssetError, cook_shader_pipeline,
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
                self.assertEqual(hashlib.sha256(decoded).hexdigest(),
                                 artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
