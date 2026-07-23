from __future__ import annotations

import hashlib
import json
import unittest

from vernon_dsl.pipeline_compile import (
    CompiledArtifact,
    CompiledStage,
    PipelineCompileError,
    TargetOptions,
    build_bundle_plan,
    canonical_json,
    external_parameters,
    inline_artifact_descriptor,
    materialize_bundle,
    merge_parameter_uses,
    parse_reflection_json,
    select_artifact,
    select_entry,
    serialize_bundle,
    validate_graphics_interfaces,
)


def _stage(stage: str, artifact: bytes, interface: dict[str, object]) -> CompiledStage:
    entry = f"{stage}_main"
    reflection = {
        "module_hash": "module-hash",
        "dependencies": ["shared.py"],
        "entries": [
            {
                "name": entry,
                "stage": stage,
                **interface,
            }
        ],
        "artifacts": [
            {
                "entry_point": entry,
                "stage": stage,
                "format": "glsl",
                "filename": f"{stage}.glsl",
            }
        ],
    }
    return CompiledStage(
        "module",
        '{"id":"module"}',
        entry,
        stage,
        TargetOptions("opengl", {"glsl_version": 330}),
        reflection,
        reflection["entries"][0],
        CompiledArtifact("glsl", artifact, f"{stage}.glsl"),
    )


class PipelineCompileTests(unittest.TestCase):
    def test_generated_sampler_and_resolution_are_internal(self) -> None:
        records = {
            "fragment": {
                "entry": "fragment_main",
                "target": "opengl",
                "interface": {
                    "arguments": [
                        {
                            "index": 0,
                            "kind": "texture",
                            "type": "!vernon.texture<2d, f32>",
                            "dtype": "f32",
                            "dimension": "2d",
                            "vernon.source_name": "image",
                            "vernon.interface": "resource",
                            "vernon.set": 0,
                            "vernon.binding": 3,
                        },
                        {
                            "index": 1,
                            "kind": "sampler",
                            "type": "!vernon.sampler",
                            "vernon.source_name": "__image_sampler",
                            "vernon.interface": "resource",
                            "vernon.implicit": "sampler",
                            "vernon.implicit_texture": "image",
                            "sampled_texture_bindings": [
                                {
                                    "set": 0,
                                    "binding": 3,
                                }
                            ],
                        },
                        {
                            "index": 2,
                            "kind": "scalar",
                            "type": "tensor<2xf32>",
                            "vernon.source_name": "__resolution",
                            "vernon.interface": "uniform",
                            "vernon.implicit": "resolution",
                        },
                    ],
                },
            },
        }
        external = external_parameters(records)
        self.assertEqual(set(external), {"image"})
        fragment = _stage("fragment", b"fragment", records["fragment"]["interface"])
        vertex = _stage(
            "vertex",
            b"vertex",
            {
                "arguments": [],
                "results": [],
            },
        )
        plan = build_bundle_plan(
            "pipeline",
            fragment.target,
            (),
            [
                (
                    (),
                    {
                        "vertex": vertex,
                        "fragment": fragment,
                    },
                )
            ],
        )
        variant = plan.variants[0].to_dict()
        self.assertEqual([row["name"] for row in variant["parameters"]], ["image"])
        self.assertEqual(
            [(row["source"], row.get("system_value")) for row in variant["internal_parameters"]],
            [("implicit_sampler", None), ("system_value", "resolution")],
        )

        unpaired = json.loads(json.dumps(records))
        del unpaired["fragment"]["interface"]["arguments"][1]["sampled_texture_bindings"]
        unpaired_fragment = _stage("fragment", b"fragment", unpaired["fragment"]["interface"])
        with self.assertRaisesRegex(PipelineCompileError, "exactly one sampled texture binding"):
            build_bundle_plan(
                "pipeline",
                fragment.target,
                (),
                [
                    (
                        (),
                        {
                            "vertex": vertex,
                            "fragment": unpaired_fragment,
                        },
                    )
                ],
            )

        legacy = json.loads(json.dumps(records))
        sampler = legacy["fragment"]["interface"]["arguments"][1]
        del sampler["vernon.implicit"]
        sampler["vernon.compiler_generated"] = True
        with self.assertRaisesRegex(PipelineCompileError, "legacy compiler-generated"):
            external_parameters(legacy)

    def test_explicit_sampler_remains_external(self) -> None:
        records = {
            "fragment": {
                "entry": "fragment_main",
                "target": "vulkan",
                "interface": {
                    "arguments": [
                        {
                            "index": 0,
                            "kind": "sampler",
                            "type": "!vernon.sampler",
                            "vernon.source_name": "linear_sampler",
                            "vernon.interface": "resource",
                            "sampled_texture_bindings": [
                                {
                                    "set": 0,
                                    "binding": 1,
                                }
                            ],
                        }
                    ],
                },
            },
        }
        self.assertEqual(set(external_parameters(records)), {"linear_sampler"})

    def test_descriptor_bound_uniform_keeps_block_member_name(self) -> None:
        records = {
            "vertex": {
                "entry": "vertex_main",
                "target": "opengl",
                "interface": {
                    "arguments": [
                        {
                            "index": 0,
                            "type": "tensor<4x4xf32>",
                            "vernon.source_name": "material",
                            "vernon.interface": "uniform",
                            "vernon.set": 0,
                            "vernon.binding": 2,
                        }
                    ],
                },
            },
        }
        uses = external_parameters(records)["material"]
        self.assertEqual(uses[0]["uniform_name"], "material._m0")

    def test_parameter_merge_and_slot_layout_are_exact(self) -> None:
        uses = [
            {
                "stage": "compute",
                "entry": "compute_main",
                "index": 0,
                "kind": "tensor",
                "type": "tensor<4xf32>",
                "dtype": "f32",
                "shape": [4],
                "interface": "storage",
                "access": "write",
            },
            {
                "stage": "vertex",
                "entry": "vertex_main",
                "index": 1,
                "kind": "tensor",
                "type": "tensor<4xf32>",
                "dtype": "f32",
                "shape": [4],
                "interface": "input",
                "access": "read",
            },
        ]
        self.assertEqual(
            merge_parameter_uses("positions", uses),
            {
                "name": "positions",
                "kind": "tensor",
                "type": "tensor<4xf32>",
                "dtype": "f32",
                "shape": [4],
                "access": "read_write",
                "uses": uses,
            },
        )

        vertex = _stage(
            "vertex",
            b"vertex",
            {
                "arguments": [
                    {
                        "index": 0,
                        "kind": "tensor",
                        "type": "tensor<2xf32>",
                        "vernon.source_name": "z_position",
                        "vernon.interface": "input",
                        "vernon.location": 0,
                    },
                    {
                        "index": 1,
                        "type": "f32",
                        "vernon.source_name": "alpha",
                        "vernon.interface": "uniform",
                    },
                ],
                "results": [
                    {
                        "type": "tensor<2xf32>",
                        "vernon.interface": "output",
                        "vernon.location": 0,
                    }
                ],
            },
        )
        fragment = _stage(
            "fragment",
            b"fragment",
            {
                "arguments": [
                    {
                        "index": 0,
                        "type": "tensor<2xf32>",
                        "vernon.source_name": "varying",
                        "vernon.interface": "input",
                        "vernon.location": 0,
                        "vernon.varying": True,
                    }
                ],
                "results": [
                    {
                        "type": "tensor<4xf32>",
                        "vernon.source_name": "color",
                        "vernon.interface": "output",
                        "vernon.location": 0,
                    }
                ],
            },
        )
        plan = build_bundle_plan(
            "pipeline",
            vertex.target,
            (),
            [
                (
                    (),
                    {
                        "vertex": vertex,
                        "fragment": fragment,
                    },
                )
            ],
        )
        parameters = plan.variants[0].to_dict()["parameters"]
        self.assertEqual([(row["name"], row["slot"]) for row in parameters], [("alpha", 0), ("z_position", 1)])
        self.assertEqual(
            parameters[0]["uses"][0]["uniform_name"],
            "alpha",
        )

    def test_variant_steps_outputs_and_graphics_interface_are_exact(self) -> None:
        compute = _stage(
            "compute",
            b"compute",
            {
                "arguments": [],
                "results": [],
            },
        )
        vertex = _stage(
            "vertex",
            b"vertex",
            {
                "arguments": [],
                "results": [
                    {
                        "type": "tensor<3xf32>",
                        "vernon.interface": "output",
                        "vernon.location": 2,
                    }
                ],
            },
        )
        fragment = _stage(
            "fragment",
            b"fragment",
            {
                "arguments": [
                    {
                        "type": "tensor<3xf32>",
                        "vernon.interface": "input",
                        "vernon.location": 2,
                        "vernon.varying": True,
                    }
                ],
                "results": [
                    {
                        "type": "tensor<4xf32>",
                        "vernon.interface": "output",
                        "vernon.location": 0,
                    }
                ],
            },
        )
        plan = build_bundle_plan(
            "pipeline",
            vertex.target,
            ("B", "A", "A"),
            [
                (
                    ("A",),
                    {
                        "compute": compute,
                        "vertex": vertex,
                        "fragment": fragment,
                    },
                )
            ],
        )
        self.assertEqual(plan.features, ("A", "B"))
        self.assertEqual(
            plan.variants[0].to_dict(),
            {
                "key": ["A"],
                "parameters": [],
                "outputs": [
                    {
                        "name": "output_0",
                        "kind": "texture",
                        "dtype": "f32",
                        "shape": [4],
                        "access": "write",
                        "location": 0,
                        "type": "tensor<4xf32>",
                    }
                ],
                "steps": [
                    {
                        "kind": "dispatch",
                        "stage": compute.id,
                    },
                    {
                        "kind": "barrier",
                        "source": "compute_write",
                        "destination": "vertex_read",
                    },
                    {
                        "kind": "draw",
                        "vertex": vertex.id,
                        "fragment": fragment.id,
                    },
                ],
            },
        )

    def test_stage_identity_and_hashing_are_deterministic(self) -> None:
        stage = _stage(
            "vertex",
            b"same",
            {
                "arguments": [],
                "results": [],
            },
        )
        repeated = _stage(
            "vertex",
            b"same",
            {
                "arguments": [],
                "results": [],
            },
        )
        changed = _stage(
            "vertex",
            b"different",
            {
                "arguments": [],
                "results": [],
            },
        )
        expected_id = hashlib.sha256(canonical_json(stage.identity).encode("utf-8")).hexdigest()
        self.assertEqual(stage.id, expected_id)
        self.assertEqual(stage.id, repeated.id)
        self.assertNotEqual(stage.id, changed.id)

        plan = build_bundle_plan("pipeline", stage.target, (), [((), {"compute": stage})])
        inline = materialize_bundle(plan, {stage.id: inline_artifact_descriptor(stage.artifact)})
        encoded = serialize_bundle(inline)
        self.assertEqual(encoded, serialize_bundle(inline))
        document = json.loads(encoded)
        unhashed = dict(document)
        digest = unhashed.pop("content_hash")
        self.assertEqual(
            digest,
            hashlib.sha256(canonical_json(unhashed).encode("utf-8")).hexdigest(),
        )

    def test_inline_and_external_storage_preserve_logical_plan(self) -> None:
        stage = _stage(
            "compute",
            b"artifact",
            {
                "arguments": [],
                "results": [],
            },
        )
        plan = build_bundle_plan("pipeline", stage.target, ("FEATURE",), [(("FEATURE",), {"compute": stage})])
        inline = materialize_bundle(plan, {stage.id: inline_artifact_descriptor(stage.artifact)})
        external = materialize_bundle(
            plan,
            {
                stage.id: {
                    "format": "glsl",
                    "storage": "external",
                    "path": f"artifacts/{stage.artifact.sha256}.glsl",
                    "size": len(stage.artifact.data),
                    "sha256": stage.artifact.sha256,
                }
            },
        )
        self.assertEqual(set(inline["stage_artifacts"]), set(external["stage_artifacts"]))
        self.assertEqual(inline["variants"], external["variants"])
        inline_record = dict(inline["stage_artifacts"][stage.id])
        external_record = dict(external["stage_artifacts"][stage.id])
        inline_record.pop("artifact")
        external_record.pop("artifact")
        self.assertEqual(inline_record, external_record)

    def test_reflection_selection_and_error_cases(self) -> None:
        reflection = parse_reflection_json(
            '{"entries":[{"name":"main","stage":"compute"}],'
            '"artifacts":[{"entry_point":"main","stage":"compute",'
            '"format":"ptx","filename":"main.ptx"}]}'
        )
        self.assertEqual(select_entry(reflection, "main")["stage"], "compute")
        self.assertEqual(select_artifact(reflection, "main", "compute")["filename"], "main.ptx")
        with self.assertRaisesRegex(PipelineCompileError, "exactly one"):
            select_entry({"entries": []}, "missing")
        with self.assertRaisesRegex(PipelineCompileError, "incompatible"):
            merge_parameter_uses(
                "value",
                [
                    {
                        "kind": "tensor",
                        "stage": "vertex",
                        "interface": "input",
                        "type": "f32",
                        "dtype": "f32",
                        "shape": [],
                    },
                    {
                        "kind": "tensor",
                        "stage": "fragment",
                        "interface": "input",
                        "type": "i32",
                        "dtype": "i32",
                        "shape": [],
                    },
                ],
            )
        with self.assertRaisesRegex(PipelineCompileError, "mismatch"):
            validate_graphics_interfaces(
                {
                    "interface": {
                        "results": [
                            {
                                "type": "f32",
                                "vernon.interface": "output",
                                "vernon.location": 0,
                            }
                        ]
                    }
                },
                {
                    "interface": {
                        "arguments": [
                            {
                                "type": "i32",
                                "vernon.interface": "input",
                                "vernon.location": 0,
                            }
                        ]
                    }
                },
            )
        stage = _stage(
            "compute",
            b"x",
            {
                "arguments": [],
                "results": [],
            },
        )
        plan = build_bundle_plan("pipeline", stage.target, (), [((), {"compute": stage})])
        with self.assertRaisesRegex(PipelineCompileError, "do not match planned"):
            materialize_bundle(plan, {})


if __name__ == "__main__":
    unittest.main()
