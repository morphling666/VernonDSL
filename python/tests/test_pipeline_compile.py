from __future__ import annotations

import hashlib
import json
import struct
import unittest
from types import SimpleNamespace

from vernon_dsl.bundle import (
    CompiledArtifact,
    CompiledStage,
    MetalTargetOptions,
    OpenGLTargetOptions,
    PipelineCompileError,
    build_bundle_plan,
    canonical_json,
    compiled_stage_from_program,
    external_parameters,
    inline_artifact_descriptor,
    make_target_options,
    materialize_bundle,
    merge_parameter_uses,
    parse_reflection_json,
    select_artifact,
    select_entry,
    serialize_bundle,
    validate_graphics_interfaces,
)
from vernon_dsl.bundle.requirements import runtime_requirements


def _scalar_layout(dtype: str) -> dict[str, object]:
    sizes = {"bool": 1, "i32": 4, "u32": 4, "f16": 2, "f32": 4, "f64": 8}
    size = sizes[dtype]
    physical = "i32" if dtype == "u32" else dtype
    canonical = f"scalar({physical},{size},{size})|dtypes={dtype}"
    return {
        "logical_type": dtype,
        "byte_size": size,
        "alignment": size,
        "layout_hash": hashlib.sha256(canonical.encode()).hexdigest(),
        "leaves": [{"path": [], "dtype": dtype, "byte_offset": 0, "scalar_count": 1}],
    }


def _interface_plan(
    profile: str,
    size: int,
    alignment: int,
    byte_strides: list[int],
) -> dict[str, object]:
    kind = {
        "host_value": "cpu_call",
        "cuda_kernel_parameter": "kernel_parameter",
        "opengl_native_uniform": "native_uniform",
    }.get(profile, "byte_transport")
    root = {
        "kind": "scalar",
        "representation": "f32",
        "offset": 0,
        "size": size,
        "alignment": alignment,
        "shape": [],
        "byte_strides": [],
        "children": [],
    }
    if byte_strides:
        root = {
            "kind": "array",
            "representation": "",
            "offset": 0,
            "size": size,
            "alignment": alignment,
            "shape": [1] * len(byte_strides),
            "byte_strides": byte_strides,
            "children": [
                {
                    "kind": "scalar",
                    "representation": "f32",
                    "offset": 0,
                    "size": byte_strides[-1],
                    "alignment": min(alignment, byte_strides[-1]),
                    "shape": [],
                    "byte_strides": [],
                    "children": [],
                }
            ],
        }
    return {
        "kind": kind,
        "profile": profile,
        "canonical_layout_hash": "test-layout-hash",
        "root": root,
    }


def _physical_layouts(size: int, alignment: int, byte_strides: list[int]) -> dict[str, object]:
    return {
        profile: _interface_plan(profile, size, alignment, byte_strides)
        for profile in (
            "host_value",
            "cuda_kernel_parameter",
            "vulkan_std140_uniform_buffer",
            "vulkan_std430_storage_buffer",
            "vulkan_push_constant",
            "opengl_native_uniform",
            "directx_constant_buffer",
            "metal_constant_buffer",
        )
    }


def _stage(stage: str, artifact: bytes, interface: dict[str, object]) -> CompiledStage:
    if not artifact.startswith(b"#version"):
        artifact = b"#version 330\n" + artifact
    interface = json.loads(json.dumps(interface))
    for argument in interface.get("arguments", []):
        if (
            argument.get("kind") not in {"texture", "sampler"}
            and "vernon.builtin" not in argument
            and not argument.get("vernon.varying", False)
            and "element_layout" not in argument
        ):
            type_name = argument.get("type", "")
            dtype = argument.get("dtype") or str(type_name).removesuffix(">").split("x")[-1]
            if dtype in {"bool", "i32", "u32", "f16", "f32", "f64"}:
                argument["element_layout"] = _scalar_layout(dtype)
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
        OpenGLTargetOptions(version=330),
        reflection,
        reflection["entries"][0],
        CompiledArtifact("glsl", artifact, f"{stage}.glsl"),
    )


class PipelineCompileTests(unittest.TestCase):
    def test_runtime_requirements_are_derived_from_artifact_headers(self) -> None:
        def stage(
            target: str,
            data: bytes,
            *,
            required_features: list[str] | None = None,
            metadata: dict[str, object] | None = None,
            interface: dict[str, object] | None = None,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                target=SimpleNamespace(target=target, options={}),
                stage="compute",
                artifact=SimpleNamespace(data=data),
                reflection={"required_features": required_features or []},
                metadata=metadata or {},
                interface=interface or {},
            )

        gl = runtime_requirements(
            "opengl",
            [stage("opengl", b"#version 430\nvoid main() {}", required_features=["textures", "compute"])],
        )
        self.assertEqual(gl["api_version"], [4, 3])
        self.assertEqual(gl["glsl_version"], 430)
        self.assertEqual(gl["profile"], "core")
        self.assertEqual(gl["features"], ["compute", "textures"])

        spirv = struct.pack("<II", 0x07230203, 0x00010300)
        vulkan = runtime_requirements(
            "vulkan",
            [stage("vulkan", spirv, interface={"workgroup_size": [8, 4, 1]})],
        )
        self.assertEqual(vulkan["spirv_version"], [1, 3])
        self.assertEqual(vulkan["compute_workgroup_size"], [8, 4, 1])

        ptx = b".version 8.1\n.target sm_75\n.address_size 64\n"
        cuda = runtime_requirements("cuda", [stage("cuda", ptx)])
        self.assertEqual(cuda["ptx_version"], [8, 1])
        self.assertEqual(cuda["minimum_compute_capability"], [7, 5])
        self.assertEqual(cuda["address_size"], 64)

        cpu = runtime_requirements(
            "cpu",
            [
                stage(
                    "cpu",
                    b"object",
                    metadata={
                        "target_triple": "x86_64-pc-windows-msvc",
                        "object_format": "coff",
                    },
                )
            ],
        )
        self.assertEqual(cpu["target_triple"], "x86_64-pc-windows-msvc")
        self.assertEqual(cpu["object_format"], "coff")
        self.assertNotIn("invocation_abi_version", cpu)
        metal_stage = stage("metal", b"MSL")
        metal_stage.target.options = {"platform": "ios"}
        metal_stage.reflection = {
            "target": {
                "kind": "metal",
                "options": {"platform": "ios"},
                "output": {"language": "msl", "version": [2, 4], "minimum_os_version": [15, 0]},
            }
        }
        metal = runtime_requirements("metal", [metal_stage])
        self.assertEqual(metal["apple_platform"], "ios")
        self.assertEqual(metal["msl_version"], [2, 4])
        self.assertEqual(metal["minimum_os_version"], [15, 0])

    def test_runtime_requirement_parsers_reject_incomplete_artifacts(self) -> None:
        stage = SimpleNamespace(
            target=SimpleNamespace(target="cuda"),
            stage="compute",
            artifact=SimpleNamespace(data=b".version 8.0\n"),
            reflection={},
            metadata={},
            interface={},
        )
        with self.assertRaisesRegex(PipelineCompileError, "incomplete PTX header"):
            runtime_requirements("cuda", [stage])

        metal_stage = SimpleNamespace(
            target=SimpleNamespace(target="metal", options={"platform": "macos"}),
            stage="compute",
            artifact=SimpleNamespace(data=b"MSL"),
            reflection={},
            metadata={},
            interface={},
        )
        with self.assertRaisesRegex(PipelineCompileError, "no valid msl_version"):
            runtime_requirements("metal", [metal_stage])
        metal_stage.target.options = {}
        metal_stage.reflection = {
            "target": {
                "kind": "metal",
                "options": {},
                "output": {"language": "msl", "version": [2, 4], "minimum_os_version": [11, 0]},
            }
        }
        with self.assertRaisesRegex(PipelineCompileError, "requires apple_platform"):
            runtime_requirements("metal", [metal_stage])

    def test_compiled_stage_uses_reflected_target_options(self) -> None:
        program = SimpleNamespace(
            ok=True,
            diagnostics="",
            reflection=json.dumps(
                {
                    "target": {
                        "kind": "metal",
                        "options": {"platform": "ios"},
                        "output": {
                            "language": "msl",
                            "version": [2, 4],
                            "minimum_os_version": [15, 0],
                        },
                    },
                    "entries": [{"name": "main", "stage": "compute"}],
                    "artifacts": [
                        {
                            "entry_point": "main",
                            "stage": "compute",
                            "filename": "main.metal",
                            "format": "msl",
                        }
                    ],
                }
            ),
            artifacts=[("main.metal", b"kernel void main() {}")],
        )
        compiled = compiled_stage_from_program(
            program,
            module="module",
            module_manifest="manifest",
            entry="main",
            target=MetalTargetOptions(),
        )
        self.assertEqual(compiled.target.options["platform"], "ios")
        requirements = runtime_requirements("metal", [compiled])
        self.assertEqual(requirements["minimum_os_version"], [15, 0])

    def test_runtime_requirements_change_content_hash_but_not_stage_identity(self) -> None:
        stage = _stage("compute", b"#version 430\nvoid main() {}", {"workgroup_size": [1, 1, 1]})
        plan = build_bundle_plan("requirements/hash", stage.target, (), [((), {"compute": stage})])
        logical = plan.logical_dict()
        content_hash = hashlib.sha256(canonical_json(logical).encode("utf-8")).hexdigest()
        changed = json.loads(json.dumps(logical))
        changed["runtime_requirements"]["glsl_version"] = 440
        changed_hash = hashlib.sha256(canonical_json(changed).encode("utf-8")).hexdigest()

        self.assertNotEqual(content_hash, changed_hash)
        self.assertEqual(set(logical["stage_artifacts"]), {stage.id})
        self.assertEqual(set(changed["stage_artifacts"]), {stage.id})

    def test_manifest_v14_autodiff_schema_omits_planning_metadata(self) -> None:
        stage = _stage("compute", b"#version 430\nvoid main() {}", {"workgroup_size": [1, 1, 1]})
        profiles = {
            name: {
                "compute": stage.id,
                "inputs": [],
                "outputs": [],
            }
            for name in ("primal", "forward_with_tape", "backward")
        }
        plan = build_bundle_plan(
            "autodiff/schema",
            stage.target,
            ("INTERNAL_FEATURE",),
            [((), {"compute": stage})],
            {
                "kind": "vjp",
                "protocol": "dynamic_v2",
                "wrt": ["value"],
                "output_cotangents": ["output"],
                "gradient_policy": "explicit",
                "identity": "internal-transform",
            },
            {
                "identity": "internal-profiles",
                "tape_bytes": 64,
                "variants": [
                    {
                        "key": [],
                        "workgroup_size": [1, 1, 1],
                        "profiles": profiles,
                    }
                ],
            },
        )

        logical = plan.logical_dict()
        self.assertNotIn("features", logical)
        self.assertNotIn("program_transform", logical)
        self.assertNotIn("autodiff_profiles", logical)
        self.assertEqual(
            set(logical["autodiff"]),
            {"kind", "protocol", "wrt", "output_cotangents", "variants"},
        )
        self.assertEqual(
            set(logical["stage_artifacts"][stage.id]),
            {"stage", "entry", "reflection"},
        )

    def test_native_options_are_scoped_to_the_selected_target(self) -> None:
        self.assertEqual(OpenGLTargetOptions(version=330).native_options, {"options": {"version": 330}})
        self.assertEqual(
            make_target_options("cpu", {"processor": "generic", "features": ["+sse2"]}).native_options,
            {"options": {"processor": "generic", "features": "+sse2"}},
        )
        self.assertEqual(make_target_options("directx").native_options, {"options": {"shader_model": 60}})
        self.assertEqual(
            make_target_options("directx", {"shader_model": 60}).native_options,
            {"options": {"shader_model": 60}},
        )
        self.assertEqual(make_target_options("cuda").native_options, {"options": {}})
        self.assertEqual(MetalTargetOptions().native_options, {"options": {"platform": "macos"}})
        self.assertEqual(
            MetalTargetOptions(platform="ios").native_options,
            {"options": {"platform": "ios"}},
        )
        with self.assertRaisesRegex(PipelineCompileError, "invalid vulkan target options"):
            make_target_options("vulkan", {"processor": "generic"})
        with self.assertRaisesRegex(PipelineCompileError, "invalid metal target options"):
            make_target_options("metal", {"shader_model": 60})
        with self.assertRaisesRegex(PipelineCompileError, "shader model must be 6.0 or newer"):
            make_target_options("directx", {"shader_model": 55})
        with self.assertRaisesRegex(PipelineCompileError, "macos.*ios"):
            MetalTargetOptions(platform="tvos")

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
                            "element_layout": _scalar_layout("f32"),
                            "vernon.source_name": "__resolution",
                            "vernon.interface": "uniform",
                            "vernon.implicit": "resolution",
                            "value_transport": "push_constant",
                            "physical_layouts": _physical_layouts(8, 8, [4]),
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
        with self.assertRaisesRegex(PipelineCompileError, "no reflected sampled texture binding"):
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
                            "kind": "tensor_value",
                            "type": "tensor<4x4xf32>",
                            "element_layout": _scalar_layout("f32"),
                            "vernon.source_name": "material",
                            "vernon.interface": "uniform",
                            "vernon.set": 0,
                            "vernon.binding": 2,
                            "value_transport": "uniform_buffer",
                            "physical_layouts": _physical_layouts(64, 16, [16, 4]),
                        }
                    ],
                },
            },
        }
        uses = external_parameters(records)["material"]
        self.assertEqual(uses[0]["uniform_name"], "material._m0")
        self.assertEqual(
            uses[0]["interface_plan"],
            _interface_plan("vulkan_std140_uniform_buffer", 64, 16, [16, 4]),
        )

    def test_reflected_static_tensor_layout_is_normalized_for_runtime(self) -> None:
        records = {
            "fragment": {
                "entry": "fragment_main",
                "target": "vulkan",
                "interface": {
                    "arguments": [
                        {
                            "index": 0,
                            "type": "tensor<2x3x5xf32>",
                            "kind": "tensor_value",
                            "dtype": "f32",
                            "element_layout": _scalar_layout("f32"),
                            "shape": [2, 3, 5],
                            "vernon.source_name": "weights",
                            "vernon.interface": "uniform",
                            "vernon.set": 0,
                            "vernon.binding": 3,
                            "value_transport": "uniform_buffer",
                            "physical_layouts": _physical_layouts(120, 4, [60, 20, 4]),
                        }
                    ],
                },
            },
        }
        use = external_parameters(records)["weights"][0]
        self.assertEqual(
            use["interface_plan"],
            _interface_plan("vulkan_std140_uniform_buffer", 120, 4, [60, 20, 4]),
        )

    def test_reflected_aggregate_tensor_uses_explicit_storage_buffer_layout(self) -> None:
        element_layout = {
            "logical_type": "!vernon.struct<ComplexAggregateVertex>",
            "layout_hash": "aggregate-layout",
            "byte_size": 44,
            "alignment": 4,
            "leaves": [
                {"path": ["position"], "dtype": "f32", "scalar_count": 2, "byte_offset": 0},
                {"path": ["payload", "object_id"], "dtype": "i32", "scalar_count": 1, "byte_offset": 8},
            ],
        }
        records = {
            "vertex": {
                "entry": "vertex_main",
                "target": "vulkan",
                "interface": {
                    "arguments": [
                        {
                            "index": 1,
                            "type": "!vernon.tensor<2x3x4x!vernon.struct<ComplexAggregateVertex>>",
                            "kind": "tensor_value",
                            "element_layout": element_layout,
                            "shape": [2, 3, 4],
                            "vernon.source_name": "aggregate",
                            "vernon.interface": "uniform",
                            "vernon.set": 0,
                            "vernon.binding": 0,
                            "value_transport": "storage_buffer",
                            "physical_layouts": _physical_layouts(1056, 4, [528, 176, 44]),
                        }
                    ],
                },
            }
        }
        use = external_parameters(records)["aggregate"][0]
        self.assertEqual(
            use["interface_plan"],
            _interface_plan("vulkan_std430_storage_buffer", 1056, 4, [528, 176, 44]),
        )

    def test_compute_tensor_resource_is_normalized_to_storage(self) -> None:
        records = {
            "compute": {
                "entry": "compute_main",
                "target": "cpu",
                "interface": {
                    "arguments": [
                        {
                            "index": 0,
                            "kind": "tensor",
                            "type": '!vernon.tensor_view<f32, [-1, -1], "write", "device">',
                            "element_layout": _scalar_layout("f32"),
                            "source_shape": [-1, -1],
                            "tensor_view_descriptor": {
                                "rank": 2,
                                "offset_binding": 1,
                                "extent_bindings": [2, 3],
                                "stride_bindings": [4, 5],
                            },
                            "access": "write",
                            "vernon.source_name": "output",
                            "vernon.interface": "resource",
                            "vernon.set": 0,
                            "vernon.binding": 0,
                        }
                    ]
                },
            }
        }

        use = external_parameters(records)["output"][0]
        self.assertEqual(use["interface"], "storage")
        self.assertEqual(use["shape"], [0, 0])
        self.assertEqual(
            use["tensor_view_descriptor"],
            {"rank": 2, "offset_binding": 1, "extent_bindings": [2, 3], "stride_bindings": [4, 5]},
        )
        self.assertNotIn("interface_plan", use)

        del records["compute"]["interface"]["arguments"][0]["vernon.binding"]
        with self.assertRaisesRegex(PipelineCompileError, "missing reflected set/binding"):
            external_parameters(records)

    def test_parameter_merge_and_slot_layout_are_exact(self) -> None:
        uses = [
            {
                "stage": "compute",
                "entry": "compute_main",
                "index": 0,
                "kind": "tensor",
                "type": "tensor<4xf32>",
                "dtype": "f32",
                "element_layout": _scalar_layout("f32"),
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
                "element_layout": _scalar_layout("f32"),
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
                "element_layout": _scalar_layout("f32"),
                "shape": [4],
                "access": "read_write",
                "uses": [
                    {"stage": "compute", "index": 0, "dtype": "f32", "shape": [4], "interface": "storage"},
                    {"stage": "vertex", "index": 1, "dtype": "f32", "shape": [4], "interface": "input"},
                ],
            },
        )

        tensor_view_use = {
            "stage": "compute",
            "entry": "compute_main",
            "index": 0,
            "kind": "tensor",
            "type": '!vernon.tensor_view<f32, [4], "read", "device">',
            "element_layout": _scalar_layout("f32"),
            "shape": [4],
            "interface": "storage",
            "access": "read",
        }
        with self.assertRaisesRegex(PipelineCompileError, "must use device address space"):
            merge_parameter_uses("missing_address_space", [tensor_view_use])
        tensor_view_use["address_space"] = "device"
        self.assertEqual(
            merge_parameter_uses("values", [tensor_view_use])["address_space"],
            "device",
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
                        "kind": "scalar",
                        "type": "f32",
                        "vernon.source_name": "alpha",
                        "vernon.interface": "uniform",
                        "value_transport": "push_constant",
                        "physical_layouts": _physical_layouts(4, 4, []),
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
                "program": {
                    "fragment": fragment.id,
                    "vertex": vertex.id,
                },
                "parameters": [],
                "outputs": [
                    {
                        "name": "output_0",
                        "kind": "texture",
                        "dtype": "f32",
                        "shape": [4],
                        "access": "write",
                        "location": 0,
                    }
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
                        "element_layout": _scalar_layout("f32"),
                        "shape": [],
                    },
                    {
                        "kind": "tensor",
                        "stage": "fragment",
                        "interface": "input",
                        "type": "i32",
                        "dtype": "i32",
                        "element_layout": _scalar_layout("i32"),
                        "shape": [],
                    },
                ],
            )
        with self.assertRaisesRegex(PipelineCompileError, "mismatch"):
            validate_graphics_interfaces(
                "vertex",
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
                "fragment",
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
