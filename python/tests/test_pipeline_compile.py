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
    ProgramCompileError,
    ProgramVariantPlan,
    assign_parameter_slots,
    build_program_manifest,
    build_program_plan,
    canonical_json,
    compiled_stage_from_program,
    external_parameters,
    fragment_outputs,
    internal_parameters,
    make_target_options,
    merge_internal_parameter_uses,
    merge_parameter_uses,
    parse_reflection_json,
    select_artifact,
    select_entry,
    serialize_bundle,
    validate_graphics_interfaces,
    with_content_hash,
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


def _vector_layout(dtype: str, shape: list[int]) -> dict[str, object]:
    sizes = {"bool": 1, "i32": 4, "u32": 4, "f16": 2, "f32": 4, "f64": 8}
    scalar = sizes[dtype]
    count = 1
    for extent in shape:
        count *= extent
    spelling = f"tensor<{'x'.join(str(extent) for extent in shape)}x{dtype}>"
    canonical = f"tensor({spelling},{scalar * count},{scalar})|dtypes={dtype}"
    return {
        "logical_type": spelling,
        "byte_size": scalar * count,
        "alignment": scalar,
        "layout_hash": hashlib.sha256(canonical.encode()).hexdigest(),
        "leaves": [
            {
                "path": [],
                "dtype": dtype,
                "byte_offset": 0,
                "scalar_count": count,
                "shape": list(shape),
            }
        ],
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
        with self.assertRaisesRegex(ProgramCompileError, "incomplete PTX header"):
            runtime_requirements("cuda", [stage])

        metal_stage = SimpleNamespace(
            target=SimpleNamespace(target="metal", options={"platform": "macos"}),
            stage="compute",
            artifact=SimpleNamespace(data=b"MSL"),
            reflection={},
            metadata={},
            interface={},
        )
        with self.assertRaisesRegex(ProgramCompileError, "no valid msl_version"):
            runtime_requirements("metal", [metal_stage])
        metal_stage.target.options = {}
        metal_stage.reflection = {
            "target": {
                "kind": "metal",
                "options": {},
                "output": {"language": "msl", "version": [2, 4], "minimum_os_version": [11, 0]},
            }
        }
        with self.assertRaisesRegex(ProgramCompileError, "requires apple_platform"):
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
            entry="main",
            target=MetalTargetOptions(),
        )
        self.assertEqual(compiled.target.options["platform"], "ios")
        requirements = runtime_requirements("metal", [compiled])
        self.assertEqual(requirements["minimum_os_version"], [15, 0])

    def test_program_variant_plan_rejects_nullable_and_legacy_shapes(self) -> None:
        with self.assertRaises(TypeError):
            ProgramVariantPlan(key=(), stage_implementations={})  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            ProgramVariantPlan(  # type: ignore[call-arg]
                key=(),
                program={},
                stage_implementations={},
                canonical_program=None,
            )
        with self.assertRaisesRegex(ProgramCompileError, "non-contract root members"):
            ProgramVariantPlan(
                (),
                {
                    "stages": {},
                    "parameters": [],
                    "storages": [],
                    "values": [],
                    "graphs": [],
                    "abi": {},
                    "shape_symbols": [],
                },
                {},
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
        with self.assertRaisesRegex(ProgramCompileError, "invalid vulkan target options"):
            make_target_options("vulkan", {"processor": "generic"})
        with self.assertRaisesRegex(ProgramCompileError, "invalid metal target options"):
            make_target_options("metal", {"shader_model": 60})
        with self.assertRaisesRegex(ProgramCompileError, "shader model must be 6.0 or newer"):
            make_target_options("directx", {"shader_model": 55})
        with self.assertRaisesRegex(ProgramCompileError, "macos.*ios"):
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
                            "kind": "image",
                            "type": '!vernon.texture<"2d", f32, "unknown", "sampled">',
                            "dtype": "f32",
                            "dimension": "2d",
                            "binding_role": "sampled",
                            "sample_result_class": "float",
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
                            "sampled_image_bindings": [
                                {
                                    "set": 0,
                                    "binding": 3,
                                }
                            ],
                        },
                        {
                            "index": 2,
                            "kind": "tensor",
                            "type": "tensor<2xf32>",
                            "shape": [2],
                            "element_layout": _scalar_layout("f32"),
                            "value_layout": _vector_layout("f32", [2]),
                            "vernon.source_name": "__resolution",
                            "vernon.interface": "system_value",
                            "vernon.implicit": "resolution",
                        },
                    ],
                },
            },
        }
        external = external_parameters(records)
        self.assertEqual(set(external), {"image"})
        internal = internal_parameters(records)
        self.assertEqual([merge_parameter_uses(name, external[name])["name"] for name in external], ["image"])
        self.assertEqual(
            [
                (row["source"], row.get("system_value"))
                for row in (merge_internal_parameter_uses(name, internal[name]) for name in sorted(internal))
            ],
            [("implicit_sampler", None), ("system_value", "resolution")],
        )

        unpaired = json.loads(json.dumps(records))
        del unpaired["fragment"]["interface"]["arguments"][1]["sampled_image_bindings"]
        with self.assertRaisesRegex(ProgramCompileError, "no reflected sampled image binding"):
            values = internal_parameters(unpaired)
            merge_internal_parameter_uses("__image_sampler", values["__image_sampler"])

        legacy = json.loads(json.dumps(records))
        sampler = legacy["fragment"]["interface"]["arguments"][1]
        del sampler["vernon.implicit"]
        sampler["vernon.compiler_generated"] = True
        with self.assertRaisesRegex(ProgramCompileError, "legacy compiler-generated"):
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
                            "sampled_image_bindings": [
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
        with self.assertRaisesRegex(ProgramCompileError, "missing reflected set/binding"):
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
        with self.assertRaisesRegex(ProgramCompileError, "must use device address space"):
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
        records = {
            "vertex": {
                "entry": vertex.entry,
                "target": vertex.target.target,
                "interface": dict(vertex.interface),
            },
            "fragment": {
                "entry": fragment.entry,
                "target": fragment.target.target,
                "interface": dict(fragment.interface),
            },
        }
        external = external_parameters(records)
        slots = assign_parameter_slots((records,))
        parameters = []
        for name in sorted(external, key=slots.__getitem__):
            parameter = merge_parameter_uses(name, external[name])
            parameter["slot"] = slots[name]
            parameters.append(parameter)
        self.assertEqual([(row["name"], row["slot"]) for row in parameters], [("alpha", 0), ("z_position", 1)])
        self.assertEqual(
            parameters[0]["uses"][0]["uniform_name"],
            "alpha",
        )

    def test_graphics_outputs_and_interface_are_exact(self) -> None:
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
        records = {
            "vertex": {"interface": dict(vertex.interface)},
            "fragment": {"interface": dict(fragment.interface)},
        }
        validate_graphics_interfaces("vertex", records["vertex"], "fragment", records["fragment"])
        self.assertEqual(
            fragment_outputs(records),
            [
                {
                    "name": "output_0",
                    "kind": "image",
                    "dtype": "f32",
                    "shape": [4],
                    "access": "write",
                    "location": 0,
                }
            ],
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

        document = with_content_hash({"type": "program", "id": "deterministic"})
        encoded = serialize_bundle(document)
        self.assertEqual(encoded, serialize_bundle(document))
        document = json.loads(encoded)
        unhashed = dict(document)
        digest = unhashed.pop("content_hash")
        self.assertEqual(
            digest,
            hashlib.sha256(canonical_json(unhashed).encode("utf-8")).hexdigest(),
        )

    def test_program_manifest_has_exact_canonical_envelope(self) -> None:
        reflection = {
            "required_features": [],
            "endpoints": [],
            "compute": {
                "workgroup_size": [1, 1, 1],
                "subgroup": None,
                "capabilities": ["direct_dispatch"],
            },
        }
        contract = {"operation": "compute", "reflection": reflection}
        contract_hash = hashlib.sha256(canonical_json(contract).encode()).hexdigest()
        artifact = CompiledArtifact("glsl", b"#version 430\nvoid main() {}", "main.glsl")
        stage = CompiledStage(
            "module",
            "main",
            "compute",
            OpenGLTargetOptions(version=430),
            {"required_features": []},
            {"workgroup_size": [1, 1, 1]},
            artifact,
            {"program_contracts": {"main": contract}},
        )
        program = {
            "stages": {"main": {"operation": "compute", "contract_hash": contract_hash}},
            "parameters": [],
            "storages": [],
            "values": [],
            "graphs": [
                {
                    "name": "forward",
                    "direction": "forward",
                    "inputs": [],
                    "captures": [],
                    "outputs": [],
                    "nodes": [],
                }
            ],
            "abi": {"boundary_slots": [], "derivative_projections": [], "tape_plans": []},
        }
        plan = build_program_plan("program", stage.target, [(("FEATURE",), {"main": stage}, program)])
        manifest = build_program_manifest(
            plan,
            {
                stage.id: {
                    "format": "glsl",
                    "storage": "external",
                    "path": f"artifacts/{artifact.sha256}.glsl",
                    "size": len(artifact.data),
                    "sha256": artifact.sha256,
                }
            },
        )
        self.assertEqual(
            set(manifest),
            {
                "compiler_contract_version",
                "program_version",
                "type",
                "id",
                "target",
                "blobs",
                "variants",
                "content_hash",
            },
        )
        variant = manifest["variants"][0]
        self.assertEqual(set(variant), {"key", "program", "artifact_system"})
        self.assertEqual(set(variant["artifact_system"]), {"runtime_requirements", "artifacts"})
        self.assertEqual(set(variant["artifact_system"]["artifacts"]), {"main"})
        self.assertNotIn("stage_bindings", variant)
        self.assertNotIn("blobs", variant["artifact_system"])
        self.assertNotIn("target", variant["artifact_system"])

    def test_reflection_selection_and_error_cases(self) -> None:
        reflection = parse_reflection_json(
            '{"entries":[{"name":"main","stage":"compute"}],'
            '"artifacts":[{"entry_point":"main","stage":"compute",'
            '"format":"ptx","filename":"main.ptx"}]}'
        )
        self.assertEqual(select_entry(reflection, "main")["stage"], "compute")
        self.assertEqual(select_artifact(reflection, "main", "compute")["filename"], "main.ptx")
        with self.assertRaisesRegex(ProgramCompileError, "exactly one"):
            select_entry({"entries": []}, "missing")
        with self.assertRaisesRegex(ProgramCompileError, "incompatible"):
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
        with self.assertRaisesRegex(ProgramCompileError, "mismatch"):
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


if __name__ == "__main__":
    unittest.main()
