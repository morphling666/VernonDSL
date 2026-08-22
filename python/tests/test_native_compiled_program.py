from __future__ import annotations

import gc
import json
import struct
import unittest

import numpy as np
import vernon_dsl as vd
from vernon_dsl import _native as native
from vernon_dsl._versions import COMPILER_CONTRACT_VERSION, PIPELINE_VERSION
from vernon_dsl.frontend.compiler import compile_source


def _versioned(module: str) -> str:
    attributes = (
        f"vernon.compiler_contract_version = {COMPILER_CONTRACT_VERSION} : i64, "
        f"vernon.pipeline_version = {PIPELINE_VERSION} : i64"
    )
    return module.replace("$VERNON_VERSION_ATTRIBUTES", attributes)


CPU_MODULE = _versioned(r"""
module attributes {$VERNON_VERSION_ATTRIBUTES} {
  func.func @increment(
      %values: !vernon.tensor_view<f32, [3], "read_write", "device"> {
        vernon.interface = "resource",
        vernon.source_name = "values",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %id: tensor<3xi32> {
        vernon.interface = "input",
        vernon.builtin = "global_invocation_id",
        vernon.dtype = "u32",
        vernon.abi_leaf_dtypes = ["u32"]
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>,
        vernon.storage_effects = [
          {kind = "read", owner = "values", region = "unknown"},
          {kind = "write", owner = "values", region = "unknown"}
        ]
      } {
    %zero = arith.constant 0 : index
    %id_i32 = tensor.extract %id[%zero] : tensor<3xi32>
    %id_x = arith.index_castui %id_i32 : i32 to index
    %value = "vernon.load"(%values, %id_x) :
        (!vernon.tensor_view<f32, [3], "read_write", "device">, index) -> f32
    %one = arith.constant 1.0 : f32
    %sum = arith.addf %value, %one : f32
    "vernon.store"(%sum, %values, %id_x) :
        (f32, !vernon.tensor_view<f32, [3], "read_write", "device">, index) -> ()
    return
  }
}
""")

DYNAMIC_CPU_MODULE = CPU_MODULE.replace("[3]", "[-1]")

CONSTANT_WRITE_CPU_MODULE = _versioned(r"""
module attributes {$VERNON_VERSION_ATTRIBUTES} {
  func.func @constant_write(
      %values: !vernon.tensor_view<f32, [1], "write", "device"> {
        vernon.interface = "resource",
        vernon.source_name = "values",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>,
        vernon.storage_effects = [
          {kind = "write", owner = "values", region = "element", indices = array<i64: 0>}
        ]
      } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    "vernon.store"(%one, %values, %zero) :
        (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
    return
  }
}
""")

MULTI_ENTRY_MODULE = _versioned(r"""
module attributes {$VERNON_VERSION_ATTRIBUTES} {
  func.func @vertex_main(
      %position: tensor<4xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      },
      %color: tensor<3xf32> {
        vernon.interface = "input", vernon.location = 1 : i64
      }) -> (
      tensor<4xf32> {
        vernon.interface = "output", vernon.builtin = "position"
      },
      tensor<3xf32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "vertex"} {
    return %position, %color : tensor<4xf32>, tensor<3xf32>
  }
  func.func @fragment_main(
      %color: tensor<3xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      }) -> (tensor<3xf32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    return %color : tensor<3xf32>
  }
}
""")

CPU_TUPLE_MODULE = _versioned(r"""
module attributes {
  vernon.frontend = "python",
  $VERNON_VERSION_ATTRIBUTES
} {
  "vernon.struct"() {
    fields = ["value:f32", "weight:f64"],
    sym_name = "ReflectedRecord"
  } : () -> ()
  func.func @tuple_first(
      %value: f32 {
        vernon.interface = "input",
        vernon.location = 0 : i64
      }) -> (f32 {
        vernon.interface = "output",
        vernon.location = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "fragment"
      } {
    %constant = arith.constant 2 : i32
    %tuple = "vernon.tuple_create"(%value, %constant)
        : (f32, i32) -> tuple<f32, i32>
    %first = "vernon.tuple_get"(%tuple) {index = 0 : i64}
        : (tuple<f32, i32>) -> f32
    return %first : f32
  }
}
""")

CPU_TUPLE_ENTRY_MODULE = _versioned(r"""
module attributes {
  vernon.frontend = "python",
  $VERNON_VERSION_ATTRIBUTES
} {
  func.func @tuple_passthrough(
      %value: tuple<f32, i32> {
        vernon.abi_leaf_dtypes = ["f32", "i32"],
        vernon.interface = "input",
        vernon.location = 0 : i64
      }
    ) -> (
      tuple<f32, i32> {
        vernon.abi_leaf_dtypes = ["f32", "i32"],
        vernon.interface = "output",
        vernon.location = 0 : i64
      }
    ) attributes {
      vernon.entry,
      vernon.stage = "compute",
      vernon.workgroup_size = array<i32: 1, 1, 1>
    } {
    return %value : tuple<f32, i32>
  }
}
""")

PROGRAM_MODULE = r"""
module {
  func.func @forward(%source: f32 {vernon.source_name = "source"})
      -> (f32 {vernon.source_name = "output"})
      attributes {
        vernon_program.graph = "forward",
        vernon_program.argument_names = ["input.source"],
        vernon_program.result_names = ["output.result"]
      } {
    %result = "vernon_program.compute"(%source) {
      callee = "Module.square",
      grid = array<i64: 1, 1, 1>,
      features = [],
      operand_names = ["source"],
      result_names = ["output"]
    } : (f32) -> f32
    func.return %result : f32
  }
}
"""

DIRECT_KERNEL_MODULE = r"""
module {
  func.func @increment(
      %values: !vernon.tensor_view<f32, [16], "read_write", "device"> {
        vernon.interface = "resource",
        vernon.source_name = "values",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    return
  }
}
"""


class CompiledProgramTests(unittest.TestCase):
    def test_program_planner_returns_kernel_compile_requests(self) -> None:
        plan = native.Compiler().plan_program_result(PROGRAM_MODULE)
        self.assertTrue(plan.ok, plan.diagnostics)
        reflection = json.loads(plan.reflection)
        self.assertEqual(len(reflection["kernel_compile_requests"]), 1)
        request = reflection["kernel_compile_requests"][0]
        self.assertEqual(request["id"], "forward:0")
        self.assertEqual(request["implementation_hint"], "Module.square")
        self.assertEqual(request["kind"], "compute")
        self.assertIn("vernon_program.compute", request["region_mlir"])
        self.assertEqual(reflection["execution"]["graphs"][0]["nodes"][0]["stage"], "forward:0")

    def test_kernel_planner_bootstraps_explicit_dispatch_controls(self) -> None:
        plan = native.Compiler().plan_kernel_result(DIRECT_KERNEL_MODULE)
        self.assertTrue(plan.ok, plan.diagnostics)
        reflection = json.loads(plan.reflection)
        request = reflection["kernel_compile_requests"][0]
        node = reflection["execution"]["graphs"][0]["nodes"][0]
        self.assertEqual(
            node["grid"],
            [
                {"control": {"argument": 1}},
                {"control": {"argument": 2}},
                {"control": {"argument": 3}},
            ],
        )
        self.assertEqual(request["grid"], node["grid"])
        self.assertEqual(request["bindings"], [{"parameter": "values", "value": 0}])
        self.assertEqual(node["resources"], [{"value": 0, "access": "read_write", "after": 4}])
        self.assertIn("func.func @increment", request["region_mlir"])

    def test_target_available_reports_compiler_capabilities(self) -> None:
        self.assertTrue(native.target_available(native.Target.CPU))
        for target in (
            native.Target.CPU,
            native.Target.CUDA,
            native.Target.VULKAN,
            native.Target.METAL,
            native.Target.DIRECTX,
            native.Target.OPENGL,
            native.Target.OPENGL_ES,
        ):
            with self.subTest(target=target):
                self.assertIsInstance(native.target_available(target), bool)

    def test_context_owned_python_gpu_resource_api_is_removed(self) -> None:
        self.assertFalse(hasattr(native.Runtime, "create_texture"))
        self.assertFalse(hasattr(native.Runtime, "import_opengl_sampler"))
        self.assertFalse(hasattr(native, "Texture"))
        self.assertFalse(hasattr(native, "Sampler"))

    def test_standalone_rhi_buffer_owns_generational_resource(self) -> None:
        try:
            host = native.RhiHost(native.RhiBackend.CUDA)
        except RuntimeError:
            self.skipTest("CUDA RHI device is unavailable")
        source = struct.pack("4I", 1, 2, 3, 4)
        buffer = host.create_buffer(len(source))
        buffer.upload(source)
        self.assertEqual(buffer.download(), source)
        runtime = host.create_runtime()
        self.assertIsNotNone(runtime)

    def test_standalone_rhi_image_upload_and_download(self) -> None:
        for backend in (native.RhiBackend.VULKAN, native.RhiBackend.DIRECTX12):
            with self.subTest(backend=backend):
                try:
                    host = native.RhiHost(backend)
                except RuntimeError:
                    continue
                source = bytes(range(16))
                image = host.create_image(2, 2)
                image.upload(source)
                self.assertEqual(image.download(), source)

    def test_standalone_rhi_three_dimensional_image_round_trip(self) -> None:
        for backend in (native.RhiBackend.VULKAN, native.RhiBackend.DIRECTX12):
            with self.subTest(backend=backend):
                try:
                    host = native.RhiHost(backend)
                except RuntimeError:
                    continue
                source = bytes(range(2 * 3 * 4 * 4))
                image = host.create_image(
                    4,
                    3,
                    native.TextureFormat.RGBA8_UNORM,
                    native.TextureDimension.TEXTURE_3D,
                    2,
                )
                self.assertEqual((image.width, image.height, image.depth), (4, 3, 2))
                image.upload(source)
                self.assertEqual(image.download(), source)

    def test_graphics_dynamic_bounds_structured_loop_compiles(self) -> None:
        module = compile_source(
            "from vernon_dsl import *\n"
            "@fragment\n"
            "def main(start: Annotated[i32, uniform()], stop: Annotated[i32, uniform()])"
            " -> i32:\n"
            "    result = 0\n"
            "    for index in range(start, stop, 1):\n"
            "        result += i32(index)\n"
            "    return result\n",
            "graphics_loop.py",
        )
        for target, options in (
            (native.Target.VULKAN, {}),
            (native.Target.OPENGL, {"version": 330}),
            (native.Target.OPENGL_ES, {"version": 310}),
        ):
            with self.subTest(target=target):
                program = native.Compiler().compile_program_result(module, target, options)
                self.assertTrue(program.ok, program.diagnostics)

    def test_spirv_struct_carried_structured_loop_compiles(self) -> None:
        module = compile_source(
            "from vernon_dsl import *\n"
            "@struct\n"
            "class State:\n"
            "    value: f32\n"
            "    active: bool\n"
            "@fragment\n"
            "def main() -> f32:\n"
            "    state = State(0.0, True)\n"
            "    for index in range(4):\n"
            "        state = State(state.value + f32(index), not state.active)\n"
            "    return state.value\n",
            "graphics_struct_loop.py",
        )
        program = native.Compiler().compile_program_result(module, native.Target.VULKAN)
        self.assertTrue(program.ok, program.diagnostics)

    def test_showcase_math_compiles_for_graphics_backends(self) -> None:
        module = compile_source(
            "from vernon_dsl import *\n"
            "@fragment\n"
            "def main(y: f32, x: f32) -> f32:\n"
            "    return floor(y) + acos(clamp(x, -1.0, 1.0)) + atan2(y, x)\n",
            "showcase_math_backends.py",
        )
        for target, options in (
            (native.Target.VULKAN, {}),
            (native.Target.DIRECTX, {}),
            (native.Target.OPENGL, {"version": 430}),
        ):
            with self.subTest(target=target):
                if target == native.Target.DIRECTX and not native.target_available(target):
                    continue
                program = native.Compiler().compile_program_result(module, target, options)
                self.assertTrue(program.ok, program.diagnostics)

    def test_spirv_dynamic_step_reports_contract_capability(self) -> None:
        module = compile_source(
            "from vernon_dsl import *\n"
            "@fragment\n"
            "def main(start: Annotated[i32, uniform()], stop: Annotated[i32, uniform()],"
            " step: Annotated[i32, uniform()]) -> i32:\n"
            "    result = 0\n"
            "    for index in range(start, stop, step):\n"
            "        result += i32(index)\n"
            "    return result\n",
            "graphics_dynamic_step.py",
        )
        program = native.Compiler().compile_program_result(module, native.Target.VULKAN)
        self.assertFalse(program.ok)
        self.assertIn(
            "SPIR-V targets do not support dynamic range steps",
            program.diagnostics,
        )

    def test_named_artifacts_and_reflection(self) -> None:
        compiler = native.Compiler()
        program = compiler.compile_program_result(MULTI_ENTRY_MODULE, native.Target.OPENGL, {"version": 330})
        self.assertTrue(program.ok, program.diagnostics)
        self.assertEqual(program.target, native.Target.OPENGL)
        self.assertEqual(json.loads(program.reflection)["target"]["options"]["version"], 330)
        self.assertEqual(len(program.artifacts), 2)
        self.assertEqual(len({name for name, _ in program.artifacts}), 2)
        self.assertTrue(all(name.endswith(".glsl") for name, _ in program.artifacts))

    def test_cpu_entry_execution_and_result_lifetime(self) -> None:

        def compile_locally() -> object:
            compiler = native.Compiler()
            result = compiler.compile_program_result(CPU_MODULE, native.Target.CPU)
            self.assertTrue(result.ok, result.diagnostics)
            return result

        program = compile_locally()
        gc.collect()
        self.assertTrue(program.has_cpu_entry("increment"))
        self.assertFalse(program.has_cpu_entry("missing"))
        reflection = json.loads(program.reflection)
        self.assertEqual(reflection["target"]["kind"], "cpu")
        self.assertTrue(reflection["target"]["options"]["triple"])
        self.assertEqual(
            reflection["entries"][0]["effects"],
            [
                {"kind": "read", "owner": "values", "region": "unknown", "indices": []},
                {"kind": "write", "owner": "values", "region": "unknown", "indices": []},
            ],
        )
        self.assertEqual(
            reflection["entries"][0]["tensor_view_write_footprints"],
            [{"version": 1, "owner": "values", "kind": "whole_view", "indices": []}],
        )

        vd.init(arch=vd.cpu)
        runtime = native.Runtime(native.RuntimeBackend.CPU)
        with self.assertRaisesRegex(RuntimeError, "CPU entry 'missing' was not found"):
            runtime.load_cpu_entry(program, "missing")
        pipeline = runtime.load_cpu_entry(program, "increment")
        del program
        gc.collect()

        values = vd.storage.from_numpy(np.array([2.0, 4.0, 6.0], dtype=np.float32))
        invocation = pipeline.invocation_builder()
        invocation.host_tensor(pipeline.parameters[0].name, values._native_host_array())
        invocation.grid(3, 1, 1).submit().wait()
        np.testing.assert_array_equal(values.to_numpy(), np.array([3.0, 5.0, 7.0], dtype=np.float32))

    def test_cpu_constant_write_requires_single_invocation(self) -> None:
        program = native.Compiler().compile_program_result(CONSTANT_WRITE_CPU_MODULE, native.Target.CPU)
        self.assertTrue(program.ok, program.diagnostics)
        runtime = native.Runtime(native.RuntimeBackend.CPU)
        pipeline = runtime.load_cpu_entry(program, "constant_write")
        values = vd.storage.from_numpy(np.array([0.0], dtype=np.float32))

        invocation = pipeline.invocation_builder()
        invocation.host_tensor(pipeline.parameters[0].name, values._native_host_array())
        invocation.grid(1, 1, 1).submit().wait()
        np.testing.assert_array_equal(values.to_numpy(), np.array([1.0], dtype=np.float32))

        invocation = pipeline.invocation_builder()
        invocation.host_tensor(pipeline.parameters[0].name, values._native_host_array())
        with self.assertRaisesRegex(RuntimeError, "dispatch grid axis 0 must equal 1"):
            invocation.grid(2, 1, 1).submit().wait()

    def test_cpu_profile_batch_compilation_preserves_order_and_options(self) -> None:
        programs = native._compile_cpu_program_results(
            [CPU_MODULE, CPU_MODULE],
            {"processor": "generic"},
        )
        self.assertEqual(len(programs), 2)
        for program in programs:
            self.assertTrue(program.ok, program.diagnostics)
            self.assertTrue(program.has_cpu_entry("increment"))
            self.assertEqual(json.loads(program.reflection)["target"]["options"]["processor"], "generic")
        with self.assertRaisesRegex(ValueError, "unknown option 'version'"):
            native._compile_cpu_program_results([CPU_MODULE], {"version": "450"})

    def test_cpu_artifact_reuses_dynamic_tensor_view_descriptor(self) -> None:
        program = native.Compiler().compile_program_result(DYNAMIC_CPU_MODULE, native.Target.CPU)
        self.assertTrue(program.ok, program.diagnostics)
        runtime = native.Runtime(native.RuntimeBackend.CPU)
        pipeline = runtime.load_cpu_entry(program, "increment")
        values = vd.storage.from_numpy(np.arange(5, dtype=np.float32))

        invocation = pipeline.invocation_builder()
        invocation.host_tensor(pipeline.parameters[0].name, values._native_host_array())
        invocation.grid(5, 1, 1).submit().wait()

        reverse = values.view(shape=(3,), strides=(-1,), offset=4, access="read_write")
        invocation = pipeline.invocation_builder()
        invocation.host_tensor(pipeline.parameters[0].name, reverse._native_host_array())
        invocation.grid(3, 1, 1).submit().wait()
        np.testing.assert_array_equal(values.to_numpy(), np.array([1.0, 2.0, 4.0, 5.0, 6.0], dtype=np.float32))

    def test_cpu_tuple_create_and_constant_extract_lowering(self) -> None:
        program = native.Compiler().compile_program_result(CPU_TUPLE_MODULE, native.Target.CPU)
        self.assertTrue(program.ok, program.diagnostics)
        self.assertTrue(program.has_cpu_entry("tuple_first"))
        reflection = json.loads(program.reflection)
        self.assertEqual(reflection["compiler_contract_version"], COMPILER_CONTRACT_VERSION)
        self.assertEqual(reflection["pipeline_version"], PIPELINE_VERSION)
        self.assertEqual(
            reflection["struct_layouts"],
            [
                {
                    "alignment": 8,
                    "field_offsets": [0, 8],
                    "fields": ["value:f32", "weight:f64"],
                    "name": "ReflectedRecord",
                    "size": 16,
                }
            ],
        )
        vulkan = native.Compiler().compile_program_result(CPU_TUPLE_MODULE, native.Target.VULKAN)
        self.assertTrue(vulkan.ok, vulkan.diagnostics)
        self.assertEqual(json.loads(vulkan.reflection)["struct_layouts"], reflection["struct_layouts"])
        opengl = native.Compiler().compile_program_result(
            CPU_TUPLE_MODULE,
            native.Target.OPENGL,
            {"version": 450},
        )
        self.assertTrue(opengl.ok, opengl.diagnostics)
        self.assertEqual(json.loads(opengl.reflection)["struct_layouts"], reflection["struct_layouts"])

    def test_cpu_tuple_entry_uses_portable_value_abi(self) -> None:
        program = native.Compiler().compile_program_result(CPU_TUPLE_ENTRY_MODULE, native.Target.CPU)
        self.assertTrue(program.ok, program.diagnostics)
        entry = json.loads(program.reflection)["entries"][0]
        self.assertEqual(entry["physical_layouts"]["host_value"]["packed_arguments_size"], 8)
        self.assertEqual(entry["physical_layouts"]["host_value"]["packed_results_size"], 8)
        self.assertEqual(
            [leaf["byte_offset"] for leaf in entry["arguments"][0]["value_layout"]["leaves"]],
            [0, 4],
        )
        self.assertEqual(
            [leaf["byte_offset"] for leaf in entry["results"][0]["value_layout"]["leaves"]],
            [0, 4],
        )

    def test_retired_duplicate_value_abi_metadata_is_rejected(self) -> None:
        source = CPU_TUPLE_ENTRY_MODULE.replace(
            'vernon.abi_leaf_dtypes = ["f32", "i32"],',
            'vernon.abi_size = 8 : i64, vernon.abi_leaf_dtypes = ["f32", "i32"],',
            1,
        )
        program = native.Compiler().compile_program_result(source, native.Target.CPU)
        self.assertFalse(program.ok)
        self.assertIn("retired duplicated Value ABI metadata", program.diagnostics)

    def test_tuple_and_struct_aggregates_lower_on_every_backend(self) -> None:
        from vernon_dsl import compile_source

        module = compile_source(
            "from vernon_dsl import *\n"
            "@struct\n"
            "class Pair:\n"
            "    first: f32\n"
            "    second: i32\n"
            "@kernel\n"
            "def aggregate_kernel() -> None:\n"
            "    pairs = Tensor([Tuple(1.0, 2), Tuple(3.0, 4)])\n"
            "    pair = pairs[1]\n"
            "    value = Pair(pair[0], pair[1])\n"
            "    copy = Pair(value.first, value.second)\n",
            "aggregate_backends.py",
        )
        layouts: list[object] = []
        for target in (
            native.Target.CPU,
            native.Target.CUDA,
            native.Target.VULKAN,
            native.Target.OPENGL,
        ):
            with self.subTest(target=target):
                program = native.Compiler().compile_program_result(module, target)
                self.assertTrue(program.ok, program.diagnostics)
                layouts.append(json.loads(program.reflection)["struct_layouts"])
        self.assertTrue(all(layout == layouts[0] for layout in layouts[1:]))

    def test_diagnostics_and_target_options(self) -> None:
        compiler = native.Compiler()
        with self.assertRaisesRegex(ValueError, "unknown option 'version'"):
            compiler.compile_program_result(CPU_MODULE, native.Target.CPU, {"version": 450})

        program = compiler.compile_program_result(CPU_MODULE, native.Target.CPU, {"processor": "generic"})
        self.assertTrue(program.ok, program.diagnostics)
        self.assertEqual(json.loads(program.reflection)["target"]["options"]["processor"], "generic")
        positional = compiler.compile_program_result(CPU_MODULE, native.Target.CPU, {"processor": "generic"})
        self.assertTrue(positional.ok, positional.diagnostics)
        self.assertEqual(json.loads(positional.reflection)["target"]["options"]["processor"], "generic")
        default_program = compiler.compile_program_result(CPU_MODULE, native.Target.CPU)
        self.assertTrue(default_program.ok, default_program.diagnostics)
        self.assertEqual(json.loads(default_program.reflection)["target"]["kind"], "cpu")


if __name__ == "__main__":
    unittest.main()
