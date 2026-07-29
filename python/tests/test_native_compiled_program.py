from __future__ import annotations

import gc
import json
import struct
import unittest

import numpy as np
import vernon_dsl as vd
from vernon_dsl import _native as native
from vernon_dsl._runtime.resources import _bind_native_argument
from vernon_dsl.frontend.compiler import compile_source

CPU_MODULE = r"""
module attributes {vernon.frontend_version = 4 : i64, vernon.value_abi_version = 1 : i64} {
  func.func @increment(
      %values: !vernon.tensor_view<f32, [3], "read_write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64,
        vernon.tensor_shape = array<i64: 3>,
        vernon.tensor_strides = array<i64: 1>,
        vernon.tensor_offset = 0 : i64
      },
      %id: index {
        vernon.interface = "input",
        vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>,
        vernon.storage_effects = [
          {kind = "read", owner = "values", region = "unknown"},
          {kind = "write", owner = "values", region = "unknown"}
        ]
      } {
    %value = "vernon.load"(%values, %id) :
        (!vernon.tensor_view<f32, [3], "read_write", "device">, index) -> f32
    %one = arith.constant 1.0 : f32
    %sum = arith.addf %value, %one : f32
    "vernon.store"(%sum, %values, %id) :
        (f32, !vernon.tensor_view<f32, [3], "read_write", "device">, index) -> ()
    return
  }
}
"""

MULTI_ENTRY_MODULE = r"""
module attributes {vernon.frontend_version = 4 : i64, vernon.value_abi_version = 1 : i64} {
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
"""

CPU_TUPLE_MODULE = r"""
module attributes {
  vernon.frontend = "python",
  vernon.frontend_version = 4 : i64,
  vernon.value_abi_version = 1 : i64
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
"""

CPU_TUPLE_ENTRY_MODULE = r"""
module attributes {
  vernon.frontend = "python",
  vernon.frontend_version = 4 : i64,
  vernon.value_abi_version = 1 : i64
} {
  func.func @tuple_passthrough(
      %value: tuple<f32, i32> {
        vernon.abi_alignment = 4 : i64,
        vernon.abi_field_offsets = array<i64: 0, 4>,
        vernon.abi_layout_hash = "b5eb9ece18126ac79faf85eeb27a05a40f9d8001124b3ee648704050af420694",
        vernon.abi_leaf_counts = array<i64: 1, 1>,
        vernon.abi_leaf_dtypes = ["f32", "i32"],
        vernon.abi_leaf_offsets = array<i64: 0, 4>,
        vernon.abi_leaf_paths = ["[0]", "[1]"],
        vernon.abi_size = 8 : i64,
        vernon.interface = "input",
        vernon.location = 0 : i64
      }
    ) -> (
      tuple<f32, i32> {
        vernon.abi_alignment = 4 : i64,
        vernon.abi_field_offsets = array<i64: 0, 4>,
        vernon.abi_layout_hash = "b5eb9ece18126ac79faf85eeb27a05a40f9d8001124b3ee648704050af420694",
        vernon.abi_leaf_counts = array<i64: 1, 1>,
        vernon.abi_leaf_dtypes = ["f32", "i32"],
        vernon.abi_leaf_offsets = array<i64: 0, 4>,
        vernon.abi_leaf_paths = ["[0]", "[1]"],
        vernon.abi_size = 8 : i64,
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
"""


class CompiledProgramTests(unittest.TestCase):
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
        host.synchronize()

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
            (native.Target.OPENGL, {"glsl_version": 330}),
            (native.Target.OPENGL_ES, {"glsl_version": 310}),
        ):
            with self.subTest(target=target):
                program = native.Compiler().compile_program_result(module, target, **options)
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
        program = compiler.compile_program_result(MULTI_ENTRY_MODULE, native.Target.OPENGL, glsl_version=330)
        self.assertTrue(program.ok, program.diagnostics)
        self.assertEqual(program.target, native.Target.OPENGL)
        self.assertEqual(program.glsl_version, 330)
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
        self.assertEqual(reflection["target"], "cpu")
        self.assertTrue(reflection["target_options"]["target_triple"])
        self.assertEqual(
            reflection["entries"][0]["effects"],
            [
                {"kind": "read", "owner": "values", "region": "unknown", "indices": []},
                {"kind": "write", "owner": "values", "region": "unknown", "indices": []},
            ],
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
        _bind_native_argument(invocation, pipeline.parameters[0], values)
        invocation.grid(3, 1, 1).invoke()
        np.testing.assert_array_equal(values.to_numpy(), np.array([3.0, 5.0, 7.0], dtype=np.float32))

    def test_cpu_tuple_create_and_constant_extract_lowering(self) -> None:
        program = native.Compiler().compile_program_result(CPU_TUPLE_MODULE, native.Target.CPU)
        self.assertTrue(program.ok, program.diagnostics)
        self.assertTrue(program.has_cpu_entry("tuple_first"))
        reflection = json.loads(program.reflection)
        self.assertEqual(reflection["value_abi_version"], 1)
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
            glsl_version=450,
        )
        self.assertTrue(opengl.ok, opengl.diagnostics)
        self.assertEqual(json.loads(opengl.reflection)["struct_layouts"], reflection["struct_layouts"])

    def test_cpu_tuple_entry_uses_portable_value_abi(self) -> None:
        program = native.Compiler().compile_program_result(CPU_TUPLE_ENTRY_MODULE, native.Target.CPU)
        self.assertTrue(program.ok, program.diagnostics)
        entry = json.loads(program.reflection)["entries"][0]
        self.assertEqual(entry["physical_layouts"]["host_value"]["packed_arguments_size"], 8)
        self.assertEqual(entry["physical_layouts"]["host_value"]["packed_results_size"], 8)
        self.assertEqual(entry["arguments"][0]["vernon.abi_field_offsets"], [0, 4])
        self.assertEqual(entry["results"][0]["vernon.abi_field_offsets"], [0, 4])

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
        invalid = compiler.compile_program_result(
            CPU_MODULE,
            native.Target.CPU,
            glsl_version=450,
        )
        self.assertFalse(invalid.ok)
        self.assertEqual(invalid.status, native.Status.INVALID_ARGUMENT)
        self.assertIn("valid only for OpenGL", invalid.diagnostics)
        self.assertEqual(invalid.glsl_version, 450)

        program = compiler.compile_program_result(CPU_MODULE, native.Target.CPU, cpu="generic", cpu_features="")
        self.assertTrue(program.ok, program.diagnostics)
        self.assertEqual(program.cpu, "generic")
        self.assertEqual(program.cpu_features, "")
        default_program = compiler.compile_program_result(CPU_MODULE, native.Target.CPU)
        self.assertTrue(default_program.ok, default_program.diagnostics)
        self.assertEqual(json.loads(default_program.reflection)["target"], "cpu")


if __name__ == "__main__":
    unittest.main()
