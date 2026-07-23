from __future__ import annotations

import gc
import importlib.util
import json
import struct
import sys
import unittest
from pathlib import Path
from types import ModuleType


def _load_native() -> ModuleType:
    if len(sys.argv) < 2 or not Path(sys.argv[1]).is_file():
        from vernon_dsl._runtime.session import _native

        return _native
    module_path = Path(sys.argv.pop(1)).resolve()
    spec = importlib.util.spec_from_file_location("_native", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = _load_native()

CPU_MODULE = r"""
module {
  func.func @increment(
      %values: !vernon.buffer<f32, "read_write"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64,
        vernon.tensor_shape = array<i64: 3>
      },
      %id: index {
        vernon.interface = "input",
        vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %value = "vernon.intrinsic"(%values, %id) {name = "buffer_load"} :
        (!vernon.buffer<f32, "read_write">, index) -> f32
    %one = arith.constant 1.0 : f32
    %sum = arith.addf %value, %one : f32
    "vernon.intrinsic"(%values, %id, %sum) {name = "buffer_store"} :
        (!vernon.buffer<f32, "read_write">, index, f32) -> ()
    return
  }
}
"""

MULTI_ENTRY_MODULE = r"""
module {
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


class CompiledProgramTests(unittest.TestCase):
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

        runtime = native.Runtime(native.RuntimeBackend.CPU)
        with self.assertRaisesRegex(RuntimeError, "CPU entry 'missing' was not found"):
            runtime.load_cpu_entry(program, "missing")
        kernel = runtime.load_cpu_entry(program, "increment")
        del program
        gc.collect()

        values = runtime.allocate(12, 4)
        values.upload(struct.pack("=3f", 2.0, 4.0, 6.0))
        kernel.launch(3, 1, 1, [values])
        self.assertEqual(struct.unpack("=3f", values.download()), (3.0, 5.0, 7.0))

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
