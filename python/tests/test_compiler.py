from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vernon_dsl import CompileError, compile_file, compile_source
from vernon_dsl.cli import main


class TypeSystemTests(unittest.TestCase):

    def test_tensor_aliases_and_fixed_resource_types(self) -> None:
        source = """
from vernon_dsl import *

@struct
class Vertex:
    position: Tensor[f32, (3,)]
    weights: Array[f32, 4]

def resources(
    a: vec3[f32],
    b: mat[2, 3, f32],
    data: Annotated[Buffer[Vertex, "read"], resource(set=1, binding=2)],
    image: Texture["2d", f32],
    sampler: Sampler,
    flag: bool,
    signed: i32,
    unsigned: u32,
    half: f16,
    double: f64,
    rate: Annotated[f32, instance(location=5, divisor=2)],
) -> None:
    pass
"""
        output = compile_source(source, "types.py")
        self.assertIn("tensor<3xf32>", output)
        self.assertIn("tensor<2x3xf32>", output)
        self.assertIn("!vernon.array<4 x f32>", output)
        self.assertIn('!vernon.buffer<!vernon.struct<"Vertex">, "read">',
                      output)
        self.assertIn('!vernon.texture<"2d", f32>', output)
        self.assertIn('vernon.interface = "resource"', output)
        self.assertIn("vernon.set = 1 : i64", output)
        self.assertIn("vernon.location = 5 : i64", output)
        self.assertIn("vernon.instance_divisor = 2 : i64", output)
        self.assertIn(": i1", output)
        self.assertIn(": f16", output)
        self.assertIn(": f64", output)


class StageTests(unittest.TestCase):

    def test_graphics_intrinsics_and_bound_uniforms(self) -> None:
        source = """
from vernon_dsl import *

@vertex
def transform(
    position: Annotated[vec3[f32], location(0)],
    transform: Annotated[mat4[f32], uniform(set=1, binding=2)],
) -> vec4[f32]:
    direction = normalize(position)
    amount = max(dot(direction, direction), 0.0)
    return matmul(transform, vec4(direction * vec3(amount, amount, amount), 1.0))
"""
        output = compile_source(source, "intrinsics.py")
        self.assertIn('name = "normalize"', output)
        self.assertIn('name = "dot"', output)
        self.assertIn('name = "construct"', output)
        self.assertIn('name = "matmul"', output)
        self.assertIn("vernon.set = 1 : i64", output)
        self.assertIn("vernon.binding = 2 : i64", output)

    def test_vertex_fragment_and_compute(self) -> None:
        source = """
from vernon_dsl import *

@vertex
def transform(
    position: Annotated[vec4[f32], location(0)],
    offset: Annotated[vec4[f32], uniform()],
) -> vec4[f32]:
    moved = position + offset
    return moved

@fragment
def shade(color: Annotated[vec4[f32], varying()]) -> f32:
    return color.x

@compute(workgroup_size=(8, 4, 1))
def update(
    values: Annotated[Buffer[f32], resource(set=0, binding=3)],
    invocation: Annotated[u32, builtin("global_invocation_id")],
) -> None:
    current = values[invocation]
    values[invocation] = current + 1.0
    for i in range(0, 4):
        local = i
"""
        output = compile_source(source, "stages.py")
        self.assertIn('vernon.stage = "vertex"', output)
        self.assertIn('vernon.stage = "fragment"', output)
        self.assertIn('vernon.stage = "compute"', output)
        self.assertIn("vernon.workgroup_size = array<i32: 8, 4, 1>", output)
        self.assertIn('"vernon.swizzle"', output)
        self.assertIn('name = "buffer_load"', output)
        self.assertIn('name = "buffer_store"', output)
        self.assertIn("scf.for", output)

    def test_if_merges_existing_values(self) -> None:
        source = """
from vernon_dsl import *

def choose(value: f32, condition: bool) -> f32:
    result = value
    if condition:
        result = value + value
    else:
        result = value - value
    return result
"""
        output = compile_source(source, "if.py")
        self.assertIn("scf.if", output)
        self.assertEqual(output.count("scf.yield"), 2)


class SafetyAndDiagnosticsTests(unittest.TestCase):

    def test_compilation_does_not_execute_input(self) -> None:
        source = """
from vernon_dsl import *

raise RuntimeError("must not execute")
"""
        with self.assertRaisesRegex(CompileError,
                                    "unsupported module-level syntax"):
            compile_source(source, "safe.py")

    def test_unsupported_syntax_has_file_line_and_column(self) -> None:
        source = """
from vernon_dsl import *

def bad(x: f32) -> f32:
    while x > 0.0:
        x = x - 1.0
    return x
"""
        with self.assertRaises(CompileError) as caught:
            compile_source(source, "shader.py")
        message = str(caught.exception)
        self.assertIn("shader.py:5:5: error:", message)
        self.assertIn("While", message)

    def test_output_is_deterministic(self) -> None:
        source = """
from vernon_dsl import *

@fragment
def main(value: f32) -> f32:
    return value * 2.0
"""
        self.assertEqual(compile_source(source, "same.py"),
                         compile_source(source, "same.py"))

    def test_cli_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.py"
            output_path = Path(directory) / "output.mlir"
            input_path.write_text(
                "from vernon_dsl import *\n@vertex\ndef main(x: f32) -> f32:\n    return x\n",
                encoding="utf-8",
            )
            self.assertEqual(main([str(input_path), "-o",
                                   str(output_path)]), 0)
            self.assertIn("func.func @main",
                          output_path.read_text(encoding="utf-8"))


class ExampleRegressionTests(unittest.TestCase):

    def test_material_and_custom_vertex_examples_compile(self) -> None:
        root = Path(__file__).parents[2]
        expected_entries = {
            "blinn_phong.py": "@blinn_phong_fragment",
            "blinn_phong_vertices.py": "@skinned_vertex",
            "planet_terrain.py": "@planet_terrain_vertex",
        }
        for filename, entry in expected_entries.items():
            with self.subTest(filename=filename):
                output = compile_file(root / "examples" / filename)
                self.assertIn(entry, output)


if __name__ == "__main__":
    unittest.main()
