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

@func
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

    def test_texture_dimensions_and_coordinate_ranks(self) -> None:
        for dimension, vector in (("2d", "vec2"), ("3d", "vec3"),
                                  ("cube", "vec3")):
            source = f"""
from vernon_dsl import *

@fragment
def sample(
    image: Annotated[Texture["{dimension}", f32], resource(set=0, binding=0)],
    sampler: Annotated[Sampler, resource(set=0, binding=1)],
    uv: {vector}[f32],
) -> vec4[f32]:
    return texture_sample(image, sampler, uv)
"""
            output = compile_source(source, f"texture_{dimension}.py")
            self.assertIn(f'!vernon.texture<"{dimension}", f32>', output)
            self.assertIn('name = "texture_sample"', output)

    def test_texture_dimension_and_coordinate_rank_are_validated(self) -> None:
        invalid_dimension = """
from vernon_dsl import *
@fragment
def sample(image: Texture["1d", f32]) -> f32:
    return 0.0
"""
        with self.assertRaisesRegex(CompileError,
                                    "texture dimension must be one of"):
            compile_source(invalid_dimension, "bad_dimension.py")

        invalid_coordinates = """
from vernon_dsl import *
@fragment
def sample(image: Texture["cube", f32], sampler: Sampler,
           uv: vec2[f32]) -> vec4[f32]:
    return texture_sample(image, sampler, uv)
"""
        with self.assertRaisesRegex(CompileError, "3-component"):
            compile_source(invalid_coordinates, "bad_coordinates.py")

    def test_swizzle_aliases_are_canonicalized(self) -> None:
        source = """
from vernon_dsl import *

@func
def aliases(color: vec4[f32]) -> vec4[f32]:
    red = color.r
    green = color.g
    blue = color.b
    alpha = color.a
    rgb = color.rgb
    rgba = color.rgba
    return vec4(rgb, alpha)
"""
        output = compile_source(source, "swizzle_aliases.py")
        for mask in ("x", "y", "z", "w", "xyz", "xyzw"):
            self.assertIn(f'mask = "{mask}"', output)
        self.assertNotIn('mask = "rgb"', output)
        self.assertNotIn('mask = "rgba"', output)

        invalid = """
from vernon_dsl import *

@func
def invalid(color: vec3[f32]) -> f32:
    return color.a
"""
        with self.assertRaisesRegex(CompileError,
                                    "swizzle 'a' is out of bounds"):
            compile_source(invalid, "invalid_swizzle_alias.py")


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

@func
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

    def test_while_and_augmented_assignment_lower_to_scf(self) -> None:
        source = """
from vernon_dsl import *

@func
def bad(x: f32) -> f32:
    while x > 0.0:
        x = x - 1.0
    return x
"""
        output = compile_source(source, "shader.py")
        self.assertIn("scf.while", output)
        self.assertIn("arith.subf", output)

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


class ModuleGraphTests(unittest.TestCase):

    def test_project_local_helper_is_namespaced_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "lighting.py"
            shader = root / "shader.py"
            helper.write_text(
                "from vernon_dsl import *\n"
                "@func\n"
                "def scale(value: f32, factor: f32) -> f32:\n"
                "    return value * factor\n",
                encoding="utf-8",
            )
            shader.write_text(
                "from vernon_dsl import *\n"
                "from lighting import scale\n"
                "@fragment\n"
                "def main(value: f32) -> f32:\n"
                "    return scale(value, 2.0)\n",
                encoding="utf-8",
            )

            output = compile_file(shader)

            self.assertIn("func.func private @__vernon_lighting__scale",
                          output)
            self.assertIn("func.call @__vernon_lighting__scale", output)
            self.assertIn("vernon.source_dependencies", output)
            self.assertIn("lighting.py=", output)
            self.assertIn("shader.py=", output)

    def test_import_cycle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("from b import helper\n",
                                       encoding="utf-8")
            (root / "b.py").write_text(
                "from a import main\n"
                "from vernon_dsl import func\n"
                "@func\n"
                "def helper(value: f32) -> f32:\n"
                "    return value\n",
                encoding="utf-8")
            with self.assertRaisesRegex(CompileError, "import cycle"):
                compile_file(root / "a.py")

    def test_recursive_helpers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recursive.py"
            path.write_text(
                "from vernon_dsl import *\n"
                "@func\n"
                "def first(value: f32) -> f32:\n"
                "    return second(value)\n"
                "@func\n"
                "def second(value: f32) -> f32:\n"
                "    return first(value)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CompileError,
                                        "recursive DSL call graph"):
                compile_file(path)

    def test_undecorated_helper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_func.py"
            path.write_text(
                "from vernon_dsl import *\n"
                "def helper(value: f32) -> f32:\n"
                "    return value\n"
                "@fragment\n"
                "def main(value: f32) -> f32:\n"
                "    return helper(value)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CompileError, "requires @func"):
                compile_file(path)


class FeatureVariantTests(unittest.TestCase):

    def test_features_specialize_interfaces_and_control_flow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variants.py"
            path.write_text(
                "from vernon_dsl import *\n"
                "INSTANCE = feature(\"INSTANCE\")\n"
                "SKIN = feature(\"SKIN\")\n"
                "@vertex\n"
                "def mesh_vertex(\n"
                "    position: vec3[f32],\n"
                "    transform: When[INSTANCE, Annotated[mat4[f32], instance()]],\n"
                "    joints: When[SKIN, vec4[u32]],\n"
                "    weights: When[SKIN, vec4[f32]],\n"
                ") -> Annotated[vec4[f32], builtin(\"position\")]:\n"
                "    result = vec4(position, 1.0)\n"
                "    if INSTANCE:\n"
                "        result = matmul(transform, result)\n"
                "    if SKIN:\n"
                "        result = result + weights\n"
                "    return result\n",
                encoding="utf-8",
            )

            static = compile_file(path)
            instanced = compile_file(path, features={"INSTANCE"})
            skinned = compile_file(path, features={"SKIN"})
            combined = compile_file(path, features={"SKIN", "INSTANCE"})

            for output in (static, instanced, skinned, combined):
                self.assertNotIn("scf.if", output)
            self.assertNotIn("vernon.instance_divisor", static)
            self.assertIn("vernon.instance_divisor", instanced)
            self.assertNotIn("vernon.location = 5", instanced)
            self.assertIn("vernon.location = 5", skinned)
            self.assertIn("vernon.location = 6", skinned)
            self.assertIn("vernon.location = 5", combined)
            self.assertIn("vernon.location = 6", combined)
            self.assertIn("vernon.location = 1", instanced)
            self.assertIn('vernon.variant_key = ["INSTANCE", "SKIN"]',
                          combined)

    def test_explicit_location_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.py"
            path.write_text(
                "from vernon_dsl import *\n"
                "@vertex\n"
                "def main(\n"
                "    transform: Annotated[mat4[f32], location(1)],\n"
                "    value: Annotated[vec4[f32], location(2)],\n"
                ") -> vec4[f32]:\n"
                "    return value\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CompileError, "location overlap"):
                compile_file(path)

    def test_disabled_value_and_unknown_feature_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.py"
            path.write_text(
                "from vernon_dsl import *\n"
                "INSTANCE = feature(\"INSTANCE\")\n"
                "@vertex\n"
                "def main(value: When[INSTANCE, f32]) -> f32:\n"
                "    return value\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CompileError, "disabled value"):
                compile_file(path)
            with self.assertRaisesRegex(CompileError, "undeclared feature"):
                compile_file(path, features={"SKIN"})

    def test_selected_entry_prunes_unrelated_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stages.py"
            path.write_text(
                "from vernon_dsl import *\n"
                "@vertex\n"
                "def vertex_main(value: f32) -> f32:\n"
                "    return value\n"
                "@fragment\n"
                "def fragment_main(value: f32) -> f32:\n"
                "    return value\n",
                encoding="utf-8",
            )
            output = compile_file(path, entry="fragment_main")
            self.assertIn("@fragment_main", output)
            self.assertNotIn("@vertex_main", output)


class ExampleRegressionTests(unittest.TestCase):

    def test_material_and_custom_vertex_examples_compile(self) -> None:
        root = Path(__file__).parents[2]
        expected_entries = {
            "blinn_phong.py": "@blinn_phong_fragment",
            "blinn_phong_vertices.py": "@skinned_vertex",
            "planet_terrain.py": "@planet_terrain_vertex",
            "shadowed_material.py": "@shadowed_fragment",
            "runtime_shader.py": "@runtime_fragment",
        }
        for filename, entry in expected_entries.items():
            with self.subTest(filename=filename):
                output = compile_file(root / "examples" / filename)
                self.assertIn(entry, output)


if __name__ == "__main__":
    unittest.main()
