from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vernon_dsl import CompileError, Compiler, compile_source
from vernon_dsl.compiler import FrontendCompileRequest
from vernon_dsl.frontend.analysis import dump_typed_model
from vernon_dsl.frontend.model import AccessMode, ConcreteType, StorageClass, TypedParameter


class LanguageVersionTests(unittest.TestCase):
    def test_tensor_addressability_is_parameter_storage_not_a_type_kind(self) -> None:
        element = ConcreteType("scalar", "f32")
        tensor = ConcreteType("tensor", "Tensor", (element, 4))
        parameter = TypedParameter(
            "output",
            tensor,
            StorageClass.ADDRESSABLE,
            AccessMode.READ_WRITE,
        )
        self.assertEqual(parameter.type.kind, "tensor")
        self.assertEqual(parameter.type.mlir, "tensor<4xf32>")
        self.assertEqual(parameter.storage, StorageClass.ADDRESSABLE)

    def test_frontend_and_semantic_identity_are_version_three(self) -> None:
        source = "from vernon_dsl import *\n@fragment\ndef main(value: float) -> float:\n    return value\n"
        output = compile_source(source, "version.py")
        self.assertIn("vernon.frontend_version = 3 : i64", output)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shader.py"
            path.write_text(source, encoding="utf-8")
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    path,
                    "main",
                    captured_constants=(("LIMIT", 3),),
                )
            )
            self.assertEqual(result.semantic_inputs["frontend_version"], 3)
            self.assertEqual(result.semantic_inputs["captured_constants"], [["LIMIT", "int", 3]])

    def test_semantic_identity_contains_concrete_helper_specializations(self) -> None:
        source = (
            "from vernon_dsl import *\n"
            "@func\n"
            "def identity(value):\n"
            "    return value\n"
            "@fragment\n"
            "def main(value: f64) -> f64:\n"
            "    return identity(value)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "specialization.py"
            path.write_text(source, encoding="utf-8")
            first = Compiler().compile_request(FrontendCompileRequest(path, "main"))
            second = Compiler().compile_request(FrontendCompileRequest(path, "main"))
        expected = [["identity", ["f64"], []]]
        self.assertEqual(first.semantic_inputs["helper_specializations"], expected)
        self.assertEqual(second.semantic_inputs["helper_specializations"], expected)
        self.assertEqual(dump_typed_model(first.typed_functions), dump_typed_model(second.typed_functions))

    def test_typed_model_records_effects_lvalues_and_branch_merges(self) -> None:
        source = (
            "from vernon_dsl import *\n"
            "@fragment\n"
            "def main(value: f32, condition: bool) -> f32:\n"
            "    result = value\n"
            "    if condition:\n"
            "        result = result + 1\n"
            "    else:\n"
            "        result = result + 2\n"
            "    return result\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "typed_model.py"
            path.write_text(source, encoding="utf-8")
            result = Compiler().compile_request(FrontendCompileRequest(path, "main"))
        function = next(function for function in result.typed_functions if function.symbol == "main")
        branch = function.body[1]
        self.assertEqual([(merge.name, merge.type.mlir) for merge in branch.branch_merges], [("result", "f32")])
        self.assertTrue(branch.children[0].lvalues)
        self.assertEqual(branch.children[0].effect.value, "write")
        self.assertIn('"operation": "add"', dump_typed_model(result.typed_functions))

    def test_removed_v2_spelling_and_array_fail_at_the_frontend(self) -> None:
        with self.assertRaisesRegex(CompileError, "unknown DSL decorator 'compute'"):
            compile_source(
                "from vernon_dsl import *\n@compute\ndef main(value: f32) -> f32:\n    return value\n",
                "compute.py",
            )
        with self.assertRaisesRegex(CompileError, "unknown DSL type constructor 'Array'"):
            compile_source(
                "from vernon_dsl import *\n@func\ndef main(value: Array[f32, 4]) -> f32:\n    return 0.0\n",
                "array.py",
            )

    def test_short_circuit_and_nested_return_are_rejected_consistently(self) -> None:
        with self.assertRaisesRegex(CompileError, "short-circuit semantics"):
            compile_source(
                "from vernon_dsl import *\n@func\ndef both(a: bool, b: bool) -> bool:\n    return a and b\n",
                "boolean.py",
            )
        with self.assertRaisesRegex(CompileError, "nested return is not supported"):
            compile_source(
                "from vernon_dsl import *\n@func\ndef stop(a: bool) -> f32:\n"
                "    while a:\n        if a:\n            return 1.0\n    return 0.0\n",
                "return.py",
            )


class NumericInferenceTests(unittest.TestCase):
    def test_safe_promotion_and_integer_true_division(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def mixed(iterations: i32, scale: f64) -> f64:\n"
            "    ratio = iterations / 2\n"
            "    return ratio + iterations * 0.02 + scale\n",
            "numeric.py",
        )
        self.assertIn("arith.divf", output)
        self.assertIn("arith.sitofp", output)
        self.assertIn("arith.extf", output)

    def test_complete_safe_scalar_promotion_lattice(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def promote(half: f16, single: f32, double: f64, signed: i32, unsigned: u32) -> f64:\n"
            "    a = half + single\n"
            "    b = a + double\n"
            "    c = signed + b\n"
            "    return unsigned + c\n",
            "promotions.py",
        )
        self.assertGreaterEqual(output.count("arith.extf"), 2)
        self.assertIn("arith.sitofp", output)
        self.assertIn("arith.uitofp", output)

    def test_unsafe_implicit_conversions_are_rejected(self) -> None:
        cases = (
            (
                "def bad(value: f64) -> f32:\n    return value\n",
                "unsafe implicit conversion from f64 to f32",
            ),
            (
                "def bad(value: f32) -> i32:\n    return value\n",
                "unsafe implicit conversion from f32 to i32",
            ),
            (
                "def bad(left: i32, right: u32) -> i32:\n    return left + right\n",
                "no safe common type",
            ),
            (
                "def bad(left: bool, right: bool) -> bool:\n    return left + right\n",
                "unsupported binary operation",
            ),
        )
        for index, (function, diagnostic) in enumerate(cases):
            with self.subTest(index=index), self.assertRaisesRegex(CompileError, diagnostic):
                compile_source(
                    f"from vernon_dsl import *\n@func\n{function}",
                    f"unsafe_{index}.py",
                )

    def test_literals_are_contextual_on_both_operand_sides(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def contextual(value: f64, iterations: i32) -> f64:\n"
            "    left = 0.25 * value\n"
            "    right = value * 2\n"
            "    ratio = 1 / iterations\n"
            "    return left + right + ratio\n",
            "contextual.py",
        )
        self.assertIn("arith.constant 0.25 : f64", output)
        self.assertIn("arith.extf", output)
        self.assertIn("arith.divf", output)

    def test_literals_are_contextual_in_calls_constructors_and_comparisons(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def takes_double(value: f64) -> f64:\n"
            "    return value\n"
            "@func\n"
            "def contexts(value: f64) -> f64:\n"
            "    called = takes_double(2)\n"
            "    vector = Vector([1, value])\n"
            "    compared = value > 0\n"
            "    if compared:\n"
            "        called = called + vector[0]\n"
            "    return called\n",
            "literal_contexts.py",
        )
        self.assertIn("func.call @takes_double", output)
        self.assertIn("tensor<2xf64>", output)
        self.assertIn("arith.cmpf", output)

    def test_scalar_literal_constraints_avoid_default_then_cast(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def takes_double(value: f64) -> f64:\n"
            "    return value\n"
            "@func\n"
            "def constrained(value: f64) -> f64:\n"
            "    return takes_double(2) + (3 + value)\n",
            "literal_constraints.py",
        )
        self.assertEqual(output.count("arith.constant 2.0 : f64"), 1)
        self.assertEqual(output.count("arith.constant 3.0 : f64"), 1)
        self.assertNotIn("arith.sitofp", output)

    def test_generic_literal_constraints_precede_specialization(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def annotated(value: f64):\n"
            "    return value\n"
            "@func\n"
            "def unconstrained(value):\n"
            "    return value\n"
            "@fragment\n"
            "def main() -> f64:\n"
            "    return annotated(1 + 2.0) + f64(unconstrained(3))\n",
            "generic_literal_constraints.py",
        )
        self.assertRegex(output, r"func\.func private @annotated__[a-f0-9]+[^(]*\(%arg0: f64")
        self.assertRegex(output, r"func\.func private @unconstrained__[a-f0-9]+[^(]*\(%arg0: i32")

    def test_partially_annotated_helper_rejects_unsafe_argument_and_result(self) -> None:
        cases = (
            (
                "def helper(value: i32):\n    return value\n",
                "value: f32",
                "cannot pass f32 as i32",
            ),
            (
                "def helper(value) -> i32:\n    return value\n",
                "value: f32",
                "cannot return f32 as i32",
            ),
        )
        for index, (helper, argument, diagnostic) in enumerate(cases):
            with self.subTest(index=index), self.assertRaisesRegex(CompileError, diagnostic):
                compile_source(
                    "from vernon_dsl import *\n"
                    f"@func\n{helper}"
                    "@fragment\n"
                    f"def main({argument}) -> i32:\n"
                    "    return helper(value)\n",
                    f"partial_annotation_{index}.py",
                )

    def test_unannotated_helpers_are_monomorphized_deterministically(self) -> None:
        source = (
            "import vernon_dsl as vd\n"
            "@vd.func\n"
            "def scale(value):\n"
            "    return value * 2\n"
            "@vd.fragment\n"
            "def single(value: vd.f32) -> vd.f32:\n"
            "    return scale(value)\n"
            "@vd.fragment\n"
            "def double(value: vd.f64) -> vd.f64:\n"
            "    return scale(value)\n"
        )
        first = compile_source(source, "specialize.py")
        second = compile_source(source, "specialize.py")
        self.assertEqual(first, second)
        self.assertEqual(first.count("func.func private @scale__"), 2)
        self.assertIn(": f32", first)
        self.assertIn(": f64", first)

    def test_helper_tensor_shapes_have_separate_specializations(self) -> None:
        output = compile_source(
            "import vernon_dsl as vd\n"
            "@vd.func\n"
            "def first(value):\n"
            "    return value[0]\n"
            "@vd.fragment\n"
            "def pair(value: vd.Tensor[vd.f32, (2,)]) -> vd.f32:\n"
            "    return first(value)\n"
            "@vd.fragment\n"
            "def triple(value: vd.Tensor[vd.f32, (3,)]) -> vd.f32:\n"
            "    return first(value)\n",
            "shape_specialization.py",
        )
        self.assertEqual(output.count("func.func private @first__"), 2)

    def test_inferred_void_helper_and_unresolved_calls_have_stable_behavior(self) -> None:
        output = compile_source(
            "import vernon_dsl as vd\n"
            "@vd.func\n"
            "def observe(value):\n"
            "    copy = value\n"
            "@vd.kernel\n"
            "def main(value: vd.f32) -> None:\n"
            "    observe(value)\n",
            "void_helper.py",
        )
        self.assertIn("func.func private @observe__", output)
        self.assertIn("func.call @observe__", output)

        with self.assertRaisesRegex(CompileError, "cannot infer call to 'missing'"):
            compile_source(
                "from vernon_dsl import *\n"
                "@func\n"
                "def unresolved(value):\n"
                "    return missing(value)\n"
                "@fragment\n"
                "def main(value: f32) -> f32:\n"
                "    return unresolved(value)\n",
                "unresolved.py",
            )

    def test_unreachable_conflicting_return_is_ignored_but_empty_value_conflict_is_rejected(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def first(value):\n"
            "    return value\n"
            "    return f64(0)\n"
            "@fragment\n"
            "def main(value: f32) -> f32:\n"
            "    return first(value)\n",
            "unreachable_return.py",
        )
        self.assertRegex(output, r"func\.func private @first__[a-f0-9]+.*-> \(f32\)")

        with self.assertRaisesRegex(CompileError, "mixes value and empty returns"):
            compile_source(
                "from vernon_dsl import *\n"
                "@func\n"
                "def bad(value) -> f32:\n"
                "    return\n"
                "@fragment\n"
                "def main(value: f32) -> f32:\n"
                "    return bad(value)\n",
                "empty_value_conflict.py",
            )

    def test_recursive_generic_specialization_is_rejected(self) -> None:
        with self.assertRaisesRegex(CompileError, "recursive helper specialization"):
            compile_source(
                "from vernon_dsl import *\n"
                "@func\n"
                "def recursive(value):\n"
                "    return recursive(value)\n"
                "@fragment\n"
                "def main(value: f32) -> f32:\n"
                "    return recursive(value)\n",
                "recursive_generic.py",
            )

    def test_imported_qualified_helper_specializes_per_feature_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "helpers.py").write_text(
                "from vernon_dsl import func\n@func\ndef identity(value):\n    return value\n",
                encoding="utf-8",
            )
            main = root / "main.py"
            main.write_text(
                "from vernon_dsl import *\n"
                "import helpers\n"
                'DOUBLE = feature("DOUBLE")\n'
                "@fragment\n"
                "def main(value: f32) -> f32:\n"
                "    selected = value\n"
                "    if DOUBLE:\n"
                "        selected = f32(helpers.identity(f64(value)))\n"
                "    else:\n"
                "        selected = helpers.identity(value)\n"
                "    return selected\n",
                encoding="utf-8",
            )
            disabled = Compiler().compile_request(FrontendCompileRequest(main, "main"))
            enabled = Compiler().compile_request(FrontendCompileRequest(main, "main", ("DOUBLE",)))

        disabled_keys = [
            argument_types for name, argument_types, _ in disabled.helper_specializations if name.endswith("identity")
        ]
        enabled_keys = [
            argument_types for name, argument_types, _ in enabled.helper_specializations if name.endswith("identity")
        ]
        self.assertEqual(disabled_keys, [("f32",)])
        self.assertEqual(enabled_keys, [("f64",)])

    def test_partial_helper_annotations_and_matrix_inference(self) -> None:
        output = compile_source(
            "import vernon_dsl as vd\n"
            "@vd.func\n"
            "def transform(matrix: vd.Tensor[vd.f32, (2, 2)], value):\n"
            "    return vd.matmul(matrix, value)\n"
            "@vd.fragment\n"
            "def main(value: vd.Tensor[vd.f32, (2,)]) -> vd.Tensor[vd.f32, (2,)]:\n"
            "    matrix = vd.Matrix([[1, 2.0], [3, 4]])\n"
            "    return transform(matrix, value)\n",
            "matrix.py",
        )
        self.assertIn("tensor<2x2xf32>", output)
        self.assertIn('name = "matmul"', output)

    def test_generic_struct_construction_and_field_inference(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@struct\n"
            "class Pair:\n"
            "    first: f64\n"
            "    second: f64\n"
            "@func\n"
            "def pair(value):\n"
            "    return Pair(value, 2)\n"
            "@fragment\n"
            "def main(value: f64) -> f64:\n"
            "    return pair(value).second\n",
            "generic_struct.py",
        )
        self.assertIn('!vernon.struct<"Pair">', output)
        self.assertIn("vernon.struct_get", output)

    def test_generic_buffer_store_infers_resource_and_widens_value(self) -> None:
        output = compile_source(
            "from typing import Annotated\n"
            "from vernon_dsl import *\n"
            "@func\n"
            "def store(output, index, value):\n"
            "    output[index] = value\n"
            "@kernel\n"
            "def main(\n"
            "    output: Annotated[Buffer[f32], resource(set=0, binding=0)],\n"
            "    index: u32,\n"
            "    value: f16,\n"
            ") -> None:\n"
            "    store(output, index, value)\n",
            "generic_buffer.py",
        )
        self.assertIn('!vernon.buffer<f32, "read_write">', output)
        self.assertIn("arith.extf", output)
        self.assertIn('name = "buffer_store"', output)

    def test_generic_buffer_store_rejects_read_only_resource(self) -> None:
        with self.assertRaisesRegex(CompileError, "cannot assign through a read-only buffer"):
            compile_source(
                "from typing import Annotated\n"
                "from vernon_dsl import *\n"
                "@func\n"
                "def store(output, value):\n"
                "    output[0] = value\n"
                "@kernel\n"
                "def main(\n"
                '    output: Annotated[Buffer[f32, "read"], resource(set=0, binding=0)],\n'
                "    value: f32,\n"
                ") -> None:\n"
                "    store(output, value)\n",
                "readonly_buffer.py",
            )

    def test_matrix_rejects_ragged_and_empty_literals(self) -> None:
        for index, expression in enumerate(("Matrix([])", "Matrix([[1], [2, 3]])")):
            with (
                self.subTest(expression=expression),
                self.assertRaisesRegex(
                    CompileError,
                    "Matrix",
                ),
            ):
                compile_source(
                    f"from vernon_dsl import *\n@func\ndef bad() -> mat2[f32]:\n    return {expression}\n",
                    f"bad_matrix_{index}.py",
                )

    def test_pythonic_and_legacy_aggregate_constructors_share_types(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def aggregates(value: f64) -> vec2[f64]:\n"
            "    pythonic = Vector([1, value])\n"
            "    legacy = vec2(1, value)\n"
            "    matrix = Matrix([[1, value], [3.0, 4]])\n"
            "    return pythonic + legacy + matmul(matrix, pythonic)\n",
            "aggregate_parity.py",
        )
        self.assertGreaterEqual(output.count("tensor<2xf64>"), 3)
        self.assertIn("tensor<2x2xf64>", output)

    def test_operator_and_intrinsic_power_share_types(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n@func\ndef powers(value: f64) -> f64:\n    return value ** 2 + pow(value, 2)\n",
            "power_parity.py",
        )
        self.assertEqual(output.count('name = "pow"'), 2)
        self.assertNotIn("arith.truncf", output)

    def test_branch_and_loop_carried_values_widen_safely(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def control_flow(condition: bool) -> f32:\n"
            "    branch = f16(1)\n"
            "    if condition:\n"
            "        branch = f32(2)\n"
            "    else:\n"
            "        branch = f16(3)\n"
            "    loop = f16(0)\n"
            "    running = condition\n"
            "    while running:\n"
            "        loop = loop + f32(1)\n"
            "        running = False\n"
            "    return branch + loop\n",
            "control_flow_widening.py",
        )
        self.assertIn("scf.if", output)
        self.assertIn("scf.while", output)
        self.assertIn("-> (f32)", output)

    def test_generic_loop_inference_converges_after_nested_specialization(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def increment(value):\n"
            "    return value + f32(1)\n"
            "@func\n"
            "def accumulate(value, running):\n"
            "    result = value\n"
            "    while running:\n"
            "        result = increment(result)\n"
            "        running = False\n"
            "    return result\n"
            "@fragment\n"
            "def main(value: f16, running: bool) -> f32:\n"
            "    return accumulate(value, running)\n",
            "generic_loop.py",
        )
        self.assertRegex(output, r"func\.func private @accumulate__[a-f0-9]+.*-> \(f32\)")
        self.assertRegex(output, r"func\.func private @increment__[a-f0-9]+.*-> \(f32\)")
        self.assertEqual(output.count("func.func private @increment__"), 1)

    def test_vector_constructor_and_intrinsic_method_share_canonical_lowering(self) -> None:
        output = compile_source(
            "import vernon_dsl as vd\n"
            "@vd.func\n"
            "def complex_sqr(z):\n"
            "    value = vd.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])\n"
            "    length = value.norm()\n"
            "    return value\n"
            "@vd.fragment\n"
            "def main(z: vd.Tensor[vd.f32, (2,)]) -> vd.Tensor[vd.f32, (2,)]:\n"
            "    return complex_sqr(z)\n",
            "vector.py",
        )
        self.assertIn('name = "construct"', output)
        self.assertIn('name = "dot"', output)
        self.assertIn("func.func private @complex_sqr__", output)

    def test_intrinsic_method_and_function_have_matching_result_types(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def lengths(value: vec2[f64]) -> f64:\n"
            "    return value.norm() + norm(value)\n",
            "norm_parity.py",
        )
        self.assertEqual(output.count('name = "dot"'), 2)
        self.assertEqual(output.count("math.sqrt"), 2)


class EntryAbiTests(unittest.TestCase):
    def test_entries_require_parameter_and_result_annotations(self) -> None:
        cases = (
            (
                "@kernel\ndef main(value) -> None:\n    pass\n",
                "entry argument 'value' requires a type annotation",
            ),
            (
                "@kernel\ndef main(value: f32):\n    pass\n",
                "entry function 'main' requires a result annotation",
            ),
        )
        for index, (function, diagnostic) in enumerate(cases):
            with self.subTest(index=index), self.assertRaisesRegex(CompileError, diagnostic):
                compile_source(
                    f"from vernon_dsl import *\n{function}",
                    f"entry_abi_{index}.py",
                )


if __name__ == "__main__":
    unittest.main()
