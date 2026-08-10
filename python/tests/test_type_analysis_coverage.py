from __future__ import annotations

import ast
import unittest

from vernon_dsl import CompileError, compile_source
from vernon_dsl.frontend.model import ConcreteType, LiteralType, TypedFunctionInstance
from vernon_dsl.frontend.type_parser import TypeParser
from vernon_dsl.frontend.type_solver import (
    can_convert,
    common_type,
    contextualize,
    default_type,
    describe,
    element_type,
    literal,
    scalar,
)
from vernon_dsl.language.scalar_types import can_implicitly_convert, canonical_scalar, common_scalar


def expression(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


class _Context:
    structs = {"Record": object()}

    @staticmethod
    def error(_node: ast.AST, message: str) -> Exception:
        return ValueError(message)


class TypeParserCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = TypeParser(_Context())

    def test_type_constructor_success_paths(self) -> None:
        expected = {
            "f32": "f32",
            "float": "f32",
            "Record": '!vernon.struct<"Record">',
            "Sampler": "!vernon.sampler",
            "Tensor[f32, (2, 4)]": "tensor<2x4xf32>",
            "Vector[f16, 4]": "tensor<4xf16>",
            "Matrix[f32, 3, 3]": "tensor<3x3xf32>",
            "TensorView[f32, (dyn,), read]": '!vernon.tensor_view<f32, [-1], "read", "device">',
            "TensorView[f32, (4, dyn), read_write]": '!vernon.tensor_view<f32, [4, -1], "read_write", "device">',
            'Texture["cube", f32]': '!vernon.texture<"cube", f32, "unknown", "sampled">',
        }
        for source, mlir in expected.items():
            with self.subTest(source=source):
                self.assertEqual(self.parser.parse_type(expression(source)).mlir, mlir)
        storage = self.parser.parse_type(expression("TensorStorage[f32]"))
        self.assertEqual((storage.kind, storage.arguments), ("tensor_storage", (ConcreteType("scalar", "f32"),)))

    def test_type_constructor_diagnostics(self) -> None:
        cases = {
            "None": "None is only valid",
            "Annotated[f32]": "Annotated requires",
            "Missing": "unknown DSL type",
            "Tensor[f32]": "Tensor requires",
            "Tensor[f32, 0]": "positive integer",
            "Tensor[f32, True]": "positive integer",
            "Tensor[f32, (None, 4)]": "positive integer",
            "vec[3, f32]": "unknown DSL type constructor",
            "mat[2, 3, f64]": "unknown DSL type constructor",
            "vec2[f32]": "unknown DSL type constructor",
            "mat4[f32]": "unknown DSL type constructor",
            "Vector[f32, f64, 2]": "Vector requires",
            "Tuple[Sampler]": "Tuple elements must be ABI-stable",
            "Tensor[Sampler, 2]": "Tensor element type must be an ABI-stable",
            "TensorStorage[f32, f32]": "TensorStorage requires one",
            "TensorStorage[Sampler]": "TensorStorage element type must be an ABI-stable",
            "Buffer[f32]": "unknown DSL type constructor 'Buffer'",
            "TensorView[f32, read]": "TensorView requires",
            "TensorView[f32, 1, read]": "TensorView shape",
            "TensorView[f32, (dyn,), missing]": "TensorView access",
            "Texture[f32]": "Texture requires",
            'Texture["1d", f32]': "texture dimension",
            "Texture[value(), f32]": "string literal or name",
            "Unknown[f32]": "unknown DSL type constructor",
        }
        for source, message in cases.items():
            with self.subTest(source=source):
                if message == "invalid syntax":
                    with self.assertRaises(SyntaxError):
                        expression(source)
                else:
                    with self.assertRaisesRegex(ValueError, message):
                        self.parser.parse(expression(source))

    def test_metadata_success_paths(self) -> None:
        expected = {
            "Annotated[f32, attribute()]": ("attribute", (-1, 0)),
            "Annotated[f32, attribute(3)]": ("attribute", (3, 0)),
            "Annotated[f32, attribute(divisor=2)]": ("attribute", (-1, 2)),
            "Annotated[f32, attribute(location=3, divisor=2)]": ("attribute", (3, 2)),
            'Annotated[f32, builtin("position")]': ("builtin", ("position",)),
            "Annotated[f32, uniform()]": ("uniform", ()),
            "Annotated[f32, uniform(set=1, binding=2)]": ("uniform", (1, 2)),
            "Annotated[f32, varying()]": ("varying", ()),
            "Annotated[f32, resource(set=1, binding=2)]": ("resource", (1, 2)),
        }
        for source, metadata in expected.items():
            with self.subTest(source=source):
                parsed = self.parser.parse(expression(source))
                self.assertEqual((parsed.metadata[0].kind, parsed.metadata[0].arguments), metadata)

    def test_metadata_diagnostics(self) -> None:
        cases = {
            "Annotated[f32, marker]": "metadata must be a call",
            "Annotated[f32, unknown()]": "unknown annotation metadata",
            "Annotated[f32, location(0)]": "unknown annotation metadata",
            "Annotated[f32, instance(location=0)]": "unknown annotation metadata",
            "Annotated[f32, varying(1)]": "wrong number",
            "Annotated[f32, attribute(divisor=-1)]": "integer or string",
            "Annotated[f32, resource(**opts)]": r"\*\*kwargs",
            "Annotated[f32, resource(set=1)]": "wrong number",
            "Annotated[f32, builtin(value=0)]": "does not accept keyword",
            "Annotated[f32, attribute('position')]": "location must be non-negative",
            "Annotated[f32, attribute(0, 'instance')]": "divisor must be non-negative",
        }
        for source, message in cases.items():
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, message):
                    self.parser.parse(expression(source))


class TypeSolverCoverageTests(unittest.TestCase):
    def test_literal_context_and_defaults(self) -> None:
        integer = literal(1)
        floating = literal(1.0)
        boolean = literal(True)
        self.assertEqual(boolean, scalar("bool"))
        self.assertEqual(default_type(integer), scalar("i32"))
        self.assertEqual(default_type(floating), scalar("f32"))
        self.assertEqual(default_type(scalar("f64")), scalar("f64"))
        self.assertEqual(contextualize(integer, scalar("f64")), scalar("f64"))
        self.assertEqual(contextualize(floating, scalar("i32")), scalar("f32"))
        self.assertEqual(contextualize(integer, scalar("bool")), scalar("i32"))
        self.assertEqual(contextualize(integer, ConcreteType("struct", "S")), scalar("i32"))
        self.assertEqual(contextualize(integer, ConcreteType("scalar", "opaque")), scalar("i32"))

    def test_common_type_success_and_rejection_paths(self) -> None:
        f32 = scalar("f32")
        f64 = scalar("f64")
        i32 = scalar("i32")
        tensor2 = ConcreteType("tensor", "Tensor", (f32, 2))
        tensor3 = ConcreteType("tensor", "Tensor", (f32, 3))
        struct = ConcreteType("struct", "S")
        self.assertEqual(element_type(tensor2), f32)
        self.assertEqual(element_type(f64), f64)
        self.assertEqual(common_type(LiteralType("integer", 1), LiteralType("integer", 2)), LiteralType("integer", 0))
        self.assertEqual(
            common_type(LiteralType("integer", 1), LiteralType("integer", 2), division=True),
            LiteralType("floating", 0.0),
        )
        self.assertEqual(common_type(LiteralType("floating", 1.0), f64), f64)
        self.assertEqual(common_type(i32, LiteralType("floating", 1.0)), scalar("f32"))
        self.assertEqual(common_type(tensor2, i32), tensor2)
        self.assertEqual(common_type(i32, tensor2), tensor2)
        self.assertEqual(common_type(struct, struct), struct)
        self.assertIsNone(common_type(struct, ConcreteType("struct", "T")))
        self.assertIsNone(common_type(tensor2, tensor3))
        self.assertIsNone(common_type(ConcreteType("tensor", "Tensor", (struct, 2)), tensor2))
        self.assertIsNone(common_type(scalar("bool"), i32))

    def test_conversion_and_description_paths(self) -> None:
        f32 = scalar("f32")
        f64 = scalar("f64")
        tensor2f32 = ConcreteType("tensor", "Tensor", (f32, 2))
        tensor2f64 = ConcreteType("tensor", "Tensor", (f64, 2))
        tensor3f64 = ConcreteType("tensor", "Tensor", (f64, 3))
        tuple_f32 = ConcreteType("tuple", "Tuple", (f32,))
        tuple_f64 = ConcreteType("tuple", "Tuple", (f64,))
        self.assertTrue(can_convert(LiteralType("integer", 1), f32))
        self.assertTrue(can_convert(f32, f64))
        self.assertTrue(can_convert(tuple_f32, tuple_f64))
        self.assertFalse(can_convert(tuple_f32, ConcreteType("tuple", "Tuple", (f64, f64))))
        self.assertTrue(can_convert(tensor2f32, tensor2f64))
        self.assertFalse(can_convert(f32, tensor2f32))
        self.assertFalse(can_convert(tensor2f32, tensor3f64))
        self.assertFalse(can_convert(ConcreteType("struct", "S"), ConcreteType("struct", "T")))
        self.assertEqual(describe(f32), "f32")
        self.assertEqual(describe(LiteralType("integer", 1)), "integer literal")

    def test_model_and_scalar_utility_edges(self) -> None:
        self.assertEqual(ConcreteType("index", "index").mlir, "index")
        self.assertEqual(ConcreteType("void", "void").mlir, "none")
        self.assertTrue(ConcreteType("index", "index").is_integer)
        self.assertTrue(ConcreteType("tensor", "Tensor", (scalar("i32"), 2)).is_integer)
        with self.assertRaisesRegex(AssertionError, "unknown type kind"):
            _ = ConcreteType("mystery", "mystery").mlir
        function = ast.parse("def helper():\n    pass\n").body[0]
        assert isinstance(function, ast.FunctionDef)
        instance = TypedFunctionInstance("module.helper", "helper", (), None, ("FEATURE",), function)
        self.assertEqual(instance.specialization_key, ("module.helper", (), ("FEATURE",)))
        self.assertIsNone(canonical_scalar("missing"))
        self.assertEqual(canonical_scalar("float"), "f32")
        self.assertEqual(common_scalar("i32", "f32"), "f32")
        self.assertEqual(common_scalar("bool", "bool"), "bool")
        self.assertIsNone(common_scalar("bool", "bool", true_division=True))
        self.assertTrue(can_implicitly_convert("f32", "f32"))

    def test_resource_helper_annotations_are_specialized(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def touch(value):\n"
            "    return\n"
            "@fragment\n"
            "def main(image: Texture['2d', f32], sampler: Sampler, uv: Vector[f32, 2]) -> Vector[f32, 4]:\n"
            "    touch(image)\n"
            "    touch(sampler)\n"
            "    return texture_sample(image, sampler, uv)\n",
            "resource_specialization.py",
        )
        self.assertEqual(output.count("func.func private @touch__"), 2)

    def test_inferred_tuple_helper_annotation(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def pair(value):\n"
            "    return (value, value)\n"
            "@fragment\n"
            "def main(value: f32) -> f32:\n"
            "    return pair(value)[0]\n",
            "tuple_specialization.py",
        )
        self.assertIn("func.func private @pair__", output)


class InferenceDiagnosticCoverageTests(unittest.TestCase):
    def assert_compile_error(self, body: str, message: str) -> None:
        with self.assertRaisesRegex(CompileError, message):
            compile_source("from vernon_dsl import *\n" + body, "coverage_case.py")

    def test_return_and_statement_diagnostics(self) -> None:
        cases = (
            (
                "@func\ndef bad(value: f32) -> None:\n    return value\n"
                "@fragment\ndef main(value: f32) -> f32:\n    bad(value)\n    return value\n",
                "void function.*returns a value",
            ),
            (
                "@func\ndef bad(value: f32) -> f32:\n    value + 1\n"
                "@fragment\ndef main(value: f32) -> f32:\n    return bad(value)\n",
                "requires a return value",
            ),
            (
                "@func\ndef bad(value: f32) -> f32:\n    return\n"
                "@fragment\ndef main(value: f32) -> f32:\n    return bad(value)\n",
                "requires a return value",
            ),
            (
                "@fragment\ndef main(value: f32) -> f32:\n    left, right = value\n    return value\n",
                "assignment target",
            ),
            (
                "@fragment\ndef main(value: f32) -> f32:\n    value[0] = 1\n    return value\n",
                "writable Storage",
            ),
            (
                "@fragment\ndef main(value: f32) -> f32:\n    result: i32 = value\n    return value\n",
                "cannot infer assignment",
            ),
            (
                "@fragment\ndef main(value: f32) -> f32:\n    value += True\n    return value\n",
                "augmented assignment",
            ),
            (
                "@fragment\ndef main(value: f32) -> f32:\n    result = [value]\n    return value\n",
                "cannot infer expression syntax",
            ),
            (
                "@fragment\ndef main(value: Vector[f32, 2], index: f32) -> f32:\n    return value[index]\n",
                "index must be an integer",
            ),
        )
        for body, message in cases:
            with self.subTest(message=message):
                self.assert_compile_error(body, message)

    def test_call_diagnostics(self) -> None:
        cases = (
            (
                "@func\ndef helper(value: f32) -> f32:\n    return value\n"
                "@fragment\ndef main(value: f32) -> f32:\n    return helper()\n",
                "expects 1 arguments",
            ),
            (
                "@func\ndef helper(value: i32) -> i32:\n    return value\n"
                "@fragment\ndef main(value: f32) -> i32:\n    return helper(value)\n",
                "cannot pass",
            ),
            (
                "@fragment\ndef main(value: f32) -> f32:\n    return resolution(value)[0]\n",
                "does not accept arguments",
            ),
            (
                "@struct\nclass Pair:\n    x: f32\n    y: f32\n"
                "@fragment\ndef main(value: f32) -> f32:\n    return Pair(value).x\n",
                "constructor requires 2",
            ),
            (
                "@fragment\ndef main(value: f32) -> f32:\n    return Vector([])[0]\n",
                "requires a non-empty sequence literal",
            ),
            (
                "@fragment\ndef main(value: f32) -> f32:\n    return Vector([value, True])[0]\n",
                "elements have incompatible types",
            ),
            (
                "@fragment\ndef main(value: f32) -> f32:\n    return matmul(value)\n",
                "matmul requires two",
            ),
            (
                "@fragment\ndef main(value: f32) -> f32:\n    return matmul(value, value)\n",
                "left operand must be a non-scalar Tensor",
            ),
            (
                "@fragment\ndef main(value: Matrix[f32, 2, 2]) -> f32:\n    return matmul(value, 1)\n",
                "right operand must be a non-scalar Tensor",
            ),
            (
                "@fragment\ndef main(value: f32) -> f32:\n    return unknown(value)\n",
                "cannot infer call",
            ),
            (
                "@fragment\ndef main(value: f32) -> f32:\n    return vec2(value, value)[0]\n",
                "cannot infer call to 'vec2'",
            ),
        )
        for body, message in cases:
            with self.subTest(message=message):
                self.assert_compile_error(body, message)


if __name__ == "__main__":
    unittest.main()
