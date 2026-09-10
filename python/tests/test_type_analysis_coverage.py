from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from language_contract_cases import (
    INFERENCE_CALL_INVALID_CASES,
    INFERENCE_STATEMENT_INVALID_CASES,
    METADATA_INVALID_CASES,
    METADATA_VALID_CASES,
    TYPE_PARSER_INVALID_CASES,
    TYPE_PARSER_VALID_CASES,
    audit_case_registry,
)
from language_contract_runner import assert_frontend_rejects, contract_oracle
from language_contract_traceability import covers_case_group
from vernon_dsl import compile_source
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
    def error(node: ast.AST, message: str) -> Exception:
        del node
        return ValueError(message)


class TypeParserCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = TypeParser(_Context())

    @covers_case_group("TYPE_PARSER_VALID_CASES", layers="F")
    def test_type_constructor_success_paths(self) -> None:
        for case in TYPE_PARSER_VALID_CASES:
            with contract_oracle(case), self.subTest(case=case.id):
                self.assertEqual(self.parser.parse_type(expression(case.source)).mlir, case.expected)
        storage = self.parser.parse_type(expression("TensorStorage[f32]"))
        self.assertEqual((storage.kind, storage.arguments), ("tensor_storage", (ConcreteType("scalar", "f32"),)))

    @covers_case_group("TYPE_PARSER_INVALID_CASES", layers="F")
    def test_type_constructor_diagnostics(self) -> None:
        for case in TYPE_PARSER_INVALID_CASES:
            assert case.expected_diagnostic is not None
            with (
                contract_oracle(case),
                self.subTest(case=case.id),
                self.assertRaisesRegex(ValueError, case.expected_diagnostic),
            ):
                self.parser.parse(expression(case.source))

    def test_contract_case_registry_is_well_formed(self) -> None:
        inventory = (Path(__file__).parents[2] / "specs/testing/language_feature_inventory.md").read_text(
            encoding="utf-8"
        )
        audit_case_registry(frozenset(re.findall(r"\bLANG-[A-Z0-9-]+\b", inventory)))

    @covers_case_group("METADATA_VALID_CASES", layers="F")
    def test_metadata_success_paths(self) -> None:
        for case in METADATA_VALID_CASES:
            with contract_oracle(case), self.subTest(case=case.id):
                parsed = self.parser.parse(expression(case.source))
                self.assertEqual((parsed.metadata[0].kind, parsed.metadata[0].arguments), case.expected)

    @covers_case_group("METADATA_INVALID_CASES", layers="F")
    def test_metadata_diagnostics(self) -> None:
        for case in METADATA_INVALID_CASES:
            assert case.expected_diagnostic is not None
            with (
                contract_oracle(case),
                self.subTest(case=case.id),
                self.assertRaisesRegex(ValueError, case.expected_diagnostic),
            ):
                self.parser.parse(expression(case.source))


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
    @covers_case_group("INFERENCE_STATEMENT_INVALID_CASES", layers="F")
    def test_return_and_statement_diagnostics(self) -> None:
        for case in INFERENCE_STATEMENT_INVALID_CASES:
            assert_frontend_rejects(self, case)

    @covers_case_group("INFERENCE_CALL_INVALID_CASES", layers="F")
    def test_call_diagnostics(self) -> None:
        for case in INFERENCE_CALL_INVALID_CASES:
            assert_frontend_rejects(self, case)


if __name__ == "__main__":
    unittest.main()
