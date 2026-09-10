from __future__ import annotations

import unittest
from collections.abc import Iterator
from contextlib import contextmanager

from language_contract_cases import LanguageContractCase, MlirOracle
from language_contract_traceability import mark_case_observed
from vernon_dsl import CompileError, _native, compile_source


def assert_verified_ir(test: unittest.TestCase, mlir: str, case: LanguageContractCase | None = None) -> None:
    result = _native.Compiler().verify_program_result(mlir)
    test.assertTrue(result.ok, result.diagnostics)
    if case is not None:
        mark_case_observed(case.id)


def assert_frontend_ir(test: unittest.TestCase, case: LanguageContractCase) -> None:
    if case.expected is None:
        raise AssertionError(f"{case.id} has no IR oracle")
    if isinstance(case.expected, MlirOracle):
        oracle = case.expected
    else:
        required = (case.expected,) if isinstance(case.expected, str) else case.expected
        if not isinstance(required, tuple) or not all(isinstance(marker, str) for marker in required):
            raise AssertionError(f"{case.id} has an invalid IR oracle")
        oracle = MlirOracle(required=required)
    if any(count < 0 for _, count in oracle.counts):
        raise AssertionError(f"{case.id} has an invalid IR oracle")
    output = compile_source(case.source, f"{case.name}.py")
    assert_verified_ir(test, output)
    with test.subTest(case=case.id):
        for marker in oracle.required:
            test.assertIn(marker, output)
        for marker in oracle.forbidden:
            test.assertNotIn(marker, output)
        for marker, count in oracle.counts:
            test.assertEqual(output.count(marker), count)
    mark_case_observed(case.id)


def assert_frontend_rejects(test: unittest.TestCase, case: LanguageContractCase) -> None:
    if case.expected_diagnostic is None:
        raise AssertionError(f"{case.id} has no expected diagnostic")
    with test.subTest(case=case.id), test.assertRaisesRegex(CompileError, case.expected_diagnostic):
        compile_source(case.source, f"{case.name}.py")
    mark_case_observed(case.id)


@contextmanager
def contract_oracle(case: LanguageContractCase) -> Iterator[None]:
    yield
    mark_case_observed(case.id)
