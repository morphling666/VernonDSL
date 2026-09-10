from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from backend_test_matrix import CapabilityUnavailable, ProbeKind, ProbeResult
from language_contract_cases import (
    CONTRACT_CASE_GROUPS,
    LANGUAGE_CONTRACT_REGISTRY,
    AcceptanceSuite,
    RuntimeOracle,
    RuntimeOracleKind,
    case_by_id,
)
from language_contract_inventory import (
    acceptance_coverage_gaps,
    audit_coverage,
    coverage_gaps,
    inventory_layer_requirements,
)
from language_contract_pytest import reject_skipped_contract_test
from language_contract_traceability import (
    ContractTestBinding,
    collect_test_bindings,
    contract_test_bindings,
    covers_case,
    mark_case_observed,
)


class LanguageContractCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = Path(__file__).parents[2]
        inventory = cls.repository / "specs/testing/language_feature_inventory.md"
        cls.requirements = inventory_layer_requirements(inventory.read_text(encoding="utf-8"))
        cls.case_groups = CONTRACT_CASE_GROUPS
        cls.bindings = collect_test_bindings(
            sorted((cls.repository / "python/tests").glob("test_*.py")),
            case_groups=cls.case_groups,
        )

    def test_inventory_assigns_layers_to_every_contract_id(self) -> None:
        self.assertTrue(self.requirements)
        self.assertTrue(all(layers for layers in self.requirements.values()))

    def test_every_inventory_contract_has_a_canonical_case(self) -> None:
        registered = {case.contract_id for case in LANGUAGE_CONTRACT_REGISTRY.cases}
        self.assertEqual(registered, self.requirements.keys())
        contracts_with_supported_regions = {
            case.contract_id
            for case in LANGUAGE_CONTRACT_REGISTRY.cases
            if case.valid_regions and case.expected_diagnostic is None
        }
        self.assertEqual(contracts_with_supported_regions, self.requirements.keys())

    def test_registered_cases_reference_inventory_contracts(self) -> None:
        for case in LANGUAGE_CONTRACT_REGISTRY.cases:
            with self.subTest(reference=case.id):
                self.assertIn(case.contract_id, self.requirements)

    def test_declared_tests_cover_required_frontend_and_ir_regions(self) -> None:
        audit_coverage(
            self.requirements,
            LANGUAGE_CONTRACT_REGISTRY.cases,
            self.bindings,
            layers=frozenset({"F", "I"}),
        )

    def test_acceptance_scenarios_cover_required_compile_artifact_and_runtime_layers(self) -> None:
        self.assertEqual(
            acceptance_coverage_gaps(
                self.requirements,
                LANGUAGE_CONTRACT_REGISTRY.cases,
                LANGUAGE_CONTRACT_REGISTRY.acceptances,
            ),
            {},
        )

    def test_acceptance_assets_are_authored_repository_sources(self) -> None:
        for acceptance in LANGUAGE_CONTRACT_REGISTRY.acceptances:
            with self.subTest(acceptance=acceptance.id):
                source_path, separator, symbol = acceptance.asset_reference.partition(":")
                self.assertEqual(separator, ":")
                self.assertTrue(symbol.isidentifier())
                self.assertTrue((self.repository / source_path).is_file())

    def test_removing_acceptance_reopens_required_layers(self) -> None:
        without_fan_out = tuple(
            acceptance for acceptance in LANGUAGE_CONTRACT_REGISTRY.acceptances if acceptance.id != "fan_out_vjp"
        )
        gaps = acceptance_coverage_gaps(
            self.requirements,
            LANGUAGE_CONTRACT_REGISTRY.cases,
            without_fan_out,
        )
        self.assertEqual(gaps["LANG-PAIR-012"], frozenset({"C", "A", "R"}))

    def test_acceptance_oracle_must_be_executed_by_its_suite(self) -> None:
        acceptance = LANGUAGE_CONTRACT_REGISTRY.acceptances[0]
        invalid = replace(
            acceptance,
            suite=AcceptanceSuite.SYNCHRONIZATION,
            oracle=RuntimeOracle(RuntimeOracleKind.FLOAT_BUFFER, (1.0,)),
        )
        with self.assertRaisesRegex(AssertionError, "is not executed by suite"):
            acceptance_coverage_gaps(
                self.requirements,
                LANGUAGE_CONTRACT_REGISTRY.cases,
                (invalid,),
            )

    def test_acceptance_runtime_oracle_requires_observations(self) -> None:
        acceptance = LANGUAGE_CONTRACT_REGISTRY.acceptances[0]
        invalid = replace(acceptance, oracle=replace(acceptance.oracle, expected=()))
        with self.assertRaisesRegex(AssertionError, "has no expected observations"):
            acceptance_coverage_gaps(
                self.requirements,
                LANGUAGE_CONTRACT_REGISTRY.cases,
                (invalid,),
            )

    def test_fake_binding_that_does_not_consume_its_case_is_rejected(self) -> None:
        source = """
from language_contract_traceability import covers_case
class FakeTests:
    @covers_case("LANG-SCALAR-001/f32", layers="F")
    def test_fake(self):
        pass
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_fake.py"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "without calling case_by_id"):
                collect_test_bindings((path,), case_groups=self.case_groups)

    def test_unknown_or_unconsumed_case_group_is_rejected(self) -> None:
        source = """
from language_contract_traceability import covers_case_group
class FakeTests:
    @covers_case_group("MISSING_CASES", layers="I")
    def test_fake(self):
        assert_verified_ir(self, "module {}")
        for case in MISSING_CASES:
            pass
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_fake.py"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "unknown case group"):
                collect_test_bindings((path,), case_groups=self.case_groups)

    def test_declared_group_must_be_iterated_by_the_test(self) -> None:
        source = """
from language_contract_traceability import covers_case_group
class FakeTests:
    @covers_case_group("TYPE_PARSER_VALID_CASES", layers="I")
    def test_fake(self):
        assert_verified_ir(self, "module {}")
        return None
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_fake.py"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "without iterating that group"):
                collect_test_bindings((path,), case_groups=self.case_groups)

    def test_ir_binding_must_call_the_canonical_verifier(self) -> None:
        source = """
from language_contract_traceability import covers_case
class FakeTests:
    @covers_case("LANG-SCALAR-001/f32", layers="I")
    def test_fake(self):
        case = case_by_id("LANG-SCALAR-001/f32")
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_fake.py"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "canonical MLIR verifier"):
                collect_test_bindings((path,), case_groups=self.case_groups)

    def test_additional_observed_layers_do_not_create_a_failure(self) -> None:
        case = next(case for case in LANGUAGE_CONTRACT_REGISTRY.cases if case.id == "LANG-SCALAR-001/f32")
        binding = ContractTestBinding((case.id,), frozenset({"F", "I"}), "synthetic")
        self.assertEqual(
            coverage_gaps(
                {"LANG-SCALAR-001": frozenset({"I"})},
                (case,),
                (binding,),
                layers=frozenset({"F", "I"}),
            ),
            {},
        )

    def test_new_inventory_requirement_is_not_implicitly_covered(self) -> None:
        requirements = dict(self.requirements)
        requirements["LANG-NEW-001"] = frozenset({"F"})
        with self.assertRaisesRegex(AssertionError, "LANG-NEW-001"):
            coverage_gaps(
                requirements,
                LANGUAGE_CONTRACT_REGISTRY.cases,
                self.bindings,
                layers=frozenset({"F", "I"}),
            )

    def test_unknown_layer_declaration_is_rejected(self) -> None:
        source = """
from language_contract_traceability import covers_case
class FakeTests:
    @covers_case("LANG-SCALAR-001/f32", layers="C")
    def test_fake(self):
        case = case_by_id("LANG-SCALAR-001/f32")
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_fake.py"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "subset of FI"):
                collect_test_bindings((path,), case_groups=self.case_groups)

    def test_inventory_layer_grammar_rejects_malformed_and_duplicate_records(self) -> None:
        malformed = "### `LANG-TEST-001`\n\n- Layers: `F`, source guards.\n"
        with self.assertRaisesRegex(ValueError, "malformed layer declaration"):
            inventory_layer_requirements(malformed)

        duplicate = "### `LANG-TEST-001`\n\n- Layers: `F`.\n### `LANG-TEST-001`\n\n- Layers: `I`.\n"
        with self.assertRaisesRegex(ValueError, "duplicate inventory contract heading"):
            inventory_layer_requirements(duplicate)

    def test_retrieving_a_case_without_running_an_oracle_gets_no_credit(self) -> None:
        @covers_case("LANG-SCALAR-001/f32", layers="F")
        def fake() -> None:
            case_by_id("LANG-SCALAR-001/f32")
            if False:
                mark_case_observed("LANG-SCALAR-001/f32")

        with self.assertRaisesRegex(AssertionError, "did not execute an oracle"):
            fake()

    def test_decorators_preserve_immutable_binding_metadata(self) -> None:
        @covers_case("LANG-SCALAR-001/f32", layers="F")
        @covers_case("LANG-SCALAR-001/float-alias", layers="I")
        def fake() -> None:
            mark_case_observed("LANG-SCALAR-001/f32")
            mark_case_observed("LANG-SCALAR-001/float-alias")

        bindings = contract_test_bindings(fake)
        self.assertIsInstance(bindings, tuple)
        self.assertEqual(
            {case_id for binding in bindings for case_id in binding.case_ids},
            {"LANG-SCALAR-001/f32", "LANG-SCALAR-001/float-alias"},
        )
        fake()

    def test_contract_bound_skip_is_converted_to_failure(self) -> None:
        @covers_case("LANG-SCALAR-001/f32", layers="F")
        def bound() -> None:
            mark_case_observed("LANG-SCALAR-001/f32")

        report = SimpleNamespace(outcome="skipped", longrepr="reason", skipped=True)
        reject_skipped_contract_test(report, bound)
        self.assertEqual(report.outcome, "failed")
        self.assertIn("LANG-SCALAR-001/f32", report.longrepr)

        ordinary_report = SimpleNamespace(outcome="skipped", longrepr="reason", skipped=True)
        reject_skipped_contract_test(ordinary_report, lambda: None)
        self.assertEqual(ordinary_report.outcome, "skipped")

        capability_report = SimpleNamespace(outcome="skipped", longrepr="reason", skipped=True)
        capability = CapabilityUnavailable(ProbeResult(ProbeKind.CAPABILITY_UNSUPPORTED, "missing required feature"))
        reject_skipped_contract_test(capability_report, bound, capability)
        self.assertEqual(capability_report.outcome, "skipped")

    def test_unexecuted_negative_case_fails_the_audit(self) -> None:
        positive = next(case for case in LANGUAGE_CONTRACT_REGISTRY.cases if case.id == "LANG-SCALAR-001/f32")
        negative = next(case for case in LANGUAGE_CONTRACT_REGISTRY.cases if case.id == "LANG-SCALAR-001/unknown-type")
        binding = ContractTestBinding((positive.id,), frozenset({"F", "I"}), "synthetic")
        with self.assertRaisesRegex(AssertionError, "diagnostic cases were not executed"):
            audit_coverage(
                {"LANG-SCALAR-001": frozenset({"F", "I"})},
                (positive, negative),
                (binding,),
                layers=frozenset({"F", "I"}),
            )

    def test_removing_required_binding_reopens_region_layer_gaps(self) -> None:
        case_id = "LANG-SEM-001/category-model"
        without_case = tuple(binding for binding in self.bindings if case_id not in binding.case_ids)
        gaps = coverage_gaps(
            self.requirements,
            LANGUAGE_CONTRACT_REGISTRY.cases,
            without_case,
            layers=frozenset({"F", "I"}),
        )
        self.assertTrue(any(contract_id == "LANG-SEM-001" for contract_id, _ in gaps))


if __name__ == "__main__":
    unittest.main()
