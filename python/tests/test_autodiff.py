from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import vernon_dsl as vd
from language_contract_cases import case_by_id
from language_contract_runner import assert_verified_ir, contract_oracle
from language_contract_traceability import covers_case
from vernon_dsl._program_assets.capture import CapturedProgram, capture_program


class AutodiffDeclarationTests(unittest.TestCase):
    @covers_case("LANG-AD-004/first-order-compute-vjp", layers="F")
    def test_compute_vjp_has_deterministic_storage_objective_identity(self) -> None:
        case = case_by_id("LANG-AD-004/first-order-compute-vjp")

        @vd.kernel
        def compute(
            value: vd.TensorView[vd.f32, (1,), vd.read],
            loss: vd.TensorView[vd.f32, (1,), vd.write],
        ) -> None:
            loss[0] = value[0] * value[0]

        first = vd.ad.vjp(compute, wrt=("value",), outputs=("loss",))
        second = vd.ad.vjp(compute, wrt=["value"], outputs=["loss"])
        from vernon_dsl._program_assets.source import load_program_asset_declaration

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.py"
            path.write_text(case.source, encoding="utf-8")
            captured = capture_program(load_program_asset_declaration(path, "asset"))
        with contract_oracle(case), self.subTest(case=case.id):
            self.assertEqual(first.transform.wrt, ("value",))
            self.assertEqual(captured.id, "compute/vjp")
            self.assertEqual(captured.variants[0].ir.vjp_wrt, ("value",))
        self.assertEqual(first.transform.output_cotangents, ("loss",))
        self.assertEqual(first.transform.identity, second.transform.identity)

        balanced = vd.ad.vjp(
            compute,
            wrt=("value",),
            outputs=("loss",),
            planning_policy="balanced",
        )
        self.assertEqual(balanced.transform.planning_policy, "balanced")
        self.assertNotEqual(first.transform.identity, balanced.transform.identity)
        with self.assertRaisesRegex(ValueError, "planning policy"):
            vd.ad.vjp(compute, wrt=("value",), outputs=("loss",), planning_policy="heuristic")

    @covers_case("LANG-AD-001/missing-storage-output", layers="F")
    def test_compute_vjp_rejects_missing_or_duplicate_paths(self) -> None:
        missing_output_case = case_by_id("LANG-AD-001/missing-storage-output")
        assert missing_output_case.expected_diagnostic is not None
        with (
            contract_oracle(missing_output_case),
            self.assertRaisesRegex(ValueError, missing_output_case.expected_diagnostic),
        ):
            exec(compile(missing_output_case.source, f"{missing_output_case.name}.py", "exec"), {})

        @vd.kernel
        def compute(
            value: vd.TensorView[vd.f32, (1,), vd.read],
            loss: vd.TensorView[vd.f32, (1,), vd.write],
        ) -> None:
            loss[0] = value[0]

        with self.assertRaisesRegex(ValueError, "canonical source paths"):
            vd.ad.vjp(compute, wrt=("value[0]",), outputs=("loss",))
        with self.assertRaisesRegex(ValueError, "canonical source paths"):
            vd.ad.vjp(compute, wrt=("value.a速",), outputs=("loss",))
        with self.assertRaisesRegex(ValueError, "unique"):
            vd.ad.vjp(compute, wrt=("value", "value"), outputs=("loss",))
        with self.assertRaisesRegex(ValueError, "requires non-empty"):
            vd.ad.vjp(compute, wrt=("value",))
        with self.assertRaisesRegex(ValueError, "outputs paths must be unique"):
            vd.ad.vjp(compute, wrt=("value",), outputs=("loss", "loss"))

    @covers_case("LANG-AD-004/graphics-tuple", layers="F")
    def test_graphics_tuple_vjp_is_not_an_authored_program_form(self) -> None:
        case = case_by_id("LANG-AD-004/graphics-tuple")
        assert case.expected_diagnostic is not None
        with contract_oracle(case), self.assertRaisesRegex(TypeError, case.expected_diagnostic):
            exec(compile(case.source, f"{case.name}.py", "exec"), {})


class AutodiffProgramCaptureTests(unittest.TestCase):
    @staticmethod
    def _capture(directory: str, body: str, name: str = "asset") -> CapturedProgram:
        from vernon_dsl._program_assets.source import load_program_asset_declaration

        source = Path(directory) / "asset.py"
        source.write_text(body, encoding="utf-8")
        return capture_program(load_program_asset_declaration(source, name))

    @covers_case("LANG-AD-001/canonical-capture", layers="FI")
    def test_the_declared_transform_reaches_canonical_capture(self) -> None:
        case = case_by_id("LANG-AD-001/canonical-capture")
        with tempfile.TemporaryDirectory() as directory:
            captured = self._capture(directory, case.source)
        self.assertEqual(captured.id, "compute/vjp")
        self.assertEqual(captured.variant_keys, ((),))
        self.assertEqual(captured.variants[0].ir.vjp_wrt, ("value",))
        self.assertTrue(captured.variants[0].ir.implementations)
        assert_verified_ir(self, captured.variants[0].ir.mlir, case)
        for implementation in captured.variants[0].ir.implementations:
            assert_verified_ir(self, implementation.mlir)

    @covers_case("LANG-PROGRAM-001/module-composition", layers="FI")
    @covers_case("LANG-PAIR-012/module-fan-in-out-versions", layers="FI")
    def test_canonical_modules_reach_verified_program_and_implementation_ir(self) -> None:
        program_case = case_by_id("LANG-PROGRAM-001/module-composition")
        fan_out_case = case_by_id("LANG-PAIR-012/module-fan-in-out-versions")
        for case in (program_case, fan_out_case):
            with self.subTest(case=case.id), tempfile.TemporaryDirectory() as directory:
                captured = self._capture(directory, case.source)
                program = captured.variants[0].ir
                assert_verified_ir(self, program.mlir, case)
                for implementation in program.implementations:
                    assert_verified_ir(self, implementation.mlir)
                self.assertGreaterEqual(len(program.implementations), 1)
                if case.contract_id == "LANG-PAIR-012":
                    self.assertEqual(program.vjp_wrt, ("source",))
                    self.assertEqual(len(program.implementations), 2)
                    self.assertEqual(program.mlir.count('"vernon_program.compute"(%v0'), 2)
                    self.assertEqual(
                        program.mlir.count('vernon_program.operand_accesses = ["read", "write"]'),
                        2,
                    )
                    self.assertIn('vernon_program.vjp_outputs = ["square", "cube"]', program.mlir)

    def test_a_named_vjp_program_describes_the_same_transform_as_an_inline_one(self) -> None:
        body = """
import vernon_dsl as vd

@vd.kernel
def compute(
    value: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss[0] = value[0] * value[0]

compute_vjp = vd.ad.vjp(compute, wrt=("value",), outputs=("loss",))
named = vd.program_asset(id="compute/vjp", program=compute_vjp)
inline = vd.program_asset(
    id="compute/vjp",
    program=vd.ad.vjp(compute, wrt=("value",), outputs=("loss",)),
)
"""
        with tempfile.TemporaryDirectory() as directory:
            named = self._capture(directory, body, "named")
            inline = self._capture(directory, body, "inline")
        self.assertEqual(named.variant_keys, inline.variant_keys)
        self.assertEqual(named.variants[0].ir.identity, inline.variants[0].ir.identity)

    def test_a_compute_vjp_without_storage_outputs_is_refused_at_declaration(self) -> None:
        @vd.kernel
        def compute(value: vd.f32) -> None:
            pass

        with self.assertRaisesRegex(ValueError, "requires non-empty writable Storage outputs"):
            vd.ad.vjp(compute, wrt=("value",))


if __name__ == "__main__":
    unittest.main()
