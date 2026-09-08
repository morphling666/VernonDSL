from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import vernon_dsl as vd
from vernon_dsl._program_assets.capture import CapturedProgram, capture_program


class AutodiffDeclarationTests(unittest.TestCase):
    def test_compute_vjp_has_deterministic_storage_objective_identity(self) -> None:
        @vd.kernel
        def compute(
            value: vd.TensorView[vd.f32, (1,), vd.read],
            loss: vd.TensorView[vd.f32, (1,), vd.write],
        ) -> None:
            loss[0] = value[0] * value[0]

        first = vd.ad.vjp(compute, wrt=("value",), outputs=("loss",))
        second = vd.ad.vjp(compute, wrt=["value"], outputs=["loss"])
        self.assertEqual(first.transform.wrt, ("value",))
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

    def test_compute_vjp_rejects_missing_or_duplicate_paths(self) -> None:
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

    def test_graphics_tuple_vjp_is_not_an_authored_program_form(self) -> None:
        @vd.vertex
        def vertex(value: vd.f32) -> vd.f32:
            return value

        @vd.fragment
        def fragment(value: vd.f32) -> vd.f32:
            return value

        with self.assertRaisesRegex(TypeError, r"must use vd\.pipeline"):
            vd.ad.vjp((vertex, fragment), wrt=("value",))


class AutodiffProgramCaptureTests(unittest.TestCase):
    @staticmethod
    def _capture(directory: str, body: str, name: str = "asset") -> CapturedProgram:
        from vernon_dsl._program_assets.source import load_program_asset_declaration

        source = Path(directory) / "asset.py"
        source.write_text(body, encoding="utf-8")
        return capture_program(load_program_asset_declaration(source, name))

    def test_the_declared_transform_reaches_canonical_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            captured = self._capture(
                directory,
                """
import vernon_dsl as vd

@vd.kernel
def compute(
    value: vd.TensorView[vd.f32, (1,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss[0] = value[0] * value[0]

asset = vd.program_asset(
    id="compute/vjp",
    program=vd.ad.vjp(compute, wrt=("value",), outputs=("loss",)),
)
""",
            )
        self.assertEqual(captured.id, "compute/vjp")
        self.assertEqual(captured.variant_keys, ((),))
        self.assertEqual(captured.variants[0].ir.vjp_wrt, ("value",))
        self.assertTrue(captured.variants[0].ir.implementations)

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
