from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import vernon_dsl as vd
from vernon_dsl.bundle import BundlePlan, CpuTargetOptions
from vernon_dsl.program_assets import ProgramCompileError


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

    def test_graphics_vjp_contract_is_unchanged(self) -> None:
        @vd.vertex
        def vertex(value: vd.f32) -> vd.f32:
            return value

        @vd.fragment
        def fragment(value: vd.f32) -> vd.f32:
            return value

        with self.assertRaisesRegex(ValueError, "custom rule set"):
            vd.ad.vjp((vertex, fragment), wrt=("value",))
        rules = vd.ad.rule_set(
            id="render/v1",
            rasterization=lambda: None,
            visibility=lambda: None,
            depth=lambda: None,
            blend=lambda: None,
            texture=lambda: None,
        )
        with self.assertRaisesRegex(ValueError, "does not accept compute Storage outputs"):
            vd.ad.vjp((vertex, fragment), wrt=("value",), outputs=("loss",), rules=rules)

    def test_bundle_schema_rejects_partial_differentiated_profiles(self) -> None:
        with self.assertRaisesRegex(ProgramCompileError, "require both"):
            BundlePlan(
                "partial",
                CpuTargetOptions(),
                (),
                (),
                (),
                transform={"kind": "vjp"},
            )


class AutodiffPipelineParsingTests(unittest.TestCase):
    """The VJP transform a cooked asset carries comes from the typed vd.ad.vjp expression.

    It used to be rebuilt from the AST and round-tripped through a dict. The typed ProgramTransformSpec is the same
    record with its own validation, so there is one description of a transform rather than two.
    """

    @staticmethod
    def _descriptor(directory: str, body: str, name: str = "asset") -> Any:
        from vernon_dsl._program_assets.cooking import _load_program_asset_declaration, _stage_path_descriptor

        source = Path(directory) / "asset.py"
        source.write_text(body, encoding="utf-8")
        return _stage_path_descriptor(_load_program_asset_declaration(source, name), source, name)

    def test_the_declared_transform_reaches_the_stage_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            descriptor = self._descriptor(
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
        self.assertIsNotNone(descriptor.transform)
        assert descriptor.transform is not None
        self.assertEqual(descriptor.transform["wrt"], ["value"])
        self.assertEqual(descriptor.transform["output_cotangents"], ["loss"])
        self.assertEqual(descriptor.stages["compute"].entry, "compute")
        manifest = json.loads(descriptor.canonical_manifest)
        self.assertEqual(manifest["transform"], descriptor.transform)

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
            named = self._descriptor(directory, body, "named")
            inline = self._descriptor(directory, body, "inline")
        self.assertEqual(named.transform, inline.transform)
        self.assertEqual(named.stages["compute"].entry, inline.stages["compute"].entry)
        self.assertEqual(named.variants, inline.variants)

    def test_a_compute_vjp_without_storage_outputs_is_refused_at_declaration(self) -> None:
        @vd.kernel
        def compute(value: vd.f32) -> None:
            pass

        with self.assertRaisesRegex(ValueError, "requires non-empty writable Storage outputs"):
            vd.ad.vjp(compute, wrt=("value",))


if __name__ == "__main__":
    unittest.main()
