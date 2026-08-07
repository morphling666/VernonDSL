from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import vernon_dsl as vd
from vernon_dsl.ad import ProgramTransformSpec
from vernon_dsl.bundle import BundlePlan, CpuTargetOptions
from vernon_dsl.pipeline_assets import PipelineCompileError, parse_python_pipeline_asset


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

    def test_vjp_protocol_participates_in_identity(self) -> None:
        dynamic = ProgramTransformSpec("vjp", ("value",), output_cotangents=("loss",), protocol="dynamic_v2")
        legacy = ProgramTransformSpec("vjp", ("value",), protocol="legacy_fixed")
        self.assertNotEqual(dynamic.identity, legacy.identity)

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
        with self.assertRaisesRegex(PipelineCompileError, "require both"):
            BundlePlan(
                "partial",
                CpuTargetOptions(),
                (),
                (),
                (),
                transform={"kind": "vjp"},
            )


class AutodiffPipelineParsingTests(unittest.TestCase):
    def test_storage_outputs_are_parsed_without_executing_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd
raise RuntimeError("must not execute")

@vd.kernel
def compute(
    value: vd.TensorView[vd.f32, (1,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss[0] = value[0] * value[0]

asset = vd.pipeline_asset(
    id="compute/vjp",
    program=vd.ad.vjp(compute, wrt=("value",), outputs=("loss",)),
)
""",
                encoding="utf-8",
            )
            descriptor = parse_python_pipeline_asset(source, "asset")
            self.assertIsNotNone(descriptor.transform)
            assert descriptor.transform is not None
            self.assertEqual(descriptor.transform["wrt"], ["value"])
            self.assertEqual(descriptor.transform["output_cotangents"], ["loss"])
            manifest = json.loads(descriptor.canonical_manifest)
            self.assertEqual(manifest["transform"], descriptor.transform)

    def test_dynamic_pipeline_requires_declared_storage_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def compute(value: vd.f32) -> None:
    pass

asset = vd.pipeline_asset(id="compute/vjp", program=vd.ad.vjp(compute, wrt=("value",)))
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PipelineCompileError, "requires non-empty writable Storage outputs"):
                parse_python_pipeline_asset(source, "asset")


if __name__ == "__main__":
    unittest.main()
