from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import vernon_dsl as vd
from native_support import native_available
from vernon_dsl import CompileError
from vernon_dsl.bundle import canonical_json
from vernon_dsl.compiler import compile_file
from vernon_dsl.program_assets import ProgramCompileError, cook_program_asset
from vernon_dsl.types import SpecializationAssignment, specialization_assignment, specialization_key

_REPOSITORY = Path(__file__).resolve().parents[2]
_FIXTURES = _REPOSITORY / "source" / "tests" / "fixtures"
_GRAPHICS_ASSET = _FIXTURES / "language_contract_graphics_asset.py"

TARGET = vd.specialization("TARGET", vd.format)
SCALE = vd.specialization("SCALE", vd.u32)


def _format(name: str, value: str) -> SpecializationAssignment:
    return SpecializationAssignment(name, "format", value)


class FormatAssignmentTests(unittest.TestCase):
    def test_a_color_format_is_the_canonical_value(self) -> None:
        self.assertEqual(specialization_key({TARGET: vd.rgba8_srgb}), (_format("TARGET", "rgba8_srgb"),))
        self.assertEqual(_format("TARGET", "rgba8_srgb").manifest["value"], {"tag": "format", "value": "rgba8_srgb"})

    def test_a_value_must_be_a_color_format(self) -> None:
        for value in (vd.d32_float, "rgba8_unorm", 3):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "requires a color texture format"):
                specialization_assignment(TARGET, value)


class FormatTargetTests(unittest.TestCase):
    def test_resolved_substitutes_the_key_format(self) -> None:
        targets = vd.target_formats(colors={0: TARGET, 1: vd.r32_uint})
        resolved = targets.resolved((_format("TARGET", "rgba8_srgb"),))
        self.assertEqual(resolved.colors, ((0, vd.rgba8_srgb), (1, vd.r32_uint)))
        self.assertEqual(targets.format_specializations, frozenset({"TARGET"}))

    def test_a_key_must_assign_exactly_the_named_formats(self) -> None:
        targets = vd.target_formats(colors={0: TARGET})
        with self.assertRaisesRegex(ValueError, "missing format specialization.*TARGET"):
            targets.resolved(())
        with self.assertRaisesRegex(ValueError, "do not name: OTHER"):
            targets.resolved((_format("OTHER", "rgba8_unorm"), _format("TARGET", "rgba8_unorm")))

    def test_only_a_format_specialization_names_a_color_target(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be declared as vd.format"):
            vd.target_formats(colors={0: SCALE})


class FormatSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        (self.root / "pyproject.toml").write_text('[project]\nname = "formats"\n', encoding="utf-8")

    def _write(self, name: str, source: str) -> Path:
        path = self.root / name
        path.write_text(source, encoding="utf-8")
        return path

    def test_device_code_cannot_read_a_format_specialization(self) -> None:
        source = self._write(
            "reads.py",
            "from vernon_dsl import *\n"
            'TARGET = specialization("TARGET", format)\n'
            "@fragment\n"
            "def main() -> f32:\n"
            "    if TARGET == 1:\n"
            "        return 1.0\n"
            "    return 0.0\n",
        )
        with self.assertRaisesRegex(CompileError, "not visible to device code"):
            compile_file(source, entry="main")

    def test_a_kernel_variant_cannot_assign_a_format(self) -> None:
        if not native_available():
            self.skipTest("native Vernon extension is not built")
        source = self._write(
            "kernel.py",
            "import vernon_dsl as vd\n"
            'TARGET = vd.specialization("TARGET", vd.format)\n'
            "@vd.kernel(workgroup_size=(1, 1, 1))\n"
            "def run(values: vd.TensorView[vd.f32, (vd.dyn,), vd.read_write]) -> None:\n"
            "    values[0] = 1.0\n"
            'asset = vd.program_asset(id="tests/kernel", program=run, variants=({TARGET: vd.rgba8_unorm},))\n',
        )
        with self.assertRaisesRegex(CompileError, "select graphics pipeline color target formats only"):
            cook_program_asset(program_asset=f"{source}:asset", output=self.root / "out", target="cpu")


class FormatAssetTests(unittest.TestCase):
    def test_lint_reads_format_variants_without_running_the_declaration(self) -> None:
        from vernon_dsl._program_assets.parsing import lint_python_program_asset

        lint = lint_python_program_asset(_GRAPHICS_ASSET, "format_specialization_asset")
        self.assertEqual(lint.variants, ((_format("TARGET", "rgba8_unorm"),), (_format("TARGET", "rgba8_srgb"),)))

    def test_lint_rejects_a_variant_value_that_is_not_a_format(self) -> None:
        from vernon_dsl._program_assets.parsing import lint_python_program_asset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text('[project]\nname = "assets"\n', encoding="utf-8")
            source = root / "asset.py"
            shutil.copyfile(_GRAPHICS_ASSET, source)
            source.write_text(
                source.read_text(encoding="utf-8").replace("{TARGET: vd.rgba8_srgb}", "{TARGET: vd.f32}"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "must name a texture format"):
                lint_python_program_asset(source, "format_specialization_asset")

    def test_graphics_asset_cooks_one_program_per_format(self) -> None:
        if not native_available():
            self.skipTest("native Vernon extension is not built")
        with tempfile.TemporaryDirectory() as directory:
            manifest = cook_program_asset(
                program_asset=f"{_GRAPHICS_ASSET}:format_specialization_asset",
                output=Path(directory),
                target="vulkan",
            )
            document = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(
            [variant["key"] for variant in document["variants"]],
            [
                [{"name": "TARGET", "value": {"tag": "format", "value": "rgba8_srgb"}}],
                [{"name": "TARGET", "value": {"tag": "format", "value": "rgba8_unorm"}}],
            ],
        )
        for variant, format in zip(document["variants"], ("rgba8_srgb", "rgba8_unorm"), strict=True):
            program = canonical_json(variant["program"])
            other = "rgba8_unorm" if format == "rgba8_srgb" else "rgba8_srgb"
            self.assertIn(f'"{format}"', program)
            self.assertNotIn(f'"{other}"', program)


if __name__ == "__main__":
    unittest.main()
