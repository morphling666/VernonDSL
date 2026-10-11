from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Annotated, get_args, get_origin

import numpy as np
import vernon_dsl as vd
from backend_test_matrix import BackendRequirements, backend_row, probe_backend, require_available
from static_specialization_shader import MODE, TINTED, mode_fragment, mode_kernel, mode_vertex
from vernon_dsl.bundle import canonical_json
from vernon_dsl.compiler import compile_file
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.frontend.runtime_types import resolved_annotations
from vernon_dsl.program_assets import ProgramCompileError, cook_program_asset
from vernon_dsl.types import SpecializationAssignment

_CONFIG = 'from vernon_dsl import *\nMODE = specialization("MODE", u32)\n'
_SHADER = """from vernon_dsl import *
import config

TINT = feature("TINT")

@func
def doubled(x: f32) -> f32:
    return x * 2.0

@fragment
def main(
    a: f32,
    b: When[config.MODE == 2, f32],
    c: When[TINT and config.MODE != 2, f32],
) -> f32:
    if config.MODE == 2:
        return doubled(a) + b
    elif config.MODE > 2 and config.MODE % 2 == 1:
        return undefined_helper(a)
    scale = 3.0 if config.MODE == 0 else 1.0
    if TINT:
        return a * scale + c
    return a * scale
"""


def _mode(value: int) -> SpecializationAssignment:
    return SpecializationAssignment("MODE", "u32", value)


def _native_available() -> bool:
    try:
        from vernon_dsl._program_assets.compile_orchestration import _native_module

        _native_module()
        return True
    except ProgramCompileError:
        return False


class StaticSpecializationCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        (self.root / "config.py").write_text(_CONFIG, encoding="utf-8")

    def _source(self, text: str, name: str = "shader.py") -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_integer_branches_select_callees_and_skip_discarded_diagnostics(self) -> None:
        source = self._source(_SHADER)

        selected = compile_file(source, specializations=(_mode(2),), entry="main")
        self.assertIn("func.call @doubled", selected)
        self.assertIn('vernon.source_name = "b"', selected)
        self.assertNotIn('vernon.source_name = "c"', selected)

        fallback = compile_file(source, specializations=(_mode(0),), entry="main")
        self.assertNotIn("@doubled", fallback)
        self.assertNotIn('vernon.source_name = "b"', fallback)
        self.assertIn("arith.constant 3.0 : f32", fallback)

        tinted = compile_file(
            source,
            specializations=(_mode(4), SpecializationAssignment("TINT", "bool", True)),
            entry="main",
        )
        self.assertNotIn('vernon.source_name = "b"', tinted)
        self.assertIn('vernon.source_name = "c", vernon.dtype = "f32"', tinted)
        self.assertIn("vernon.location = 2 : i64", tinted, "interface locations are assigned before evaluation")
        self.assertIn("arith.constant 1.0 : f32", tinted)

        with self.assertRaisesRegex(CompileError, "undefined_helper"):
            compile_file(source, specializations=(_mode(3),), entry="main")

    def test_whole_module_compilation_accepts_scalar_specializations(self) -> None:
        output = compile_file(self._source(_SHADER), specializations=(_mode(2),))
        self.assertIn("func.call @doubled", output)

    def test_unassigned_scalar_specialization_is_rejected_where_used(self) -> None:
        with self.assertRaisesRegex(CompileError, "missing specialization 'MODE'"):
            compile_file(self._source(_SHADER), entry="main")

    def test_assignment_kind_must_match_the_declaration(self) -> None:
        with self.assertRaisesRegex(CompileError, "declared as u32, not i32"):
            compile_file(
                self._source(_SHADER),
                specializations=(SpecializationAssignment("MODE", "i32", 2),),
                entry="main",
            )

    def test_condition_mixing_runtime_values_is_rejected(self) -> None:
        source = self._source(
            "from vernon_dsl import *\n"
            "import config\n"
            "@fragment\n"
            "def main(x: f32) -> f32:\n"
            "    if config.MODE == 2 and x > 0.0:\n"
            "        return x\n"
            "    return 0.0\n"
        )
        with self.assertRaisesRegex(CompileError, "specialization condition mixes runtime values"):
            compile_file(source, specializations=(_mode(2),), entry="main")

    def test_name_bound_only_in_a_discarded_branch_is_undefined(self) -> None:
        source = self._source(
            "from vernon_dsl import *\n"
            "import config\n"
            "@fragment\n"
            "def main(x: f32) -> f32:\n"
            "    if config.MODE == 1:\n"
            "        y = x\n"
            "    return y\n"
        )
        self.assertIn("func.return", compile_file(source, specializations=(_mode(1),), entry="main"))
        with self.assertRaises(CompileError):
            compile_file(source, specializations=(_mode(0),), entry="main")

    def test_disabled_predicate_value_is_rejected_outside_its_guard(self) -> None:
        source = self._source(
            "from vernon_dsl import *\n"
            "import config\n"
            "@fragment\n"
            "def main(x: f32, y: When[config.MODE > 1, f32]) -> f32:\n"
            "    return x + y\n"
        )
        self.assertIn("arith.addf", compile_file(source, specializations=(_mode(2),), entry="main"))
        with self.assertRaisesRegex(CompileError, "disabled value 'y'"):
            compile_file(source, specializations=(_mode(1),), entry="main")

    def test_when_condition_must_be_compile_time(self) -> None:
        source = self._source(
            "from vernon_dsl import *\n@fragment\ndef main(x: f32, y: When[unknown_flag, f32]) -> f32:\n    return x\n"
        )
        with self.assertRaisesRegex(CompileError, "When condition must be a compile-time expression"):
            compile_file(source, entry="main")


class HostAnnotationTests(unittest.TestCase):
    def test_compound_predicates_resolve_to_the_wrapped_type(self) -> None:
        key = (SpecializationAssignment("MODE", "u32", 3), SpecializationAssignment("TINTED", "bool", True))
        tint = resolved_annotations(mode_fragment.function, key)["tint"]
        self.assertIs(get_origin(tint), Annotated)
        self.assertEqual(get_args(tint)[0], vd.Vector[vd.f32, 4])
        scale = resolved_annotations(mode_kernel._function, (SpecializationAssignment("MODE", "u32", 1),))["scale"]
        self.assertIs(scale, vd.f32)

    def test_unassigned_specializations_still_evaluate(self) -> None:
        self.assertIs(resolved_annotations(mode_kernel._function)["scale"], vd.f32)

    def test_eager_specialization_operators_explain_deferred_annotations(self) -> None:
        with self.assertRaisesRegex(TypeError, "from __future__ import annotations"):
            MODE > 1  # noqa: B015
        with self.assertRaisesRegex(TypeError, "from __future__ import annotations"):
            bool(TINTED)
        self.assertEqual({MODE: 2}[vd.specialization("MODE", vd.u32)], 2)


class StaticSpecializationKernelTests(unittest.TestCase):
    def test_kernel_parameters_follow_the_when_predicate(self) -> None:
        vd.init(arch=vd.cpu)
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        mode_kernel(output, grid=(1, 1, 1), specializations={MODE: 1})
        np.testing.assert_array_equal(output.to_numpy(), np.array((-9.0,), dtype=np.float32))
        mode_kernel(output, np.float32(0.5), grid=(1, 1, 1), specializations={MODE: 3})
        np.testing.assert_array_equal(output.to_numpy(), np.array((1.5,), dtype=np.float32))
        with self.assertRaisesRegex(TypeError, "expects 1 launch arguments"):
            mode_kernel(output, np.float32(0.5), grid=(1, 1, 1), specializations={MODE: 1})


def _render_target(texture: vd.Texture) -> vd.RenderPass:
    return vd.render_pass(
        vd.RenderTarget.from_attachments(colors={0: texture}),
        color=vd.clear((0.0, 0.0, 0.0, 0.0)),
    )


def _assert_mode_pipeline_renders(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)), dtype=np.float32))
    expected = {0: (255, 0, 0, 255), 1: (0, 255, 0, 255), 2: (0, 0, 255, 255)}
    for mode, color in expected.items():
        target = vd.Texture.zeros(shape=(8, 8))
        render = vd.pipeline(mode_vertex, mode_fragment, specializations={MODE: mode})
        render(position=positions, render_pass=_render_target(target))
        np.testing.assert_array_equal(target.to_numpy()[4, 4], np.array(color, dtype=np.uint8))

    target = vd.Texture.zeros(shape=(8, 8))
    tinted = vd.pipeline(mode_vertex, mode_fragment, specializations={MODE: 2, TINTED: True})
    tint = np.array((1.0, 1.0, 0.0, 1.0), dtype=np.float32)
    tinted(position=positions, tint=tint, render_pass=_render_target(target))
    np.testing.assert_array_equal(target.to_numpy()[4, 4], np.array((255, 255, 0, 255), dtype=np.uint8))
    untinted = vd.pipeline(mode_vertex, mode_fragment, specializations={MODE: 0, TINTED: True})
    with test.assertRaisesRegex(TypeError, "unexpected pipeline argument"):
        untinted(position=positions, tint=tint, render_pass=_render_target(target))


class OpenGLStaticSpecializationTests(unittest.TestCase):
    def setUp(self) -> None:
        require_available(
            probe_backend(backend_row(vd.opengl), BackendRequirements(graphics=True, minimum_api_version=(4, 3)))
        )

    def test_integer_specializations_select_pipeline_code_and_parameters(self) -> None:
        _assert_mode_pipeline_renders(self)


class VulkanStaticSpecializationTests(unittest.TestCase):
    def setUp(self) -> None:
        require_available(probe_backend(backend_row(vd.vulkan), BackendRequirements(graphics=True)))

    def test_integer_specializations_select_pipeline_code_and_parameters(self) -> None:
        _assert_mode_pipeline_renders(self)


class StaticSpecializationAssetTests(unittest.TestCase):
    def test_graphics_asset_cooks_integer_variants(self) -> None:
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "modes.py"
            source.write_text(
                """
from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd

MODE = vd.specialization("MODE", vd.u32)


@vd.vertex
def vertex_main(
    vertices: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([vertices, 0.0, 1.0])


@vd.fragment
def fragment_main(
    tint: vd.When[MODE >= 1, Annotated[vd.Vector[vd.f32, 4], vd.uniform()]],
) -> vd.Vector[vd.f32, 4]:
    if MODE >= 1:
        return tint
    return vd.Vector([1.0, 1.0, 1.0, 1.0])


asset = vd.program_asset(
    id="shaders/modes",
    program=vd.pipeline(
        vertex_main,
        fragment_main,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
    variants=({MODE: 0}, {MODE: 1}),
)
""",
                encoding="utf-8",
            )
            manifest = cook_program_asset(
                program_asset=f"{source}:asset",
                output=Path(directory) / "out",
                target="vulkan",
            )
            document = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(
            [variant["key"] for variant in document["variants"]],
            [
                [{"name": "MODE", "value": {"tag": "u32", "value": 0}}],
                [{"name": "MODE", "value": {"tag": "u32", "value": 1}}],
            ],
        )
        boundaries = [
            sorted(slot["path"] for slot in variant["program"]["abi"]["boundary_slots"])
            for variant in document["variants"]
        ]
        self.assertEqual(boundaries, [["vertices"], ["tint", "vertices"]])
        deployed = {
            canonical_json(
                {
                    stage: [module["blob"] for module in artifact["modules"]]
                    for stage, artifact in variant["artifact_system"]["artifacts"].items()
                }
            )
            for variant in document["variants"]
        }
        self.assertEqual(len(deployed), 2)


if __name__ == "__main__":
    unittest.main()
