from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import vernon_dsl as vd
from backend_test_matrix import BackendRequirements, backend_row, probe_backend, require_available
from type_specialization_materials import Flat, Glossy, Lookup, Matte, Scaled, SurfaceParams
from type_specialization_template import ELEMENT, MATERIAL, SURFACE, convert, shade, surface_fragment, surface_vertex
from vernon_dsl import CompileError, Compiler
from vernon_dsl.bundle import canonical_json
from vernon_dsl.cli import main
from vernon_dsl.compiler import FrontendCompileRequest, compile_file
from vernon_dsl.frontend.cache import frontend_cache
from vernon_dsl.program_assets import ProgramCompileError, cook_program_asset
from vernon_dsl.types import SpecializationAssignment, specialization_assignment, specialization_key, type_identity

_REPOSITORY = Path(__file__).resolve().parents[2]
_MATERIALS = "python.tests.type_specialization_materials"
_FIXTURES = _REPOSITORY / "source" / "tests" / "fixtures"

_MATTE = "from vernon_dsl import *\n@binding_group\nclass Matte:\n    gain: f32\n"
_GLOSSY = "from vernon_dsl import *\n@binding_group\nclass Glossy:\n    gain: f32\n    offset: f32\n"
_TEMPLATE = """from vernon_dsl import *
import glossy

MATERIAL = specialization("MATERIAL", type)

@fragment
def main(material: MATERIAL, extra: When[MATERIAL == glossy.Glossy, f32]) -> f32:
    if MATERIAL == glossy.Glossy:
        return material.gain + material.offset + extra
    elif MATERIAL != i32:
        return material.gain * 3.0
    return 0.0
"""


def _type(name: str, identity: str) -> SpecializationAssignment:
    return SpecializationAssignment(name, "type", identity)


def _native_available() -> bool:
    try:
        from vernon_dsl._program_assets.compile_orchestration import _native_module

        _native_module()
        return True
    except ProgramCompileError:
        return False


def _import_file(path: Path):
    spec = importlib.util.spec_from_file_location(f"_type_specialization_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


class TypeIdentityTests(unittest.TestCase):
    def test_declared_and_scalar_types_have_canonical_identities(self) -> None:
        self.assertEqual(type_identity(Matte), f"{_MATERIALS}:Matte")
        self.assertEqual(type_identity(SurfaceParams), f"{_MATERIALS}:SurfaceParams")
        self.assertEqual(type_identity(vd.i32), "i32")
        key = specialization_key({MATERIAL: Glossy, ELEMENT: vd.f16})
        self.assertEqual(key, (_type("ELEMENT", "f16"), _type("MATERIAL", f"{_MATERIALS}:Glossy")))
        self.assertIs(key[1].host_value, Glossy)
        self.assertEqual(key[1], _type("MATERIAL", f"{_MATERIALS}:Glossy"), "the host class takes no part in the key")

    def test_value_must_be_a_declared_or_scalar_type(self) -> None:
        for value in (3, float, object):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "declared type or scalar type"):
                specialization_assignment(MATERIAL, value)

    def test_nested_class_has_no_portable_identity(self) -> None:
        @vd.struct
        class Local:
            value: vd.f32

        with self.assertRaisesRegex(TypeError, "no portable identity"):
            type_identity(Local)

    def test_variants_reject_types_of_two_projects_sharing_one_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            modules = []
            for project in ("first", "second"):
                root = Path(directory) / project
                (root / "lib").mkdir(parents=True)
                (root / "pyproject.toml").write_text(f'[project]\nname = "{project}"\n', encoding="utf-8")
                (root / "lib" / "materials.py").write_text(_MATTE, encoding="utf-8")
                modules.append(_import_file(root / "lib" / "materials.py"))
            first, second = (module.Matte for module in modules)
            self.assertEqual(type_identity(first), type_identity(second))
            with self.assertRaisesRegex(ValueError, "share the identity 'lib.materials:Matte'"):
                vd.program_asset(id="tests/clash", program=shade, variants=({MATERIAL: first}, {MATERIAL: second}))

    def test_class_outside_a_project_root_has_no_portable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rootless.py"
            path.write_text(_MATTE, encoding="utf-8")
            module = _import_file(path)
        with self.assertRaisesRegex(TypeError, "no portable identity"):
            type_identity(module.Matte)


class TypeSpecializationKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        vd.init(arch=vd.cpu)
        self.weights = vd.storage.from_numpy(np.array((1.0, 2.0, 3.0), dtype=np.float32))

    def test_binding_group_type_selects_the_parameter_and_its_methods(self) -> None:
        output = vd.storage.zeros(dtype=vd.f32, shape=(3,))
        lookup = Lookup(weights=self.weights)
        shade(Matte(gain=np.float32(2.0), lookup=lookup), output, grid=(3, 1, 1), specializations={MATERIAL: Matte})
        np.testing.assert_array_equal(output.to_numpy(), np.array((2.0, 4.0, 6.0), dtype=np.float32))
        glossy = Glossy(gain=np.float32(2.0), offset=np.float32(0.5), lookup=lookup)
        shade(glossy, output, grid=(3, 1, 1), specializations={MATERIAL: Glossy})
        np.testing.assert_array_equal(output.to_numpy(), np.array((2.5, 4.5, 6.5), dtype=np.float32))

    def test_scalar_type_selects_storage_element_and_conversion(self) -> None:
        output = vd.storage.zeros(dtype=vd.i32, shape=(3,))
        convert(self.weights, output, grid=(3, 1, 1), specializations={ELEMENT: vd.i32})
        np.testing.assert_array_equal(output.to_numpy(), np.array((2, 4, 6), dtype=np.int32))

    def test_unassigned_type_specialization_is_rejected_where_used(self) -> None:
        output = vd.storage.zeros(dtype=vd.f32, shape=(3,))
        with self.assertRaisesRegex(CompileError, "missing type specialization 'MATERIAL'"):
            shade(Matte(gain=np.float32(2.0), lookup=Lookup(weights=self.weights)), output, grid=(3, 1, 1))

    def test_group_root_wrt_expands_per_specialization_key(self) -> None:
        transformed = vd.ad.vjp(shade, wrt=("material",), outputs=("output",))
        matte = specialization_key({MATERIAL: Matte})
        glossy = specialization_key({MATERIAL: Glossy})
        self.assertEqual(transformed.transform_for(matte).wrt, ("material.gain", "material.lookup.weights"))
        self.assertEqual(
            transformed.transform_for(glossy).wrt,
            ("material.gain", "material.lookup.weights", "material.offset"),
        )


class TypeSpecializationCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        Compiler.clear_cache()
        self.addCleanup(Compiler.clear_cache)
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        (self.root / "pyproject.toml").write_text('[project]\nname = "materials"\n', encoding="utf-8")
        (self.root / "matte.py").write_text(_MATTE, encoding="utf-8")
        (self.root / "glossy.py").write_text(_GLOSSY, encoding="utf-8")
        (self.root / "plain.py").write_text("class Plain:\n    pass\n", encoding="utf-8")
        self.template = self.root / "template.py"
        self.template.write_text(_TEMPLATE, encoding="utf-8")

    def _compile(self, identity: str, source: Path | None = None) -> str:
        return compile_file(source or self.template, specializations=(_type("MATERIAL", identity),), entry="main")

    def test_static_comparisons_and_when_follow_the_bound_type(self) -> None:
        glossy = self._compile("glossy:Glossy")
        self.assertIn('vernon.source_name = "extra"', glossy)
        self.assertIn("arith.addf", glossy)
        self.assertNotIn("arith.mulf", glossy)
        matte = self._compile("matte:Matte")
        self.assertNotIn('vernon.source_name = "extra"', matte)
        self.assertIn("arith.mulf", matte)

    def test_type_specialization_compares_only_for_equality(self) -> None:
        source = self.root / "truthy.py"
        source.write_text(
            "from vernon_dsl import *\n"
            'MATERIAL = specialization("MATERIAL", type)\n'
            "@fragment\n"
            "def main(material: MATERIAL) -> f32:\n"
            "    if MATERIAL:\n"
            "        return material.gain\n"
            "    return 0.0\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(CompileError, "compares only with == and !="):
            self._compile("matte:Matte", source)

    def test_identity_must_name_a_declared_type_in_the_project(self) -> None:
        with self.assertRaisesRegex(CompileError, "missing type specialization 'MATERIAL'"):
            compile_file(self.template, entry="main")
        with self.assertRaisesRegex(CompileError, "'plain:Plain' is not one"):
            self._compile("plain:Plain")
        with self.assertRaisesRegex(CompileError, "names no source module"):
            self._compile("absent:Thing")
        with self.assertRaisesRegex(CompileError, "declared as type, not u32"):
            compile_file(self.template, specializations=(SpecializationAssignment("MATERIAL", "u32", 1),), entry="main")

    def test_cli_binds_a_type_identity(self) -> None:
        output = self.root / "glossy.mlir"
        arguments = [str(self.template), "-o", str(output), "--entry", "main"]
        self.assertEqual(main([*arguments, "--specialization", "MATERIAL:type=glossy:Glossy"]), 0)
        self.assertIn('vernon.source_name = "extra"', output.read_text(encoding="utf-8"))
        self.assertEqual(main([*arguments, "--specialization", "MATERIAL:type=absent:Thing"]), 1)

    def test_cache_entries_depend_only_on_the_bound_type_source(self) -> None:
        matte = FrontendCompileRequest(self.template, "main", (_type("MATERIAL", "matte:Matte"),))
        glossy = FrontendCompileRequest(self.template, "main", (_type("MATERIAL", "glossy:Glossy"),))
        matte_result = Compiler().compile_request(matte)
        glossy_result = Compiler().compile_request(glossy)
        self.assertNotEqual(matte_result.specialized_source, glossy_result.specialized_source)
        self.assertTrue(any(name.endswith("matte.py") for name, _ in matte_result.dependencies))
        self.assertFalse(any(name.endswith("matte.py") for name, _ in glossy_result.dependencies))

        (self.root / "matte.py").write_text(_MATTE.replace("gain: f32", "gain: f32\n    unused: f32"), encoding="utf-8")
        self.assertIsNone(frontend_cache.get(matte))
        self.assertIs(frontend_cache.get(glossy), glossy_result)


class TypeSpecializationAssetTests(unittest.TestCase):
    def test_lint_reads_type_variants_without_running_the_declaration(self) -> None:
        from vernon_dsl._program_assets.parsing import lint_python_program_asset

        lint = lint_python_program_asset(_FIXTURES / "type_specialization_program_asset.py", "asset")
        materials = "source.tests.fixtures.type_specialization_fixture_materials"
        self.assertEqual(
            lint.variants,
            ((_type("MATERIAL", f"{materials}:Matte"),), (_type("MATERIAL", f"{materials}:Glossy"),)),
        )

    def test_lint_rejects_a_variant_value_that_is_not_a_type(self) -> None:
        from vernon_dsl._program_assets.parsing import lint_python_program_asset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text('[project]\nname = "assets"\n', encoding="utf-8")
            for name in ("type_specialization_fixture_template.py", "type_specialization_fixture_materials.py"):
                shutil.copyfile(_FIXTURES / name, root / name)
            source = root / "asset.py"
            source.write_text(
                (_FIXTURES / "type_specialization_program_asset.py")
                .read_text(encoding="utf-8")
                .replace("{MATERIAL: Matte}", "{MATERIAL: 3}"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramCompileError, "requires a declared type or scalar type"):
                lint_python_program_asset(source, "asset")

    def test_graphics_asset_cooks_one_variant_per_type(self) -> None:
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        surfaces = "source.tests.fixtures.language_contract_graphics_asset"
        with tempfile.TemporaryDirectory() as directory:
            manifest = cook_program_asset(
                program_asset=f"{_FIXTURES / 'language_contract_graphics_asset.py'}:type_specialization_asset",
                output=Path(directory),
                target="vulkan",
            )
            document = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(
            [variant["key"] for variant in document["variants"]],
            [
                [{"name": "SURFACE", "value": {"tag": "type", "value": f"{surfaces}:Cool"}}],
                [{"name": "SURFACE", "value": {"tag": "type", "value": f"{surfaces}:Warm"}}],
            ],
        )
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

    def test_kernel_cooks_type_specialization_without_postponed_annotations(self) -> None:
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        source = (
            "from typing import Annotated\n"
            "import vernon_dsl as vd\n"
            'OPERATOR = vd.specialization("OPERATOR", vd.type)\n'
            "@vd.binding_group\n"
            "class Gain:\n"
            "    gain: vd.f32\n"
            "    @vd.func\n"
            "    def apply(self, x: vd.f32) -> vd.f32:\n"
            "        return x * self.gain\n"
            "@vd.kernel(workgroup_size=(1, 1, 1))\n"
            "def run(\n"
            "    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read_write],\n"
            "    operator: OPERATOR,\n"
            '    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],\n'
            ") -> None:\n"
            "    values[gid[0]] = operator.apply(values[gid[0]])\n"
            'asset = vd.program_asset(id="tests/eager", program=run, variants=({OPERATOR: Gain},))\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text('[project]\nname = "assets"\n', encoding="utf-8")
            (root / "eager.py").write_text(source, encoding="utf-8")
            manifest = cook_program_asset(program_asset=f"{root / 'eager.py'}:asset", output=root / "out", target="cpu")
            document = json.loads(manifest.read_text(encoding="utf-8"))
        (variant,) = document["variants"]
        self.assertEqual(variant["key"], [{"name": "OPERATOR", "value": {"tag": "type", "value": "eager:Gain"}}])
        self.assertIn("operator.gain", [slot["path"] for slot in variant["program"]["abi"]["boundary_slots"]])

    def test_type_value_from_another_project_specializes_a_library_template(self) -> None:
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        library_ops = (
            "from typing import Annotated\n"
            "import vernon_dsl as vd\n"
            'OPERATOR = vd.specialization("OPERATOR", vd.type)\n'
            "@vd.struct(shared=True)\n"
            "class Scale:\n"
            "    factor: vd.f32\n"
            "@vd.kernel(workgroup_size=(1, 1, 1))\n"
            "def run(\n"
            "    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read_write],\n"
            "    operator: OPERATOR,\n"
            '    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],\n'
            ") -> None:\n"
            "    values[gid[0]] = operator.apply(values[gid[0]])\n"
        )
        boost = (
            "import vernon_dsl as vd\n"
            "from lib.ops import Scale\n"
            "from game.helpers import twice\n"
            "@vd.binding_group\n"
            "class Boost:\n"
            "    scale: Scale\n"
            "    @vd.func\n"
            "    def apply(self, x: vd.f32) -> vd.f32:\n"
            "        return twice(x * self.scale.factor)\n"
        )
        helpers = "import vernon_dsl as vd\n@vd.func\ndef twice(x: vd.f32) -> vd.f32:\n    return x * 2.0\n"
        asset = (
            "import vernon_dsl as vd\n"
            "from lib.ops import OPERATOR, run\n"
            "from game.boost import Boost\n"
            'asset = vd.program_asset(id="game/boost", program=run, variants=({OPERATOR: Boost},))\n'
        )
        expected_key = [{"name": "OPERATOR", "value": {"tag": "type", "value": "game.boost:Boost"}}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library, content, clash = root / "library", root / "content", root / "clash"
            for project, files in (
                (library, {"lib/ops.py": library_ops}),
                (content, {"game/boost.py": boost, "game/helpers.py": helpers, "game/asset.py": asset}),
                (
                    clash,
                    {
                        "lib/ops.py": library_ops,
                        "arcade/helpers.py": helpers,
                        "arcade/boost.py": boost.replace("game.", "arcade."),
                        "arcade/asset.py": asset.replace("game", "arcade"),
                    },
                ),
            ):
                (project / "pyproject.toml").parent.mkdir(parents=True, exist_ok=True)
                (project / "pyproject.toml").write_text(f'[project]\nname = "{project.name}"\n', encoding="utf-8")
                for relative, text in files.items():
                    (project / relative).parent.mkdir(parents=True, exist_ok=True)
                    (project / relative).write_text(text, encoding="utf-8")
            sys.path[:0] = [str(library), str(content), str(clash)]
            try:
                static = cook_program_asset(
                    program_asset=f"{content / 'game' / 'asset.py'}:asset", output=root / "static", target="vulkan"
                )
                ops = importlib.import_module("lib.ops")
                declaration = vd.program_asset(
                    id="game/boost",
                    program=ops.run,
                    variants=({ops.OPERATOR: importlib.import_module("game.boost").Boost},),
                )
                evaluated = cook_program_asset(program_asset=declaration, output=root / "evaluated", target="vulkan")
                documents = [path.read_text(encoding="utf-8") for path in (static, evaluated)]

                # The clashing project's own lib/ops.py shadows the library's for its modules.
                with self.assertRaisesRegex(Exception, "share the project module path 'lib/ops.py'"):
                    cook_program_asset(
                        program_asset=f"{clash / 'arcade' / 'asset.py'}:asset", output=root / "out", target="vulkan"
                    )
            finally:
                for package in ("lib", "game", "arcade"):
                    for name in [name for name in sys.modules if name == package or name.startswith(f"{package}.")]:
                        sys.modules.pop(name)
                del sys.path[:3]
            for document in documents:
                self.assertNotIn(root.as_posix(), document.replace("\\\\", "/"))
                (variant,) = json.loads(document)["variants"]
                self.assertEqual(variant["key"], expected_key)
                self.assertIn("operator.scale", [slot["path"] for slot in variant["program"]["abi"]["boundary_slots"]])
            self.assertEqual(json.loads(documents[0])["content_hash"], json.loads(documents[1])["content_hash"])

    def test_declaration_object_cooks_caller_selected_variants(self) -> None:
        if not _native_available():
            self.skipTest("native Vernon extension is not built")
        path = _FIXTURES / "language_contract_graphics_asset.py"
        specification = importlib.util.spec_from_file_location("_language_contract_graphics_asset", path)
        assert specification is not None and specification.loader is not None
        fixture = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = fixture
        try:
            specification.loader.exec_module(fixture)
            authored = fixture.type_specialization_asset
            declaration = vd.program_asset(
                id="runtime/caller-selected", program=authored.program, variants=({fixture.SURFACE: fixture.Warm},)
            )
            with tempfile.TemporaryDirectory() as directory:
                manifest = cook_program_asset(program_asset=declaration, output=Path(directory), target="vulkan")
                document = json.loads(manifest.read_text(encoding="utf-8"))
        finally:
            sys.modules.pop(specification.name, None)
        self.assertEqual(document["id"], "runtime/caller-selected")
        warm = "source.tests.fixtures.language_contract_graphics_asset:Warm"
        self.assertEqual(
            [variant["key"] for variant in document["variants"]],
            [[{"name": "SURFACE", "value": {"tag": "type", "value": warm}}]],
        )


def _assert_surface_pipeline_renders(test: unittest.TestCase) -> None:
    positions = vd.storage.from_numpy(np.array(((-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)), dtype=np.float32))
    params = SurfaceParams(np.array((0.25, 0.5, 0.0, 0.5), dtype=np.float32), vd.f32(2.0))
    expected = {Flat: (64, 128, 0, 128), Scaled: (128, 255, 0, 255)}
    for group, color in expected.items():
        target = vd.Texture.zeros(shape=(8, 8))
        render = vd.pipeline(surface_vertex, surface_fragment, specializations={SURFACE: group})
        render(
            position=positions,
            surface=group(params=params),
            render_pass=vd.render_pass(
                vd.RenderTarget.from_attachments(colors={0: target}), color=vd.clear((0.0, 0.0, 0.0, 0.0))
            ),
        )
        np.testing.assert_allclose(target.to_numpy()[4, 4], np.array(color), atol=1)


class OpenGLTypeSpecializationTests(unittest.TestCase):
    def setUp(self) -> None:
        require_available(
            probe_backend(backend_row(vd.opengl), BackendRequirements(graphics=True, minimum_api_version=(4, 3)))
        )

    def test_type_specializations_select_the_fragment_binding_group(self) -> None:
        _assert_surface_pipeline_renders(self)


class VulkanTypeSpecializationTests(unittest.TestCase):
    def setUp(self) -> None:
        require_available(probe_backend(backend_row(vd.vulkan), BackendRequirements(graphics=True)))

    def test_type_specializations_select_the_fragment_binding_group(self) -> None:
        _assert_surface_pipeline_renders(self)


if __name__ == "__main__":
    unittest.main()
