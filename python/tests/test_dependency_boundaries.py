from __future__ import annotations

import ast
import unittest
from pathlib import Path

import vernon_dsl as vd
from vernon_dsl.pipeline_assets import cook_pipeline_asset, encode_runtime_stage, parse_python_pipeline_asset

ROOT = Path(__file__).parents[1] / "vernon_dsl"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            modules.add(f"{prefix}{node.module or ''}")
    return modules


class DependencyBoundaryTests(unittest.TestCase):
    def test_runtime_resource_implementations_live_outside_session(self) -> None:
        expected_modules = {
            vd.interop.RawBuffer: "vernon_dsl._runtime.tensor",
            vd.TensorLayout: "vernon_dsl._runtime.tensor",
            vd.TensorStorage: "vernon_dsl._runtime.tensor",
            vd.TensorView: "vernon_dsl._runtime.tensor",
            vd.Texture: "vernon_dsl._runtime.texture",
        }
        for resource, expected_module in expected_modules.items():
            with self.subTest(resource=resource.__name__):
                self.assertEqual(resource.__module__, expected_module)
        self.assertEqual(vd.Tensor.__module__, "vernon_dsl.types")
        self.assertEqual(vd.Kernel.__module__, "vernon_dsl._runtime.kernel")
        self.assertEqual(vd.Pipeline.__module__, "vernon_dsl._runtime.pipeline")
        self.assertEqual(vd.PrimitiveTopology.__module__, "vernon_dsl._runtime.pipeline")

    def test_runtime_resource_modules_follow_dependency_boundaries(self) -> None:
        runtime = ROOT / "_runtime"
        common_tree = ast.parse((runtime / "resource_common.py").read_text(encoding="utf-8"))
        common_definitions = {
            node.name
            for node in common_tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertEqual(common_definitions, {"_session_state"})

        for leaf in ("tensor.py", "texture.py", "sampler.py"):
            with self.subTest(module=leaf):
                self.assertNotIn(".binding", imported_modules(runtime / leaf))

        facade_tree = ast.parse((runtime / "resources.py").read_text(encoding="utf-8"))
        self.assertFalse(
            any(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) for node in facade_tree.body)
        )

        for path in runtime.rglob("*.py"):
            if path.name == "resources.py":
                continue
            with self.subTest(module=path.name):
                self.assertNotIn(".resources", imported_modules(path))

    def test_shader_artifact_io_lives_in_its_implementation_module(self) -> None:
        self.assertEqual(
            encode_runtime_stage.__module__,
            "vernon_dsl._shader_assets.artifact_io",
        )
        self.assertEqual(
            parse_python_pipeline_asset.__module__,
            "vernon_dsl._shader_assets.parsing",
        )
        self.assertEqual(
            cook_pipeline_asset.__module__,
            "vernon_dsl._shader_assets.cooking",
        )

    def test_frontend_does_not_import_runtime(self) -> None:
        for path in sorted((ROOT / "frontend").glob("*.py")):
            with self.subTest(path=path.name):
                imports = imported_modules(path)
                self.assertFalse(
                    any("_runtime" in module or module.endswith(".runtime") for module in imports),
                    imports,
                )

    def test_bundle_planning_does_not_import_frontend_or_runtime(self) -> None:
        for path in sorted((ROOT / "bundle").glob("*.py")):
            with self.subTest(path=path.name):
                imports = imported_modules(path)
                self.assertFalse(
                    any(
                        "frontend" in module or "_runtime" in module or module.endswith(".runtime")
                        for module in imports
                    ),
                    imports,
                )


if __name__ == "__main__":
    unittest.main()
