from __future__ import annotations

import ast
import unittest
from pathlib import Path

import vernon_dsl as vd
from vernon_dsl.shader_assets import cook_shader_pipeline, encode_runtime_stage, parse_python_pipeline_asset

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
        for resource in (vd.Tensor, vd.TensorLayout, vd.TensorView, vd.Texture):
            with self.subTest(resource=resource.__name__):
                self.assertEqual(resource.__module__, "vernon_dsl._runtime.resources")
        self.assertEqual(vd.Kernel.__module__, "vernon_dsl._runtime.kernel")
        self.assertEqual(vd.Pipeline.__module__, "vernon_dsl._runtime.pipeline")
        self.assertEqual(vd.PrimitiveTopology.__module__, "vernon_dsl._runtime.pipeline")

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
            cook_shader_pipeline.__module__,
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
