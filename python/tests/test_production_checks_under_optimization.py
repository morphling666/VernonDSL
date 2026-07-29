from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_optimized(script: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": "python", "PYTHONOPTIMIZE": "1"}
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class ProductionChecksUnderOptimizationTests(unittest.TestCase):
    def test_optimization_is_active_in_subprocess(self) -> None:
        result = _run_optimized("import sys; sys.exit(0 if not __debug__ else 2)")
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

    def test_tensor_view_layout_validation_raises_under_optimization(self) -> None:
        script = textwrap.dedent(
            """
            import sys
            import tempfile
            from pathlib import Path

            from vernon_dsl import Compiler
            from vernon_dsl.compiler import FrontendCompileRequest

            source = (
                "from vernon_dsl import *\\n"
                "@kernel\\n"
                "def read(value: TensorView[f32, (2, dyn), read]) -> None:\\n"
                "    pass\\n"
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid_view.py"
                path.write_text(source, encoding="utf-8")
                try:
                    Compiler().compile_request(
                        FrontendCompileRequest(
                            path,
                            "read",
                            tensor_view_layouts=(("value", "<f4", (3, 4), (4, 1), 0),),
                        )
                    )
                except ValueError as error:
                    if "dimension 0 is 3, expected 2" in str(error):
                        sys.exit(0)
                    raise
            sys.exit("expected ValueError for invalid TensorView specialization")
            """
        )
        result = _run_optimized(script)
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

    def test_execution_graph_requires_declared_graph_under_optimization(self) -> None:
        script = textwrap.dedent(
            """
            import sys

            from vernon_dsl._runtime.execution_graph import ComputePass

            class BrokenPass(ComputePass):
                def execute(self, encoder, resources):
                    pass

            try:
                BrokenPass("broken")._native_execute(object())
            except RuntimeError as error:
                if "requires a declared execution graph" in str(error):
                    sys.exit(0)
                raise
            sys.exit("expected RuntimeError for missing execution graph")
            """
        )
        result = _run_optimized(script)
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

    def test_render_pass_requires_declared_graph_under_optimization(self) -> None:
        script = textwrap.dedent(
            """
            import sys

            from vernon_dsl._runtime.execution_graph import RenderPass

            class BrokenPass(RenderPass):
                def execute(self, encoder, resources):
                    pass

            try:
                BrokenPass("broken")._native_execute(object())
            except RuntimeError as error:
                if "requires a declared execution graph" in str(error):
                    sys.exit(0)
                raise
            sys.exit("expected RuntimeError for missing execution graph")
            """
        )
        result = _run_optimized(script)
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
