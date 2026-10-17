from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vernon_dsl.compiler import compile_file
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.frontend.model import ConcreteType
from vernon_dsl.types import SpecializationAssignment

_SPECIALIZED = """from vernon_dsl import *

SCALE = specialization("SCALE", u32)


# Comments and blank lines are not part of the syntax tree.
@kernel(workgroup_size=(1, 1, 1))
def main(output: TensorView[u32, (1,), write]) -> None:
    # The store below is on line 11.
    scaled = SCALE // 2
    output[0] = scaled + 1.5
"""

_HELPERS = """from vernon_dsl import *


# The combined project places this helper after the root module's definitions.
@func
def widen(value: f32) -> f32:
    return value


@func
def broken(value: f32) -> i32:
    halved = value * 0.5
    narrowed: i32 = halved
    return narrowed
"""

_SHADER = """from vernon_dsl import *
import helpers


@fragment
def main(value: f32) -> i32:
    return helpers.broken(helpers.widen(value))
"""


class DiagnosticLocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def _write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_specialized_entry_reports_the_line_it_was_written_on(self) -> None:
        source = self._write("specialized.py", _SPECIALIZED)
        with self.assertRaises(CompileError) as raised:
            compile_file(source, specializations=(SpecializationAssignment("SCALE", "u32", 4),), entry="main")
        self.assertEqual(raised.exception.location.filename, str(source))
        self.assertEqual((raised.exception.location.line, raised.exception.location.column), (11, 17))
        self.assertEqual(raised.exception.message, "cannot store floating literal as u32")

    def test_an_imported_helper_reports_its_own_file_and_line(self) -> None:
        helpers = self._write("helpers.py", _HELPERS)
        shader = self._write("shader.py", _SHADER)
        with self.assertRaises(CompileError) as raised:
            compile_file(shader, entry="main")
        self.assertEqual(Path(raised.exception.location.filename).resolve(), helpers.resolve())
        self.assertEqual((raised.exception.location.line, raised.exception.location.column), (13, 21))

    def test_types_are_named_as_annotations_write_them(self) -> None:
        u32 = ConcreteType("scalar", "u32")
        self.assertEqual(u32.spelling, "u32")
        self.assertEqual(
            ConcreteType("tensor_view", "TensorView", (u32, ("?", 4), "write", "device")).spelling,
            "TensorView[u32, (dyn, 4), write]",
        )
        self.assertEqual(ConcreteType("tensor", "Tensor", (u32, 3)).spelling, "Tensor[u32, (3,)]")


if __name__ == "__main__":
    unittest.main()
