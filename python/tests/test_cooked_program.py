from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from vernon_dsl._runtime.cooked_program import CookedProgram
from vernon_dsl._runtime.session import cpu


class CookedProgramTests(unittest.TestCase):
    def test_fixed_grid_module_vjp_does_not_require_public_grid_values(self) -> None:
        parameters = (
            SimpleNamespace(name="source", access=0, slot=0),
            SimpleNamespace(name="output", access=1, slot=1),
        )
        native_pullback = SimpleNamespace()
        linked_vjp = object()
        native = SimpleNamespace(
            parameters=parameters,
            derivative_groups=(
                ("gradient", "source", ("source",)),
                ("cotangent", "output", ("output",)),
            ),
            program_abi={
                "boundary_slots": [
                    {"slot": 0, "path": "source", "role": "input"},
                    {"slot": 1, "path": "output", "role": "output"},
                ]
            },
        )
        state = SimpleNamespace(arch=cpu)
        program = CookedProgram(b"", "", ())
        loaded = SimpleNamespace(native=native, linked_vjp=lambda session: linked_vjp)
        calls = []

        def call_program_vjp(vjp, executable, invocation, targets, bindings):
            calls.append((vjp, dict(invocation.inputs), dict(targets), dict(bindings)))
            return native_pullback

        with (
            mock.patch.object(program, "_load", return_value=loaded),
            mock.patch(
                "vernon_dsl._runtime.cooked_program._execution_context",
                return_value=SimpleNamespace(session=state, identity=1),
            ),
            mock.patch("vernon_dsl._runtime.program_autodiff._call_program_vjp", side_effect=call_program_vjp),
        ):
            output, pullback = program._vjp({"source": 1, "output": 2}, None)
            self.assertEqual(output, {})
            self.assertIs(pullback.native, native_pullback)
            self.assertEqual(calls, [(linked_vjp, {"source": 1}, {"output": 2}, {"source": 1, "output": 2})])

            with self.assertRaisesRegex(ValueError, "owns its launch grids"):
                program._vjp({"source": 1, "output": 2}, (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
