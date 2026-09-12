from __future__ import annotations

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from vernon_dsl._runtime.cooked_program import CookedProgram
from vernon_dsl._runtime.session import cpu


class _BindingCache:
    @contextmanager
    def invocation(self, executable, context):
        yield object()

    def bind_argument(self, builder, executable, parameter, value) -> None:
        pass


class CookedProgramTests(unittest.TestCase):
    def test_fixed_grid_module_vjp_does_not_require_public_grid_values(self) -> None:
        parameters = (
            SimpleNamespace(name="source", access=0),
            SimpleNamespace(name="output", access=1),
        )
        native_pullback = object()
        native = SimpleNamespace(
            parameters=parameters,
            derivative_groups=(
                ("gradient", "source", ("source",)),
                ("cotangent", "output", ("output",)),
            ),
            program_vjp_bound=lambda builder, bindings: ({}, native_pullback),
        )
        native_module = SimpleNamespace(
            ACCESS_READ=0,
            ACCESS_WRITE=1,
            ACCESS_READ_WRITE=2,
        )
        state = SimpleNamespace(
            native=native_module,
            arch=cpu,
        )
        program = CookedProgram(b"", "", ())
        loaded = SimpleNamespace(native=native, binding_cache=_BindingCache())

        with (
            mock.patch.object(program, "_load", return_value=loaded),
            mock.patch(
                "vernon_dsl._runtime.cooked_program._execution_context",
                return_value=SimpleNamespace(session=state, identity=1),
            ),
        ):
            output, pullback = program._vjp({"source": 1, "output": 2}, None)
            self.assertEqual(output, {})
            self.assertIs(pullback.native, native_pullback)

            with self.assertRaisesRegex(ValueError, "owns its launch grids"):
                program._vjp({"source": 1, "output": 2}, (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
