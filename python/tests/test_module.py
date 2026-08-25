from __future__ import annotations

import importlib
import unittest
from dataclasses import dataclass
from unittest import mock

import numpy as np
import vernon_dsl as vd
from vernon_dsl._runtime.session import RuntimeUnavailableError
from vernon_dsl.program_frontend import BuiltinDslProvider, DirectKernelDslProvider


@vd.kernel(workgroup_size=(1, 1, 1))
def module_square(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0]


@vd.kernel(workgroup_size=(1, 1, 1))
def module_cube(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0] * source[0]


@vd.struct
class ModulePair:
    left: vd.f32
    right: vd.f32


@vd.kernel(workgroup_size=(1, 1, 1))
def module_copy_pair(
    source: vd.TensorView[ModulePair, (1,), vd.read],
    output: vd.TensorView[ModulePair, (1,), vd.write],
) -> None:
    output[0] = source[0]


class PowerBranch(vd.Module):
    def __init__(self, *, power: int, grid: tuple[int, int, int]):
        super().__init__()
        if power not in {2, 3}:
            raise ValueError("PowerBranch supports square or cube")
        self.power = power
        self.grid = grid

    def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
        output = vd.empty_like(source)
        if self.power == 2:
            module_square(source, output, grid=self.grid)
        else:
            module_cube(source, output, grid=self.grid)
        return output


@dataclass
class FanInOutput:
    square: vd.TensorStorage
    cube: vd.TensorStorage


class FanIn(vd.Module):
    def __init__(self, *, grid: tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        self.grid = grid
        self.square = PowerBranch(power=2, grid=grid)
        self.cube = PowerBranch(power=3, grid=grid)

    def forward(self, source: vd.TensorStorage) -> FanInOutput:
        return FanInOutput(self.square(source), self.cube(source))


class AnnotatedSquare(vd.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        source: vd.TensorView[vd.f32, (1,), vd.read],
    ) -> vd.TensorStorage:
        output = vd.empty_like(source)
        module_square(source, output)
        return output


class AggregateCopy(vd.Module):
    def forward(
        self,
        source: vd.TensorView[ModulePair, (1,), vd.read],
    ) -> vd.TensorStorage:
        output = vd.empty_like(source)
        module_copy_pair(source, output)
        return output


class ModuleTests(unittest.TestCase):
    def test_direct_provider_returns_original_frontend_implementation(self) -> None:
        provider = DirectKernelDslProvider("module { func.func @direct() }", "direct")
        implementation = provider.lower(
            {"kind": "compute", "implementation_hint": "direct"},
            {},
        )
        self.assertIsNotNone(implementation)
        assert implementation is not None
        self.assertEqual(implementation.entry, "direct")
        self.assertEqual(implementation.mlir, "module { func.func @direct() }")
        self.assertIsNone(provider.lower({"kind": "render", "implementation_hint": "direct"}, {}))

    def test_fusion_dsl_provider_lowers_builtin_add_request(self) -> None:
        values = {
            0: {"id": 0, "dtype": "f32", "shape": [4]},
            1: {"id": 1, "dtype": "f32", "shape": [4]},
            2: {"id": 2, "dtype": "f32", "shape": [4]},
        }
        request = {
            "implementation_hint": "vernon.builtin.add",
            "bindings": [
                {"parameter": "left", "value": 0},
                {"parameter": "right", "value": 1},
                {"parameter": "output", "value": 2},
            ],
        }

        implementation = BuiltinDslProvider().lower(request, values)

        self.assertIsNotNone(implementation)
        assert implementation is not None
        self.assertEqual(implementation.callee, "vernon.builtin.add")
        self.assertEqual(implementation.entry, "_program_add_f32_rank1")
        self.assertIn('vernon.source_name = "output"', implementation.mlir)

    def test_fusion_dsl_provider_lowers_rank_two_builtin_add_with_view_shape(
        self,
    ) -> None:
        values = {
            0: {"id": 0, "dtype": "f32", "shape": [2, 3]},
            1: {"id": 1, "dtype": "f32", "shape": [2, 3]},
            2: {"id": 2, "dtype": "f32", "shape": [2, 3]},
        }
        request = {
            "implementation_hint": "vernon.builtin.add",
            "bindings": [
                {"parameter": "left", "value": 0},
                {"parameter": "right", "value": 1},
                {"parameter": "output", "value": 2},
            ],
        }

        implementation = BuiltinDslProvider().lower(request, values)

        self.assertIsNotNone(implementation)
        assert implementation is not None
        self.assertEqual(implementation.entry, "_program_add_f32_rank2")
        self.assertIn('"vernon.get_shape"', implementation.mlir)
        self.assertIn("tensor<2xi32>", implementation.mlir)

    @classmethod
    def setUpClass(cls) -> None:
        try:
            importlib.import_module("vernon_dsl._native")
        except (ImportError, OSError) as error:
            raise unittest.SkipTest("native Vernon compiler and runtime are unavailable") from error
        vd.init(arch=vd.cpu)

    def test_init_registers_children_and_uses_native_program_ad_abi(self) -> None:
        module = FanIn()
        self.assertEqual(tuple(module.modules), ("square", "cube"))
        self.assertEqual(
            tuple(name for name, _ in module.named_modules()),
            ("", "square", "cube"),
        )

        source = vd.storage.from_numpy(np.array([2.0], dtype=np.float32))
        outputs, pullback = vd.ad.vjp(
            module,
            wrt=("source",),
            outputs=("square", "cube"),
        )(source)

        np.testing.assert_array_equal(outputs.square.to_numpy(), np.array([4.0], dtype=np.float32))
        np.testing.assert_array_equal(outputs.cube.to_numpy(), np.array([8.0], dtype=np.float32))
        specialization = next(iter(module._program_cache.values()))
        signature = specialization.pipeline.program_ad_signature
        self.assertEqual([value["path"] for value in signature["inputs"]], ["source"])
        self.assertEqual([value["path"] for value in signature["outputs"]], ["square", "cube"])
        self.assertEqual([value["path"] for value in signature["cotangents"]], ["square", "cube"])
        self.assertEqual([value["path"] for value in signature["gradients"]], ["source"])
        self.assertEqual([value["value_id"] for value in signature["captures"]], [0])

        gradients = pullback(
            {
                "square": np.array([1.0], dtype=np.float32),
                "cube": np.array([2.0], dtype=np.float32),
            }
        )

        np.testing.assert_array_equal(gradients["source"].to_numpy(), np.array([28.0], dtype=np.float32))

    def test_calls_hold_independent_outputs_and_tapes(self) -> None:
        module = FanIn()
        program = vd.ad.vjp(module, wrt=("source",), outputs=("square", "cube"))
        _, first = program(vd.storage.from_numpy(np.array([2.0], dtype=np.float32)))
        _, second = program(vd.storage.from_numpy(np.array([3.0], dtype=np.float32)))

        first_gradient = first(
            {
                "square": np.ones((1,), dtype=np.float32),
                "cube": np.zeros((1,), dtype=np.float32),
            }
        )
        second_gradient = second(
            {
                "square": np.zeros((1,), dtype=np.float32),
                "cube": np.ones((1,), dtype=np.float32),
            }
        )

        np.testing.assert_array_equal(first_gradient["source"].to_numpy(), np.array([4.0], dtype=np.float32))
        np.testing.assert_array_equal(second_gradient["source"].to_numpy(), np.array([27.0], dtype=np.float32))

    def test_semantic_add_uses_builtin_program_stage(self) -> None:
        module = FanIn()
        _, pullback = vd.ad.vjp(
            module,
            wrt=("source",),
            outputs=("square", "cube"),
        )(vd.storage.from_numpy(np.array([2.0], dtype=np.float32)))
        gradients = pullback(
            {
                "square": np.array([1.0], dtype=np.float32),
                "cube": np.array([2.0], dtype=np.float32),
            }
        )
        np.testing.assert_array_equal(gradients["source"].to_numpy(), np.array([28.0], dtype=np.float32))

    def test_plain_module_call_does_not_compile_or_retain_vjp(self) -> None:
        module = FanIn()
        with mock.patch(
            "vernon_dsl._runtime.autodiff._compile_direct_vjp",
            side_effect=AssertionError("plain Module call compiled VJP"),
        ):
            outputs = module(vd.storage.from_numpy(np.array([2.0], dtype=np.float32)))
        np.testing.assert_array_equal(outputs.square.to_numpy(), np.array([4.0], dtype=np.float32))
        np.testing.assert_array_equal(outputs.cube.to_numpy(), np.array([8.0], dtype=np.float32))
        module(vd.storage.from_numpy(np.array([3.0], dtype=np.float32)))
        self.assertEqual(len(module._program_cache), 1)

    def test_primal_cache_hit_does_not_reexecute_forward(self) -> None:
        module = AnnotatedSquare()
        first = module(vd.storage.from_numpy(np.array([2.0], dtype=np.float32)).view(access="read"))
        with mock.patch.object(
            AnnotatedSquare,
            "forward",
            side_effect=AssertionError("cache hit reexecuted forward"),
        ):
            second = module(vd.storage.from_numpy(np.array([3.0], dtype=np.float32)).view(access="read"))

        np.testing.assert_array_equal(first.to_numpy(), np.array([4.0], dtype=np.float32))
        np.testing.assert_array_equal(second.to_numpy(), np.array([9.0], dtype=np.float32))

    def test_gpu_primal_cache_hit_executes_native_program_graph(self) -> None:
        try:
            vd.init(arch=vd.metal)
        except RuntimeUnavailableError:
            self.skipTest("Metal runtime is unavailable")
        try:
            module = FanIn()
            module(vd.storage.from_numpy(np.array([2.0], dtype=np.float32)))
            specialization = next(iter(module._program_cache.values()))
            self.assertIsNotNone(specialization.native_plan)
            with mock.patch(
                "vernon_dsl._runtime.kernel.Kernel.__call__",
                side_effect=AssertionError("native cache hit called Python Kernel.__call__"),
            ):
                outputs = module(vd.storage.from_numpy(np.array([3.0], dtype=np.float32)))
            np.testing.assert_array_equal(outputs.square.to_numpy(), np.array([9.0], dtype=np.float32))
            np.testing.assert_array_equal(outputs.cube.to_numpy(), np.array([27.0], dtype=np.float32))
        finally:
            vd.init(arch=vd.cpu)

    def test_pipeline_asset_retains_initialized_module(self) -> None:
        module = AnnotatedSquare()
        asset = vd.pipeline_asset(id="modules/annotated-square", program=module)
        self.assertIs(asset.program, module)

    def test_module_frontend_reflects_nested_kernel_parameter_types(self) -> None:
        from vernon_dsl.program import _parse_module_program

        parsed = _parse_module_program(FanIn())

        self.assertEqual(parsed.forward.arguments[0].role, "input.source")
        logical_type = parsed.forward.arguments[0].type.logical
        self.assertEqual(logical_type.arguments[0].name, "f32")
        self.assertEqual(logical_type.arguments[1:], ((1,), "read_write", "device"))
        self.assertEqual(
            tuple(
                operation.name for operation in parsed.forward.operations if operation.kind == "vernon_program.compute"
            ),
            (
                "FanIn.square.module_square",
                "FanIn.cube.module_cube",
            ),
        )
        self.assertEqual(
            tuple(operation.name for operation in parsed.forward.operations if operation.kind == "vernon.intrinsic"),
            ("empty_like", "empty_like"),
        )
        self.assertEqual(parsed.forward.operations[0].operands[0][0], "source")
        self.assertIsNone(parsed.backward)
        self.assertIn("func.func @forward", parsed.mlir)
        self.assertIn('name = "empty_like"', parsed.mlir)
        self.assertIn('operand_names = ["source", "output"]', parsed.mlir)
        self.assertEqual(
            tuple((implementation.callee, implementation.entry) for implementation in parsed.implementations),
            (
                ("FanIn.cube.module_cube", "module_cube"),
                ("FanIn.square.module_square", "module_square"),
            ),
        )

    def test_module_frontend_reuses_canonical_aggregate_logical_types(self) -> None:
        from vernon_dsl.program import _parse_module_program

        parsed = _parse_module_program(AggregateCopy())

        self.assertEqual(parsed.structs[0][0], "ModulePair")
        self.assertEqual(parsed.forward.arguments[0].type.logical.arguments[0].name, "ModulePair")
        self.assertIn('"vernon.struct"()', parsed.mlir)
        self.assertIn(
            '!vernon.tensor_view<!vernon.struct<"ModulePair">, [1], "read_write", "device">',
            parsed.mlir,
        )

    def test_module_frontend_rejects_ambiguous_unused_parameter(self) -> None:
        from vernon_dsl.program import _parse_module_program

        class Ambiguous(vd.Module):
            def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
                return vd.zeros(dtype=vd.f32, shape=(1,))

        with self.assertRaisesRegex(TypeError, "cannot infer Module parameter 'source'"):
            _parse_module_program(Ambiguous())

    def test_forward_rejects_host_from_numpy(self) -> None:
        from vernon_dsl.frontend.module_ast import ModuleParameterType, interpret_module_forward

        class HostAlloc(vd.Module):
            def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
                return vd.storage.from_numpy(np.array([1.0], dtype=np.float32))

        with self.assertRaisesRegex(TypeError, "host session"):
            interpret_module_forward(
                HostAlloc(),
                {"source": ModuleParameterType(vd.f32, (1,))},
            )

    def test_typed_storage_activity_drives_program_dependencies(self) -> None:
        lowered = module_square._lower()
        entry = next(
            function for function in lowered.frontend.typed_functions if function.source.name == module_square._entry
        )
        assert entry.storage_activity is not None
        self.assertEqual(entry.storage_activity.readable_roots, frozenset({"source"}))
        self.assertEqual(entry.storage_activity.writable_roots, frozenset({"output"}))
        self.assertEqual(entry.storage_activity.dependencies_for("output"), frozenset({"source"}))

    def test_init_must_call_super_and_cannot_capture_runtime_storage(self) -> None:
        class MissingSuper(vd.Module):
            def __init__(self) -> None:
                pass

            def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
                return source

        with self.assertRaisesRegex(RuntimeError, "super"):
            MissingSuper()(vd.storage.zeros(dtype=vd.f32, shape=(1,)))

        module = FanIn()
        with self.assertRaisesRegex(TypeError, "cannot capture"):
            module.state = vd.storage.zeros(dtype=vd.f32, shape=(1,))


if __name__ == "__main__":
    unittest.main()
