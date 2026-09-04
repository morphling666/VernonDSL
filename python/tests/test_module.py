from __future__ import annotations

import importlib
import unittest
from dataclasses import dataclass
from unittest import mock

import numpy as np
import vernon_dsl as vd
from vernon_dsl._runtime.session import RuntimeUnavailableError
from vernon_dsl.program_frontend import BuiltinDslProvider


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


@vd.kernel(workgroup_size=(1, 1, 1))
def module_scale(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
    factor: vd.i32,
) -> None:
    output[0] = source[0] * vd.f32(factor)


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


class ScaleByHostConstant(vd.Module):
    def __init__(self) -> None:
        super().__init__()
        self.factor = np.int32(3)

    def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
        output = vd.empty_like(source)
        module_scale(source, output, self.factor, grid=(1, 1, 1))
        return output


class MixedGrid(vd.Module):
    def forward(self, source: vd.TensorStorage, groups_x: vd.u32) -> vd.TensorStorage:
        output = vd.empty_like(source)
        module_square(source, output, grid=(groups_x, 2, 1))
        return output


def _builtin_program_value(value_id: int, dtype: str, shape: list[int]) -> dict[str, object]:
    rank_shape = ", ".join("-1" for _ in shape)
    return {
        "id": value_id,
        "type": f'!vernon.tensor_view<{dtype}, [{rank_shape}], "read_write", "device">',
        "dtype": dtype,
        "shape": shape,
        "value_layout": {
            "scope": "element",
            "layout_hash": f"test-{dtype}",
            "byte_size": 4,
            "alignment": 4,
            "leaves": [
                {
                    "path": [],
                    "dtype": dtype,
                    "byte_offset": 0,
                    "scalar_count": 1,
                    "shape": [],
                }
            ],
        },
    }


class ModuleTests(unittest.TestCase):
    def test_fusion_dsl_provider_lowers_builtin_add_request(self) -> None:
        values = {index: _builtin_program_value(index, "f32", [4]) for index in range(3)}
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
        values = {index: _builtin_program_value(index, "f32", [2, 3]) for index in range(3)}
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
        self.assertIn("global_invocation_id", implementation.mlir)
        self.assertIn('"vernon.get_shape"', implementation.mlir)
        self.assertIn("tensor<2xi32>", implementation.mlir)
        self.assertIn('!vernon.tensor_view<f32, [-1, -1], "write", "device">', implementation.mlir)
        self.assertNotIn("2x3", implementation.mlir)

    def test_fusion_dsl_provider_lowers_dynamic_rank_builtin_add(self) -> None:
        values = {index: _builtin_program_value(index, "f32", [-1, -1]) for index in range(3)}
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
        self.assertIn("[-1, -1]", implementation.mlir)

    def test_fusion_dsl_provider_lowers_builtin_copy_request(self) -> None:
        values = {index: _builtin_program_value(index, "f32", [4]) for index in range(2)}
        request = {
            "implementation_hint": "vernon.builtin.copy",
            "bindings": [
                {"parameter": "source", "value": 0},
                {"parameter": "output", "value": 1},
            ],
        }

        implementation = BuiltinDslProvider().lower(request, values)

        self.assertIsNotNone(implementation)
        assert implementation is not None
        self.assertEqual(implementation.callee, "vernon.builtin.copy")
        self.assertEqual(implementation.entry, "_program_copy_f32_rank1")

    def test_fusion_dsl_provider_selects_copy_kernel_by_program_value_rank(self) -> None:
        rank_one = BuiltinDslProvider().lower(
            {
                "implementation_hint": "vernon.builtin.copy",
                "bindings": [
                    {"parameter": "source", "value": 0},
                    {"parameter": "output", "value": 1},
                ],
            },
            {index: _builtin_program_value(index, "f32", [1]) for index in range(2)},
        )
        rank_two = BuiltinDslProvider().lower(
            {
                "implementation_hint": "vernon.builtin.copy",
                "bindings": [
                    {"parameter": "source", "value": 0},
                    {"parameter": "output", "value": 1},
                ],
            },
            {index: _builtin_program_value(index, "f32", [-1, -1]) for index in range(2)},
        )

        self.assertIsNotNone(rank_one)
        self.assertIsNotNone(rank_two)
        assert rank_one is not None and rank_two is not None
        self.assertEqual(rank_one.callee, "vernon.builtin.copy")
        self.assertEqual(rank_two.callee, "vernon.builtin.copy")
        self.assertEqual(rank_one.entry, "_program_copy_f32_rank1")
        self.assertEqual(rank_two.entry, "_program_copy_f32_rank2")
        self.assertIn("[-1]", rank_one.mlir)
        self.assertIn("[-1, -1]", rank_two.mlir)
        self.assertNotIn("[-1, -1]", rank_one.mlir)

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

    def test_vjp_after_primal_kernel_load_reuses_interned_cpu_entries(self) -> None:
        module = FanIn()
        primal = module(vd.storage.from_numpy(np.array([2.0], dtype=np.float32)))
        np.testing.assert_array_equal(primal.square.to_numpy(), np.array([4.0], dtype=np.float32))
        np.testing.assert_array_equal(primal.cube.to_numpy(), np.array([8.0], dtype=np.float32))

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

    def test_module_definition_is_immutable_and_shared_by_class(self) -> None:
        first = AnnotatedSquare()
        second = AnnotatedSquare()
        definition = type(first)._module_definition

        self.assertIs(definition, type(second)._module_definition)
        self.assertIs(definition.original_function, AnnotatedSquare.__dict__["forward"])
        self.assertEqual(tuple(definition.signature.parameters), ("source",))
        self.assertIn("source", definition.resolved_annotations)
        self.assertEqual(definition.source_identity[2], "AnnotatedSquare.forward")
        with self.assertRaises(AttributeError):
            definition.__setattr__("signature", definition.signature)

    def test_vjp_cache_hit_does_not_recapture_or_reparse(self) -> None:
        module = AnnotatedSquare()
        transformed = vd.ad.vjp(module, wrt=("source",), outputs=("output",))
        transformed(vd.storage.from_numpy(np.array([2.0], dtype=np.float32)).view(access="read"))

        with (
            mock.patch(
                "vernon_dsl.program._capture_module",
                side_effect=AssertionError("VJP cache hit recaptured forward"),
            ),
            mock.patch(
                "vernon_dsl.program_frontend.parse_program",
                side_effect=AssertionError("VJP cache hit reparsed Program"),
            ),
        ):
            output, _ = transformed(vd.storage.from_numpy(np.array([3.0], dtype=np.float32)).view(access="read"))

        np.testing.assert_array_equal(output.to_numpy(), np.array([9.0], dtype=np.float32))

    def test_gpu_primal_cache_hit_executes_canonical_program(self) -> None:
        try:
            vd.init(arch=vd.metal)
        except RuntimeUnavailableError:
            self.skipTest("Metal runtime is unavailable")
        try:
            module = FanIn()
            module(vd.storage.from_numpy(np.array([2.0], dtype=np.float32)))
            specialization = next(iter(module._program_cache.values()))
            self.assertIsNotNone(specialization.native_program)
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
        asset = vd.program_asset(id="modules/annotated-square", program=module)
        self.assertIs(asset.program, module)

    def test_module_frontend_reflects_nested_kernel_parameter_types(self) -> None:
        from vernon_dsl.program import _parse_module_program

        parsed = _parse_module_program(FanIn())

        self.assertIn("func.func @forward", parsed.mlir)
        self.assertNotIn("func.func @backward", parsed.mlir)
        self.assertIn(
            '%v0: !vernon.tensor_view<f32, [1], "read_write", "device"> {vernon.source_name = "source"',
            parsed.mlir,
        )
        self.assertIn('debug_name = "FanIn.square.module_square"', parsed.mlir)
        self.assertIn('debug_name = "FanIn.cube.module_cube"', parsed.mlir)
        self.assertEqual(parsed.mlir.count('debug_name = "empty_like"'), 2)
        self.assertIn('operand_names = ["source", "output"]', parsed.mlir)
        self.assertEqual(
            tuple((implementation.callee, implementation.entry) for implementation in parsed.implementations),
            (
                ("FanIn.cube.module_cube", "module_cube"),
                ("FanIn.square.module_square", "module_square"),
            ),
        )

    def test_host_static_kernel_scalars_are_implementation_constants(self) -> None:
        from vernon_dsl.program import _parse_module_program

        parsed = _parse_module_program(ScaleByHostConstant())
        self.assertEqual(
            parsed.implementations[0].host_constants,
            (("factor", 3),),
        )
        self.assertIn('constant_names = ["factor"]', parsed.mlir)

    def test_grid_axes_mix_static_and_program_value_controls(self) -> None:
        from vernon_dsl.program import _parse_module_program

        parsed = _parse_module_program(MixedGrid())
        self.assertIn("grid = array<i64: 1, 2, 1>", parsed.mlir)
        self.assertIn("vernon_program.grid_control_arguments = array<i64: 1, -1, -1>", parsed.mlir)

    def test_module_frontend_reuses_canonical_aggregate_logical_types(self) -> None:
        from vernon_dsl.program import _parse_module_program

        parsed = _parse_module_program(AggregateCopy())

        self.assertEqual(parsed.structs[0][0], "ModulePair")
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
        from vernon_dsl.frontend.module_ast import interpret_module_forward
        from vernon_dsl.frontend.runtime_types import RuntimeParameterDescriptor

        class HostAlloc(vd.Module):
            def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
                return vd.storage.from_numpy(np.array([1.0], dtype=np.float32))

        with self.assertRaisesRegex(TypeError, "host session"):
            interpret_module_forward(
                HostAlloc(),
                {"source": RuntimeParameterDescriptor.storage(vd.f32, (1,))},
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
