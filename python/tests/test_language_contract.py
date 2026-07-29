from __future__ import annotations

import ast
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import vernon_dsl as vd
from vernon_dsl import CompileError, Compiler, compile_source
from vernon_dsl.compiler import FrontendCompileRequest
from vernon_dsl.frontend.abi import attribute_layout, value_abi_layout
from vernon_dsl.frontend.analysis import dump_typed_model, typed_effect_data, typed_model_data
from vernon_dsl.frontend.model import (
    AccessMode,
    AtomicEffect,
    BarrierEffect,
    ConcreteType,
    Effect,
    EffectScope,
    MemoryOrdering,
    ResourceEffect,
    StorageEffect,
    StorageEffectKind,
    StorageOwner,
    StorageOwnerKind,
    StorageRegion,
    StorageRegionKind,
    Termination,
    TypedFunctionInstance,
    TypedParameter,
    TypedStatement,
)
from vernon_dsl.language.stage_registry import (
    ENTRY_DECORATOR_STAGES,
    GRAPHICS_STAGE_ORDER,
    STAGE_REGISTRY_VERSION,
    validate_graphics_topology,
)
from vernon_dsl.shader_contracts import ATOMIC_OPERATION_NAMES, DEVICE_ONLY_OPERATION_NAMES


class LanguageVersionTests(unittest.TestCase):
    def test_stage_registry_is_versioned_and_canonical(self) -> None:
        self.assertEqual(STAGE_REGISTRY_VERSION, 2)
        self.assertEqual(
            ENTRY_DECORATOR_STAGES,
            {"kernel": "compute", "vertex": "vertex", "fragment": "fragment"},
        )
        self.assertEqual(
            validate_graphics_topology(GRAPHICS_STAGE_ORDER),
            ("vertex", "fragment"),
        )
        self.assertEqual(validate_graphics_topology(("vertex", "fragment")), ("vertex", "fragment"))
        with self.assertRaisesRegex(ValueError, "topology order"):
            validate_graphics_topology(reversed(GRAPHICS_STAGE_ORDER))

    def test_v3_vector_and_matrix_aliases_are_removed(self) -> None:
        for name in ("vec", "mat", "vec2", "vec3", "vec4", "mat2", "mat3", "mat4"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(vd, name))

    def test_tensor_values_and_tensor_view_resources_are_distinct(self) -> None:
        element = ConcreteType("scalar", "f32")
        tensor = ConcreteType("tensor", "Tensor", (element, 4))
        view = ConcreteType("tensor_view", "TensorView", (element, (4,), "write", "device"))
        parameter = TypedParameter("output", view, AccessMode.WRITE)

        self.assertEqual(tensor.mlir, "tensor<4xf32>")
        self.assertEqual(
            parameter.type.mlir,
            '!vernon.tensor_view<f32, [4], "write", "device">',
        )
        self.assertEqual(parameter.access, AccessMode.WRITE)

    def test_portable_value_abi_layout_is_deterministic(self) -> None:
        f16 = ConcreteType("scalar", "f16")
        f32 = ConcreteType("scalar", "f32")
        f64 = ConcreteType("scalar", "f64")
        boolean = ConcreteType("scalar", "bool")
        tuple_type = ConcreteType("tuple", "Tuple", (f16, f64, boolean))
        vertex = ConcreteType("struct", "Vertex")
        fields = {
            "Vertex": (
                ("position", ConcreteType("tensor", "Tensor", (f32, 3))),
                ("weight", f64),
            )
        }

        tuple_layout = value_abi_layout(tuple_type, fields.__getitem__)
        self.assertEqual(
            (tuple_layout.size, tuple_layout.alignment, tuple_layout.field_offsets),
            (24, 8, (0, 8, 16)),
        )
        vertex_layout = value_abi_layout(vertex, fields.__getitem__)
        self.assertEqual(
            (vertex_layout.size, vertex_layout.alignment, vertex_layout.field_offsets),
            (24, 8, (0, 16)),
        )
        tensor_layout = value_abi_layout(
            ConcreteType("tensor", "Tensor", (vertex, 2)),
            fields.__getitem__,
        )
        self.assertEqual(
            (tensor_layout.size, tensor_layout.alignment, tensor_layout.element_stride),
            (48, 8, 24),
        )

        output = compile_source(
            "from vernon_dsl import *\n@struct\nclass Vertex:\n    position: Tensor[f32, (3,)]\n    weight: f64\n",
            "abi_layout.py",
        )
        self.assertIn("vernon.value_abi_version = 1 : i64", output)
        self.assertIn('abi_leaf_dtypes = ["f32", "f64"]', output)
        self.assertNotIn("abi_alignment", output)
        self.assertNotIn("abi_field_offsets", output)
        self.assertNotIn("abi_size", output)

        shared_output = compile_source(
            "from vernon_dsl import *\n"
            "@func(shared=True)\n"
            "def preserve(values: Tensor[Tuple[f32, i32], (2,)])"
            " -> Tensor[Tuple[f32, i32], (2,)]:\n"
            "    return values\n",
            "shared_abi_layout.py",
        )
        self.assertIn("vernon.shared", shared_output)
        self.assertIn('vernon.abi_leaf_dtypes = ["f32", "i32", "f32", "i32"]', shared_output)
        self.assertNotIn("vernon.abi_alignment", shared_output)
        self.assertNotIn("vernon.abi_element_stride", shared_output)
        self.assertNotIn("vernon.abi_size", shared_output)

    def test_aggregate_value_and_attribute_layout_share_recursive_leaves(self) -> None:
        f16 = ConcreteType("scalar", "f16")
        f32 = ConcreteType("scalar", "f32")
        u32 = ConcreteType("scalar", "u32")
        vertex = ConcreteType("struct", "Vertex")
        fields = {
            "Vertex": (
                ("position", ConcreteType("tensor", "Tensor", (f32, 3))),
                ("object_id", u32),
                ("uv", ConcreteType("tensor", "Tensor", (f16, 2))),
            )
        }

        value = value_abi_layout(vertex, fields.__getitem__)
        attributes = attribute_layout(vertex, fields.__getitem__)

        self.assertEqual((value.size, value.alignment, value.field_offsets), (20, 4, (0, 12, 16)))
        self.assertEqual(
            tuple((leaf.path, leaf.dtype, leaf.byte_offset, leaf.scalar_count) for leaf in value.leaves),
            (
                (("position",), "f32", 0, 3),
                (("object_id",), "u32", 12, 1),
                (("uv",), "f16", 16, 2),
            ),
        )
        self.assertEqual(tuple(leaf.shape for leaf in value.leaves), ((3,), (), (2,)))
        self.assertEqual(
            tuple(
                (leaf.location_offset, leaf.path, leaf.dtype, leaf.component_count, leaf.byte_offset)
                for leaf in attributes.leaves
            ),
            (
                (0, ("position",), "f32", 3, 0),
                (1, ("object_id",), "u32", 1, 12),
                (2, ("uv",), "f16", 2, 16),
            ),
        )
        self.assertEqual(len(value.layout_hash), 64)

    def test_aggregate_attribute_leaves_do_not_cross_tensor_element_or_field_boundaries(self) -> None:
        f32 = ConcreteType("scalar", "f32")
        i32 = ConcreteType("scalar", "i32")
        element = ConcreteType(
            "tuple",
            "Tuple",
            (ConcreteType("tensor", "Tensor", (f32, 4)), i32),
        )
        value_type = ConcreteType("tensor", "Tensor", (element, 2, 3))

        layout = attribute_layout(value_type, lambda _: ())

        self.assertEqual(layout.location_span, 12)
        self.assertEqual(
            tuple((leaf.path, leaf.dtype, leaf.byte_offset) for leaf in layout.leaves[:4]),
            (
                ((0, 0, 0), "f32", 0),
                ((0, 0, 1), "i32", 16),
                ((0, 1, 0), "f32", 20),
                ((0, 1, 1), "i32", 36),
            ),
        )

    def test_frontend_and_semantic_identity_are_version_three(self) -> None:
        source = "from vernon_dsl import *\n@fragment\ndef main(value: float) -> float:\n    return value\n"
        output = compile_source(source, "version.py")
        self.assertIn("vernon.frontend_version = 4 : i64", output)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shader.py"
            path.write_text(source, encoding="utf-8")
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    path,
                    "main",
                    captured_constants=(("LIMIT", 3),),
                )
            )
            self.assertEqual(result.semantic_inputs["frontend_version"], 4)
            self.assertEqual(result.semantic_inputs["captured_constants"], [["LIMIT", "int", 3]])

    def test_semantic_identity_contains_concrete_helper_specializations(self) -> None:
        source = (
            "from vernon_dsl import *\n"
            "@func\n"
            "def identity(value):\n"
            "    return value\n"
            "@fragment\n"
            "def main(value: f64) -> f64:\n"
            "    return identity(value)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "specialization.py"
            path.write_text(source, encoding="utf-8")
            first = Compiler().compile_request(FrontendCompileRequest(path, "main"))
            second = Compiler().compile_request(FrontendCompileRequest(path, "main"))
        expected = [["identity", ["f64"], []]]
        self.assertEqual(first.semantic_inputs["helper_specializations"], expected)
        self.assertEqual(second.semantic_inputs["helper_specializations"], expected)
        self.assertEqual(dump_typed_model(first.typed_functions), dump_typed_model(second.typed_functions))

    def test_frontend_cache_hits_and_invalidates_transitive_dependencies(self) -> None:
        Compiler.clear_cache()
        self.addCleanup(Compiler.clear_cache)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "helper.py"
            shader = root / "shader.py"
            helper.write_text(
                "from vernon_dsl import *\n@func\ndef scale(value: f32) -> f32:\n    return value * 2.0\n",
                encoding="utf-8",
            )
            shader.write_text(
                "from vernon_dsl import *\n"
                "from helper import scale\n"
                "@fragment\n"
                "def main(value: f32) -> f32:\n"
                "    return scale(value)\n",
                encoding="utf-8",
            )
            request = FrontendCompileRequest(shader, "main")
            first = Compiler().compile_request(request)
            second = Compiler().compile_request(request)
            self.assertIs(first, second)

            helper.write_text(
                "from vernon_dsl import *\n@func\ndef scale(value: f32) -> f32:\n    return value * 3.0\n",
                encoding="utf-8",
            )
            third = Compiler().compile_request(request)

        self.assertIsNot(first, third)
        self.assertNotEqual(first.dependencies, third.dependencies)
        self.assertNotEqual(first.mlir, third.mlir)

    def test_failed_frontend_requests_reproduce_current_diagnostics(self) -> None:
        Compiler.clear_cache()
        self.addCleanup(Compiler.clear_cache)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.py"
            path.write_text(
                "from vernon_dsl import *\n@fragment\ndef main(value: f32):\n    return value\n",
                encoding="utf-8",
            )
            request = FrontendCompileRequest(path, "main")
            with self.assertRaisesRegex(CompileError, "requires a result annotation"):
                Compiler().compile_request(request)

            path.write_text(
                "from vernon_dsl import *\n@fragment\ndef main(value: f32) -> f32:\n    return missing\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CompileError, "cannot infer unknown value 'missing'"):
                Compiler().compile_request(request)

            path.write_text(
                "from vernon_dsl import *\n@fragment\ndef main(value: f32) -> f32:\n    return value\n",
                encoding="utf-8",
            )
            result = Compiler().compile_request(request)

        self.assertIn("func.func @main", result.mlir)

    def test_runtime_workgroup_size_is_a_semantic_compile_input(self) -> None:
        Compiler.clear_cache()
        self.addCleanup(Compiler.clear_cache)
        source = "from vernon_dsl import *\n@kernel\ndef main() -> None:\n    pass\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workgroup.py"
            path.write_text(source, encoding="utf-8")
            first = Compiler().compile_request(FrontendCompileRequest(path, "main", workgroup_size=(2, 3, 1)))
            second = Compiler().compile_request(FrontendCompileRequest(path, "main", workgroup_size=(4, 1, 1)))

        self.assertIn("vernon.workgroup_size = array<i32: 2, 3, 1>", first.mlir)
        self.assertIn("vernon.workgroup_size = array<i32: 4, 1, 1>", second.mlir)
        self.assertNotEqual(first.mlir, second.mlir)

    def test_graphics_stages_share_project_parsing_and_dependency_discovery(self) -> None:
        Compiler.clear_cache()
        self.addCleanup(Compiler.clear_cache)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper_source = "from vernon_dsl import *\n@func\ndef tint(value):\n    return value * 2.0\n"
            shader_source = (
                "from vernon_dsl import *\n"
                "from helper import tint\n"
                "@vertex\n"
                "def vertex_main(value: Vector[f32, 4]) -> Vector[f32, 4]:\n"
                "    return tint(value)\n"
                "@fragment\n"
                "def fragment_main(value: f32) -> f32:\n"
                "    return tint(value)\n"
            )
            helper = root / "helper.py"
            shader = root / "shader.py"
            helper.write_text(helper_source, encoding="utf-8")
            shader.write_text(shader_source, encoding="utf-8")

            parsed_sources: list[str] = []
            original_parse = ast.parse

            def record_parse(source: str, *args: object, **kwargs: object) -> ast.AST:
                parsed_sources.append(source)
                return original_parse(source, *args, **kwargs)

            with mock.patch("vernon_dsl.module_graph.ast.parse", side_effect=record_parse):
                vertex = Compiler().compile_request(FrontendCompileRequest(shader, "vertex_main"))
                fragment = Compiler().compile_request(FrontendCompileRequest(shader, "fragment_main"))

        self.assertEqual(parsed_sources.count(shader_source), 1)
        self.assertEqual(parsed_sources.count(helper_source), 1)
        self.assertEqual(vertex.dependencies, fragment.dependencies)

    def test_typed_model_records_effects_lvalues_and_branch_merges(self) -> None:
        source = (
            "from vernon_dsl import *\n"
            "@fragment\n"
            "def main(value: f32, condition: bool) -> f32:\n"
            "    result = value\n"
            "    if condition:\n"
            "        result = result + 1\n"
            "    else:\n"
            "        result = result + 2\n"
            "    return result\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "typed_model.py"
            path.write_text(source, encoding="utf-8")
            result = Compiler().compile_request(FrontendCompileRequest(path, "main"))
        function = next(function for function in result.typed_functions if function.symbol == "main")
        branch = function.body[1]
        self.assertEqual([(merge.name, merge.type.mlir) for merge in branch.branch_merges], [("result", "f32")])
        self.assertTrue(branch.children[0].lvalues)
        self.assertEqual(branch.children[0].effects, ())
        self.assertEqual(branch.children[0].effect, Effect.PURE)
        self.assertEqual(branch.effect, Effect.PURE)
        self.assertIn('"operation": "add"', dump_typed_model(result.typed_functions))

    def test_structured_storage_effects_track_owner_and_region(self) -> None:
        source = (
            "from typing import Annotated\n"
            "from vernon_dsl import *\n"
            "@kernel\n"
            "def main(\n"
            "    output: TensorView[f32, (dyn,), write],\n"
            "    source: TensorView[f32, (dyn,), read],\n"
            "    gid: Annotated[Tensor[u32, (3,)], builtin('global_invocation_id')],\n"
            ") -> None:\n"
            "    local = source[2]\n"
            "    output[gid[0]] = local\n"
            "    if gid[0] > 0:\n"
            "        output[1] = source[gid[0]]\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured_effects.py"
            path.write_text(source, encoding="utf-8")
            first = Compiler().compile_request(FrontendCompileRequest(path, "main"))
            second = Compiler().compile_request(FrontendCompileRequest(path, "main"))

        function = next(function for function in first.typed_functions if function.symbol == "main")
        read_static, write_dynamic, branch = function.body
        self.assertEqual(
            read_static.effects,
            (
                StorageEffect(
                    StorageEffectKind.READ,
                    StorageOwner(StorageOwnerKind.PARAMETER, "source"),
                    StorageRegion(StorageRegionKind.ELEMENT, (2,)),
                ),
            ),
        )
        self.assertEqual(
            write_dynamic.effects,
            (
                StorageEffect(
                    StorageEffectKind.WRITE,
                    StorageOwner(StorageOwnerKind.PARAMETER, "output"),
                    StorageRegion(StorageRegionKind.UNKNOWN),
                ),
            ),
        )
        self.assertEqual(
            branch.children[0].effects,
            (
                StorageEffect(
                    StorageEffectKind.READ,
                    StorageOwner(StorageOwnerKind.PARAMETER, "source"),
                    StorageRegion(StorageRegionKind.UNKNOWN),
                ),
                StorageEffect(
                    StorageEffectKind.WRITE,
                    StorageOwner(StorageOwnerKind.PARAMETER, "output"),
                    StorageRegion(StorageRegionKind.ELEMENT, (1,)),
                ),
            ),
        )
        self.assertEqual(read_static.effect, Effect.READ)
        self.assertEqual(write_dynamic.effect, Effect.WRITE)
        self.assertEqual(branch.effect, Effect.WRITE)
        self.assertEqual(
            function.effects,
            (
                read_static.effects[0],
                write_dynamic.effects[0],
                branch.children[0].effects[0],
                branch.children[0].effects[1],
            ),
        )
        self.assertEqual(dump_typed_model(first.typed_functions), dump_typed_model(second.typed_functions))
        self.assertEqual(first.semantic_inputs["entry_effects"], second.semantic_inputs["entry_effects"])
        self.assertEqual(
            first.semantic_inputs["entry_effects"],
            [typed_effect_data(effect) for effect in function.effects],
        )
        self.assertNotIn("allocation", str(first.semantic_inputs["entry_effects"]))
        self.assertNotIn("transfer", str(first.semantic_inputs["entry_effects"]))
        self.assertIn("vernon.storage_effects = [", first.mlir)
        self.assertEqual(
            typed_model_data(first.typed_functions)[0]["body"][0]["effects"],
            [
                {
                    "kind": "read",
                    "owner": "source",
                    "owner_kind": "parameter",
                    "region": {"kind": "element", "indices": [2]},
                }
            ],
        )

    def test_value_and_resource_reads_are_not_storage_effects(self) -> None:
        value_source = (
            "from vernon_dsl import *\n@fragment\ndef main(value: Tensor[f32, (2,)]) -> f32:\n    return value[0]\n"
        )
        resource_source = (
            "from vernon_dsl import *\n"
            "@fragment\n"
            "def main(image: Texture['2d', f32], sampler: Sampler, uv: Vector[f32, 2]) -> Vector[f32, 4]:\n"
            "    return texture_sample(image, sampler, uv)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            value_path = Path(directory) / "value_effect.py"
            value_path.write_text(value_source, encoding="utf-8")
            value_result = Compiler().compile_request(FrontendCompileRequest(value_path, "main"))
            resource_path = Path(directory) / "resource_effect.py"
            resource_path.write_text(resource_source, encoding="utf-8")
            resource_result = Compiler().compile_request(FrontendCompileRequest(resource_path, "main"))

        value_statement = value_result.typed_functions[0].body[0]
        resource_statement = resource_result.typed_functions[0].body[0]
        self.assertEqual(value_statement.effects, ())
        self.assertEqual(value_statement.effect, Effect.PURE)
        self.assertEqual(resource_statement.effects, (ResourceEffect("texture_sample", "image"),))
        self.assertEqual(resource_statement.effect, Effect.READ)
        self.assertEqual(
            typed_effect_data(resource_statement.effects[0]),
            {"kind": "resource_read", "operation": "texture_sample", "owner": "image"},
        )

    def test_atomic_and_barrier_effect_records_are_deterministic(self) -> None:
        function_source = ast.parse("@kernel\ndef main() -> None:\n    pass\n").body[0]
        assert isinstance(function_source, ast.FunctionDef)
        atomic = AtomicEffect(
            "add",
            StorageOwner(StorageOwnerKind.PARAMETER, "values"),
            StorageRegion(StorageRegionKind.ELEMENT, (3,)),
            MemoryOrdering.RELAXED,
            EffectScope.DEVICE,
        )
        barrier = BarrierEffect(MemoryOrdering.ACQUIRE_RELEASE, EffectScope.WORKGROUP)
        statement = TypedStatement(
            function_source.body[0],
            Termination.FALLTHROUGH,
            (atomic, barrier),
        )
        function = TypedFunctionInstance(
            "main",
            "main",
            (),
            None,
            (),
            function_source,
            (statement,),
            effects=(atomic, barrier),
        )

        model = typed_model_data((function,))[0]
        self.assertEqual(model["body"][0]["effect"], "write")
        self.assertEqual(
            model["effects"],
            [
                {
                    "kind": "atomic",
                    "operation": "add",
                    "owner": "values",
                    "owner_kind": "parameter",
                    "region": {"kind": "element", "indices": [3]},
                    "ordering": "relaxed",
                    "scope": "device",
                },
                {
                    "kind": "barrier",
                    "ordering": "acquire_release",
                    "scope": "workgroup",
                },
            ],
        )
        self.assertEqual(dump_typed_model((function,)), dump_typed_model((function,)))

    def test_synchronization_registry_matches_public_device_only_api(self) -> None:
        synchronization_operations = ATOMIC_OPERATION_NAMES | {
            "storage_barrier",
            "workgroup_storage",
            "workgroup_barrier",
        }
        self.assertEqual(
            ATOMIC_OPERATION_NAMES,
            {"atomic_add", "atomic_exchange", "atomic_max", "atomic_min"},
        )
        self.assertTrue(synchronization_operations <= DEVICE_ONLY_OPERATION_NAMES)

        calls = {
            **{name: (None, 0, 1) for name in ATOMIC_OPERATION_NAMES},
            "storage_barrier": (),
            "workgroup_storage": (vd.i32,),
            "workgroup_barrier": (),
        }
        for name, arguments in calls.items():
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, rf"^{name} is device-only"):
                (
                    getattr(vd, name)(*arguments, shape=(4,))
                    if name == "workgroup_storage"
                    else getattr(vd, name)(*arguments)
                )

    def test_workgroup_storage_atomics_and_barriers_have_typed_effects(self) -> None:
        source = (
            "from vernon_dsl import *\n"
            "@kernel\n"
            "def main(output: TensorView[i32, (dyn,), write]) -> None:\n"
            "    signed = workgroup_storage(i32, shape=(4,))\n"
            "    unsigned = workgroup_storage(u32, shape=(2,))\n"
            "    signed[0] = 4\n"
            "    unsigned[0] = u32(4)\n"
            "    workgroup_barrier()\n"
            "    added = atomic_add(signed, 0, 1)\n"
            "    minimum = atomic_min(signed, 1, 2)\n"
            "    maximum = atomic_max(signed, 2, 3)\n"
            "    exchanged = atomic_exchange(signed, 3, 4)\n"
            "    unsigned_minimum = atomic_min(unsigned, 0, 2)\n"
            "    unsigned_maximum = atomic_max(unsigned, 1, 3)\n"
            "    output[0] = added + minimum + maximum + exchanged\n"
            "    storage_barrier()\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synchronization.py"
            path.write_text(source, encoding="utf-8")
            result = Compiler().compile_request(FrontendCompileRequest(path, "main"))

        self.assertEqual(result.mlir.count('"vernon.workgroup_alloc"'), 2)
        self.assertEqual(result.mlir.count('"vernon.store"'), 3)
        self.assertEqual(result.mlir.count('"vernon.atomic"'), 6)
        for atomic_kind in ("add", "min", "max", "exchange", "umin", "umax"):
            with self.subTest(atomic_kind=atomic_kind):
                self.assertEqual(result.mlir.count(f'atomic_kind = "{atomic_kind}"'), 1)
        atomic_effects = [effect for effect in result.typed_functions[0].effects if isinstance(effect, AtomicEffect)]
        self.assertTrue(all(effect.owner.kind is StorageOwnerKind.WORKGROUP_LOCAL for effect in atomic_effects))
        self.assertEqual(
            Counter((effect.operation, effect.scope) for effect in atomic_effects),
            Counter(
                {
                    ("add", EffectScope.WORKGROUP): 1,
                    ("min", EffectScope.WORKGROUP): 2,
                    ("max", EffectScope.WORKGROUP): 2,
                    ("exchange", EffectScope.WORKGROUP): 1,
                }
            ),
        )
        barriers = [effect for effect in result.typed_functions[0].effects if isinstance(effect, BarrierEffect)]
        self.assertEqual(
            barriers,
            [
                BarrierEffect(MemoryOrdering.ACQUIRE_RELEASE, EffectScope.WORKGROUP),
                BarrierEffect(MemoryOrdering.ACQUIRE_RELEASE, EffectScope.DEVICE),
            ],
        )

    def test_rank_two_aggregate_workgroup_storage_uses_typed_tensor_view_ops(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@struct\n"
            "class Pair:\n"
            "    left: i32\n"
            "    right: f32\n"
            "@kernel\n"
            "def main() -> None:\n"
            "    values = workgroup_storage(Pair, shape=(2, 3))\n"
            "    values[1, 2] = Pair(7, 2.5)\n"
            "    loaded = values[1, 2]\n",
            "aggregate_workgroup.py",
        )
        view = '!vernon.tensor_view<!vernon.struct<"Pair">, [2, 3], "read_write", "workgroup">'
        self.assertIn(view, output)
        self.assertIn('"vernon.store"', output)
        self.assertIn('"vernon.load"', output)
        self.assertNotIn("strides = array", output)

    def test_all_storage_tensor_view_atomics_use_device_scope(self) -> None:
        source = (
            "from vernon_dsl import *\n"
            "@kernel\n"
            "def main(\n"
            "    signed: TensorView[i32, (dyn,), read_write],\n"
            "    unsigned: TensorView[u32, (dyn,), read_write],\n"
            ") -> None:\n"
            "    added = atomic_add(signed, 0, 1)\n"
            "    minimum = atomic_min(signed, 1, 2)\n"
            "    maximum = atomic_max(signed, 2, 3)\n"
            "    exchanged = atomic_exchange(signed, 3, 4)\n"
            "    unsigned_minimum = atomic_min(unsigned, 0, 2)\n"
            "    unsigned_maximum = atomic_max(unsigned, 1, 3)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "storage_atomic.py"
            path.write_text(source, encoding="utf-8")
            result = Compiler().compile_request(FrontendCompileRequest(path, "main"))

        self.assertEqual(result.mlir.count('"vernon.atomic"'), 6)
        for atomic_kind in ("add", "min", "max", "exchange", "umin", "umax"):
            with self.subTest(atomic_kind=atomic_kind):
                self.assertEqual(result.mlir.count(f'atomic_kind = "{atomic_kind}"'), 1)
        atomic_effects = [effect for effect in result.typed_functions[0].effects if isinstance(effect, AtomicEffect)]
        self.assertTrue(all(effect.owner.kind is StorageOwnerKind.PARAMETER for effect in atomic_effects))
        self.assertEqual(
            Counter((effect.operation, effect.scope) for effect in atomic_effects),
            Counter(
                {
                    ("add", EffectScope.DEVICE): 1,
                    ("min", EffectScope.DEVICE): 2,
                    ("max", EffectScope.DEVICE): 2,
                    ("exchange", EffectScope.DEVICE): 1,
                }
            ),
        )

        for operation in sorted(ATOMIC_OPERATION_NAMES):
            with (
                self.subTest(readonly_operation=operation),
                self.assertRaisesRegex(CompileError, "requires a writable TensorView"),
            ):
                compile_source(
                    "from vernon_dsl import *\n"
                    "@kernel\n"
                    "def bad(values: TensorView[i32, (dyn,), read]) -> None:\n"
                    f"    {operation}(values, 0, 1)\n",
                    f"readonly_{operation}.py",
                )

    def test_atomic_prefix_does_not_reserve_helper_names(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def atomic_decoy(value: i32) -> i32:\n"
            "    return value\n"
            "@kernel\n"
            "def main(destination: TensorView[i32, (dyn,), write]) -> None:\n"
            "    destination[0] = atomic_decoy(7)\n",
            "atomic_prefix_helper.py",
        )
        self.assertIn("func.call @atomic_decoy(", output)

    def test_atomic_owner_must_be_a_named_storage_value(self) -> None:
        with self.assertRaisesRegex(CompileError, "requires a named storage owner"):
            compile_source(
                "from vernon_dsl import *\n"
                "@kernel\n"
                "def bad(values: TensorView[i32, (dyn,), read_write]) -> None:\n"
                "    atomic_add(values if True else values, 0, 1)\n",
                "atomic_owner_expression.py",
            )

    def test_workgroup_synchronization_rejects_non_compute_and_invalid_types(self) -> None:
        with self.assertRaisesRegex(CompileError, "only in compute kernels"):
            compile_source(
                "from vernon_dsl import *\n@fragment\ndef bad() -> None:\n    workgroup_barrier()\n",
                "fragment_barrier.py",
            )
        with self.assertRaisesRegex(CompileError, "positive compile-time integers"):
            compile_source(
                "from vernon_dsl import *\n"
                "@kernel\n"
                "def bad() -> None:\n"
                "    values = workgroup_storage(f32, shape=(0,))\n",
                "invalid_workgroup.py",
            )

    def test_pure_helper_rejects_propagated_storage_effects(self) -> None:
        with self.assertRaisesRegex(CompileError, "pure helper 'copy' has Storage effects"):
            compile_source(
                "from vernon_dsl import *\n"
                "@func\n"
                "def copy(output: TensorView[f32, (dyn,), write], source: TensorView[f32, (dyn,), read]) -> None:\n"
                "    output[0] = source[0]\n"
                "@kernel\n"
                "def main(output: TensorView[f32, (dyn,), write], source: TensorView[f32, (dyn,), read]) -> None:\n"
                "    copy(output, source)\n",
                "effectful_helper.py",
            )

    def test_helper_call_rejects_known_incompatible_aliases(self) -> None:
        with self.assertRaisesRegex(CompileError, "helper call 'combine' has incompatible aliased Storage effects"):
            compile_source(
                "from vernon_dsl import *\n"
                "@func\n"
                "def combine(\n"
                "    left: TensorView[f32, (dyn,), read_write],\n"
                "    right: TensorView[f32, (dyn,), read_write],\n"
                ") -> None:\n"
                "    left[0] = right[0]\n"
                "@kernel\n"
                "def main(data: TensorView[f32, (dyn,), read_write]) -> None:\n"
                "    combine(data, data)\n",
                "aliased_helper.py",
            )

    def test_removed_v2_spelling_and_array_fail_at_the_frontend(self) -> None:
        with self.assertRaisesRegex(CompileError, "unknown DSL decorator 'compute'"):
            compile_source(
                "from vernon_dsl import *\n@compute\ndef main(value: f32) -> f32:\n    return value\n",
                "compute.py",
            )
        with self.assertRaisesRegex(CompileError, "unknown DSL type constructor 'Array'"):
            compile_source(
                "from vernon_dsl import *\n@func\ndef main(value: Array[f32, 4]) -> f32:\n    return 0.0\n",
                "array.py",
            )

    def test_tensor_elements_are_recursively_abi_stable_values(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@struct\n"
            "class Particle:\n"
            "    position: Tensor[f32, (3,)]\n"
            "    lifetime: f32\n"
            "@func\n"
            "def identity(value: Tensor[Particle, (4,)]) -> Tensor[Particle, (4,)]:\n"
            "    return value\n",
            "aggregate_elements.py",
        )
        self.assertIn('!vernon.tensor<!vernon.struct<"Particle">, [4]>', output)

    def test_nested_tensor_types_canonicalize_by_shape_composition(self) -> None:
        scalar = ConcreteType("scalar", "f32")
        nested = ConcreteType(
            "tensor",
            "Tensor",
            (ConcreteType("tensor", "Tensor", (scalar, 3)), 8),
        )
        flat = ConcreteType("tensor", "Tensor", (scalar, 8, 3))
        self.assertEqual(nested, flat)

        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def identity(\n"
            "    value: Tensor[Tensor[f32, (3,)], (8,)],\n"
            ") -> Tensor[f32, (8, 3)]:\n"
            "    return value\n",
            "nested_tensor.py",
        )
        self.assertIn("tensor<8x3xf32>", output)
        self.assertNotIn("tensor<8xtensor", output)

    def test_tensor_rejects_storage_and_resource_elements(self) -> None:
        cases = (
            ("TensorView[f32, (dyn,), read]", "Storage 'TensorView'"),
            ('Texture["2d", f32]', "Resource 'Texture'"),
            ("Sampler", "Resource 'Sampler'"),
        )
        for index, (element, diagnostic) in enumerate(cases):
            with self.subTest(element=element), self.assertRaisesRegex(CompileError, diagnostic):
                compile_source(
                    f"from vernon_dsl import *\n@func\ndef bad(value: Tensor[{element}, (2,)]) -> None:\n    pass\n",
                    f"invalid_tensor_element_{index}.py",
                )

    def test_struct_fields_must_be_finite_abi_stable_values(self) -> None:
        cases = (
            (
                "@struct\nclass Bad:\n    data: TensorView[f32, (dyn,), read]\n",
                "Struct field 'Bad.data' must be an ABI-stable Value",
            ),
            (
                "@struct\nclass Recursive:\n    next: Recursive\n",
                "Struct field 'Recursive.next' must be an ABI-stable Value",
            ),
        )
        for index, (declaration, diagnostic) in enumerate(cases):
            with self.subTest(index=index), self.assertRaisesRegex(CompileError, diagnostic):
                compile_source(
                    f"from vernon_dsl import *\n{declaration}",
                    f"invalid_struct_{index}.py",
                )

    def test_lazy_short_circuit_boolean_expressions_yield_values(self) -> None:
        source = (
            "from vernon_dsl import *\n"
            "@func\n"
            "def choose(a: bool, b: bool, c: bool) -> bool:\n"
            "    return a and b and c\n"
            "@kernel\n"
            "def main(\n"
            "    output: TensorView[i32, (dyn,), write],\n"
            "    source: TensorView[i32, (dyn,), read],\n"
            "    enabled: bool,\n"
            ") -> None:\n"
            "    if enabled and source[0] > 0:\n"
            "        output[0] = 1\n"
        )
        output = compile_source(source, "short_circuit.py")
        self.assertGreaterEqual(output.count("scf.if"), 3)
        self.assertNotIn("arith.andi", output)
        expression_if = output.index("scf.if", output.index("@main"))
        load = output.index('"vernon.load"', expression_if)
        outer_if = output.index("scf.if", load)
        self.assertLess(expression_if, load)
        self.assertLess(load, outer_if)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short_circuit.py"
            path.write_text(source, encoding="utf-8")
            result = Compiler().compile_request(FrontendCompileRequest(path, "main"))
        main = next(function for function in result.typed_functions if function.symbol == "main")
        boolean_expression = next(
            expression for expression in main.body[0].expressions if expression.operation == "and"
        )
        self.assertEqual([value.mlir for value in boolean_expression.operand_types], ["i1", "i1"])

        with self.assertRaisesRegex(CompileError, "and/or operands must be bool Values"):
            compile_source(
                "from vernon_dsl import *\n@func\ndef bad(a: f32, b: f32) -> f32:\n    return a or b\n",
                "numeric_short_circuit.py",
            )

    def test_nested_return_uses_structured_payload_state(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n@func\ndef stop(a: bool) -> f32:\n"
            "    while a:\n        if a:\n            return 1.0\n    return 0.0\n",
            "return.py",
        )
        self.assertIn("scf.while", output)
        self.assertEqual(output.count("func.return"), 1)

        with self.assertRaisesRegex(CompileError, "may exit without returning a value"):
            compile_source(
                "from vernon_dsl import *\n@func\ndef incomplete(a: bool) -> f32:\n    if a:\n        return 1.0\n",
                "incomplete_return.py",
            )

    def test_conditional_expressions_unify_types_and_merge_effects(self) -> None:
        source = (
            "from vernon_dsl import *\n"
            "@func\n"
            "def choose(condition: bool, narrow: f32, wide: f64) -> f64:\n"
            "    return narrow if condition else wide\n"
            "@kernel\n"
            "def main(\n"
            "    output: TensorView[f32, (dyn,), write],\n"
            "    left: TensorView[f32, (dyn,), read],\n"
            "    right: TensorView[f32, (dyn,), read],\n"
            "    condition: bool,\n"
            ") -> None:\n"
            "    output[0] = left[0] if condition else right[0]\n"
        )
        output = compile_source(source, "conditional_expression.py")
        self.assertGreaterEqual(output.count("scf.if"), 2)
        self.assertIn("arith.extf", output)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditional_expression.py"
            path.write_text(source, encoding="utf-8")
            result = Compiler().compile_request(FrontendCompileRequest(path, "main"))
        main = next(function for function in result.typed_functions if function.symbol == "main")
        conditional = next(
            expression for expression in main.body[0].expressions if expression.operation == "conditional"
        )
        self.assertEqual([value.mlir for value in conditional.operand_types], ["i1", "f32", "f32"])
        self.assertEqual(
            [(effect.kind.value, effect.owner.name) for effect in main.effects],
            [("read", "left"), ("read", "right"), ("write", "output")],
        )
        self.assertEqual(
            result.semantic_inputs["entry_effects"],
            [typed_effect_data(effect) for effect in main.effects],
        )

        with self.assertRaisesRegex(CompileError, "conditional expression branches have incompatible types"):
            compile_source(
                "from vernon_dsl import *\n"
                "@func\n"
                "def bad(condition: bool, value: f32) -> f32:\n"
                "    return value if condition else False\n",
                "invalid_conditional_expression.py",
            )

    def test_typed_break_and_continue_target_nearest_loop(self) -> None:
        source = (
            "from vernon_dsl import *\n"
            "@func\n"
            "def sum_odds(limit: i32) -> i32:\n"
            "    index = 0\n"
            "    total = 0\n"
            "    while index < limit:\n"
            "        index += 1\n"
            "        if index % 2 == 0:\n"
            "            continue\n"
            "        if index > 7:\n"
            "            break\n"
            "        total += index\n"
            "    return total\n"
            "@fragment\n"
            "def main(limit: i32) -> i32:\n"
            "    return sum_odds(limit)\n"
        )
        output = compile_source(source, "loop_control.py")
        self.assertIn("scf.while", output)
        self.assertIn("arith.select", output)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loop_control.py"
            path.write_text(source, encoding="utf-8")
            result = Compiler().compile_request(FrontendCompileRequest(path, "main"))
        function = next(function for function in result.typed_functions if function.qualified_name == "sum_odds")
        loop = function.body[2]
        exits = [
            child
            for statement in loop.children
            for child in statement.children
            if child.termination in {Termination.BREAK, Termination.CONTINUE}
        ]
        self.assertEqual(
            [(statement.termination.value, statement.loop_depth) for statement in exits],
            [("continue", 1), ("break", 1)],
        )
        self.assertEqual(function.body[-1].return_type, ConcreteType("scalar", "i32"))

        for keyword in ("break", "continue"):
            with (
                self.subTest(keyword=keyword),
                self.assertRaisesRegex(CompileError, f"{keyword} is only valid inside a loop"),
            ):
                compile_source(
                    f"from vernon_dsl import *\n@func\ndef bad() -> None:\n    {keyword}\n",
                    f"invalid_{keyword}.py",
                )

    def test_dynamic_range_contract_and_type_rules(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def total(start: i32, stop: i32, step: i32) -> i32:\n"
            "    result = 0\n"
            "    for index in range(start, stop, step):\n"
            "        result += i32(index)\n"
            "    return result\n",
            "dynamic_range.py",
        )
        self.assertIn("cf.assert", output)
        self.assertIn("scf.while", output)
        self.assertNotIn("scf.for", output)

        with self.assertRaisesRegex(CompileError, "range step must not be zero"):
            compile_source(
                "from vernon_dsl import *\n"
                "@func\n"
                "def zero_step() -> i32:\n"
                "    result = 0\n"
                "    for index in range(0, 4, 0):\n"
                "        result += i32(index)\n"
                "    return result\n",
                "zero_range.py",
            )

        with self.assertRaisesRegex(CompileError, "range u32 arguments require an explicit i32 conversion"):
            compile_source(
                "from vernon_dsl import *\n"
                "@func\n"
                "def unsigned_range(stop: u32) -> i32:\n"
                "    result = 0\n"
                "    for index in range(stop):\n"
                "        result += i32(index)\n"
                "    return result\n",
                "unsigned_range.py",
            )


class NumericInferenceTests(unittest.TestCase):
    def test_safe_promotion_and_integer_true_division(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def mixed(iterations: i32, scale: f64) -> f64:\n"
            "    ratio = iterations / 2\n"
            "    return ratio + iterations * 0.02 + scale\n",
            "numeric.py",
        )
        self.assertIn("arith.divf", output)
        self.assertIn("arith.sitofp", output)
        self.assertIn("arith.extf", output)

    def test_complete_safe_scalar_promotion_lattice(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def promote(half: f16, single: f32, double: f64, signed: i32, unsigned: u32) -> f64:\n"
            "    a = half + single\n"
            "    b = a + double\n"
            "    c = signed + b\n"
            "    return unsigned + c\n",
            "promotions.py",
        )
        self.assertGreaterEqual(output.count("arith.extf"), 2)
        self.assertIn("arith.sitofp", output)
        self.assertIn("arith.uitofp", output)

    def test_unsafe_implicit_conversions_are_rejected(self) -> None:
        cases = (
            (
                "def bad(value: f64) -> f32:\n    return value\n",
                "unsafe implicit conversion from f64 to f32",
            ),
            (
                "def bad(value: f32) -> i32:\n    return value\n",
                "unsafe implicit conversion from f32 to i32",
            ),
            (
                "def bad(left: i32, right: u32) -> i32:\n    return left + right\n",
                "no safe common type",
            ),
            (
                "def bad(left: bool, right: bool) -> bool:\n    return left + right\n",
                "unsupported binary operation",
            ),
        )
        for index, (function, diagnostic) in enumerate(cases):
            with self.subTest(index=index), self.assertRaisesRegex(CompileError, diagnostic):
                compile_source(
                    f"from vernon_dsl import *\n@func\n{function}",
                    f"unsafe_{index}.py",
                )

    def test_literals_are_contextual_on_both_operand_sides(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def contextual(value: f64, iterations: i32) -> f64:\n"
            "    left = 0.25 * value\n"
            "    right = value * 2\n"
            "    ratio = 1 / iterations\n"
            "    return left + right + ratio\n",
            "contextual.py",
        )
        self.assertIn("arith.constant 0.25 : f64", output)
        self.assertIn("arith.extf", output)
        self.assertIn("arith.divf", output)

    def test_literals_are_contextual_in_calls_constructors_and_comparisons(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def takes_double(value: f64) -> f64:\n"
            "    return value\n"
            "@func\n"
            "def contexts(value: f64) -> f64:\n"
            "    called = takes_double(2)\n"
            "    vector = Vector([1, value])\n"
            "    compared = value > 0\n"
            "    if compared:\n"
            "        called = called + vector[0]\n"
            "    return called\n",
            "literal_contexts.py",
        )
        self.assertIn("func.call @takes_double", output)
        self.assertIn("tensor<2xf64>", output)
        self.assertIn("arith.cmpf", output)

    def test_scalar_literal_constraints_avoid_default_then_cast(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def takes_double(value: f64) -> f64:\n"
            "    return value\n"
            "@func\n"
            "def constrained(value: f64) -> f64:\n"
            "    return takes_double(2) + (3 + value)\n",
            "literal_constraints.py",
        )
        self.assertEqual(output.count("arith.constant 2.0 : f64"), 1)
        self.assertEqual(output.count("arith.constant 3.0 : f64"), 1)
        self.assertNotIn("arith.sitofp", output)

    def test_generic_literal_constraints_precede_specialization(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def annotated(value: f64):\n"
            "    return value\n"
            "@func\n"
            "def unconstrained(value):\n"
            "    return value\n"
            "@fragment\n"
            "def main() -> f64:\n"
            "    return annotated(1 + 2.0) + f64(unconstrained(3))\n",
            "generic_literal_constraints.py",
        )
        self.assertRegex(output, r"func\.func private @annotated__[a-f0-9]+[^(]*\(%arg0: f64")
        self.assertRegex(output, r"func\.func private @unconstrained__[a-f0-9]+[^(]*\(%arg0: i32")

    def test_partially_annotated_helper_rejects_unsafe_argument_and_result(self) -> None:
        cases = (
            (
                "def helper(value: i32):\n    return value\n",
                "value: f32",
                "cannot pass f32 as i32",
            ),
            (
                "def helper(value) -> i32:\n    return value\n",
                "value: f32",
                "cannot return f32 as i32",
            ),
        )
        for index, (helper, argument, diagnostic) in enumerate(cases):
            with self.subTest(index=index), self.assertRaisesRegex(CompileError, diagnostic):
                compile_source(
                    "from vernon_dsl import *\n"
                    f"@func\n{helper}"
                    "@fragment\n"
                    f"def main({argument}) -> i32:\n"
                    "    return helper(value)\n",
                    f"partial_annotation_{index}.py",
                )

    def test_unannotated_helpers_are_monomorphized_deterministically(self) -> None:
        source = (
            "import vernon_dsl as vd\n"
            "@vd.func\n"
            "def scale(value):\n"
            "    return value * 2\n"
            "@vd.fragment\n"
            "def single(value: vd.f32) -> vd.f32:\n"
            "    return scale(value)\n"
            "@vd.fragment\n"
            "def double(value: vd.f64) -> vd.f64:\n"
            "    return scale(value)\n"
        )
        first = compile_source(source, "specialize.py")
        second = compile_source(source, "specialize.py")
        self.assertEqual(first, second)
        self.assertEqual(first.count("func.func private @scale__"), 2)
        self.assertIn(": f32", first)
        self.assertIn(": f64", first)

    def test_helper_tensor_shapes_have_separate_specializations(self) -> None:
        output = compile_source(
            "import vernon_dsl as vd\n"
            "@vd.func\n"
            "def first(value):\n"
            "    return value[0]\n"
            "@vd.fragment\n"
            "def pair(value: vd.Tensor[vd.f32, (2,)]) -> vd.f32:\n"
            "    return first(value)\n"
            "@vd.fragment\n"
            "def triple(value: vd.Tensor[vd.f32, (3,)]) -> vd.f32:\n"
            "    return first(value)\n",
            "shape_specialization.py",
        )
        self.assertEqual(output.count("func.func private @first__"), 2)

    def test_inferred_void_helper_and_unresolved_calls_have_stable_behavior(self) -> None:
        output = compile_source(
            "import vernon_dsl as vd\n"
            "@vd.func\n"
            "def observe(value):\n"
            "    copy = value\n"
            "@vd.kernel\n"
            "def main(value: vd.f32) -> None:\n"
            "    observe(value)\n",
            "void_helper.py",
        )
        self.assertIn("func.func private @observe__", output)
        self.assertIn("func.call @observe__", output)

        with self.assertRaisesRegex(CompileError, "cannot infer call to 'missing'"):
            compile_source(
                "from vernon_dsl import *\n"
                "@func\n"
                "def unresolved(value):\n"
                "    return missing(value)\n"
                "@fragment\n"
                "def main(value: f32) -> f32:\n"
                "    return unresolved(value)\n",
                "unresolved.py",
            )

    def test_unreachable_conflicting_return_is_ignored_but_empty_value_conflict_is_rejected(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def first(value):\n"
            "    return value\n"
            "    return f64(0)\n"
            "@fragment\n"
            "def main(value: f32) -> f32:\n"
            "    return first(value)\n",
            "unreachable_return.py",
        )
        self.assertRegex(output, r"func\.func private @first__[a-f0-9]+.*-> \(f32\)")

        with self.assertRaisesRegex(CompileError, "mixes value and empty returns"):
            compile_source(
                "from vernon_dsl import *\n"
                "@func\n"
                "def bad(value) -> f32:\n"
                "    return\n"
                "@fragment\n"
                "def main(value: f32) -> f32:\n"
                "    return bad(value)\n",
                "empty_value_conflict.py",
            )

    def test_recursive_generic_specialization_is_rejected(self) -> None:
        with self.assertRaisesRegex(CompileError, "recursive helper specialization"):
            compile_source(
                "from vernon_dsl import *\n"
                "@func\n"
                "def recursive(value):\n"
                "    return recursive(value)\n"
                "@fragment\n"
                "def main(value: f32) -> f32:\n"
                "    return recursive(value)\n",
                "recursive_generic.py",
            )

    def test_imported_qualified_helper_specializes_per_feature_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "helpers.py").write_text(
                "from vernon_dsl import func\n@func\ndef identity(value):\n    return value\n",
                encoding="utf-8",
            )
            main = root / "main.py"
            main.write_text(
                "from vernon_dsl import *\n"
                "import helpers\n"
                'DOUBLE = feature("DOUBLE")\n'
                "@fragment\n"
                "def main(value: f32) -> f32:\n"
                "    selected = value\n"
                "    if DOUBLE:\n"
                "        selected = f32(helpers.identity(f64(value)))\n"
                "    else:\n"
                "        selected = helpers.identity(value)\n"
                "    return selected\n",
                encoding="utf-8",
            )
            disabled = Compiler().compile_request(FrontendCompileRequest(main, "main"))
            enabled = Compiler().compile_request(FrontendCompileRequest(main, "main", ("DOUBLE",)))

        disabled_keys = [
            argument_types for name, argument_types, _ in disabled.helper_specializations if name.endswith("identity")
        ]
        enabled_keys = [
            argument_types for name, argument_types, _ in enabled.helper_specializations if name.endswith("identity")
        ]
        self.assertEqual(disabled_keys, [("f32",)])
        self.assertEqual(enabled_keys, [("f64",)])

    def test_partial_helper_annotations_and_matrix_inference(self) -> None:
        output = compile_source(
            "import vernon_dsl as vd\n"
            "@vd.func\n"
            "def transform(matrix: vd.Tensor[vd.f32, (2, 2)], value):\n"
            "    return vd.matmul(matrix, value)\n"
            "@vd.fragment\n"
            "def main(value: vd.Tensor[vd.f32, (2,)]) -> vd.Tensor[vd.f32, (2,)]:\n"
            "    matrix = vd.Matrix([[1, 2.0], [3, 4]])\n"
            "    return transform(matrix, value)\n",
            "matrix.py",
        )
        self.assertIn("tensor<2x2xf32>", output)
        self.assertIn('name = "matmul"', output)

    def test_generic_struct_construction_and_field_inference(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@struct\n"
            "class Pair:\n"
            "    first: f64\n"
            "    second: f64\n"
            "@func\n"
            "def pair(value):\n"
            "    return Pair(value, 2)\n"
            "@fragment\n"
            "def main(value: f64) -> f64:\n"
            "    return pair(value).second\n",
            "generic_struct.py",
        )
        self.assertIn('!vernon.struct<"Pair">', output)
        self.assertIn("vernon.struct_get", output)

    def test_kernel_store_infers_resource_and_widens_value(self) -> None:
        output = compile_source(
            "from typing import Annotated\n"
            "from vernon_dsl import *\n"
            "@kernel\n"
            "def main(\n"
            "    output: Annotated[TensorView[f32, (dyn,), read_write], resource(set=0, binding=0)],\n"
            "    index: u32,\n"
            "    value: f16,\n"
            ") -> None:\n"
            "    output[index] = value\n",
            "kernel_tensor_view.py",
        )
        self.assertIn(
            '!vernon.tensor_view<f32, [-1], "read_write", "device">',
            output,
        )
        self.assertIn("arith.extf", output)
        self.assertIn('"vernon.store"', output)

    def test_store_rejects_read_only_resource(self) -> None:
        with self.assertRaisesRegex(CompileError, "cannot assign through a read-only TensorView"):
            compile_source(
                "from typing import Annotated\n"
                "from vernon_dsl import *\n"
                "@func\n"
                "def store(output, value):\n"
                "    output[0] = value\n"
                "@kernel\n"
                "def main(\n"
                "    output: Annotated[TensorView[f32, (dyn,), read], resource(set=0, binding=0)],\n"
                "    value: f32,\n"
                ") -> None:\n"
                "    store(output, value)\n",
                "readonly_tensor_view.py",
            )

    def test_matrix_rejects_ragged_and_empty_literals(self) -> None:
        for index, expression in enumerate(("Matrix([])", "Matrix([[1], [2, 3]])")):
            with (
                self.subTest(expression=expression),
                self.assertRaisesRegex(
                    CompileError,
                    "Matrix",
                ),
            ):
                compile_source(
                    f"from vernon_dsl import *\n@func\ndef bad() -> Matrix[f32, 2, 2]:\n    return {expression}\n",
                    f"bad_matrix_{index}.py",
                )

    def test_vector_and_matrix_constructors_share_tensor_types(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def aggregates(value: f64) -> Vector[f64, 2]:\n"
            "    vector = Vector([1, value])\n"
            "    matrix = Matrix([[1, value], [3.0, 4]])\n"
            "    return vector + matmul(matrix, vector)\n",
            "aggregate_parity.py",
        )
        self.assertGreaterEqual(output.count("tensor<2xf64>"), 2)
        self.assertIn("tensor<2x2xf64>", output)

    def test_tensor_is_the_canonical_rectangular_value_constructor(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def tensor_value(value: f64) -> Tensor[f64, (2, 2, 1)]:\n"
            "    return Tensor([[[1], [value]], [[3.0], [4]]])\n",
            "tensor_constructor.py",
        )
        self.assertIn("tensor<2x2x1xf64>", output)
        self.assertIn('name = "construct"', output)

        with self.assertRaisesRegex(CompileError, "non-empty rectangular"):
            compile_source(
                "from vernon_dsl import *\n"
                "@func\n"
                "def ragged() -> Tensor[f32, (2, 2)]:\n"
                "    return Tensor([[1], [2, 3]])\n",
                "ragged_tensor.py",
            )

    def test_vector_and_matrix_are_tensor_rank_aliases(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def aliases(\n"
            "    vector: Vector[f32, 2],\n"
            "    matrix: Matrix[f32, 2, 2],\n"
            ") -> Tensor[f32, (2,)]:\n"
            "    return vector + matmul(matrix, vector)\n",
            "rank_aliases.py",
        )
        self.assertIn("tensor<2xf32>", output)
        self.assertIn("tensor<2x2xf32>", output)

    def test_tuple_construction_constant_indexing_and_destructuring(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def make(value: f64) -> Tuple[f64, i32]:\n"
            "    return (value, 2)\n"
            "@func\n"
            "def consume(value: f64) -> f64:\n"
            "    pair = make(value)\n"
            "    first, second = pair\n"
            "    return first + f64(second) + pair[0]\n",
            "tuple_values.py",
        )
        self.assertIn("tuple<f64, i32>", output)
        self.assertIn('"vernon.tuple_create"', output)
        self.assertIn('"vernon.tuple_get"', output)

    def test_tensor_may_contain_homogeneous_tuple_values(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def records(value: f32) -> Tensor[Tuple[f32, i32], (2,)]:\n"
            "    return Tensor([Tuple(value, 1), Tuple(value, 2)])\n",
            "tuple_tensor.py",
        )
        self.assertIn("!vernon.tensor<tuple<f32, i32>, [2]>", output)

    def test_tuple_indexing_requires_an_in_bounds_constant(self) -> None:
        cases = (
            ("pair[index]", "Tuple indexing requires an integer literal"),
            ("pair[2]", "Tuple index is out of bounds"),
        )
        for index, (expression, diagnostic) in enumerate(cases):
            with self.subTest(expression=expression), self.assertRaisesRegex(CompileError, diagnostic):
                compile_source(
                    "from vernon_dsl import *\n"
                    "@func\n"
                    "def bad(pair: Tuple[f32, i32], index: i32) -> f32:\n"
                    f"    return {expression}\n",
                    f"tuple_index_{index}.py",
                )

    def test_operator_and_intrinsic_power_share_types(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n@func\ndef powers(value: f64) -> f64:\n    return value ** 2 + pow(value, 2)\n",
            "power_parity.py",
        )
        self.assertEqual(output.count('name = "pow"'), 2)
        self.assertNotIn("arith.truncf", output)

    def test_branch_and_loop_carried_values_widen_safely(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def control_flow(condition: bool) -> f32:\n"
            "    branch = f16(1)\n"
            "    if condition:\n"
            "        branch = f32(2)\n"
            "    else:\n"
            "        branch = f16(3)\n"
            "    loop = f16(0)\n"
            "    running = condition\n"
            "    while running:\n"
            "        loop = loop + f32(1)\n"
            "        running = False\n"
            "    return branch + loop\n",
            "control_flow_widening.py",
        )
        self.assertIn("scf.if", output)
        self.assertIn("scf.while", output)
        self.assertIn("-> (f32)", output)

    def test_generic_loop_inference_converges_after_nested_specialization(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def increment(value):\n"
            "    return value + f32(1)\n"
            "@func\n"
            "def accumulate(value, running):\n"
            "    result = value\n"
            "    while running:\n"
            "        result = increment(result)\n"
            "        running = False\n"
            "    return result\n"
            "@fragment\n"
            "def main(value: f16, running: bool) -> f32:\n"
            "    return accumulate(value, running)\n",
            "generic_loop.py",
        )
        self.assertRegex(output, r"func\.func private @accumulate__[a-f0-9]+.*-> \(f32\)")
        self.assertRegex(output, r"func\.func private @increment__[a-f0-9]+.*-> \(f32\)")
        self.assertEqual(output.count("func.func private @increment__"), 1)

    def test_vector_constructor_and_intrinsic_method_share_canonical_lowering(self) -> None:
        output = compile_source(
            "import vernon_dsl as vd\n"
            "@vd.func\n"
            "def complex_sqr(z):\n"
            "    value = vd.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])\n"
            "    length = value.norm()\n"
            "    return value\n"
            "@vd.fragment\n"
            "def main(z: vd.Tensor[vd.f32, (2,)]) -> vd.Tensor[vd.f32, (2,)]:\n"
            "    return complex_sqr(z)\n",
            "vector.py",
        )
        self.assertIn('name = "construct"', output)
        self.assertIn('name = "dot"', output)
        self.assertIn("func.func private @complex_sqr__", output)

    def test_intrinsic_method_and_function_have_matching_result_types(self) -> None:
        output = compile_source(
            "from vernon_dsl import *\n"
            "@func\n"
            "def lengths(value: Vector[f64, 2]) -> f64:\n"
            "    return value.norm() + norm(value)\n",
            "norm_parity.py",
        )
        self.assertEqual(output.count('name = "dot"'), 2)
        self.assertEqual(output.count("math.sqrt"), 2)


class EntryAbiTests(unittest.TestCase):
    def test_entries_require_parameter_and_result_annotations(self) -> None:
        cases = (
            (
                "@kernel\ndef main(value) -> None:\n    pass\n",
                "entry argument 'value' requires a type annotation",
            ),
            (
                "@kernel\ndef main(value: f32):\n    pass\n",
                "entry function 'main' requires a result annotation",
            ),
        )
        for index, (function, diagnostic) in enumerate(cases):
            with self.subTest(index=index), self.assertRaisesRegex(CompileError, diagnostic):
                compile_source(
                    f"from vernon_dsl import *\n{function}",
                    f"entry_abi_{index}.py",
                )


if __name__ == "__main__":
    unittest.main()
