from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import vernon_dsl as vd
import vernon_dsl._runtime.autodiff as runtime_autodiff
from vernon_dsl.compiler import Compiler, FrontendCompileRequest
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.frontend.autodiff_profiles import derivative_groups_from_paths
from vernon_dsl.frontend.structured_vjp import build_structured_vjp
from vernon_dsl.host_values import tangent_layout

from python.tests.storage_vjp_direct_fixture import (
    NestedRecord,
    Particle,
    TensorProductLeaf,
    TensorProductRecord,
    aggregate_field_objective_vjp,
    aggregate_multi_output_objective_vjp,
    aggregate_output_objective_vjp,
    aggregate_output_pair_objective_vjp,
    aggregate_storage_objective_vjp,
    aliased_inputs_objective_vjp,
    branch_scratch_objective_vjp,
    dynamic_for_objective_vjp,
    dynamic_gather_objective_vjp,
    dynamic_scratch_gather_objective_vjp,
    dynamic_while_objective_vjp,
    loop_scratch_objective_vjp,
    nested_aggregate_objective_vjp,
    nested_branch_accumulation_objective_vjp,
    nested_dynamic_scratch_objective_vjp,
    overwrite_objective_vjp,
    partially_dynamic_objective_vjp,
    scratch_objective_vjp,
    shared_nested_inputs_objective_vjp,
    storage_objective_vjp,
    vector_scratch_objective_vjp,
)


class StorageVjpContractTests(unittest.TestCase):
    def compile_source(
        self,
        source: str,
        *,
        transform: vd.ad.ProgramTransformSpec | None = None,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "storage_vjp.py"
            path.write_text(source, encoding="utf-8")
            return Compiler().compile_request(FrontendCompileRequest(path, "objective", program_transform=transform))

    def test_derivative_groups_reject_duplicate_declared_and_leaf_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "group paths must be unique"):
            derivative_groups_from_paths("gradient", ("source", "source"), ("source.value",))
        with self.assertRaisesRegex(ValueError, "leaf paths must be unique"):
            derivative_groups_from_paths(
                "gradient",
                ("source",),
                ("source.value", "source.value"),
            )

    def test_direct_vjp_rejects_boolean_grid_dimensions(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        with self.assertRaisesRegex(ValueError, "three positive integers"):
            storage_objective_vjp(values, loss, grid=(True, 1, 1))

    def test_dynamic_void_storage_objective_is_accepted(self) -> None:
        from vernon_dsl import _native  # pyright: ignore[reportAttributeAccessIssue]

        transform = vd.ad.ProgramTransformSpec(
            "vjp",
            ("values",),
            output_cotangents=("loss",),
        )
        result = self.compile_source(
            """
import vernon_dsl as vd

@vd.kernel
def objective(
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss[0] = values[0] * values[0]
""",
            transform=transform,
        )
        self.assertIsNone(result.autodiff_profiles)
        structured = build_structured_vjp(_native, result, transform)
        backward = structured.profiles["backward"]
        self.assertIn('!vernon.tensor_view<f32, [-1], "write", "device">', backward)
        self.assertNotIn("-> tensor<-1", backward)
        compiled = _native.Compiler().compile_program_result(backward, _native.Target.CPU)
        self.assertTrue(compiled.ok, compiled.diagnostics)

    def test_frontend_nested_canonical_ranges_reach_native_vjp_profiles(self) -> None:
        from vernon_dsl import _native  # pyright: ignore[reportAttributeAccessIssue]

        transform = vd.ad.ProgramTransformSpec(
            "vjp",
            ("values",),
            output_cotangents=("loss",),
        )
        result = self.compile_source(
            """
import vernon_dsl as vd

@vd.kernel
def objective(
    values: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    target: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    count: vd.i32,
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    total = 0.0
    for y in range(count):
        for x in range(count):
            difference = values[y, x] - target[y, x]
            total = total + difference * difference
    loss[0] = total
""",
            transform=transform,
        )
        self.assertEqual(result.mlir.count("scf.for"), 2)
        self.assertNotIn("scf.while", result.mlir)

        structured = build_structured_vjp(_native, result, transform)
        self.assertEqual(structured.residual_storage_kind, "none")
        forward = structured.profiles["forward_with_tape"]
        backward = structured.profiles["backward"]
        self.assertEqual(forward.count("scf.for"), 2)
        self.assertEqual(backward.count("scf.for"), 2)
        self.assertNotIn("scf.while", forward)
        self.assertNotIn("scf.while", backward)

    def test_disjoint_no_tape_profile_compiles_for_gpu_targets(self) -> None:
        from vernon_dsl import _native  # pyright: ignore[reportAttributeAccessIssue]

        transform = vd.ad.ProgramTransformSpec(
            "vjp",
            ("values",),
            output_cotangents=("loss",),
        )
        result = self.compile_source(
            """
from typing import Annotated
import vernon_dsl as vd

@vd.kernel(workgroup_size=(4, 1, 1))
def objective(
    values: vd.TensorView[vd.f32, (4,), vd.read],
    loss: vd.TensorView[vd.f32, (4,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    loss[gid[0]] = values[gid[0]] * values[gid[0]]
""",
            transform=transform,
        )
        structured = build_structured_vjp(_native, result, transform)
        self.assertEqual(structured.residual_storage_kind, "none")
        for target in (
            _native.Target.CUDA,
            _native.Target.VULKAN,
            _native.Target.METAL,
            _native.Target.OPENGL,
        ):
            with self.subTest(target=target.name):
                compiled = _native.Compiler().compile_program_result(
                    structured.profiles["backward"],
                    target,
                )
                self.assertTrue(compiled.ok, compiled.diagnostics)

    def test_non_void_compute_kernel_is_rejected(self) -> None:
        with self.assertRaisesRegex(CompileError, "must return None"):
            self.compile_source(
                """
import vernon_dsl as vd

@vd.kernel
def objective(value: vd.f32) -> vd.f32:
    return value
""",
                transform=vd.ad.ProgramTransformSpec(
                    "vjp",
                    ("value",),
                    output_cotangents=("loss",),
                ),
            )

    def test_output_must_be_writable_storage(self) -> None:
        source = """
import vernon_dsl as vd

@vd.kernel
def objective(
    values: vd.TensorView[vd.f32, (1,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.read],
) -> None:
    pass
"""
        with self.assertRaisesRegex(CompileError, "must be writable"):
            self.compile_source(
                source,
                transform=vd.ad.ProgramTransformSpec(
                    "vjp",
                    ("values",),
                    output_cotangents=("loss",),
                ),
            )

    def test_unselected_writable_storage_is_not_an_objective(self) -> None:
        result = self.compile_source(
            """
import vernon_dsl as vd

@vd.kernel
def objective(
    values: vd.TensorView[vd.f32, (1,), vd.read],
    scratch: vd.TensorView[vd.f32, (1,), vd.read_write],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    scratch[0] = values[0]
    loss[0] = scratch[0] * scratch[0]
""",
            transform=vd.ad.ProgramTransformSpec(
                "vjp",
                ("values",),
                output_cotangents=("loss",),
            ),
        )
        assert result.request.program_transform is not None
        self.assertEqual(result.request.program_transform.output_cotangents, ("loss",))

    def test_native_transform_roots_reverse_mode_at_selected_store(self) -> None:
        from vernon_dsl import _native  # pyright: ignore[reportAttributeAccessIssue]

        transform = vd.ad.ProgramTransformSpec(
            "vjp",
            ("values",),
            output_cotangents=("loss",),
        )
        result = self.compile_source(
            """
import vernon_dsl as vd

@vd.kernel
def objective(
    values: vd.TensorView[vd.f32, (2,), vd.read],
    ignored: vd.TensorView[vd.f32, (1,), vd.write],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    ignored[0] = values[1]
    loss[0] = values[0] * values[0]
""",
            transform=transform,
        )
        structured = build_structured_vjp(_native, result, transform)
        self.assertEqual(structured.transform.output_cotangents, ("loss",))
        self.assertIn('vernon.source_name = "loss"', structured.profiles["backward"])
        self.assertNotIn('vernon.source_name = "ignored"', structured.profiles["backward"])

    def test_direct_storage_objective_returns_owned_gradient(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        output, pullback = storage_objective_vjp(values, loss, grid=(1, 1, 1))
        self.assertIsNone(output)
        self.assertEqual(float(loss.to_numpy()[0]), 13.0)
        gradients = pullback(None)
        self.assertIsInstance(gradients["values"], vd.TensorStorage)
        np.testing.assert_array_equal(
            gradients["values"].to_numpy(),
            np.array([6.0, 1.0], dtype=np.float32),
        )
        repeated = pullback(None)
        self.assertIsNot(repeated["values"], gradients["values"])
        np.testing.assert_array_equal(
            repeated["values"].to_numpy(),
            np.array([6.0, 1.0], dtype=np.float32),
        )

    def test_pullback_concurrent_apply_uses_fresh_invocation_state(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        _, pullback = storage_objective_vjp(values, loss, grid=(1, 1, 1))

        with ThreadPoolExecutor(max_workers=2) as executor:
            gradients = list(executor.map(lambda _: pullback(None), range(2)))

        self.assertIsNot(gradients[0]["values"], gradients[1]["values"])
        for gradient in gradients:
            np.testing.assert_array_equal(
                gradient["values"].to_numpy(),
                np.array([6.0, 1.0], dtype=np.float32),
            )

    def test_aggregate_storage_gradient_uses_one_packed_tangent_owner(self) -> None:
        vd.init(arch=vd.cpu)
        particles = vd.storage.zeros(dtype=Particle, shape=(1,))
        values = particles.to_numpy()
        values["velocity"][0] = np.array([2.0, -3.0], dtype=np.float16)
        values["mass"][0] = np.float32(4.0)
        values["tag"][0] = np.int32(7)
        particles.copy_from_numpy(values)
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))

        _, pullback = aggregate_storage_objective_vjp(particles, loss, grid=(1, 1, 1))
        gradient = pullback(None)["particles"]

        self.assertIsInstance(gradient, vd.TensorStorage)
        self.assertEqual(gradient.shape, (1,))
        self.assertEqual(gradient.dtype.names, ("velocity", "mass"))
        np.testing.assert_array_equal(
            gradient["velocity"].to_numpy(),
            np.array([[4.0, -6.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(gradient["mass"].to_numpy(), np.array([8.0], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-differentiable"):
            gradient["tag"]

    def test_tensor_of_products_projects_and_unpacks_structural_tangent_leaves(
        self,
    ) -> None:
        top_level = tangent_layout(vd.Tensor[TensorProductLeaf, (2,)])
        self.assertEqual(top_level.project("0.value").dtype, np.dtype(np.float32))
        self.assertEqual(top_level.project("1.value").dtype, np.dtype(np.float32))

        tangent = vd.storage.tangent_zeros(dtype=TensorProductRecord, shape=(1,))
        tangent["entries.0.value"].copy_from_numpy(np.array([2.0], dtype=np.float32))
        tangent["entries.1.value"].copy_from_numpy(np.array([3.0], dtype=np.float32))

        values = tangent.to_values()
        self.assertEqual(float(values[0].entries[0].value), 2.0)
        self.assertEqual(float(values[0].entries[1].value), 3.0)
        self.assertIsNone(values[0].entries[0].tag)
        with self.assertRaisesRegex(ValueError, "non-differentiable"):
            tangent["entries.0.tag"]

    def test_aggregate_storage_output_accepts_one_packed_cotangent(self) -> None:
        vd.init(arch=vd.cpu)
        particles = vd.storage.zeros(dtype=Particle, shape=(1,))
        values = particles.to_numpy()
        values["velocity"][0] = np.array([2.0, -3.0], dtype=np.float16)
        values["mass"][0] = np.float32(4.0)
        particles.copy_from_numpy(values)
        output = vd.storage.zeros(dtype=Particle, shape=(1,))
        _, pullback = aggregate_output_objective_vjp(particles, output, grid=(1, 1, 1))

        cotangent = vd.storage.tangent_zeros(dtype=Particle, shape=(1,))
        cotangent["velocity"].copy_from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
        cotangent["mass"].copy_from_numpy(np.array([3.0], dtype=np.float32))
        gradient = pullback(cotangent)["particles"]

        np.testing.assert_array_equal(
            gradient["velocity"].to_numpy(),
            np.array([[1.0, 2.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(gradient["mass"].to_numpy(), np.array([24.0], dtype=np.float32))

    def test_aggregate_output_cotangent_gathers_through_its_view_descriptor(
        self,
    ) -> None:
        vd.init(arch=vd.cpu)
        particles = vd.storage.zeros(dtype=Particle, shape=(2,))
        values = particles.to_numpy()
        values["velocity"][:] = np.array([[2.0, -3.0], [5.0, 6.0]], dtype=np.float16)
        values["mass"][:] = np.array([4.0, 5.0], dtype=np.float32)
        particles.copy_from_numpy(values)
        output_owner = vd.storage.zeros(dtype=Particle, shape=(3,))
        output = output_owner.view(shape=(2,), strides=(-1,), offset=2, access="write")
        _, pullback = aggregate_output_pair_objective_vjp(particles, output, grid=(1, 1, 1))

        cotangent = vd.storage.tangent_zeros(dtype=Particle, shape=(3,))
        cotangent["velocity"].copy_from_numpy(np.array([[9.0, 9.0], [1.0, 2.0], [4.0, 5.0]], dtype=np.float32))
        cotangent["mass"].copy_from_numpy(np.array([9.0, 3.0, 7.0], dtype=np.float32))
        gradient = pullback(cotangent)["particles"]

        np.testing.assert_array_equal(
            gradient["velocity"].to_numpy(),
            np.array([[4.0, 5.0], [1.0, 2.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(gradient["mass"].to_numpy(), np.array([56.0, 30.0], dtype=np.float32))

    def test_multiple_aggregate_output_cotangents_are_grouped_independently(
        self,
    ) -> None:
        vd.init(arch=vd.cpu)
        particles = vd.storage.zeros(dtype=Particle, shape=(1,))
        values = particles.to_numpy()
        values["velocity"][0] = np.array([2.0, -3.0], dtype=np.float16)
        values["mass"][0] = np.float32(4.0)
        particles.copy_from_numpy(values)
        first = vd.storage.zeros(dtype=Particle, shape=(1,))
        second = vd.storage.zeros(dtype=Particle, shape=(1,))
        _, pullback = aggregate_multi_output_objective_vjp(particles, first, second, grid=(1, 1, 1))

        first_cotangent = vd.storage.tangent_zeros(dtype=Particle, shape=(1,))
        first_cotangent["velocity"].copy_from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
        first_cotangent["mass"].copy_from_numpy(np.array([3.0], dtype=np.float32))
        second_cotangent = vd.storage.tangent_zeros(dtype=Particle, shape=(1,))
        second_cotangent["velocity"].copy_from_numpy(np.array([[4.0, 5.0]], dtype=np.float32))
        second_cotangent["mass"].copy_from_numpy(np.array([7.0], dtype=np.float32))

        gradient = pullback({"first": first_cotangent, "second": second_cotangent})["particles"]
        np.testing.assert_array_equal(
            gradient["velocity"].to_numpy(),
            np.array([[5.0, 7.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(gradient["mass"].to_numpy(), np.array([45.0], dtype=np.float32))

    def test_constant_aggregate_store_rejects_multi_invocation_grid(self) -> None:
        vd.init(arch=vd.cpu)
        particles = vd.storage.zeros(dtype=Particle, shape=(1,))
        values = particles.to_numpy()
        values["velocity"][0] = np.array([2.0, -3.0], dtype=np.float16)
        values["mass"][0] = np.float32(4.0)
        particles.copy_from_numpy(values)
        output = vd.storage.zeros(dtype=Particle, shape=(1,))
        with self.assertRaisesRegex(ValueError, "dispatch grid axis 0 must equal 1"):
            aggregate_output_objective_vjp(particles, output, grid=(2, 1, 1))

    def test_nested_input_and_output_route_every_tangent_leaf(self) -> None:
        vd.init(arch=vd.cpu)
        source = vd.storage.zeros(dtype=NestedRecord, shape=(2,))
        values = source.to_numpy()
        values["particle"]["position"][0] = np.array([2.0, -3.0], dtype=np.float32)
        values["particle"]["weight"][0] = np.float64(4.0)
        values["particle"]["tag"][0] = np.int32(17)
        values["pair"]["0"][0] = np.float32(5.0)
        values["pair"]["1"][0] = np.int32(19)
        values["samples"][0] = np.array([7.0, 11.0], dtype=np.float32)
        source.copy_from_numpy(values)
        output = vd.storage.zeros(dtype=NestedRecord, shape=(2,))

        _, pullback = nested_aggregate_objective_vjp(source, output, grid=(1, 1, 1))
        output_values = output.to_numpy()
        np.testing.assert_array_equal(
            output_values["particle"]["position"][0],
            np.array([4.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(output_values["particle"]["weight"], np.array([16.0, 0.0], dtype=np.float64))
        np.testing.assert_array_equal(output_values["pair"]["0"], np.array([10.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(output_values["samples"][0], np.array([-21.0, 16.0], dtype=np.float32))

        cotangent = vd.storage.tangent_zeros(dtype=NestedRecord, shape=(2,))
        cotangent["particle.position"].copy_from_numpy(np.array([[1.5, -2.0], [9.0, 9.0]], dtype=np.float32))
        cotangent["particle.weight"].copy_from_numpy(np.array([0.25, 9.0], dtype=np.float64))
        cotangent["pair.0"].copy_from_numpy(np.array([3.0, 9.0], dtype=np.float32))
        cotangent["samples"].copy_from_numpy(np.array([[4.0, -5.0], [9.0, 9.0]], dtype=np.float32))
        gradient = pullback(cotangent)["source"]

        # Stage-local aggregate carriers must not overwrite the canonical
        # logical Storage or any independently owned cotangent leaf.
        np.testing.assert_array_equal(source.to_numpy(), values)
        np.testing.assert_array_equal(
            cotangent["particle.position"].to_numpy(),
            np.array([[1.5, -2.0], [9.0, 9.0]], dtype=np.float32),
        )
        self.assertEqual(
            gradient.dtype.names,
            ("particle", "pair", "samples"),
        )
        np.testing.assert_array_equal(
            gradient["particle.position"].to_numpy(),
            np.array([[21.0, 26.0], [0.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            gradient["particle.weight"].to_numpy(),
            np.array([2.0, 0.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            gradient["pair.0"].to_numpy(),
            np.array([-1.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            gradient["samples"].to_numpy(),
            np.array([[-12.0, -5.0], [0.0, 0.0]], dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "non-differentiable"):
            gradient["particle.tag"]
        with self.assertRaisesRegex(ValueError, "non-differentiable"):
            gradient["pair.1"]

    def test_nested_inputs_sharing_one_storage_accumulate_into_one_tangent_owner(
        self,
    ) -> None:
        vd.init(arch=vd.cpu)
        source = vd.storage.zeros(dtype=NestedRecord, shape=(2,))
        values = source.to_numpy()
        values["particle"]["position"][0] = np.array([2.0, -3.0], dtype=np.float32)
        values["particle"]["weight"][0] = np.float64(4.0)
        values["particle"]["tag"][0] = np.int32(17)
        values["pair"]["0"][0] = np.float32(5.0)
        values["pair"]["1"][0] = np.int32(19)
        values["samples"][0] = np.array([7.0, 11.0], dtype=np.float32)
        source.copy_from_numpy(values)
        output = vd.storage.zeros(dtype=NestedRecord, shape=(2,))

        _, pullback = shared_nested_inputs_objective_vjp(
            source,
            source,
            output,
            grid=(1, 1, 1),
        )
        cotangent = vd.storage.tangent_zeros(dtype=NestedRecord, shape=(2,))
        cotangent["particle.position"].copy_from_numpy(np.array([[1.5, -2.0], [9.0, 9.0]], dtype=np.float32))
        cotangent["particle.weight"].copy_from_numpy(np.array([0.25, 9.0], dtype=np.float64))
        cotangent["pair.0"].copy_from_numpy(np.array([3.0, 9.0], dtype=np.float32))
        cotangent["samples"].copy_from_numpy(np.array([[4.0, -5.0], [9.0, 9.0]], dtype=np.float32))
        gradients = pullback(cotangent)

        self.assertIs(gradients["left"], gradients["right"])
        gradient = gradients["left"]
        np.testing.assert_array_equal(
            gradient["particle.position"].to_numpy(),
            np.array([[6.0, 12.0], [0.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            gradient["particle.weight"].to_numpy(),
            np.array([2.0, 0.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            gradient["pair.0"].to_numpy(),
            np.array([30.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            gradient["samples"].to_numpy(),
            np.array([[56.0, -110.0], [0.0, 0.0]], dtype=np.float32),
        )

    def test_declared_aggregate_field_path_returns_the_owner_tangent(self) -> None:
        vd.init(arch=vd.cpu)
        particles = vd.storage.zeros(dtype=Particle, shape=(1,))
        values = particles.to_numpy()
        values["velocity"][0] = np.array([2.0, -3.0], dtype=np.float16)
        values["mass"][0] = np.float32(4.0)
        particles.copy_from_numpy(values)
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))

        _, pullback = aggregate_field_objective_vjp(particles, loss, grid=(1, 1, 1))
        gradients = pullback(None)

        self.assertEqual(tuple(gradients), ("particles.velocity",))
        np.testing.assert_array_equal(
            gradients["particles.velocity"]["velocity"].to_numpy(),
            np.array([[4.0, -6.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            gradients["particles.velocity"]["mass"].to_numpy(),
            np.zeros((1,), dtype=np.float32),
        )

    def test_dynamic_storage_shapes_share_one_direct_vjp_compilation(self) -> None:
        vd.init(arch=vd.cpu)
        runtime_autodiff.clear_vjp_cache()
        for values_array, expected_gradient in (
            (
                np.array([3.0, 4.0], dtype=np.float32),
                np.array([6.0, 1.0], dtype=np.float32),
            ),
            (
                np.array([5.0, 6.0, 7.0], dtype=np.float32),
                np.array([10.0, 1.0, 0.0], dtype=np.float32),
            ),
        ):
            values = vd.storage.from_numpy(values_array)
            loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
            _, pullback = storage_objective_vjp(values, loss, grid=(1, 1, 1))
            cotangent = np.array([1.0], dtype=np.float32)
            cotangent_before = cotangent.copy()
            np.testing.assert_array_equal(pullback(cotangent)["values"].to_numpy(), expected_gradient)
            np.testing.assert_array_equal(cotangent, cotangent_before)
        self.assertEqual(
            len(runtime_autodiff._kernel_state(storage_objective_vjp.program, storage_objective_vjp).compiled),
            1,
        )

    def test_partially_dynamic_storage_shape_preserves_static_extents(self) -> None:
        vd.init(arch=vd.cpu)
        runtime_autodiff.clear_vjp_cache()
        for outer, inner in ((3, 5), (7, 9)):
            shape = (outer, 2, 4, inner)
            values_array = np.zeros(shape, dtype=np.float32)
            values_array[0, 1, 3, 0] = 3.0
            values_array[outer - 1, 0, 0, inner - 1] = 4.0
            values = vd.storage.from_numpy(values_array)
            loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))

            _, pullback = partially_dynamic_objective_vjp(
                values,
                np.int32(outer),
                np.int32(inner),
                loss,
                grid=(1, 1, 1),
            )
            expected = np.zeros(shape, dtype=np.float32)
            expected[0, 1, 3, 0] = 6.0
            expected[outer - 1, 0, 0, inner - 1] = 1.0
            np.testing.assert_array_equal(pullback(None)["values"].to_numpy(), expected)
        self.assertEqual(
            len(
                runtime_autodiff._kernel_state(
                    partially_dynamic_objective_vjp.program,
                    partially_dynamic_objective_vjp,
                ).compiled
            ),
            1,
        )

        invalid = vd.storage.zeros(dtype=vd.f32, shape=(3, 3, 4, 5))
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        with self.assertRaisesRegex(ValueError, "bound shape for Program value 0 conflicts"):
            partially_dynamic_objective_vjp(invalid, np.int32(3), np.int32(5), loss, grid=(1, 1, 1))

    def test_tensor_view_gradient_scatter_targets_its_owner_descriptor(self) -> None:
        vd.init(arch=vd.cpu)
        owner = vd.storage.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        values = owner.view(shape=(2,), strides=(2,), offset=1, access="read")
        loss_owner = vd.storage.zeros(dtype=vd.f32, shape=(3,))
        loss = loss_owner.view(shape=(1,), strides=(1,), offset=1, access="read_write")
        output, pullback = storage_objective_vjp(values, loss, grid=(1, 1, 1))
        self.assertIsNone(output)
        np.testing.assert_array_equal(loss_owner.to_numpy(), np.array([0.0, 8.0, 0.0], dtype=np.float32))
        del values
        del owner
        gradient = pullback(None)["values"]
        self.assertIsInstance(gradient, vd.TensorStorage)
        np.testing.assert_array_equal(
            gradient.to_numpy(),
            np.array([0.0, 4.0, 0.0, 1.0], dtype=np.float32),
        )

    def test_aliased_wrt_views_share_one_accumulated_gradient_owner(self) -> None:
        vd.init(arch=vd.cpu)
        owner = vd.storage.from_numpy(np.array([2.0, 3.0, 4.0], dtype=np.float32))
        left = owner.view(shape=(2,), strides=(1,), offset=0, access="read")
        right = owner.view(shape=(2,), strides=(1,), offset=1, access="read")
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        _, pullback = aliased_inputs_objective_vjp(left, right, loss, grid=(1, 1, 1))
        gradients = pullback(None)
        self.assertIs(gradients["left"], gradients["right"])
        np.testing.assert_array_equal(
            gradients["left"].to_numpy(),
            np.array([4.0, 0.0, 8.0], dtype=np.float32),
        )

    def test_overlapping_read_write_views_are_rejected_before_staging(self) -> None:
        vd.init(arch=vd.cpu)
        owner = vd.storage.from_numpy(np.array([1.0, 2.0, 0.0], dtype=np.float32))
        values = owner.view(shape=(2,), strides=(1,), offset=0, access="read")
        loss = owner.view(shape=(1,), strides=(1,), offset=1, access="read_write")
        with self.assertRaisesRegex(ValueError, "overlap"):
            storage_objective_vjp(values, loss, grid=(1, 1, 1))

    def test_cpu_invocation_reduction_is_deterministic(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        _, pullback = storage_objective_vjp(values, loss, grid=(4, 1, 1))
        expected = np.array([24.0, 4.0], dtype=np.float32)
        cotangent = np.ones((1, 1, 4, 1), dtype=np.float32)
        for _ in range(3):
            np.testing.assert_array_equal(pullback(cotangent)["values"].to_numpy(), expected)

    def test_direct_storage_objective_reverses_scratch_write(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        scratch = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
        _, pullback = scratch_objective_vjp(values, scratch, loss, grid=(1, 1, 1))
        np.testing.assert_array_equal(
            pullback(None)["values"].to_numpy(),
            np.array([24.0], dtype=np.float32),
        )

    def test_overwrite_clears_the_replaced_storage_adjoint(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        scratch = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
        _, pullback = overwrite_objective_vjp(values, scratch, loss, grid=(1, 1, 1))
        np.testing.assert_array_equal(
            pullback(None)["values"].to_numpy(),
            np.array([54.0], dtype=np.float32),
        )

    def test_descriptor_scatter_supports_vector_storage_elements(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        scratch = vd.storage.zeros(dtype=vd.Vector[vd.f32, 2], shape=(1, 1, 1))
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
        _, pullback = vector_scratch_objective_vjp(values, scratch, loss, grid=(1, 1, 1))
        np.testing.assert_array_equal(
            pullback(None)["values"].to_numpy(),
            np.array([78.0], dtype=np.float32),
        )

    def test_descriptor_scatter_reverses_loop_scratch_writes(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
        scratch = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1, 2))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
        _, pullback = loop_scratch_objective_vjp(values, scratch, output, grid=(1, 1, 1))
        np.testing.assert_array_equal(
            pullback(None)["values"].to_numpy(),
            np.array([24.0, 32.0], dtype=np.float32),
        )

    def test_storage_effect_merge_reverses_the_executed_branch(self) -> None:
        vd.init(arch=vd.cpu)
        for primal, expected in ((3.0, 24.0), (-3.0, -54.0)):
            values = vd.storage.from_numpy(np.array([primal], dtype=np.float32))
            scratch = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
            output = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
            _, pullback = branch_scratch_objective_vjp(values, scratch, output, grid=(1, 1, 1))
            np.testing.assert_array_equal(
                pullback(None)["values"].to_numpy(),
                np.array([expected], dtype=np.float32),
            )

    def test_dynamic_for_has_no_compile_time_trip_cap(self) -> None:
        vd.init(arch=vd.cpu)
        primal = np.float32(1.25)
        for count in (0, 1, 32, 1500):
            values = vd.storage.from_numpy(np.array([primal], dtype=np.float32))
            output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
            result, pullback = dynamic_for_objective_vjp(
                values,
                np.int32(count),
                output,
                grid=(1, 1, 1),
            )
            self.assertIsNone(result)
            scale = np.float32(1.0 + 0.25 * count)
            np.testing.assert_allclose(output.to_numpy(), np.array([(primal * scale) ** 2], dtype=np.float32))
            if count == 32:
                values.copy_from_numpy(np.array([100.0], dtype=np.float32))
            np.testing.assert_allclose(
                pullback(None)["values"].to_numpy(),
                np.array([2.0 * primal * scale * scale], dtype=np.float32),
                rtol=2.0e-5,
                atol=2.0e-5,
            )
            if count == 1:
                np.testing.assert_allclose(
                    pullback(np.array([2.0], dtype=np.float32))["values"].to_numpy(),
                    np.array([4.0 * primal * scale * scale], dtype=np.float32),
                    rtol=2.0e-5,
                    atol=2.0e-5,
                )

    def test_dynamic_loop_gather_accumulates_every_storage_element(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        _, pullback = dynamic_gather_objective_vjp(values, np.int32(4), output, grid=(1, 1, 1))
        np.testing.assert_array_equal(
            pullback(None)["values"].to_numpy(),
            np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32),
        )

    def test_dynamic_loop_scratch_versions_propagate_every_element(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        scratch = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1, 4))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
        _, pullback = dynamic_scratch_gather_objective_vjp(
            values,
            scratch,
            np.int32(4),
            output,
            grid=(1, 1, 1),
        )
        np.testing.assert_array_equal(
            pullback(None)["values"].to_numpy(),
            np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32),
        )

    def test_nested_dynamic_scratch_versions_propagate_every_element(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        scratch = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1, 3, 4))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
        _, pullback = nested_dynamic_scratch_objective_vjp(
            values,
            scratch,
            np.int32(4),
            np.int32(3),
            output,
            grid=(1, 1, 1),
        )
        np.testing.assert_array_equal(
            pullback(None)["values"].to_numpy(),
            np.array([6.0, 12.0, 18.0, 24.0], dtype=np.float32),
        )

    def test_nested_branch_accumulation_propagates_all_executed_interiors(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        _, pullback = nested_branch_accumulation_objective_vjp(
            values,
            np.int32(4),
            np.int32(4),
            output,
            grid=(1, 1, 1),
        )
        np.testing.assert_array_equal(
            pullback(None)["values"].to_numpy(),
            np.array([0.0, 8.0, 12.0, 0.0], dtype=np.float32),
        )

    def test_dynamic_while_replays_break_continue_and_storage_effects(self) -> None:
        vd.init(arch=vd.cpu)
        primal = np.float32(3.0)
        cases = (
            (0, 1, 8),
            (8, 9, 12),
            (8, 3, 6),
            (1500, 1497, 1499),
        )
        for limit, start, stop in cases:
            values = vd.storage.from_numpy(np.array([primal], dtype=np.float32))
            scratch = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
            output = vd.storage.zeros(dtype=vd.f32, shape=(1, 1, 1))
            _, pullback = dynamic_while_objective_vjp(
                values,
                scratch,
                np.int32(limit),
                np.int32(start),
                np.int32(stop),
                output,
                grid=(1, 1, 1),
            )
            active = max(min(limit, stop) - start + 1, 0)
            accumulated = np.float32(0.5 * active) * primal
            expected_gradient = np.float32(0.5 * active * active) * primal
            np.testing.assert_allclose(
                output.to_numpy(),
                np.array([[[accumulated * accumulated]]], dtype=np.float32),
            )
            for _ in range(2):
                np.testing.assert_allclose(
                    pullback(None)["values"].to_numpy(),
                    np.array([expected_gradient], dtype=np.float32),
                    rtol=2.0e-5,
                    atol=2.0e-5,
                )


if __name__ == "__main__":
    unittest.main()
