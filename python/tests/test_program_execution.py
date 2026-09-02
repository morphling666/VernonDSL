from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import vernon_dsl as vd
from vernon_dsl._runtime.session import RuntimeUnavailableError

from python.tests.storage_vjp_direct_fixture import (
    Particle,
    aggregate_storage_objective,
    aggregate_storage_objective_vjp,
)


@vd.kernel(workgroup_size=(1, 1, 1))
def program_increment(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = source[gid[0]] + 1.0


@vd.kernel(workgroup_size=(1, 1, 1))
def program_add_parameter(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    amount: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = source[gid[0]] + amount


@vd.kernel(workgroup_size=(1, 1, 1))
def program_square(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0]


@vd.kernel(workgroup_size=(1, 1, 1))
def program_strided_square_sum(
    source: vd.TensorView[vd.f32, (2,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0] + source[1] * source[1]


@vd.kernel(workgroup_size=(1, 1, 1))
def program_cube(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0] * source[0]


@vd.kernel(workgroup_size=(1, 1, 1))
def program_product(
    left: vd.TensorView[vd.f32, (1,), vd.read],
    right: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = left[0] * right[0]


@vd.kernel(workgroup_size=(1, 1, 1))
def program_scale(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    factor: vd.f32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * factor


@dataclass
class BranchOutputs:
    square: vd.TensorStorage
    cube: vd.TensorStorage


@dataclass
class AggregateOutputs:
    first: vd.TensorStorage
    second: vd.TensorStorage


@dataclass
class UnrelatedOutputs:
    unrelated: vd.TensorStorage
    square: vd.TensorStorage


class IncrementChain(vd.Module):
    def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
        intermediate = vd.empty_like(source)
        output = vd.empty_like(source)
        program_increment(intermediate, source, grid=(4, 1, 1))
        program_increment(output, intermediate, grid=(4, 1, 1))
        return output


class ParameterizedIncrement(vd.Module):
    def forward(self, source: vd.TensorStorage, amount: vd.f32) -> vd.TensorStorage:
        output = vd.empty_like(source)
        program_add_parameter(output, source, amount, grid=(4, 1, 1))
        return output


class Square(vd.Module):
    def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
        output = vd.empty_like(source)
        program_square(source, output, grid=(1, 1, 1))
        return output


class SquareChain(vd.Module):
    def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
        intermediate = vd.empty_like(source)
        output = vd.empty_like(source)
        program_square(source, intermediate, grid=(1, 1, 1))
        program_square(intermediate, output, grid=(1, 1, 1))
        return output


class FanOut(vd.Module):
    def forward(self, source: vd.TensorStorage) -> BranchOutputs:
        square = vd.empty_like(source)
        cube = vd.empty_like(source)
        program_square(source, square, grid=(1, 1, 1))
        program_cube(source, cube, grid=(1, 1, 1))
        return BranchOutputs(square, cube)


class Product(vd.Module):
    def forward(self, left: vd.TensorStorage, right: vd.TensorStorage) -> vd.TensorStorage:
        output = vd.empty_like(left)
        program_product(left, right, output, grid=(1, 1, 1))
        return output


class AliasedProduct(vd.Module):
    def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
        output = vd.empty_like(source)
        program_product(source, source, output, grid=(1, 1, 1))
        return output


class Scale(vd.Module):
    def forward(self, source: vd.TensorStorage, factor: vd.f32) -> vd.TensorStorage:
        output = vd.empty_like(source)
        program_scale(source, factor, output, grid=(1, 1, 1))
        return output


class StridedSquareSum(vd.Module):
    def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
        output = vd.zeros(dtype=vd.f32, shape=(1,))
        program_strided_square_sum(source, output, grid=(1, 1, 1))
        return output


class AggregateObjective(vd.Module):
    def forward(self, particles: vd.TensorStorage) -> vd.TensorStorage:
        loss = vd.zeros(dtype=vd.f32, shape=(1,))
        aggregate_storage_objective(particles, loss, grid=(1, 1, 1))
        return loss


class AggregateFanOut(vd.Module):
    def forward(self, particles: vd.TensorStorage) -> AggregateOutputs:
        first = vd.zeros(dtype=vd.f32, shape=(1,))
        second = vd.zeros(dtype=vd.f32, shape=(1,))
        aggregate_storage_objective(particles, first, grid=(1, 1, 1))
        aggregate_storage_objective(particles, second, grid=(1, 1, 1))
        return AggregateOutputs(first, second)


class UnrelatedAndSquare(vd.Module):
    def forward(self, unrelated: vd.TensorStorage, source: vd.TensorStorage) -> UnrelatedOutputs:
        unrelated_output = vd.empty_like(unrelated)
        square_output = vd.empty_like(source)
        program_increment(unrelated_output, unrelated, grid=(1, 1, 1))
        program_square(source, square_output, grid=(1, 1, 1))
        return UnrelatedOutputs(unrelated_output, square_output)


def particle_storage() -> vd.TensorStorage:
    particles = vd.storage.zeros(dtype=Particle, shape=(1,))
    values = particles.to_numpy()
    values["velocity"][0] = np.array([2.0, -3.0], dtype=np.float16)
    values["mass"][0] = np.float32(4.0)
    particles.copy_from_numpy(values)
    return particles


def assert_particle_gradient(test: unittest.TestCase, gradient: object) -> None:
    assert isinstance(gradient, vd.TensorStorage)
    np.testing.assert_array_equal(
        gradient["velocity"].to_numpy(),
        np.array([[4.0, -6.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        gradient["mass"].to_numpy(),
        np.array([8.0], dtype=np.float32),
    )


class ProgramExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        vd.init(arch=vd.cpu)

    def test_module_orders_dependent_kernels(self) -> None:
        source = vd.storage.from_numpy(np.arange(4, dtype=np.float32))
        module = IncrementChain()

        first = module(source)
        second = module(vd.storage.from_numpy(np.arange(4, dtype=np.float32) + 10.0))

        np.testing.assert_array_equal(first.to_numpy(), np.arange(4, dtype=np.float32) + 2.0)
        np.testing.assert_array_equal(second.to_numpy(), np.arange(4, dtype=np.float32) + 12.0)
        self.assertEqual(len(module._program_cache), 1)

    def test_module_scalar_argument_is_bound_per_invocation(self) -> None:
        source = vd.storage.from_numpy(np.arange(4, dtype=np.float32))
        module = ParameterizedIncrement()

        first = module(source, np.float32(2.0))
        second = module(source, np.float32(5.0))

        np.testing.assert_array_equal(first.to_numpy(), np.arange(4, dtype=np.float32) + 2.0)
        np.testing.assert_array_equal(second.to_numpy(), np.arange(4, dtype=np.float32) + 5.0)

    def test_single_kernel_vjp_supports_direct_and_module_surfaces(self) -> None:
        source = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        direct_output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        direct = vd.ad.vjp(program_square, wrt=("source",), outputs=("output",))

        output, direct_pullback = direct(source, direct_output, grid=(1, 1, 1))
        module_output, module_pullback = vd.ad.vjp(
            Square(),
            wrt=("source",),
            outputs=("output",),
        )(source)

        self.assertIsNone(output)
        np.testing.assert_array_equal(direct_output.to_numpy(), np.array([9.0], dtype=np.float32))
        np.testing.assert_array_equal(module_output.to_numpy(), np.array([9.0], dtype=np.float32))
        np.testing.assert_array_equal(
            direct_pullback(None)["source"].to_numpy(),
            np.array([6.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            module_pullback({"output": np.ones((1,), dtype=np.float32)})["source"].to_numpy(),
            np.array([6.0], dtype=np.float32),
        )

    def test_module_vjp_composes_multi_kernel_chain(self) -> None:
        source = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        output, pullback = vd.ad.vjp(
            SquareChain(),
            wrt=("source",),
            outputs=("output",),
        )(source)

        np.testing.assert_array_equal(output.to_numpy(), np.array([81.0], dtype=np.float32))
        gradient = pullback({"output": np.ones((1,), dtype=np.float32)})["source"]
        np.testing.assert_allclose(gradient.to_numpy(), np.array([108.0], dtype=np.float32), rtol=1e-5)

    def test_module_pullbacks_retain_independent_forward_state(self) -> None:
        transformed = vd.ad.vjp(SquareChain(), wrt=("source",), outputs=("output",))
        _, first = transformed(vd.storage.from_numpy(np.array([2.0], dtype=np.float32)))
        _, second = transformed(vd.storage.from_numpy(np.array([3.0], dtype=np.float32)))

        np.testing.assert_array_equal(
            first({"output": np.ones((1,), dtype=np.float32)})["source"].to_numpy(),
            np.array([32.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            second({"output": np.ones((1,), dtype=np.float32)})["source"].to_numpy(),
            np.array([108.0], dtype=np.float32),
        )

    def test_module_vjp_accumulates_branch_fan_in(self) -> None:
        source = vd.storage.from_numpy(np.array([2.0], dtype=np.float32))
        outputs, pullback = vd.ad.vjp(
            FanOut(),
            wrt=("source",),
            outputs=("square", "cube"),
        )(source)

        np.testing.assert_array_equal(outputs.square.to_numpy(), np.array([4.0], dtype=np.float32))
        np.testing.assert_array_equal(outputs.cube.to_numpy(), np.array([8.0], dtype=np.float32))
        gradient = pullback(
            {
                "square": np.ones((1,), dtype=np.float32),
                "cube": np.full((1,), 2.0, dtype=np.float32),
            }
        )["source"]
        np.testing.assert_array_equal(gradient.to_numpy(), np.array([28.0], dtype=np.float32))

    def test_module_vjp_returns_multiple_input_gradients(self) -> None:
        left = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        right = vd.storage.from_numpy(np.array([5.0], dtype=np.float32))
        output, pullback = vd.ad.vjp(
            Product(),
            wrt=("left", "right"),
            outputs=("output",),
        )(left, right)

        np.testing.assert_array_equal(output.to_numpy(), np.array([15.0], dtype=np.float32))
        gradients = pullback({"output": np.ones((1,), dtype=np.float32)})
        np.testing.assert_array_equal(gradients["left"].to_numpy(), np.array([5.0], dtype=np.float32))
        np.testing.assert_array_equal(gradients["right"].to_numpy(), np.array([3.0], dtype=np.float32))

    def test_module_vjp_accumulates_aliased_operands_once_per_use(self) -> None:
        output, pullback = vd.ad.vjp(
            AliasedProduct(),
            wrt=("source",),
            outputs=("output",),
        )(vd.storage.from_numpy(np.array([3.0], dtype=np.float32)))

        np.testing.assert_array_equal(output.to_numpy(), np.array([9.0], dtype=np.float32))
        gradient = pullback({"output": np.ones((1,), dtype=np.float32)})["source"]
        np.testing.assert_array_equal(gradient.to_numpy(), np.array([6.0], dtype=np.float32))

    def test_module_vjp_differentiates_scalar_argument(self) -> None:
        source = vd.storage.from_numpy(np.array([4.0], dtype=np.float32))
        output, pullback = vd.ad.vjp(
            Scale(),
            wrt=("factor",),
            outputs=("output",),
        )(source, np.float32(3.0))

        np.testing.assert_array_equal(output.to_numpy(), np.array([12.0], dtype=np.float32))
        gradient = pullback({"output": np.ones((1,), dtype=np.float32)})["factor"]
        self.assertEqual(float(np.asarray(gradient)), 4.0)

    def test_aggregate_vjp_supports_direct_and_module_root_paths(self) -> None:
        direct_particles = particle_storage()
        direct_loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        _, direct_pullback = aggregate_storage_objective_vjp(
            direct_particles,
            direct_loss,
            grid=(1, 1, 1),
        )
        module_output, module_pullback = vd.ad.vjp(
            AggregateObjective(),
            wrt=("particles",),
            outputs=("output",),
        )(particle_storage())

        np.testing.assert_array_equal(direct_loss.to_numpy(), np.array([29.0], dtype=np.float32))
        np.testing.assert_array_equal(module_output.to_numpy(), np.array([29.0], dtype=np.float32))
        assert_particle_gradient(self, direct_pullback(None)["particles"])
        assert_particle_gradient(
            self,
            module_pullback({"output": np.ones((1,), dtype=np.float32)})["particles"],
        )

    def test_module_aggregate_vjp_supports_leaf_and_root_selection(self) -> None:
        _, leaf_pullback = vd.ad.vjp(
            AggregateObjective(),
            wrt=("particles.velocity", "particles.mass"),
            outputs=("output",),
        )(particle_storage())
        _, root_pullback = vd.ad.vjp(
            AggregateObjective(),
            wrt=("particles",),
            outputs=("output",),
        )(particle_storage())
        cotangent = {"output": np.ones((1,), dtype=np.float32)}

        leaf_gradient = leaf_pullback(cotangent)["particles"]
        root_gradient = root_pullback(cotangent)["particles"]
        assert_particle_gradient(self, leaf_gradient)
        assert_particle_gradient(self, root_gradient)

    def test_module_aggregate_vjp_accumulates_branch_fan_in(self) -> None:
        outputs, pullback = vd.ad.vjp(
            AggregateFanOut(),
            wrt=("particles",),
            outputs=("first", "second"),
        )(particle_storage())

        np.testing.assert_array_equal(outputs.first.to_numpy(), np.array([29.0], dtype=np.float32))
        np.testing.assert_array_equal(outputs.second.to_numpy(), np.array([29.0], dtype=np.float32))
        gradient = pullback(
            {
                "first": np.ones((1,), dtype=np.float32),
                "second": np.ones((1,), dtype=np.float32),
            }
        )["particles"]
        np.testing.assert_array_equal(
            gradient["velocity"].to_numpy(),
            np.array([[8.0, -12.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            gradient["mass"].to_numpy(),
            np.array([16.0], dtype=np.float32),
        )

    def test_module_vjp_ignores_unrelated_non_differentiable_branch(self) -> None:
        unrelated = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        source = vd.storage.from_numpy(np.array([2.0], dtype=np.float32))
        outputs, pullback = vd.ad.vjp(
            UnrelatedAndSquare(),
            wrt=("source",),
            outputs=("square",),
        )(unrelated, source)

        np.testing.assert_array_equal(outputs.unrelated.to_numpy(), np.array([4.0], dtype=np.float32))
        np.testing.assert_array_equal(outputs.square.to_numpy(), np.array([4.0], dtype=np.float32))
        gradient = pullback({"square": np.ones((1,), dtype=np.float32)})["source"]
        np.testing.assert_array_equal(gradient.to_numpy(), np.array([4.0], dtype=np.float32))

    def test_texture_view_validation_remains_on_direct_resource_surface(self) -> None:
        texture = vd.Texture.zeros(shape=(4, 4))
        with self.assertRaisesRegex(ValueError, "format is incompatible"):
            texture.view(format=vd.r16_float)
        with self.assertRaisesRegex(ValueError, "dimension is incompatible"):
            texture.view(dimension="3d")
        with self.assertRaisesRegex(ValueError, "aspects are incompatible"):
            texture.view(aspects=("depth",))

        cube = vd.Texture.cube(np.zeros((6, 4, 4, 4), dtype=np.uint8))
        face = cube.view(dimension="2d", base_array_layer=2, array_layer_count=1)
        self.assertEqual(face.dimension, "2d")
        with self.assertRaisesRegex(ValueError, "exactly one array layer"):
            cube.view(dimension="2d")


class ProgramGpuExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        for architecture in (vd.metal, vd.vulkan, vd.opengl):
            try:
                if architecture == vd.opengl:
                    vd.init(arch=architecture, api_version=(4, 3))
                else:
                    vd.init(arch=architecture)
                return
            except RuntimeUnavailableError:
                pass
        raise unittest.SkipTest("no GPU compute runtime is available")

    @classmethod
    def tearDownClass(cls) -> None:
        vd.init(arch=vd.cpu)

    def test_gpu_module_vjp_retains_non_contiguous_view_layout(self) -> None:
        owner = vd.storage.from_numpy(np.array([10.0, 2.0, 20.0, 3.0], dtype=np.float32))
        source = owner.view(shape=(2,), strides=(-2,), offset=3, access="read")
        output, pullback = vd.ad.vjp(
            StridedSquareSum(),
            wrt=("source",),
            outputs=("output",),
        )(source)

        np.testing.assert_array_equal(output.to_numpy(), np.array([13.0], dtype=np.float32))
        gradient = pullback({"output": np.ones((1,), dtype=np.float32)})["source"]
        np.testing.assert_array_equal(
            gradient.to_numpy(),
            np.array([0.0, 4.0, 0.0, 6.0], dtype=np.float32),
        )

    def test_gpu_module_vjp_accumulates_branch_fan_in(self) -> None:
        outputs, pullback = vd.ad.vjp(
            FanOut(),
            wrt=("source",),
            outputs=("square", "cube"),
        )(vd.storage.from_numpy(np.array([2.0], dtype=np.float32)))

        np.testing.assert_array_equal(outputs.square.to_numpy(), np.array([4.0], dtype=np.float32))
        np.testing.assert_array_equal(outputs.cube.to_numpy(), np.array([8.0], dtype=np.float32))
        gradient = pullback(
            {
                "square": np.ones((1,), dtype=np.float32),
                "cube": np.full((1,), 2.0, dtype=np.float32),
            }
        )["source"]
        np.testing.assert_array_equal(gradient.to_numpy(), np.array([28.0], dtype=np.float32))

    def test_gpu_module_vjp_preserves_structured_storage_gradients(self) -> None:
        output, pullback = vd.ad.vjp(
            AggregateObjective(),
            wrt=("particles",),
            outputs=("output",),
        )(particle_storage())

        np.testing.assert_array_equal(output.to_numpy(), np.array([29.0], dtype=np.float32))
        gradient = pullback({"output": np.ones((1,), dtype=np.float32)})["particles"]
        assert_particle_gradient(self, gradient)


if __name__ == "__main__":
    unittest.main()
