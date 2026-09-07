from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import vernon_dsl as vd
from vernon_dsl._program_assets.capture import capture_program
from vernon_dsl._runtime.session import RuntimeUnavailableError
from vernon_dsl.program_assets import cook_program_asset

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


@vd.kernel(workgroup_size=(2, 2, 2))
def program_texture_round_trip(
    image: vd.Texture["3d", vd.rgba32_float, vd.read_write],  # noqa: F722
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    coordinate = vd.Vector([vd.i32(gid[0]), vd.i32(gid[1]), vd.i32(gid[2])])
    value = vd.texture_load(image, coordinate)
    vd.texture_store(image, coordinate, value + vd.Vector([1.0, 2.0, 3.0, 4.0]))


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


@vd.struct(shared=True)
class ProgramValueRecord:
    weight: vd.f32
    offset: vd.Vector[vd.f32, 2]


@vd.struct
class GpuParticle:
    velocity: vd.Vector[vd.f32, 2]
    mass: vd.f32
    tag: vd.i32


@vd.kernel(workgroup_size=(1, 1, 1))
def gpu_particle_objective(
    particles: vd.TensorView[GpuParticle, (1,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    particle = particles[0]
    loss[0] = (
        particle.velocity.x * particle.velocity.x
        + particle.velocity.y * particle.velocity.y
        + particle.mass * particle.mass
    )


@vd.kernel(workgroup_size=(1, 1, 1))
def program_value_parameters(
    output: vd.TensorView[vd.f32, (1,), vd.write],
    scalar: vd.f32,
    vector: vd.Vector[vd.f32, 2],
    matrix: vd.Matrix[vd.f32, 2, 2],
    pair: vd.Tuple[vd.f32, vd.Vector[vd.f32, 2]],
    tensor: vd.Tensor[vd.f32, (2,)],
    record: ProgramValueRecord,
) -> None:
    output[0] = scalar + vector[1] + matrix[1, 0] + pair[0] + pair[1][0] + tensor[1] + record.weight + record.offset[0]  # pyright: ignore[reportIndexIssue]


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


class PersistentlyBoundIncrement(vd.Module):
    def forward(
        self,
        output: vd.TensorStorage,
        source: vd.TensorStorage,
        amount: vd.f32,
    ) -> vd.TensorStorage:
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


class ValueParameters(vd.Module):
    def forward(
        self,
        scalar: vd.f32,
        vector: vd.Vector[vd.f32, 2],
        matrix: vd.Matrix[vd.f32, 2, 2],
        pair: vd.Tuple[vd.f32, vd.Vector[vd.f32, 2]],
        tensor: vd.Tensor[vd.f32, (2,)],
        record: ProgramValueRecord,
    ) -> vd.TensorStorage:
        output = vd.zeros(dtype=vd.f32, shape=(1,))
        program_value_parameters(output, scalar, vector, matrix, pair, tensor, record, grid=(1, 1, 1))
        return output


class StridedSquareSum(vd.Module):
    def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
        output = vd.zeros(dtype=vd.f32, shape=(1,))
        program_strided_square_sum(source, output, grid=(1, 1, 1))
        return output


class TextureRoundTrip(vd.Module):
    def forward(
        self,
        image: vd.Texture["3d", vd.rgba32_float, vd.read_write],  # noqa: F722
        marker: vd.TensorStorage,
    ) -> vd.TensorStorage:
        output = vd.empty_like(marker)
        program_increment(output, marker, grid=(1, 1, 1))
        program_texture_round_trip(image, grid=(1, 1, 1))
        return output


class AggregateObjective(vd.Module):
    def forward(self, particles: vd.TensorStorage) -> vd.TensorStorage:
        loss = vd.zeros(dtype=vd.f32, shape=(1,))
        aggregate_storage_objective(particles, loss, grid=(1, 1, 1))
        return loss


class GpuAggregateObjective(vd.Module):
    def forward(self, particles: vd.TensorStorage) -> vd.TensorStorage:
        loss = vd.zeros(dtype=vd.f32, shape=(1,))
        gpu_particle_objective(particles, loss, grid=(1, 1, 1))
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

    def test_module_vjp_rejects_forward_only_texture_resources(self) -> None:
        expression = vd.ad.vjp(TextureRoundTrip(), wrt=("marker",), outputs=("output",))
        declaration = vd.program_asset(id="resources/vjp", program=expression)
        with self.assertRaisesRegex(ValueError, "Texture and Sampler are forward-only resources"):
            capture_program(declaration)

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
        self.assertEqual(len(module._program_cache), 1)

    def test_module_allows_concurrent_invocation_snapshots(self) -> None:
        module = ParameterizedIncrement()
        module(vd.storage.from_numpy(np.arange(4, dtype=np.float32)), np.float32(0.0))

        def invoke(index: int) -> np.ndarray:
            source = vd.storage.from_numpy(np.arange(4, dtype=np.float32) + index)
            return module(source, np.float32(index)).to_numpy()

        with ThreadPoolExecutor(max_workers=4) as executor:
            actual = tuple(executor.map(invoke, range(8)))

        for index, value in enumerate(actual):
            np.testing.assert_array_equal(value, np.arange(4, dtype=np.float32) + 2 * index)

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

    def test_module_binds_all_abi_stable_value_categories(self) -> None:
        transformed = ValueParameters()
        arguments = (
            vd.f32(1),
            vd.Vector([2.0, 3.0]),
            vd.Matrix([[4.0, 5.0], [6.0, 7.0]]),
            (vd.f32(8), vd.Vector([9.0, 10.0])),
            vd.Tensor([11.0, 12.0]),
            ProgramValueRecord(vd.f32(13), vd.Vector([14.0, 15.0])),  # type: ignore[call-arg]
        )

        first = transformed(*arguments)
        second = transformed(vd.f32(2), *arguments[1:])

        np.testing.assert_array_equal(first.to_numpy(), np.array([66.0], dtype=np.float32))
        np.testing.assert_array_equal(second.to_numpy(), np.array([67.0], dtype=np.float32))

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

    def test_gpu_module_binds_texture_as_forward_resource(self) -> None:
        image = vd.Texture.from_numpy(
            np.zeros((2, 2, 2, 4), dtype=np.float32),
            dimension="3d",
            format=vd.rgba32_float,
            usage=("storage", "transfer_source", "transfer_destination"),
        )
        marker = vd.storage.from_numpy(np.array([7.0], dtype=np.float32))

        result = TextureRoundTrip()(image, marker)

        np.testing.assert_array_equal(result.to_numpy(), np.array([8.0], dtype=np.float32))
        expected = np.zeros((2, 2, 2, 4), dtype=np.float32)
        expected[...] = [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_array_equal(image.to_numpy(), expected)

    def test_gpu_module_reuses_unchanged_value_and_storage_bindings(self) -> None:
        source = vd.storage.from_numpy(np.arange(4, dtype=np.float32))
        output = vd.storage.from_numpy(np.full(4, -1.0, dtype=np.float32))
        module = PersistentlyBoundIncrement()

        first = module(output, source, np.float32(2.0)).to_numpy().copy()
        specialization = next(iter(module._program_cache.values()))
        first_telemetry = dict(specialization.native_program.binding_telemetry)
        unchanged = module(output, source, np.float32(2.0)).to_numpy().copy()
        unchanged_telemetry = dict(specialization.native_program.binding_telemetry)
        changed = module(output, source, np.float32(5.0)).to_numpy().copy()
        changed_telemetry = dict(specialization.native_program.binding_telemetry)
        source.copy_from_numpy(np.arange(4, dtype=np.float32) + 10.0)
        dirty_storage = module(output, source, np.float32(5.0)).to_numpy().copy()
        dirty_telemetry = dict(specialization.native_program.binding_telemetry)

        np.testing.assert_array_equal(first, np.arange(4, dtype=np.float32) + 2.0)
        np.testing.assert_array_equal(unchanged, np.arange(4, dtype=np.float32) + 2.0)
        np.testing.assert_array_equal(changed, np.arange(4, dtype=np.float32) + 5.0)
        np.testing.assert_array_equal(dirty_storage, np.arange(4, dtype=np.float32) + 15.0)
        self.assertEqual(unchanged_telemetry["prepare_count"] - first_telemetry["prepare_count"], 0)
        self.assertEqual(unchanged_telemetry["upload_bytes"] - first_telemetry["upload_bytes"], 0)
        self.assertEqual(unchanged_telemetry["upload_ranges"] - first_telemetry["upload_ranges"], 0)
        self.assertEqual(changed_telemetry["prepare_count"] - unchanged_telemetry["prepare_count"], 1)
        self.assertEqual(changed_telemetry["upload_bytes"] - unchanged_telemetry["upload_bytes"], 4)
        self.assertEqual(changed_telemetry["upload_ranges"] - unchanged_telemetry["upload_ranges"], 1)
        self.assertEqual(dirty_telemetry["prepare_count"] - changed_telemetry["prepare_count"], 0)
        self.assertEqual(
            dirty_telemetry["upload_bytes"] - changed_telemetry["upload_bytes"],
            source.to_numpy().nbytes,
        )
        self.assertEqual(dirty_telemetry["upload_ranges"] - changed_telemetry["upload_ranges"], 1)

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
        particles = vd.storage.zeros(dtype=GpuParticle, shape=(1,))
        values = particles.to_numpy()
        values["velocity"][0] = np.array([2.0, -3.0], dtype=np.float32)
        values["mass"][0] = np.float32(4.0)
        values["tag"][0] = np.int32(7)
        particles.copy_from_numpy(values)
        output, pullback = vd.ad.vjp(
            GpuAggregateObjective(),
            wrt=("particles",),
            outputs=("output",),
        )(particles)

        np.testing.assert_array_equal(output.to_numpy(), np.array([29.0], dtype=np.float32))
        gradient = pullback({"output": np.ones((1,), dtype=np.float32)})["particles"]
        np.testing.assert_array_equal(
            gradient["velocity"].to_numpy(),
            np.array([[4.0, -6.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(gradient["mass"].to_numpy(), np.array([8.0], dtype=np.float32))


class CookedProgramGpuVjpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._artifacts = tempfile.TemporaryDirectory(prefix="vernon-cooked-vjp-")
        for architecture, target, options in (
            (vd.metal, "metal", {}),
            (vd.vulkan, "vulkan", {}),
            (vd.opengl, "opengl", {"api_version": (4, 3)}),
        ):
            try:
                vd.init(arch=architecture, **options)
            except RuntimeUnavailableError:
                continue
            cls._architecture = architecture
            try:
                cls._manifest = cook_program_asset(
                    program_asset=f"{Path(__file__).with_name('cooked_vjp_program_asset_fixture.py')}:square_vjp_asset",
                    output=Path(cls._artifacts.name) / "bundle",
                    target=target,
                )
            except Exception:
                vd.init(arch=vd.cpu)
                cls._artifacts.cleanup()
                raise
            return
        cls._artifacts.cleanup()
        raise unittest.SkipTest("no GPU compute runtime is available")

    @classmethod
    def tearDownClass(cls) -> None:
        vd.init(arch=vd.cpu)
        cls._artifacts.cleanup()

    def test_cooked_square_vjp_loads_and_reuses_pullback(self) -> None:
        program = vd.load_program(self._manifest)
        source = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))

        _, pullback = program.vjp({"source": source, "output": output}, (1, 1, 1))

        np.testing.assert_array_equal(output.to_numpy(), np.array([9.0], dtype=np.float32))
        first = pullback({"output": np.array([1.0], dtype=np.float32)})["source"]
        first_values = first.to_numpy().copy()
        second = pullback({"output": np.array([2.0], dtype=np.float32)})["source"]
        self.assertIsNot(first, second)
        np.testing.assert_array_equal(first_values, np.array([6.0], dtype=np.float32))
        np.testing.assert_array_equal(first.to_numpy(), np.array([6.0], dtype=np.float32))
        np.testing.assert_array_equal(second.to_numpy(), np.array([12.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
