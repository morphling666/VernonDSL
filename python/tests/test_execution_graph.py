from __future__ import annotations

import unittest
from typing import Annotated, Any, cast
from unittest import mock

import numpy as np
import vernon_dsl as vd
from vernon_dsl._runtime.autodiff import _StructuredPullback
from vernon_dsl._runtime.session import RuntimeUnavailableError

from python.tests.storage_vjp_direct_fixture import Particle, aggregate_storage_objective_vjp


@vd.kernel(workgroup_size=(1, 1, 1))
def graph_increment(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = source[gid[0]] + 1.0


@vd.kernel(workgroup_size=(1, 1, 1))
def graph_add_parameter(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    amount: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = source[gid[0]] + amount


@vd.kernel(workgroup_size=(1, 1, 1))
def graph_square(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0]


@vd.kernel(workgroup_size=(1, 1, 1))
def graph_strided_square_sum(
    source: vd.TensorView[vd.f32, (2,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0] + source[1] * source[1]


@vd.struct
class GraphParticle:
    velocity: vd.Vector[vd.f32, 2]
    mass: vd.f32
    tag: vd.i32


@vd.kernel
def graph_aggregate_storage_objective(
    particles: vd.TensorView[GraphParticle, (1,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    particle = particles[0]
    loss[0] = (
        particle.velocity.x * particle.velocity.x
        + particle.velocity.y * particle.velocity.y
        + particle.mass * particle.mass
    )


@vd.kernel(workgroup_size=(1, 1, 1))
def graph_heavy_square(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    value = source[0]
    value = value * value
    value = value * value
    value = value * value
    value = value * value
    value = value * value
    value = value * value
    value = value * value
    output[0] = value * value


@vd.kernel(workgroup_size=(1, 1, 1))
def graph_checkpoint_product(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.read_write],
) -> None:
    output[0] = source[0] * output[0]


@vd.kernel(workgroup_size=(1, 1, 1))
def graph_cube(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0] * source[0]


@vd.kernel(workgroup_size=(1, 1, 1))
def graph_product(
    left: vd.TensorView[vd.f32, (1,), vd.read],
    right: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = left[0] * right[0]


@vd.kernel(workgroup_size=(1, 1, 1))
def graph_scale(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    factor: vd.f32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * factor


graph_square_vjp = vd.ad.vjp(graph_square, wrt=("source",), outputs=("output",))
graph_strided_square_sum_vjp = vd.ad.vjp(graph_strided_square_sum, wrt=("source",), outputs=("output",))
graph_aggregate_storage_objective_vjp = vd.ad.vjp(
    graph_aggregate_storage_objective,
    wrt=("particles",),
    outputs=("loss",),
)
graph_heavy_square_vjp = vd.ad.vjp(graph_heavy_square, wrt=("source",), outputs=("output",))
graph_checkpoint_product_vjp = vd.ad.vjp(graph_checkpoint_product, wrt=("source",), outputs=("output",))
graph_cube_vjp = vd.ad.vjp(graph_cube, wrt=("source",), outputs=("output",))
graph_product_vjp = vd.ad.vjp(graph_product, wrt=("left", "right"), outputs=("output",))
graph_scale_vjp = vd.ad.vjp(graph_scale, wrt=("factor",), outputs=("output",))


class RecordingComputePass(vd.ComputePass):
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        read: object | None = None,
        write: object | None = None,
    ):
        super().__init__(name)
        self.events = events
        self.read_resource = read
        self.write_resource = write

    def declare(self) -> None:
        if self.read_resource is not None:
            self.read(self.read_resource)
        if self.write_resource is not None:
            self.write(self.write_resource)

    def execute(self, encoder: vd.ComputeEncoder, resources: vd.ExecutionResources) -> None:
        self.events.append(self.name)


class InvocationComputePass(vd.ComputePass):
    def __init__(self, name: str, invocation: vd.PipelineInvocation):
        super().__init__(name)
        self.invocation = invocation

    def declare(self) -> None:
        self.invocation.declare(self)

    def execute(self, encoder: vd.ComputeEncoder, resources: vd.ExecutionResources) -> None:
        self.invocation.encode(encoder, resources)


class FailingComputePass(vd.ComputePass):
    def __init__(self, events: list[str]):
        super().__init__("failure")
        self.events = events
        self.side_effect = True

    def declare(self) -> None:
        pass

    def execute(self, encoder: vd.ComputeEncoder, resources: vd.ExecutionResources) -> None:
        self.events.append(self.name)
        raise RuntimeError("intentional graph failure")


class ParameterComputePass(vd.ComputePass):
    def __init__(self, name: str, parameter: vd.ExecutionParameter, values: list[object]):
        super().__init__(name)
        self.parameter = parameter
        self.values = values
        self.side_effect = True

    def declare(self) -> None:
        pass

    def execute(self, encoder: vd.ComputeEncoder, resources: vd.ExecutionResources) -> None:
        self.values.append(resources.resolve(self.parameter))


class CpuExecutionGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        vd.init(arch=vd.cpu)

    def test_cpu_uses_native_hazard_schedule(self) -> None:
        graph = vd.ExecutionGraph()
        events: list[str] = []
        shared = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        graph.add_pass(RecordingComputePass("write", events, write=shared))
        graph.add_pass(RecordingComputePass("read", events, read=shared))
        graph._passes[-1].side_effect = True

        plan = graph.compile()
        plan.submit().wait()

        self.assertEqual([execution_pass.name for execution_pass in plan.schedule], ["write", "read"])
        self.assertEqual(events, ["write", "read"])
        self.assertEqual(len(plan.scopes[1].barriers), 1)

    def test_autodiff_checkpoint_planning_validates_memory_budget(self) -> None:
        graph = vd.ExecutionGraph()
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            graph.plan_autodiff_checkpoints(memory_budget=-1)

    def test_autodiff_budget_accepts_no_tape_plan_and_charges_gradients(self) -> None:
        def make_graph() -> vd.ExecutionGraph:
            source = vd.storage.from_numpy(np.array([2.0], dtype=np.float32))
            intermediate = vd.storage.zeros(dtype=vd.f32, shape=(1,))
            objective = vd.storage.zeros(dtype=vd.f32, shape=(1,))
            graph = vd.ExecutionGraph()
            previous = graph.differentiable_input("source", source)
            for name, storage in (("first", intermediate), ("second", objective)):
                output = (
                    graph.objective("loss", storage)
                    if name == "second"
                    else graph.import_resource(storage, exported=False)
                )
                graph.add_pass(
                    vd.VjpComputePass(
                        name,
                        graph_square_vjp,
                        {"source": previous, "output": output},
                        grid=(1, 1, 1),
                    )
                )
                previous = output
            return graph

        pullback = make_graph().compile().vjp()
        retained_primal_bytes = 2 * (np.dtype(np.float32).itemsize + np.dtype(np.uint64).itemsize)
        transaction_bytes = 2 * np.dtype(np.float32).itemsize
        backward_value_bytes = 3 * 2 * np.dtype(np.float32).itemsize
        minimum_budget = retained_primal_bytes + backward_value_bytes
        self.assertEqual(pullback.logical_residual_bytes, 0)
        self.assertEqual(pullback.resident_tape_bytes, 0)
        self.assertEqual(pullback.allocated_tape_bytes, 0)
        self.assertEqual(pullback.retained_allocation_bytes, retained_primal_bytes)
        self.assertGreaterEqual(pullback.peak_runtime_managed_bytes, retained_primal_bytes + transaction_bytes)

        graph = make_graph()
        graph.plan_autodiff_checkpoints(memory_budget=minimum_budget)
        compiled = graph.compile()
        checkpoint_plan = compiled.autodiff_checkpoint_plan
        assert checkpoint_plan is not None
        self.assertEqual(checkpoint_plan["logical_residual_bytes"], 0)
        self.assertEqual(checkpoint_plan["retained_allocation_bytes"], retained_primal_bytes)
        self.assertEqual(checkpoint_plan["transaction_bytes"], transaction_bytes)
        self.assertEqual(checkpoint_plan["restoration_bytes"], 0)
        self.assertEqual(checkpoint_plan["backward_value_bytes"], backward_value_bytes)
        self.assertEqual(
            [(item["version"], item["source"]) for item in checkpoint_plan["required_versions"]],
            [(0, "retained_owner"), (1, "retained_owner")],
        )

        graph = make_graph()
        graph.plan_autodiff_checkpoints(memory_budget=minimum_budget - 1)
        with self.assertRaisesRegex(ValueError, "cannot satisfy the memory budget"):
            graph.compile()

    def test_compile_rejects_foreign_dependencies_and_freezes_compiled_passes(self) -> None:
        first_graph = vd.ExecutionGraph()
        first_pass = first_graph.add_pass(RecordingComputePass("first", []))
        first_pass.side_effect = True
        second_graph = vd.ExecutionGraph()
        second_pass = second_graph.add_pass(RecordingComputePass("second", []))
        second_pass.depends_on(first_pass)

        with self.assertRaisesRegex(ValueError, "outside this graph"):
            second_graph.compile()

        plan = first_graph.compile()
        self.assertEqual([execution_pass.name for execution_pass in plan.schedule], ["first"])
        with self.assertRaisesRegex(RuntimeError, "after its execution graph has been compiled"):
            first_pass.side_effect = True
        with self.assertRaisesRegex(RuntimeError, "after its execution graph has been compiled"):
            first_pass.depends_on(RecordingComputePass("late", []))

    def test_cpu_kernel_invocations_execute_through_graph(self) -> None:
        source = vd.storage.from_numpy(np.arange(4, dtype=np.float32))
        intermediate = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        output = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        graph = vd.ExecutionGraph()
        graph.add_pass(InvocationComputePass("first", cast(Any, graph_increment).invocation(intermediate, source)))
        graph.add_pass(InvocationComputePass("second", cast(Any, graph_increment).invocation(output, intermediate)))

        plan = graph.compile()
        plan.submit().wait()

        np.testing.assert_array_equal(output.to_numpy(), np.arange(4, dtype=np.float32) + 2.0)
        self.assertEqual([execution_pass.name for execution_pass in plan.schedule], ["first", "second"])

    def test_parameterized_kernel_invocation_resolves_each_submission_snapshot(self) -> None:
        source = vd.storage.from_numpy(np.arange(4, dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        graph = vd.ExecutionGraph()
        amount = graph.parameter("amount")
        invocation = cast(Any, graph_add_parameter).invocation(output, source, amount)
        graph.add_pass(
            InvocationComputePass(
                "parameterized",
                invocation,
            )
        )
        plan = graph.compile()
        bindings = plan.create_bindings({amount: np.float32(2.0)})

        plan.submit(bindings).wait()
        np.testing.assert_array_equal(output.to_numpy(), np.arange(4, dtype=np.float32) + 2.0)
        plan.submit(bindings).wait()
        np.testing.assert_array_equal(output.to_numpy(), np.arange(4, dtype=np.float32) + 2.0)
        bindings.update({amount: np.float32(5.0)})
        plan.submit(bindings).wait()
        np.testing.assert_array_equal(output.to_numpy(), np.arange(4, dtype=np.float32) + 5.0)

    def test_parameter_bindings_validate_ownership_and_preserve_unchanged_values(self) -> None:
        graph = vd.ExecutionGraph()
        dynamic = graph.parameter("dynamic")
        constant = graph.parameter("constant")
        with self.assertRaisesRegex(ValueError, "already defined"):
            graph.parameter("dynamic")
        values: list[object] = []
        constant_values: list[object] = []
        graph.add_pass(ParameterComputePass("dynamic", dynamic, values))
        graph.add_pass(ParameterComputePass("constant", constant, constant_values))
        plan = graph.compile()

        with self.assertRaisesRegex(ValueError, "every parameter"):
            plan.create_bindings({dynamic: np.array([1.0], dtype=np.float32)})
        foreign_graph = vd.ExecutionGraph()
        foreign = foreign_graph.parameter("foreign")
        with self.assertRaisesRegex(ValueError, "does not belong"):
            plan.create_bindings({dynamic: 1, foreign: 2})

        constant_value = np.array([7.0], dtype=np.float32)
        bindings = plan.create_bindings(
            {
                dynamic: np.array([1.0], dtype=np.float32),
                constant: constant_value,
            }
        )
        plan.submit(bindings).wait()
        first = values[-1]
        bindings.update({dynamic: np.array([2.0], dtype=np.float32)})
        plan.submit(bindings).wait()

        self.assertIsNot(values[0], values[1])
        self.assertIs(constant_values[0], constant_values[1])
        np.testing.assert_array_equal(cast(Any, first), np.array([1.0], dtype=np.float32))
        constant_value[0] = 99.0
        bindings.update({dynamic: np.array([3.0], dtype=np.float32)})
        plan.submit(bindings).wait()
        np.testing.assert_array_equal(cast(Any, values[-1]), np.array([3.0], dtype=np.float32))
        np.testing.assert_array_equal(cast(Any, constant_values[-1]), np.array([7.0], dtype=np.float32))

    def test_compiled_plan_and_bindings_reject_runtime_reinitialization(self) -> None:
        graph = vd.ExecutionGraph()
        parameter = graph.parameter("value")
        graph.add_pass(ParameterComputePass("parameter", parameter, []))
        plan = graph.compile()
        bindings = plan.create_bindings({parameter: np.float32(1.0)})
        submission = plan.submit(bindings)

        vd.init(arch=vd.cpu)

        submission.wait()
        with self.assertRaisesRegex(RuntimeError, "runtime generation"):
            plan.submit(bindings)
        with self.assertRaisesRegex(RuntimeError, "runtime generation"):
            plan.create_bindings({parameter: np.float32(2.0)})
        with self.assertRaisesRegex(RuntimeError, "runtime generation"):
            bindings.update({parameter: np.float32(2.0)})

    def test_cpu_callback_failure_is_propagated_and_stops_execution(self) -> None:
        events: list[str] = []
        graph = vd.ExecutionGraph()
        graph.add_pass(FailingComputePass(events))
        after = RecordingComputePass("after", events)
        after.side_effect = True
        graph.add_pass(after)

        plan = graph.compile()
        with self.assertRaisesRegex(RuntimeError, "intentional graph failure"):
            plan.submit()

        self.assertEqual(events, ["failure"])

    def test_vjp_compute_pass_ordinary_submission_uses_primal_pipeline(self) -> None:
        source = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        graph = vd.ExecutionGraph()
        execution_pass = vd.VjpComputePass(
            "square",
            graph_square_vjp,
            {"source": graph.import_resource(source), "output": graph.import_resource(output)},
            grid=(1, 1, 1),
        )
        graph.add_pass(execution_pass)

        with mock.patch.object(
            execution_pass,
            "_native_vjp_forward",
            side_effect=AssertionError("ordinary submission allocated an autodiff tape"),
        ) as forward:
            graph.compile().submit().wait()

        forward.assert_not_called()
        np.testing.assert_array_equal(output.to_numpy(), np.array([9.0], dtype=np.float32))

    def test_graph_vjp_composes_multi_pass_chain_and_retains_destroyed_plan(self) -> None:
        source = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        intermediate = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        graph = vd.ExecutionGraph()
        source_resource = graph.differentiable_input("source", source)
        intermediate_resource = graph.import_resource(intermediate, exported=False)
        loss_resource = graph.objective("loss", loss)
        graph.add_pass(
            vd.VjpComputePass(
                "square-source",
                graph_square_vjp,
                {"source": source_resource, "output": intermediate_resource},
                grid=(1, 1, 1),
            )
        )
        graph.add_pass(
            vd.VjpComputePass(
                "square-intermediate",
                graph_square_vjp,
                {"source": intermediate_resource, "output": loss_resource},
                grid=(1, 1, 1),
            )
        )

        plan = graph.compile()
        pullback = plan.vjp()
        del graph
        del plan

        self.assertEqual(float(loss.to_numpy()[0]), 81.0)
        with mock.patch.object(
            _StructuredPullback,
            "apply_logical",
            side_effect=AssertionError("graph reverse called Python pullback orchestration"),
        ) as apply_logical:
            gradients = pullback(None)
        apply_logical.assert_not_called()
        np.testing.assert_array_equal(gradients["source"].to_numpy(), np.array([108.0], dtype=np.float32))
        epsilon = 1.0e-3
        finite_difference = ((3.0 + epsilon) ** 4 - (3.0 - epsilon) ** 4) / (2.0 * epsilon)
        self.assertAlmostEqual(float(gradients["source"].to_numpy()[0]), finite_difference, places=3)

    def test_graph_vjp_uses_retained_segment_owners_and_remains_reusable(self) -> None:
        source = vd.storage.from_numpy(np.array([1.0], dtype=np.float32))
        ping = vd.storage.from_numpy(np.array([1.0], dtype=np.float32))
        pong = vd.storage.from_numpy(np.array([1.0], dtype=np.float32))
        graph = vd.ExecutionGraph()
        previous = graph.differentiable_input("source", source)
        ping_resource = graph.import_resource(ping, exported=False)
        pong_resource = graph.objective("loss", pong)
        for index, output in enumerate((ping_resource, pong_resource, ping_resource, pong_resource)):
            graph.add_pass(
                vd.VjpComputePass(
                    f"product-{index}",
                    graph_checkpoint_product_vjp,
                    {"source": previous, "output": output},
                    grid=(1, 1, 1),
                )
            )
            previous = output
        graph.plan_autodiff_checkpoints(memory_budget=512)

        with mock.patch.object(
            vd.TensorStorage,
            "to_numpy",
            side_effect=AssertionError("checkpoint used NumPy"),
        ) as read:
            compiled = graph.compile()
            pullback = compiled.vjp()
        read.assert_not_called()
        checkpoint_plan = compiled.autodiff_checkpoint_plan
        assert checkpoint_plan is not None
        self.assertEqual(checkpoint_plan["memory_budget"], 512)
        self.assertEqual(checkpoint_plan["initial_state_bytes"], 0)
        self.assertEqual(checkpoint_plan["restoration_bytes"], 0)
        self.assertGreater(checkpoint_plan["transaction_bytes"], 0)
        self.assertEqual(checkpoint_plan["logical_residual_bytes"], 0)
        self.assertGreaterEqual(
            checkpoint_plan["retained_allocation_bytes"],
            checkpoint_plan["logical_residual_bytes"],
        )
        self.assertEqual(checkpoint_plan["checkpoint_resources"], [])
        self.assertTrue(all(item["source"] == "retained_owner" for item in checkpoint_plan["required_versions"]))
        self.assertEqual(pullback.checkpoint_bytes, 0)
        self.assertGreaterEqual(
            pullback.peak_runtime_managed_bytes,
            pullback.allocated_tape_bytes + pullback.checkpoint_bytes,
        )
        expected_loss = np.float32(1.0)
        self.assertAlmostEqual(float(pong.to_numpy()[0]), float(expected_loss), places=5)
        expected_gradient = np.float32(1.0)
        source.copy_from_numpy(np.array([2.0], dtype=np.float32))
        for _ in range(2):
            gradient = pullback(None)["source"].to_numpy()
            self.assertAlmostEqual(float(gradient[0]), float(expected_gradient), places=4)
            self.assertAlmostEqual(float(pong.to_numpy()[0]), float(expected_loss), places=5)
            np.testing.assert_array_equal(source.to_numpy(), np.array([2.0], dtype=np.float32))

    def test_graph_vjp_rejects_fan_in_without_program_graph_lowering(self) -> None:
        source = vd.storage.from_numpy(np.array([2.0], dtype=np.float32))
        square = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        cube = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        graph = vd.ExecutionGraph()
        source_resource = graph.differentiable_input("source", source)
        square_resource = graph.objective("square", square)
        cube_resource = graph.objective("cube", cube)
        graph.add_pass(
            vd.VjpComputePass(
                "square",
                graph_square_vjp,
                {"source": source_resource, "output": square_resource},
                grid=(1, 1, 1),
            )
        )
        graph.add_pass(
            vd.VjpComputePass(
                "cube",
                graph_cube_vjp,
                {"source": source_resource, "output": cube_resource},
                grid=(1, 1, 1),
            )
        )

        pullback = graph.compile().vjp()
        with self.assertRaisesRegex(RuntimeError, "Program Operation Graph"):
            pullback(
                {
                    "square": np.array([1.0], dtype=np.float32),
                    "cube": np.array([2.0], dtype=np.float32),
                }
            )

    def test_graph_vjp_returns_multiple_input_gradients_and_backward_submission(self) -> None:
        left = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        right = vd.storage.from_numpy(np.array([5.0], dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        graph = vd.ExecutionGraph()
        left_resource = graph.differentiable_input("left", left)
        right_resource = graph.differentiable_input("right", right)
        output_resource = graph.objective("output", output)
        graph.add_pass(
            vd.VjpComputePass(
                "product",
                graph_product_vjp,
                {"left": left_resource, "right": right_resource, "output": output_resource},
                grid=(1, 1, 1),
            )
        )

        compiled = graph.compile()
        pullback = compiled.vjp()
        second_pullback = compiled.vjp()
        backward = pullback.submit(None)

        self.assertIs(backward.state, vd.SubmissionState.SUCCEEDED)
        backward.wait()
        np.testing.assert_array_equal(backward.gradients["left"].to_numpy(), np.array([5.0], dtype=np.float32))
        np.testing.assert_array_equal(backward.gradients["right"].to_numpy(), np.array([3.0], dtype=np.float32))
        self.assertEqual(pullback.reverse_python_callback_count, 2)
        second_pullback(
            {
                "output": np.ones((1,), dtype=np.float32),
            }
        )
        self.assertEqual(second_pullback.reverse_python_callback_count, 1)
        self.assertEqual(pullback.reverse_python_callback_count, 2)

    def test_graph_vjp_does_not_double_count_aliased_pipeline_gradients(self) -> None:
        source = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        graph = vd.ExecutionGraph()
        source_resource = graph.differentiable_input("source", source)
        output_resource = graph.objective("output", output)
        graph.add_pass(
            vd.VjpComputePass(
                "square-through-product",
                graph_product_vjp,
                {"left": source_resource, "right": source_resource, "output": output_resource},
                grid=(1, 1, 1),
            )
        )
        graph.plan_autodiff_checkpoints(memory_budget=128)

        compiled = graph.compile()
        checkpoint_plan = compiled.autodiff_checkpoint_plan
        assert checkpoint_plan is not None
        self.assertEqual(checkpoint_plan["retained_allocation_bytes"], 24)
        self.assertEqual(checkpoint_plan["backward_value_bytes"], 16)
        self.assertLessEqual(checkpoint_plan["peak_bytes"], 128)
        required = checkpoint_plan["required_versions"]
        self.assertEqual(len(required), 2)
        self.assertEqual(len({item["resource"] for item in required}), 1)
        self.assertEqual({item["version"] for item in required}, {0})
        gradient = compiled.vjp()(None)["source"]

        np.testing.assert_array_equal(gradient.to_numpy(), np.array([6.0], dtype=np.float32))

    def test_graph_vjp_differentiates_execution_value_parameters(self) -> None:
        source = vd.storage.from_numpy(np.array([4.0], dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        graph = vd.ExecutionGraph()
        factor = graph.parameter("factor")
        graph.differentiable_input("factor", factor)
        output_resource = graph.objective("output", output)
        graph.add_pass(
            vd.VjpComputePass(
                "scale",
                graph_scale_vjp,
                {"source": source, "factor": factor, "output": output_resource},
                grid=(1, 1, 1),
            )
        )
        plan = graph.compile()
        bindings = plan.create_bindings({factor: np.float32(3.0)})

        gradient = plan.vjp(bindings)(None)["factor"]

        self.assertEqual(float(np.asarray(gradient)), 4.0)

    def test_graph_vjp_preserves_structured_storage_gradients(self) -> None:
        particles = vd.storage.zeros(dtype=Particle, shape=(1,))
        values = particles.to_numpy()
        values["velocity"][0] = np.array([2.0, -3.0], dtype=np.float16)
        values["mass"][0] = np.float32(4.0)
        particles.copy_from_numpy(values)
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        graph = vd.ExecutionGraph()
        particles_resource = graph.differentiable_input("particles", particles)
        loss_resource = graph.objective("loss", loss)
        graph.add_pass(
            vd.VjpComputePass(
                "aggregate",
                aggregate_storage_objective_vjp,
                {"particles": particles_resource, "loss": loss_resource},
                grid=(1, 1, 1),
            )
        )

        gradient = graph.compile().vjp()(None)["particles"]

        np.testing.assert_array_equal(
            gradient["velocity"].to_numpy(),
            np.array([[4.0, -6.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(gradient["mass"].to_numpy(), np.array([8.0], dtype=np.float32))

    def test_texture_view_rejects_incompatible_reinterpretations(self) -> None:
        texture = vd.Texture.zeros(shape=(4, 4))
        with self.assertRaisesRegex(ValueError, "format is incompatible"):
            texture.view(format=cast(Any, vd.r16_float))
        with self.assertRaisesRegex(ValueError, "dimension is incompatible"):
            texture.view(dimension="3d")
        with self.assertRaisesRegex(ValueError, "aspects are incompatible"):
            texture.view(aspects=("depth",))

    def test_cube_face_view_selects_one_layer(self) -> None:
        texture = vd.Texture.cube(np.zeros((6, 4, 4, 4), dtype=np.uint8))
        face = texture.view(dimension="2d", base_array_layer=2, array_layer_count=1)
        self.assertEqual(face.dimension, "2d")
        with self.assertRaisesRegex(ValueError, "exactly one array layer"):
            texture.view(dimension="2d")


class RecordingRenderPass(vd.RenderPass):
    def __init__(
        self,
        name: str,
        events: list[str],
        target: vd.RenderTarget,
        invocation: vd.PipelineInvocation | None = None,
    ):
        super().__init__(name)
        self.events = events
        self.target = target
        self.invocation = invocation

    def declare(self) -> None:
        self.attachments(self.target)

    def execute(self, encoder: vd.GraphicsEncoder, resources: vd.ExecutionResources) -> None:
        self.events.append(self.name)
        if self.invocation is not None:
            self.invocation.encode(encoder, resources)


class GpuExecutionGraphAutodiffTests(unittest.TestCase):
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

    def test_gpu_graph_vjp_retains_non_contiguous_tensor_view_layout(self) -> None:
        owner = vd.storage.from_numpy(np.array([10.0, 2.0, 20.0, 3.0], dtype=np.float32))
        source = owner.view(shape=(2,), strides=(-2,), offset=3, access="read")
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        graph = vd.ExecutionGraph()
        source_resource = graph.differentiable_input("source", source)
        output_resource = graph.objective("output", output)
        graph.add_pass(
            vd.VjpComputePass(
                "strided-square-sum",
                graph_strided_square_sum_vjp,
                {"source": source_resource, "output": output_resource},
                grid=(1, 1, 1),
            )
        )

        gradient = graph.compile().vjp()(None)["source"]

        np.testing.assert_array_equal(output.to_numpy(), np.array([13.0], dtype=np.float32))
        np.testing.assert_array_equal(gradient.to_numpy(), np.array([0.0, 4.0, 0.0, 6.0], dtype=np.float32))

    def test_gpu_graph_vjp_allows_unrelated_non_differentiable_compute_pass(self) -> None:
        unrelated_source = vd.storage.from_numpy(np.array([3.0], dtype=np.float32))
        unrelated_output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        source = vd.storage.from_numpy(np.array([2.0], dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        graph = vd.ExecutionGraph()
        graph.add_pass(
            InvocationComputePass(
                "unrelated", cast(Any, graph_increment).invocation(unrelated_output, unrelated_source)
            )
        )
        source_resource = graph.differentiable_input("source", source)
        graph.add_pass(
            vd.VjpComputePass(
                "square",
                graph_square_vjp,
                {"source": source_resource, "output": graph.objective("output", output)},
                grid=(1, 1, 1),
            )
        )

        gradient = graph.compile().vjp()(None)["source"]

        np.testing.assert_array_equal(unrelated_output.to_numpy(), np.array([4.0], dtype=np.float32))
        np.testing.assert_array_equal(gradient.to_numpy(), np.array([4.0], dtype=np.float32))

    def test_gpu_graph_vjp_rejects_unlowered_branch_fan_in(self) -> None:
        source = vd.storage.from_numpy(np.array([2.0], dtype=np.float32))
        square = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        cube = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        graph = vd.ExecutionGraph()
        source_resource = graph.differentiable_input("source", source)
        graph.add_pass(
            vd.VjpComputePass(
                "square",
                graph_square_vjp,
                {"source": source_resource, "output": graph.objective("square", square)},
                grid=(1, 1, 1),
            )
        )
        graph.add_pass(
            vd.VjpComputePass(
                "cube",
                graph_cube_vjp,
                {"source": source_resource, "output": graph.objective("cube", cube)},
                grid=(1, 1, 1),
            )
        )
        pullback = graph.compile().vjp()
        with self.assertRaisesRegex(RuntimeError, "Program Operation Graph"):
            pullback(
                {
                    "square": np.array([1.0], dtype=np.float32),
                    "cube": np.array([2.0], dtype=np.float32),
                }
            )

    def test_gpu_graph_vjp_preserves_structured_storage_gradients(self) -> None:
        particles = vd.storage.zeros(dtype=GraphParticle, shape=(1,))
        values = particles.to_numpy()
        values["velocity"][0] = np.array([2.0, -3.0], dtype=np.float16)
        values["mass"][0] = np.float32(4.0)
        particles.copy_from_numpy(values)
        loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        graph = vd.ExecutionGraph()
        particles_resource = graph.differentiable_input("particles", particles)
        loss_resource = graph.objective("loss", loss)
        graph.add_pass(
            vd.VjpComputePass(
                "aggregate",
                graph_aggregate_storage_objective_vjp,
                {"particles": particles_resource, "loss": loss_resource},
                grid=(1, 1, 1),
            )
        )

        gradient = graph.compile().vjp()(None)["particles"]

        np.testing.assert_array_equal(gradient["velocity"].to_numpy(), np.array([[4.0, -6.0]], dtype=np.float32))
        np.testing.assert_array_equal(gradient["mass"].to_numpy(), np.array([8.0], dtype=np.float32))

    def test_gpu_graph_vjp_rejects_unlowered_aggregate_fan_in(self) -> None:
        particles = vd.storage.zeros(dtype=GraphParticle, shape=(1,))
        values = particles.to_numpy()
        values["velocity"][0] = np.array([2.0, -3.0], dtype=np.float16)
        values["mass"][0] = np.float32(4.0)
        particles.copy_from_numpy(values)
        first_loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        second_loss = vd.storage.zeros(dtype=vd.f32, shape=(1,))
        graph = vd.ExecutionGraph()
        particles_resource = graph.differentiable_input("particles", particles)
        graph.add_pass(
            vd.VjpComputePass(
                "first",
                graph_aggregate_storage_objective_vjp,
                {"particles": particles_resource, "loss": graph.objective("first_loss", first_loss)},
                grid=(1, 1, 1),
            )
        )
        graph.add_pass(
            vd.VjpComputePass(
                "second",
                graph_aggregate_storage_objective_vjp,
                {"particles": particles_resource, "loss": graph.objective("second_loss", second_loss)},
                grid=(1, 1, 1),
            )
        )
        pullback = graph.compile().vjp()
        with self.assertRaisesRegex(RuntimeError, "Program Operation Graph"):
            pullback(
                {
                    "first_loss": np.ones((1,), dtype=np.float32),
                    "second_loss": np.ones((1,), dtype=np.float32),
                }
            )


class ExecutionGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            vd.init(arch=vd.opengl, api_version=(4, 3))
        except RuntimeUnavailableError as error:
            raise unittest.SkipTest("OpenGL 4.3 compute runtime is unavailable") from error

    @classmethod
    def tearDownClass(cls) -> None:
        vd.init(arch=vd.cpu)

    def test_hazards_schedule_producer_before_consumer(self) -> None:
        graph = vd.ExecutionGraph()
        events: list[str] = []
        source = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        intermediate = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        output = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        graph.add_pass(RecordingComputePass("producer", events, read=source, write=intermediate))
        graph.add_pass(RecordingComputePass("consumer", events, read=intermediate, write=output))

        plan = graph.compile()
        plan.submit().wait()

        self.assertEqual([execution_pass.name for execution_pass in plan.schedule], ["producer", "consumer"])
        self.assertEqual(events, ["producer", "consumer"])
        self.assertEqual(len(plan.scopes[1].barriers), 1)

    def test_sampled_depth_target_and_texture_share_graph_identity(self) -> None:
        graph = vd.ExecutionGraph()
        events: list[str] = []
        color = vd.Texture.zeros(shape=(8, 8))
        target = vd.RenderTarget(shape=(8, 8)).attach_color(0, cast(Any, color)).attach_depth()
        depth = target.depth_texture
        writer = RecordingRenderPass("shadow", events, target)
        reader = RecordingComputePass("sample", events, read=depth)
        reader.side_effect = True
        graph.add_pass(writer)
        graph.add_pass(reader)

        plan = graph.compile()

        self.assertEqual([execution_pass.name for execution_pass in plan.schedule], ["shadow", "sample"])
        self.assertEqual(len(plan.scopes[1].barriers), 1)

    def test_texture_views_track_disjoint_mips_without_false_barriers(self) -> None:
        texture = vd.Texture.zeros(
            shape=(8, 8),
            mip_levels=2,
            usage=("sampled", "storage", "transfer_source", "transfer_destination"),
        )
        mip0 = texture.view(base_mip_level=0, mip_level_count=1)
        mip1 = texture.view(base_mip_level=1, mip_level_count=1)
        graph = vd.ExecutionGraph()
        events: list[str] = []
        for execution_pass in (
            RecordingComputePass("write-mip-0", events, write=mip0),
            RecordingComputePass("read-mip-1", events, read=mip1),
            RecordingComputePass("read-mip-0", events, read=mip0),
        ):
            execution_pass.side_effect = True
            graph.add_pass(execution_pass)

        plan = graph.compile()

        self.assertEqual(len(plan.scopes), 3)
        self.assertEqual(len(plan.scopes[1].barriers), 0)
        self.assertEqual(len(plan.scopes[2].barriers), 1)
        self.assertEqual(plan.scopes[2].barriers[0].base_mip_level, 0)
        self.assertEqual(plan.scopes[2].barriers[0].mip_level_count, 1)

    def test_dependency_cycle_is_rejected(self) -> None:
        graph = vd.ExecutionGraph()
        events: list[str] = []
        first = graph.add_pass(RecordingComputePass("first", events))
        second = graph.add_pass(RecordingComputePass("second", events))
        first.side_effect = True
        second.side_effect = True
        first.depends_on(second)
        second.depends_on(first)

        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            graph.compile()

    def test_culls_transient_work_without_live_consumers(self) -> None:
        graph = vd.ExecutionGraph()
        events: list[str] = []
        transient = graph.import_resource(vd.storage.zeros(dtype=vd.f32, shape=(4,)), exported=False)
        output = graph.import_resource(vd.storage.zeros(dtype=vd.f32, shape=(4,)))
        graph.add_pass(RecordingComputePass("dead", events, write=transient))
        graph.add_pass(RecordingComputePass("live", events, write=output))

        plan = graph.compile()
        plan.submit().wait()

        self.assertEqual([execution_pass.name for execution_pass in plan.schedule], ["live"])
        self.assertEqual(events, ["live"])

    def test_compatible_render_passes_fuse_and_compute_splits(self) -> None:
        graph = vd.ExecutionGraph()
        events: list[str] = []
        color = vd.Texture.zeros(shape=(8, 8))
        target = vd.RenderTarget(shape=color.shape).attach_color(0, cast(Any, color))
        graph.add_pass(RecordingRenderPass("first", events, target))
        graph.add_pass(RecordingRenderPass("second", events, target))
        compute = RecordingComputePass("compute", events)
        compute.side_effect = True
        graph.add_pass(compute)

        plan = graph.compile()

        self.assertEqual([scope.kind for scope in plan.scopes], ["render", "compute"])
        self.assertEqual([execution_pass.name for execution_pass in plan.scopes[0].passes], ["first", "second"])

    def test_no_merge_splits_matching_render_targets(self) -> None:
        graph = vd.ExecutionGraph()
        color = vd.Texture.zeros(shape=(8, 8))
        target = vd.RenderTarget(shape=color.shape).attach_color(0, cast(Any, color))
        first = RecordingRenderPass("first", [], target)
        second = RecordingRenderPass("second", [], target)
        second.no_merge = True
        graph.add_pass(first)
        graph.add_pass(second)

        plan = graph.compile()

        self.assertEqual([scope.kind for scope in plan.scopes], ["render", "render"])

        fused_graph = vd.ExecutionGraph()
        fused_graph.add_pass(RecordingRenderPass("first", [], target))
        fused_graph.add_pass(RecordingRenderPass("second", [], target))
        fused_plan = fused_graph.compile()

        self.assertEqual([scope.kind for scope in fused_plan.scopes], ["render"])

    def test_resource_from_another_graph_is_rejected(self) -> None:
        first = vd.ExecutionGraph()
        foreign = first.import_resource(vd.storage.zeros(dtype=vd.f32, shape=(4,)))
        second = vd.ExecutionGraph()
        execution_pass = RecordingComputePass("foreign", [], read=foreign)
        execution_pass.side_effect = True
        second.add_pass(execution_pass)

        with self.assertRaisesRegex(ValueError, "does not belong"):
            second.compile()

    def test_unified_pipeline_invocation_checks_encoder_kind(self) -> None:
        invocation = vd.PipelineInvocation("compute", lambda encoder, resources: None)
        color = vd.Texture.zeros(shape=(4, 4))
        target = vd.RenderTarget(shape=color.shape).attach_color(0, cast(Any, color))
        render_pass = RecordingRenderPass("render", [], target, invocation)
        graph = vd.ExecutionGraph()
        graph.add_pass(render_pass)

        plan = graph.compile()
        with self.assertRaisesRegex(TypeError, "compute invocation"):
            plan.submit()


if __name__ == "__main__":
    unittest.main()
