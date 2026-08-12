from __future__ import annotations

import unittest
from typing import Annotated

import numpy as np
import vernon_dsl as vd
from vernon_dsl._runtime.session import RuntimeUnavailableError


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
        graph.add_pass(InvocationComputePass("first", graph_increment.invocation(intermediate, source)))
        graph.add_pass(InvocationComputePass("second", graph_increment.invocation(output, intermediate)))

        plan = graph.compile()
        plan.submit().wait()

        np.testing.assert_array_equal(output.to_numpy(), np.arange(4, dtype=np.float32) + 2.0)
        self.assertEqual([execution_pass.name for execution_pass in plan.schedule], ["first", "second"])

    def test_parameterized_kernel_invocation_resolves_each_submission_snapshot(self) -> None:
        source = vd.storage.from_numpy(np.arange(4, dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        graph = vd.ExecutionGraph()
        amount = graph.parameter("amount")
        invocation = graph_add_parameter.invocation(output, source, amount)
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
        np.testing.assert_array_equal(first, np.array([1.0], dtype=np.float32))
        constant_value[0] = 99.0
        bindings.update({dynamic: np.array([3.0], dtype=np.float32)})
        plan.submit(bindings).wait()
        np.testing.assert_array_equal(values[-1], np.array([3.0], dtype=np.float32))
        np.testing.assert_array_equal(constant_values[-1], np.array([7.0], dtype=np.float32))

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

    def test_texture_view_rejects_incompatible_reinterpretations(self) -> None:
        texture = vd.Texture.zeros(shape=(4, 4))
        with self.assertRaisesRegex(ValueError, "format is incompatible"):
            texture.view(format=vd.r16_float)
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
        target = vd.RenderTarget(shape=(8, 8)).attach_color(0, color).attach_depth()
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
        target = vd.RenderTarget(shape=color.shape).attach_color(0, color)
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
        target = vd.RenderTarget(shape=color.shape).attach_color(0, color)
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
        target = vd.RenderTarget(shape=color.shape).attach_color(0, color)
        render_pass = RecordingRenderPass("render", [], target, invocation)
        graph = vd.ExecutionGraph()
        graph.add_pass(render_pass)

        plan = graph.compile()
        with self.assertRaisesRegex(TypeError, "compute invocation"):
            plan.submit()


if __name__ == "__main__":
    unittest.main()
