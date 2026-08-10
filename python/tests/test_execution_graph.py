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

        graph.execute()

        self.assertEqual([execution_pass.name for execution_pass in graph.schedule], ["write", "read"])
        self.assertEqual(events, ["write", "read"])
        self.assertEqual(len(graph.scopes[1].barriers), 1)

    def test_cpu_kernel_invocations_execute_through_graph(self) -> None:
        source = vd.storage.from_numpy(np.arange(4, dtype=np.float32))
        intermediate = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        output = vd.storage.zeros(dtype=vd.f32, shape=(4,))
        graph = vd.ExecutionGraph()
        graph.add_pass(InvocationComputePass("first", graph_increment.invocation(intermediate, source)))
        graph.add_pass(InvocationComputePass("second", graph_increment.invocation(output, intermediate)))

        graph.execute()

        np.testing.assert_array_equal(output.to_numpy(), np.arange(4, dtype=np.float32) + 2.0)
        self.assertEqual([execution_pass.name for execution_pass in graph.schedule], ["first", "second"])

    def test_cpu_callback_failure_is_propagated_and_stops_execution(self) -> None:
        events: list[str] = []
        graph = vd.ExecutionGraph()
        graph.add_pass(FailingComputePass(events))
        after = RecordingComputePass("after", events)
        after.side_effect = True
        graph.add_pass(after)

        with self.assertRaisesRegex(RuntimeError, "intentional graph failure"):
            graph.execute()

        self.assertEqual(events, ["failure"])


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

        graph.execute()

        self.assertEqual([execution_pass.name for execution_pass in graph.schedule], ["producer", "consumer"])
        self.assertEqual(events, ["producer", "consumer"])
        self.assertEqual(len(graph.scopes[1].barriers), 1)

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

        graph.compile()

        self.assertEqual([execution_pass.name for execution_pass in graph.schedule], ["shadow", "sample"])
        self.assertEqual(len(graph.scopes[1].barriers), 1)

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

        graph.execute()

        self.assertEqual([execution_pass.name for execution_pass in graph.schedule], ["live"])
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

        graph.compile()

        self.assertEqual([scope.kind for scope in graph.scopes], ["render", "compute"])
        self.assertEqual([execution_pass.name for execution_pass in graph.scopes[0].passes], ["first", "second"])

    def test_no_merge_splits_matching_render_targets(self) -> None:
        graph = vd.ExecutionGraph()
        color = vd.Texture.zeros(shape=(8, 8))
        target = vd.RenderTarget(shape=color.shape).attach_color(0, color)
        first = RecordingRenderPass("first", [], target)
        second = RecordingRenderPass("second", [], target)
        second.no_merge = True
        graph.add_pass(first)
        graph.add_pass(second)

        graph.compile()

        self.assertEqual([scope.kind for scope in graph.scopes], ["render", "render"])

        second.no_merge = False
        graph.compile()

        self.assertEqual([scope.kind for scope in graph.scopes], ["render"])

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
        invocation = vd.PipelineInvocation("compute", lambda encoder: None)
        color = vd.Texture.zeros(shape=(4, 4))
        target = vd.RenderTarget(shape=color.shape).attach_color(0, color)
        render_pass = RecordingRenderPass("render", [], target, invocation)
        graph = vd.ExecutionGraph()
        graph.add_pass(render_pass)

        with self.assertRaisesRegex(TypeError, "compute invocation"):
            graph.execute()


if __name__ == "__main__":
    unittest.main()
