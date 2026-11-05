"""Kernels called through a linked schedule (specs/runtime/architecture_v2.md) agree with the Program path."""

from __future__ import annotations

import contextlib
import dataclasses
import unittest
from collections.abc import Iterator
from typing import Annotated, Any
from unittest import mock

import numpy as np
import vernon_dsl as vd
from backend_test_matrix import BackendRequirements, backend_row, probe_backend, require_available
from vernon_dsl import program as program_module
from vernon_dsl._runtime.source_epoch import source_epoch
from vernon_dsl._runtime.tensor import TensorStorage


@vd.kernel(workgroup_size=(8, 1, 1))
def scaled_sum(
    x: vd.TensorView[vd.f64, (vd.dyn,), vd.read],
    y: vd.TensorView[vd.f64, (vd.dyn,), vd.read],
    out: vd.TensorView[vd.f64, (vd.dyn,), vd.write],
    a: vd.f64,
    count: vd.u32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    i = gid[0]
    if i < count:
        out[i] = a * x[i] + y[i]


@vd.kernel(workgroup_size=(4, 1, 1))
def row_sums(
    table: vd.TensorView[vd.i32, (vd.dyn, 3), vd.read],
    out: vd.TensorView[vd.i32, (vd.dyn,), vd.read_write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    i = gid[0]
    if i < out.shape[0]:
        out[i] = out[i] + table[i, 0] + table[i, 1] + table[i, 2]


class LinkedScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        vd.init(arch=vd.cpu)
        self.rng = np.random.default_rng(3)

    def _sum_inputs(self, n: int) -> tuple[np.ndarray, np.ndarray, vd.TensorStorage, vd.TensorStorage]:
        xs, ys = self.rng.random(n), self.rng.random(n)
        return xs, ys, vd.storage.from_numpy(xs), vd.storage.from_numpy(ys)

    def test_first_call_links_and_later_calls_agree(self) -> None:
        xs, ys, x, y = self._sum_inputs(100)
        out = vd.storage.zeros(dtype=vd.f64, shape=(100,))
        scaled_sum(x, y, out, 2.0, 100, grid=(13, 1, 1))
        self.assertIsNotNone(scaled_sum._linked)
        np.testing.assert_array_equal(out.to_numpy(), 2.0 * xs + ys)
        version = out._control.version
        scaled_sum(x, y, out, np.float64(3.0), np.uint32(100), grid=(13, 1, 1))
        np.testing.assert_array_equal(out.to_numpy(), 3.0 * xs + ys)
        self.assertGreater(out._control.version, version)
        self.assertEqual(out._control.host_version, out._control.version)

    def test_views_pass_their_offset_extent_and_stride(self) -> None:
        xs, ys, x, y = self._sum_inputs(64)
        out = vd.storage.zeros(dtype=vd.f64, shape=(64,))
        scaled_sum(x, y, out, 1.0, 64, grid=(8, 1, 1))
        forward = x.view(shape=(20,), strides=(3,), offset=2, access="read")
        backward = y.view(shape=(20,), strides=(-2,), offset=50, access="read")
        target = out.view(shape=(20,), strides=(2,), offset=5, access="write")
        expected = out.to_numpy()
        expected[5:45:2] = 0.5 * xs[2:62:3] + ys[50:10:-2]
        scaled_sum(forward, backward, target, 0.5, 20, grid=(3, 1, 1))
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_rank_two_views_and_read_write_outputs(self) -> None:
        table = self.rng.integers(-50, 50, size=(9, 3), dtype=np.int32)
        out = vd.storage.from_numpy(np.arange(9, dtype=np.int32))
        stored = vd.storage.from_numpy(table)
        row_sums(stored, out, grid=(3, 1, 1))
        row_sums(stored, out, grid=(3, 1, 1))
        self.assertIsNotNone(row_sums._linked)
        np.testing.assert_array_equal(out.to_numpy(), np.arange(9, dtype=np.int32) + 2 * table.sum(axis=1))

    def test_unsupported_arguments_fall_back_with_the_program_diagnostics(self) -> None:
        _, _, x, y = self._sum_inputs(16)
        out = vd.storage.zeros(dtype=vd.f64, shape=(16,))
        scaled_sum(x, y, out, 1.0, 16, grid=(2, 1, 1))
        self.assertIsNotNone(scaled_sum._linked)
        with self.assertRaisesRegex(TypeError, "dtype float32 does not match f64"):
            scaled_sum(x, y, vd.storage.zeros(dtype=vd.f32, shape=(16,)), 1.0, 16, grid=(2, 1, 1))
        with self.assertRaisesRegex(ValueError, "grid must contain three positive integers"):
            scaled_sum(x, y, out, 1.0, 16, grid=(2, 0, 1))
        with self.assertRaisesRegex(TypeError, "access 'read' does not satisfy 'write'"):
            scaled_sum(x, y, out.view(access="read"), 1.0, 16, grid=(2, 1, 1))
        overlapping = out.view(shape=(4, 4), strides=(1, 1), access="write")
        with self.assertRaises((TypeError, ValueError)):
            scaled_sum(x, y, overlapping, 1.0, 16, grid=(2, 1, 1))

    def test_a_written_owner_borrowed_twice_takes_the_program_path(self) -> None:
        xs, ys, x, y = self._sum_inputs(16)
        scaled_sum(x, y, vd.storage.zeros(dtype=vd.f64, shape=(16,)), 1.0, 16, grid=(2, 1, 1))
        expected = xs.copy()
        expected[8:16] = 2.0 * xs[0:8] + ys[0:8]
        source = x.view(shape=(8,), strides=(1,), offset=0, access="read")
        target = x.view(shape=(8,), strides=(1,), offset=8, access="write")
        scaled_sum(source, y, target, 2.0, 8, grid=(1, 1, 1))
        np.testing.assert_array_equal(x.to_numpy(), expected)

    def test_a_written_view_that_overlaps_another_argument_is_rejected_on_every_call(self) -> None:
        # The CPU executable marks buffers `noalias`, which holds only while every call with overlap is rejected.
        xs, ys, x, y = self._sum_inputs(16)
        scaled_sum(x, y, vd.storage.zeros(dtype=vd.f64, shape=(16,)), 1.0, 16, grid=(2, 1, 1))
        self.assertIsNotNone(scaled_sum._linked)
        shared = xs.copy()
        left, right = TensorStorage(shared), TensorStorage(shared)
        for _ in range(3):
            with self.assertRaisesRegex(ValueError, "overlap"):
                scaled_sum(x, y, x, 1.0, 16, grid=(2, 1, 1))
            with self.assertRaisesRegex(ValueError, "overlap"):
                scaled_sum(left, y, right, 1.0, 16, grid=(2, 1, 1))
        np.testing.assert_array_equal(x.to_numpy(), xs)
        np.testing.assert_array_equal(shared, xs)

    def test_wait_returns_at_once_on_the_cpu(self) -> None:
        xs, ys, x, y = self._sum_inputs(16)
        out = vd.storage.zeros(dtype=vd.f64, shape=(16,))
        for _ in range(2):
            scaled_sum(x, y, out, 2.0, 16, grid=(2, 1, 1))
        out.wait()
        out.view(access="read").wait()
        np.testing.assert_array_equal(out._array, 2.0 * xs + ys)

    def test_a_new_source_epoch_relinks(self) -> None:
        _, _, x, y = self._sum_inputs(8)
        out = vd.storage.zeros(dtype=vd.f64, shape=(8,))
        scaled_sum(x, y, out, 1.0, 8, grid=(1, 1, 1))
        linked = scaled_sum._linked
        source_epoch.bump()
        scaled_sum(x, y, out, 1.0, 8, grid=(1, 1, 1))
        self.assertIsNot(scaled_sum._linked, linked)
        self.assertEqual(scaled_sum._linked.epoch, source_epoch.value)

    def test_a_new_session_relinks(self) -> None:
        _, _, x, y = self._sum_inputs(8)
        out = vd.storage.zeros(dtype=vd.f64, shape=(8,))
        scaled_sum(x, y, out, 1.0, 8, grid=(1, 1, 1))
        session = vd.RuntimeSession(arch=vd.cpu)
        with session:
            scaled_sum(x, y, out, 4.0, 8, grid=(1, 1, 1))
            self.assertEqual(scaled_sum._linked.session, session.identity)
        np.testing.assert_array_equal(out.to_numpy(), 4.0 * x.to_numpy() + y.to_numpy())


class CudaLinkedScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        require_available(probe_backend(backend_row(vd.cuda), BackendRequirements(gpu=True, f64=True, compute=True)))
        self.rng = np.random.default_rng(5)

    def tearDown(self) -> None:
        vd.init(arch=vd.cpu)

    def test_linked_calls_use_and_publish_the_session_residency(self) -> None:
        xs, ys = self.rng.random(100), self.rng.random(100)
        x, y = vd.storage.from_numpy(xs), vd.storage.from_numpy(ys)
        out = vd.storage.zeros(dtype=vd.f64, shape=(100,))
        scaled_sum(x, y, out, 2.0, 100, grid=(13, 1, 1))
        np.testing.assert_array_equal(out.to_numpy(), 2.0 * xs + ys)
        schedule = scaled_sum._linked.schedule
        version = out._control.version
        self.assertTrue(schedule.call((x, y, out, 3.0, 100), (13, 1, 1)))
        control = out._control
        self.assertEqual(control.version, version + 1)
        self.assertTrue(control.coherence.device_dirty)
        self.assertEqual(max(r.version for r in control.residencies._entries.values()), control.version)
        np.testing.assert_array_equal(out.to_numpy(), 3.0 * xs + ys)

    def test_a_host_write_falls_back_once_to_upload(self) -> None:
        xs, ys = self.rng.random(16), self.rng.random(16)
        x, y = vd.storage.from_numpy(xs), vd.storage.from_numpy(ys)
        out = vd.storage.zeros(dtype=vd.f64, shape=(16,))
        scaled_sum(x, y, out, 1.0, 16, grid=(2, 1, 1))
        schedule = scaled_sum._linked.schedule
        x[:] = np.ones(16)
        self.assertFalse(schedule.call((x, y, out, 1.0, 16), (2, 1, 1)))
        scaled_sum(x, y, out, 1.0, 16, grid=(2, 1, 1))
        self.assertTrue(schedule.call((x, y, out, 2.0, 16), (2, 1, 1)))
        np.testing.assert_array_equal(out.to_numpy(), 2.0 + ys)

    def test_views_and_rank_two_read_write_arguments(self) -> None:
        xs, ys = self.rng.random(64), self.rng.random(64)
        x, y = vd.storage.from_numpy(xs), vd.storage.from_numpy(ys)
        out = vd.storage.zeros(dtype=vd.f64, shape=(64,))
        forward = x.view(shape=(20,), strides=(3,), offset=2, access="read")
        backward = y.view(shape=(20,), strides=(-2,), offset=50, access="read")
        target = out.view(shape=(20,), strides=(2,), offset=5, access="write")
        expected = np.zeros(64)
        expected[5:45:2] = 0.5 * xs[2:62:3] + ys[50:10:-2]
        for _ in range(2):
            scaled_sum(forward, backward, target, 0.5, 20, grid=(3, 1, 1))
        np.testing.assert_array_equal(out.to_numpy(), expected)

        table = self.rng.integers(-50, 50, size=(9, 3), dtype=np.int32)
        sums = vd.storage.from_numpy(np.arange(9, dtype=np.int32))
        stored = vd.storage.from_numpy(table)
        for _ in range(3):
            row_sums(stored, sums, grid=(3, 1, 1))
        self.assertIsNotNone(row_sums._linked)
        np.testing.assert_array_equal(sums.to_numpy(), np.arange(9, dtype=np.int32) + 3 * table.sum(axis=1))

    def test_wait_completes_the_last_write_without_a_download(self) -> None:
        xs, ys = self.rng.random(4096), self.rng.random(4096)
        x, y = vd.storage.from_numpy(xs), vd.storage.from_numpy(ys)
        out = vd.storage.zeros(dtype=vd.f64, shape=(4096,))
        scaled_sum(x, y, out, 1.0, 4096, grid=(512, 1, 1))
        np.testing.assert_array_equal(out.to_numpy(), xs + ys)
        points = []
        for scale in (2.0, 3.0):
            scaled_sum(x, y, out, scale, 4096, grid=(512, 1, 1))
            points.append(out._control.residencies.latest().write_point)
            out.view(access="read").wait()
        self.assertGreater(points[1], points[0])
        self.assertTrue(out._control.coherence.device_dirty)
        np.testing.assert_array_equal(out._array, xs + ys)
        np.testing.assert_array_equal(out.to_numpy(), 3.0 * xs + ys)
        out.wait()


@vd.kernel(workgroup_size=(8, 1, 1))
def accumulate(
    total: vd.TensorView[vd.f32, (vd.dyn,), vd.read_write],
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    i = gid[0]
    if i < source.shape[0]:
        total[i] = total[i] + source[i]


@vd.kernel(workgroup_size=(8, 1, 1))
def scale(
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    factor: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    i = gid[0]
    if i < source.shape[0]:
        output[i] = source[i] * factor


class Doubled(vd.Module):
    """Three nodes over a zeroed allocation that is not an output, so the schedule holds it in its arena."""

    def forward(self, source: vd.TensorView[vd.f32, (vd.dyn,), vd.read], factor: vd.f32) -> vd.TensorStorage:
        total = vd.zeros_like(source)
        output = vd.empty_like(source)
        accumulate(total, source, grid=(128, 1, 1))
        accumulate(total, source, grid=(128, 1, 1))
        scale(total, output, factor, grid=(128, 1, 1))
        return output


@dataclasses.dataclass
class Scaled:
    twice: vd.TensorStorage
    thrice: vd.TensorStorage


class FanOut(vd.Module):
    def forward(self, source: vd.TensorView[vd.f32, (vd.dyn,), vd.read]) -> Scaled:
        twice = vd.empty_like(source)
        thrice = vd.empty_like(source)
        scale(source, twice, 2.0, grid=(128, 1, 1))
        scale(source, thrice, 3.0, grid=(128, 1, 1))
        return Scaled(twice, thrice)


class FixedSum(vd.Module):
    def forward(self, source: vd.TensorView[vd.f32, (vd.dyn,), vd.read]) -> vd.TensorStorage:
        total = vd.zeros(dtype=vd.f32, shape=(16,))
        accumulate(total, source, grid=(2, 1, 1))
        accumulate(total, source, grid=(2, 1, 1))
        return total


class ScaledChain(vd.Module):
    """Two nodes whose backward graph captures the scalar factor and the intermediate as a residual."""

    def forward(self, source: vd.TensorView[vd.f32, (vd.dyn,), vd.read], factor: vd.f32) -> vd.TensorStorage:
        middle = vd.empty_like(source)
        output = vd.empty_like(source)
        scale(source, middle, factor, grid=(128, 1, 1))
        scale(middle, output, factor, grid=(128, 1, 1))
        return output


@vd.kernel(workgroup_size=(8, 1, 1))
def weigh(
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    weight: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    i = gid[0]
    if i < source.shape[0]:
        output[i] = source[i] * weight[i]


class WeightedChain(vd.Module):
    """Three weighted layers and a scaled one, so the backward graph captures each layer's input as a residual and
    a scalar kernel parameter follows buffer arguments."""

    def forward(
        self,
        source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
        w0: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
        w1: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
        w2: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
        factor: vd.f32,
    ) -> vd.TensorStorage:
        a = vd.empty_like(source)
        b = vd.empty_like(source)
        c = vd.empty_like(source)
        output = vd.empty_like(source)
        weigh(source, w0, a, grid=(128, 1, 1))
        weigh(a, w1, b, grid=(128, 1, 1))
        weigh(b, w2, c, grid=(128, 1, 1))
        scale(c, output, factor, grid=(128, 1, 1))
        return output


@vd.kernel(workgroup_size=(8, 1, 1))
def weigh_repeatedly(
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    weight: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    count: vd.i32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    i = gid[0]
    if i < source.shape[0]:
        result = source[i]
        for _ in range(count):
            result = result * weight[i]
        output[i] = result


class RepeatedChain(vd.Module):
    """Two layers of a kernel with a dynamic loop, so each node keeps an autodiff tape its backward node replays."""

    def forward(
        self,
        source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
        weight: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
        count: vd.i32,
    ) -> vd.TensorStorage:
        middle = vd.empty_like(source)
        output = vd.empty_like(source)
        weigh_repeatedly(source, weight, middle, count, grid=(128, 1, 1))
        weigh_repeatedly(middle, weight, output, count, grid=(128, 1, 1))
        return output


def repeated_chain_gradients(
    data: np.ndarray, weight: np.ndarray, count: int, cotangent: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "source": cotangent * weight ** (2 * count),
        "weight": cotangent * 2 * count * data * weight ** (2 * count - 1),
    }


def check_two_program_graph(test: unittest.TestCase) -> None:
    """Two Programs linked as one schedule: the first's output is the second's source, which the schedule keeps in its
    arena, and both take one factor."""
    from vernon_dsl._runtime.session import _session_state, cpu
    from vernon_dsl._runtime.tensor import TensorView

    module = ScaledChain()
    module(vd.storage.from_numpy(np.ones(4, dtype=np.float32)), 1.0)
    state = _session_state()
    specialization = next(iter(module._program_cache._partition(state).snapshot.values()))
    deployment = specialization.native_program.deployment
    program, graphs = program_module._linked_program(state, deployment)
    test.assertEqual(
        [row["path"] for row in program_module._graph_slots(program, "forward")], ["source", "factor", "output"]
    )
    used = {node["stage"] for node in graphs["forward"]["nodes"]}
    stages = [stage for stage in deployment.linked_stages if stage[0] in used]
    node = (deployment.program_json, "forward", stages, [], [])
    key = None if state.arch == cpu else state.identity
    graph = state.native._ScheduleKernel.link_graph(
        state.native_runtime,
        [node, node],
        [([(0, 2), (1, 0)], True), ([(0, 1), (1, 1)], False)],
        TensorStorage,
        TensorView,
        key,
    )
    test.assertEqual(graph.node_slots, [[0, 1, None], [None, 1, 2]])
    test.assertEqual(graph.slot_extents[2], [(0, 0, 0)])
    for n in (10, 333):
        data = np.linspace(-1.0, 1.0, n).astype(np.float32)
        source = vd.storage.from_numpy(data)
        output = vd.storage.zeros(dtype=vd.f32, shape=(n,))
        if key is not None:
            test.assertTrue(program_module._ensure_device_residency(source, state))
            test.assertTrue(program_module._ensure_device_residency(output, state))
        test.assertTrue(graph.call((source, 1.5, output), None))
        np.testing.assert_allclose(output.to_numpy(), data * 1.5**4, rtol=1e-6)


class WeightedChainChecks:
    """A linked VJP over several layers: each pullback reads its own forward call's residuals, so pullbacks taken
    from more forward calls than a CUDA graph's record ring holds still differentiate their own calls, in any order."""

    def check_weighted_chain(self, test: unittest.TestCase) -> None:
        rng = np.random.default_rng(11)
        transformed = vd.ad.vjp(WeightedChain(), wrt=("source", "w0", "w1", "w2"), outputs=None)
        for n in (200, 1000, 200):
            weights = [rng.standard_normal(n).astype(np.float32) for _ in range(3)]
            stored = [vd.storage.from_numpy(weight) for weight in weights]
            calls = []
            for index in range(6):
                data = rng.standard_normal(n).astype(np.float32)
                factor = 0.5 + index
                output, pullback = transformed(vd.storage.from_numpy(data), *stored, factor)
                calls.append((data, factor, output, pullback))
            for data, factor, output, pullback in reversed(calls):
                w0, w1, w2 = weights
                np.testing.assert_allclose(output.to_numpy(), data * w0 * w1 * w2 * factor, rtol=1e-5)
                cotangent = rng.standard_normal(n).astype(np.float32)
                gradients = pullback(cotangent)
                expected = {
                    "source": cotangent * factor * w0 * w1 * w2,
                    "w0": cotangent * factor * data * w1 * w2,
                    "w1": cotangent * factor * data * w0 * w2,
                    "w2": cotangent * factor * data * w0 * w1,
                }
                for name, value in expected.items():
                    np.testing.assert_allclose(gradients[name].to_numpy(), value, rtol=1e-5, atol=1e-6)
            kinds = [type(pullback).__name__ for *_, pullback in calls]
            test.assertEqual(kinds[1:], ["_LinkedPullback"] * 5)

    def check_bound_chain(self, test: unittest.TestCase) -> None:
        """A VJP with its weights bound: every pullback writes the weights' gradients into the same Storages, reads
        weights written between steps, and differentiates a weight written between a forward call and its pullback as
        of the forward call."""
        rng = np.random.default_rng(13)
        n = 300
        weights = [rng.standard_normal(n).astype(np.float32) for _ in range(3)]
        stored = [vd.storage.from_numpy(weight) for weight in weights]
        supplied = vd.storage.zeros(dtype=vd.f32, shape=(n,))
        transformed = vd.ad.vjp(WeightedChain(), wrt=("source", "w0", "w1", "w2"), outputs=None)
        bound = transformed.bind(w0=stored[0], w1=stored[1], w2=stored[2], gradients={"w1": supplied})
        test.assertIs(bound.gradients["w1"], supplied)
        kinds = []
        for step in range(5):
            if step == 3:
                weights[0] = rng.standard_normal(n).astype(np.float32)
                stored[0][:] = weights[0]
            data = rng.standard_normal(n).astype(np.float32)
            factor = 1.5 + step
            output, pullback = bound(vd.storage.from_numpy(data), factor)
            kinds.append(type(pullback).__name__)
            w0, w1, w2 = weights
            np.testing.assert_allclose(output.to_numpy(), data * w0 * w1 * w2 * factor, rtol=1e-5)
            cotangent = rng.standard_normal(n).astype(np.float32)
            gradients = pullback(cotangent)
            expected = {
                "source": cotangent * factor * w0 * w1 * w2,
                "w0": cotangent * factor * data * w1 * w2,
                "w1": cotangent * factor * data * w0 * w2,
                "w2": cotangent * factor * data * w0 * w1,
            }
            for name, value in expected.items():
                np.testing.assert_allclose(gradients[name].to_numpy(), value, rtol=1e-5, atol=1e-6)
            for name in ("w0", "w1", "w2"):
                test.assertIs(gradients[name], bound.gradients[name])
        test.assertEqual(kinds[1:], ["_BoundPullback"] * 4)
        w0, w1, w2 = weights
        _, pullback = bound(vd.storage.from_numpy(np.ones(n, dtype=np.float32)), 1.0)
        stored[2][:] = np.zeros(n, dtype=np.float32)
        gradients = pullback(np.ones(n, dtype=np.float32))
        np.testing.assert_allclose(gradients["w0"].to_numpy(), w1 * w2, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(gradients["w2"].to_numpy(), w0 * w1, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(stored[2].to_numpy(), np.zeros(n, dtype=np.float32))
        output, pullback = bound(vd.storage.from_numpy(np.ones(n, dtype=np.float32)), 1.0)
        np.testing.assert_array_equal(output.to_numpy(), np.zeros(n, dtype=np.float32))
        np.testing.assert_allclose(pullback(np.ones(n, dtype=np.float32))["w0"].to_numpy(), np.zeros(n), atol=1e-6)
        with test.assertRaisesRegex(ValueError, "aliases a bound input"):
            transformed.bind(w0=stored[0], gradients={"w0": stored[0]})
        with test.assertRaisesRegex(TypeError, "names no forward"):
            transformed.bind(weight=stored[0])

    def check_checkpointed_chain(self, test: unittest.TestCase) -> None:
        """A checkpointed VJP keeps the residuals of the last layers that fit its memory budget and runs the others
        again at the pullback, with several pullbacks in flight; every budget differentiates as keeping all does."""
        rng = np.random.default_rng(23)
        n = 64
        weights = [rng.standard_normal(n).astype(np.float32) for _ in range(3)]
        stored = [vd.storage.from_numpy(weight) for weight in weights]

        def run(budget: int, policy: str = "balanced") -> Any:
            module = WeightedChain()
            module.checkpoint_memory_budget = budget
            transformed = vd.ad.vjp(module, wrt=("source", "w0", "w1", "w2"), outputs=None, planning_policy=policy)
            calls = []
            for index in range(3):
                data = rng.standard_normal(n).astype(np.float32)
                output, pullback = transformed(vd.storage.from_numpy(data), *stored, 0.5 + index)
                calls.append((data, 0.5 + index, output, pullback))
            w0, w1, w2 = weights
            for data, factor, output, pullback in reversed(calls):
                np.testing.assert_allclose(output.to_numpy(), data * w0 * w1 * w2 * factor, rtol=1e-5)
                cotangent = rng.standard_normal(n).astype(np.float32)
                gradients = pullback(cotangent)
                expected = {
                    "source": cotangent * factor * w0 * w1 * w2,
                    "w0": cotangent * factor * data * w1 * w2,
                    "w1": cotangent * factor * data * w0 * w2,
                    "w2": cotangent * factor * data * w0 * w1,
                }
                for name, value in expected.items():
                    np.testing.assert_allclose(gradients[name].to_numpy(), value, rtol=1e-5, atol=1e-6)
            return calls[-1][3]

        everything = run(1 << 40)
        plan = everything.checkpoint_plan
        kept = plan["logical_residual_bytes"]
        initial = plan["initial_state_bytes"]
        test.assertGreater(kept, 0)
        test.assertEqual(everything.logical_residual_bytes, kept)
        test.assertEqual(plan["restoration_bytes"], 0)
        test.assertEqual(everything.recomputation_factor, 1.0)
        nothing = run(initial)
        test.assertEqual(nothing.logical_residual_bytes, 0)
        test.assertEqual(nothing.checkpoint_plan["restoration_bytes"], kept)
        test.assertLessEqual(nothing.checkpoint_plan["peak_bytes"], initial)
        some = run(initial + kept - 1)
        test.assertLess(some.logical_residual_bytes, kept)
        test.assertGreater(some.recomputation_factor, 1.0)
        test.assertLess(some.recomputation_factor, nothing.recomputation_factor)
        test.assertEqual(run(1 << 40, "min_memory").recomputation_factor, 2.0)
        with test.assertRaisesRegex(ValueError, "checkpoint memory budget"):
            run(initial - 1)

        module = RepeatedChain()
        module.checkpoint_memory_budget = 1 << 40
        transformed = vd.ad.vjp(module, wrt=("source", "weight"), outputs=None, planning_policy="balanced")
        weight = rng.uniform(0.5, 1.5, n).astype(np.float32)
        calls = []
        for count in (1, 3, 2):
            data = rng.uniform(0.5, 1.5, n).astype(np.float32)
            calls.append((data, count, *transformed(vd.storage.from_numpy(data), vd.storage.from_numpy(weight), count)))
        for data, count, output, pullback in reversed(calls):
            test.assertEqual(pullback.recomputation_factor, 2.0)
            np.testing.assert_allclose(output.to_numpy(), data * weight ** (2 * count), rtol=1e-5)
            cotangent = rng.standard_normal(n).astype(np.float32)
            gradients = pullback(cotangent)
            for name, value in repeated_chain_gradients(data, weight, count, cotangent).items():
                np.testing.assert_allclose(gradients[name].to_numpy(), value, rtol=1e-4, atol=1e-5)


@contextlib.contextmanager
def linked_calls() -> Iterator[list[bool]]:
    """Records, per call of a Module specialization that has linked, whether its linked schedule ran it."""
    calls: list[bool] = []
    original = program_module._PrimalSpecialization._call_linked

    def counted(self: Any, inputs: Any) -> Any:
        result = original(self, inputs)
        calls.append(result is not None)
        return result

    with mock.patch.object(program_module._PrimalSpecialization, "_call_linked", counted):
        yield calls


class LinkedModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        vd.init(arch=vd.cpu)

    def test_float_constants_print_as_mlir_float_literals(self) -> None:
        from vernon_dsl.program_frontend.parser import _constant_attribute

        self.assertEqual(_constant_attribute(2.0), "2.0 : f64")
        self.assertEqual(_constant_attribute(1e20), "1.0e+20 : f64")
        self.assertEqual(_constant_attribute(-2.5e-7), "-2.5e-07 : f64")
        self.assertEqual(_constant_attribute(0.1), "0.1 : f64")
        self.assertEqual(_constant_attribute(float("inf")), "0x7FF0000000000000 : f64")

    def test_a_module_links_before_its_first_call_and_keeps_its_transients_in_the_arena(self) -> None:
        module = Doubled()
        with linked_calls() as calls:
            for n in (10, 1000):
                data = np.arange(n, dtype=np.float32)
                source = vd.storage.from_numpy(data)
                for _ in range(3):
                    np.testing.assert_array_equal(module(source, 1.5).to_numpy(), data * 3.0)
        # A new extent reuses the schedule with a larger arena.
        self.assertEqual(calls, [True] * 6)

    def test_a_module_with_two_outputs_links(self) -> None:
        module = FanOut()
        data = np.linspace(-1.0, 1.0, 50, dtype=np.float32)
        source = vd.storage.from_numpy(data)
        with linked_calls() as calls:
            for _ in range(2):
                outputs = module(source)
                np.testing.assert_array_equal(outputs.twice.to_numpy(), data * 2.0)
                np.testing.assert_array_equal(outputs.thrice.to_numpy(), data * 3.0)
        self.assertEqual(calls, [True, True])

    def test_a_static_output_allocation_links_and_is_zeroed_per_call(self) -> None:
        module = FixedSum()
        data = np.arange(16, dtype=np.float32)
        source = vd.storage.from_numpy(data)
        with linked_calls() as calls:
            for _ in range(3):
                np.testing.assert_array_equal(module(source).to_numpy(), data * 2.0)
        self.assertEqual(calls, [True] * 3)

    def test_a_strided_view_input_links(self) -> None:
        module = Doubled()
        data = np.arange(40, dtype=np.float32)
        source = vd.storage.from_numpy(data).view(shape=(10,), strides=(4,), offset=1, access="read")
        with linked_calls() as calls:
            for _ in range(2):
                np.testing.assert_array_equal(module(source, 1.0).to_numpy(), data[1::4] * 2.0)
        self.assertEqual(calls, [True, True])

    def test_a_vjp_links_its_forward_and_backward_graphs(self) -> None:
        transformed = vd.ad.vjp(ScaledChain(), wrt=("source",), outputs=None)
        data = np.arange(100, dtype=np.float32)
        cotangent = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        pullbacks = []
        for factor in (1.5, 2.0, 3.0):
            output, pullback = transformed(vd.storage.from_numpy(data), factor)
            pullbacks.append(type(pullback).__name__)
            np.testing.assert_allclose(output.to_numpy(), data * factor * factor, rtol=1e-6)
            np.testing.assert_allclose(pullback(cotangent)["source"].to_numpy(), cotangent * factor * factor, rtol=1e-6)
        self.assertEqual(pullbacks[1:], ["_LinkedPullback", "_LinkedPullback"])

    def test_a_linked_pullback_reads_inputs_written_since_its_forward_as_of_the_forward(self) -> None:
        rng = np.random.default_rng(5)
        values = [rng.standard_normal(8).astype(np.float32) for _ in range(4)]
        source, w0, w1, w2 = (vd.storage.from_numpy(value) for value in values)
        transformed = vd.ad.vjp(WeightedChain(), wrt=("source", "w0", "w1", "w2"), outputs=None)
        transformed(source, w0, w1, w2, 2.0)
        _, pullback = transformed(source, w0, w1, w2, 2.0)
        _, unwritten = transformed(source, w0, w1, w2, 2.0)
        self.assertEqual(type(pullback).__name__, "_LinkedPullback")
        source[:] = np.zeros(8, dtype=np.float32)
        w1.copy_from_numpy(np.full(8, 3.0, dtype=np.float32))
        data, v0, v1, v2 = values
        for result in (pullback(), pullback(), unwritten()):
            np.testing.assert_allclose(result["w0"].to_numpy(), 2.0 * data * v1 * v2, rtol=1e-5)
            np.testing.assert_allclose(result["source"].to_numpy(), 2.0 * v0 * v1 * v2, rtol=1e-5)
        np.testing.assert_array_equal(source.to_numpy(), np.zeros(8, dtype=np.float32))
        np.testing.assert_array_equal(w1.to_numpy(), np.full(8, 3.0, dtype=np.float32))
        _, pullback = transformed(source, w0, w1, w2, 2.0)
        np.testing.assert_allclose(pullback()["w2"].to_numpy(), np.zeros(8), atol=1e-6)

    def test_read_write_kernel_gradients_count_the_cotangent_once(self) -> None:
        total = vd.storage.from_numpy(np.zeros(8, dtype=np.float32))
        source = vd.storage.from_numpy(np.arange(8, dtype=np.float32))
        _, pullback = vd.ad.vjp(accumulate, wrt=("total", "source"), outputs=("total",))(total, source, grid=(1, 1, 1))
        cotangent = np.arange(1, 9, dtype=np.float32)
        gradients = pullback({"total": cotangent})
        np.testing.assert_array_equal(gradients["total"].to_numpy(), cotangent)
        np.testing.assert_array_equal(gradients["source"].to_numpy(), cotangent)

    def test_a_module_accumulating_in_place_twice_has_the_chain_rule_gradient(self) -> None:
        transformed = vd.ad.vjp(Doubled(), wrt=("source",), outputs=None)
        data = np.arange(64, dtype=np.float32)
        for factor in (1.5, 2.0):
            output, pullback = transformed(vd.storage.from_numpy(data), factor)
            np.testing.assert_allclose(output.to_numpy(), 2.0 * factor * data)
            np.testing.assert_allclose(pullback(np.full(64, 0.5, dtype=np.float32))["source"].to_numpy(), factor)

    def test_a_multi_layer_vjp_keeps_each_calls_residuals_in_its_own_block(self) -> None:
        WeightedChainChecks().check_weighted_chain(self)

    def test_a_bound_vjp_writes_the_gradients_of_its_bound_inputs_in_place(self) -> None:
        WeightedChainChecks().check_bound_chain(self)

    def test_a_checkpointed_vjp_runs_what_it_did_not_keep_again(self) -> None:
        WeightedChainChecks().check_checkpointed_chain(self)

    def test_a_vjp_of_kernels_with_tapes_keeps_each_calls_tapes_in_its_block(self) -> None:
        rng = np.random.default_rng(17)
        transformed = vd.ad.vjp(RepeatedChain(), wrt=("source", "weight"), outputs=None)
        n = 300
        weight = rng.uniform(0.5, 1.5, n).astype(np.float32)
        stored = vd.storage.from_numpy(weight)
        calls = []
        for count in (1, 2, 3, 2, 0):
            data = rng.uniform(0.5, 1.5, n).astype(np.float32)
            output, pullback = transformed(vd.storage.from_numpy(data), stored, count)
            calls.append((data, count, output, pullback))
        for data, count, output, pullback in reversed(calls):
            np.testing.assert_allclose(output.to_numpy(), data * weight ** (2 * count), rtol=1e-5)
            cotangent = rng.standard_normal(n).astype(np.float32)
            gradients = pullback(cotangent)
            for name, value in repeated_chain_gradients(data, weight, count, cotangent).items():
                np.testing.assert_allclose(gradients[name].to_numpy(), value, rtol=1e-4, atol=1e-5)
        self.assertEqual([type(pullback).__name__ for *_, pullback in calls[1:]], ["_LinkedPullback"] * 4)

    def test_a_bound_vjp_of_kernels_with_tapes_replays_its_own_forward(self) -> None:
        rng = np.random.default_rng(19)
        n = 200
        weight = rng.uniform(0.5, 1.5, n).astype(np.float32)
        bound = vd.ad.vjp(RepeatedChain(), wrt=("source", "weight"), outputs=None).bind(
            weight=vd.storage.from_numpy(weight)
        )
        kinds = []
        for count in (2, 3, 1, 2):
            data = rng.uniform(0.5, 1.5, n).astype(np.float32)
            output, pullback = bound(vd.storage.from_numpy(data), count)
            kinds.append(type(pullback).__name__)
            np.testing.assert_allclose(output.to_numpy(), data * weight ** (2 * count), rtol=1e-5)
            cotangent = rng.standard_normal(n).astype(np.float32)
            gradients = pullback(cotangent)
            for name, value in repeated_chain_gradients(data, weight, count, cotangent).items():
                np.testing.assert_allclose(gradients[name].to_numpy(), value, rtol=1e-4, atol=1e-5)
        self.assertEqual(kinds[1:], ["_BoundPullback"] * 3)

    def test_two_programs_link_as_one_graph(self) -> None:
        check_two_program_graph(self)


class CudaLinkedModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        require_available(probe_backend(backend_row(vd.cuda), BackendRequirements(gpu=True, compute=True)))

    def tearDown(self) -> None:
        vd.init(arch=vd.cpu)

    def test_a_module_links_and_writes_fresh_outputs_on_the_device(self) -> None:
        module = Doubled()
        data = np.arange(1000, dtype=np.float32)
        source = vd.storage.from_numpy(data)
        with linked_calls() as calls:
            for factor in (1.0, 2.0, 0.5):
                output = module(source, factor)
                self.assertTrue(output._control.coherence.device_dirty or factor == 1.0)
                np.testing.assert_array_equal(output.to_numpy(), data * 2.0 * factor)
        self.assertEqual(calls, [True] * 3)

    def test_a_vjp_pulls_back_on_the_device(self) -> None:
        transformed = vd.ad.vjp(ScaledChain(), wrt=("source",), outputs=None)
        data = np.arange(1000, dtype=np.float32)
        pullbacks = []
        for factor in (1.5, 2.0, 3.0):
            output, pullback = transformed(vd.storage.from_numpy(data), factor)
            pullbacks.append(type(pullback).__name__)
            np.testing.assert_allclose(output.to_numpy(), data * factor * factor, rtol=1e-6)
            np.testing.assert_allclose(pullback()["source"].to_numpy(), np.full(1000, factor * factor), rtol=1e-6)
        self.assertEqual(pullbacks[1:], ["_LinkedPullback", "_LinkedPullback"])

    def test_a_multi_layer_vjp_keeps_each_calls_residuals_in_its_own_block(self) -> None:
        WeightedChainChecks().check_weighted_chain(self)

    def test_a_bound_vjp_writes_the_gradients_of_its_bound_inputs_in_place(self) -> None:
        WeightedChainChecks().check_bound_chain(self)

    def test_a_checkpointed_vjp_runs_what_it_did_not_keep_again(self) -> None:
        WeightedChainChecks().check_checkpointed_chain(self)

    def test_two_programs_link_as_one_graph(self) -> None:
        check_two_program_graph(self)

    def test_a_module_relinks_nothing_when_its_extents_change(self) -> None:
        module = Doubled()
        with linked_calls() as calls:
            # Doubled runs 1024 lanes, so each extent fits its fixed grid.
            for n in (10, 1000, 10, 500):
                data = np.arange(n, dtype=np.float32)
                source = vd.storage.from_numpy(data)
                outputs = [module(source, factor) for factor in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
                for factor, output in zip((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), outputs, strict=True):
                    np.testing.assert_array_equal(output.to_numpy(), data * 2.0 * factor)
        self.assertEqual(calls, [True] * 24)

    def test_an_in_place_kernel_vjp_differentiates_its_source(self) -> None:
        total = vd.storage.from_numpy(np.zeros(8, dtype=np.float32))
        source = vd.storage.from_numpy(np.arange(8, dtype=np.float32))
        _, pullback = vd.ad.vjp(accumulate, wrt=("source",), outputs=("total",))(total, source, grid=(1, 1, 1))
        cotangent = np.arange(1, 9, dtype=np.float32)
        np.testing.assert_array_equal(pullback({"total": cotangent})["source"].to_numpy(), cotangent)

    def test_an_in_place_kernel_vjp_wrt_its_updated_storage_is_rejected_with_a_diagnostic(self) -> None:
        total = vd.storage.from_numpy(np.zeros(8, dtype=np.float32))
        source = vd.storage.from_numpy(np.arange(8, dtype=np.float32))
        with self.assertRaisesRegex(Exception, "internal Storage adjoint"):
            vd.ad.vjp(accumulate, wrt=("total", "source"), outputs=("total",))(total, source, grid=(1, 1, 1))


if __name__ == "__main__":
    unittest.main()
