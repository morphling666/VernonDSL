from __future__ import annotations

import unittest

import numpy as np
import vernon_dsl as vd


@vd.kernel
def graph_add_one(values: vd.TensorView[vd.i32, 1, vd.read_write]) -> None:
    values[0] += 1


@vd.kernel
def graph_double(values: vd.TensorView[vd.i32, 1, vd.read_write]) -> None:
    values[0] *= 2


class ExecutionGraphTests(unittest.TestCase):
    @staticmethod
    def _backends() -> list[object]:
        result = [vd.cpu]
        for backend in (vd.cuda, vd.vulkan, vd.opengl, vd.opengles):
            try:
                vd.init(arch=backend)
            except RuntimeError:
                continue
            result.append(backend)
        return result

    def test_graph_identity_is_deterministic_and_ignores_owner_identity(self) -> None:
        vd.init(arch=vd.cpu)
        left = vd.storage.zeros(dtype=vd.i32, shape=(1,))
        right = vd.storage.zeros(dtype=vd.i32, shape=(1,))
        first = vd.ExecutionGraph()
        first.dispatch(graph_add_one, left)
        second = vd.ExecutionGraph()
        second.dispatch(graph_add_one, right)
        self.assertEqual(first.semantic_inputs, second.semantic_inputs)

    def test_unordered_alias_hazard_and_missing_transition_are_rejected(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.zeros(dtype=vd.i32, shape=(1,))
        unordered = vd.ExecutionGraph()
        unordered.dispatch(graph_add_one, values)
        unordered.dispatch(graph_double, values, depends_on=())
        with self.assertRaisesRegex(ValueError, "unordered graph hazard"):
            unordered.validate()

        missing = vd.ExecutionGraph(automatic_transitions=False)
        first = missing.dispatch(graph_add_one, values)
        missing.dispatch(graph_double, values, depends_on=(first,))
        with self.assertRaisesRegex(ValueError, "missing resource transition"):
            missing.validate()

        explicit = vd.ExecutionGraph(automatic_transitions=False)
        producer = explicit.dispatch(graph_add_one, values)
        transition = explicit.barrier(
            values,
            source="compute_write",
            destination="compute_read_write",
            depends_on=(producer,),
        )
        explicit.dispatch(graph_double, values, depends_on=(transition,))
        explicit.validate()

    def test_disjoint_unordered_views_are_valid(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.zeros(dtype=vd.i32, shape=(2,))
        left = values.view(shape=(1,), offset=0)
        right = values.view(shape=(1,), offset=1)
        graph = vd.ExecutionGraph()
        graph.dispatch(graph_add_one, left)
        graph.dispatch(graph_double, right, depends_on=())
        graph.validate()

    def test_graph_rejects_runtime_generation_change(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.zeros(dtype=vd.i32, shape=(1,))
        graph = vd.ExecutionGraph()
        graph.dispatch(graph_add_one, values)
        vd.init(arch=vd.cpu)
        with self.assertRaisesRegex(RuntimeError, "different runtime generation"):
            graph.run()

    def test_cooked_graph_asset_reloads_and_runs_with_new_bindings(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        try:
            vd.init(arch=vd.vulkan)
        except RuntimeError:
            self.skipTest("Vulkan runtime unavailable")
        specialization = vd.storage.zeros(dtype=vd.i32, shape=(1,))
        graph = vd.ExecutionGraph()
        graph.dispatch(graph_add_one, specialization)
        graph.dispatch(graph_double, specialization)
        with TemporaryDirectory() as directory:
            manifest = graph.cook(Path(directory) / "ordered", target="vulkan")
            self.assertTrue(manifest.is_file())
            self.assertTrue((manifest.parent / "artifacts").is_dir())

            asset = vd.load_execution_graph_asset(manifest)
            self.assertEqual(asset.parameters, ("node_0_values", "node_1_values"))
            self.assertEqual(len(asset.steps), 3)
            values = vd.storage.zeros(dtype=vd.i32, shape=(1,))
            asset.run({"node_0_values": values, "node_1_values": values})
            np.testing.assert_array_equal(values.to_numpy(), np.array((2,), dtype=np.int32))

    def test_multi_dispatch_native_order_and_slot_aliasing(self) -> None:
        for backend in self._backends():
            with self.subTest(backend=backend.name):
                vd.init(arch=backend)
                left = vd.storage.zeros(dtype=vd.i32, shape=(1,))
                right = vd.storage.zeros(dtype=vd.i32, shape=(1,))
                graph = vd.ExecutionGraph()
                graph.dispatch(graph_add_one, left)
                graph.dispatch(graph_double, left)
                graph.dispatch(graph_add_one, right)
                graph.run()
                np.testing.assert_array_equal(left.to_numpy(), np.array((2,), dtype=np.int32))
                np.testing.assert_array_equal(right.to_numpy(), np.array((1,), dtype=np.int32))

    def test_single_kernel_call_uses_graph_path(self) -> None:
        vd.init(arch=vd.cpu)
        values = vd.storage.zeros(dtype=vd.i32, shape=(1,))
        graph_add_one(values)
        np.testing.assert_array_equal(values.to_numpy(), np.array((1,), dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
