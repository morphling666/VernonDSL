from __future__ import annotations

import unittest

from vernon_dsl.frontend.abi import attribute_layout
from vernon_dsl.frontend.model import ConcreteType


class AttributeAbiTests(unittest.TestCase):
    @staticmethod
    def _layout(dtype: str, shape: tuple[int, ...]):
        value_type = ConcreteType(
            "tensor",
            "Tensor",
            (ConcreteType("scalar", dtype), *shape),
        )
        return attribute_layout(value_type, lambda _: ())

    def test_rank_does_not_change_flat_leaf_plan(self) -> None:
        flat = self._layout("f32", (10,))
        ranked = self._layout("f32", (2, 5))
        self.assertEqual(flat.leaves, ranked.leaves)
        self.assertEqual(
            tuple(leaf.component_count for leaf in ranked.leaves),
            (4, 4, 2),
        )

    def test_dtype_controls_components_per_location(self) -> None:
        self.assertEqual(
            tuple(leaf.component_count for leaf in self._layout("f64", (5,)).leaves),
            (2, 2, 1),
        )
        self.assertEqual(
            tuple(leaf.component_count for leaf in self._layout("f16", (9,)).leaves),
            (4, 4, 1),
        )

    def test_all_numeric_attribute_dtypes_are_planned(self) -> None:
        for dtype in ("i32", "u32", "f16", "f32", "f64"):
            with self.subTest(dtype=dtype):
                layout = self._layout(dtype, (2, 3, 5))
                self.assertGreater(layout.location_span, 0)
                self.assertEqual(layout.leaves[0].location_offset, 0)

    def test_non_numeric_or_dynamic_shapes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "numeric vertex attribute"):
            self._layout("bool", (4,))
        with self.assertRaisesRegex(ValueError, "positive static shape"):
            self._layout("f32", (2, 0))


if __name__ == "__main__":
    unittest.main()
