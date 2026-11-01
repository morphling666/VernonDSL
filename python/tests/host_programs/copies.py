"""Explicit copies: `.copy()` of lists, dicts and storages of values, and `copy.deepcopy` of class instances and of
containers of them. Each copy is independent of its source."""

from __future__ import annotations

import copy

import vernon_dsl as vd
from vernon_dsl import TensorStorage, dyn, f32, i32


@vd.host
class Part:
    weight: i32 = 1
    tags: list[i32]
    name: str = "part"


@vd.host(effects="pure")
def cloned(part: Part) -> Part:
    return copy.deepcopy(part)


@vd.host
class Shelf:
    parts: list[Part]
    spare: Part
    counts: list[i32]
    names: dict[str, i32]
    indexed: dict[i32, Part]
    heights: TensorStorage[f32, (dyn,)]

    def fill(self, count: i32) -> i32:
        for index in range(count):
            part = Part()
            part.weight = index + 1
            part.tags.append(index * 10)
            self.parts.append(part)
            self.counts.append(index)
        self.names["a"] = 1
        self.names["bb"] = 2
        self.spare.tags.append(5)
        self.indexed[3] = Part()
        self.heights = vd.zeros(dtype=f32, shape=(count,))
        self.heights[0] = 2.5
        return len(self.parts)

    def copy_values(self) -> i32:
        counts = self.counts.copy()
        counts.append(100)
        names = self.names.copy()
        names["ccc"] = 3
        total = len(counts) * 1000 + len(self.counts) * 100 + len(names) * 10 + len(self.names)
        self.counts = counts.copy()
        return total

    def copy_parts(self) -> i32:
        parts = copy.deepcopy(self.parts)
        parts[0].weight += 50
        parts[0].tags.append(7)
        kept = self.parts[0].weight * 100 + len(self.parts[0].tags) * 10
        self.parts = copy.deepcopy(parts)
        return kept + self.parts[0].weight + len(self.parts[0].tags) * 1000

    def copy_spare(self) -> i32:
        spare = copy.deepcopy(self.spare)
        spare.tags.append(6)
        spare.name = "copied"
        self.parts.append(copy.deepcopy(spare))
        self.spare = spare
        return len(self.parts) * 100 + len(self.spare.tags) * 10 + len(self.spare.name)

    def copy_indexed(self) -> i32:
        indexed = copy.deepcopy(self.indexed)
        indexed[3].weight = 9
        indexed[4] = copy.deepcopy(self.parts[0])
        self.indexed = copy.deepcopy(indexed)
        return len(self.indexed) * 100 + self.indexed[3].weight * 10 + self.indexed[4].weight

    def copy_heights(self) -> f32:
        heights = self.heights.copy()
        heights[0] = 10.0
        kept = self.heights[0]
        self.heights = heights.copy()
        doubled: TensorStorage[f32, (dyn,)] = copy.deepcopy(self.heights)
        doubled[1] = 4.0
        return kept + self.heights[0] + doubled[1] + self.heights[1]

    def recopy(self) -> i32:
        self.parts = copy.deepcopy(self.parts)
        self.counts = self.counts.copy()
        self.spare = copy.deepcopy(self.spare)
        return len(self.parts) + len(self.counts) + len(self.spare.tags)

    def clone_first(self) -> i32:
        twin = cloned(self.parts[0])
        twin.weight += 1
        return twin.weight * 10 + self.parts[0].weight
