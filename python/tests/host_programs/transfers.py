"""Externs taking lists and dicts as spans and returning owned strs, lists and dicts, run alike in Python mode and
AOT."""

import vernon_dsl as vd
from vernon_dsl import f32, i32


@vd.struct(shared=True)
class Sample:
    weight: f32
    count: i32


@vd.extern(symbol="transfers_total", effects="read")
def total(values: list[i32], weights: dict[i32, f32]) -> f32: ...


@vd.extern(symbol="transfers_samples", effects="read")
def sample_total(samples: list[Sample]) -> i32: ...


@vd.extern(symbol="transfers_name", effects="read", abi="status")
def name(id: i32) -> str: ...


@vd.extern(symbol="transfers_range", effects="read", abi="status")
def make_range(count: i32) -> list[i32]: ...


@vd.extern(symbol="transfers_table", effects="read", abi="status")
def table(count: i32) -> dict[i32, Sample]: ...


@vd.host
class Inventory:
    ids: list[i32]
    weights: dict[i32, f32]
    label: str = "none"
    samples: dict[i32, Sample]

    def fill(self, count: i32) -> i32:
        self.ids = make_range(count)
        for id in self.ids:
            self.weights[id] = f32(id) * f32(0.5)
        if count > 1:
            del self.weights[self.ids[0]]
        return len(self.ids)

    def weigh(self) -> f32:
        return total(self.ids, self.weights)

    def local_spans(self, count: i32) -> f32:
        values = make_range(count)
        values.append(100)
        weights: dict[i32, f32] = {}
        weights[7] = f32(2.0)
        return total(values, weights)

    def rename(self, id: i32) -> i32:
        self.label = name(id)
        return len(self.label)

    def rename_failing(self, id: i32) -> i32:
        # A failed call leaves the field as it was.
        self.label = name(-id)
        return len(self.label)

    def label_length(self) -> i32:
        return len(self.label)

    def measured(self) -> f32:
        # len is no call that could change the span, so it may follow it.
        return total(self.ids, self.weights) + f32(len(self.ids))

    def tabulate(self, count: i32) -> i32:
        self.samples = table(count)
        kept: list[Sample] = []
        for sample in self.samples.values():
            kept.append(sample)
        return sample_total(kept) + len(self.samples)

    def repeated(self, rounds: i32) -> i32:
        # Each call receives into the same records, which the previous round's move left empty.
        seen = 0
        for step in range(rounds):
            text = name(step)
            seen += len(text)
            values = make_range(step)
            seen += len(values)
        return seen
