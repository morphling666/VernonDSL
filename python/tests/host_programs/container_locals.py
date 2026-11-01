"""list and dict locals: declared in a function, used in place, and released on every exit."""

from enum import IntEnum

import vernon_dsl as vd
from vernon_dsl import f32, i32


@vd.enum
class Mode(IntEnum):
    IDLE = 0
    WALK = 1
    RUN = 4


@vd.struct(shared=True)
class Hit:
    damage: f32
    critical: bool


@vd.extern(symbol="locals_score", effects="read", abi="status")
def score(value: i32) -> i32: ...


@vd.host(effects="pure")
def tally(first: Mode, second: Mode) -> i32:
    counts: dict[Mode, i32] = {}
    counts[first] = 1
    if second in counts:
        counts[second] += 1
    else:
        counts[second] = 1
    return counts[first] * 10 + i32(len(counts))


@vd.host
class Arena:
    hits: list[Hit]
    names: list[i32]

    @vd.host(effects="pure")
    def tally(self, first: Mode, second: Mode) -> i32:
        counts: dict[Mode, i32] = {}
        counts[first] = 1
        if second in counts:
            counts[second] += 1
        else:
            counts[second] = 1
        return counts[first] * 10 + i32(len(counts))

    @vd.host(effects="pure")
    def evens(self, count: i32) -> i32:
        values: list[i32] = []
        for index in range(count):
            if index % 2 == 0:
                values.append(index * index)
        total = 0
        for value in values:
            total += value
        if len(values) > 2:
            total += values.pop() * 1000
        return total + i32(len(values))

    @vd.host(effects="pure")
    def words(self, which: i32) -> i32:
        seen: dict[str, i32] = {}
        text = "red"
        for step in range(5):
            if (step + which) % 3 == 0:
                text = "green"
            elif (step + which) % 3 == 1:
                text = f"blue{step % 2}"
            seen[text] = seen.get(text, 0) + 1
        total = 0
        for word, times in seen.items():
            total = total * 10 + times + i32(len(word))
        del seen["red" if "red" in seen else text]
        return total * 10 + i32(len(seen))

    @vd.host(effects="pure")
    def batches(self, rounds: i32) -> i32:
        total = 0
        for round in range(rounds):
            # Declared in the loop, so each round starts from an empty dict.
            batch: dict[i32, i32] = {}
            for item in range(round + 1):
                batch[item % 2] = batch.get(item % 2, 0) + item
            total = total * 10 + i32(len(batch))
        return total

    @vd.host(effects="pure")
    def strongest(self, count: i32) -> f32:
        hits: list[Hit] = []
        for index in range(count):
            hits.append(Hit(f32(index) * 1.5, index % 2 == 1))
        best = f32(0.0)
        for hit in hits:
            if hit.critical and hit.damage > best:
                best = hit.damage
        return best

    @vd.host(effects="pure")
    def early(self, stop: i32) -> i32:
        values: list[i32] = []
        for index in range(10):
            values.append(index)
            if index == stop:
                return i32(len(values))
        values.clear()
        return -1

    @vd.host(effects="pure")
    def missing(self, key: i32) -> i32:
        counts: dict[i32, i32] = {}
        counts[1] = 10
        return counts[key]

    @vd.host(effects="read")
    def scored(self, value: i32) -> i32:
        values: list[i32] = []
        values.append(value)
        return score(values[0])

    def critical_total(self) -> f32:
        critical: list[Hit] = []
        for hit in self.hits:
            if hit.critical:
                critical.append(hit)
        total = f32(0.0)
        for hit in critical:
            total += hit.damage
        self.names.append(i32(len(critical)))
        return total
