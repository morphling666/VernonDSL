"""Host classes owning lists and strs: every container operation, run alike in Python mode and AOT."""

import vernon_dsl as vd
from vernon_dsl import Vector, f32, i32


@vd.struct(shared=True)
class Item:
    weight: f32
    count: i32


@vd.extern(symbol="inventory_log")
def log(message: str, value: f32) -> None: ...


@vd.host
class Inventory:
    """A named bag of items, with the weights it has seen."""

    owner: str = "nobody"
    items: list[Item]
    weights: list[f32] = []
    flags: list[bool]
    path: list[Vector[f32, 3]]

    def add(self, weight: f32, count: i32) -> i32:
        self.items.append(Item(weight, count))
        self.weights.append(weight)
        self.flags.append(count > 1)
        return len(self.items)

    def total_weight(self) -> f32:
        total = f32(0.0)
        for item in self.items:
            total += item.weight * f32(item.count)
        return total

    def heaviest(self) -> f32:
        best = f32(0.0)
        for weight in self.weights:
            best = max(best, weight)
        return best

    def weight_at(self, index: i32) -> f32:
        return self.weights[index]

    def count_at(self, index: i32) -> i32:
        return self.items[index].count

    def flagged(self) -> i32:
        count = 0
        for i in range(len(self.flags)):
            if self.flags[i]:
                count += 1
        return count

    def double_weights(self) -> None:
        for i in range(len(self.weights)):
            self.weights[i] *= f32(2.0)

    def drop_last(self) -> f32:
        self.items.pop()
        self.flags.pop()
        return self.weights.pop()

    def grow_while_iterating(self) -> i32:
        # Iteration reads the live length, so appending in the body visits the new items too.
        visited = 0
        for weight in self.weights:
            visited += 1
            if weight > f32(0.0) and len(self.weights) < 4:
                self.weights.append(-weight)
        return visited

    def walk(self, step: f32) -> f32:
        self.path.append(Vector([step, 0.0, 0.0]))
        self.path.append(Vector([step, step, 0.0]))
        last = self.path[-1]
        return last.x + last.y

    def rename(self, longer: bool) -> i32:
        if longer:
            self.owner = "a traveller with a long name"
        else:
            self.owner = "ann"
        log(self.owner, f32(len(self.items)))
        return 1 if self.owner == "ann" else 0

    def same_owner(self, other_name_is_ann: bool) -> bool:
        if other_name_is_ann:
            return self.owner == "ann"
        return self.owner != "ann"

    def empty(self) -> None:
        self.items = []
        self.weights.clear()
        self.flags = []
        self.path.clear()
