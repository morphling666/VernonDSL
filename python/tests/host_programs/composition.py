"""Host classes composed of Host classes that own lists, dicts, strs and TensorStorages."""

import vernon_dsl as vd
from vernon_dsl import TensorStorage, f32, i32


@vd.host
class Bag:
    """Owns one of each memory-owning kind."""

    items: list[i32]
    counts: dict[i32, i32]
    label: str = "bag"
    grid: TensorStorage[f32, (vd.dyn,)]

    def add(self, value: i32) -> i32:
        self.items.append(value)
        self.counts[value % 3] = self.counts.get(value % 3, 0) + 1
        self.label = f"{self.label}+"
        if len(self.grid) < len(self.items):
            self.grid = vd.zeros(dtype=f32, shape=(len(self.items) * 2,))
        self.grid[len(self.items) - 1] = f32(value)
        return len(self.items)

    def total(self) -> i32:
        total = 0
        for value in self.items:
            total += value
        return total


@vd.host
class Shelf:
    """A class nested two levels down."""

    bag: Bag
    moves: i32 = 0


@vd.host
class Owner:
    """Composes classes through fields and reaches into them directly and through their methods."""

    left: Bag
    right: Bag
    shelf: Shelf

    def fill(self, count: i32) -> i32:
        for value in range(count):
            self.left.add(value)
            self.right.add(value * 10)
            self.shelf.bag.add(value + 100)
            self.shelf.moves += 1
        return self.left.total() + self.right.total() + self.shelf.bag.total()

    def direct(self) -> i32:
        self.left.items.append(1000)
        self.shelf.bag.counts[7] = 70
        del self.right.counts[0]
        return len(self.left.items) + self.shelf.bag.counts[7] + len(self.right.counts)

    def text(self) -> i32:
        return len(self.left.label) * 100 + len(self.shelf.bag.label)

    def cells(self) -> f32:
        return self.left.grid[2] + self.right.grid[2] + f32(len(self.shelf.bag.grid))

    def reset(self) -> i32:
        self.left.items.clear()
        self.right.counts.clear()
        return len(self.left.items) + len(self.right.counts) + self.shelf.moves
