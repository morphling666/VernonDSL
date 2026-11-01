"""Host class values: borrowed parameters, owned locals and results, moves, lists and dicts of classes, and a tree
whose nodes hold a list of their own class."""

from __future__ import annotations

import vernon_dsl as vd
from vernon_dsl import i32


@vd.host
class Item:
    weight: i32 = 1
    tags: list[i32]
    label: str = "item"

    @vd.host(effects="pure")
    def heavier(self, amount: i32) -> i32:
        self.weight += amount
        self.tags.append(amount)
        return self.weight


@vd.host
class Node:
    """A tree: each node owns its children."""

    value: i32 = 0
    children: list[Node]

    @vd.host(effects="pure")
    def add(self, value: i32) -> i32:
        child = Node()
        child.value = value
        self.children.append(child)
        return len(self.children)

    @vd.host(effects="pure")
    def total(self) -> i32:
        total = self.value
        for child in self.children:
            total += child.total()
        return total

    @vd.host(effects="pure")
    def depth(self) -> i32:
        deepest = 0
        for child in self.children:
            deepest = max(deepest, child.depth())
        return deepest + 1


@vd.host(effects="pure")
def make_item(weight: i32) -> Item:
    item = Item()
    item.weight = weight
    item.tags.append(weight * 2)
    return item


@vd.host(effects="pure")
def weigh(item: Item, extra: i32) -> i32:
    return item.heavier(extra) + len(item.tags)


@vd.host
class Bag:
    items: list[Item]
    named: dict[i32, Item]
    best: Item
    tree: Node

    @vd.host(effects="pure")
    def fill(self, count: i32) -> i32:
        for index in range(count):
            self.items.append(make_item(index + 1))
        self.named[7] = make_item(70)
        self.named[8] = Item()
        return len(self.items) + len(self.named)

    @vd.host(effects="pure")
    def total(self) -> i32:
        total = 0
        for item in self.items:
            total += item.weight + len(item.tags)
        for key, item in self.named.items():
            total += key * item.weight
        return total

    @vd.host(effects="pure")
    def bump(self) -> i32:
        for item in self.items:
            item.heavier(1)
        return weigh(self.items[0], 5) + weigh(self.named[7], 1)

    @vd.host(effects="pure")
    def swap_best(self) -> i32:
        best = self.items.pop()
        best.label = "best"
        self.best = best
        self.items[0] = Item()
        return self.best.weight + len(self.items)

    @vd.host(effects="pure")
    def take(self, key: i32) -> i32:
        taken = self.named.pop(key)
        weight = taken.weight
        self.items.append(taken)
        return weight

    @vd.host(effects="pure")
    def drop(self) -> i32:
        del self.named[8]
        self.items.pop()
        return len(self.named) * 100 + len(self.items)

    @vd.host(effects="pure")
    def grow(self) -> i32:
        self.tree.add(1)
        self.tree.add(2)
        self.tree.children[0].add(3)
        self.tree.children[0].children[0].add(4)
        return self.tree.total() * 10 + self.tree.depth()

    @vd.host(effects="pure")
    def alias(self) -> i32:
        best = self.best
        best.weight += 100
        return self.best.weight

    @vd.host(effects="pure")
    def local_items(self, count: i32) -> i32:
        kept: list[Item] = []
        for index in range(count):
            kept.append(make_item(index))
        total = 0
        for item in kept:
            total += item.weight
        moved = kept.pop()
        self.items.append(moved)
        return total * 10 + len(kept)

    @vd.host(effects="pure")
    def reset(self) -> i32:
        self.items.clear()
        self.named = {}
        self.best = Item()
        return len(self.items) + len(self.named) + self.best.weight
