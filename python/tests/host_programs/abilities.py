"""A second Host module of the systems asset: it imports types, externs, and a Host class from `systems`, and names
a helper as `systems` names an exported function."""

import vernon_dsl as vd
from systems import Entity, Health, Timer, read, write
from vernon_dsl import TensorStorage, dyn, f32, f64, i32


@vd.func(shared=True)
def scaled(value: f32, factor: f32) -> f32:
    return value * factor * f32(2.0)


@vd.host(meta=vd.meta(role="system", order=20))
class Poison:
    """Drains its target's health each time its timer fires."""

    rate: f32 = 0.5
    timer: Timer
    ticks: i32 = 0
    name: str = "poison"
    drains: list[f32]
    """What each tick drained, oldest first."""
    hits: dict[Entity, f32]
    """What the poison drained from each target."""
    levels: dict[str, i32]
    """The strengthening each name was given at."""

    def update(self, target: Entity, dt: f64) -> f32:
        if not self.timer.tick(dt):
            return f32(0.0)
        health = read(target, Health)
        drained = min(scaled(self.rate, f32(dt)), health.current)
        write(target, Health(health.current - drained, health.maximum))
        self.ticks += 1
        self.drains.append(drained)
        self.hits[target] = self.hits.get(target, f32(0.0)) + drained
        return drained

    def total_drained(self) -> f32:
        total = f32(0.0)
        for drained in self.drains:
            total += drained
        return total

    def strengthen(self) -> i32:
        """Doubles the rate the first time; returns how many times the poison is now strengthened."""

        if self.name == "poison":
            self.name = "strong " + self.name
            self.rate *= f32(2.0)
            self.levels[self.name] = len(self.levels) + 1
            return 1
        return 2 if self.name == "strong poison" else 0


@vd.host
class Trail:
    """Where a poison spread, as a grid of what it left behind."""

    cells: TensorStorage[f32, (dyn, dyn)]
    marks: TensorStorage[i32, (2,)]

    def spread(self, rows: i32, columns: i32) -> i32:
        self.cells = vd.zeros(dtype=f32, shape=(rows, columns))
        for row in range(rows):
            for column in range(columns):
                self.cells[row, column] = f32(row * 10 + column)
        self.marks[0] = rows
        self.marks[-1] = columns
        return len(self.cells)

    def total(self) -> f32:
        rows, columns = self.cells.shape
        total = f32(0.0)
        for row in range(rows):
            for column in range(columns):
                total += self.cells[row, column]
        return total
