from typing import Annotated

import vernon_dsl as vd
from components import Entity, Transform, get_transform, set_transform
from timing import Timer
from vernon_dsl import Vector, f32, f64, i32


@vd.host(meta=vd.meta(role="system", order=10, tags=("movement", "demo")))
class Mover:
    """Raises its target every update, recording its speed each time the timer fires."""

    target: Entity
    speed: Annotated[f32, vd.meta(min=0.0, max=10.0, step=-0.5)] = 2.0
    """How fast the target rises, in meters per second."""
    rest: Transform = Transform(Vector([0.0, 1.0, 0.0]), 1.0)
    history: Vector[f32, 4]
    timer: Timer
    updates: i32 = 0

    @vd.host(effects="write")
    def update(self, dt: f64) -> i32:
        if self.timer.tick(dt):
            self._record(self.speed)
        transform = get_transform(self.target)
        rise = Vector([0.0, self.speed * f32(dt), 0.0])
        set_transform(self.target, Transform(transform.position + rise, transform.scale))
        self.updates += 1
        return self.updates

    @vd.host(effects="pure")
    def _record(self, value: f32) -> None:
        self.history = self.history.at[self.timer.fired % 4].set(value)

    @vd.host(effects="pure")
    def height_after(self, seconds: f64) -> f32:
        return self.rest.position.y + self.speed * f32(seconds)


@vd.host(effects="pure")
def scaled(value: f32, factor: i32) -> f32:
    return value * f32(factor)
