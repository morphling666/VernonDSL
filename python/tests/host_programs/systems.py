from typing import Annotated, TypeVar

import vernon_dsl as vd
from vernon_dsl import Vector, f32, f64, i32


@vd.handle(c_type="DemoEntity")
class Entity:
    pass


@vd.struct(shared=True)
class Transform:
    position: Vector[f32, 3]
    scale: f32


@vd.extern(symbol="demo_get_transform", effects="read", abi="status")
def get_transform(entity: Entity) -> Transform: ...


@vd.extern(symbol="demo_set_transform", effects="write", abi="status")
def set_transform(entity: Entity, value: Transform) -> None: ...


@vd.host
class Timer:
    """Fires once per period of accumulated time."""

    period: Annotated[f64, vd.meta(min=0.0, unit="s")] = 0.5
    elapsed: f64 = 0.0
    fired: i32

    @vd.host(effects="pure")
    def tick(self, dt: f64) -> bool:
        self.elapsed += dt
        if self.elapsed >= self.period:
            self.elapsed -= self.period
            self.fired += 1
            return True
        return False


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


T = TypeVar("T")


@vd.struct(shared=True, meta=vd.meta(role="component", name="demo:Health"))
class Health:
    """Hit points."""

    current: Annotated[f32, vd.meta(min=0.0)]
    """What remains."""
    maximum: f32


@vd.extern(symbol="demo_read", effects="read", abi="status")
def read(entity: Entity, type: type[T]) -> T: ...


@vd.extern(symbol="demo_write", effects="write", abi="status")
def write(entity: Entity, type: type[T], value: T) -> None: ...


@vd.host
class Regen:
    rate: f32 = 1.0

    def update(self, target: Entity, dt: f64) -> f32:
        health = read(target, Health)
        healed = min(health.current + self.rate * f32(dt), health.maximum)
        write(target, Health, Health(healed, health.maximum))
        return healed + read(target, Transform).scale
