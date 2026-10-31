"""What every system of the demo game shares: the entity handle, component Structs, and the engine's externs."""

from typing import Annotated, TypeVar

import vernon_dsl as vd
from vernon_dsl import Vector, f32

T = TypeVar("T")


@vd.handle(c_type="DemoEntity")
class Entity:
    pass


@vd.struct(shared=True)
class Transform:
    position: Vector[f32, 3]
    scale: f32


@vd.struct(shared=True, meta=vd.meta(role="component", name="demo:Health"))
class Health:
    """Hit points."""

    current: Annotated[f32, vd.meta(min=0.0)]
    """What remains."""
    maximum: f32


@vd.extern(symbol="demo_get_transform", effects="read", abi="status")
def get_transform(entity: Entity) -> Transform: ...


@vd.extern(symbol="demo_set_transform", effects="write", abi="status")
def set_transform(entity: Entity, value: Transform) -> None: ...


@vd.extern(symbol="demo_read", effects="read", abi="status")
def read(entity: Entity, type: type[T]) -> T: ...


@vd.extern(symbol="demo_write", effects="write", abi="status")
def write(entity: Entity, type: type[T], value: T) -> None: ...
