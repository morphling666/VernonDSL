import vernon_dsl as vd
from vernon_dsl import Vector, f32, i32, i64, u64


@vd.handle(c_type="DemoEntity")
class Entity:
    pass


@vd.struct(shared=True, c_type="DemoTransform")
class Transform:
    position: Vector[f32, 3]
    scale: f32


@vd.extern(symbol="demo_find", effects="read")
def find(name: str) -> Entity: ...


@vd.extern(symbol="demo_get_transform", effects="read", abi="status")
def get_transform(entity: Entity) -> Transform: ...


@vd.extern(symbol="demo_set_transform", effects="write", abi="status")
def set_transform(entity: Entity, value: Transform) -> None: ...


@vd.extern(symbol="demo_log", effects="io")
def log(message: str, value: f32) -> None: ...


@vd.func(shared=True)
def lift(position: Vector[f32, 3], height: f32) -> Vector[f32, 3]:
    return position + Vector([0.0, height, 0.0])


@vd.host(effects="write")
def raise_entity(height: f32, steps: i32) -> f32:
    entity = find("player")
    if entity == Entity():
        return -1.0
    transform = get_transform(entity)
    position = transform.position
    for _ in range(steps):
        position = lift(position, height)
    set_transform(entity, Transform(position, transform.scale * 2.0))
    return position.y


@vd.host(effects="pure")
def integer_rules(a: i32, b: i32) -> Vector[i32, 2]:
    return Vector([a // b, a % b])


@vd.host(effects="pure")
def wide(a: i64, b: u64) -> i64:
    total: i64 = a
    i = 0
    while i < 3:
        total = total * i64(2)
        i += 1
    return total + i64(b)


@vd.host(effects="pure")
def bad_effect() -> f32:
    return 1.0
