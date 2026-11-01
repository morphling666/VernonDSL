"""IntEnums at every place a Host value goes: parameters, results, Struct fields, dict keys, externs."""

from enum import IntEnum

import vernon_dsl as vd
from vernon_dsl import Tuple, f32, i32, i64


@vd.enum(c_type="DemoMode", meta=vd.meta(cpp="demo::Mode"))
class Mode(IntEnum):
    """How a body moves."""

    IDLE = 0
    WALK = 1
    RUN = 4
    BACK = -2
    JOG = 1


@vd.enum
class Light(IntEnum):
    RED = 0
    GREEN = 1


@vd.struct(shared=True)
class Body:
    mode: Mode
    speed: f32


@vd.extern(symbol="demo_mode_speed", effects="pure")
def mode_speed(mode: Mode) -> f32: ...


@vd.extern(symbol="demo_next_mode", effects="pure")
def next_mode(mode: Mode) -> Mode: ...


@vd.host(effects="pure")
def faster(mode: Mode) -> Mode:
    if mode == Mode.IDLE:
        return Mode.WALK
    if mode != Mode.RUN:
        return Mode.RUN
    return mode


@vd.host(effects="pure")
def code(mode: Mode) -> i64:
    return i64(mode) * 10


@vd.host(effects="pure")
def from_code(value: i32) -> bool:
    return Mode.RUN == next_mode(Mode.WALK) and value == i32(Mode.RUN)


@vd.host(effects="pure")
def walk(speed: f32) -> Body:
    return Body(Mode.JOG, speed + mode_speed(Mode.WALK))


@vd.host(effects="pure")
def body_speed(body: Body) -> f32:
    if body.mode == Mode.BACK:
        return -body.speed
    return body.speed


@vd.host
class Tally:
    counts: dict[Mode, i32]
    last: Mode = Mode.RUN

    @vd.host(effects="write")
    def add(self, mode: Mode) -> i32:
        if mode in self.counts:
            self.counts[mode] += 1
        else:
            self.counts[mode] = 1
        self.last = mode
        return self.counts[mode]


@vd.host(effects="pure")
def lights(light: Light) -> Tuple[Light, Mode]:
    return (light, Mode.BACK)
