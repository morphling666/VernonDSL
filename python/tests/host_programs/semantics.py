"""Host programs at the edges where integer, float, and conversion rules differ between languages."""

import vernon_dsl as vd
from vernon_dsl import (
    Matrix,
    Tensor,
    Tuple,
    Vector,
    clamp,
    cos,
    cross,
    dot,
    exp,
    f32,
    f64,
    i32,
    i64,
    log,
    norm,
    normalize,
    sin,
    sqrt,
    u32,
    u64,
)


@vd.struct(shared=True)
class Pair:
    low: i32
    high: f64
    flag: bool


@vd.extern(symbol="semantics_count", effects="write")
def count(value: i32) -> bool: ...


@vd.host(effects="pure")
def integer_division(a: i32, b: i32) -> Vector[i32, 2]:
    return Vector([a // b, a % b])


@vd.host(effects="pure")
def unsigned_division(a: u32, b: u32) -> Vector[u32, 2]:
    return Vector([a // b, a % b])


@vd.host(effects="pure")
def wide_division(a: i64, b: i64) -> Vector[i64, 2]:
    return Vector([a // b, a % b])


@vd.host(effects="pure")
def wrapping(a: i32, b: i32) -> Vector[i32, 3]:
    return Vector([a + b, a - b, a * b])


@vd.host(effects="pure")
def true_division(a: i32, b: i32) -> f32:
    return a / b


@vd.host(effects="pure")
def float_remainder(a: f32, b: f32) -> f32:
    return a % b


@vd.host(effects="pure")
def double_remainder(a: f64, b: f64) -> f64:
    return a % b


@vd.host(effects="pure")
def to_integers(value: f32) -> Tuple[i32, u32, i64, u64]:
    return Tuple(i32(value), u32(value), i64(value), u64(value))


@vd.host(effects="pure")
def conversions(a: i32, b: u64) -> Tuple[u64, i64, f32, f64, bool, i32]:
    return Tuple(u64(a), i64(b), f32(b), f64(a), bool(a), i32(b))


@vd.host(effects="pure")
def to_bool(value: f32) -> bool:
    return bool(value)


@vd.host(effects="pure")
def float_compare(a: f32, b: f32) -> Vector[i32, 6]:
    return Vector([i32(a == b), i32(a != b), i32(a < b), i32(a <= b), i32(a > b), i32(a >= b)])


@vd.host(effects="pure")
def unsigned_compare(a: u32, b: u32) -> bool:
    return a < b


@vd.host(effects="pure")
def selection(a: f32, b: f32) -> Vector[f32, 3]:
    return Vector([min(a, b), max(a, b), clamp(a, 0.0, 1.0)])


@vd.host(effects="pure")
def negation(a: f32, b: i32) -> Tuple[f32, i32, i32]:
    return Tuple(-a, -b, abs(b))


@vd.host(effects="pure")
def rounding(a: f32, b: f32) -> Vector[f32, 4]:
    return Vector([a + b, a * b, a / b, sqrt(abs(a))])


@vd.host(effects="pure")
def vectors(a: Vector[f32, 3], b: Vector[f32, 3]) -> Tuple[f32, Vector[f32, 3], Vector[f32, 3], f32]:
    return Tuple(dot(a, b), cross(a, b), normalize(a), norm(b))


@vd.host(effects="pure")
def broadcast(a: Vector[f32, 3], s: f32) -> Vector[f32, 3]:
    return a * s + 1.0


@vd.host(effects="pure")
def matrix_index(m: Matrix[f32, 2, 3], row: i32, column: i32) -> f32:
    return m[row, column]


@vd.host(effects="pure")
def rank3_index(t: Tensor[f32, 2, 2, 2], i: i32, j: i32, k: i32) -> f32:
    return t[i, j, k]


@vd.host(effects="pure")
def loops(limit: i32, step: i32) -> Vector[i32, 3]:
    total = 0
    iterations = 0
    for i in range(limit, -limit, step):
        iterations += 1
        if i == 3:
            continue
        if i == -5:
            break
        total += i
    else:
        total += 1000
    j = 0
    while j < limit:
        j += 2
    else:
        j += 100
    return Vector([total, iterations, j])


@vd.host(effects="write")
def short_circuit(a: i32, b: i32) -> Vector[i32, 2]:
    left = 0
    if count(a) and count(b):
        left = 1
    right = 0
    if count(a) or count(b):
        right = 1
    return Vector([left, right])


@vd.host(effects="pure")
def choose(flag: bool, a: f32, b: f32) -> f32:
    return a if flag else b * 2.0


@vd.func(shared=True)
def make_pair(low: i32, high: f64) -> Pair:
    return Pair(low, high, low > 0)


@vd.host(effects="pure")
def structs(low: i32, high: f64) -> Pair:
    pair = make_pair(low, high)
    return Pair(pair.low * 2, pair.high / 2.0, not pair.flag)


@vd.host(effects="pure")
def replaced(low: i32, high: f64) -> Pair:
    pair = make_pair(low, high)
    return vd.replace(pair, flag=not pair.flag, low=pair.low + 1)


@vd.host(effects="pure")
def updated(v: Vector[f32, 3], index: i32, value: f32) -> Vector[f32, 3]:
    changed = v.at[index].set(value).at[-1].add(10.0)
    return changed + v * 0.0


@vd.host(effects="pure")
def sliced(m: Matrix[f32, 3, 4], i: i32) -> Tuple[Vector[f32, 2], Matrix[f32, 2, 2], f32]:
    window = m[1, i : i + 2]
    corner = m[:2, -2:]
    changed = m.at[1:3, i : i + 2].set(corner * 10.0).at[:, -1].add(100.0).at[0, ::2].set(-1.0)
    return window, corner[::-1, ::-1], changed[2, 3] + changed[0, 2]


@vd.host(effects="pure")
def transcendental(a: f32) -> Vector[f32, 5]:
    return Vector([sin(a), cos(a), exp(a), log(abs(a) + 1.0), pow(abs(a), 1.5)])


@vd.host(effects="pure")
def bits(a: i32, b: i32) -> Vector[i32, 6]:
    return Vector([a & b, a | b, a ^ b, ~a, a << b, a >> b])


@vd.host(effects="pure")
def unsigned_bits(a: u32, b: u32) -> Vector[u32, 3]:
    return Vector([a << b, a >> b, ~a])


@vd.host(effects="pure")
def wide_bits(a: u64, b: u64) -> Vector[u64, 2]:
    mixed = a
    mixed ^= mixed >> b
    return Vector([mixed, a << b])


@vd.host(effects="pure")
def float_floor_division(a: f32, b: f32) -> f32:
    return a // b


@vd.host(effects="pure")
def vector_masks(a: Vector[f32, 3], b: Vector[f32, 3]) -> Tuple[Vector[f32, 3], bool, bool, f32]:
    return Tuple(vd.select(a < b, a, b), vd.any(a < b), vd.all(a == b), vd.select(a[0] < 0.0, -1.0, 1.0))


@vd.host(effects="pure")
def integer_vectors(a: Vector[i32, 3], b: Vector[i32, 3]) -> Tuple[i32, Vector[i32, 3], Vector[i32, 3]]:
    return Tuple(dot(a, b), cross(a, b), vd.select(a > b, a, b))


@vd.host(effects="pure")
def literal_widths() -> Tuple[u64, i64, u32]:
    return Tuple(u64(18446744073709551615), i64(-9223372036854775807) - i64(1), u32(4294967295))
