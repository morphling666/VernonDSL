"""The Host binding of tally, a small C++ library: the schema vernon_dsl.bindgen generates its C API and glue from.

Regenerate with `python -m vernon_dsl.bindgen generate python/tests/bindgen_sample/tally.py --output-dir
python/tests/bindgen_sample/generated`.
"""

from enum import IntEnum
from typing import TypeVar

import vernon_dsl as vd
from vernon_dsl import Vector, f32, i32, i64, u64

T = TypeVar("T")

# Every status function but a pure one asks tally::api::allow first: a counter labelled "frozen" refuses changes.
vd.library("tally", version=1, meta=vd.meta(cpp_namespace="tally::api", effect_check="allow"))


@vd.handle(meta=vd.meta(cpp="tally::Counter", header="tally/counter.h"))
class Counter:
    """A running tally; destroyed with counter_destroy."""


@vd.handle(meta=vd.meta(cpp="tally::Cursor", header="tally/counter.h", depends_on="Counter"))
class Cursor:
    """A reader of a Counter's history; destroying the Counter destroys it."""


@vd.handle(meta=vd.meta(cpp="tally::Ticket", header="tally/counter.h", convert="convert"))
class Ticket:
    """A counted amount's position in a Counter's history; tally::Ticket is its value, not an object."""


@vd.enum
class Mode(IntEnum):
    SUM = 0
    MAX = 1


@vd.struct(shared=True)
class Stats:
    count: i32
    total: i64
    recent: Vector[f32, 3]


@vd.struct(shared=True, meta=vd.meta(cpp="tally::Sample", header="tally/counter.h"))
class Sample:
    """An amount and its weight; tally::Sample has this layout."""

    amount: i32
    weight: f32


@vd.struct(shared=True)
class Marker:
    """No fields: counter_store keeps one as a flag."""


@vd.struct(shared=True, meta=vd.meta(cpp="tally::Range", header="tally/counter.h", convert="convert"))
class Range:
    """Inclusive bounds; tally::Range holds them as int64_t, so the library converts."""

    low: i32
    high: i32


@vd.extern(effects="io", abi="status")
def counter_create(mode: Mode) -> Counter: ...


@vd.extern(effects="io", abi="status", meta=vd.meta(destroys="counter"))
def counter_destroy(counter: Counter) -> None: ...


@vd.extern(effects="write", abi="status")
def counter_add(counter: Counter, amount: i32) -> None:
    """Counts amount; a negative amount fails with status 10."""
    ...


@vd.extern(effects="read", abi="status")
def counter_stats(counter: Counter) -> Stats: ...


@vd.extern(effects="write", abi="status", meta=vd.meta(unfreezes=True))
def counter_label(counter: Counter, label: str) -> i32:
    """Names the counter and returns the name's length in bytes; a frozen counter allows it, to be renamed."""
    ...


@vd.extern(effects="write", abi="status")
def counter_add_all(counter: Counter, amounts: list[i32]) -> None: ...


@vd.extern(effects="write", abi="status")
def counter_add_weighted(counter: Counter, samples: dict[i32, Sample]) -> f32:
    """Counts each sample's amount and returns the sum of key * weight, in the dict's order."""
    ...


@vd.extern(effects="read", abi="status")
def counter_name(counter: Counter) -> str: ...


@vd.extern(effects="read", abi="status")
def counter_history(counter: Counter) -> list[i32]: ...


@vd.extern(effects="read", abi="status")
def counter_histogram(counter: Counter) -> dict[i32, i32]:
    """How often each amount was counted, by first occurrence."""
    ...


@vd.extern(effects="read", abi="status")
def counter_samples(counter: Counter) -> list[Sample]: ...


@vd.extern(effects="read", abi="status")
def counter_range(counter: Counter) -> Range: ...


@vd.extern(effects="read", abi="status")
def counter_in_range(counter: Counter, range: Range) -> i32: ...


@vd.extern(effects="write", abi="status")
def counter_store(counter: Counter, value: T) -> None:
    """Keeps one value of each Struct type with the counter, keyed by the type's name; a frozen counter still
    stores a Marker."""
    ...


@vd.extern(effects="read", abi="status")
def counter_load(counter: Counter, type: type[T]) -> T: ...


@vd.extern(effects="read", abi="status")
def counter_has(counter: Counter, type: type[T]) -> bool: ...


@vd.extern(effects="read", abi="status")
def counter_clamp(counter: Counter, ranges: list[Range]) -> list[Range]:
    """Each range cut to the counted amounts' range."""
    ...


@vd.extern(effects="read", abi="status")
def counter_ticket(counter: Counter) -> Ticket:
    """The ticket of the last counted amount."""
    ...


@vd.extern(effects="read", abi="status")
def counter_tickets(counter: Counter) -> list[Ticket]: ...


@vd.extern(effects="read", abi="status")
def counter_redeem(counter: Counter, tickets: list[Ticket]) -> i32:
    """The sum of the tickets' amounts; a ticket past the history fails with status 14."""
    ...


@vd.extern(effects="pure")
def ticket_position(ticket: Ticket) -> i32: ...


@vd.extern(effects="io", abi="status")
def cursor_create(counter: Counter) -> Cursor: ...


@vd.extern(effects="io", abi="status", meta=vd.meta(destroys="cursor"))
def cursor_destroy(cursor: Cursor) -> None: ...


@vd.extern(effects="write", abi="status")
def cursor_next(cursor: Cursor) -> i32:
    """The next amount of the counter's history; past its end fails with status 15."""
    ...


@vd.extern(effects="read")
def live_cursors() -> i32: ...


@vd.extern(effects="read")
def counter_exists(bits: u64) -> bool:
    """Whether a Counter handle's bits name a live Counter."""
    ...


@vd.extern(effects="io", abi="status")
def counter_destroy_all() -> None:
    """Destroys every Counter, and their Cursors with them."""
    ...


@vd.extern(effects="pure")
def scaled_total(stats: Stats, factor: f32) -> f32: ...


@vd.extern(effects="read")
def live_counters() -> i32:
    """How many Counters exist."""
    ...
