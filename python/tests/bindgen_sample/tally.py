"""The Host binding of tally, a small C++ library: the schema vernon_dsl.bindgen generates its C API and glue from.

Regenerate with `python -m vernon_dsl.bindgen generate python/tests/bindgen_sample/tally.py --output-dir
python/tests/bindgen_sample/generated`.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TypeVar

import vernon_dsl as vd
from vernon_dsl import Vector, f32, i32, i64, u64

T = TypeVar("T")

# Every status function but a pure one asks tally::api::allow first: a counter labelled "frozen" refuses changes.
vd.library("tally", version=1, meta=vd.meta(cpp_namespace="tally::api", effect_check="allow"))


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


@vd.handle(meta=vd.meta(cpp="tally::Ticket", header="tally/counter.h", convert="convert"))
class Ticket:
    """A counted amount's position in a Counter's history; tally::Ticket is its value, not an object."""

    @vd.extern(effects="pure")
    def position(self) -> i32: ...


@vd.handle(meta=vd.meta(cpp="tally::Counter", header="tally/counter.h"))
class Counter:
    """A running tally; destroyed with counter.destroy()."""

    @staticmethod
    @vd.extern(effects="io", abi="status")
    def create(mode: Mode) -> Counter: ...

    @vd.extern(effects="io", abi="status", meta=vd.meta(destroys="counter"))
    def destroy(self) -> None: ...

    @vd.extern(effects="write", abi="status")
    def add(self, amount: i32) -> None:
        """Counts amount; a negative amount fails with status 10."""
        ...

    @vd.extern(effects="read", abi="status")
    def stats(self) -> Stats: ...

    @vd.extern(effects="write", abi="status", meta=vd.meta(unfreezes=True))
    def label(self, label: str) -> i32:
        """Names the counter and returns the name's length in bytes; a frozen counter allows it, to be renamed."""
        ...

    @vd.extern(effects="write", abi="status")
    def add_all(self, amounts: list[i32]) -> None: ...

    @vd.extern(effects="write", abi="status")
    def add_weighted(self, samples: dict[i32, Sample]) -> f32:
        """Counts each sample's amount and returns the sum of key * weight, in the dict's order."""
        ...

    @vd.extern(effects="read", abi="status")
    def name(self) -> str: ...

    @vd.extern(effects="read", abi="status")
    def history(self) -> list[i32]: ...

    @vd.extern(effects="read", abi="status")
    def histogram(self) -> dict[i32, i32]:
        """How often each amount was counted, by first occurrence."""
        ...

    @vd.extern(effects="read", abi="status")
    def samples(self) -> list[Sample]: ...

    @vd.extern(effects="read", abi="status")
    def range(self) -> Range: ...

    @vd.extern(effects="read", abi="status")
    def in_range(self, range: Range) -> i32: ...

    @vd.extern(effects="write", abi="status")
    def store(self, value: T) -> None:
        """Keeps one value of each Struct type with the counter, keyed by the type's name; a frozen counter still
        stores a Marker."""
        ...

    @vd.extern(effects="read", abi="status")
    def load(self, type: type[T]) -> T: ...

    @vd.extern(effects="read", abi="status")
    def has(self, type: type[T]) -> bool: ...

    @vd.extern(effects="read", abi="status")
    def clamp(self, ranges: list[Range]) -> list[Range]:
        """Each range cut to the counted amounts' range."""
        ...

    @vd.extern(effects="read", abi="status")
    def ticket(self) -> Ticket:
        """The ticket of the last counted amount."""
        ...

    @vd.extern(effects="read", abi="status")
    def tickets(self) -> list[Ticket]: ...

    @vd.extern(effects="read", abi="status")
    def redeem(self, tickets: list[Ticket]) -> i32:
        """The sum of the tickets' amounts; a ticket past the history fails with status 14."""
        ...

    @staticmethod
    @vd.extern(effects="read", abi="status")
    def exists(bits: u64) -> bool:
        """Whether a Counter handle's bits name a live Counter."""
        ...

    @staticmethod
    @vd.extern(effects="io", abi="status")
    def destroy_all() -> None:
        """Destroys every Counter, and their Cursors with them."""
        ...


@vd.handle(meta=vd.meta(cpp="tally::Cursor", header="tally/counter.h", depends_on="Counter"))
class Cursor:
    """A reader of a Counter's history; destroying the Counter destroys it."""

    @staticmethod
    @vd.extern(effects="io", abi="status")
    def create(counter: Counter) -> Cursor: ...

    @vd.extern(effects="io", abi="status", meta=vd.meta(destroys="cursor"))
    def destroy(self) -> None: ...

    @vd.extern(effects="write", abi="status")
    def next(self) -> i32:
        """The next amount of the counter's history; past its end fails with status 15."""
        ...

    @vd.extern(effects="read", abi="status")
    def counter_bits(self) -> u64:
        """The bits of the live Counter handle the cursor reads."""
        ...


@vd.extern(effects="read", abi="status")
def live_cursors() -> i32: ...


@vd.extern(effects="pure")
def scaled_total(stats: Stats, factor: f32) -> f32: ...


@vd.extern(effects="read", abi="status")
def live_counters() -> i32:
    """How many Counters exist."""
    ...
