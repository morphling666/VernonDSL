"""Host code over the tally library, run in Python mode through its nanobind module and linked as an AOT object."""

import vernon_dsl as vd
from tally import Counter, Mode, Range, Sample, live_counters, scaled_total
from vernon_dsl import f32, i32, i64


@vd.struct(shared=True)
class Note:
    """A value the counter keeps for its user, which tally sees only as bytes."""

    id: i32
    weight: f32


@vd.host(effects="io")
def run(count: i32) -> i64:
    counter = Counter.create(Mode.SUM)
    for index in range(count):
        counter.add(index + 1)
    stats = counter.stats()
    length = counter.label("apples")
    counter.destroy()
    return stats.total * 1000 + i64(stats.count * 10 + length)


@vd.host(effects="io")
def peak(count: i32) -> f32:
    counter = Counter.create(Mode.MAX)
    for index in range(count):
        counter.add(index * 3)
    stats = counter.stats()
    counter.destroy()
    return scaled_total(stats, 0.5) + stats.recent[0]


@vd.host(effects="io")
def negative() -> i32:
    counter = Counter.create(Mode.SUM)
    counter.add(-4)
    return 0


@vd.host(effects="io")
def stale() -> i32:
    counter = Counter.create(Mode.SUM)
    counter.destroy()
    counter.add(1)
    return 0


@vd.host(effects="read")
def alive() -> i32:
    return live_counters()


@vd.host(effects="io")
def spans(count: i32) -> f32:
    counter = Counter.create(Mode.SUM)
    amounts: list[i32] = []
    for index in range(count):
        amounts.append(index * 2)
    counter.add_all(amounts)
    samples: dict[i32, Sample] = {}
    samples[3] = Sample(5, 0.5)
    samples[1] = Sample(7, 2.0)
    samples[3] = Sample(9, 1.5)
    weighted = counter.add_weighted(samples)
    stats = counter.stats()
    counter.destroy()
    return weighted * 100.0 + f32(stats.total)


@vd.host(effects="io")
def owned(count: i32) -> i64:
    counter = Counter.create(Mode.SUM)
    for index in range(count):
        counter.add(index % 3)
    counter.label("pears")
    name = counter.name()
    history: list[i32] = counter.history()
    counts: dict[i32, i32] = counter.histogram()
    samples = counter.samples()
    counter.destroy()
    checksum = i64(len(name)) * 1000000 + i64(len(history)) * 10000
    for amount, seen in counts.items():
        checksum += i64(amount * 100 + seen)
    weight = 0.0
    for sample in samples:
        weight += sample.weight
    return checksum * 100 + i64(weight * 2.0)


@vd.host(effects="io")
def nameless() -> i32:
    counter = Counter.create(Mode.SUM)
    name = counter.name()
    return len(name)


@vd.host(effects="io")
def ranged() -> i32:
    counter = Counter.create(Mode.SUM)
    amounts: list[i32] = []
    amounts.append(4)
    amounts.append(9)
    amounts.append(1)
    amounts.append(6)
    counter.add_all(amounts)
    bounds = counter.range()
    inside = counter.in_range(Range(bounds.low + 1, bounds.high))
    counter.destroy()
    return bounds.low * 10000 + bounds.high * 100 + inside


@vd.host(effects="io")
def stored(id: i32) -> f32:
    counter = Counter.create(Mode.SUM)
    before = counter.has(Note)
    counter.store(Note(id, 2.5))
    after = counter.has(Note)
    note = counter.load(Note)
    counter.destroy()
    flags = (0.0 if before else 1000.0) + (100.0 if after else 0.0)
    return flags + f32(note.id) * note.weight
