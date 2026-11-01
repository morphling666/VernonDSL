"""Keyword construction: `Cls(field=value)` sets named fields over the defaults and `Point(1, y=2)` builds a Struct,
alike in Python mode and AOT; a class instance named this way moves into its field."""

from __future__ import annotations

import ctypes
import gc
import textwrap
from pathlib import Path

import pytest
from host_aot import LinkedHost, toolchain
from test_host_strs import _CountingAllocator
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostModule, load_host_module

_PROGRAM = """
import vernon_dsl as vd
from vernon_dsl import f64, i32


@vd.extern(symbol="crew_log")
def log(message: str, value: i32) -> None: ...


@vd.struct(shared=True)
class Extent:
    width: i32
    height: i32


@vd.host
class Badge:
    count: i32 = 7
    label: str = "badge"


@vd.host
class Post:
    name: str
    size: Extent
    badge: Badge
    scale: f64 = 2.0
    tags: list[i32]

    def report(self) -> i32:
        log(self.name, self.size.width * 100 + self.size.height)
        log(self.badge.label, self.badge.count)
        return i32(self.scale * 10.0) + len(self.tags)


@vd.host
def build(width: i32, label: str) -> i32:
    badge = Badge(label=f"{label}!")
    post = Post(name=label, size=Extent(width, height=width * 2), badge=badge)
    total = post.report()
    other = Post(badge=Badge(count=3), scale=0.5)
    other.name = other.badge.label
    total += other.report()
    post.badge = Badge()
    total += post.report()
    return total


@vd.host
class Station:
    post: Post
    size: Extent = Extent(height=1, width=2)

    def staff(self, label: str) -> i32:
        self.post = Post(name=self.post.name + label, size=self.size, badge=Badge(count=len(label)))
        return self.post.report()
"""

_STAFFED = ["a", "bc", ""]


def _registry(logged: list[tuple[str, int]]) -> ExternRegistry:
    return ExternRegistry().bind("crew_log", lambda message, value: logged.append((str(message), int(value))))


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "crew.py"
    source.write_text(_PROGRAM, encoding="utf-8")
    return source


def _python_mode(source: Path) -> tuple[list[int], list[tuple[str, int]]]:
    logged: list[tuple[str, int]] = []
    module = HostModule(source, _registry(logged), eager=False)
    counting = _CountingAllocator()
    module.runtime.allocator.value = ctypes.addressof(counting.record)
    results = [int(module.call("build", 4, "crew"))]
    station = module.namespace.Station()
    results += [int(station.staff(label)) for label in _STAFFED]
    del station
    gc.collect()
    module.runtime.allocator.value = None
    assert counting.live == {}
    return results, logged


def test_keywords_set_fields_over_the_defaults(tmp_path: Path) -> None:
    results, logged = _python_mode(_source(tmp_path))
    assert logged[:6] == [
        ("crew", 408),
        ("crew!", 7),
        ("badge", 0),
        ("badge", 3),
        ("crew", 408),
        ("badge", 7),
    ]
    assert results[0] == 20 + 5 + 20
    assert results[1:] == [20, 20, 20]
    assert logged[6:] == [("a", 201), ("badge", 1), ("abc", 201), ("badge", 2), ("abc", 201), ("badge", 0)]


@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_keyword_construction_runs_alike_in_python_mode_and_aot(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    source = _source(tmp_path)
    python, python_logged = _python_mode(source)
    logged: list[tuple[str, int]] = []
    library = LinkedHost(load_programs(source), tmp_path / "aot", _registry(logged))
    table = library.table
    counting = _CountingAllocator()
    table.allocator[0] = ctypes.addressof(counting.record)
    results = [int(library.call("build", 4, "crew"))]
    descriptor = next(
        table.types[index].contents
        for index in range(table.type_count)
        if table.types[index].contents.name == b"Station"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Station__init", address)
    results += [int(library.call("Station_staff", address, label)) for label in _STAFFED]
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)
    table.allocator[0] = None

    assert library.errors == []
    assert results == python
    assert logged == python_logged
    assert counting.live == {}, "the replaced instances, their strs and the temporaries are released"
    assert counting.mismatched == []


_PRELUDE = """
import vernon_dsl as vd
from vernon_dsl import i32


@vd.struct(shared=True)
class Extent:
    width: i32
    height: i32


@vd.host
class Badge:
    count: i32 = 7
    tags: list[i32]


@vd.host
class Post:
    badge: Badge
"""


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("@vd.host\ndef f() -> i32:\n    return Badge(size=1).count\n", "Badge has no field 'size'"),
        ("@vd.host\ndef f() -> i32:\n    return Badge(1).count\n", "names the fields it sets"),
        ("@vd.host\ndef f() -> i32:\n    return Badge(count=1.5).count\n", "cannot pass"),
        (
            "@vd.host\ndef f() -> i32:\n    tags: list[i32] = []\n    return Badge(tags=tags).count\n",
            "starts empty",
        ),
        ("@vd.host\ndef f() -> i32:\n    return Extent(1).width\n", "'height' is unset"),
        ("@vd.host\ndef f() -> i32:\n    return Extent(1, width=2).width\n", "sets field 'width' twice"),
        (
            "@vd.host\ndef f() -> None:\n    badge = Badge()\n    post = Post(badge=badge)\n    badge.count = 1\n",
            "'badge' was moved",
        ),
        ("@vd.host\ndef f(badge: Badge) -> None:\n    post = Post(badge=badge)\n", "moved into field 'badge'"),
    ],
    ids=["unknown", "positional", "type", "container", "missing", "twice", "use-after-move", "borrowed"],
)
def test_keyword_construction_diagnostics(tmp_path: Path, body: str, message: str) -> None:
    source = tmp_path / "broken.py"
    source.write_text(_PRELUDE + "\n\n" + textwrap.dedent(body), encoding="utf-8")
    with pytest.raises(CompileError) as error:
        load_host_module(source)
    assert message in str(error.value)
