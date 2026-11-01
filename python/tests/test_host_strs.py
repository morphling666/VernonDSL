"""Host str locals and f-strings: Python mode runs the f-strings themselves, and compiled code formats alike."""

from __future__ import annotations

import ctypes
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from host_aot import LinkedHost, toolchain
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostModule, load_host_module

PROGRAMS = Path(__file__).parent / "host_programs"
MESSAGES = PROGRAMS / "messages.py"
sys.path.insert(0, str(PROGRAMS))


def _native_available() -> bool:
    try:
        from vernon_dsl import _native

        return hasattr(_native, "_host_runtime_symbols")
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")

_SCRIPT: list[tuple[str, tuple[Any, ...]]] = [
    ("report", (7, 0.1, 12.345, False)),
    ("report", (2**40, 1.5, 0.0, True)),
    ("numbers", (42, 0xDEADBEEF, -12345, 0.1)),
    ("numbers", (-7, 2**64 - 1, 9, -2.5e-7)),
    ("numbers", (0, 0, 0, 3.0e20)),
    ("titled", (False,)),
    ("same_captain", (0,)),
    ("same_captain", (1,)),
    ("titled", (True,)),
    ("same_captain", (0,)),
    ("count_words", (0,)),
    ("count_words", (1,)),
    ("count_words", (2,)),
    ("count_words", (1,)),
    ("count_words", (0,)),
    ("summary", ()),
    ("forget_while_listing", ()),
    ("summary", ()),
    ("field_then_call", ()),
    ("early_exit", (True,)),
    ("early_exit", (False,)),
    ("unicode", ()),
]


def _registry(logged: list[tuple[str, int]]) -> ExternRegistry:
    def stamp(message: Any) -> int:
        logged.append((f"stamp {message}", -1))
        return len(str(message))

    return (
        ExternRegistry()
        .bind("messages_log", lambda message, value: logged.append((str(message), int(value))))
        .bind("messages_stamp", stamp)
    )


def _outcome(result: Any) -> Any:
    return None if result is None else int(result)


class _CountingAllocator:
    """A `VhAllocator` over Python buffers that counts what compiled code takes and gives back."""

    _ALLOCATE = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64)
    _RELEASE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64)

    class Record(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("user", ctypes.c_void_p),
            ("allocate", ctypes.c_void_p),
            ("release", ctypes.c_void_p),
        ]

    def __init__(self) -> None:
        self.live: dict[int, tuple[Any, int]] = {}
        self.allocations = 0
        self.mismatched: list[int] = []
        self._callbacks = (self._ALLOCATE(self._allocate), self._RELEASE(self._release))
        self.record = self.Record(
            ctypes.sizeof(self.Record),
            None,
            ctypes.cast(self._callbacks[0], ctypes.c_void_p),
            ctypes.cast(self._callbacks[1], ctypes.c_void_p),
        )

    def _allocate(self, _user: Any, size: int, alignment: int) -> int:
        buffer = ctypes.create_string_buffer(size + alignment)
        address = ctypes.addressof(buffer)
        address += -address % alignment
        self.live[address] = (buffer, size)
        self.allocations += 1
        return address

    def _release(self, _user: Any, pointer: int | None, size: int, _alignment: int) -> None:
        if pointer is None:
            return
        _, allocated = self.live.pop(pointer)
        if allocated != size:
            self.mismatched.append(pointer)


def _python_mode() -> tuple[list[Any], list[tuple[str, int]], Any]:
    logged: list[tuple[str, int]] = []
    module = HostModule(MESSAGES, _registry(logged), eager=False)
    logbook = module.namespace.Logbook()
    return [_outcome(getattr(logbook, name)(*args)) for name, args in _SCRIPT], logged, logbook


def test_python_mode_formats_with_python() -> None:
    results, logged, logbook = _python_mode()

    # An f32 formats as Python formats the double it widens to.
    assert logged[0] == ("ship 7 at 0.10000000149011612 m/s, fuel 12.35 (under way, docked=False)", 71)
    assert logged[2][0] == (
        "[   42|42   |  42   |00042|+42] deadbeef DEADBEEF 33653337357 11011110101011011011111011101111"
        " -12345 -12345 -  12345 0.10000000149011612 0.100   1.00e-01 0.1****** 1.000000E-01"
    )
    assert results[6:10] == [1, 0, 48, 0]
    assert ("before", 17) in logged, "the field is read before the call that rewrites it"
    assert str(logbook.captain) == "--Grand Admiral of the Fleet--"
    assert ("forgot calm", 2) in logged
    assert logged[-1] == ("[   größe] [éégrößeéé]", 5)


@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_strs_format_alike_in_python_mode_and_aot(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    python, python_logged, _ = _python_mode()
    logged: list[tuple[str, int]] = []
    library = LinkedHost(load_programs(MESSAGES), tmp_path, _registry(logged))
    table = library.table
    counting = _CountingAllocator()
    table.allocator[0] = ctypes.addressof(counting.record)
    descriptor = next(
        table.types[index].contents
        for index in range(table.type_count)
        if table.types[index].contents.name == b"Logbook"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Logbook__init", address)
    aot = [_outcome(library.call(f"Logbook_{name}", address, *args)) for name, args in _SCRIPT]
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)
    table.allocator[0] = None

    assert library.errors == []
    assert logged == python_logged
    assert aot == python
    assert counting.allocations > 20
    assert counting.live == {}, "every owned str and temporary is released, on early returns too"
    assert counting.mismatched == []


_BADGES = """
import vernon_dsl as vd
from vernon_dsl import i32


@vd.extern(symbol="badges_log")
def log(message: str, value: i32) -> None: ...


@vd.host
def shout(word: str, times: i32) -> i32:
    log(word, times)
    word = word + "!"
    i = 0
    while i < times:
        word += "!"
        i += 1
    log(word, len(word))
    return len(word)


@vd.host
class Badge:
    name: str = "anon"

    def rename(self, name: str) -> None:
        self.name = "[" + name + "]"
        log(name, len(self.name))

    def run(self, count: i32) -> i32:
        self.rename(self.name)
        self.rename(f"crew {count}")
        if count > 2:
            return shout(self.name, count)
        return shout("größe", count)
"""


def _badges(tmp_path: Path) -> Path:
    source = tmp_path / "badges.py"
    source.write_text(_BADGES, encoding="utf-8")
    return source


def test_str_parameters_are_copies_of_the_callers_text(tmp_path: Path) -> None:
    logged: list[tuple[str, int]] = []
    registry = ExternRegistry().bind("badges_log", lambda message, value: logged.append((str(message), int(value))))
    module = HostModule(_badges(tmp_path), registry, eager=False)
    badge = module.namespace.Badge()
    assert badge.run(1) == len("größe!!")
    assert badge.name == "[crew 1]"
    assert logged[0] == ("anon", len("[anon]"))
    assert module.call("shout", "hey", 2) == len("hey!!!")


@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_str_parameters_pass_alike_in_python_mode_and_aot(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    source = _badges(tmp_path)
    script = [1, 3, 0]

    python_logged: list[tuple[str, int]] = []
    registry = ExternRegistry().bind(
        "badges_log", lambda message, value: python_logged.append((str(message), int(value)))
    )
    badge = HostModule(source, registry, eager=False).namespace.Badge()
    python = [badge.run(count) for count in script]

    logged: list[tuple[str, int]] = []
    registry = ExternRegistry().bind("badges_log", lambda message, value: logged.append((str(message), int(value))))
    library = LinkedHost(load_programs(source), tmp_path / "aot", registry)
    table = library.table
    counting = _CountingAllocator()
    table.allocator[0] = ctypes.addressof(counting.record)
    descriptor = next(
        table.types[index].contents for index in range(table.type_count) if table.types[index].contents.name == b"Badge"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Badge__init", address)
    aot = [library.call("Badge_run", address, count) for count in script]
    shouted = library.call("shout", "hey", 2)
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)
    table.allocator[0] = None

    assert library.errors == []
    assert [int(result) for result in aot] == python
    assert int(shouted) == len("hey!!!")
    assert logged[: len(python_logged)] == python_logged
    assert counting.live == {}, "parameter copies and argument temporaries are released"
    assert counting.mismatched == []


def _compile_error(tmp_path: Path, body: str) -> str:
    source = tmp_path / "bad_strs.py"
    source.write_text(
        "import vernon_dsl as vd\n"
        "from vernon_dsl import f32, i32\n\n\n"
        "@vd.extern(symbol='bad_log')\n"
        "def log(message: str) -> None: ...\n\n\n" + textwrap.dedent(body),
        encoding="utf-8",
    )
    with pytest.raises(CompileError) as raised:
        load_host_module(source)
    return str(raised.value)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("@vd.host\nclass Bad:\n    def f(self, x: i32) -> None:\n        log(f'{x!r}')\n", "conversion"),
        ("@vd.host\nclass Bad:\n    def f(self, x: i32) -> None:\n        log(f'{x=}')\n", "conversion"),
        ("@vd.host\nclass Bad:\n    def f(self, x: i32) -> None:\n        log(f'{x:q}')\n", "format"),
        ("@vd.host\nclass Bad:\n    def f(self, x: f32) -> None:\n        log(f'{x:d}')\n", "format"),
        ("@vd.host\nclass Bad:\n    def f(self, x: i32) -> None:\n        log(f'{x:{x}}')\n", "replacement"),
        ("@vd.host\nclass Bad:\n    def f(self, x: bool) -> None:\n        log(f'{x:>5}')\n", "bool"),
        ("@vd.host\nclass Bad:\n    def f(self) -> None:\n        s = 'a'\n        log(f'{s:+}')\n", "format"),
        ("@vd.host\nclass Bad:\n    def f(self) -> str:\n        return 'a'\n", "str"),
        ("@vd.host\nclass Bad:\n    def f(self) -> None:\n        s = 'a'\n        s -= 'b'\n", "+="),
        ("@vd.host\nclass Bad:\n    def f(self) -> None:\n        s = 'a'\n        log(s - s)\n", "only with +"),
        ("@vd.func\ndef helper(x: i32) -> i32:\n    s = f'{x}'\n    return x\n", "str"),
        ("@vd.func(shared=True)\ndef helper(s: str) -> i32:\n    return 0\n", "str"),
        ("@vd.host\ndef named(s: str) -> str:\n    return s\n", "str"),
    ],
)
def test_str_misuse_is_reported(tmp_path: Path, body: str, message: str) -> None:
    assert message in _compile_error(tmp_path, body)
