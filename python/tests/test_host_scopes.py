"""Scoped externs: `@extern(closed_by=end)` opens a scope that `with` closes by calling `end` on the opener's handle
however the block exits, in Python mode and in compiled code alike."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from host_aot import LinkedHost, toolchain
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostExecutionError, HostModule, load_host_module

_GUI = """
from __future__ import annotations

import vernon_dsl as vd
from vernon_dsl import i32


@vd.handle
class Ui:
    @staticmethod
    @vd.extern(symbol="gui_make")
    def make(id: i32) -> Ui: ...

    @vd.extern(symbol="gui_end", abi="status")
    def end(self) -> None: ...

    @vd.extern(symbol="gui_window", abi="status", closed_by=end)
    def window(self, name: str) -> i32: ...

    @vd.extern(symbol="gui_group", closed_by=end)
    def group(self, depth: i32) -> None: ...

    @vd.extern(symbol="gui_note", abi="status")
    def note(self, value: i32) -> None: ...
"""

_FRAME = """
import vernon_dsl as vd
from vernon_dsl import i32

from gui import Ui


@vd.host
def frame(id: i32, stop: i32) -> i32:
    ui = Ui.make(id)
    total = 0
    with ui.window("Scene") as tag:
        total += tag
        for depth in range(4):
            with ui.group(depth):
                if depth == stop:
                    break
                if depth == 1:
                    continue
                ui.note(depth)
        with Ui.make(id + 1).group(9), ui.group(10):
            ui.note(99)
    return total


@vd.host
def early(id: i32) -> i32:
    ui = Ui.make(id)
    with ui.window("Early") as tag:
        with ui.group(1):
            return tag + 1
    return 0


@vd.host
def failing(id: i32, value: i32) -> None:
    ui = Ui.make(id)
    with ui.window("Outer"):
        with ui.group(1):
            ui.note(value)
        ui.note(1)


@vd.host
def opening(id: i32) -> None:
    ui = Ui.make(id)
    with ui.window("Broken"):
        ui.note(5)
"""

# A window named "Broken" fails to open, a negative note fails, and every scope of Ui 13 fails to close.
_SCRIPT: list[tuple[str, tuple[int, ...]]] = [
    ("frame", (1, 9)),
    ("frame", (2, 2)),
    ("early", (3,)),
    ("failing", (4, -1)),
    ("failing", (13, 2)),
    ("opening", (5,)),
]


class _Gui:
    def __init__(self) -> None:
        self.log: list[tuple[Any, ...]] = []

    def registry(self) -> ExternRegistry:
        return (
            ExternRegistry()
            .bind("gui_make", self.make)
            .bind("gui_end", self.end)
            .bind("gui_window", self.window)
            .bind("gui_group", lambda ui, depth: self.log.append(("group", int(ui), int(depth))))
            .bind("gui_note", self.note)
        )

    def make(self, id: int) -> int:
        self.log.append(("make", int(id)))
        return int(id)

    def end(self, ui: Any) -> None:
        self.log.append(("end", int(ui)))
        if int(ui) == 13:
            raise HostExecutionError(8, "cannot close")

    def window(self, ui: Any, name: str) -> int:
        self.log.append(("window", int(ui), str(name)))
        if name == "Broken":
            raise HostExecutionError(5, "cannot open")
        return 10 * int(ui)

    def note(self, ui: Any, value: int) -> None:
        self.log.append(("note", int(ui), int(value)))
        if value < 0:
            raise HostExecutionError(6, "bad note")


def _sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: str = _FRAME, gui: str = _GUI) -> Path:
    (tmp_path / "gui.py").write_text(gui, encoding="utf-8")
    source = tmp_path / "frame.py"
    source.write_text(frame, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return source


def _run(call: Any, gui: _Gui) -> tuple[Any, list[tuple[Any, ...]]]:
    gui.log.clear()
    try:
        result = call()
    except HostExecutionError as error:
        result = ("failed", error.status)
    return (None if result is None else result), list(gui.log)


def _python_mode(source: Path) -> list[tuple[Any, list[tuple[Any, ...]]]]:
    gui = _Gui()
    module = HostModule(source, gui.registry(), eager=False)
    return [_run(lambda name=name, args=args: module.call(name, *args), gui) for name, args in _SCRIPT]


def test_scopes_close_on_every_exit_in_python_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = _python_mode(_sources(tmp_path, monkeypatch))
    groups = [("group", 1, depth) for depth in range(4)]
    assert results[0] == (
        10,
        [
            ("make", 1),
            ("window", 1, "Scene"),
            groups[0],
            ("note", 1, 0),
            ("end", 1),
            groups[1],
            ("end", 1),
            groups[2],
            ("note", 1, 2),
            ("end", 1),
            groups[3],
            ("note", 1, 3),
            ("end", 1),
            ("make", 2),
            ("group", 2, 9),
            ("group", 1, 10),
            ("note", 1, 99),
            ("end", 1),
            ("end", 2),
            ("end", 1),
        ],
    )
    # break at depth 2 closes that depth's group before leaving the loop.
    assert results[1][1][:10] == [
        ("make", 2),
        ("window", 2, "Scene"),
        ("group", 2, 0),
        ("note", 2, 0),
        ("end", 2),
        ("group", 2, 1),
        ("end", 2),
        ("group", 2, 2),
        ("end", 2),
        ("make", 3),
    ]
    # return closes both scopes after evaluating its value.
    assert results[2] == (31, [("make", 3), ("window", 3, "Early"), ("group", 3, 1), ("end", 3), ("end", 3)])
    # A failure inside closes every scope and stands.
    assert results[3] == (
        ("failed", 6),
        [("make", 4), ("window", 4, "Outer"), ("group", 4, 1), ("note", 4, -1), ("end", 4), ("end", 4)],
    )
    # A failing closer fails the call, and the scopes outside it still close.
    assert results[4] == (
        ("failed", 8),
        [("make", 13), ("window", 13, "Outer"), ("group", 13, 1), ("note", 13, 2), ("end", 13), ("end", 13)],
    )
    # A scope that fails to open has nothing to close.
    assert results[5] == (("failed", 5), [("make", 5), ("window", 5, "Broken")])


@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_scopes_close_alike_when_compiled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    source = _sources(tmp_path, monkeypatch)
    python = _python_mode(source)
    gui = _Gui()
    library = LinkedHost(load_programs(source), tmp_path / "aot", gui.registry())
    compiled = [_run(lambda name=name, args=args: library.call(name, *args), gui) for name, args in _SCRIPT]
    assert compiled == python


_BAD_CLOSER = _GUI.replace("def end(self) -> None: ...", "def end(self) -> i32: ...").replace(
    'symbol="gui_end", abi="status"', 'symbol="gui_end"'
)


@pytest.mark.parametrize(
    ("body", "message", "gui"),
    [
        ('ui.window("x")', "'ui_window' opens a scope that 'ui_end' closes", _GUI),
        ('tag = ui.window("x")', "call it as the subject of a with statement", _GUI),
        ("with Ui.make(2):\n    pass", "its subject calls an extern declared closed_by", _GUI),
        ("with ui.group(1) as g:\n    pass", "returns nothing to bind", _GUI),
        ("with ui.group(1):\n    pass", "closer 'ui_end' takes only the Ui", _BAD_CLOSER),
    ],
)
def test_scope_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, message: str, gui: str) -> None:
    frame = textwrap.dedent(
        """
        import vernon_dsl as vd

        from gui import Ui


        @vd.host
        def broken() -> None:
            ui = Ui.make(1)
        """
    ) + textwrap.indent(body, "    ")
    with pytest.raises(CompileError) as error:
        load_host_module(_sources(tmp_path, monkeypatch, frame, gui))
    assert message in str(error.value)
