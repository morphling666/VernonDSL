"""Handle methods: externs a `@handle` class body declares, as `Body.push`, the C function `body_push(body, f)`,
called as `body.push(f)`, and `@staticmethod` ones, as `Body.create(mass)`, in Host code compiled and in Python
mode, where handles are values of their class."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from host_aot import LinkedHost, toolchain
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostModule, load_host_module

_RIG = """
from __future__ import annotations

import vernon_dsl as vd
from vernon_dsl import f64


@vd.handle
class Body:
    @staticmethod
    @vd.extern(symbol="rig_body_create")
    def create(mass: f64) -> Body: ...

    @vd.extern(symbol="rig_body_push", effects="write")
    def push(self, force: f64) -> None: ...

    @vd.extern(symbol="rig_body_speed", effects="read")
    def speed(self) -> f64: ...


@vd.extern(symbol="rig_label_body")
def label_body(body: Body, name: str) -> None: ...
"""

_WALK = """
import vernon_dsl as vd
from vernon_dsl import f64

from rig import Body


@vd.host
class Pusher:
    body: Body

    def run(self, force: f64) -> f64:
        self.body.push(force)
        self.body.push(force * 2.0)
        return self.body.speed()


@vd.host
def simulate(mass: f64) -> f64:
    pusher = Pusher(body=Body.create(mass))
    other = Body.create(mass * 2.0)
    other.push(4.0)
    return pusher.run(1.0) + other.speed()
"""


class _Rig:
    def __init__(self) -> None:
        self.masses: list[float] = []
        self.momenta: dict[int, float] = {}

    def registry(self) -> ExternRegistry:
        return (
            ExternRegistry()
            .bind("rig_body_create", self.create)
            .bind("rig_body_push", self.push)
            .bind("rig_body_speed", lambda body: self.momenta[int(body)] / self.masses[int(body) - 1])
            .bind("rig_label_body", lambda body, name: None)
        )

    def create(self, mass: float) -> int:
        self.masses.append(float(mass))
        self.momenta[len(self.masses)] = 0.0
        return len(self.masses)

    def push(self, body: Any, force: float) -> None:
        self.momenta[int(body)] += float(force)


def _sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walk: str = _WALK, rig: str = _RIG) -> Path:
    (tmp_path / "rig.py").write_text(rig, encoding="utf-8")
    source = tmp_path / "walk.py"
    source.write_text(walk, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return source


def test_handle_methods_run_in_python_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _Rig()
    module = HostModule(_sources(tmp_path, monkeypatch), rig.registry(), eager=False)
    assert module.call("simulate", 2.0) == pytest.approx(3.0 / 2.0 + 4.0 / 4.0)

    body_class = module.namespace.Body
    assert body_class.push.__name__ == "push" and body_class.push.symbol == "rig_body_push"
    assert not hasattr(body_class, "label_body"), "an extern declared outside the class is no method"
    pusher = module.namespace.Pusher()
    pusher.body = body_class(1)
    assert pusher.run(2.0) == pytest.approx((3.0 + 6.0) / 2.0)
    assert type(pusher.body) is body_class and pusher.body == body_class(1) and pusher.body.bits == 1


@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_handle_methods_compile_to_extern_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    source = _sources(tmp_path, monkeypatch)
    python = HostModule(source, _Rig().registry(), eager=False).call("simulate", 2.0)
    library = LinkedHost(load_programs(source), tmp_path / "aot", _Rig().registry())
    assert float(library.call("simulate", 2.0)) == pytest.approx(float(python))
    assert library.errors == []


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("body.fly()", "'Body' has no method 'fly'"),
        ("body.label_body('x')", "'Body' has no method 'label_body'"),
        ("body.push()", "push"),
    ],
)
def test_handle_method_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, message: str) -> None:
    walk = textwrap.dedent(
        f"""
        import vernon_dsl as vd
        from vernon_dsl import f64

        from rig import Body


        @vd.host
        def broken() -> None:
            body = Body.create(1.0)
            {body}
        """
    )
    with pytest.raises(CompileError) as error:
        load_host_module(_sources(tmp_path, monkeypatch, walk))
    assert message in str(error.value)


def test_handle_bodies_declare_only_externs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _RIG.replace(
        '    @vd.extern(symbol="rig_body_speed", effects="read")\n',
        "    def doubled(self, force: f64) -> f64:\n        return force * 2.0\n\n"
        '    @vd.extern(symbol="rig_body_speed", effects="read")\n',
    )
    with pytest.raises(CompileError) as error:
        load_host_module(_sources(tmp_path, monkeypatch, rig=rig))
    assert "handle 'Body' declares only externs" in str(error.value)


@pytest.mark.parametrize(
    ("walk", "rig_tail", "message"),
    [
        (
            "from rig import body_create\n\n\n@vd.host\ndef broken() -> None:\n    body_create(1.0)\n",
            "",
            "'body_create' is the C name of 'Body.create'; Host code calls Body.create(...)",
        ),
        (
            "import rig\n\n\n@vd.host\ndef broken() -> None:\n    rig.body_push(rig.Body(), 1.0)\n",
            "",
            "'body_push' is the C name of 'Body.push'; Host code calls body.push(...) on a Body",
        ),
        (
            "from rig import Body, shove\n\n\n@vd.host\ndef broken() -> None:\n    shove(Body.create(1.0))\n",
            "\n\n@vd.host\ndef shove(body: Body) -> None:\n    body_push(body, 1.0)\n",
            "'body_push' is the C name of 'Body.push'",
        ),
    ],
)
def test_c_names_of_handle_methods_are_not_host_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walk: str, rig_tail: str, message: str
) -> None:
    walk = "import vernon_dsl as vd\n" + walk + "\n\n@vd.host\ndef main() -> None:\n    broken()\n"
    with pytest.raises(CompileError) as error:
        load_host_module(_sources(tmp_path, monkeypatch, walk, rig=_RIG + rig_tail))
    assert message in str(error.value)
