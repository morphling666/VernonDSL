"""Host modules importing other packages: a module of another installed package, found along `sys.path` or through
an editable install's finder, compiles into the importer's object under its import name, and runs in Python mode
through an ordinary import."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import shutil
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
from host_aot import LinkedHost, toolchain
from test_host_class_values import _native_available
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostModule
from vernon_dsl.hostlang.aot import compile_host, load_programs
from vernon_dsl.source_identity import module_identity

_SHAPES = """
import vernon_dsl as vd
from vernon_dsl import f64


@vd.host
class Box:
    w: f64
    h: f64

    def area(self) -> f64:
        return self.w * self.h


@vd.host
def double(x: f64) -> f64:
    return x * 2.0
"""

_LEVEL = """
import vernon_dsl as vd
from vernon_dsl import f64
{imports}


@vd.host
def total(w: f64, h: f64) -> f64:
    box = Box()
    box.w = w
    box.h = h
    return double(box.area())
"""

_PACKAGES = {"geo", "game", "plain"}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


def _project(root: Path, files: dict[str, str]) -> Path:
    """A `src/` layout project; returns its `src` directory."""

    _write(root / "pyproject.toml", f"[project]\nname = '{root.name}'\nversion = '0'\n")
    for name, text in files.items():
        _write(root / "src" / name, text)
    return root / "src"


def _level(game: Path, imports: str) -> Path:
    return _write(game / "game" / "level.py", _LEVEL.format(imports=imports))


class _EditableFinder:
    """Maps top-level packages to directories, as setuptools' editable installs do from `sys.meta_path`."""

    def __init__(self, packages: dict[str, Path]):
        self.packages = packages

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        top, _, rest = fullname.partition(".")
        if top not in self.packages:
            return None
        base = self.packages[top].joinpath(*rest.split(".")) if rest else self.packages[top]
        if (base / "__init__.py").is_file():
            return importlib.util.spec_from_file_location(
                fullname, base / "__init__.py", submodule_search_locations=[str(base)]
            )
        if base.with_suffix(".py").is_file():
            return importlib.util.spec_from_file_location(fullname, base.with_suffix(".py"))
        return None


@pytest.fixture
def projects(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """(geo's src, game's src): `geo.shapes` declares Box and double; `game` is an empty package."""

    geo = _project(tmp_path / "geo_project", {"geo/__init__.py": "", "geo/shapes.py": _SHAPES})
    game = _project(tmp_path / "game_project", {"game/__init__.py": ""})
    before = set(sys.modules)
    importlib.invalidate_caches()
    yield geo, game
    for name in set(sys.modules) - before:
        if name.split(".")[0] in _PACKAGES:
            del sys.modules[name]


@pytest.mark.parametrize("install", ["sys_path", "editable_finder"])
def test_a_host_module_compiles_and_runs_code_of_another_installed_package(
    projects: tuple[Path, Path], install: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    geo, game = projects
    if install == "sys_path":
        monkeypatch.syspath_prepend(str(geo))
    else:
        monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, _EditableFinder({"geo": geo / "geo"})])
    level = _level(game, "from geo.shapes import Box, double")

    artifacts = compile_host(level, prefix="game_")
    assert [entry["symbol"] for entry in artifacts.reflection["entries"]] == ["game_total"]
    assert "__vernon_geo_shapes__double" in artifacts.ir
    assert HostModule(level, ExternRegistry()).call("total", 2.0, 3.0) == 12.0


@pytest.mark.parametrize("imports", ["from game.shapes import Box, double", "from .shapes import Box, double"])
def test_a_src_layout_package_imports_its_own_modules(projects: tuple[Path, Path], imports: str) -> None:
    _, game = projects
    _write(game / "game" / "shapes.py", _SHAPES)
    level = _level(game, imports)

    assert "__vernon_game_shapes__double" in compile_host(level).ir
    assert HostModule(level, ExternRegistry()).call("total", 2.0, 3.0) == 12.0


def test_a_package_module_has_its_import_name_in_a_checkout_and_installed(
    projects: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geo, game = projects
    installed = tmp_path / "site-packages"
    shutil.copytree(geo / "geo", installed / "geo")
    monkeypatch.syspath_prepend(str(installed))
    level = _level(game, "from geo.shapes import Box, double")

    assert module_identity(geo / "geo" / "shapes.py") == module_identity(installed / "geo" / "shapes.py")
    assert module_identity(installed / "geo" / "shapes.py") == "geo.shapes"
    assert module_identity(installed / "geo" / "__init__.py") == "geo"
    ir = compile_host(level).ir
    assert "__vernon_geo_shapes__double" in ir
    assert installed.as_posix() not in ir


def test_a_package_inside_namespace_packages_is_named_by_its_project_path(tmp_path: Path) -> None:
    project = tmp_path / "engine_project"
    _write(project / "pyproject.toml", '[project]\nname = "engine_project"\n')
    _write(project / "front" / "render" / "shader" / "__init__.py", "")
    _write(project / "front" / "render" / "shader" / "surfaces.py", "")

    assert module_identity(project / "front" / "render" / "shader" / "surfaces.py") == "front.render.shader.surfaces"


@pytest.mark.parametrize("found", [True, False])
def test_names_a_python_import_brings_in_are_rejected_in_host_code(
    projects: tuple[Path, Path], tmp_path: Path, found: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, game = projects
    if found:
        # The same classes, but the module does not import vernon_dsl, so it is a Python import Host code cannot see.
        plain = tmp_path / "plain_project"
        _write(plain / "plain" / "__init__.py", "")
        _write(plain / "plain" / "shapes.py", "class Box:\n    pass\n\n\ndef double(x):\n    return x * 2\n")
        monkeypatch.syspath_prepend(str(plain))
    level = _level(game, "from plain.shapes import Box, double")

    with pytest.raises(CompileError, match="'Box' comes from 'plain.shapes', which is not a Host source"):
        compile_host(level)


@pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")
@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_the_linked_object_runs_code_of_another_package(
    projects: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geo, game = projects
    monkeypatch.syspath_prepend(str(geo))
    level = _level(game, "from geo.shapes import Box, double")

    library = LinkedHost(load_programs(level), tmp_path / "aot", ExternRegistry())
    assert library.call("total", 2.0, 3.0) == 12.0
