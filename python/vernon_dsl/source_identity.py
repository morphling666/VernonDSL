from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from pathlib import Path


def source_text_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def source_file_digest(path: Path) -> str | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return source_text_digest(source)


def project_root(start: Path, is_file: Callable[[Path], bool] = Path.is_file) -> Path | None:
    """The nearest directory at or above `start` that contains `pyproject.toml`."""

    for candidate in (start, *start.parents):
        if is_file(candidate / "pyproject.toml"):
            return candidate.resolve()
    return None


def project_module_name(path: Path, root: Path) -> str | None:
    """The checkout-independent dotted name of a source file under `root`, or None when it has none.

    `project_module_path` inverts it, so every component must be an identifier: a directory named `a.b` would make
    the dotted spelling ambiguous.
    """

    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return None
    if relative.suffix != ".py":
        return None
    parts = relative.with_suffix("").parts
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def project_module_path(name: str, root: Path) -> Path | None:
    parts = name.split(".")
    if not all(part.isidentifier() for part in parts):
        return None
    return root.joinpath(*parts).with_suffix(".py")


def import_root(path: Path, is_file: Callable[[Path], bool] = Path.is_file) -> Path | None:
    """The directory Python imports the module of a regular package from: the first ancestor holding no
    `__init__.py`. None for a module outside a regular package.

    The directory must be one Python imports from: the project root or its `src/` directory (the layouts packaging
    tools install from), a `sys.path` entry, or any directory outside a project (an installed wheel). A regular
    package nested in namespace packages under a project, `<root>/a/b/pkg/__init__.py` with no `a/__init__.py`, is
    imported as `a.b.pkg`, never as `pkg`, so it has no import root of its own.
    """

    directory = path.resolve().parent
    if not is_file(directory / "__init__.py"):
        return None
    while is_file(directory / "__init__.py"):
        directory = directory.parent
    project = project_root(directory, is_file)
    if project is None or directory in (project, project / "src") or directory in _search_path():
        return directory
    return None


def _search_path() -> set[Path]:
    entries: set[Path] = set()
    for entry in sys.path:
        try:
            entries.add(Path(entry or ".").resolve())
        except (OSError, ValueError):
            continue
    return entries


def package_module_name(path: Path, is_file: Callable[[Path], bool] = Path.is_file) -> str | None:
    """The dotted name Python imports a module of a regular package by, so a checkout, an editable install and an
    installed wheel name it alike; None outside a regular package."""

    root = import_root(path, is_file)
    if root is None or path.suffix != ".py":
        return None
    relative = path.resolve().relative_to(root).with_suffix("")
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    if not all(part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def module_identity(path: Path) -> str | None:
    """A module's checkout-independent name: its import name inside a regular package, else its path under the
    nearest project root."""

    name = package_module_name(path)
    if name is not None:
        return name
    root = project_root(path.parent)
    return project_module_name(path, root) if root is not None else None
