from __future__ import annotations

import hashlib
from pathlib import Path


def source_text_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def source_file_digest(path: Path) -> str | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return source_text_digest(source)


def project_root(start: Path) -> Path | None:
    """The nearest directory at or above `start` that contains `pyproject.toml`."""

    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
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
