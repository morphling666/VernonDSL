from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

_dll_directories: list[Any] = []


def load_native() -> Any | None:
    """Load the packaged extension or this checkout's development build."""

    try:
        from . import _native as packaged_native

        return packaged_native
    except ImportError:
        pass

    try:
        return importlib.import_module("_native")
    except ImportError:
        pass

    repository = Path(__file__).resolve().parents[2]
    source_build = repository / "build" / "source"
    candidates = (
        source_build / "Release",
        source_build / "Debug",
        source_build / "RelWithDebInfo",
        source_build / "MinSizeRel",
        source_build,
    )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if os.name == "nt":
            _dll_directories.append(os.add_dll_directory(str(candidate)))
        sys.path.insert(0, str(candidate))
        try:
            return importlib.import_module("_native")
        except ImportError:
            sys.path.pop(0)
    return None


def require_native(purpose: str) -> Any:
    native = load_native()
    if native is None:
        raise RuntimeError(
            f"{purpose} requires vernon_dsl._native; build the native extension or install a wheel containing it"
        )
    return native
