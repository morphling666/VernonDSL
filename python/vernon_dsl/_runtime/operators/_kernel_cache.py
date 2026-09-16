"""Load generated @vd.kernel sources from a durable tempfile."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

_generated_kernels: dict[tuple[Any, ...], Any] = {}
_generated_directories: list[tempfile.TemporaryDirectory[str]] = []


def load_generated_kernel(key: tuple[Any, ...], entry: str, source: str, prefix: str) -> Any:
    cached = _generated_kernels.get(key)
    if cached is not None:
        return cached
    directory = tempfile.TemporaryDirectory(prefix=f"vernon-{prefix}-")
    path = Path(directory.name) / f"{entry}.py"
    path.write_text(source, encoding="utf-8")
    module_name = f"vernon_dsl._runtime._generated_{prefix}_{abs(hash(key))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load generated {prefix} kernel {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, entry)
    _generated_directories.append(directory)
    _generated_kernels[key] = kernel
    return kernel


__all__ = ["load_generated_kernel"]
