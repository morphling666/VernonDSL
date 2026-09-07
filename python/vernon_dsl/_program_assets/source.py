"""Evaluation of a typed Program Asset declaration from Python source."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from ..diagnostics import ProgramCompileError
from .declaration import ProgramAssetDeclaration


def _importable_module_name(source: Path) -> str | None:
    resolved = source.resolve()
    candidates: list[tuple[int, str]] = []
    for entry in sys.path:
        root = Path(entry or ".").resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        parts = relative.with_suffix("").parts
        if len(parts) < 2 or any(not part.isidentifier() for part in parts):
            continue
        package = ".".join(parts[:-1])
        try:
            package_spec = importlib.util.find_spec(package)
        except (ImportError, ModuleNotFoundError, AttributeError):
            continue
        if package_spec is not None:
            candidates.append((len(parts), ".".join(parts)))
    return min(candidates)[1] if candidates else None


def load_program_asset_declaration(source: Path, declaration_name: str) -> ProgramAssetDeclaration:
    module_name = _importable_module_name(source)
    if module_name is None:
        module_name = f"_vernon_program_asset_{hashlib.sha256(str(source).encode()).hexdigest()[:20]}"
    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise ProgramCompileError(f"cannot load Program Asset source {source}")
    module = importlib.util.module_from_spec(specification)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    fallback_search_path = module_name.startswith("_vernon_program_asset_")
    if fallback_search_path:
        sys.path.insert(0, str(source.parent))
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise ProgramCompileError(f"cannot evaluate Program Asset source {source}: {error}") from error
    finally:
        if fallback_search_path:
            sys.path.pop(0)
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    declaration = getattr(module, declaration_name, None)
    if not isinstance(declaration, ProgramAssetDeclaration):
        raise ProgramCompileError(f"Program Asset declaration '{declaration_name}' did not evaluate canonically")
    return declaration


__all__ = ["load_program_asset_declaration"]
