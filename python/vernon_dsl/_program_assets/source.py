"""Evaluation of a typed Program Asset declaration from Python source."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from ..diagnostics import ProgramCompileError
from .declaration import ProgramAssetDeclaration


def load_program_asset_declaration(source: Path, declaration_name: str) -> ProgramAssetDeclaration:
    module_name = f"_vernon_program_asset_{hashlib.sha256(str(source).encode()).hexdigest()[:20]}"
    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise ProgramCompileError(f"cannot load Program Asset source {source}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    sys.path.insert(0, str(source.parent))
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise ProgramCompileError(f"cannot evaluate Program Asset source {source}: {error}") from error
    finally:
        sys.path.pop(0)
        sys.modules.pop(module_name, None)
    declaration = getattr(module, declaration_name, None)
    if not isinstance(declaration, ProgramAssetDeclaration):
        raise ProgramCompileError(f"Program Asset declaration '{declaration_name}' did not evaluate canonically")
    return declaration


__all__ = ["load_program_asset_declaration"]
