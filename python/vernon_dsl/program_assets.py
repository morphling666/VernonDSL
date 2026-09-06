"""Public Program Asset declaration and cooking facade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._runtime.cooked_program import CookedProgram

from ._program_assets.artifact_io import encode_runtime_stage
from ._program_assets.cooking import cook_program_asset
from ._program_assets.declaration import ProgramAssetDeclaration, program_asset
from ._program_assets.parsing import ProgramAssetLint, lint_python_program_asset
from .bundle import ProgramCompileError


def load_program(
    manifest: str | Path,
    *,
    features: tuple[str, ...] = (),
) -> CookedProgram:
    from ._runtime.cooked_program import load_program as load

    return load(manifest, features=features)


__all__ = [
    "ProgramAssetDeclaration",
    "ProgramAssetLint",
    "ProgramCompileError",
    "cook_program_asset",
    "encode_runtime_stage",
    "lint_python_program_asset",
    "load_program",
    "program_asset",
]
