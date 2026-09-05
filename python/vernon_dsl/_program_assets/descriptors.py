from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ShaderModuleDescriptor:
    id: str
    source: Path
    manifest_path: Path
    canonical_manifest: str


@dataclass(frozen=True)
class ShaderStageReference:
    module: str
    entry: str


@dataclass(frozen=True)
class ShaderProgramDescriptor:
    id: str
    stages: dict[str, ShaderStageReference]
    variants: tuple[tuple[str, ...], ...]
    manifest_path: Path
    canonical_manifest: str
    modules: dict[str, ShaderModuleDescriptor]
    transform: Mapping[str, Any] | None = None
