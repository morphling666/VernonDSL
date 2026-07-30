from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
class ShaderPipelineDescriptor:
    id: str
    stages: dict[str, ShaderStageReference]
    variants: tuple[tuple[str, ...], ...]
    manifest_path: Path
    canonical_manifest: str
    modules: dict[str, ShaderModuleDescriptor]
