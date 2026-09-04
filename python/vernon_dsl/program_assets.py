"""Public pipeline asset declaration and cooking facade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._runtime.autodiff import CookedVjpPipeline
    from ._runtime.cooked_pipeline import CookedPipeline

from ._shader_assets.artifact_io import encode_runtime_stage
from ._shader_assets.cooking import cook_pipeline_asset
from ._shader_assets.declaration import PipelineAssetDeclaration, pipeline_asset
from ._shader_assets.descriptors import ShaderModuleDescriptor, ShaderPipelineDescriptor, ShaderStageReference
from ._shader_assets.parsing import parse_python_pipeline_asset
from .bundle import PipelineCompileError


def load_cooked_vjp_asset(
    manifest: str | Path,
    *,
    features: tuple[str, ...] = (),
) -> CookedVjpPipeline:
    from ._runtime.autodiff import load_cooked_vjp_asset as load

    return load(manifest, features=features)


def load_pipeline(
    manifest: str | Path,
    *,
    features: tuple[str, ...] = (),
) -> CookedPipeline:
    from ._runtime.cooked_pipeline import load_pipeline as load

    return load(manifest, features=features)


__all__ = [
    "PipelineAssetDeclaration",
    "PipelineCompileError",
    "ShaderModuleDescriptor",
    "ShaderPipelineDescriptor",
    "ShaderStageReference",
    "cook_pipeline_asset",
    "encode_runtime_stage",
    "load_cooked_vjp_asset",
    "load_pipeline",
    "parse_python_pipeline_asset",
    "pipeline_asset",
]
