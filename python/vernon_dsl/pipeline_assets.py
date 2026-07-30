"""Public pipeline asset declaration and cooking facade."""

from ._shader_assets.artifact_io import encode_runtime_stage
from ._shader_assets.cooking import cook_pipeline_asset
from ._shader_assets.declaration import PipelineAssetDeclaration, pipeline_asset
from ._shader_assets.descriptors import ShaderModuleDescriptor, ShaderPipelineDescriptor, ShaderStageReference
from ._shader_assets.parsing import parse_python_pipeline_asset
from .bundle import PipelineCompileError

__all__ = [
    "PipelineAssetDeclaration",
    "PipelineCompileError",
    "ShaderModuleDescriptor",
    "ShaderPipelineDescriptor",
    "ShaderStageReference",
    "cook_pipeline_asset",
    "encode_runtime_stage",
    "parse_python_pipeline_asset",
    "pipeline_asset",
]
