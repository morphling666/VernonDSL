from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .._versions import PIPELINE_VERSION
from ..bundle import (
    CompiledStage,
    OpenGLTargetOptions,
    PipelineCompileError,
    TargetOptions,
    build_bundle_plan,
    canonical_json,
    compiled_stage_from_program,
    make_target_options,
    materialize_bundle,
)
from ..compiler import compile_file
from ..language.stage_registry import validate_stage_target
from ..module_graph import load_project
from .artifact_io import write_external_artifact
from .descriptors import ShaderModuleDescriptor, ShaderStageReference
from .parsing import parse_python_pipeline_asset, pipeline_asset_reference


def _native_module() -> Any:
    try:
        from .. import _native as native
    except (ImportError, OSError):
        raise PipelineCompileError(
            "shader cooking requires vernon_dsl._native; build the native extension or install a wheel containing it"
        ) from None
    return native


def _native_target(native: Any, target: str) -> Any:
    try:
        name = {
            "cpu": "CPU",
            "cuda": "CUDA",
            "vulkan": "VULKAN",
            "metal": "METAL",
            "directx": "DIRECTX",
            "opengl": "OPENGL",
            "opengles": "OPENGL_ES",
        }[target]
    except KeyError:
        raise PipelineCompileError(f"target '{target}' is not available through the native compiler") from None
    try:
        return getattr(native.Target, name)
    except AttributeError:
        raise PipelineCompileError(f"target '{target}' is not available through the native compiler") from None


def _cpu_object_format(filename: str, target_triple: str) -> str:
    if Path(filename).suffix == ".obj":
        return "coff"
    normalized = target_triple.lower()
    if "wasm" in normalized:
        return "wasm"
    if any(name in normalized for name in ("apple", "darwin", "macos", "ios")):
        return "macho"
    return "elf"


def _cpu_stage_metadata(stage: CompiledStage) -> dict[str, Any]:
    symbol = stage.interface.get("symbol")
    reflected_target = stage.reflection.get("target")
    reflected_options = reflected_target.get("options") if isinstance(reflected_target, Mapping) else None
    if not isinstance(symbol, str) or not symbol:
        raise PipelineCompileError(f"CPU compiler reflection has no exported symbol for {stage.entry}")
    if not isinstance(reflected_options, Mapping):
        raise PipelineCompileError("CPU compiler reflection has no target options")
    target_triple = reflected_options.get("triple")
    if not isinstance(target_triple, str) or not target_triple:
        raise PipelineCompileError("CPU compiler reflection has no normalized target triple")
    metadata: dict[str, Any] = {
        "symbol": symbol,
        "pipeline_version": PIPELINE_VERSION,
        "target_triple": target_triple,
        "object_format": _cpu_object_format(stage.artifact.filename, target_triple),
    }
    for source_name, metadata_name in (("processor", "cpu"), ("features", "cpu_features")):
        value = reflected_options.get(source_name)
        if isinstance(value, str):
            metadata[metadata_name] = value
        elif source_name == "features" and isinstance(value, list) and all(isinstance(item, str) for item in value):
            metadata[metadata_name] = ",".join(value)
    return metadata


def _compile_stage(
    module: ShaderModuleDescriptor,
    reference: ShaderStageReference,
    stage: str,
    variant: tuple[str, ...],
    target: TargetOptions,
    compiler: Any,
    native_target: Any,
    mlir: str | None = None,
) -> CompiledStage:
    mlir = mlir or compile_file(module.source, features=variant, entry=reference.entry)
    compiled = compiled_stage_from_program(
        compiler.compile_program_result(mlir, native_target, **target.native_options),
        module=module.id,
        module_manifest=module.canonical_manifest,
        entry=reference.entry,
        target=target,
    )
    if compiled.stage != stage:
        raise PipelineCompileError(f"compiler reflected {reference.entry} as {compiled.stage}, expected {stage}")
    if target.target == "cpu":
        compiled = CompiledStage(
            compiled.module,
            compiled.module_manifest,
            compiled.entry,
            compiled.stage,
            compiled.target,
            compiled.reflection,
            compiled.interface,
            compiled.artifact,
            _cpu_stage_metadata(compiled),
        )
        if compiled.artifact.format != "relocatable_object":
            raise PipelineCompileError("CPU cooking requires the compiler relocatable object artifact")
    return compiled


def cook_pipeline_asset(
    *,
    pipeline_asset: str | Path,
    output: str | Path,
    target: TargetOptions | str | None = None,
) -> Path:
    source, descriptor_name = pipeline_asset_reference(pipeline_asset)
    pipeline = parse_python_pipeline_asset(source, descriptor_name)
    if target is None:
        target = OpenGLTargetOptions()
    elif isinstance(target, str):
        target = make_target_options(target)
    target_name = target.target
    if target_name == "cpu" and set(pipeline.stages) != {"compute"}:
        raise PipelineCompileError("CPU pipeline bundles support one compute stage and no graphics or barrier steps")
    for stage in pipeline.stages:
        try:
            validate_stage_target(stage, target_name)
        except ValueError as error:
            raise PipelineCompileError(str(error)) from None
    resolved_target = target
    native = _native_module()
    native_target = _native_target(native, target_name)
    selected_modules: dict[str, ShaderModuleDescriptor] = {}
    declared_features: set[str] = set()
    for stage, reference in pipeline.stages.items():
        module = pipeline.modules.get(reference.module)
        if module is None:
            raise PipelineCompileError(f"pipeline {stage} stage references unknown module '{reference.module}'")
        selected_modules[stage] = module
        declared_features.update(load_project(module.source).features)
    for variant in pipeline.variants:
        unknown = set(variant) - declared_features
        if unknown:
            raise PipelineCompileError("variant requests undeclared feature(s): " + ", ".join(sorted(unknown)))

    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    planned_variants: list[tuple[tuple[str, ...], dict[str, CompiledStage]]] = []
    compile_cache: dict[tuple[str, str, str, str], CompiledStage] = {}
    compiler = native.Compiler()
    for variant in pipeline.variants:
        stages: dict[str, CompiledStage] = {}
        for stage, reference in pipeline.stages.items():
            module = selected_modules[stage]
            mlir = compile_file(module.source, features=variant, entry=reference.entry)
            key = (
                reference.module,
                reference.entry,
                hashlib.sha256(mlir.encode()).hexdigest(),
                canonical_json(resolved_target.spec),
            )
            compiled = compile_cache.get(key)
            if compiled is None:
                compiled = _compile_stage(
                    module, reference, stage, variant, resolved_target, compiler, native_target, mlir
                )
                compile_cache[key] = compiled
            stages[stage] = compiled
        planned_variants.append((variant, stages))

    compiled_targets = {
        canonical_json(stage.target.spec): stage.target for _, stages in planned_variants for stage in stages.values()
    }
    if len(compiled_targets) != 1:
        raise PipelineCompileError("compiler returned inconsistent target options across pipeline stages")
    manifest_target = next(iter(compiled_targets.values()))
    plan = build_bundle_plan(
        pipeline.id,
        manifest_target,
        sorted({feature for variant in pipeline.variants for feature in variant}),
        planned_variants,
    )
    descriptors = {
        stage.id: write_external_artifact(
            output_path, stage.artifact.data, stage.artifact.format, stage.stage, stage.artifact.filename
        )
        for stage in plan.stages
    }
    bundle = materialize_bundle(plan, descriptors)
    manifest = output_path / f"{output_path.name}.pipeline.json"
    manifest.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


__all__ = ["cook_pipeline_asset"]
