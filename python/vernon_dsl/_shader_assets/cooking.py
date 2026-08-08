from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .._versions import PIPELINE_VERSION
from ..ad import ProgramTransformSpec
from ..bundle import (
    BundlePlan,
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
from ..compiler import Compiler, FrontendCompileRequest, compile_file
from ..frontend.autodiff_profiles import build_autodiff_profile_plan
from ..frontend.structured_vjp import (
    build_structured_vjp,
    is_structured_vjp_abi_eligible,
    resolve_vjp_transform,
)
from ..language.stage_registry import validate_stage_target
from ..module_graph import load_project
from .artifact_io import write_external_artifact
from .cpu_registration import write_cpu_static_registration
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
    for source_name, metadata_name in (
        ("processor", "cpu"),
        ("features", "cpu_features"),
    ):
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
    if pipeline.transform is not None and set(pipeline.stages) != {"compute"}:
        raise PipelineCompileError(
            "initial VJP asset cooking supports compute pipelines only; no primal-only substitute was emitted"
        )
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
    planned_variants: list[tuple[tuple[str, ...], dict[str, CompiledStage]]] = []
    differentiated_variants: list[dict[str, Any]] = []
    differentiated_stages: dict[str, CompiledStage] = {}
    compile_cache: dict[tuple[str, str, str, str], CompiledStage] = {}
    compiler = native.Compiler()
    transform = None
    resolved_transform = None
    if pipeline.transform is not None:
        transform_values = dict(pipeline.transform)
        transform_values.pop("identity", None)
        transform = ProgramTransformSpec(
            kind=str(transform_values["kind"]),
            wrt=tuple(transform_values["wrt"]),
            rule_set=transform_values.get("rule_set"),
            rule_set_identity=transform_values.get("rule_set_identity"),
            output_cotangents=tuple(transform_values.get("output_cotangents", ())),
            gradient_policy=str(transform_values["gradient_policy"]),
            accumulation_policy=str(transform_values["accumulation_policy"]),
            tape_policy=str(transform_values["tape_policy"]),
            protocol=str(transform_values.get("protocol", "dynamic_v2")),
            derivative_rules_version=int(transform_values["derivative_rules_version"]),
        )
    for variant in pipeline.variants:
        stages: dict[str, CompiledStage] = {}
        for stage, reference in pipeline.stages.items():
            module = selected_modules[stage]
            mlir = compile_file(
                module.source,
                features=variant,
                entry=reference.entry,
            )
            key = (
                reference.module,
                reference.entry,
                hashlib.sha256(mlir.encode()).hexdigest(),
                canonical_json(resolved_target.spec),
            )
            compiled = compile_cache.get(key)
            if compiled is None:
                compiled = _compile_stage(
                    module,
                    reference,
                    stage,
                    variant,
                    resolved_target,
                    compiler,
                    native_target,
                    mlir,
                )
                compile_cache[key] = compiled
            stages[stage] = compiled
        planned_variants.append((variant, stages))
        if transform is not None:
            frontend = None
            if target_name == "cpu":
                frontend = Compiler().compile_request(
                    FrontendCompileRequest(
                        selected_modules["compute"].source,
                        pipeline.stages["compute"].entry,
                        variant,
                    )
                )
            if transform.protocol == "dynamic_v2":
                if frontend is None or not is_structured_vjp_abi_eligible(frontend, transform):
                    raise PipelineCompileError(
                        "dynamic_v2 requires a structured CPU VJP; "
                        "use protocol='legacy_fixed' only for an explicitly identified native profile"
                    )
                try:
                    structured = build_structured_vjp(native, frontend, transform)
                except ValueError as error:
                    raise PipelineCompileError(str(error)) from None
                variant_transform = structured.transform
                profile_plan = structured.plan
                profile_modules = structured.profiles
            else:
                from ..frontend.autodiff_native import (
                    AutodiffNativeLoweringError,
                    emit_native_autodiff_modules,
                )

                legacy_frontend = Compiler().compile_request(
                    FrontendCompileRequest(
                        selected_modules["compute"].source,
                        pipeline.stages["compute"].entry,
                        variant,
                        program_transform=transform,
                    )
                )
                if legacy_frontend.program_graph is None or legacy_frontend.autodiff_profiles is None:
                    raise PipelineCompileError("VJP frontend produced no differentiated profile plan")
                variant_transform = resolve_vjp_transform(
                    transform,
                    legacy_frontend.program_graph.reverse.cotangent_paths,
                )
                profile_plan = build_autodiff_profile_plan(variant_transform, legacy_frontend.program_graph)
                try:
                    profile_modules = emit_native_autodiff_modules(
                        legacy_frontend.program_graph,
                        profile_plan,
                        target=target_name,
                    )
                except AutodiffNativeLoweringError as error:
                    raise PipelineCompileError(str(error)) from None
            if resolved_transform is None:
                resolved_transform = variant_transform
            elif resolved_transform.output_cotangents != variant_transform.output_cotangents:
                raise PipelineCompileError("VJP variants must expose the same canonical output cotangent paths")
            profile_programs: dict[str, dict[str, Any]] = {}
            module = selected_modules["compute"]
            for profile_name, profile_mlir in profile_modules.items():
                profile = next(value for value in profile_plan.profiles if value.name == profile_name)
                profile_manifest = canonical_json(
                    {
                        "module": module.id,
                        "profile": profile_name,
                        "profiles_identity": profile_plan.identity,
                        "transform_identity": variant_transform.identity,
                    }
                )
                profile_module = ShaderModuleDescriptor(
                    f"{module.id}/ad/{profile_name}",
                    module.source,
                    module.manifest_path,
                    profile_manifest,
                )
                profile_stage = _compile_stage(
                    profile_module,
                    ShaderStageReference(profile_module.id, profile.symbol),
                    "compute",
                    variant,
                    resolved_target,
                    compiler,
                    native_target,
                    profile_mlir,
                )
                differentiated_stages[profile_stage.id] = profile_stage
                profile_programs[profile_name] = {
                    "compute": profile_stage.id,
                    "inputs": [binding.to_dict() for binding in profile.inputs],
                    "outputs": [binding.to_dict() for binding in profile.outputs],
                }
            primal_profile = next(value for value in profile_plan.profiles if value.name == "primal")
            profile_programs["primal"] = {
                "compute": stages["compute"].id,
                "inputs": [binding.to_dict() for binding in primal_profile.inputs],
                "outputs": [binding.to_dict() for binding in primal_profile.outputs],
            }
            differentiated_variants.append(
                {
                    "key": list(variant),
                    "workgroup_size": list(profile_plan.launch.workgroup_size),
                    "profiles": {name: profile_programs[name] for name in ("primal", "forward_with_tape", "backward")},
                }
            )

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
    if transform is not None:
        assert resolved_transform is not None
        transform = resolved_transform
        plan = BundlePlan(
            plan.pipeline_id,
            plan.target,
            plan.features,
            plan.variants,
            (
                *plan.stages,
                *(differentiated_stages[key] for key in sorted(differentiated_stages)),
            ),
            transform.to_dict(),
            {"variants": differentiated_variants},
        )
    output_path.mkdir(parents=True, exist_ok=True)
    descriptors = {
        stage.id: write_external_artifact(
            output_path,
            stage.artifact.data,
            stage.artifact.format,
            stage.stage,
            stage.artifact.filename,
        )
        for stage in plan.stages
    }
    bundle = materialize_bundle(plan, descriptors)
    if target_name == "cpu":
        write_cpu_static_registration(
            output_path,
            [str(stage.metadata.get("symbol", "")) for stage in plan.stages],
        )
    manifest = output_path / f"{output_path.name}.pipeline.json"
    manifest.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


__all__ = ["cook_pipeline_asset"]
