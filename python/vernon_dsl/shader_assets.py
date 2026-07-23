from __future__ import annotations

import ast
import hashlib
import json
import keyword
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .compiler import compile_file
from .module_graph import load_project
from .pipeline_compile import (
    CompiledArtifact,
    CompiledStage,
    PipelineCompileError,
    TargetOptions,
    build_bundle_plan,
    canonical_json,
    compiled_stage_from_program,
    inline_artifact_descriptor,
    materialize_bundle,
    serialize_bundle,
)
from .types import Feature

ShaderAssetError = PipelineCompileError


@dataclass(frozen=True)
class PipelineAssetDeclaration:
    id: str
    stages: dict[str, Callable[..., Any]]
    variants: tuple[tuple[Feature, ...], ...]
    targets: dict[str, dict[str, Any]]


def pipeline_asset(
    *,
    id: str,
    compute: Callable[..., Any] | None = None,
    vertex: Callable[..., Any] | None = None,
    fragment: Callable[..., Any] | None = None,
    variants: Iterable[Iterable[Feature]] = ((),),
    targets: Mapping[str, Mapping[str, Any]],
) -> PipelineAssetDeclaration:
    """Declare a cookable pipeline without creating a runtime pipeline.

    The cooker reads this call from the source AST and does not execute it.
    This implementation exists for editor completion and normal Python imports.
    """
    stages = {
        name: value
        for name, value in (("compute", compute), ("vertex", vertex), ("fragment", fragment))
        if value is not None
    }
    return PipelineAssetDeclaration(
        id=id,
        stages=stages,
        variants=tuple(tuple(key) for key in variants),
        targets={name: dict(options) for name, options in targets.items()},
    )


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
    targets: dict[str, dict[str, Any]]
    manifest_path: Path
    canonical_manifest: str
    modules: dict[str, ShaderModuleDescriptor]


def _canonical_json(value: Any) -> str:
    return canonical_json(value)


def encode_runtime_stage(record: dict[str, Any], artifact: bytes) -> dict[str, Any]:
    artifact_format = record.get("format")
    if not isinstance(artifact_format, str) or not artifact_format:
        raise ShaderAssetError("runtime stage artifact format is missing")
    return {
        **record,
        "artifact": inline_artifact_descriptor(CompiledArtifact(artifact_format, artifact)),
    }


def serialize_runtime_pipeline_bundle(bundle: dict[str, Any]) -> bytes:
    return serialize_bundle(bundle)


def _artifact_extension(artifact_format: str, stage: str, original_name: str = "") -> str:
    if artifact_format == "glsl":
        return {
            "vertex": ".vert.glsl",
            "fragment": ".frag.glsl",
            "compute": ".comp.glsl",
        }.get(stage, ".glsl")
    if artifact_format == "gles":
        return {
            "vertex": ".vert.gles",
            "fragment": ".frag.gles",
            "compute": ".comp.gles",
        }.get(stage, ".gles")
    if artifact_format == "ptx":
        return ".ptx"
    if artifact_format == "spirv":
        return ".spv"
    if artifact_format == "native_library":
        suffix = Path(original_name).suffix
        return suffix or ".native"
    if artifact_format == "relocatable_object":
        suffix = Path(original_name).suffix
        return suffix if suffix in {".o", ".obj"} else ".o"
    suffix = Path(original_name).suffix
    return suffix or f".{artifact_format}"


def _write_external_artifact(
    output: Path, artifact: bytes, artifact_format: str, stage: str, original_name: str = ""
) -> dict[str, Any]:
    digest = hashlib.sha256(artifact).hexdigest()
    extension = _artifact_extension(artifact_format, stage, original_name)
    relative_path = Path("artifacts") / f"{digest}{extension}"
    artifact_path = output / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        if artifact_path.read_bytes() != artifact:
            raise ShaderAssetError(f"content-addressed artifact collision: {relative_path}")
    else:
        artifact_path.write_bytes(artifact)
    return {
        "format": artifact_format,
        "storage": "external",
        "path": relative_path.as_posix(),
        "size": len(artifact),
        "sha256": digest,
    }


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _pipeline_asset_reference(value: str | Path) -> tuple[Path, str]:
    spelling = str(value)
    marker = spelling.rfind(".py:")
    if marker < 0:
        raise ShaderAssetError(
            "pipeline asset input must be a Python descriptor reference in source.py:descriptor_name form"
        )
    source = Path(spelling[: marker + 3]).resolve()
    descriptor_name = spelling[marker + 4 :]
    if not descriptor_name or not descriptor_name.isidentifier() or keyword.iskeyword(descriptor_name):
        raise ShaderAssetError("pipeline asset reference requires a valid Python descriptor name after source.py:")
    return source, descriptor_name


def _feature_bindings(tree: ast.Module) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if not isinstance(value, ast.Call) or _call_name(value.func).split(".")[-1] != "feature":
            continue
        if (
            len(value.args) != 1
            or value.keywords
            or not isinstance(value.args[0], ast.Constant)
            or not isinstance(value.args[0].value, str)
        ):
            raise ShaderAssetError("feature declarations used by pipeline assets require one string literal")
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            raise ShaderAssetError("feature declarations used by pipeline assets require a simple name")
        bindings[targets[0].id] = value.args[0].value
    return bindings


def _literal_keyword(keywords: dict[str, ast.expr], name: str) -> Any:
    node = keywords.get(name)
    if node is None:
        raise ShaderAssetError(f"pipeline_asset declaration requires '{name}'")
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        raise ShaderAssetError(f"pipeline_asset '{name}' must be a literal value") from None


def parse_python_pipeline_asset(source: str | Path, descriptor_name: str) -> ShaderPipelineDescriptor:
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise ShaderAssetError(f"pipeline asset source does not exist: {source_path}")
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as error:
        raise ShaderAssetError(f"cannot parse pipeline asset source {source_path}: {error}") from None

    declaration: ast.Call | None = None
    for statement in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value
        if (
            len(targets) == 1
            and isinstance(targets[0], ast.Name)
            and targets[0].id == descriptor_name
            and isinstance(value, ast.Call)
            and _call_name(value.func).split(".")[-1] == "pipeline_asset"
        ):
            if declaration is not None:
                raise ShaderAssetError(f"duplicate pipeline asset declaration: {descriptor_name}")
            declaration = value
    if declaration is None:
        raise ShaderAssetError(f"pipeline asset declaration '{descriptor_name}' was not found in {source_path}")
    if declaration.args:
        raise ShaderAssetError("pipeline_asset accepts keyword arguments only")
    keywords = {keyword.arg: keyword.value for keyword in declaration.keywords if keyword.arg is not None}
    if len(keywords) != len(declaration.keywords):
        raise ShaderAssetError("pipeline_asset does not accept expanded keyword arguments")
    supported = {"id", "compute", "vertex", "fragment", "variants", "targets"}
    unknown = set(keywords) - supported
    if unknown:
        raise ShaderAssetError("unknown pipeline_asset argument(s): " + ", ".join(sorted(unknown)))

    pipeline_id = _literal_keyword(keywords, "id")
    if not isinstance(pipeline_id, str) or not pipeline_id:
        raise ShaderAssetError("pipeline asset id must be a non-empty string")

    stage_functions: dict[str, str] = {}
    for stage in ("compute", "vertex", "fragment"):
        node = keywords.get(stage)
        if node is None:
            continue
        if not isinstance(node, ast.Name):
            raise ShaderAssetError(f"pipeline_asset {stage} must reference a function in the same module")
        stage_functions[stage] = node.id
    stage_names = set(stage_functions)
    if stage_names not in ({"compute"}, {"vertex", "fragment"}, {"compute", "vertex", "fragment"}):
        raise ShaderAssetError("pipeline must contain vertex+fragment, compute+vertex+fragment, or one compute stage")

    definitions = {
        statement.name: {
            _call_name(decorator.func if isinstance(decorator, ast.Call) else decorator).split(".")[-1]
            for decorator in statement.decorator_list
        }
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef)
    }
    expected_decorators = {
        "compute": {"compute", "kernel"},
        "vertex": {"vertex"},
        "fragment": {"fragment"},
    }
    for stage, entry in stage_functions.items():
        if entry not in definitions or not (definitions[entry] & expected_decorators[stage]):
            raise ShaderAssetError(f"pipeline_asset {stage} entry '{entry}' has the wrong stage decorator")

    features = _feature_bindings(tree)
    variants_node = keywords.get("variants")
    if variants_node is None:
        variants: tuple[tuple[str, ...], ...] = ((),)
    elif not isinstance(variants_node, (ast.Tuple, ast.List)):
        raise ShaderAssetError("pipeline_asset variants must be a tuple or list")
    else:
        parsed_variants: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for variant_node in variants_node.elts:
            if not isinstance(variant_node, (ast.Tuple, ast.List)):
                raise ShaderAssetError("each pipeline asset variant must be a tuple or list")
            names: list[str] = []
            for feature_node in variant_node.elts:
                if not isinstance(feature_node, ast.Name) or feature_node.id not in features:
                    raise ShaderAssetError("pipeline asset variants must reference locally declared features")
                names.append(features[feature_node.id])
            key = tuple(names)
            if list(key) != sorted(key) or len(set(key)) != len(key):
                raise ShaderAssetError(f"pipeline asset variant is not canonical: {list(key)}")
            if key in seen:
                raise ShaderAssetError(f"duplicate pipeline asset variant: {list(key)}")
            seen.add(key)
            parsed_variants.append(key)
        variants = tuple(parsed_variants)
    if not variants:
        raise ShaderAssetError("pipeline must include at least one variant")
    if len(variants) > 16:
        raise ShaderAssetError(f"pipeline declares {len(variants)}, exceeding variant cap 16")

    targets = _literal_keyword(keywords, "targets")
    if not isinstance(targets, dict) or not targets:
        raise ShaderAssetError("pipeline_asset targets must be a non-empty dictionary")
    normalized_targets: dict[str, dict[str, Any]] = {}
    for target, options in targets.items():
        if not isinstance(target, str) or not isinstance(options, dict):
            raise ShaderAssetError("pipeline target options must be objects")
        normalized_targets[target] = options

    module_id = f"python/{source_path.stem}"
    source_manifest = {
        "type": "python_pipeline_asset",
        "source": source_path.name,
        "name": descriptor_name,
        "id": pipeline_id,
        "stages": stage_functions,
        "variants": [list(key) for key in variants],
        "targets": normalized_targets,
    }
    module = ShaderModuleDescriptor(module_id, source_path, source_path, _canonical_json(source_manifest))
    stages = {stage: ShaderStageReference(module_id, entry) for stage, entry in stage_functions.items()}
    declared_features = set(load_project(source_path).features)
    requested_features = {name for variant in variants for name in variant}
    unknown_features = requested_features - declared_features
    if unknown_features:
        raise ShaderAssetError("variant requests undeclared feature(s): " + ", ".join(sorted(unknown_features)))
    return ShaderPipelineDescriptor(
        pipeline_id,
        stages,
        variants,
        normalized_targets,
        source_path,
        _canonical_json(source_manifest),
        {module_id: module},
    )


def _native_module() -> Any:
    # Runtime owns native-module discovery so interactive and offline compilation
    # select the same packaged extension and development-build fallback.
    from . import runtime

    if runtime._native is None:
        raise ShaderAssetError(
            "shader cooking requires vernon_dsl._native; build the native extension or install a wheel containing it"
        )
    return runtime._native


def _native_target(native: Any, target: str) -> Any:
    targets = {
        "cpu": native.Target.CPU,
        "cuda": native.Target.CUDA,
        "vulkan": native.Target.VULKAN,
        "metal": native.Target.METAL,
        "opengl": native.Target.OPENGL,
        "opengles": native.Target.OPENGL_ES,
    }
    try:
        return targets[target]
    except KeyError:
        raise ShaderAssetError(f"target '{target}' is not available through the native compiler") from None


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
    reflected_options = stage.reflection.get("target_options")
    if not isinstance(symbol, str) or not symbol:
        raise ShaderAssetError(f"CPU compiler reflection has no exported symbol for {stage.entry}")
    if not isinstance(reflected_options, Mapping):
        raise ShaderAssetError("CPU compiler reflection has no target options")
    target_triple = reflected_options.get("target_triple")
    if not isinstance(target_triple, str) or not target_triple:
        raise ShaderAssetError("CPU compiler reflection has no normalized target triple")
    metadata: dict[str, Any] = {
        "symbol": symbol,
        "cpu_invocation_abi_version": 1,
        "target_triple": target_triple,
        "object_format": _cpu_object_format(stage.artifact.filename, target_triple),
    }
    for name in ("cpu", "cpu_features"):
        value = reflected_options.get(name)
        if isinstance(value, str):
            metadata[name] = value
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
    if mlir is None:
        mlir = compile_file(module.source, features=variant, entry=reference.entry)
    program = compiler.compile_program_result(mlir, native_target, **target.native_options)
    compiled = compiled_stage_from_program(
        program,
        module=module.id,
        module_manifest=module.canonical_manifest,
        entry=reference.entry,
        target=target,
    )
    if compiled.stage != stage:
        raise ShaderAssetError(f"compiler reflected {reference.entry} as {compiled.stage}, expected {stage}")
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
            raise ShaderAssetError("CPU cooking requires the compiler relocatable object artifact")
    return compiled


def cook_shader_pipeline(*, pipeline_asset: str | Path, output: str | Path, target: str = "opengl") -> Path:
    """Cook with the owning native compiler result."""
    source, descriptor_name = _pipeline_asset_reference(pipeline_asset)
    pipeline = parse_python_pipeline_asset(source, descriptor_name)
    modules = pipeline.modules
    if target not in pipeline.targets:
        raise ShaderAssetError(f"pipeline does not declare requested target '{target}'")
    if target == "cpu" and set(pipeline.stages) != {"compute"}:
        raise ShaderAssetError("CPU pipeline bundles support one compute stage and no graphics or barrier steps")
    target_options = TargetOptions(target, pipeline.targets[target])
    native = _native_module()
    native_target = _native_target(native, target)
    native_compiler = native.Compiler()
    selected_modules: dict[str, ShaderModuleDescriptor] = {}
    declared_features: set[str] = set()
    for stage, reference in pipeline.stages.items():
        module = modules.get(reference.module)
        if module is None:
            raise ShaderAssetError(f"pipeline {stage} stage references unknown module '{reference.module}'")
        selected_modules[stage] = module
        declared_features.update(load_project(module.source).features)
    for variant in pipeline.variants:
        unknown = set(variant) - declared_features
        if unknown:
            raise ShaderAssetError("variant requests undeclared feature(s): " + ", ".join(sorted(unknown)))

    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    planned_variants: list[tuple[tuple[str, ...], dict[str, CompiledStage]]] = []
    compile_cache: dict[tuple[str, str, str, str], CompiledStage] = {}
    for variant in pipeline.variants:
        stages_for_variant: dict[str, CompiledStage] = {}
        for stage, reference in pipeline.stages.items():
            mlir = compile_file(selected_modules[stage].source, features=variant, entry=reference.entry)
            cache_key = (
                reference.module,
                reference.entry,
                hashlib.sha256(mlir.encode("utf-8")).hexdigest(),
                canonical_json(
                    {
                        "target": target_options.target,
                        "options": dict(target_options.options),
                    }
                ),
            )
            compiled = compile_cache.get(cache_key)
            if compiled is None:
                compiled = _compile_stage(
                    selected_modules[stage],
                    reference,
                    stage,
                    variant,
                    target_options,
                    native_compiler,
                    native_target,
                    mlir,
                )
                compile_cache[cache_key] = compiled
            stages_for_variant[stage] = compiled
        planned_variants.append((variant, stages_for_variant))

    plan = build_bundle_plan(
        pipeline.id,
        target_options,
        sorted({feature for variant in pipeline.variants for feature in variant}),
        planned_variants,
    )
    descriptors = {
        stage.id: _write_external_artifact(
            output_path, stage.artifact.data, stage.artifact.format, stage.stage, stage.artifact.filename
        )
        for stage in plan.stages
    }
    pipeline_bundle = materialize_bundle(plan, descriptors)
    manifest_path = output_path / f"{output_path.name}.pipeline.json"
    manifest_path.write_text(
        json.dumps(pipeline_bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest_path
