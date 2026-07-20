from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compiler import compile_file
from .module_graph import load_project


class ShaderAssetError(ValueError):
    pass


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


def _canonical_json(value: Any) -> str:
    return json.dumps(value,
                      sort_keys=True,
                      separators=(",", ":"),
                      ensure_ascii=False)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShaderAssetError(
            f"cannot read shader manifest {path}: {error}") from None
    if not isinstance(value, dict):
        raise ShaderAssetError(f"shader manifest must be an object: {path}")
    return value


def parse_shader_module_manifest(path: str | Path) -> ShaderModuleDescriptor:
    manifest_path = Path(path).resolve()
    value = _read_manifest(manifest_path)
    if value.get("schema_version") != 1 or value.get(
            "type") != "shader_module":
        raise ShaderAssetError(
            f"invalid shader-module manifest header: {manifest_path}")
    module_id = value.get("id")
    source = value.get("source")
    if not isinstance(module_id, str) or not module_id or not isinstance(
            source, str) or not source:
        raise ShaderAssetError(
            f"shader-module manifest requires id and source: {manifest_path}")
    source_path = (manifest_path.parent / source).resolve()
    try:
        source_path.relative_to(manifest_path.parent.resolve())
    except ValueError:
        raise ShaderAssetError(
            f"shader module source escapes its manifest directory: {source}"
        ) from None
    if not source_path.is_file():
        raise ShaderAssetError(
            f"shader module source does not exist: {source}")
    return ShaderModuleDescriptor(module_id, source_path, manifest_path,
                                  _canonical_json(value))


def parse_shader_pipeline_manifest(
        path: str | Path) -> ShaderPipelineDescriptor:
    manifest_path = Path(path).resolve()
    value = _read_manifest(manifest_path)
    if value.get("schema_version") != 1 or value.get(
            "type") != "shader_pipeline":
        raise ShaderAssetError(
            f"invalid shader-pipeline manifest header: {manifest_path}")
    pipeline_id = value.get("id")
    raw_stages = value.get("stages")
    raw_variants = value.get("variants")
    targets = value.get("targets")
    if not isinstance(pipeline_id, str) or not pipeline_id:
        raise ShaderAssetError("shader pipeline id must be a non-empty string")
    if not isinstance(raw_stages, dict) or not isinstance(
            raw_variants, dict) or not isinstance(targets, dict):
        raise ShaderAssetError(
            "shader pipeline requires stages, variants, and targets")
    stage_names = set(raw_stages)
    graphics = stage_names == {"vertex", "fragment"}
    compute = stage_names == {"compute"}
    compute_graphics = stage_names == {"compute", "vertex", "fragment"}
    if not graphics and not compute and not compute_graphics:
        raise ShaderAssetError(
            "pipeline must contain vertex+fragment, compute+vertex+fragment, "
            "or one compute stage")
    stages: dict[str, ShaderStageReference] = {}
    for stage, reference in raw_stages.items():
        if not isinstance(reference, dict) or not isinstance(
                reference.get("module"), str) or not isinstance(
                    reference.get("entry"), str):
            raise ShaderAssetError(f"invalid {stage} stage reference")
        stages[stage] = ShaderStageReference(reference["module"],
                                             reference["entry"])
    include = raw_variants.get("include")
    max_variants = raw_variants.get("max_variants", 16)
    if not isinstance(include, list) or not isinstance(
            max_variants, int) or max_variants <= 0:
        raise ShaderAssetError(
            "variants.include and a positive max_variants are required")
    if len(include) > max_variants:
        raise ShaderAssetError(
            f"pipeline declares {len(include)} variants, exceeding cap {max_variants}"
        )
    variants: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for raw_key in include:
        if not isinstance(raw_key, list) or any(
                not isinstance(name, str) or not name for name in raw_key):
            raise ShaderAssetError(
                "each included variant must be a list of feature names")
        key = tuple(sorted(set(raw_key)))
        if len(key) != len(raw_key) or key in seen:
            raise ShaderAssetError(
                f"duplicate or non-canonical variant: {raw_key}")
        seen.add(key)
        variants.append(key)
    if not variants:
        raise ShaderAssetError("pipeline must include at least one variant")
    normalized_targets: dict[str, dict[str, Any]] = {}
    for target, options in targets.items():
        if not isinstance(target, str) or not isinstance(options, dict):
            raise ShaderAssetError("pipeline target options must be objects")
        normalized_targets[target] = options
    return ShaderPipelineDescriptor(pipeline_id, stages, tuple(variants),
                                    normalized_targets, manifest_path,
                                    _canonical_json(value))


def discover_shader_modules(
        asset_root: str | Path) -> dict[str, ShaderModuleDescriptor]:
    root = Path(asset_root).resolve()
    modules: dict[str, ShaderModuleDescriptor] = {}
    for path in sorted(root.rglob("*.shader-module.json")):
        descriptor = parse_shader_module_manifest(path)
        if descriptor.id in modules:
            raise ShaderAssetError(
                f"duplicate shader module id: {descriptor.id}")
        modules[descriptor.id] = descriptor
    return modules


def _entry_reflection(reflection: dict[str, Any],
                      entry: str) -> dict[str, Any]:
    matches = [
        value for value in reflection.get("entries", [])
        if isinstance(value, dict) and value.get("name") == entry
    ]
    if len(matches) != 1:
        raise ShaderAssetError(
            f"compiler reflection does not contain exactly one '{entry}' entry"
        )
    return matches[0]


def _compile_stage(module: ShaderModuleDescriptor,
                   reference: ShaderStageReference, stage: str,
                   variant: tuple[str,
                                  ...], target: str, target_options: dict[str,
                                                                          Any],
                   compiler: Path, output: Path) -> tuple[str, dict[str, Any]]:
    mlir = compile_file(module.source, features=variant, entry=reference.entry)
    with tempfile.TemporaryDirectory(
            prefix="vernon_shader_stage_") as directory:
        temporary = Path(directory)
        mlir_path = temporary / "stage.mlir"
        artifact_directory = temporary / "artifacts"
        mlir_path.write_text(mlir, encoding="utf-8", newline="\n")
        command = [
            str(compiler), "--target", target,
            str(mlir_path), "--output-dir",
            str(artifact_directory)
        ]
        glsl_version = target_options.get("glsl_version", 0)
        if glsl_version:
            if not isinstance(glsl_version, int):
                raise ShaderAssetError("glsl_version must be an integer")
            command.extend(("--glsl-version", str(glsl_version)))
        result = subprocess.run(command,
                                capture_output=True,
                                text=True,
                                check=False)
        if result.returncode != 0:
            diagnostics = result.stderr.strip() or result.stdout.strip()
            raise ShaderAssetError(
                f"native compilation failed for {reference.entry}: {diagnostics}"
            )
        reflection_path = artifact_directory / "reflection.json"
        reflection = _read_manifest(reflection_path)
        artifact_rows = [
            value for value in reflection.get("artifacts", [])
            if isinstance(value, dict) and value.get("entry_point") ==
            reference.entry and value.get("stage") == stage
        ]
        if len(artifact_rows) != 1:
            raise ShaderAssetError(
                f"compiler did not emit exactly one {stage} artifact for {reference.entry}"
            )
        artifact_row = artifact_rows[0]
        artifact_name = artifact_row.get("filename")
        if not isinstance(artifact_name, str):
            raise ShaderAssetError("compiler artifact filename is invalid")
        artifact_data = (artifact_directory / artifact_name).read_bytes()
        entry_reflection = _entry_reflection(reflection, reference.entry)
        identity = {
            "cache_version": 1,
            "compiler_version": 1,
            "module": module.id,
            "module_manifest": module.canonical_manifest,
            "entry": reference.entry,
            "stage": stage,
            "target": target,
            "target_options": target_options,
            "dependencies": reflection.get("dependencies", []),
            "interface": entry_reflection,
            "artifact_sha256": hashlib.sha256(artifact_data).hexdigest(),
        }
        stage_id = hashlib.sha256(
            _canonical_json(identity).encode("utf-8")).hexdigest()
        suffix = Path(artifact_name).suffix
        cooked_filename = f"stages/{stage_id}.{stage}{suffix}"
        cooked_path = output / cooked_filename
        cooked_path.parent.mkdir(parents=True, exist_ok=True)
        if not cooked_path.exists():
            cooked_path.write_bytes(artifact_data)
        record = {
            "id": stage_id,
            "module": module.id,
            "entry": reference.entry,
            "stage": stage,
            "target": target,
            "format": artifact_row.get("format"),
            "filename": cooked_filename,
            "module_hash": reflection.get("module_hash"),
            "dependencies": reflection.get("dependencies", []),
            "interface": entry_reflection,
        }
        return stage_id, record


def _interface_by_location(values: list[dict[str, Any]],
                           interface: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        if value.get(
                "vernon.interface") != interface or "vernon.builtin" in value:
            continue
        location = value.get("vernon.location")
        value_type = value.get("type")
        if isinstance(location, int) and isinstance(value_type, str):
            result[location] = value_type
    return result


def _validate_graphics_interfaces(vertex: dict[str, Any],
                                  fragment: dict[str, Any]) -> None:
    outputs = _interface_by_location(vertex["interface"].get("results", []),
                                     "output")
    inputs = _interface_by_location(fragment["interface"].get("arguments", []),
                                    "input")
    for location, value_type in inputs.items():
        if outputs.get(location) != value_type:
            raise ShaderAssetError(
                f"vertex/fragment interface mismatch at location {location}")


def cook_shader_pipeline(*,
                         pipeline_manifest: str | Path,
                         asset_root: str | Path,
                         compiler: str | Path,
                         output: str | Path,
                         target: str = "opengl") -> Path:
    pipeline = parse_shader_pipeline_manifest(pipeline_manifest)
    modules = discover_shader_modules(asset_root)
    compiler_path = Path(compiler).resolve()
    if not compiler_path.is_file():
        raise ShaderAssetError(
            f"native Vernon compiler does not exist: {compiler_path}")
    if target not in pipeline.targets:
        raise ShaderAssetError(
            f"pipeline does not declare requested target '{target}'")
    target_options = pipeline.targets[target]
    selected_modules: dict[str, ShaderModuleDescriptor] = {}
    declared_features: set[str] = set()
    for stage, reference in pipeline.stages.items():
        module = modules.get(reference.module)
        if module is None:
            raise ShaderAssetError(
                f"pipeline {stage} stage references unknown module '{reference.module}'"
            )
        selected_modules[stage] = module
        declared_features.update(load_project(module.source).features)
    for variant in pipeline.variants:
        unknown = set(variant) - declared_features
        if unknown:
            raise ShaderAssetError("variant requests undeclared feature(s): " +
                                   ", ".join(sorted(unknown)))

    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    stage_records: dict[str, dict[str, Any]] = {}
    variants: list[dict[str, Any]] = []
    compile_cache: dict[tuple[str, str, tuple[str, ...]],
                        tuple[str, dict[str, Any]]] = {}
    for variant in pipeline.variants:
        mapping: dict[str, str] = {}
        records_for_variant: dict[str, dict[str, Any]] = {}
        for stage, reference in pipeline.stages.items():
            cache_key = (reference.module, reference.entry, variant)
            compiled = compile_cache.get(cache_key)
            if compiled is None:
                compiled = _compile_stage(selected_modules[stage], reference,
                                          stage, variant, target,
                                          target_options, compiler_path,
                                          output_path)
                compile_cache[cache_key] = compiled
            stage_id, record = compiled
            stage_records.setdefault(stage_id, record)
            mapping[stage] = stage_id
            records_for_variant[stage] = record
        if {"vertex", "fragment"}.issubset(pipeline.stages):
            _validate_graphics_interfaces(records_for_variant["vertex"],
                                          records_for_variant["fragment"])
        variants.append({"key": list(variant), "stages": mapping})

    bundle: dict[str, Any] = {
        "schema_version": 2,
        "type": "compiled_shader_bundle",
        "id": pipeline.id,
        "target": target,
        "target_options": target_options,
        "features": sorted(declared_features),
        "variants": variants,
        "stage_artifacts": {
            key: stage_records[key]
            for key in sorted(stage_records)
        },
        "source_manifest": json.loads(pipeline.canonical_manifest),
    }
    bundle["content_hash"] = hashlib.sha256(
        _canonical_json(bundle).encode("utf-8")).hexdigest()
    manifest_path = output_path / "shader.json"
    manifest_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) +
        "\n",
        encoding="utf-8",
        newline="\n")
    return manifest_path
