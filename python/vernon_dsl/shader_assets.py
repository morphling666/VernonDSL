from __future__ import annotations

import ast
import base64
import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .compiler import compile_file
from .module_graph import load_project
from .types import Feature


class ShaderAssetError(ValueError):
    pass


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
    variants: Iterable[Iterable[Feature]] = ((), ),
    targets: Mapping[str, Mapping[str, Any]],
) -> PipelineAssetDeclaration:
    """Declare a cookable pipeline without creating a runtime pipeline.

    The cooker reads this call from the source AST and does not execute it.
    This implementation exists for editor completion and normal Python imports.
    """
    stages = {
        name: value
        for name, value in (("compute", compute), ("vertex", vertex),
                            ("fragment", fragment)) if value is not None
    }
    return PipelineAssetDeclaration(
        id=id,
        stages=stages,
        variants=tuple(tuple(key) for key in variants),
        targets={
            name: dict(options)
            for name, options in targets.items()
        },
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
    modules: dict[str, ShaderModuleDescriptor] | None = None


def _canonical_json(value: Any) -> str:
    return json.dumps(value,
                      sort_keys=True,
                      separators=(",", ":"),
                      ensure_ascii=False)


def _with_content_hash(document: dict[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("content_hash", None)
    result["content_hash"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")).hexdigest()
    return result


def encode_runtime_stage(record: dict[str, Any],
                         artifact: bytes) -> dict[str, Any]:
    artifact_format = record.get("format")
    if not isinstance(artifact_format, str) or not artifact_format:
        raise ShaderAssetError("runtime stage artifact format is missing")
    digest = hashlib.sha256(artifact).hexdigest()
    encoding = "base64" if artifact_format == "spirv" else "utf8"
    try:
        data = (base64.b64encode(artifact).decode("ascii")
                if encoding == "base64" else artifact.decode("utf-8"))
    except UnicodeDecodeError:
        raise ShaderAssetError(
            f"{artifact_format} runtime artifact is not UTF-8") from None
    return {
        **record,
        "artifact": {
            "format": artifact_format,
            "storage": "inline",
            "encoding": encoding,
            "data": data,
            "size": len(artifact),
            "sha256": digest,
        },
    }


def serialize_runtime_pipeline_bundle(bundle: dict[str, Any]) -> bytes:
    return (_canonical_json(_with_content_hash(bundle)) + "\n").encode("utf-8")


def _artifact_extension(artifact_format: str, stage: str,
                        original_name: str = "") -> str:
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
    suffix = Path(original_name).suffix
    return suffix or f".{artifact_format}"


def _write_external_artifact(output: Path, artifact: bytes,
                             artifact_format: str, stage: str,
                             original_name: str = "") -> dict[str, Any]:
    digest = hashlib.sha256(artifact).hexdigest()
    extension = _artifact_extension(artifact_format, stage, original_name)
    relative_path = Path("artifacts") / f"{digest}{extension}"
    artifact_path = output / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        if artifact_path.read_bytes() != artifact:
            raise ShaderAssetError(
                f"content-addressed artifact collision: {relative_path}")
    else:
        artifact_path.write_bytes(artifact)
    return {
        "format": artifact_format,
        "storage": "external",
        "path": relative_path.as_posix(),
        "size": len(artifact),
        "sha256": digest,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShaderAssetError(
            f"cannot read shader manifest {path}: {error}") from None
    if not isinstance(value, dict):
        raise ShaderAssetError(f"shader manifest must be an object: {path}")
    return value


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _pipeline_source_reference(
        value: str | Path) -> tuple[Path, str] | None:
    spelling = str(value)
    marker = spelling.rfind(".py:")
    if marker < 0:
        return None
    source = Path(spelling[:marker + 3]).resolve()
    descriptor_name = spelling[marker + 4:]
    if not descriptor_name:
        raise ShaderAssetError(
            "Python pipeline asset reference requires a descriptor name")
    return source, descriptor_name


def _feature_bindings(tree: ast.Module) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if not isinstance(value, ast.Call) or _call_name(
                value.func).split(".")[-1] != "feature":
            continue
        if len(value.args) != 1 or value.keywords or not isinstance(
                value.args[0], ast.Constant) or not isinstance(
                    value.args[0].value, str):
            raise ShaderAssetError(
                "feature declarations used by pipeline assets require one string literal"
            )
        targets = statement.targets if isinstance(statement,
                                                   ast.Assign) else [
                                                       statement.target
                                                   ]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            raise ShaderAssetError(
                "feature declarations used by pipeline assets require a simple name"
            )
        bindings[targets[0].id] = value.args[0].value
    return bindings


def _literal_keyword(keywords: dict[str, ast.expr], name: str) -> Any:
    node = keywords.get(name)
    if node is None:
        raise ShaderAssetError(
            f"pipeline_asset declaration requires '{name}'")
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        raise ShaderAssetError(
            f"pipeline_asset '{name}' must be a literal value") from None


def parse_python_pipeline_asset(
        source: str | Path, descriptor_name: str) -> ShaderPipelineDescriptor:
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise ShaderAssetError(
            f"pipeline asset source does not exist: {source_path}")
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"),
                         filename=str(source_path))
    except (OSError, SyntaxError) as error:
        raise ShaderAssetError(
            f"cannot parse pipeline asset source {source_path}: {error}") from None

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
        if (len(targets) == 1 and isinstance(targets[0], ast.Name)
                and targets[0].id == descriptor_name
                and isinstance(value, ast.Call)
                and _call_name(value.func).split(".")[-1] == "pipeline_asset"):
            if declaration is not None:
                raise ShaderAssetError(
                    f"duplicate pipeline asset declaration: {descriptor_name}")
            declaration = value
    if declaration is None:
        raise ShaderAssetError(
            f"pipeline asset declaration '{descriptor_name}' was not found in {source_path}"
        )
    if declaration.args:
        raise ShaderAssetError(
            "pipeline_asset accepts keyword arguments only")
    keywords = {
        keyword.arg: keyword.value
        for keyword in declaration.keywords if keyword.arg is not None
    }
    if len(keywords) != len(declaration.keywords):
        raise ShaderAssetError(
            "pipeline_asset does not accept expanded keyword arguments")
    supported = {
        "id", "compute", "vertex", "fragment", "variants", "targets"
    }
    unknown = set(keywords) - supported
    if unknown:
        raise ShaderAssetError("unknown pipeline_asset argument(s): " +
                               ", ".join(sorted(unknown)))

    pipeline_id = _literal_keyword(keywords, "id")
    if not isinstance(pipeline_id, str) or not pipeline_id:
        raise ShaderAssetError("pipeline asset id must be a non-empty string")

    stage_functions: dict[str, str] = {}
    for stage in ("compute", "vertex", "fragment"):
        node = keywords.get(stage)
        if node is None:
            continue
        if not isinstance(node, ast.Name):
            raise ShaderAssetError(
                f"pipeline_asset {stage} must reference a function in the same module"
            )
        stage_functions[stage] = node.id
    stage_names = set(stage_functions)
    if stage_names not in ({"compute"}, {"vertex", "fragment"},
                            {"compute", "vertex", "fragment"}):
        raise ShaderAssetError(
            "pipeline must contain vertex+fragment, compute+vertex+fragment, or one compute stage"
        )

    definitions = {
        statement.name: {
            _call_name(decorator.func if isinstance(decorator, ast.Call) else
                       decorator).split(".")[-1]
            for decorator in statement.decorator_list
        }
        for statement in tree.body if isinstance(statement, ast.FunctionDef)
    }
    expected_decorators = {
        "compute": {
            "compute", "kernel"
        },
        "vertex": {
            "vertex"
        },
        "fragment": {
            "fragment"
        },
    }
    for stage, entry in stage_functions.items():
        if entry not in definitions or not (
                definitions[entry] & expected_decorators[stage]):
            raise ShaderAssetError(
                f"pipeline_asset {stage} entry '{entry}' has the wrong stage decorator"
            )

    features = _feature_bindings(tree)
    variants_node = keywords.get("variants")
    if variants_node is None:
        variants: tuple[tuple[str, ...], ...] = ((), )
    elif not isinstance(variants_node, (ast.Tuple, ast.List)):
        raise ShaderAssetError(
            "pipeline_asset variants must be a tuple or list")
    else:
        parsed_variants: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for variant_node in variants_node.elts:
            if not isinstance(variant_node, (ast.Tuple, ast.List)):
                raise ShaderAssetError(
                    "each pipeline asset variant must be a tuple or list")
            names: list[str] = []
            for feature_node in variant_node.elts:
                if not isinstance(feature_node,
                                  ast.Name) or feature_node.id not in features:
                    raise ShaderAssetError(
                        "pipeline asset variants must reference locally declared features"
                    )
                names.append(features[feature_node.id])
            key = tuple(names)
            if list(key) != sorted(key) or len(set(key)) != len(key):
                raise ShaderAssetError(
                    f"pipeline asset variant is not canonical: {list(key)}")
            if key in seen:
                raise ShaderAssetError(
                    f"duplicate pipeline asset variant: {list(key)}")
            seen.add(key)
            parsed_variants.append(key)
        variants = tuple(parsed_variants)
    if not variants:
        raise ShaderAssetError("pipeline must include at least one variant")
    if len(variants) > 16:
        raise ShaderAssetError(
            f"pipeline declares {len(variants)}, exceeding variant cap 16")

    targets = _literal_keyword(keywords, "targets")
    if not isinstance(targets, dict) or not targets:
        raise ShaderAssetError(
            "pipeline_asset targets must be a non-empty dictionary")
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
    module = ShaderModuleDescriptor(module_id, source_path, source_path,
                                    _canonical_json(source_manifest))
    stages = {
        stage: ShaderStageReference(module_id, entry)
        for stage, entry in stage_functions.items()
    }
    declared_features = set(load_project(source_path).features)
    requested_features = {
        name
        for variant in variants for name in variant
    }
    unknown_features = requested_features - declared_features
    if unknown_features:
        raise ShaderAssetError("variant requests undeclared feature(s): " +
                               ", ".join(sorted(unknown_features)))
    return ShaderPipelineDescriptor(
        pipeline_id,
        stages,
        variants,
        normalized_targets,
        source_path,
        _canonical_json(source_manifest),
        {module_id: module},
    )


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
    source_reference = _pipeline_source_reference(path)
    if source_reference is not None:
        return parse_python_pipeline_asset(*source_reference)
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
        command = [str(compiler), "--target", target, str(mlir_path)]
        if target == "cpu":
            command.extend(("--compute-bundle", str(artifact_directory)))
        else:
            command.extend(("--output-dir", str(artifact_directory)))
        glsl_version = target_options.get("glsl_version", 0)
        if glsl_version:
            if not isinstance(glsl_version, int):
                raise ShaderAssetError("glsl_version must be an integer")
            command.extend(("--glsl-version", str(glsl_version)))
        result = subprocess.run(command,
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                                check=False)
        if result.returncode != 0:
            diagnostics = result.stderr.strip() or result.stdout.strip()
            raise ShaderAssetError(
                f"native compilation failed for {reference.entry}: {diagnostics}"
            )
        if target == "cpu":
            manifest = _read_manifest(artifact_directory / "compute.json")
            symbol = manifest.get("symbol")
            artifact_name = manifest.get("artifact")
            artifact_size = manifest.get("artifact_size")
            artifact_sha256 = manifest.get("artifact_sha256")
            operating_system = manifest.get("operating_system")
            architecture = manifest.get("architecture")
            abi_version = manifest.get("cpu_invocation_abi_version")
            if (manifest.get("schema_version") != 2
                    or manifest.get("target") != "cpu"
                    or manifest.get("artifact_format") != "native_library"
                    or manifest.get("entry") != reference.entry
                    or symbol != f"__vernon_cpu_{reference.entry}"
                    or not isinstance(artifact_name, str) or not artifact_name
                    or not isinstance(artifact_size, int)
                    or not isinstance(artifact_sha256, str)
                    or len(artifact_sha256) != 64
                    or not isinstance(operating_system, str)
                    or not isinstance(architecture, str)
                    or not isinstance(abi_version, int)):
                raise ShaderAssetError(
                    f"compiler emitted an invalid CPU bundle for {reference.entry}"
                )
            relative_artifact = Path(artifact_name)
            if relative_artifact.is_absolute(
            ) or ".." in relative_artifact.parts:
                raise ShaderAssetError(
                    "CPU bundle artifact path must stay inside its bundle")
            artifact_data = (artifact_directory /
                             relative_artifact).read_bytes()
            digest = hashlib.sha256(artifact_data).hexdigest()
            if len(artifact_data
                   ) != artifact_size or digest != artifact_sha256:
                raise ShaderAssetError(
                    "CPU bundle artifact size or SHA-256 does not match its manifest"
                )
            reflection = manifest.get("reflection")
            if not isinstance(reflection, dict):
                raise ShaderAssetError(
                    "CPU bundle reflection must be an object")
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
                "artifact_sha256": digest,
            }
            stage_id = hashlib.sha256(
                _canonical_json(identity).encode("utf-8")).hexdigest()
            artifact = _write_external_artifact(
                output, artifact_data, "native_library", stage, artifact_name)
            record = {
                "id": stage_id,
                "module": module.id,
                "entry": reference.entry,
                "stage": stage,
                "target": target,
                "format": "native_library",
                "artifact": artifact,
                "symbol": symbol,
                "operating_system": operating_system,
                "architecture": architecture,
                "cpu_invocation_abi_version": abi_version,
                "module_hash": reflection.get("module_hash"),
                "dependencies": reflection.get("dependencies", []),
                "interface": entry_reflection,
                "reflection": reflection,
            }
            return stage_id, record
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
        artifact_format = artifact_row.get("format")
        if not isinstance(artifact_format, str) or not artifact_format:
            raise ShaderAssetError("compiler artifact format is invalid")
        artifact = _write_external_artifact(output, artifact_data,
                                            artifact_format, stage,
                                            artifact_name)
        record = {
            "id": stage_id,
            "module": module.id,
            "entry": reference.entry,
            "stage": stage,
            "target": target,
            "format": artifact_format,
            "artifact": artifact,
            "module_hash": reflection.get("module_hash"),
            "dependencies": reflection.get("dependencies", []),
            "interface": entry_reflection,
            "reflection": reflection,
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


def _dtype_and_shape(type_name: object) -> tuple[str | None, list[int]]:
    if not isinstance(type_name, str):
        return None, []
    if not type_name.startswith("tensor<") or not type_name.endswith(">"):
        return type_name, []
    parts = type_name[7:-1].split("x")
    if not parts:
        return None, []
    return parts[-1], [
        0 if dimension == "?" else int(dimension)
        for dimension in parts[:-1]
    ]


def _external_parameters(
        records: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    parameters: dict[str, list[dict[str, Any]]] = {}
    for stage in ("compute", "vertex", "fragment"):
        record = records.get(stage)
        if record is None:
            continue
        for row in record["interface"].get("arguments", []):
            if "vernon.builtin" in row or row.get("vernon.varying", False):
                continue
            name = row.get("vernon.source_name")
            interface = row.get("vernon.interface")
            if not isinstance(name, str) or not name or not isinstance(
                    interface, str):
                raise ShaderAssetError(
                    f"{stage} external argument is missing source metadata")
            inferred_dtype, inferred_shape = _dtype_and_shape(row.get("type"))
            use = {
                "stage": stage,
                "entry": record["entry"],
                "index": row.get("index"),
                "kind": row.get("kind", "scalar"),
                "type": row.get("type"),
                "dtype": (row.get("dtype") or row.get("vernon.dtype")
                          or inferred_dtype),
                "shape": row.get("shape", inferred_shape),
                "interface": interface,
                "access": row.get("access", "read"),
            }
            for key in ("vernon.location", "vernon.instance_divisor",
                        "vernon.set", "vernon.binding"):
                if key in row:
                    use[key] = row[key]
            if interface == "uniform":
                use["uniform_name"] = (
                    f"{record['entry']}_arg_{int(row['index'])}._m0")
            parameters.setdefault(name, []).append(use)
    return parameters


def _merge_parameter_uses(name: str, uses: list[dict[str,
                                                     Any]]) -> dict[str, Any]:
    first = uses[0]
    kind = "texture" if first["kind"] == "texture" else (
        "inline" if first["interface"] == "uniform" or
        (first["stage"] == "compute" and first["kind"] != "tensor") else
        "tensor")
    for use in uses[1:]:
        other_kind = "texture" if use["kind"] == "texture" else (
            "inline" if use["interface"] == "uniform" or
            (use["stage"] == "compute" and use["kind"] != "tensor") else
            "tensor")
        if (other_kind != kind or use.get("type") != first.get("type")
                or use.get("dtype") != first.get("dtype")
                or use.get("shape", []) != first.get("shape", [])):
            raise ShaderAssetError(f"incompatible pipeline parameter {name!r}")
    access_values = {str(use.get("access", "read")) for use in uses}
    access = ("read_write" if "read_write" in access_values
              or access_values == {"read", "write"} else next(
                  iter(access_values)))
    return {
        "name": name,
        "kind": kind,
        "type": first.get("type"),
        "dtype": first.get("dtype"),
        "shape": first.get("shape", []),
        "access": access,
        "uses": uses,
    }


def _fragment_outputs(
        records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fragment = records.get("fragment")
    if fragment is None:
        return []
    outputs: list[dict[str, Any]] = []
    for row in fragment["interface"].get("results", []):
        if "vernon.location" not in row:
            continue
        location = row["vernon.location"]
        type_name = row.get("type")
        dtype, shape = _dtype_and_shape(type_name)
        outputs.append({
            "name": row.get("vernon.source_name") or f"output_{location}",
            "kind": "texture",
            "dtype": dtype,
            "shape": shape,
            "access": "write",
            "location": location,
            "type": type_name,
        })
    return outputs


def cook_shader_pipeline(*,
                         pipeline_manifest: str | Path,
                         asset_root: str | Path,
                         compiler: str | Path,
                         output: str | Path,
                         target: str = "opengl") -> Path:
    pipeline = parse_shader_pipeline_manifest(pipeline_manifest)
    modules = pipeline.modules or discover_shader_modules(asset_root)
    compiler_path = Path(compiler).resolve()
    if not compiler_path.is_file():
        raise ShaderAssetError(
            f"native Vernon compiler does not exist: {compiler_path}")
    if target not in pipeline.targets:
        raise ShaderAssetError(
            f"pipeline does not declare requested target '{target}'")
    if target == "cpu" and set(pipeline.stages) != {"compute"}:
        raise ShaderAssetError(
            "CPU pipeline bundles support one compute stage and no graphics "
            "or barrier steps")
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
    records_by_variant: list[dict[str, dict[str, Any]]] = []
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
        records_by_variant.append(records_for_variant)

    parameter_names = sorted({
        name
        for records in records_by_variant
        for name in _external_parameters(records)
    })
    slots = {name: slot for slot, name in enumerate(parameter_names)}
    pipeline_variants: list[dict[str, Any]] = []
    for variant, records in zip(pipeline.variants, records_by_variant):
        external = _external_parameters(records)
        parameters = []
        for name in parameter_names:
            uses = external.get(name)
            if uses is None:
                continue
            parameter = _merge_parameter_uses(name, uses)
            parameter["slot"] = slots[name]
            parameters.append(parameter)
        steps: list[dict[str, Any]] = []
        if "compute" in records:
            steps.append({
                "kind":
                "dispatch",
                "stage":
                variants[len(pipeline_variants)]["stages"]["compute"],
            })
        if "compute" in records and "vertex" in records:
            steps.append({
                "kind": "barrier",
                "source": "compute_write",
                "destination": "vertex_read",
            })
        if "vertex" in records:
            steps.append({
                "kind":
                "draw",
                "vertex":
                variants[len(pipeline_variants)]["stages"]["vertex"],
                "fragment":
                variants[len(pipeline_variants)]["stages"]["fragment"],
            })
        pipeline_variants.append({
            "key": list(variant),
            "parameters": parameters,
            "outputs": _fragment_outputs(records),
            "steps": steps,
        })

    pipeline_bundle: dict[str, Any] = {
        "schema_version": 2,
        "invocation_abi_version": 1,
        "type": "pipeline",
        "id": pipeline.id,
        "target": target,
        "target_options": target_options,
        "features": sorted(declared_features),
        "variants": pipeline_variants,
        "stage_artifacts": {
            key: stage_records[key]
            for key in sorted(stage_records)
        },
    }
    pipeline_bundle = _with_content_hash(pipeline_bundle)
    manifest_path = output_path / f"{output_path.name}.pipeline.json"
    manifest_path.write_text(
        json.dumps(pipeline_bundle,
                   indent=2,
                   sort_keys=True,
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n")
    legacy_manifest = output_path / "shader.json"
    if legacy_manifest.exists():
        legacy_manifest.unlink()
    return manifest_path
