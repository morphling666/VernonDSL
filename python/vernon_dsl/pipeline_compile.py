from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class PipelineCompileError(ValueError):
    pass


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({
        key: value[key]
        for key in sorted(value)
    })


def canonical_json(value: Any) -> str:
    return json.dumps(value,
                      sort_keys=True,
                      separators=(",", ":"),
                      ensure_ascii=False)


def content_hash(value: Mapping[str, Any]) -> str:
    unhashed = dict(value)
    unhashed.pop("content_hash", None)
    return hashlib.sha256(
        canonical_json(unhashed).encode("utf-8")).hexdigest()


def with_content_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_hash", None)
    result["content_hash"] = content_hash(result)
    return result


@dataclass(frozen=True)
class TargetOptions:
    target: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target:
            raise PipelineCompileError("target must be a non-empty string")
        options = dict(self.options)
        glsl_version = options.get("glsl_version")
        if glsl_version is not None:
            if self.target not in {"opengl", "opengles"}:
                raise PipelineCompileError(
                    "glsl_version is valid only for OpenGL targets")
            if (not isinstance(glsl_version, int)
                    or isinstance(glsl_version, bool) or glsl_version <= 0):
                raise PipelineCompileError(
                    "glsl_version must be a positive integer")
        for name in ("target_triple", "cpu", "cpu_features"):
            value = options.get(name)
            if value is not None and not isinstance(value, str):
                raise PipelineCompileError(f"{name} must be a string")
        object.__setattr__(self, "options", _frozen_mapping(options))

    @property
    def native_options(self) -> dict[str, Any]:
        return {
            "glsl_version": int(self.options.get("glsl_version", 0)),
            "target_triple": str(self.options.get("target_triple", "")),
            "cpu": str(self.options.get("cpu", "")),
            "cpu_features": str(self.options.get("cpu_features", "")),
        }


@dataclass(frozen=True)
class CompiledArtifact:
    format: str
    data: bytes
    filename: str = ""

    def __post_init__(self) -> None:
        if not self.format:
            raise PipelineCompileError("compiled artifact format is missing")
        object.__setattr__(self, "data", bytes(self.data))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class CompiledStage:
    module: str
    module_manifest: str
    entry: str
    stage: str
    target: TargetOptions
    reflection: Mapping[str, Any]
    interface: Mapping[str, Any]
    artifact: CompiledArtifact
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entry or not self.stage:
            raise PipelineCompileError("compiled stage requires entry and stage")
        object.__setattr__(self, "reflection",
                           _frozen_mapping(self.reflection))
        object.__setattr__(self, "interface", _frozen_mapping(self.interface))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "cache_version": 2,
            "compiler_version": 1,
            "module": self.module,
            "entry": self.entry,
            "stage": self.stage,
            "target": self.target.target,
            "target_options": dict(self.target.options),
            "dependencies": self.reflection.get("dependencies", []),
            "interface": dict(self.interface),
            "artifact_sha256": self.artifact.sha256,
        }

    @property
    def id(self) -> str:
        return hashlib.sha256(
            canonical_json(self.identity).encode("utf-8")).hexdigest()

    def logical_record(self) -> dict[str, Any]:
        record = {
            "id": self.id,
            "module": self.module,
            "entry": self.entry,
            "stage": self.stage,
            "target": self.target.target,
            "format": self.artifact.format,
            "module_hash": self.reflection.get("module_hash"),
            "dependencies": self.reflection.get("dependencies", []),
            "interface": dict(self.interface),
            "reflection": dict(self.reflection),
            **dict(self.metadata),
        }
        if not self.module:
            record.pop("module")
        return record


@dataclass(frozen=True)
class VariantPlan:
    key: tuple[str, ...]
    stages: Mapping[str, str]
    parameters: tuple[Mapping[str, Any], ...]
    internal_parameters: tuple[Mapping[str, Any], ...]
    outputs: tuple[Mapping[str, Any], ...]
    steps: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", _frozen_mapping(self.stages))
        for name in ("parameters", "internal_parameters", "outputs", "steps"):
            values = tuple(_frozen_mapping(value)
                           for value in getattr(self, name))
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "key": list(self.key),
            "parameters": [dict(value) for value in self.parameters],
            "outputs": [dict(value) for value in self.outputs],
            "steps": [dict(value) for value in self.steps],
        }
        if self.internal_parameters:
            result["internal_parameters"] = [
                dict(value) for value in self.internal_parameters
            ]
        return result


@dataclass(frozen=True)
class BundlePlan:
    pipeline_id: str
    target: TargetOptions
    features: tuple[str, ...]
    variants: tuple[VariantPlan, ...]
    stages: tuple[CompiledStage, ...]

    def logical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "invocation_abi_version": 3,
            "type": "pipeline",
            "id": self.pipeline_id,
            "target": self.target.target,
            "target_options": dict(self.target.options),
            "features": list(self.features),
            "variants": [variant.to_dict() for variant in self.variants],
            "stage_artifacts": {
                stage.id: stage.logical_record()
                for stage in sorted(self.stages, key=lambda value: value.id)
            },
        }


def parse_reflection_json(reflection: str | bytes |
                          Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(reflection, Mapping):
        return dict(reflection)
    try:
        value = json.loads(reflection)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PipelineCompileError(
            f"compiler reflection is invalid JSON: {error}") from None
    if not isinstance(value, dict):
        raise PipelineCompileError("compiler reflection must be an object")
    return value


def select_entry(reflection: Mapping[str, Any],
                 entry: str) -> dict[str, Any]:
    matches = [
        value for value in reflection.get("entries", [])
        if isinstance(value, dict) and value.get("name") == entry
    ]
    if len(matches) != 1:
        raise PipelineCompileError(
            f"compiler reflection does not contain exactly one '{entry}' entry"
        )
    return matches[0]


def select_artifact(reflection: Mapping[str, Any], entry: str,
                    stage: str) -> dict[str, Any]:
    matches = [
        value for value in reflection.get("artifacts", [])
        if isinstance(value, dict) and value.get("entry_point") == entry
        and value.get("stage") == stage
    ]
    if len(matches) != 1:
        raise PipelineCompileError(
            f"compiler did not emit exactly one {stage} artifact for {entry}")
    return matches[0]


def select_artifact_bytes(
    artifacts: Sequence[tuple[str, bytes]],
    artifact_row: Mapping[str, Any],
) -> tuple[str, bytes]:
    filename = artifact_row.get("filename")
    if not isinstance(filename, str) or not filename:
        raise PipelineCompileError("compiler artifact filename is invalid")
    matches = [(name, bytes(data)) for name, data in artifacts
               if name == filename]
    if len(matches) != 1:
        raise PipelineCompileError(
            f"compiler produced no unique artifact {filename!r}")
    return matches[0]


def compiled_stage_from_program(
    program: Any,
    *,
    module: str,
    module_manifest: str,
    entry: str,
    target: TargetOptions,
    metadata: Mapping[str, Any] = MappingProxyType({}),
) -> CompiledStage:
    """Normalize one owning compiler result into the shared stage model."""
    if not bool(program.ok):
        raise PipelineCompileError(
            str(program.diagnostics) or f"native compilation failed for {entry}")
    reflection = parse_reflection_json(program.reflection)
    interface = select_entry(reflection, entry)
    stage = interface.get("stage")
    if not isinstance(stage, str) or not stage:
        raise PipelineCompileError(
            f"compiler reflection has no stage for {entry}")
    artifact_row = select_artifact(reflection, entry, stage)
    artifact_name, artifact_data = select_artifact_bytes(
        program.artifacts, artifact_row)
    artifact_format = artifact_row.get("format")
    if not isinstance(artifact_format, str) or not artifact_format:
        raise PipelineCompileError("compiler artifact format is invalid")
    return CompiledStage(
        module,
        module_manifest,
        entry,
        stage,
        target,
        reflection,
        interface,
        CompiledArtifact(artifact_format, artifact_data, artifact_name),
        metadata,
    )


def dtype_and_shape(type_name: object) -> tuple[str | None, list[int]]:
    if not isinstance(type_name, str):
        return None, []
    if not type_name.startswith("tensor<") or not type_name.endswith(">"):
        return type_name, []
    parts = type_name[7:-1].split("x")
    if not parts:
        return None, []
    try:
        shape = [
            0 if dimension == "?" else int(dimension)
            for dimension in parts[:-1]
        ]
    except ValueError:
        raise PipelineCompileError(
            f"invalid reflected tensor type {type_name!r}") from None
    return parts[-1], shape


def _backend_name(name: str) -> str:
    result = "".join(
        character if character.isascii() and
        (character.isalnum() or character == "_") else "_"
        for character in name)
    return f"_{result}" if not result or result[0].isdigit() else result


def _internal_parameter_source(row: Mapping[str, Any]) -> str | None:
    implicit = row.get("vernon.implicit")
    system_value = (implicit
                    if implicit == "resolution"
                    else row.get("vernon.system_value"))
    if system_value is not None:
        if system_value != "resolution":
            raise PipelineCompileError(
                f"unsupported compiler system value {system_value!r}")
        return "system_value"
    if row.get("kind") == "sampler" and (
            implicit == "sampler"
            or row.get("vernon.implicit_sampler") is True
            or row.get("vernon.compiler_generated") is True):
        return "implicit_sampler"
    return None


def reflected_parameters(
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]],
           dict[str, list[dict[str, Any]]]]:
    external: dict[str, list[dict[str, Any]]] = {}
    internal: dict[str, list[dict[str, Any]]] = {}
    for stage in ("compute", "vertex", "fragment"):
        record = records.get(stage)
        if record is None:
            continue
        interface = record.get("interface", {})
        if not isinstance(interface, Mapping):
            raise PipelineCompileError(
                f"{stage} stage interface must be an object")
        for row in interface.get("arguments", []):
            if not isinstance(row, Mapping):
                raise PipelineCompileError(
                    f"{stage} interface argument must be an object")
            if "vernon.builtin" in row or row.get("vernon.varying", False):
                continue
            internal_source = _internal_parameter_source(row)
            name = row.get("vernon.source_name")
            interface_name = row.get("vernon.interface")
            if internal_source is not None and (not isinstance(name, str)
                                                or not name):
                name = (f"__vernon_{internal_source}_{stage}_"
                        f"{row.get('index', 0)}")
            if (not isinstance(name, str) or not name
                    or not isinstance(interface_name, str)):
                raise PipelineCompileError(
                    f"{stage} external argument is missing source metadata")
            inferred_dtype, inferred_shape = dtype_and_shape(row.get("type"))
            use = {
                "stage": stage,
                "entry": record["entry"],
                "index": row.get("index"),
                "kind": row.get("kind", "scalar"),
                "type": row.get("type"),
                "dtype": (row.get("dtype") or row.get("vernon.dtype")
                          or inferred_dtype),
                "shape": row.get("shape", inferred_shape),
                "interface": interface_name,
                "access": row.get("access", "read"),
                "dimension": row.get("dimension"),
            }
            if internal_source is not None:
                use["internal_source"] = internal_source
                if internal_source == "system_value":
                    use["system_value"] = "resolution"
            for key in ("vernon.location", "vernon.instance_divisor",
                        "vernon.set", "vernon.binding",
                        "sampled_texture_set", "sampled_texture_binding",
                        "sampled_texture_bindings"):
                if key in row:
                    use[key] = row[key]
            backend_name = _backend_name(name)
            if interface_name == "uniform":
                if (record.get("target") in {"opengl", "opengles", "metal"}
                        and "vernon.binding" not in row):
                    use["uniform_name"] = backend_name
                else:
                    use["uniform_name"] = f"{backend_name}._m0"
            elif row.get("kind") == "texture":
                use["uniform_name"] = backend_name
            table = internal if internal_source is not None else external
            table.setdefault(name, []).append(use)
    return external, internal


def external_parameters(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return reflected_parameters(records)[0]


def internal_parameters(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return reflected_parameters(records)[1]


def classify_parameter_use(use: Mapping[str, Any]) -> str:
    kind = use.get("kind")
    if kind in {"texture", "sampler"}:
        return str(kind)
    return "tensor"


def merge_parameter_uses(name: str,
                         uses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not uses:
        raise PipelineCompileError(
            f"pipeline parameter {name!r} has no uses")
    normalized = [dict(use) for use in uses]
    first = normalized[0]
    kind = classify_parameter_use(first)
    for use in normalized[1:]:
        incompatible_layout = (
            kind != "tensor"
            and (use.get("type") != first.get("type")
                 or use.get("shape", []) != first.get("shape", [])))
        if (classify_parameter_use(use) != kind
                or use.get("dtype") != first.get("dtype")
                or incompatible_layout):
            raise PipelineCompileError(
                f"incompatible pipeline parameter {name!r}")
    representative = (next(
        (use for use in normalized if use.get("stage") != "compute"), first)
                      if kind == "tensor" else first)
    access_values = {str(use.get("access", "read")) for use in normalized}
    access = ("read_write" if "read_write" in access_values
              or access_values == {"read", "write"} else
              next(iter(access_values)))
    parameter = {
        "name": name,
        "kind": kind,
        "type": representative.get("type"),
        "dtype": representative.get("dtype"),
        "shape": representative.get("shape", []),
        "access": access,
        "dimension": first.get("dimension"),
        "uses": normalized,
    }
    return {key: value for key, value in parameter.items()
            if value is not None}


def merge_internal_parameter_uses(
        name: str, uses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parameter = merge_parameter_uses(name, uses)
    sources = {use.get("internal_source") for use in uses}
    if len(sources) != 1 or None in sources:
        raise PipelineCompileError(
            f"inconsistent internal pipeline parameter {name!r}")
    source = next(iter(sources))
    parameter["source"] = source
    if source == "system_value":
        values = {use.get("system_value") for use in uses}
        if values != {"resolution"}:
            raise PipelineCompileError(
                f"inconsistent resolution system value {name!r}")
        parameter["system_value"] = "resolution"
        if parameter.get("dtype") != "f32" or parameter.get("shape") != [2]:
            raise PipelineCompileError(
                "resolution system value must have reflected type tensor<2xf32>"
            )
    elif parameter.get("kind") != "sampler":
        raise PipelineCompileError(
            "implicit sampler metadata must annotate a sampler argument")
    return parameter


def assign_parameter_slots(
    records_by_variant: Sequence[Mapping[str, Mapping[str, Any]]],
) -> dict[str, int]:
    names = sorted({
        name
        for records in records_by_variant
        for name in external_parameters(records)
    })
    return {name: slot for slot, name in enumerate(names)}


def interface_by_location(values: Sequence[Mapping[str, Any]],
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


def validate_graphics_interfaces(vertex: Mapping[str, Any],
                                 fragment: Mapping[str, Any]) -> None:
    vertex_interface = vertex.get("interface", {})
    fragment_interface = fragment.get("interface", {})
    outputs = interface_by_location(vertex_interface.get("results", []),
                                    "output")
    inputs = interface_by_location(fragment_interface.get("arguments", []),
                                   "input")
    for location, value_type in inputs.items():
        if outputs.get(location) != value_type:
            raise PipelineCompileError(
                f"vertex/fragment interface mismatch at location {location}")


def fragment_outputs(
    records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fragment = records.get("fragment")
    if fragment is None:
        return []
    outputs: list[dict[str, Any]] = []
    interface = fragment.get("interface", {})
    for row in interface.get("results", []):
        if "vernon.location" not in row:
            continue
        location = row["vernon.location"]
        type_name = row.get("type")
        dtype, shape = dtype_and_shape(type_name)
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


def build_steps(stage_ids: Mapping[str, str]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if "compute" in stage_ids:
        steps.append({"kind": "dispatch", "stage": stage_ids["compute"]})
    if "compute" in stage_ids and "vertex" in stage_ids:
        steps.append({
            "kind": "barrier",
            "source": "compute_write",
            "destination": "vertex_read",
        })
    if "vertex" in stage_ids:
        if "fragment" not in stage_ids:
            raise PipelineCompileError(
                "draw step requires vertex and fragment stages")
        steps.append({
            "kind": "draw",
            "vertex": stage_ids["vertex"],
            "fragment": stage_ids["fragment"],
        })
    return steps


def plan_variant(key: Sequence[str],
                 records: Mapping[str, Mapping[str, Any]],
                 slots: Mapping[str, int]) -> VariantPlan:
    if "vertex" in records or "fragment" in records:
        if not {"vertex", "fragment"}.issubset(records):
            raise PipelineCompileError(
                "graphics variants require vertex and fragment stages")
        validate_graphics_interfaces(records["vertex"], records["fragment"])
    external = external_parameters(records)
    internal = internal_parameters(records)
    parameters = []
    for name in sorted(external, key=lambda value: slots[value]):
        parameter = merge_parameter_uses(name, external[name])
        parameter["slot"] = slots[name]
        parameters.append(parameter)
    internal_rows = [
        merge_internal_parameter_uses(name, internal[name])
        for name in sorted(internal)
    ]
    stage_ids = {
        stage: str(record["id"])
        for stage, record in records.items()
    }
    return VariantPlan(tuple(key), stage_ids, tuple(parameters),
                       tuple(internal_rows),
                       tuple(fragment_outputs(records)),
                       tuple(build_steps(stage_ids)))


def build_bundle_plan(pipeline_id: str, target: TargetOptions,
                      features: Sequence[str],
                      variants: Sequence[tuple[Sequence[str],
                                               Mapping[str,
                                                       CompiledStage]]]
                      ) -> BundlePlan:
    records_by_variant = [{
        name: stage.logical_record()
        for name, stage in stages.items()
    } for _, stages in variants]
    slots = assign_parameter_slots(records_by_variant)
    variant_plans = tuple(
        plan_variant(key, records, slots)
        for (key, _), records in zip(variants, records_by_variant, strict=True))
    unique_stages = {
        stage.id: stage
        for _, stages in variants for stage in stages.values()
    }
    return BundlePlan(pipeline_id, target, tuple(sorted(set(features))),
                      variant_plans,
                      tuple(unique_stages[key]
                            for key in sorted(unique_stages)))


def inline_artifact_descriptor(artifact: CompiledArtifact) -> dict[str, Any]:
    encoding = "base64" if artifact.format == "spirv" else "utf8"
    try:
        data = (base64.b64encode(artifact.data).decode("ascii")
                if encoding == "base64" else artifact.data.decode("utf-8"))
    except UnicodeDecodeError:
        raise PipelineCompileError(
            f"{artifact.format} runtime artifact is not UTF-8") from None
    return {
        "format": artifact.format,
        "storage": "inline",
        "encoding": encoding,
        "data": data,
        "size": len(artifact.data),
        "sha256": artifact.sha256,
    }


def materialize_bundle(
    plan: BundlePlan,
    artifact_descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    document = plan.logical_dict()
    records = document["stage_artifacts"]
    if set(records) != set(artifact_descriptors):
        raise PipelineCompileError(
            "artifact descriptors do not match planned stages")
    for stage_id, descriptor in artifact_descriptors.items():
        if descriptor.get("sha256") != next(
                stage.artifact.sha256 for stage in plan.stages
                if stage.id == stage_id):
            raise PipelineCompileError(
                f"artifact descriptor digest does not match stage {stage_id}")
        records[stage_id]["artifact"] = dict(descriptor)
    return with_content_hash(document)


def serialize_bundle(bundle: Mapping[str, Any]) -> bytes:
    return (canonical_json(with_content_hash(bundle)) + "\n").encode("utf-8")
