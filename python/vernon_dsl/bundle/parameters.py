from __future__ import annotations

from typing import Any, Mapping, Sequence

from .types import PipelineCompileError


def dtype_and_shape(type_name: object) -> tuple[str | None, list[int]]:
    if not isinstance(type_name, str):
        return None, []
    if not type_name.startswith("tensor<") or not type_name.endswith(">"):
        return type_name, []
    parts = type_name[7:-1].split("x")
    if not parts:
        return None, []
    try:
        shape = [0 if dimension == "?" else int(dimension) for dimension in parts[:-1]]
    except ValueError:
        raise PipelineCompileError(f"invalid reflected tensor type {type_name!r}") from None
    return parts[-1], shape


def _backend_name(name: str) -> str:
    result = "".join(
        character if character.isascii() and (character.isalnum() or character == "_") else "_" for character in name
    )
    return f"_{result}" if not result or result[0].isdigit() else result


def _uniform_layout(row: Mapping[str, Any], dtype: str | None, shape: Sequence[int]) -> dict[str, Any] | None:
    size = row.get("physical_size")
    alignment = row.get("physical_alignment")
    storage_class = row.get("proposed_storage_class")
    if not isinstance(size, int) or size <= 0 or not isinstance(alignment, int) or alignment <= 0:
        return None
    scalar_sizes = {"bool": 1, "f16": 2, "i32": 4, "u32": 4, "f32": 4, "f64": 8}
    element_size = scalar_sizes.get(dtype or "")
    if element_size is None:
        return None
    reflected_strides = row.get("array_strides")
    if isinstance(reflected_strides, list) and len(reflected_strides) == len(shape):
        byte_strides = reflected_strides
    elif len(shape) == 2 and isinstance(row.get("matrix_stride"), int):
        byte_strides = [element_size, row["matrix_stride"]]
    elif len(shape) == 1:
        byte_strides = [element_size]
    elif not shape:
        byte_strides = []
    else:
        return None
    result = {
        "storage": "uniform_buffer" if storage_class == "Uniform" else "inline",
        "size": size,
        "alignment": alignment,
        "byte_strides": byte_strides,
    }
    matrix_order = row.get("matrix_order")
    if isinstance(matrix_order, str):
        result["matrix_order"] = matrix_order
    return result


def _internal_parameter_source(row: Mapping[str, Any]) -> str | None:
    legacy_markers = ("vernon.compiler_generated", "vernon.implicit_sampler", "vernon.system_value")
    if any(marker in row for marker in legacy_markers):
        raise PipelineCompileError("legacy compiler-generated parameter metadata is unsupported")
    implicit = row.get("vernon.implicit")
    if implicit == "resolution":
        return "system_value"
    if implicit == "sampler":
        if row.get("kind") != "sampler":
            raise PipelineCompileError("implicit sampler metadata must annotate a sampler argument")
        return "implicit_sampler"
    if implicit is not None:
        raise PipelineCompileError(f"unsupported compiler-generated parameter {implicit!r}")
    return None


def reflected_parameters(
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    external: dict[str, list[dict[str, Any]]] = {}
    internal: dict[str, list[dict[str, Any]]] = {}
    for stage in ("compute", "vertex", "fragment"):
        record = records.get(stage)
        if record is None:
            continue
        interface = record.get("interface", {})
        if not isinstance(interface, Mapping):
            raise PipelineCompileError(f"{stage} stage interface must be an object")
        for row in interface.get("arguments", []):
            if not isinstance(row, Mapping):
                raise PipelineCompileError(f"{stage} interface argument must be an object")
            if "vernon.builtin" in row or row.get("vernon.varying", False):
                continue
            internal_source = _internal_parameter_source(row)
            name = row.get("vernon.source_name")
            interface_name = row.get("vernon.interface")
            if internal_source is not None and (not isinstance(name, str) or not name):
                name = f"__vernon_{internal_source}_{stage}_{row.get('index', 0)}"
            if not isinstance(name, str) or not name or not isinstance(interface_name, str):
                raise PipelineCompileError(f"{stage} external argument is missing source metadata")
            inferred_dtype, inferred_shape = dtype_and_shape(row.get("type"))
            if row.get("kind") == "tensor" and "shape" not in row:
                rank = row.get("rank")
                if isinstance(rank, int) and rank >= 0:
                    inferred_shape = [0] * rank
            use = {
                "stage": stage,
                "entry": record["entry"],
                "index": row.get("index"),
                "kind": row.get("kind", "scalar"),
                "type": row.get("type"),
                "dtype": (row.get("dtype") or row.get("vernon.dtype") or inferred_dtype),
                "shape": row.get("shape", inferred_shape),
                "interface": interface_name,
                "access": row.get("access", "read"),
                "dimension": row.get("dimension"),
            }
            if (
                interface_name == "uniform" or (stage == "compute" and row.get("kind") == "tensor_value")
            ) and "uniform_layout" not in row:
                layout = _uniform_layout(row, use["dtype"], use["shape"])
                if layout is not None:
                    use["uniform_layout"] = layout
            if internal_source is not None:
                use["internal_source"] = internal_source
                if internal_source == "system_value":
                    use["system_value"] = "resolution"
            for key in (
                "vernon.location",
                "vernon.instance_divisor",
                "vernon.set",
                "vernon.binding",
                "sampled_texture_bindings",
                "uniform_layout",
                "location_span",
                "attribute_leaves",
            ):
                if key in row:
                    use[key] = row[key]
            if stage == "compute" and record.get("target") in {"opengl", "opengles"} and "vernon.binding" not in use:
                use["vernon.set"] = 0
                use["vernon.binding"] = int(row.get("index", 0))
            backend_name = _backend_name(name)
            if interface_name == "uniform":
                if record.get("target") in {"opengl", "opengles", "metal", "directx"} and "vernon.binding" not in row:
                    use["uniform_name"] = backend_name
                else:
                    use["uniform_name"] = f"{backend_name}._m0"
            elif row.get("kind") == "texture":
                use["uniform_name"] = backend_name
            table = internal if internal_source is not None else external
            table.setdefault(name, []).append(use)
    return external, internal


def external_parameters(records: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return reflected_parameters(records)[0]


def internal_parameters(records: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return reflected_parameters(records)[1]


def classify_parameter_use(use: Mapping[str, Any]) -> str:
    kind = use.get("kind")
    if kind in {"texture", "sampler"}:
        return str(kind)
    return "tensor"


def merge_parameter_uses(name: str, uses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not uses:
        raise PipelineCompileError(f"pipeline parameter {name!r} has no uses")
    normalized = [dict(use) for use in uses]
    first = normalized[0]
    kind = classify_parameter_use(first)
    for use in normalized[1:]:
        incompatible_layout = kind != "tensor" and (
            use.get("type") != first.get("type") or use.get("shape", []) != first.get("shape", [])
        )
        if classify_parameter_use(use) != kind or use.get("dtype") != first.get("dtype") or incompatible_layout:
            raise PipelineCompileError(f"incompatible pipeline parameter {name!r}")
    representative = (
        next((use for use in normalized if use.get("stage") != "compute"), first) if kind == "tensor" else first
    )
    access_values = {str(use.get("access", "read")) for use in normalized}
    access = (
        "read_write"
        if "read_write" in access_values or access_values == {"read", "write"}
        else next(iter(access_values))
    )
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
    return {key: value for key, value in parameter.items() if value is not None}


def merge_internal_parameter_uses(name: str, uses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parameter = merge_parameter_uses(name, uses)
    sources = {use.get("internal_source") for use in uses}
    if len(sources) != 1 or None in sources:
        raise PipelineCompileError(f"inconsistent internal pipeline parameter {name!r}")
    source = next(iter(sources))
    parameter["source"] = source
    if source == "system_value":
        values = {use.get("system_value") for use in uses}
        if values != {"resolution"}:
            raise PipelineCompileError(f"inconsistent resolution system value {name!r}")
        parameter["system_value"] = "resolution"
        if parameter.get("dtype") != "f32" or parameter.get("shape") != [2]:
            raise PipelineCompileError("resolution system value must have reflected type tensor<2xf32>")
        if any(use.get("sampled_texture_bindings") for use in uses):
            raise PipelineCompileError("resolution system value cannot pair sampled textures")
    else:
        if parameter.get("kind") != "sampler":
            raise PipelineCompileError("implicit sampler metadata must annotate a sampler argument")
        for use in uses:
            bindings = use.get("sampled_texture_bindings")
            if (
                not isinstance(bindings, list)
                or len(bindings) != 1
                or not isinstance(bindings[0], Mapping)
                or not isinstance(bindings[0].get("set"), int)
                or not isinstance(bindings[0].get("binding"), int)
            ):
                raise PipelineCompileError("implicit sampler must have exactly one sampled texture binding")
    return parameter


def assign_parameter_slots(records_by_variant: Sequence[Mapping[str, Mapping[str, Any]]]) -> dict[str, int]:
    names = sorted({name for records in records_by_variant for name in external_parameters(records)})
    return {name: slot for slot, name in enumerate(names)}


def interface_by_location(values: Sequence[Mapping[str, Any]], interface: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        if value.get("vernon.interface") != interface or "vernon.builtin" in value:
            continue
        location = value.get("vernon.location")
        value_type = value.get("type")
        if isinstance(location, int) and isinstance(value_type, str):
            result[location] = value_type
    return result


def validate_graphics_interfaces(vertex: Mapping[str, Any], fragment: Mapping[str, Any]) -> None:
    vertex_interface = vertex.get("interface", {})
    fragment_interface = fragment.get("interface", {})
    outputs = interface_by_location(vertex_interface.get("results", []), "output")
    inputs = interface_by_location(fragment_interface.get("arguments", []), "input")
    for location, value_type in inputs.items():
        if outputs.get(location) != value_type:
            raise PipelineCompileError(f"vertex/fragment interface mismatch at location {location}")


def fragment_outputs(records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
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
        outputs.append(
            {
                "name": row.get("vernon.source_name") or f"output_{location}",
                "kind": "texture",
                "dtype": dtype,
                "shape": shape,
                "access": "write",
                "location": location,
                "type": type_name,
            }
        )
    return outputs


__all__ = [
    "assign_parameter_slots",
    "classify_parameter_use",
    "dtype_and_shape",
    "external_parameters",
    "fragment_outputs",
    "interface_by_location",
    "internal_parameters",
    "merge_internal_parameter_uses",
    "merge_parameter_uses",
    "reflected_parameters",
    "validate_graphics_interfaces",
]
