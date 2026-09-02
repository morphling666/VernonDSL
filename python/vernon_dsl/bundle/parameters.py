from __future__ import annotations

from typing import Any, Mapping, Sequence

from .._mlir import ranked_tensor_parts
from .types import PipelineCompileError


def dtype_and_shape(type_name: object) -> tuple[str | None, list[int]]:
    if not isinstance(type_name, str):
        return None, []
    tensor = ranked_tensor_parts(type_name)
    if tensor is None:
        return type_name, []
    shape, dtype = tensor
    try:
        extents = [0 if dimension == "?" else int(dimension) for dimension in shape]
    except ValueError:
        raise PipelineCompileError(f"invalid reflected tensor type {type_name!r}") from None
    return dtype, extents


def _backend_name(name: str) -> str:
    result = "".join(
        character if character.isascii() and (character.isalnum() or character == "_") else "_" for character in name
    )
    return f"_{result}" if not result or result[0].isdigit() else result


def _physical_value_profile(target: object, transport: object) -> tuple[str, str]:
    if target == "cpu":
        return "host_value", "host_value"
    if target == "cuda":
        return "cuda_kernel_parameter", "kernel_parameter"
    if target == "metal":
        return "metal_constant_buffer", transport if isinstance(transport, str) else "constant_buffer"
    if not isinstance(transport, str):
        raise PipelineCompileError("packed value argument is missing its reflected transport")
    if transport == "storage_buffer":
        return "vulkan_std430_storage_buffer", transport
    if target == "vulkan":
        if transport == "uniform_buffer":
            return "vulkan_std140_uniform_buffer", transport
        if transport == "push_constant":
            return "vulkan_push_constant", transport
    if target in {"opengl", "opengles"}:
        if transport == "uniform_buffer":
            return "vulkan_std140_uniform_buffer", transport
        if transport == "push_constant":
            return "opengl_native_uniform", "native_uniform"
    if target == "directx" and transport in {"uniform_buffer", "push_constant"}:
        return "directx_constant_buffer", transport
    raise PipelineCompileError(f"unsupported physical value transport {transport!r} for target {target!r}")


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
    for stage, record in records.items():
        stage_kind = record.get("stage", stage)
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
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(interface_name, str)
                or row.get("kind") not in {"scalar", "tensor_value", "tensor", "image", "sampler"}
            ):
                raise PipelineCompileError(f"{stage} external argument is missing source or kind metadata")
            inferred_dtype, inferred_shape = dtype_and_shape(row.get("type"))
            reflected_shape = row.get("shape")
            if row.get("kind") == "tensor" and isinstance(row.get("source_shape"), list):
                reflected_shape = [
                    0 if isinstance(extent, int) and extent < 0 else extent for extent in row["source_shape"]
                ]
            if row.get("kind") == "tensor" and reflected_shape is None:
                rank = row.get("rank")
                if isinstance(rank, int) and rank >= 0:
                    inferred_shape = [0] * rank
            use = {
                "stage": stage,
                "entry": record["entry"],
                "index": row.get("index"),
                "kind": row["kind"],
                "type": row.get("type"),
                "shape": reflected_shape if reflected_shape is not None else inferred_shape,
                "interface": interface_name,
                "access": row.get("access", "read"),
                "address_space": row.get("address_space"),
                "dimension": row.get("dimension"),
                "binding_role": row.get("binding_role"),
                "autodiff_role": row.get("vernon.autodiff_role"),
                "autodiff_source": row.get("vernon.autodiff_source"),
                "autodiff_carrier": row.get("vernon.autodiff_carrier"),
                "sample_result_class": row.get("sample_result_class"),
                "exact_storage_format": row.get("exact_storage_format"),
            }
            if "tensor_view_descriptor" in row:
                use["tensor_view_descriptor"] = row["tensor_view_descriptor"]
            physical_layouts = row.get("physical_layouts")
            compute_plan = None
            if isinstance(physical_layouts, Mapping):
                compute_plan = next(
                    (
                        plan
                        for plan in physical_layouts.values()
                        if isinstance(plan, Mapping) and plan.get("kind") in {"cpu_call", "kernel_parameter"}
                    ),
                    None,
                )
            compute_static_tensor_value = (
                stage_kind == "compute"
                and row.get("kind") == "tensor"
                and isinstance(compute_plan, Mapping)
                and compute_plan.get("kind") in {"cpu_call", "kernel_parameter"}
            )
            packed_value = interface_name == "uniform" or (
                stage_kind == "compute"
                and (row["kind"] not in {"tensor", "image", "sampler"} or compute_static_tensor_value)
            )
            if packed_value and stage_kind == "compute":
                use["interface"] = "value"
            elif stage_kind == "compute" and row.get("kind") == "tensor":
                use["interface"] = "storage"
            dtype = row.get("dtype") or row.get("vernon.dtype") or inferred_dtype
            if dtype is not None:
                use["dtype"] = dtype
            element_layout = row.get("element_layout")
            value_layout = row.get("value_layout")
            if isinstance(element_layout, Mapping):
                use["element_layout"] = dict(element_layout)
            if isinstance(value_layout, Mapping):
                use["value_layout"] = dict(value_layout)
            elif isinstance(element_layout, Mapping):
                use["value_layout"] = dict(element_layout)
            elif row.get("kind") not in {"image", "sampler"}:
                raise PipelineCompileError(f"{stage} value argument is missing canonical value_layout")
            if packed_value:
                profile, transport = _physical_value_profile(record.get("target"), row.get("value_transport"))
                selected_layout = physical_layouts.get(profile) if isinstance(physical_layouts, Mapping) else None
                if not isinstance(selected_layout, Mapping):
                    raise PipelineCompileError(f"{stage} packed value argument is missing profile {profile!r}")
                use["interface_plan"] = dict(selected_layout)
                use["transport"] = transport
            if internal_source is not None:
                use["internal_source"] = internal_source
                if internal_source == "system_value":
                    use["system_value"] = "resolution"
            for key in (
                "vernon.location",
                "vernon.instance_divisor",
                "vernon.set",
                "vernon.binding",
                "sampled_image_bindings",
                "location_span",
                "attribute_leaves",
            ):
                if key in row:
                    use[key] = row[key]
            descriptor_required = (
                use["interface"] == "storage"
                or (interface_name == "resource" and row.get("kind") in {"tensor", "image"})
                or (interface_name == "uniform" and use.get("transport") in {"uniform_buffer", "storage_buffer"})
            )
            if descriptor_required and ("vernon.set" not in use or "vernon.binding" not in use):
                raise PipelineCompileError(f"{stage} descriptor-backed argument is missing reflected set/binding")
            if row.get("kind") == "sampler" and not use.get("sampled_image_bindings"):
                raise PipelineCompileError(f"{stage} sampler argument has no reflected sampled image binding")
            backend_name = _backend_name(name)
            if interface_name == "uniform":
                if record.get("target") in {"opengl", "opengles", "metal", "directx"} and "vernon.binding" not in row:
                    use["uniform_name"] = backend_name
                else:
                    use["uniform_name"] = f"{backend_name}._m0"
            elif row.get("kind") == "image":
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
    if kind in {"image", "sampler"}:
        return str(kind)
    return "tensor"


def merge_parameter_uses(name: str, uses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not uses:
        raise PipelineCompileError(f"pipeline parameter {name!r} has no uses")
    normalized = [dict(use) for use in uses]
    first = normalized[0]
    kind = classify_parameter_use(first)
    if any(not isinstance(use.get("type"), str) or not use["type"] for use in normalized):
        raise PipelineCompileError(f"pipeline parameter {name!r} is missing its logical type")
    for use in normalized[1:]:
        incompatible_layout = kind != "tensor" and (
            use.get("type") != first.get("type") or use.get("shape", []) != first.get("shape", [])
        )
        incompatible_tensor_layout = kind == "tensor" and (
            use.get("element_layout") != first.get("element_layout")
            or use.get("value_layout") != first.get("value_layout")
        )
        if (
            classify_parameter_use(use) != kind
            or (kind != "tensor" and use.get("dtype") != first.get("dtype"))
            or incompatible_layout
            or incompatible_tensor_layout
        ):
            raise PipelineCompileError(f"incompatible pipeline parameter {name!r}")
    representative = (
        next((use for use in normalized if use.get("stage") != "compute"), first) if kind == "tensor" else first
    )
    element_layout = representative.get("element_layout")
    value_layout = representative.get("value_layout")
    access_values = {str(use.get("access", "read")) for use in normalized}
    tensor_view = kind == "tensor" and any(
        str(use.get("type", "")).startswith("!vernon.tensor_view<") for use in normalized
    )
    address_spaces = {use.get("address_space") for use in normalized}
    if tensor_view and address_spaces != {"device"}:
        raise PipelineCompileError(f"pipeline TensorView parameter {name!r} must use device address space")
    access = (
        "read_write"
        if "read_write" in access_values or access_values == {"read", "write"}
        else next(iter(access_values))
    )
    parameter = {
        "name": name,
        "kind": kind,
        "type": representative.get("type"),
        "shape": representative.get("shape", []),
        "access": access,
        "address_space": "device" if tensor_view else None,
        "dimension": first.get("dimension"),
        "binding_role": first.get("binding_role"),
        "sample_result_class": first.get("sample_result_class"),
        "exact_storage_format": first.get("exact_storage_format"),
        "uses": normalized,
    }
    if kind == "tensor":
        if element_layout is not None:
            parameter["element_layout"] = element_layout
        if value_layout is not None:
            parameter["value_layout"] = value_layout
    else:
        parameter["dtype"] = representative.get("dtype")
    for use in normalized:
        for key in (
            "entry",
            "kind",
            "type",
            "access",
            "address_space",
            "dimension",
            "binding_role",
            "sample_result_class",
            "exact_storage_format",
            "internal_source",
            "system_value",
            "location_span",
            "element_layout",
            "autodiff_role",
            "autodiff_source",
            "autodiff_carrier",
        ):
            use.pop(key, None)
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
        leaves = parameter.get("value_layout", {}).get("leaves", [])
        if (
            parameter.get("shape") != [2]
            or len(leaves) != 1
            or leaves[0].get("dtype") != "f32"
            or leaves[0].get("scalar_count") != 2
            or leaves[0].get("shape") != [2]
        ):
            raise PipelineCompileError("resolution system value must have reflected type tensor<2xf32>")
        if any(use.get("sampled_image_bindings") for use in uses):
            raise PipelineCompileError("resolution system value cannot pair sampled images")
    else:
        if parameter.get("kind") != "sampler":
            raise PipelineCompileError("implicit sampler metadata must annotate a sampler argument")
        for use in uses:
            bindings = use.get("sampled_image_bindings")
            if (
                not isinstance(bindings, list)
                or not bindings
                or any(
                    not isinstance(binding, Mapping)
                    or not isinstance(binding.get("set"), int)
                    or isinstance(binding.get("set"), bool)
                    or binding["set"] < 0
                    or not isinstance(binding.get("binding"), int)
                    or isinstance(binding.get("binding"), bool)
                    or binding["binding"] < 0
                    for binding in bindings
                )
            ):
                raise PipelineCompileError("implicit sampler requires reflected sampled image bindings")
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


def validate_graphics_interfaces(
    producer_stage: str,
    producer: Mapping[str, Any],
    consumer_stage: str,
    consumer: Mapping[str, Any],
) -> None:
    producer_interface = producer.get("interface", {})
    consumer_interface = consumer.get("interface", {})
    outputs = interface_by_location(producer_interface.get("results", []), "output")
    inputs = interface_by_location(consumer_interface.get("arguments", []), "input")
    for location, value_type in inputs.items():
        if outputs.get(location) != value_type:
            raise PipelineCompileError(f"{producer_stage}/{consumer_stage} interface mismatch at location {location}")


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
                "kind": "image",
                "dtype": dtype,
                "shape": shape,
                "access": "write",
                "location": location,
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
