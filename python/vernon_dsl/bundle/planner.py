from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from ..language.stage_registry import (
    GRAPHICS_STAGES,
    STAGE_BY_KIND,
    validate_graphics_topology,
    validate_stage_target,
)
from .parameters import (
    assign_parameter_slots,
    external_parameters,
    fragment_outputs,
    internal_parameters,
    merge_internal_parameter_uses,
    merge_parameter_uses,
    reflected_parameters,
    validate_graphics_interfaces,
)
from .reflection import (
    compiled_stage_from_program,
    parse_reflection_json,
    select_artifact,
    select_artifact_bytes,
    select_entry,
)
from .serialize import (
    canonical_json,
    content_hash,
    inline_artifact_descriptor,
    materialize_bundle,
    serialize_bundle,
    with_content_hash,
)
from .types import BundlePlan, CompiledArtifact, CompiledStage, PipelineCompileError, TargetOptions, VariantPlan


def _parameter_dtype(parameter: Mapping[str, Any]) -> str:
    dtype = parameter.get("dtype")
    if isinstance(dtype, str) and dtype:
        return dtype
    for layout_name in ("value_layout", "element_layout"):
        layout = parameter.get(layout_name)
        if not isinstance(layout, Mapping):
            continue
        leaves = layout.get("leaves")
        if isinstance(leaves, list) and leaves and isinstance(leaves[0], Mapping):
            dtype = leaves[0].get("dtype")
            if isinstance(dtype, str) and dtype:
                return dtype
    return "opaque"


def _single_compute_execution(
    records: Mapping[str, Mapping[str, Any]],
    parameters: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    record = records["compute"]
    values: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    results: list[int] = []
    for value_id, parameter in enumerate(parameters):
        access = parameter.get("access")
        if access not in {"read", "write", "read_write"}:
            raise PipelineCompileError("compute pipeline parameter has no canonical resource access")
        output = access in {"write", "read_write"}
        values.append(
            {
                "id": value_id,
                "name": parameter["name"],
                "type": parameter.get("type", parameter.get("kind", "opaque")),
                "dtype": _parameter_dtype(parameter),
                "shape": list(parameter.get("shape", ())),
                **(
                    {"value_layout": copy.deepcopy(parameter["value_layout"])}
                    if "value_layout" in parameter
                    else {"value_layout": copy.deepcopy(parameter["element_layout"])}
                    if "element_layout" in parameter
                    else {}
                ),
                "external": True,
                "output": output,
            }
        )
        bindings.append({"parameter": parameter["name"], "value": value_id})
        resources.append({"value": value_id, "access": access})
        if output:
            results.append(value_id)
    node = {
        "id": 0,
        "name": str(record["entry"]),
        "kind": "compute",
        "stage": "compute",
        "operands": list(range(len(values))),
        "results": [],
        "dependencies": [],
        "bindings": bindings,
        "resources": resources,
        # Standalone dispatch geometry remains invocation-time state. The
        # validated single-node fast path ignores this topology placeholder.
        "grid": [1, 1, 1],
    }
    return {
        "values": values,
        "graphs": [
            {
                "name": "forward",
                "direction": "forward",
                "arguments": list(range(len(values))),
                "results": results,
                "nodes": [node],
            }
        ],
    }


def plan_variant(
    key: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    slots: Mapping[str, int],
) -> VariantPlan:
    graphics = tuple(
        sorted(
            (stage for stage in records if stage in GRAPHICS_STAGES),
            key=lambda stage: STAGE_BY_KIND[stage].graphics_order or 0,
        )
    )
    if ("compute" in records) == bool(graphics):
        raise PipelineCompileError("pipeline variant must contain either one compute program or one graphics program")
    if graphics:
        try:
            validate_graphics_topology(graphics)
        except ValueError as error:
            raise PipelineCompileError(str(error)) from None
        for producer, consumer in zip(graphics, graphics[1:], strict=False):
            validate_graphics_interfaces(producer, records[producer], consumer, records[consumer])
    external = external_parameters(records)
    internal = internal_parameters(records)
    parameters = []
    for name in sorted(external, key=lambda value: slots[value]):
        parameter = merge_parameter_uses(name, external[name])
        parameter["slot"] = slots[name]
        parameters.append(parameter)
    internal_rows = [merge_internal_parameter_uses(name, internal[name]) for name in sorted(internal)]
    stage_ids = {stage: str(record["id"]) for stage, record in records.items()}
    execution = _single_compute_execution(records, parameters) if "compute" in records else None
    return VariantPlan(
        tuple(key),
        stage_ids,
        tuple(parameters),
        tuple(internal_rows),
        tuple(fragment_outputs(records)),
        execution,
    )


def build_bundle_plan(
    pipeline_id: str,
    target: TargetOptions,
    features: Sequence[str],
    variants: Sequence[tuple[Sequence[str], Mapping[str, CompiledStage]]],
    transform: Mapping[str, Any] | None = None,
    autodiff_profiles: Mapping[str, Any] | None = None,
) -> BundlePlan:
    records_by_variant = [
        {
            name: {
                "id": stage.id,
                "entry": stage.entry,
                "target": stage.target.target,
                "interface": dict(stage.interface),
            }
            for name, stage in stages.items()
        }
        for _, stages in variants
    ]
    slots = assign_parameter_slots(records_by_variant)
    variant_plans = tuple(
        plan_variant(key, records, slots) for (key, _), records in zip(variants, records_by_variant, strict=True)
    )
    unique_stages = {stage.id: stage for _, stages in variants for stage in stages.values()}
    for stage in unique_stages.values():
        try:
            validate_stage_target(stage.stage, target.target)
        except ValueError as error:
            raise PipelineCompileError(str(error)) from None
    return BundlePlan(
        pipeline_id,
        target,
        tuple(sorted(set(features))),
        variant_plans,
        tuple(unique_stages[key] for key in sorted(unique_stages)),
        transform,
        autodiff_profiles,
    )


def build_program_bundle_plan(
    pipeline_id: str,
    target: TargetOptions,
    features: Sequence[str],
    variants: Sequence[tuple[Sequence[str], Mapping[str, Any], Mapping[str, CompiledStage]]],
    *,
    canonical_execution: bool = False,
    canonical_program: Mapping[str, Any] | None = None,
) -> BundlePlan:
    """Compose executable Program graphs with their selected kernel stages.

    ``canonical_program`` installs compiler-finalized execution directly.
    Canonical execution is never synthesized in Python.
    """
    if canonical_execution and canonical_program is None:
        raise PipelineCompileError("canonical execution requires C++ finalized Program output")
    planned_variants: list[VariantPlan] = []
    unique_stages: dict[str, CompiledStage] = {}
    for key, execution, stages in variants:
        graph_values = execution.get("values")
        graphs = execution.get("graphs")
        if not isinstance(graph_values, list) or not isinstance(graphs, list) or not graphs:
            raise PipelineCompileError("Program compiler reflection has no executable graph")
        values_by_id = {
            value["id"]: value
            for value in graph_values
            if isinstance(value, Mapping) and isinstance(value.get("id"), int)
        }
        if len(values_by_id) != len(graph_values):
            raise PipelineCompileError("Program compiler reflection contains invalid values")
        boundary_ids: set[int] = set()
        referenced: dict[str, str] = {}
        stage_nodes: dict[str, Mapping[str, Any]] = {}
        for graph in graphs:
            if not isinstance(graph, Mapping) or not isinstance(graph.get("nodes"), list):
                raise PipelineCompileError("Program compiler reflection contains an invalid graph")
            for field in ("arguments", "results"):
                ids = graph.get(field)
                if not isinstance(ids, list) or any(not isinstance(value_id, int) for value_id in ids):
                    raise PipelineCompileError("Program compiler reflection contains invalid graph boundaries")
                boundary_ids.update(ids)
            for node in graph["nodes"]:
                if not isinstance(node, Mapping):
                    raise PipelineCompileError("Program compiler reflection contains an invalid node")
                stage = node.get("stage")
                kind = node.get("kind")
                if not isinstance(stage, str) or not stage or kind not in {"compute", "render"}:
                    raise PipelineCompileError("Program compiler reflection contains an invalid node implementation")
                implementation_kind = "compute" if kind == "compute" else "graphics"
                previous = referenced.setdefault(stage, implementation_kind)
                if previous != implementation_kind:
                    raise PipelineCompileError(f"Program stage {stage!r} is used by incompatible node kinds")
                stage_nodes[stage] = node
        if set(stages) != set(referenced):
            missing = sorted(set(referenced) - set(stages))
            unused = sorted(set(stages) - set(referenced))
            detail = []
            if missing:
                detail.append("missing " + ", ".join(repr(value) for value in missing))
            if unused:
                detail.append("unused " + ", ".join(repr(value) for value in unused))
            raise PipelineCompileError("Program implementation stages do not match its graph: " + "; ".join(detail))
        program: dict[str, str] = {}
        records: dict[str, dict[str, Any]] = {}
        for name, stage in stages.items():
            expected = referenced[name]
            if stage.stage != expected:
                raise PipelineCompileError(
                    f"Program stage {name!r} requires {expected}, compiler produced {stage.stage}"
                )
            try:
                validate_stage_target(stage.stage, target.target)
            except ValueError as error:
                raise PipelineCompileError(str(error)) from None
            if stage.target.spec != target.spec:
                raise PipelineCompileError(f"Program stage {name!r} target does not match the bundle target")
            program[name] = stage.id
            unique_stages[stage.id] = stage
            interface = copy.deepcopy(dict(stage.interface))
            node = stage_nodes[name]
            raw_bindings = node.get("bindings")
            if not isinstance(raw_bindings, list):
                raise PipelineCompileError(f"Program stage {name!r} has no value bindings")
            bindings = {
                binding["parameter"]: binding["value"]
                for binding in raw_bindings
                if isinstance(binding, Mapping)
                and isinstance(binding.get("parameter"), str)
                and isinstance(binding.get("value"), int)
            }
            for row in interface.get("arguments", []):
                if not isinstance(row, dict):
                    continue
                parameter = row.get("vernon.source_name")
                value_id = bindings.get(parameter)
                if value_id is None and row.get("vernon.autodiff_role") == "cotangent":
                    value_id = bindings.get(f"cotangent.{parameter}")
                value = values_by_id.get(value_id)
                if value is not None:
                    row["vernon.source_name"] = value["name"]
            records[name] = {
                "id": stage.id,
                "entry": stage.entry,
                "stage": stage.stage,
                "target": stage.target.target,
                "interface": interface,
            }
        reflected_external, reflected_internal = reflected_parameters(records)
        value_ids_by_name = {value["name"]: value_id for value_id, value in values_by_id.items()}

        def attach_value_layout(
            name: str,
            parameter: dict[str, Any],
            *,
            value_ids_by_name: Mapping[str, int] = value_ids_by_name,
            values_by_id: Mapping[int, Mapping[str, Any]] = values_by_id,
        ) -> None:
            value_id = value_ids_by_name.get(name)
            if value_id is None:
                return
            value = values_by_id.get(value_id)
            if value is None:
                return
            value_type = value.get("type")
            if isinstance(value_type, str) and (
                value_type == "!vernon.ad_tape"
                or value_type.startswith("!vernon.ad_tape<")
                or value_type.startswith("!vernon.texture<")
                or value_type.startswith("!vernon.sampler")
            ):
                return
            layout = parameter.get("value_layout", parameter.get("element_layout"))
            if layout is None:
                raise PipelineCompileError(f"Program value {name!r} has no canonical layout")
            canonical = value.get("value_layout")
            if not isinstance(canonical, Mapping):
                raise PipelineCompileError(f"Program value {name!r} has no compiler-authoritative canonical layout")
            parameter.pop("element_layout", None)
            parameter["value_layout"] = copy.deepcopy(canonical)
            parameter["type"] = value["type"]
            parameter["shape"] = copy.deepcopy(value["shape"])
            parameter.pop("address_space", None)

        def preserve_released_tensor_layout(name: str, parameter: dict[str, Any]) -> None:
            if canonical_execution or parameter.get("kind") != "tensor":
                return
            if "value_layout" in parameter:
                parameter["element_layout"] = parameter.pop("value_layout")

        external_rows: list[dict[str, Any]] = []
        internal_rows: list[dict[str, Any]] = []
        for slot, name in enumerate(
            sorted(
                (name for name in reflected_external if value_ids_by_name.get(name) in boundary_ids),
                key=lambda candidate: value_ids_by_name[candidate],
            )
        ):
            parameter = merge_parameter_uses(name, reflected_external[name])
            parameter["slot"] = slot
            preserve_released_tensor_layout(name, parameter)
            if canonical_execution:
                attach_value_layout(name, parameter)
            external_rows.append(parameter)
        for name in sorted(
            (name for name in reflected_external if value_ids_by_name.get(name) not in boundary_ids),
            key=lambda candidate: value_ids_by_name[candidate],
        ):
            parameter = merge_parameter_uses(name, reflected_external[name])
            parameter["source"] = "program_value"
            preserve_released_tensor_layout(name, parameter)
            if canonical_execution:
                attach_value_layout(name, parameter)
            internal_rows.append(parameter)
        for name in sorted(reflected_internal):
            parameter = merge_internal_parameter_uses(name, reflected_internal[name])
            preserve_released_tensor_layout(name, parameter)
            if canonical_execution:
                attach_value_layout(name, parameter)
            internal_rows.append(parameter)
        planned_variants.append(
            VariantPlan(
                tuple(key),
                program,
                tuple(external_rows),
                tuple(internal_rows),
                (),
                copy.deepcopy(dict(canonical_program))
                if canonical_execution and canonical_program is not None
                else copy.deepcopy(dict(execution)),
            )
        )
    return BundlePlan(
        pipeline_id,
        target,
        tuple(sorted(set(features))),
        tuple(planned_variants),
        tuple(unique_stages[key] for key in sorted(unique_stages)),
    )


__all__ = [
    "BundlePlan",
    "CompiledArtifact",
    "CompiledStage",
    "PipelineCompileError",
    "TargetOptions",
    "VariantPlan",
    "build_bundle_plan",
    "build_program_bundle_plan",
    "canonical_json",
    "compiled_stage_from_program",
    "content_hash",
    "inline_artifact_descriptor",
    "materialize_bundle",
    "parse_reflection_json",
    "plan_variant",
    "select_artifact",
    "select_artifact_bytes",
    "select_entry",
    "serialize_bundle",
    "with_content_hash",
]
