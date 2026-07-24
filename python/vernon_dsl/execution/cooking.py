from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .._runtime.resources import TensorStorage, TensorView
from .._shader_assets.artifact_io import write_external_artifact
from .._shader_assets.cooking import _cpu_stage_metadata, _native_module, _native_target
from ..bundle.parameters import external_parameters
from ..bundle.planner import compiled_stage_from_program, plan_variant
from ..bundle.serialize import canonical_json, materialize_bundle
from ..bundle.types import BundlePlan, CompiledStage, PipelineCompileError, TargetOptions, VariantPlan
from .graph import ComputeDispatch, ExecutionGraph, GraphicsDraw, ResourceBarrier
from .lower import infer_dispatch_grid
from .validate import validate_graph


def _compile_stage(
    node: ComputeDispatch,
    target: TargetOptions,
    native_target: Any,
    compiler: Any,
) -> tuple[CompiledStage, Any, set[str], set[str], Mapping[str, Any]]:
    frontend, function, builtins, _ = node.kernel._lower(node.arguments)
    program = compiler.compile_program_result(frontend.mlir, native_target, **target.native_options)
    if not program.ok:
        raise PipelineCompileError(program.diagnostics)
    stage = compiled_stage_from_program(
        program,
        module=f"execution_graph/node_{node.handle.index}",
        module_manifest=canonical_json(frontend.semantic_inputs),
        entry=node.kernel._entry,
        target=target,
    )
    if target.target == "cpu":
        stage = CompiledStage(
            stage.module,
            stage.module_manifest,
            stage.entry,
            stage.stage,
            stage.target,
            stage.reflection,
            stage.interface,
            stage.artifact,
            _cpu_stage_metadata(stage),
        )
    names = [value.arg for value in function.args.args if value.arg not in builtins]
    return (
        stage,
        function,
        builtins,
        node.kernel._writable_parameters(function),
        dict(zip(names, node.arguments, strict=True)),
    )


def cook_execution_graph(
    graph: ExecutionGraph,
    *,
    output: str | Path,
    target: str,
    target_options: Mapping[str, Any] | None = None,
) -> Path:
    """Cook a bound compute graph into a target-specific schema-2 asset."""

    validate_graph(graph)
    if any(isinstance(node, GraphicsDraw) for node in graph.nodes):
        raise PipelineCompileError(
            "execution graph asset cooking currently supports compute dispatch and barrier nodes"
        )
    native = _native_module()
    options = TargetOptions(target, target_options or {})
    native_target = _native_target(native, target)
    compiler = native.Compiler()
    stages: dict[str, CompiledStage] = {}
    parameters: list[dict[str, Any]] = []
    internal_parameters: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    slot = 0
    previous_dispatch = False

    for node in graph.nodes:
        if isinstance(node, ResourceBarrier):
            steps.append(
                {
                    "kind": "barrier",
                    "source": node.source,
                    "destination": node.destination,
                }
            )
            previous_dispatch = False
            continue
        assert isinstance(node, ComputeDispatch)
        stage, function, builtins, writable, values = _compile_stage(node, options, native_target, compiler)
        stages[stage.id] = stage
        record = stage.logical_record()
        names = external_parameters({"compute": record})
        local_plan = plan_variant((), {"compute": record}, {name: index for index, name in enumerate(sorted(names))})
        for row_value in local_plan.parameters:
            row = dict(row_value)
            source_name = str(row["name"])
            row["name"] = f"node_{node.handle.index}_{source_name}"
            row["slot"] = slot
            row["uses"] = [{**use, "stage": stage.id} for use in row["uses"]]
            value = values[source_name]
            if isinstance(value, (TensorStorage, TensorView)):
                row["shape"] = [0] * len(value.shape)
            parameters.append(row)
            slot += 1
        for value in local_plan.internal_parameters:
            row = dict(value)
            name = str(row["name"])
            row["uses"] = [{**use, "stage": stage.id} for use in row["uses"]]
            existing = internal_parameters.get(name)
            if existing is None:
                internal_parameters[name] = row
            else:
                existing["uses"] = [*existing["uses"], *row["uses"]]
        if graph.automatic_transitions and previous_dispatch:
            steps.append(
                {
                    "kind": "barrier",
                    "source": "compute_write",
                    "destination": "compute_read_write",
                }
            )
        grid = infer_dispatch_grid(node, function, builtins, writable)
        steps.append({"kind": "dispatch", "stage": stage.id, "grid": list(grid)})
        previous_dispatch = True

    identity = canonical_json(graph.semantic_inputs)
    plan = BundlePlan(
        f"execution_graph/{hashlib.sha256(identity.encode()).hexdigest()}",
        options,
        (),
        (
            VariantPlan(
                (),
                {},
                tuple(parameters),
                tuple(internal_parameters[name] for name in sorted(internal_parameters)),
                (),
                tuple(steps),
            ),
        ),
        tuple(stages[name] for name in sorted(stages)),
    )
    output_path = Path(output).resolve()
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
    manifest = output_path / f"{output_path.name}.execution.json"
    manifest.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


__all__ = ["cook_execution_graph"]
