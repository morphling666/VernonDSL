from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from .._runtime.operators import ImplementationUnavailable
from .._versions import PIPELINE_VERSION
from ..ad import ProgramTransformSpec
from ..bundle import (
    BundlePlan,
    CompiledStage,
    OpenGLTargetOptions,
    PipelineCompileError,
    TargetOptions,
    build_bundle_plan,
    build_program_bundle_plan,
    canonical_json,
    compiled_stage_from_program,
    make_target_options,
    materialize_bundle,
    parse_reflection_json,
)
from ..bundle.requirements import runtime_requirements
from ..compiler import Compiler, FrontendCompileRequest, compile_file
from ..frontend.structured_vjp import (
    build_structured_vjp,
    is_structured_vjp_abi_eligible,
)
from ..language.stage_registry import validate_stage_target
from ..module_graph import load_project
from ..program_frontend import (
    BuiltinDslProvider,
    CapturedDslProvider,
    CapturedVjpDslProvider,
    ParsedProgram,
    ProgramImplementation,
    ProviderChain,
)
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
    retained_programs: list[tuple[CompiledStage, Any]] | None = None,
) -> CompiledStage:
    mlir = mlir or compile_file(module.source, features=variant, entry=reference.entry)
    result = compiler.compile_program_result(mlir, native_target, **target.native_options)
    compiled = compiled_stage_from_program(
        result,
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
    if retained_programs is not None:
        retained_programs.append((compiled, result))
    return compiled


def _specialize_program_implementations(
    implementations: Sequence[ProgramImplementation],
    native: Any,
) -> tuple[ProgramImplementation, ...]:
    specialized: list[ProgramImplementation] = []
    for implementation in implementations:
        if not implementation.host_constants:
            specialized.append(implementation)
            continue
        names = [name for name, _ in implementation.host_constants]
        values = [value for _, value in implementation.host_constants]
        try:
            mlir = native._specialize_kernel_constants(implementation.mlir, implementation.entry, names, values)
        except ValueError as error:
            raise PipelineCompileError(
                f"cannot specialize Program callee {implementation.callee!r} host constants: {error}"
            ) from None
        specialized.append(
            ProgramImplementation(
                implementation.callee,
                implementation.entry,
                implementation.kind,
                mlir,
                implementation.host_constants,
            )
        )
    return tuple(specialized)


def _compile_program_bundle_plan(
    parsed: ParsedProgram,
    *,
    pipeline_id: str,
    variant: tuple[str, ...],
    target: TargetOptions,
    compiler: Any,
    native: Any,
    native_target: Any,
    retained_programs: list[tuple[CompiledStage, Any]] | None = None,
    canonical_execution: bool = False,
) -> BundlePlan:
    planned = compiler.plan_program_result(parsed.mlir)
    if not bool(planned.ok):
        raise PipelineCompileError(str(planned.diagnostics) or "Program planning failed")
    reflection = parse_reflection_json(planned.reflection)
    execution = reflection.get("execution")
    if not isinstance(execution, Mapping):
        raise PipelineCompileError("Program compiler reflection has no execution graph")
    requests = reflection.get("kernel_compile_requests")
    if not isinstance(requests, list):
        raise PipelineCompileError("Program compiler reflection has no kernel compile requests")

    stages: dict[str, CompiledStage] = {}
    compiled_implementations: dict[tuple[str, str], CompiledStage] = {}
    source = Path(parsed.provenance[0]) if parsed.provenance else Path()
    reflected_values = execution.get("values")
    if not isinstance(reflected_values, list):
        raise PipelineCompileError("Program compiler reflection has no reflected values")
    values = {
        value["id"]: value
        for value in reflected_values
        if isinstance(value, Mapping) and isinstance(value.get("id"), int)
    }
    implementations = _specialize_program_implementations(parsed.implementations, native)
    providers = ProviderChain(
        (
            CapturedDslProvider(implementations),
            CapturedVjpDslProvider(implementations, native),
            BuiltinDslProvider(),
        )
    )
    vjp_requests: list[Mapping[str, Any]] = []
    other_requests: list[Mapping[str, Any]] = []
    for request in requests:
        if not isinstance(request, Mapping):
            raise PipelineCompileError("Program planner returned an invalid kernel compile request")
        hint = request.get("implementation_hint")
        if isinstance(hint, str) and hint.endswith(".vjp"):
            vjp_requests.append(request)
        else:
            other_requests.append(request)
    for request in (*vjp_requests, *other_requests):
        request_id = request.get("id")
        hint = request.get("implementation_hint")
        kind = request.get("kind")
        if not isinstance(request_id, str) or not isinstance(hint, str) or kind not in {"compute", "render"}:
            raise PipelineCompileError("Program planner returned incomplete kernel compile request metadata")
        try:
            implementation = providers.lower(request, values)
        except ImplementationUnavailable as error:
            raise PipelineCompileError(
                f"Python DSL provider cannot lower Program request {request_id!r}: {error}"
            ) from None
        if implementation is None:
            raise PipelineCompileError(
                f"no Python DSL implementation provider can lower Program request {request_id!r} ({hint!r})"
            )
        if implementation.kind != ("compute" if kind == "compute" else "graphics"):
            raise PipelineCompileError(f"Program request {request_id!r} selected an incompatible implementation")
        # Builtin copy/add share one hint across ranks. Rank (and dtype) live on
        # the generated entry; extents stay vd.dyn from the view descriptor.
        compile_key = (hint, implementation.entry)
        stage = compiled_implementations.get(compile_key)
        if stage is None:
            module_manifest = canonical_json(
                {
                    "program": parsed.identity,
                    "implementation_hint": hint,
                    "entry": implementation.entry,
                    "kind": implementation.kind,
                }
            )
            module = ShaderModuleDescriptor(
                f"{pipeline_id}/{hint}/{implementation.entry}",
                source,
                source,
                module_manifest,
            )
            stage = _compile_stage(
                module,
                ShaderStageReference(module.id, implementation.entry),
                implementation.kind,
                variant,
                target,
                compiler,
                native_target,
                implementation.mlir,
                retained_programs,
            )
            compiled_implementations[compile_key] = stage
        stages[request_id] = stage
    finalized = compiler.finalize_program_result(
        planned.reflection,
        [
            (
                request_id,
                stage.id,
                stage.entry,
                canonical_json(dict(stage.reflection)),
            )
            for request_id, stage in sorted(stages.items())
        ],
    )
    if not bool(finalized.ok):
        raise PipelineCompileError(str(finalized.diagnostics) or "Program finalization failed")
    finalized_reflection = parse_reflection_json(finalized.reflection)
    execution = finalized_reflection.get("execution")
    if not isinstance(execution, Mapping):
        raise PipelineCompileError("finalized Program reflection has no execution graph")
    reflected_targets = {stage.target for stage in stages.values()}
    if len(reflected_targets) != 1:
        raise PipelineCompileError("Program stages disagree on their reflected target")
    canonical_program = finalized_reflection.get("canonical_program")
    contracts = finalized_reflection.get("stage_contracts")
    if canonical_execution and (not isinstance(canonical_program, Mapping) or not isinstance(contracts, Mapping)):
        raise PipelineCompileError("C++ Program finalization returned no canonical deployment")
    plan = build_program_bundle_plan(
        pipeline_id,
        reflected_targets.pop(),
        variant,
        [(variant, execution, stages)],
        canonical_execution=canonical_execution,
        canonical_program=(canonical_program if isinstance(canonical_program, Mapping) else None),
    )
    if not canonical_execution:
        return plan
    assert isinstance(canonical_program, Mapping)
    assert isinstance(contracts, Mapping)
    canonical_stage_rows = canonical_program.get("stages")
    if not isinstance(canonical_stage_rows, Mapping) or set(canonical_stage_rows) != set(stages):
        raise PipelineCompileError("canonical Program stages do not exactly cover logical compute requests")
    if set(contracts) != set(stages) or any(not isinstance(contract, Mapping) for contract in contracts.values()):
        raise PipelineCompileError("canonical stage contracts do not exactly cover logical compute requests")
    contracts_by_implementation: dict[str, dict[str, Mapping[str, Any]]] = {}
    implementations_by_stage: dict[str, dict[str, Mapping[str, Any]]] = {}
    implementations = finalized_reflection.get("target_implementations")
    for logical_stage, stage in stages.items():
        contracts_by_implementation.setdefault(stage.id, {})[logical_stage] = contracts[logical_stage]
        implementation = implementations.get(logical_stage) if isinstance(implementations, Mapping) else None
        if isinstance(implementation, Mapping):
            implementations_by_stage.setdefault(stage.id, {})[logical_stage] = implementation
    canonical_stages = tuple(
        CompiledStage(
            stage.module,
            stage.module_manifest,
            stage.entry,
            stage.stage,
            stage.target,
            stage.reflection,
            stage.interface,
            stage.artifact,
            {
                **stage.metadata,
                "program_contracts": contracts_by_implementation[stage.id],
                "program_implementations": implementations_by_stage.get(stage.id, {}),
            },
        )
        for stage in plan.stages
    )
    return BundlePlan(
        plan.pipeline_id,
        plan.target,
        plan.features,
        plan.variants,
        canonical_stages,
    )


def _load_pipeline_asset_declaration(source: Path, descriptor_name: str) -> Any:
    module_name = f"_vernon_pipeline_asset_{hashlib.sha256(str(source).encode()).hexdigest()[:20]}"
    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise PipelineCompileError(f"cannot load pipeline asset source {source}")
    module = importlib.util.module_from_spec(specification)
    inserted_path = str(source.parent)
    sys.modules[module_name] = module
    sys.path.insert(0, inserted_path)
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise PipelineCompileError(f"cannot evaluate pipeline asset source {source}: {error}") from error
    finally:
        sys.path.pop(0)
        sys.modules.pop(module_name, None)
    declaration = getattr(module, descriptor_name, None)
    from .declaration import PipelineAssetDeclaration

    if not isinstance(declaration, PipelineAssetDeclaration):
        raise PipelineCompileError(f"pipeline asset declaration '{descriptor_name}' did not evaluate canonically")
    return declaration


def _materialize_plan(plan: BundlePlan, output_path: Path, target_name: str) -> Path:
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


def _canonical_deployment(
    plan: BundlePlan,
    artifact_descriptors: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, str]]:
    if len(plan.variants) != 1:
        raise PipelineCompileError("canonical deployment currently requires exactly one variant")
    variant = plan.variants[0]
    if variant.execution is None:
        raise PipelineCompileError("canonical deployment has no Program")
    stages = {stage.id: stage for stage in plan.stages}
    canonical_stage_rows = variant.execution.get("stages")
    if not isinstance(canonical_stage_rows, Mapping):
        raise PipelineCompileError("canonical deployment has no logical stages")
    blobs: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    stage_bindings: dict[str, str] = {}
    for logical_stage, implementation_stage in variant.program.items():
        stage = stages.get(implementation_stage)
        descriptor = artifact_descriptors.get(implementation_stage)
        contracts = stage.metadata.get("program_contracts") if stage is not None else None
        contract = contracts.get(logical_stage) if isinstance(contracts, Mapping) else None
        canonical_stage = canonical_stage_rows.get(logical_stage)
        if not isinstance(descriptor, Mapping) or not isinstance(contract, Mapping):
            raise PipelineCompileError(f"canonical stage {logical_stage} is incomplete")
        if not isinstance(canonical_stage, Mapping):
            raise PipelineCompileError(f"canonical Program has no logical stage {logical_stage!r}")
        path = descriptor.get("path")
        digest = descriptor.get("sha256")
        byte_length = descriptor.get("size")
        symbol = stage.metadata.get("symbol")
        entry_point = symbol if isinstance(symbol, str) else stage.entry
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not isinstance(byte_length, int)
            or not isinstance(entry_point, str)
        ):
            raise PipelineCompileError(f"canonical stage {logical_stage} has invalid artifact metadata")
        blobs[digest] = {
            "byte_length": byte_length,
            "sha256": digest,
            "location": {"tag": "external", "uri": path},
        }
        requirements = runtime_requirements(plan.target.target, (stage,))
        if requirements is None:
            raise PipelineCompileError(f"canonical stage {logical_stage} has no runtime requirements")
        requirements = dict(requirements)
        requirements.pop("compute_workgroup_size", None)
        reflection = contract.get("reflection")
        if not isinstance(reflection, Mapping):
            raise PipelineCompileError(f"canonical stage {logical_stage} has no portable reflection")
        contract_hash = hashlib.sha256(canonical_json(dict(contract)).encode("utf-8")).hexdigest()
        if canonical_stage.get("contract_hash") != contract_hash:
            raise PipelineCompileError(f"canonical stage {logical_stage} contract hash disagrees with its Program")
        artifacts[logical_stage] = {
            "tag": "stage",
            "operation": "compute",
            "contract_hash": contract_hash,
            "runtime_requirements": requirements,
            "modules": [
                {
                    "role": "compute",
                    "format": stage.artifact.format,
                    "entry_point": entry_point,
                    "blob": digest,
                    "offset": 0,
                    "byte_length": byte_length,
                    "sha256": digest,
                }
            ],
            "reflection": copy.deepcopy(dict(reflection)),
        }
        implementations = stage.metadata.get("program_implementations")
        implementation = implementations.get(logical_stage) if isinstance(implementations, Mapping) else None
        if isinstance(implementation, Mapping):
            artifacts[logical_stage]["implementation"] = copy.deepcopy(dict(implementation))
        stage_bindings[logical_stage] = logical_stage
    return (
        copy.deepcopy(dict(variant.execution)),
        {
            "target": copy.deepcopy(plan.target.spec),
            "blobs": blobs,
            "artifacts": artifacts,
        },
        stage_bindings,
    )


def _direct_compiled_stage(
    result: Any,
    *,
    target: TargetOptions,
    entry: str,
) -> CompiledStage:
    compiled = compiled_stage_from_program(
        result,
        module=f"interactive/kernel/{entry}",
        module_manifest=canonical_json({"entry": entry}),
        entry=entry,
        target=target,
    )
    if compiled.stage != "compute":
        raise PipelineCompileError("direct Kernel canonical deployment requires a compute stage")
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
    return compiled


def _canonical_kernel_deployment(
    compiled: CompiledStage,
    finalized_reflection: Mapping[str, Any],
    *,
    target: TargetOptions,
    request_id: str,
    output: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, str], CompiledStage]:
    program = finalized_reflection.get("canonical_program")
    contracts = finalized_reflection.get("stage_contracts")
    if not isinstance(program, Mapping) or not isinstance(contracts, Mapping):
        raise PipelineCompileError("C++ Program finalization returned no canonical deployment")
    contract = contracts.get(request_id)
    canonical_stages = program.get("stages")
    canonical_stage = canonical_stages.get(request_id) if isinstance(canonical_stages, Mapping) else None
    if not isinstance(contract, Mapping) or not isinstance(canonical_stage, Mapping):
        raise PipelineCompileError("C++ Program finalization returned an incomplete direct stage contract")
    contract_hash = canonical_stage.get("contract_hash")
    if not isinstance(contract_hash, str) or not contract_hash:
        raise PipelineCompileError("C++ Program finalization returned no direct stage contract hash")
    descriptor = write_external_artifact(
        output,
        compiled.artifact.data,
        compiled.artifact.format,
        compiled.stage,
        compiled.artifact.filename,
    )
    digest = descriptor["sha256"]
    requirements = runtime_requirements(target.target, (compiled,))
    if requirements is None:
        raise PipelineCompileError("direct Kernel has no canonical runtime requirements")
    requirements = dict(requirements)
    requirements.pop("compute_workgroup_size", None)
    entry_point = compiled.metadata.get("symbol", compiled.entry)
    reflection = contract.get("reflection")
    if not isinstance(reflection, Mapping):
        raise PipelineCompileError("C++ Program finalization returned no portable direct stage reflection")
    implementations = finalized_reflection.get("target_implementations")
    implementation = implementations.get(request_id) if isinstance(implementations, Mapping) else None
    artifact_system = {
        "target": copy.deepcopy(target.spec),
        "blobs": {
            digest: {
                "byte_length": descriptor["size"],
                "sha256": digest,
                "location": {"tag": "external", "uri": descriptor["path"]},
            }
        },
        "artifacts": {
            compiled.id: {
                "tag": "stage",
                "operation": "compute",
                "contract_hash": contract_hash,
                "runtime_requirements": requirements,
                "modules": [
                    {
                        "role": "compute",
                        "format": compiled.artifact.format,
                        "entry_point": entry_point,
                        "blob": digest,
                        "offset": 0,
                        "byte_length": descriptor["size"],
                        "sha256": digest,
                    }
                ],
                "reflection": copy.deepcopy(dict(reflection)),
            }
        },
    }
    if isinstance(implementation, Mapping):
        artifact_system["artifacts"][compiled.id]["implementation"] = copy.deepcopy(dict(implementation))
    return (
        copy.deepcopy(dict(program)),
        artifact_system,
        {request_id: compiled.id},
        compiled,
    )


def _canonical_graphics_deployment(
    compiled: Sequence[CompiledStage],
    finalized_reflection: Mapping[str, Any],
    *,
    target: TargetOptions,
    request_id: str,
    artifact_id: str,
    output: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, str]]:
    program = finalized_reflection.get("canonical_program")
    contracts = finalized_reflection.get("stage_contracts")
    if not isinstance(program, Mapping) or not isinstance(contracts, Mapping):
        raise PipelineCompileError("C++ graphics finalization returned no canonical deployment")
    contract = contracts.get(request_id)
    canonical_stages = program.get("stages")
    canonical_stage = canonical_stages.get(request_id) if isinstance(canonical_stages, Mapping) else None
    if not isinstance(contract, Mapping) or not isinstance(canonical_stage, Mapping):
        raise PipelineCompileError("C++ graphics finalization returned an incomplete StageContract")
    contract_hash = canonical_stage.get("contract_hash")
    reflection = contract.get("reflection")
    if not isinstance(contract_hash, str) or not isinstance(reflection, Mapping):
        raise PipelineCompileError("C++ graphics finalization returned invalid portable reflection")
    implementations = finalized_reflection.get("target_implementations")
    implementation = implementations.get(request_id) if isinstance(implementations, Mapping) else None
    requirements = runtime_requirements(target.target, tuple(compiled))
    if requirements is None:
        raise PipelineCompileError("graphics Program has no canonical runtime requirements")
    blobs: dict[str, Any] = {}
    modules: list[dict[str, Any]] = []
    for stage in compiled:
        descriptor = write_external_artifact(
            output,
            stage.artifact.data,
            stage.artifact.format,
            stage.stage,
            stage.artifact.filename,
        )
        digest = descriptor["sha256"]
        blobs[digest] = {
            "byte_length": descriptor["size"],
            "sha256": digest,
            "location": {"tag": "external", "uri": descriptor["path"]},
        }
        modules.append(
            {
                "role": stage.stage,
                "format": stage.artifact.format,
                "entry_point": stage.metadata.get("symbol", stage.entry),
                "blob": digest,
                "offset": 0,
                "byte_length": descriptor["size"],
                "sha256": digest,
            }
        )
    modules.sort(key=lambda module: ("vertex", "fragment").index(module["role"]))
    artifact_system = {
        "target": copy.deepcopy(target.spec),
        "blobs": blobs,
        "artifacts": {
            artifact_id: {
                "tag": "stage",
                "operation": "graphics",
                "contract_hash": contract_hash,
                "runtime_requirements": dict(requirements),
                "modules": modules,
                "reflection": copy.deepcopy(dict(reflection)),
            }
        },
    }
    implementation_row: dict[str, Any] = dict(implementation) if isinstance(implementation, Mapping) else {}
    if target.target == "metal":
        slots: list[Any] = []
        for compiled_stage in compiled:
            reflected_slots = compiled_stage.reflection.get("metal_resource_slots")
            if isinstance(reflected_slots, list):
                slots.extend(copy.deepcopy(slot) for slot in reflected_slots if isinstance(slot, Mapping))
        if "endpoints" not in implementation_row:
            implementation_row["endpoints"] = []
        implementation_row["metal_resource_slots"] = slots
    if implementation_row:
        artifact_system["artifacts"][artifact_id]["implementation"] = copy.deepcopy(implementation_row)
    return copy.deepcopy(dict(program)), artifact_system, {request_id: artifact_id}


def _compile_module_bundle_plan(
    declaration: Any,
    pipeline: Any,
    target: TargetOptions,
    native: Any,
    native_target: Any,
    retained_programs: list[tuple[CompiledStage, Any]] | None = None,
) -> BundlePlan:
    from ..module import Module
    from ..program import _parse_module_program

    if not isinstance(declaration.program, Module):
        raise PipelineCompileError("pipeline asset declared a host Program that is not a Vernon Module")
    parsed = _parse_module_program(declaration.program)
    compiler = native.Compiler()
    variant_plans = [
        _compile_program_bundle_plan(
            parsed,
            pipeline_id=pipeline.id,
            variant=variant,
            target=target,
            compiler=compiler,
            native=native,
            native_target=native_target,
            retained_programs=retained_programs,
            canonical_execution=True,
        )
        for variant in pipeline.variants
    ]
    targets = {canonical_json(plan.target.spec): plan.target for plan in variant_plans}
    if len(targets) != 1:
        raise PipelineCompileError("compiler returned inconsistent target options across Program variants")
    stages = {stage.id: stage for plan in variant_plans for stage in plan.stages}
    return BundlePlan(
        pipeline.id,
        next(iter(targets.values())),
        tuple(sorted({feature for variant in pipeline.variants for feature in variant})),
        tuple(variant for variant_plan in variant_plans for variant in variant_plan.variants),
        tuple(stages[key] for key in sorted(stages)),
    )


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
    if pipeline.transform is not None and pipeline.program_kind == "stages" and set(pipeline.stages) != {"compute"}:
        raise PipelineCompileError(
            "graphics VJP asset cooking is not supported; automatic differentiation requires a compute pipeline"
        )
    if target_name == "cpu" and pipeline.program_kind == "stages" and set(pipeline.stages) != {"compute"}:
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
    if pipeline.program_kind == "module":
        if pipeline.transform is not None:
            raise PipelineCompileError("Module VJP asset cooking is not enabled until canonical primal Programs ship")
        declaration = _load_pipeline_asset_declaration(source, descriptor_name)
        plan = _compile_module_bundle_plan(
            declaration,
            pipeline,
            resolved_target,
            native,
            native_target,
        )
        return _materialize_plan(plan, output_path, target_name)
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
            planning_policy=str(transform_values.get("planning_policy", "min_memory")),
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
            frontend = Compiler().compile_request(
                FrontendCompileRequest(
                    selected_modules["compute"].source,
                    pipeline.stages["compute"].entry,
                    variant,
                )
            )
            if not is_structured_vjp_abi_eligible(frontend, transform):
                raise PipelineCompileError("autodiff requires a structured VJP with explicit writable Storage outputs")
            try:
                structured = build_structured_vjp(native, frontend, transform)
            except ValueError as error:
                raise PipelineCompileError(str(error)) from None
            variant_transform = structured.transform
            profile_plan = structured.plan
            profile_modules = structured.profiles
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
                    "residual_storage": structured.residual_storage_kind,
                    "static_tape_bytes_hint": profile_plan.tape_bytes,
                    "required_primal_paths": list(profile_plan.required_primal_paths),
                    "source_kind_counts": dict(profile_plan.source_kind_counts),
                    "cost_components": dict(profile_plan.cost_components),
                    "selected_policy": profile_plan.selected_policy,
                    "whole_dispatch_retention_permitted": profile_plan.whole_dispatch_retention_permitted,
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
    return _materialize_plan(plan, output_path, target_name)


__all__ = ["cook_pipeline_asset"]
