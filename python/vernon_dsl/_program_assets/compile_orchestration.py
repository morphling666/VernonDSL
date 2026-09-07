"""Compilation orchestration for captured Program Assets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from .._runtime.operators import ImplementationUnavailable
from .._versions import PROGRAM_VERSION
from ..bundle import (
    BundlePlan,
    CompiledArtifact,
    CompiledStage,
    ProgramCompileError,
    TargetOptions,
    build_program_plan,
    canonical_json,
    compiled_stage_from_program,
    parse_reflection_json,
)
from ..program_frontend import (
    BuiltinDslProvider,
    CapturedDslProvider,
    CapturedVjpDslProvider,
    ParsedProgram,
    ProgramImplementation,
    ProviderChain,
)
from .capture import CapturedProgram
from .deployment_validation import validate_canonical_deployment


def _native_module() -> Any:
    try:
        from .. import _native as native
    except (ImportError, OSError):
        raise ProgramCompileError(
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
        raise ProgramCompileError(f"target '{target}' is not available through the native compiler") from None
    try:
        return getattr(native.Target, name)
    except AttributeError:
        raise ProgramCompileError(f"target '{target}' is not available through the native compiler") from None


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
        raise ProgramCompileError(f"CPU compiler reflection has no exported symbol for {stage.entry}")
    if not isinstance(reflected_options, Mapping):
        raise ProgramCompileError("CPU compiler reflection has no target options")
    target_triple = reflected_options.get("triple")
    if not isinstance(target_triple, str) or not target_triple:
        raise ProgramCompileError("CPU compiler reflection has no normalized target triple")
    metadata: dict[str, Any] = {
        "symbol": symbol,
        "program_version": PROGRAM_VERSION,
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
    module_id: str,
    entry: str,
    stage: str,
    variant: tuple[str, ...],
    target: TargetOptions,
    compiler: Any,
    native_target: Any,
    mlir: str,
    retained_programs: list[tuple[CompiledStage, Any]] | None = None,
) -> CompiledStage:
    result = compiler.compile_program_result(mlir, native_target, **target.native_options)
    compiled = compiled_stage_from_program(
        result,
        module=module_id,
        entry=entry,
        target=target,
    )
    if compiled.stage != stage:
        raise ProgramCompileError(f"compiler reflected {entry} as {compiled.stage}, expected {stage}")
    if target.target == "cpu":
        compiled = CompiledStage(
            compiled.module,
            compiled.entry,
            compiled.stage,
            compiled.target,
            compiled.reflection,
            compiled.interface,
            compiled.artifact,
            _cpu_stage_metadata(compiled),
        )
        if compiled.artifact.format != "relocatable_object":
            raise ProgramCompileError("CPU cooking requires the compiler relocatable object artifact")
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
            raise ProgramCompileError(
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


def _compile_program_variant(
    parsed: ParsedProgram,
    *,
    program_id: str,
    variant: tuple[str, ...],
    target: TargetOptions,
    compiler: Any,
    native: Any,
    native_target: Any,
    retained_programs: list[tuple[CompiledStage, Any]] | None = None,
) -> tuple[TargetOptions, tuple[str, ...], Mapping[str, CompiledStage], Mapping[str, Any]]:
    planned = compiler.plan_program_result(parsed.mlir)
    if not bool(planned.ok):
        raise ProgramCompileError(str(planned.diagnostics) or "Program planning failed")
    reflection = parse_reflection_json(planned.reflection)
    execution = reflection.get("program_plan")
    if not isinstance(execution, Mapping):
        raise ProgramCompileError("Program compiler reflection has no typed Program plan")
    requests = reflection.get("kernel_compile_requests")
    if not isinstance(requests, list):
        raise ProgramCompileError("Program compiler reflection has no kernel compile requests")

    stages: dict[str, CompiledStage] = {}
    compiled_implementations: dict[tuple[str, str], CompiledStage] = {}
    graphics_implementations: dict[str, tuple[CompiledStage, tuple[CompiledStage, ...]]] = {}
    finalize_rows: list[tuple[str, str, str, str]] = []
    reflected_values = execution.get("values")
    if not isinstance(reflected_values, list):
        raise ProgramCompileError("Program compiler reflection has no reflected values")
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
            BuiltinDslProvider(native=native),
        )
    )
    vjp_requests: list[Mapping[str, Any]] = []
    other_requests: list[Mapping[str, Any]] = []
    for request in requests:
        if not isinstance(request, Mapping):
            raise ProgramCompileError("Program planner returned an invalid kernel compile request")
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
            raise ProgramCompileError("Program planner returned incomplete kernel compile request metadata")
        try:
            implementation = providers.lower(request, values)
        except ImplementationUnavailable as error:
            raise ProgramCompileError(
                f"Python DSL provider cannot lower Program request {request_id!r}: {error}"
            ) from None
        if implementation is None:
            raise ProgramCompileError(
                f"no Python DSL implementation provider can lower Program request {request_id!r} ({hint!r})"
            )
        if implementation.kind != ("compute" if kind == "compute" else "graphics"):
            raise ProgramCompileError(f"Program request {request_id!r} selected an incompatible implementation")
        if implementation.kind == "graphics":
            cached_graphics = graphics_implementations.get(hint)
            if cached_graphics is None:
                if not implementation.graphics_stages:
                    raise ProgramCompileError(f"graphics Program implementation {hint!r} has no shader stages")
                compiled_graphics: list[CompiledStage] = []
                for role, entry, stage_mlir in implementation.graphics_stages:
                    module_id = f"{program_id}/{hint}/{entry}"
                    compiled_graphics.append(
                        _compile_stage(
                            module_id,
                            entry,
                            role,
                            variant,
                            target,
                            compiler,
                            native_target,
                            stage_mlir,
                            retained_programs,
                        )
                    )
                offsets: list[int] = []
                payload = bytearray()
                modules: list[dict[str, Any]] = []
                for compiled_stage in compiled_graphics:
                    offsets.append(len(payload))
                    payload.extend(compiled_stage.artifact.data)
                    modules.append(
                        {
                            "role": compiled_stage.stage,
                            "format": compiled_stage.artifact.format,
                            "entry_point": compiled_stage.metadata.get("symbol", compiled_stage.entry),
                            "offset": offsets[-1],
                            "byte_length": len(compiled_stage.artifact.data),
                            "sha256": compiled_stage.artifact.sha256,
                        }
                    )
                representative = CompiledStage(
                    f"{program_id}/{hint}",
                    implementation.entry,
                    "graphics",
                    target,
                    {},
                    {},
                    CompiledArtifact("graphics_stage_group", bytes(payload), "graphics-stage-group.bin"),
                    {
                        "graphics_modules": tuple(modules),
                        "graphics_compiled_stages": tuple(compiled_graphics),
                    },
                )
                cached_graphics = (representative, tuple(compiled_graphics))
                graphics_implementations[hint] = cached_graphics
            stage, compiled_graphic_stages = cached_graphics
            stages[request_id] = stage
            finalize_rows.extend(
                (
                    request_id,
                    stage.id,
                    compiled_stage.entry,
                    canonical_json(dict(compiled_stage.reflection)),
                )
                for compiled_stage in compiled_graphic_stages
            )
            continue
        compile_key = (hint, implementation.entry)
        stage = compiled_implementations.get(compile_key)
        if stage is None:
            module_id = f"{program_id}/{hint}/{implementation.entry}"
            stage = _compile_stage(
                module_id,
                implementation.entry,
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
        finalize_rows.append(
            (
                request_id,
                stage.id,
                stage.entry,
                canonical_json(dict(stage.reflection)),
            )
        )
    finalized = compiler.finalize_program_result(
        planned.reflection,
        finalize_rows,
    )
    if not bool(finalized.ok):
        raise ProgramCompileError(str(finalized.diagnostics) or "Program finalization failed")
    finalized_reflection = parse_reflection_json(finalized.reflection)
    reflected_targets = {stage.target for stage in stages.values()}
    if len(reflected_targets) != 1:
        raise ProgramCompileError("Program stages disagree on their reflected target")
    canonical_program = finalized_reflection.get("canonical_program")
    contracts = finalized_reflection.get("stage_contracts")
    if not isinstance(canonical_program, Mapping) or not isinstance(contracts, Mapping):
        raise ProgramCompileError("C++ Program finalization returned no canonical deployment")
    validate_canonical_deployment(canonical_program)
    reflected_target = reflected_targets.pop()
    if set(contracts) != set(stages) or any(not isinstance(contract, Mapping) for contract in contracts.values()):
        raise ProgramCompileError("canonical stage contracts do not exactly cover logical compute requests")
    contracts_by_implementation: dict[str, dict[str, Mapping[str, Any]]] = {}
    implementations_by_stage: dict[str, dict[str, Mapping[str, Any]]] = {}
    target_implementations = finalized_reflection.get("target_implementations")
    for logical_stage, stage in stages.items():
        contracts_by_implementation.setdefault(stage.id, {})[logical_stage] = contracts[logical_stage]
        implementation = (
            target_implementations.get(logical_stage) if isinstance(target_implementations, Mapping) else None
        )
        if isinstance(implementation, Mapping):
            implementations_by_stage.setdefault(stage.id, {})[logical_stage] = implementation
    canonical_stages = {
        stage.id: CompiledStage(
            stage.module,
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
        for stage in {stage.id: stage for stage in stages.values()}.values()
    }
    return (
        reflected_target,
        variant,
        {logical_stage: canonical_stages[stage.id] for logical_stage, stage in stages.items()},
        canonical_program,
    )


def compile_captured_program(
    captured: CapturedProgram,
    target: TargetOptions,
    retained_programs: list[tuple[CompiledStage, Any]] | None = None,
) -> BundlePlan:
    """Compile canonical captured variants into a target-specific bundle plan."""
    if not isinstance(captured, CapturedProgram):
        raise TypeError("compile_captured_program requires a CapturedProgram")
    native = _native_module()
    native_target = _native_target(native, target.target)
    compiler = native.Compiler()
    compiled_variants = [
        _compile_program_variant(
            variant.ir,
            program_id=captured.id,
            variant=variant.key,
            target=target,
            compiler=compiler,
            native=native,
            native_target=native_target,
            retained_programs=retained_programs,
        )
        for variant in captured.variants
    ]
    targets = {canonical_json(compiled[0].spec): compiled[0] for compiled in compiled_variants}
    if len(targets) != 1:
        raise ProgramCompileError("compiler returned inconsistent target options across Program variants")
    variants = [(key, stages, program) for _, key, stages, program in compiled_variants]
    return build_program_plan(captured.id, next(iter(targets.values())), variants)


__all__ = ["compile_captured_program"]
