from __future__ import annotations

import ast
import atexit
import hashlib
import inspect
import json
import tempfile
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from .._dtypes import NUMPY_DTYPE_BY_SCALAR
from .._versions import COMPILER_CONTRACT_VERSION, PIPELINE_VERSION
from ..bundle import canonical_json, make_target_options, parse_reflection_json
from ..compiler import Compiler, FrontendCompileRequest, FrontendCompileResult
from ..frontend.model import ConcreteType
from ..host_values import pack_host_value
from ..types import TypeExpr, _Scalar
from .binding import _DispatchBorrowLease, _PersistentBindingTable
from .resource_common import _session_state
from .tensor import TensorStorage, TensorView
from .texture import _TextureResource


@dataclass(frozen=True)
class _LoweredKernel:
    """Source + annotations compiled to MLIR. No target, no runtime tensors."""

    frontend: FrontendCompileResult
    function: ast.FunctionDef
    source: ast.FunctionDef
    builtins: tuple[str, ...]


@dataclass
class _CompiledKernel:
    mlir: str
    function: ast.FunctionDef
    frontend: FrontendCompileResult
    builtin_names: tuple[str, ...]
    program: Any
    canonical_directory: tempfile.TemporaryDirectory[str]
    canonical_program: bytes
    canonical_artifact_system: bytes
    canonical_stage_bindings: dict[str, str]
    canonical_stage: Any
    dependency_hashes: tuple[tuple[Path, str], ...] = ()
    native: Any | None = None
    native_generation: int = -1


class _DependencyTracked(Protocol):
    dependency_hashes: tuple[tuple[Path, str], ...]


class Kernel:
    """Device kernel: lower source to MLIR, specialize MLIR to a native artifact, bind tensors at launch."""

    _cache: ClassVar[dict[str, _CompiledKernel]] = {}
    _instances: ClassVar[weakref.WeakSet[Kernel]] = weakref.WeakSet()

    def __init__(self, function: Any, *, workgroup_size: tuple[int, int, int] = (1, 1, 1)):
        if len(workgroup_size) != 3 or any(not isinstance(value, int) or value <= 0 for value in workgroup_size):
            raise ValueError("workgroup_size must contain three positive integers")
        self.__name__ = function.__name__
        self.__module__ = function.__module__
        self.__doc__ = function.__doc__
        self.__vernon_dsl__ = ("compute", {"workgroup_size": workgroup_size})
        self._function = function
        self._file = Path(inspect.getsourcefile(function) or "")
        self._entry = function.__name__
        self._workgroup_size = workgroup_size
        self._globals = function.__globals__
        self._direct_binding_caches: dict[tuple[Any, ...], _PersistentBindingTable] = {}
        self.compile_count = 0
        self._instances.add(self)

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        from .autodiff import clear_vjp_cache

        clear_vjp_cache()

    @classmethod
    def invalidate_loaded(cls) -> None:
        for compiled in cls._cache.values():
            compiled.native = None
            compiled.native_generation = -1
        for kernel in cls._instances:
            for cache in kernel._direct_binding_caches.values():
                cache.clear()

    def _resolve_dependency(self, spelling: str) -> Path:
        path = Path(spelling)
        if path.is_absolute():
            return path
        for parent in (self._file.parent, *self._file.parents):
            candidate = parent / path
            if candidate.is_file():
                return candidate
        return path

    def _dependency_hashes(self, frontend: FrontendCompileResult) -> tuple[tuple[Path, str], ...]:
        return tuple(
            (self._resolve_dependency(str(path)), str(digest))
            for path, digest in frontend.semantic_inputs.get("dependencies", ())
        )

    @staticmethod
    def _dependencies_current(compiled: _DependencyTracked) -> bool:
        try:
            return all(
                hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest() == digest
                for path, digest in compiled.dependency_hashes
            )
        except OSError:
            return False

    @staticmethod
    def _argument_signature(value: Any) -> tuple[Any, ...]:
        if isinstance(value, (TensorStorage, TensorView)):
            return ("tensor",)
        if isinstance(value, _TextureResource):
            return ("texture", tuple(value.shape))
        return ("value", type(value).__module__, type(value).__qualname__)

    def _dispatch_key(
        self,
        arguments: tuple[Any, ...],
        features: tuple[str, ...],
        target: str,
        target_options: tuple[tuple[str, Any], ...],
    ) -> tuple[Any, ...]:
        source_digest = hashlib.sha256(self._file.read_bytes()).hexdigest()
        constants = tuple(
            sorted(
                (name, type(value).__name__, repr(value))
                for name, value in self._globals.items()
                if isinstance(value, (int, float, bool))
            )
        )
        return (
            str(self._file.resolve()),
            source_digest,
            self._entry,
            self._workgroup_size,
            target,
            target_options,
            features,
            tuple(self._argument_signature(value) for value in arguments),
            constants,
        )

    @staticmethod
    def _load_native(compiled: _CompiledKernel, state: Any) -> None:
        if compiled.native is not None and compiled.native_generation == state._runtime_generation:
            return
        cpu_stages = (
            [
                (
                    compiled.canonical_stage.metadata["symbol"],
                    compiled.canonical_stage.entry,
                    compiled.program,
                )
            ]
            if state._architecture == state.cpu
            else []
        )
        compiled.native = state._native_runtime.load_canonical_endpoint(
            compiled.canonical_program,
            compiled.canonical_artifact_system,
            compiled.canonical_directory.name,
            compiled.canonical_stage_bindings,
            cpu_stages,
        )
        compiled.native_generation = state._runtime_generation

    @staticmethod
    def _annotation_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    @classmethod
    def _builtin_parameters(cls, function: ast.FunctionDef) -> tuple[str, ...]:
        return tuple(
            argument.arg
            for argument in function.args.args
            if argument.annotation
            and any(
                isinstance(node, ast.Call)
                and cls._annotation_name(node.func) == "builtin"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                for node in ast.walk(argument.annotation)
            )
        )

    @classmethod
    def _tensor_view_access(cls, annotation: ast.expr | None) -> str | None:
        if not isinstance(annotation, ast.Subscript) or cls._annotation_name(annotation.value) != "TensorView":
            return None
        arguments = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
        if len(arguments) != 3:
            return None
        access = cls._annotation_name(arguments[2])
        return access if access in {"read", "write", "read_write"} else None

    @classmethod
    def _is_static_tensor_annotation(cls, annotation: ast.expr | None) -> bool:
        if isinstance(annotation, ast.Subscript) and cls._annotation_name(annotation.value) == "Annotated":
            annotation = annotation.slice.elts[0] if isinstance(annotation.slice, ast.Tuple) else annotation.slice
        return isinstance(annotation, ast.Subscript) and cls._annotation_name(annotation.value) == "Tensor"

    def _normalize_arguments(
        self,
        function: ast.FunctionDef,
        builtins: tuple[str, ...],
        arguments: tuple[Any, ...],
    ) -> tuple[list[str], tuple[Any, ...]]:
        user_parameters = [argument.arg for argument in function.args.args if argument.arg not in builtins]
        if len(arguments) != len(user_parameters):
            raise TypeError(f"{self._entry} expects {len(user_parameters)} launch arguments")
        declared_access = {
            argument.arg: self._tensor_view_access(argument.annotation)
            for argument in function.args.args
            if argument.arg in user_parameters
        }
        normalized = tuple(
            value._full_view(declared_access[name])
            if isinstance(value, TensorStorage) and declared_access[name] is not None
            else value
            for name, value in zip(user_parameters, arguments, strict=True)
        )
        return user_parameters, normalized

    def _validate_tensor_view_arguments(
        self,
        frontend: FrontendCompileResult,
        user_parameters: list[str],
        arguments: tuple[Any, ...],
    ) -> tuple[Any, _DispatchBorrowLease] | None:
        entry = next(
            (function for function in frontend.typed_functions if function.source.name == self._entry),
            None,
        )
        if entry is None:
            raise RuntimeError("compiled kernel has no typed entry function")
        typed_parameters = {parameter.name: parameter for parameter in entry.parameters}
        access_compatibility = {
            "read": {"read", "read_write"},
            "write": {"write", "read_write"},
            "read_write": {"read_write"},
        }
        annotations = inspect.get_annotations(self._function, eval_str=True)

        def dsl_signature(value_type: Any) -> Any:
            if value_type.kind == "scalar":
                return ("scalar", value_type.name)
            if value_type.kind == "struct":
                return ("struct", value_type.name)
            if value_type.kind == "tuple":
                return ("tuple", tuple(dsl_signature(item) for item in value_type.arguments))
            if value_type.kind == "tensor":
                return (
                    "tensor",
                    dsl_signature(value_type.arguments[0]),
                    tuple(value_type.arguments[1:]),
                )
            return (value_type.kind, value_type.name)

        def runtime_signature(annotation: Any) -> Any:
            if isinstance(annotation, _Scalar):
                return ("scalar", annotation.name)
            if isinstance(annotation, TypeExpr):
                if annotation.name == "Tuple":
                    return ("tuple", tuple(runtime_signature(item) for item in annotation.arguments))
                if annotation.name == "Tensor" and len(annotation.arguments) == 2:
                    return (
                        "tensor",
                        runtime_signature(annotation.arguments[0]),
                        tuple(annotation.arguments[1]),
                    )
                if annotation.name in {"Vector", "Matrix"}:
                    rank = 1 if annotation.name == "Vector" else 2
                    if len(annotation.arguments) == rank + 1:
                        return (
                            "tensor",
                            runtime_signature(annotation.arguments[0]),
                            tuple(annotation.arguments[1:]),
                        )
            if isinstance(annotation, type) and getattr(annotation, "__vernon_dsl__", (None, {}))[0] == "struct":
                return ("struct", annotation.__name__)
            return None

        for name, value in zip(user_parameters, arguments, strict=True):
            parameter = typed_parameters[name]
            if parameter.type.kind != "tensor_view":
                if parameter.type.kind == "tensor" and isinstance(value, (TensorStorage, TensorView)):
                    element = parameter.type.arguments[0]
                    expected_shape = tuple(parameter.type.arguments[1:])
                    element_type = value.element_type if isinstance(value, TensorView) else value._element_type
                    if tuple(value.shape) != expected_shape:
                        raise TypeError(
                            f"kernel Tensor argument {name!r} has shape {tuple(value.shape)}, expected {expected_shape}"
                        )
                    if not isinstance(element, ConcreteType):
                        raise RuntimeError(f"kernel Tensor argument {name!r} has an unresolved element type")
                    if runtime_signature(element_type) != dsl_signature(element):
                        raise TypeError(f"kernel Tensor argument {name!r} element type does not match {element.name}")
                    continue
                if isinstance(value, (TensorStorage, TensorView)):
                    raise TypeError(f"kernel argument {name!r} is runtime storage but its annotation is not TensorView")
                if parameter.type.kind == "struct":
                    annotation = annotations.get(name)
                    if isinstance(value, Mapping):
                        if runtime_signature(annotation) != dsl_signature(parameter.type):
                            raise RuntimeError(f"kernel Struct argument {name!r} has an inconsistent host annotation")
                        pack_host_value(annotation, value, name)
                    elif runtime_signature(type(value)) != dsl_signature(parameter.type):
                        raise TypeError(
                            f"kernel Struct argument {name!r} has type {type(value).__name__}, "
                            f"expected {parameter.type.name}"
                        )
                continue
            if not isinstance(value, TensorView):
                raise TypeError(f"kernel argument {name!r} must be a TensorView")
            element, shape, declared_access, address_space = parameter.type.arguments
            if not isinstance(element, ConcreteType):
                raise RuntimeError(f"kernel TensorView argument {name!r} has an unresolved element type")
            if not isinstance(shape, tuple):
                raise RuntimeError(f"kernel TensorView argument {name!r} has an unresolved shape")
            if address_space != "device":
                raise RuntimeError(f"kernel parameter {name!r} has non-device TensorView address space")
            if len(value.shape) != len(shape):
                raise TypeError(
                    f"kernel TensorView argument {name!r} has rank {len(value.shape)}, expected {len(shape)}"
                )
            for dimension, (actual, expected) in enumerate(zip(value.shape, shape, strict=True)):
                if expected != "?" and actual != expected:
                    raise TypeError(
                        f"kernel TensorView argument {name!r} dimension {dimension} is {actual}, expected {expected}"
                    )
            if element.kind == "scalar":
                matches_element = value.dtype == NUMPY_DTYPE_BY_SCALAR[element.name]
            else:
                matches_element = runtime_signature(value.element_type) == dsl_signature(element)
            if not matches_element:
                raise TypeError(
                    f"kernel TensorView argument {name!r} dtype {value.dtype} does not match {element.name}"
                )
            if value.access not in access_compatibility[declared_access]:
                raise TypeError(
                    f"kernel TensorView argument {name!r} access {value.access!r} does not satisfy {declared_access!r}"
                )

    def _annotation_shapes(self, frontend: FrontendCompileResult) -> dict[str, tuple[int, ...]]:
        entry = next(
            (function for function in frontend.typed_functions if function.source.name == self._entry),
            None,
        )
        if entry is None:
            return {}
        shapes: dict[str, tuple[int, ...]] = {}
        for parameter in entry.parameters:
            if parameter.type.kind != "tensor_view":
                continue
            shape = parameter.type.arguments[1]
            if isinstance(shape, tuple) and all(isinstance(extent, int) for extent in shape):
                shapes[parameter.name] = tuple(int(extent) for extent in shape)
        return shapes

    def _bind_launch(self, lowered: _LoweredKernel, arguments: tuple[Any, ...]) -> None:
        user_parameters, normalized = self._normalize_arguments(lowered.source, lowered.builtins, arguments)
        self._validate_tensor_view_arguments(lowered.frontend, user_parameters, normalized)

    def _session_target(self) -> tuple[Any, Any, Any]:
        state = _session_state()
        if state._native is None or state._native_runtime is None:
            raise RuntimeError(f"{state._architecture.name} kernel execution requires the native runtime")
        target = {
            state.cpu: state._native.Target.CPU,
            state.cuda: state._native.Target.CUDA,
            state.vulkan: state._native.Target.VULKAN,
            state.directx: state._native.Target.DIRECTX,
            state.metal: state._native.Target.METAL,
            state.opengl: state._native.Target.OPENGL,
            state.opengles: state._native.Target.OPENGL_ES,
        }.get(state._architecture)
        options = make_target_options(
            state._architecture.name,
            {"version": state._interactive_glsl_version()}
            if state._architecture in {state.opengl, state.opengles}
            else {},
        )
        return state, target, options

    def _lower(
        self,
        features: tuple[str, ...] = (),
        *,
        autodiff_planning_policy: str | None = None,
    ) -> _LoweredKernel:
        source = self._file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(self._file))
        function = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == self._entry),
            None,
        )
        if function is None:
            raise RuntimeError("kernel functions must be top-level definitions in files")
        builtins = self._builtin_parameters(function)
        loaded_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        constants = tuple(
            (name, value)
            for name, value in self._globals.items()
            if name in loaded_names and isinstance(value, (int, float, bool))
        )
        request = FrontendCompileRequest(
            self._file,
            self._entry,
            features,
            (),
            constants,
            self._workgroup_size,
            autodiff_planning_policy=autodiff_planning_policy,
        )
        frontend = Compiler().compile_request(request)
        specialized_tree = ast.parse(frontend.specialized_source, filename=str(self._file))
        specialized_function = next(
            node for node in specialized_tree.body if isinstance(node, ast.FunctionDef) and node.name == self._entry
        )
        return _LoweredKernel(frontend, specialized_function, function, builtins)

    def compile_artifact(self, *arguments: Any, target: str) -> tuple[bytes, str]:
        state = _session_state()
        if state._native is None:
            raise RuntimeError("native artifact compilation requires vernon_dsl._native")
        targets = {
            "cpu": state._native.Target.CPU,
            "cuda": state._native.Target.CUDA,
            "vulkan": state._native.Target.VULKAN,
            "directx": state._native.Target.DIRECTX,
            "metal": state._native.Target.METAL,
            "opengl": state._native.Target.OPENGL,
            "opengles": state._native.Target.OPENGL_ES,
        }
        if target not in targets:
            raise ValueError("target must be cpu, cuda, vulkan, directx, metal, opengl, or opengles")
        lowered = self._lower()
        if arguments:
            self._bind_launch(lowered, arguments)
        options = make_target_options(
            target,
            {"version": 430} if target == "opengl" else {"version": 310} if target == "opengles" else {},
        )
        program = state._native.Compiler().compile_program_result(
            lowered.frontend.mlir, targets[target], **options.native_options
        )
        if not program.ok:
            raise RuntimeError(program.diagnostics)
        if len(program.artifacts) != 1:
            raise RuntimeError("kernel compilation must produce exactly one artifact")
        return bytes(program.artifacts[0][1]), str(program.reflection)

    def specialize(
        self,
        features: tuple[str, ...] = (),
        *,
        lowered: _LoweredKernel | None = None,
    ) -> _CompiledKernel:
        """Lowered MLIR + session target → native artifact.

        Runtime tensors, GraphBuffer extents, and other invoke-time shapes are
        not inputs. Static annotation extents may be recorded; `vd.dyn` stays
        dynamic until C++ bind.
        """

        state, target, options = self._session_target()
        lowered = lowered or self._lower(features)
        frontend = lowered.frontend
        key = hashlib.sha256(
            canonical_json(
                {
                    "compiler_contract_version": COMPILER_CONTRACT_VERSION,
                    "pipeline_version": PIPELINE_VERSION,
                    "frontend": frontend.semantic_inputs,
                    "target": options.spec,
                }
            ).encode()
        ).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            if not cached.dependency_hashes:
                cached.dependency_hashes = self._dependency_hashes(frontend)
            if self._dependencies_current(cached):
                self._load_native(cached, state)
                return cached
            del self._cache[key]
        if target is None:
            raise RuntimeError(f"unsupported kernel architecture {state._architecture.name!r}")
        native_compiler = state._native.Compiler()
        planned = native_compiler.plan_kernel_result(frontend.mlir)
        if not planned.ok:
            raise RuntimeError(planned.diagnostics)
        plan_reflection = parse_reflection_json(planned.reflection)
        requests = plan_reflection.get("kernel_compile_requests")
        execution = plan_reflection.get("program_plan")
        values = execution.get("values") if isinstance(execution, Mapping) else None
        if not isinstance(requests, list) or len(requests) != 1 or not isinstance(requests[0], Mapping):
            raise RuntimeError("direct kernel planning must produce exactly one compile request")
        if not isinstance(values, list):
            raise RuntimeError("direct kernel planning produced no executable values")
        request = requests[0]
        request_id = request.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError("direct kernel planning produced an invalid compile request id")
        from ..program_frontend.providers import DirectKernelDslProvider

        implementation = DirectKernelDslProvider(frontend.mlir, self._entry).lower(
            request,
            {value["id"]: value for value in values if isinstance(value, Mapping) and isinstance(value.get("id"), int)},
        )
        if implementation is None:
            raise RuntimeError("direct kernel provider did not match the C++ compile request")
        program = native_compiler.compile_program_result(implementation.mlir, target, **options.native_options)
        if not program.ok:
            raise RuntimeError(program.diagnostics)
        from .._shader_assets.cooking import _canonical_kernel_deployment, _direct_compiled_stage

        canonical_directory = tempfile.TemporaryDirectory(prefix="vernon-kernel-")
        concrete_shapes = dict(self._annotation_shapes(frontend))
        canonical_stage = _direct_compiled_stage(program, target=options, entry=implementation.entry)
        finalized = native_compiler.finalize_program_result(
            planned.reflection,
            [
                (
                    request_id,
                    canonical_stage.id,
                    canonical_stage.entry,
                    canonical_json(dict(canonical_stage.reflection)),
                )
            ],
            [(request_id, name, list(shape)) for name, shape in sorted(concrete_shapes.items())],
        )
        if not finalized.ok:
            raise RuntimeError(finalized.diagnostics)
        canonical_program, artifact_system, stage_bindings, canonical_stage = _canonical_kernel_deployment(
            canonical_stage,
            parse_reflection_json(finalized.reflection),
            target=options,
            request_id=request_id,
            output=Path(canonical_directory.name),
        )
        cached = _CompiledKernel(
            frontend.mlir,
            lowered.function,
            frontend,
            lowered.builtins,
            program,
            canonical_directory,
            json.dumps(canonical_program, sort_keys=True, separators=(",", ":")).encode(),
            json.dumps(artifact_system, sort_keys=True, separators=(",", ":")).encode(),
            dict(stage_bindings),
            canonical_stage,
            self._dependency_hashes(frontend),
        )
        self._cache[key] = cached
        self.compile_count += 1
        self._load_native(cached, state)
        return cached

    def _bind_direct_arguments(
        self,
        compiled: _CompiledKernel,
        builder: Any,
        arguments: tuple[Any, ...],
        user_parameters: list[str],
        binding_cache: _PersistentBindingTable,
    ) -> None:
        static_tensor_names = {
            argument.arg
            for argument in compiled.function.args.args
            if self._is_static_tensor_annotation(argument.annotation)
        }
        annotations = inspect.get_annotations(self._function, eval_str=True)
        for user_name, parameter, value in zip(user_parameters, compiled.native.parameters, arguments, strict=True):
            binding_cache.bind_argument(
                builder,
                compiled.native,
                parameter,
                value,
                host_value=user_name in static_tensor_names,
                annotation=annotations.get(user_name),
            )

    @staticmethod
    def _direct_parameter_accesses(
        compiled: _CompiledKernel,
        user_parameters: list[str],
        state: Any,
    ) -> dict[str, str]:
        if compiled.native is None:
            raise RuntimeError("kernel native program is not loaded")
        native_parameters = list(compiled.native.parameters)
        if len(native_parameters) != len(user_parameters):
            raise RuntimeError("finalized kernel endpoint ABI does not match its Python arguments")
        return {
            name: (
                "read"
                if parameter.access == state._native.ACCESS_READ
                else "write"
                if parameter.access == state._native.ACCESS_WRITE
                else "read_write"
            )
            for name, parameter in zip(user_parameters, native_parameters, strict=True)
        }

    def _invoke_direct(
        self,
        arguments: tuple[Any, ...],
        grid: tuple[int, int, int] | None,
        features: tuple[str, ...] = (),
        binding_cache: _PersistentBindingTable | None = None,
        return_submission: bool = False,
    ) -> tuple[Any, _DispatchBorrowLease] | None:
        state = _session_state()
        lowered = self._lower(features)
        self._bind_launch(lowered, arguments)
        compiled = self.specialize(features, lowered=lowered)
        user_parameters = [
            argument.arg for argument in compiled.function.args.args if argument.arg not in compiled.builtin_names
        ]
        parameter_accesses = self._direct_parameter_accesses(compiled, user_parameters, state)
        if grid is None:
            writable_shapes = {
                value.shape
                for name, value in zip(user_parameters, arguments, strict=True)
                if parameter_accesses[name] != "read"
                and isinstance(value, (TensorStorage, TensorView, _TextureResource))
            }
            if not writable_shapes:
                raise TypeError("grid is required when no writable Tensor domain can be inferred")
            if len(writable_shapes) != 1:
                raise ValueError("all writable Tensor arguments must have the same shape")
            shape = next(iter(writable_shapes))
            if len(shape) > 3:
                raise ValueError("inferred compute grids require Tensor rank zero through three")
            extent = tuple(reversed(shape)) + (1,) * (3 - len(shape))
            grid = tuple((value + size - 1) // size for value, size in zip(extent, self._workgroup_size, strict=True))
        if len(grid) != 3 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid):
            raise ValueError("grid must contain three positive integers")
        if state._native_runtime is None:
            raise RuntimeError(f"{state._architecture.name} kernel execution requires the native runtime")
        if compiled.native is None:
            raise RuntimeError("kernel native program is not loaded")
        if binding_cache is None:
            binding_cache = _PersistentBindingTable()
        dispatch_borrows = [
            (name, value, parameter_accesses[name])
            for name, value in zip(user_parameters, arguments, strict=True)
            if isinstance(value, (TensorStorage, TensorView, _TextureResource))
        ]
        lease = _DispatchBorrowLease(dispatch_borrows) if return_submission else None
        try:
            with binding_cache.invocation(compiled.native) as builder:
                self._bind_direct_arguments(compiled, builder, arguments, user_parameters, binding_cache)
                # The loaded single-node pipeline exposes the canonical Program's
                # groups_x/y/z control arguments through this direct-dispatch facade.
                builder.grid(*grid)
                submission = builder.submit()
                if not return_submission:
                    submission.wait()
        except Exception:
            if lease is not None:
                lease.release()
            raise
        for name, value in zip(user_parameters, arguments, strict=True):
            if (
                state._architecture != state.cpu
                and parameter_accesses[name] != "read"
                and isinstance(value, (TensorStorage, TensorView, _TextureResource))
            ):
                value._mark_device_dirty()
        if return_submission:
            assert lease is not None
            return submission, lease
        return None

    def __call__(
        self,
        *arguments: Any,
        grid: tuple[int, int, int] | None = None,
        features: tuple[str, ...] = (),
    ) -> None:
        self._submit_direct(arguments, grid, features)

    def _submit_direct(
        self,
        arguments: tuple[Any, ...],
        grid: tuple[int, int, int] | None,
        features: tuple[str, ...],
    ) -> None:
        cache_key = (
            features,
            tuple(self._argument_signature(value) for value in arguments),
        )
        binding_cache = self._direct_binding_caches.setdefault(cache_key, _PersistentBindingTable())
        result = self._invoke_direct(
            arguments,
            grid,
            features,
            binding_cache=binding_cache,
            return_submission=True,
        )
        if result is None:
            raise RuntimeError("direct kernel invocation did not produce a submission")
        native_submission, lease = result
        try:
            native_submission.wait()
        finally:
            lease.release()


atexit.register(Kernel.clear_cache)

__all__ = ["Kernel"]
