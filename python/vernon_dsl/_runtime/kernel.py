from __future__ import annotations

import ast
import atexit
import hashlib
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..bundle import TargetOptions, canonical_json
from ..compiler import Compiler, FrontendCompileRequest, FrontendCompileResult
from ..frontend.model import AccessMode, StorageEffect, StorageEffectKind
from ..types import TypeExpr, _Scalar
from .execution_graph import (
    ComputeEncoder,
    ComputePass,
    ExecutionGraph,
    ExecutionResources,
    PipelineInvocation,
)
from .resources import TensorStorage, TensorView, _bind_native_argument, _dispatch_borrow_scope


def _session_state() -> Any:
    return importlib.import_module("vernon_dsl._runtime.session")


class _ImmediateComputePass(ComputePass):
    def __init__(self, name: str, invocation: PipelineInvocation):
        super().__init__(name)
        self._invocation = invocation

    def declare(self) -> None:
        self._invocation.declare(self)
        self.side_effect = True

    def execute(self, encoder: ComputeEncoder, resources: ExecutionResources) -> None:
        self._invocation.encode(encoder, resources)


@dataclass
class _CompiledKernel:
    mlir: str
    function: ast.FunctionDef
    builtin_names: tuple[str, ...]
    writable_names: tuple[str, ...]
    program: Any
    dependency_hashes: tuple[tuple[Path, str], ...] = ()
    native: Any | None = None
    native_generation: int = -1


class Kernel:
    _cache: ClassVar[dict[str, _CompiledKernel]] = {}
    _dispatch_cache: ClassVar[dict[tuple[Any, ...], _CompiledKernel]] = {}

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
        self.compile_count = 0

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        cls._dispatch_cache.clear()

    @classmethod
    def invalidate_loaded(cls) -> None:
        for compiled in cls._cache.values():
            compiled.native = None
            compiled.native_generation = -1

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
    def _dependencies_current(compiled: _CompiledKernel) -> bool:
        try:
            return all(
                hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest() == digest
                for path, digest in compiled.dependency_hashes
            )
        except OSError:
            return False

    @staticmethod
    def _argument_signature(value: Any) -> tuple[Any, ...]:
        if isinstance(value, TensorStorage):
            return ("storage", value.dtype.str, value.shape)
        if isinstance(value, TensorView):
            layout = value.layout
            return (
                "view",
                value.dtype.str,
                value.shape,
                layout.element_strides,
                layout.element_offset,
                value.access,
            )
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
    def _load_native(compiled: _CompiledKernel, state: Any, entry: str) -> None:
        if compiled.native is not None and compiled.native_generation == state._runtime_generation:
            return
        if state._architecture == state.cpu:
            compiled.native = state._native_runtime.load_cpu_entry(compiled.program, entry)
        else:
            if len(compiled.program.artifacts) != 1:
                raise RuntimeError("kernel compilation must produce exactly one artifact")
            compiled.native = state._native_runtime.load(
                compiled.program.artifacts[0][1],
                compiled.program.reflection,
                entry,
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
                and node.args[0].value == "global_invocation_id"
                for node in ast.walk(argument.annotation)
            )
        )

    @staticmethod
    def _writable_parameters(frontend: FrontendCompileResult) -> tuple[str, ...]:
        entry = next(
            (function for function in frontend.typed_functions if function.symbol == frontend.request.entry),
            None,
        )
        if entry is None:
            raise RuntimeError("compiled kernel has no typed entry function")
        parameter_names = {parameter.name for parameter in entry.parameters}
        writable = {
            parameter.name
            for parameter in entry.parameters
            if parameter.type.kind == "tensor_view" and parameter.access is not AccessMode.READ
        }
        writable.update(
            effect.owner.name
            for effect in entry.effects
            if isinstance(effect, StorageEffect)
            and effect.kind is StorageEffectKind.WRITE
            and effect.owner.name in parameter_names
        )
        return tuple(sorted(writable))

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

    def _validate_tensor_view_arguments(
        self,
        frontend: FrontendCompileResult,
        user_parameters: list[str],
        arguments: tuple[Any, ...],
    ) -> None:
        entry = next(
            (function for function in frontend.typed_functions if function.source.name == self._entry),
            None,
        )
        if entry is None:
            raise RuntimeError("compiled kernel has no typed entry function")
        typed_parameters = {parameter.name: parameter for parameter in entry.parameters}
        scalar_dtypes = {
            "bool": np.dtype(np.bool_),
            "i32": np.dtype(np.int32),
            "u32": np.dtype(np.uint32),
            "f16": np.dtype(np.float16),
            "f32": np.dtype(np.float32),
            "f64": np.dtype(np.float64),
        }
        access_compatibility = {
            "read": {"read", "read_write"},
            "write": {"write", "read_write"},
            "read_write": {"read_write"},
        }

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
                    if runtime_signature(element_type) != dsl_signature(element):
                        raise TypeError(f"kernel Tensor argument {name!r} element type does not match {element.name}")
                    continue
                if isinstance(value, (TensorStorage, TensorView)):
                    raise TypeError(f"kernel argument {name!r} is runtime storage but its annotation is not TensorView")
                continue
            if not isinstance(value, TensorView):
                raise TypeError(f"kernel argument {name!r} must be a TensorView")
            element, rank, declared_access = parameter.type.arguments
            if len(value.shape) != rank:
                raise TypeError(f"kernel TensorView argument {name!r} has rank {len(value.shape)}, expected {rank}")
            if element.kind == "scalar":
                matches_element = value.dtype == scalar_dtypes[element.name]
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

    def _lower(
        self,
        arguments: tuple[Any, ...],
        features: tuple[str, ...] = (),
    ) -> tuple[
        FrontendCompileResult,
        ast.FunctionDef,
        tuple[str, ...],
        dict[str, TensorStorage | TensorView],
    ]:
        source = self._file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(self._file))
        function = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == self._entry),
            None,
        )
        if function is None:
            raise RuntimeError("kernel functions must be top-level definitions in files")
        builtins = self._builtin_parameters(function)
        user_parameters = [argument.arg for argument in function.args.args if argument.arg not in builtins]
        if len(arguments) != len(user_parameters):
            raise TypeError(f"{self._entry} expects {len(user_parameters)} launch arguments")
        declared_access = {
            argument.arg: self._tensor_view_access(argument.annotation)
            for argument in function.args.args
            if argument.arg in user_parameters
        }
        normalized_arguments = tuple(
            value._full_view(declared_access[name])
            if isinstance(value, TensorStorage) and declared_access[name] is not None
            else value
            for name, value in zip(user_parameters, arguments, strict=True)
        )
        tensors = {
            name: value
            for name, value in zip(user_parameters, normalized_arguments, strict=True)
            if isinstance(value, (TensorStorage, TensorView))
        }
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
            tuple(
                (name, value.dtype.str, value.shape)
                for name, value in tensors.items()
                if isinstance(value, TensorStorage)
            ),
            tuple(
                (
                    name,
                    value.dtype.str,
                    value.shape,
                    value.layout.element_strides,
                    value.layout.element_offset,
                )
                for name, value in tensors.items()
                if isinstance(value, TensorView)
            ),
            constants,
            self._workgroup_size,
        )
        frontend = Compiler().compile_request(request)
        self._validate_tensor_view_arguments(frontend, user_parameters, normalized_arguments)
        specialized_tree = ast.parse(frontend.specialized_source, filename=str(self._file))
        specialized_function = next(
            node for node in specialized_tree.body if isinstance(node, ast.FunctionDef) and node.name == self._entry
        )
        return frontend, specialized_function, builtins, tensors

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
        frontend, _, _, _ = self._lower(arguments)
        options = TargetOptions(
            target,
            {"glsl_version": 430} if target == "opengl" else {"glsl_version": 310} if target == "opengles" else {},
        )
        program = state._native.Compiler().compile_program_result(
            frontend.mlir, targets[target], **options.native_options
        )
        if not program.ok:
            raise RuntimeError(program.diagnostics)
        if len(program.artifacts) != 1:
            raise RuntimeError("kernel compilation must produce exactly one artifact")
        return bytes(program.artifacts[0][1]), str(program.reflection)

    def _compile(self, arguments: tuple[Any, ...], features: tuple[str, ...] = ()) -> _CompiledKernel:
        state = _session_state()
        if state._native is None or state._native_runtime is None:
            raise RuntimeError(f"{state._architecture.name} kernel execution requires the native runtime")
        target = (
            {
                state.cpu: state._native.Target.CPU,
                state.cuda: state._native.Target.CUDA,
                state.vulkan: state._native.Target.VULKAN,
                state.directx: state._native.Target.DIRECTX,
                state.opengl: state._native.Target.OPENGL,
                state.opengles: state._native.Target.OPENGL_ES,
            }.get(state._architecture)
            if state._native is not None
            else None
        )
        options = TargetOptions(
            state._architecture.name,
            {"glsl_version": state._interactive_glsl_version()}
            if state._architecture in {state.opengl, state.opengles}
            else {},
        )
        dispatch_key = self._dispatch_key(arguments, features, options.target, tuple(sorted(options.options.items())))
        cached = self._dispatch_cache.get(dispatch_key)
        if cached is not None and self._dependencies_current(cached):
            self._load_native(cached, state, self._entry)
            return cached

        frontend, function, builtins, _ = self._lower(arguments, features)
        key = hashlib.sha256(
            canonical_json(
                {
                    "version": 3,
                    "frontend": frontend.semantic_inputs,
                    "target": options.target,
                    "target_options": dict(options.options),
                }
            ).encode()
        ).hexdigest()
        cached = self._cache.get(key)
        if cached is None:
            assert target is not None
            program = state._native.Compiler().compile_program_result(frontend.mlir, target, **options.native_options)
            if not program.ok:
                raise RuntimeError(program.diagnostics)
            cached = _CompiledKernel(
                frontend.mlir,
                function,
                builtins,
                self._writable_parameters(frontend),
                program,
                self._dependency_hashes(frontend),
            )
            self._cache[key] = cached
            self.compile_count += 1
        elif not cached.dependency_hashes:
            cached.dependency_hashes = self._dependency_hashes(frontend)
        self._dispatch_cache[dispatch_key] = cached
        self._load_native(cached, state, self._entry)
        return cached

    def _invoke_direct(
        self,
        arguments: tuple[Any, ...],
        grid: tuple[int, int, int] | None,
        features: tuple[str, ...] = (),
        encoder: ComputeEncoder | None = None,
    ) -> None:
        state = _session_state()
        compiled = self._compile(arguments, features)
        user_parameters = [
            argument.arg for argument in compiled.function.args.args if argument.arg not in compiled.builtin_names
        ]
        if grid is None:
            writable_shapes = {
                value.shape
                for name, value in zip(user_parameters, arguments, strict=True)
                if name in compiled.writable_names and isinstance(value, (TensorStorage, TensorView))
            }
            if not writable_shapes:
                raise TypeError("grid is required when no writable Tensor domain can be inferred")
            if len(writable_shapes) != 1:
                raise ValueError("all writable Tensor arguments must have the same shape")
            shape = next(iter(writable_shapes))
            if not 1 <= len(shape) <= 3:
                raise ValueError("inferred compute grids require Tensor rank one through three")
            grid = tuple(reversed(shape)) + (1,) * (3 - len(shape))
        if len(grid) != 3 or any(not isinstance(value, int) or value <= 0 for value in grid):
            raise ValueError("grid must contain three positive integers")
        assert state._native_runtime is not None and compiled.native is not None
        dispatch_borrows = [
            (name, value, "write" if name in compiled.writable_names else "read")
            for name, value in zip(user_parameters, arguments, strict=True)
            if isinstance(value, (TensorStorage, TensorView))
        ]
        with _dispatch_borrow_scope(dispatch_borrows):
            builder = compiled.native.invocation_builder()
            static_tensor_names = {
                argument.arg
                for argument in compiled.function.args.args
                if self._is_static_tensor_annotation(argument.annotation)
            }
            for user_name, parameter, value in zip(user_parameters, compiled.native.parameters, arguments, strict=True):
                _bind_native_argument(
                    builder,
                    parameter,
                    value,
                    host_value=state._architecture == state.cuda and user_name in static_tensor_names,
                )
            builder.grid(*grid)
            if encoder is None:
                builder.invoke()
            else:
                builder.encode(encoder._native)
            for name, value in zip(user_parameters, arguments, strict=True):
                if (
                    state._architecture != state.cpu
                    and name in compiled.writable_names
                    and isinstance(value, (TensorStorage, TensorView))
                ):
                    value._mark_device_dirty()

    def __call__(
        self,
        *arguments: Any,
        grid: tuple[int, int, int] | None = None,
        features: tuple[str, ...] = (),
    ) -> None:
        state = _session_state()
        if state._architecture == state.cpu:
            self._invoke_direct(tuple(arguments), grid, features)
            return
        invocation = self.invocation(*arguments, grid=grid, features=features)

        graph = ExecutionGraph()
        graph.add_pass(_ImmediateComputePass(f"{self.__name__} immediate", invocation))
        try:
            graph.execute()
        finally:
            graph._dispose_native()

    def _declare_invocation(
        self,
        arguments: tuple[Any, ...],
        features: tuple[str, ...],
        execution_pass: ComputePass,
    ) -> None:
        state = _session_state()
        compiled = self._compile(arguments, features)
        for parameter, value in zip(compiled.native.parameters, arguments, strict=True):
            if not isinstance(value, (TensorStorage, TensorView)):
                continue
            if parameter.access == state._native.ACCESS_READ:
                execution_pass.read(value)
            elif parameter.access == state._native.ACCESS_WRITE:
                execution_pass.write(value)
            else:
                execution_pass.read_write(value)

    def invocation(
        self,
        *arguments: Any,
        grid: tuple[int, int, int] | None = None,
        features: tuple[str, ...] = (),
    ) -> PipelineInvocation:
        captured = tuple(arguments)
        return PipelineInvocation(
            "compute",
            lambda encoder: self._invoke_direct(captured, grid, features, encoder),
            lambda execution_pass: self._declare_invocation(captured, features, execution_pass),
        )


atexit.register(Kernel.clear_cache)

__all__ = ["Kernel"]
