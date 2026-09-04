from __future__ import annotations

import ast
import atexit
import dataclasses
import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from .._dtypes import NUMPY_DTYPE_BY_SCALAR
from .._versions import COMPILER_CONTRACT_VERSION, PIPELINE_VERSION
from ..bundle import PipelineCompileError, canonical_json, make_target_options
from ..compiler import Compiler, FrontendCompileRequest, FrontendCompileResult
from ..frontend.model import ConcreteType
from ..host_values import pack_host_value
from ..types import TypeExpr, _Scalar
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
    frontend: FrontendCompileResult
    builtin_names: tuple[str, ...]
    invocation: Any
    specialization: Any
    dependency_hashes: tuple[tuple[Path, str], ...] = ()


class _DependencyTracked(Protocol):
    dependency_hashes: tuple[tuple[Path, str], ...]


class Kernel:
    """Device kernel: lower source to MLIR, specialize MLIR to a native artifact, bind tensors at launch."""

    _cache: ClassVar[dict[str, _CompiledKernel]] = {}

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
        from .autodiff import clear_vjp_cache

        clear_vjp_cache()

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

    def _specialization_key(
        self,
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
            constants,
        )

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
        normalized: list[Any] = []
        for name, value in zip(user_parameters, arguments, strict=True):
            access = declared_access[name]
            normalized.append(value._full_view(access) if isinstance(value, TensorStorage) and access else value)
        return user_parameters, tuple(normalized)

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
            if not isinstance(declared_access, str):
                raise RuntimeError(f"kernel TensorView argument {name!r} has unresolved access")
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
        lowered = lowered or self._lower(features)
        frontend = lowered.frontend
        state, target, options = self._session_target()
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
                return cached
            del self._cache[key]
        if target is None:
            raise RuntimeError(f"unsupported kernel architecture {state._architecture.name!r}")
        entry = next(
            (function for function in frontend.typed_functions if function.source.name == self._entry),
            None,
        )
        if entry is None or entry.storage_activity is None:
            raise RuntimeError("compiled kernel has no typed storage activity")
        annotations = inspect.get_annotations(self._function, eval_str=True)
        parameter_names = tuple(
            name for name in inspect.signature(self._function).parameters if name not in lowered.builtins
        )
        from ..frontend.runtime_types import runtime_parameter_descriptor
        from ..program import _one_node_program

        parameter_types: dict[str, Any] = {}
        for name in parameter_names:
            annotation = annotations.get(name)
            if annotation is None:
                raise TypeError(f"kernel argument {name!r} requires a runtime annotation")
            parameter_types[name] = runtime_parameter_descriptor(annotation)
        from ..types import u32

        for axis in "xyz":
            parameter_types[f"__grid_{axis}"] = runtime_parameter_descriptor(u32)

        template, invocation, parsed = _one_node_program(
            self,
            parameter_types,
            lambda capture, inputs: capture.capture_kernel(
                self,
                tuple(inputs[name] for name in parameter_names),
                tuple(inputs[f"__grid_{axis}"] for axis in "xyz"),
                features,
                lowered=lowered,
            ),
            None,
        )
        from .program_autodiff import compile_program

        try:
            specialization = compile_program(parsed, template)
        except PipelineCompileError as error:
            raise RuntimeError(str(error)) from error
        cached = _CompiledKernel(
            frontend,
            lowered.builtins,
            invocation,
            specialization,
            self._dependency_hashes(frontend),
        )
        self._cache[key] = cached
        self.compile_count += 1
        return cached

    def _invoke(
        self,
        arguments: tuple[Any, ...],
        grid: tuple[int, int, int] | None,
        features: tuple[str, ...] = (),
    ) -> None:
        lowered = self._lower(features)
        user_parameters, normalized = self._normalize_arguments(lowered.source, lowered.builtins, arguments)
        self._validate_tensor_view_arguments(lowered.frontend, user_parameters, normalized)
        entry = next(function for function in lowered.frontend.typed_functions if function.source.name == self._entry)
        writable = entry.storage_activity.writable_roots if entry.storage_activity is not None else frozenset()
        if grid is None:
            writable_shapes = {
                value.shape
                for name, value in zip(user_parameters, normalized, strict=True)
                if name in writable and isinstance(value, (TensorStorage, TensorView, _TextureResource))
            }
            if not writable_shapes:
                raise TypeError("grid is required when no writable Tensor domain can be inferred")
            if len(writable_shapes) != 1:
                raise ValueError("all writable Tensor arguments must have the same shape")
            shape = next(iter(writable_shapes))
            if len(shape) > 3:
                raise ValueError("inferred compute grids require Tensor rank zero through three")
            extent = tuple(reversed(shape)) + (1,) * (3 - len(shape))
            grid = (
                (extent[0] + self._workgroup_size[0] - 1) // self._workgroup_size[0],
                (extent[1] + self._workgroup_size[1] - 1) // self._workgroup_size[1],
                (extent[2] + self._workgroup_size[2] - 1) // self._workgroup_size[2],
            )
        if len(grid) != 3 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid):
            raise ValueError("grid must contain three positive integers")
        compiled = self.specialize(features, lowered=lowered)
        actual = dict(zip(user_parameters, normalized, strict=True))
        outputs = {name: actual[name] for name in compiled.invocation.outputs}
        actual.update({f"__grid_{axis}": value for axis, value in zip("xyz", grid, strict=True)})
        compiled.specialization.invoke(dataclasses.replace(compiled.invocation, inputs=actual, outputs=outputs))

    def __call__(
        self,
        *arguments: Any,
        grid: tuple[int, int, int] | None = None,
        features: tuple[str, ...] = (),
    ) -> None:
        self._invoke(arguments, grid, features)


atexit.register(Kernel.clear_cache)

__all__ = ["Kernel"]
