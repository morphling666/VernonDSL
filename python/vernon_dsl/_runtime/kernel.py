from __future__ import annotations

import ast
import atexit
import hashlib
import importlib
import inspect
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..compiler import Compiler, FrontendCompileRequest, FrontendCompileResult
from ..pipeline_compile import TargetOptions, canonical_json
from .resources import Tensor


def _session_state() -> Any:
    return importlib.import_module("vernon_dsl._runtime.session")


@dataclass
class _CompiledKernel:
    mlir: str
    function: ast.FunctionDef
    builtin_names: tuple[str, ...]
    writable_names: tuple[str, ...]
    program: Any
    native: Any | None = None
    native_generation: int = -1


class Kernel:
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

    @classmethod
    def invalidate_loaded(cls) -> None:
        for compiled in cls._cache.values():
            compiled.native = None
            compiled.native_generation = -1

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
    def _writable_parameters(function: ast.FunctionDef) -> tuple[str, ...]:
        parameters = {argument.arg for argument in function.args.args}
        writable: set[str] = set()
        for node in ast.walk(function):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                while isinstance(target, ast.Subscript):
                    target = target.value
                if isinstance(target, ast.Name) and target.id in parameters:
                    writable.add(target.id)
        return tuple(sorted(writable))

    def _lower(
        self,
        arguments: tuple[Any, ...],
        features: tuple[str, ...] = (),
    ) -> tuple[FrontendCompileResult, ast.FunctionDef, tuple[str, ...], dict[str, Tensor]]:
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
        tensors = {
            name: value for name, value in zip(user_parameters, arguments, strict=True) if isinstance(value, Tensor)
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
            tuple((name, value.dtype.str, value.shape) for name, value in tensors.items()),
            constants,
            self._workgroup_size,
        )
        frontend = Compiler().compile_request(request)
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
            "metal": state._native.Target.METAL,
            "opengl": state._native.Target.OPENGL,
            "opengles": state._native.Target.OPENGL_ES,
        }
        if target not in targets:
            raise ValueError("target must be cpu, cuda, vulkan, metal, opengl, or opengles")
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
        frontend, function, builtins, _ = self._lower(arguments, features)
        target = (
            {
                state.cpu: state._native.Target.CPU,
                state.cuda: state._native.Target.CUDA,
                state.vulkan: state._native.Target.VULKAN,
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
        if state._native is None or state._native_runtime is None:
            raise RuntimeError(f"{state._architecture.name} kernel execution requires the native runtime")
        if cached is None:
            assert target is not None
            program = state._native.Compiler().compile_program_result(frontend.mlir, target, **options.native_options)
            if not program.ok:
                raise RuntimeError(program.diagnostics)
            cached = _CompiledKernel(
                frontend.mlir,
                function,
                builtins,
                self._writable_parameters(function),
                program,
            )
            self._cache[key] = cached
            self.compile_count += 1
        if cached.native is None or cached.native_generation != state._runtime_generation:
            if state._architecture == state.cpu:
                cached.native = state._native_runtime.load_cpu_entry(cached.program, self._entry)
            else:
                if len(cached.program.artifacts) != 1:
                    raise RuntimeError("kernel compilation must produce exactly one artifact")
                cached.native = state._native_runtime.load(
                    cached.program.artifacts[0][1],
                    cached.program.reflection,
                    self._entry,
                )
            cached.native_generation = state._runtime_generation
        return cached

    def __call__(
        self,
        *arguments: Any,
        grid: tuple[int, int, int] | None = None,
        _pipeline_features: tuple[str, ...] = (),
    ) -> None:
        state = _session_state()
        compiled = self._compile(arguments, _pipeline_features)
        user_parameters = [
            argument.arg for argument in compiled.function.args.args if argument.arg not in compiled.builtin_names
        ]
        if grid is None:
            writable_shapes = {
                value.shape
                for name, value in zip(user_parameters, arguments, strict=True)
                if name in compiled.writable_names and isinstance(value, Tensor)
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
        native_values: list[Any] = []
        parameters = (
            argument for argument in compiled.function.args.args if argument.arg not in compiled.builtin_names
        )
        for parameter, value in zip(parameters, arguments, strict=True):
            if isinstance(value, Tensor):
                native_values.append(value._resident_buffer())
            else:
                annotation = self._annotation_name(parameter.annotation)
                if annotation == "u32":
                    native_values.append(struct.pack("<I", int(value)))
                elif annotation == "i32":
                    native_values.append(struct.pack("<i", int(value)))
                elif annotation == "f64":
                    native_values.append(struct.pack("<d", float(value)))
                else:
                    native_values.append(struct.pack("<f", float(value)))
        compiled.native.launch(*grid, native_values)
        state._native_runtime.synchronize()
        for name, value in zip(user_parameters, arguments, strict=True):
            if name in compiled.writable_names and isinstance(value, Tensor):
                value._mark_device_dirty()


atexit.register(Kernel.clear_cache)

__all__ = ["Kernel"]
