from __future__ import annotations

import ast
import atexit
import hashlib
import importlib
import inspect
import os
import struct
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, get_args, get_origin, get_type_hints

import numpy as np

from .compiler import (
    Compiler,
    FrontendCompileRequest,
    FrontendCompileResult,
)
from .pipeline_compile import (
    CompiledStage,
    PipelineCompileError,
    TargetOptions,
    build_bundle_plan,
    canonical_json,
    compiled_stage_from_program,
    inline_artifact_descriptor,
    materialize_bundle,
    serialize_bundle,
)
from .types import Annotation, TypeExpr, _Scalar

_native_dll_directories: list[Any] = []


def _load_native() -> Any | None:
    try:
        from . import _native as packaged_native

        return packaged_native
    except ImportError:
        pass

    try:
        return importlib.import_module("_native")
    except ImportError:
        pass

    # CMake places development builds beside the compiler/runtime libraries,
    # rather than inside the source package. Discover only this checkout's
    # conventional build directories so examples work without PYTHONPATH.
    repository = Path(__file__).resolve().parents[2]
    source_build = repository / "build" / "source"
    candidates = (
        source_build / "Release",
        source_build / "Debug",
        source_build / "RelWithDebInfo",
        source_build / "MinSizeRel",
        source_build,
    )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if os.name == "nt":
            _native_dll_directories.append(os.add_dll_directory(
                str(candidate)))
        sys.path.insert(0, str(candidate))
        try:
            return importlib.import_module("_native")
        except ImportError:
            sys.path.pop(0)
    return None


try:
    _native = _load_native()
except (ImportError, OSError):
    _native = None


@dataclass(frozen=True)
class _Architecture:
    name: str


cpu = _Architecture("cpu")
cuda = _Architecture("cuda")
vulkan = _Architecture("vulkan")
opengl = _Architecture("opengl")
opengles = _Architecture("opengles")
_architecture = cpu
_native_runtime: Any | None = None
_runtime_generation = 0
_api_version: tuple[int, int] | None = None
_external_opengl_contexts: dict[_Architecture, tuple[int, int, int,
                                                     tuple[int, int]]] = {}


@dataclass(frozen=True)
class PrimitiveTopology:
    name: str
    vertices_per_primitive: int


triangles = PrimitiveTopology("triangles", 3)
lines = PrimitiveTopology("lines", 2)
points = PrimitiveTopology("points", 1)


def init(*,
         arch: _Architecture = cpu,
         api_version: tuple[int, int] | None = None) -> None:
    global _architecture, _native_runtime, _runtime_generation, _api_version
    if arch not in {cpu, cuda, vulkan, opengl, opengles}:
        raise ValueError("unsupported VernonDSL runtime architecture")
    if api_version is not None and (arch not in {opengl, opengles}
                                    or len(api_version) != 2 or any(
                                        not isinstance(value, int) or value < 0
                                        for value in api_version)):
        raise ValueError(
            "api_version is a (major, minor) pair for OpenGL runtimes")
    kernel_type = globals().get("Kernel")
    if kernel_type is not None:
        kernel_type.invalidate_loaded()
    if arch in {cpu, cuda, vulkan, opengl, opengles}:
        if _native is None:
            raise RuntimeError(
                f"{arch.name} requires vernon_dsl._native; build the Release native "
                "targets or install a wheel containing the native module")
        backend = {
            cpu: _native.RuntimeBackend.CPU,
            cuda: _native.RuntimeBackend.CUDA,
            vulkan: _native.RuntimeBackend.VULKAN,
            opengl: _native.RuntimeBackend.OPENGL,
            opengles: _native.RuntimeBackend.OPENGL_ES,
        }[arch]
        if arch in {opengl, opengles}:
            external = _external_opengl_contexts.get(arch)
            if external is None:
                raise RuntimeError(
                    f"{arch.name} requires a registered host-owned external context"
                )
            user_data, make_current, get_proc_address, registered_version = external
            requested = api_version or registered_version
            _native_runtime = _native.Runtime.create_external_opengl(
                backend, user_data, make_current, get_proc_address, *requested)
        elif not _native.runtime_available(backend):
            raise RuntimeError(
                f"{arch.name} loader or a usable device is unavailable")
        else:
            requested = api_version or (4, 3)
            _native_runtime = _native.Runtime(backend, *requested)
    _architecture = arch
    _api_version = api_version or ((4,
                                    3) if arch in {opengl, opengles} else None)
    _runtime_generation += 1


def register_external_opengl_context(
    *,
    arch: _Architecture,
    user_data: int,
    make_current: int,
    get_proc_address: int,
    api_version: tuple[int, int],
) -> None:
    if arch not in {opengl, opengles}:
        raise ValueError(
            "external contexts are only valid for OpenGL backends")
    if (not all(
            isinstance(value, int) and value >= 0
            for value in (user_data, make_current, get_proc_address))
            or not make_current or not get_proc_address):
        raise ValueError(
            "external context callbacks must be non-zero addresses")
    if (len(api_version) != 2 or any(not isinstance(value, int) or value < 0
                                     for value in api_version)):
        raise ValueError("api_version must be a non-negative major/minor pair")
    _external_opengl_contexts[arch] = (user_data, make_current,
                                       get_proc_address, api_version)


def _interactive_glsl_version() -> int:
    if _architecture not in {opengl, opengles}:
        return 0
    major, minor = _api_version or ((4, 3) if _architecture == opengl else
                                    (3, 1))
    return major * 100 + minor * 10


class CpuAotRuntime:
    """Low-level deployable CPU AOT bundle runtime."""

    def __init__(self) -> None:
        if _native is None:
            raise RuntimeError("native runtime module is unavailable")
        self._native = _native.Runtime(_native.RuntimeBackend.CPU)

    def allocate(self, size: int, alignment: int = 16) -> Any:
        return self._native.allocate(size, alignment)

    def load_bundle(self, directory: str | Path) -> Any:
        return self._native.load_compute_bundle(str(Path(directory).resolve()))


_NUMPY_DTYPES = {
    "bool": np.dtype(np.bool_),
    "i32": np.dtype(np.int32),
    "u32": np.dtype(np.uint32),
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple[int, ...]
    byte_strides: tuple[int, ...]
    byte_offset: int = 0
    components: tuple[int, ...] | None = None


class Tensor:
    """Contiguous row-major runtime Tensor and annotation constructor."""

    def __init__(self, array: np.ndarray):
        if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
            raise ValueError("Tensor storage must be a contiguous NumPy array")
        self._array = array
        self._native_buffer: Any | None = None
        self._native_generation = -1
        self._host_version = 1
        self._uploaded_version = 0
        self._device_dirty = False
        self._allocation_count = 0
        self._upload_count = 0
        self._download_count = 0

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments, )
        return TypeExpr("Tensor", arguments)

    @staticmethod
    def _dtype(dtype: _Scalar) -> np.dtype[Any]:
        if not isinstance(dtype, _Scalar) or dtype.name not in _NUMPY_DTYPES:
            raise TypeError("dtype must be a Vernon scalar type")
        return _NUMPY_DTYPES[dtype.name]

    @classmethod
    def zeros(cls, *, dtype: _Scalar, shape: tuple[int, ...]) -> "Tensor":
        return cls(np.zeros(shape, dtype=cls._dtype(dtype), order="C"))

    @classmethod
    def empty(cls, *, dtype: _Scalar, shape: tuple[int, ...]) -> "Tensor":
        return cls(np.empty(shape, dtype=cls._dtype(dtype), order="C"))

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "Tensor":
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a NumPy ndarray")
        if array.dtype not in _NUMPY_DTYPES.values():
            raise TypeError(f"unsupported Tensor dtype {array.dtype}")
        return cls(np.array(array, copy=True, order="C"))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._array.dtype

    @property
    def layout(self) -> TensorLayout:
        return TensorLayout(self.shape, self._array.strides)

    def swizzle(self, components: str) -> "TensorView":
        if len(self.shape) < 2:
            raise ValueError(
                "Tensor swizzle requires a trailing component axis")
        spelling = "xyzw"
        aliases = "rgba"
        indices = tuple(
            spelling.find(value) if value in spelling else aliases.find(value)
            for value in components)
        if (not indices or any(index < 0 or index >= self.shape[-1]
                               for index in indices)):
            raise ValueError("Tensor swizzle is outside the component axis")
        start = indices[0]
        if indices != tuple(range(start, start + len(indices))):
            raise ValueError(
                "runtime Tensor swizzles must select contiguous components in storage order"
            )
        return TensorView(self, start, len(indices))

    def to_numpy(self) -> np.ndarray:
        self.synchronize()
        return self._array.copy(order="C")

    def _borrowed_array(self) -> np.ndarray:
        self.synchronize()
        return self._array

    def copy_from_numpy(self, array: np.ndarray) -> None:
        if (not isinstance(array, np.ndarray) or array.dtype != self.dtype
                or array.shape != self.shape or not array.flags.c_contiguous):
            raise ValueError(
                "upload requires matching dtype, shape, and contiguity")
        np.copyto(self._array, array)
        self._host_version += 1
        self._device_dirty = False

    def synchronize(self) -> None:
        if not self._device_dirty or self._native_buffer is None:
            return
        downloaded = np.frombuffer(self._native_buffer.download(),
                                   dtype=self.dtype).reshape(self.shape)
        np.copyto(self._array, downloaded)
        self._download_count += 1
        self._device_dirty = False
        self._host_version += 1
        self._uploaded_version = self._host_version

    def _resident_buffer(self) -> Any:
        if _native_runtime is None:
            raise RuntimeError("native Tensor residency requires a runtime")
        if (self._native_buffer is None
                or self._native_generation != _runtime_generation):
            self._native_buffer = _native_runtime.allocate(
                self._array.nbytes, self.dtype.itemsize)
            self._native_generation = _runtime_generation
            self._uploaded_version = 0
            self._device_dirty = False
            self._allocation_count += 1
        if self._uploaded_version != self._host_version:
            self._native_buffer.upload(self._array.tobytes(order="C"))
            self._uploaded_version = self._host_version
            self._upload_count += 1
        return self._native_buffer

    def _mark_device_dirty(self) -> None:
        self._device_dirty = True


class TensorView:
    """A zero-copy contiguous component selection from a resident Tensor."""

    def __init__(self, owner: Tensor, first_component: int,
                 component_count: int):
        self._owner = owner
        self._first_component = first_component
        self._component_count = component_count

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._owner.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return (*self._owner.shape[:-1], self._component_count)

    @property
    def layout(self) -> TensorLayout:
        offset = self._first_component * self.dtype.itemsize
        return TensorLayout(
            self.shape, self._owner._array.strides, offset,
            tuple(
                range(self._first_component,
                      self._first_component + self._component_count)))

    def to_numpy(self) -> np.ndarray:
        self._owner.synchronize()
        stop = self._first_component + self._component_count
        return np.array(self._owner._array[..., self._first_component:stop],
                        copy=True,
                        order="C")

    def _borrowed_array(self) -> np.ndarray:
        self._owner.synchronize()
        stop = self._first_component + self._component_count
        return self._owner._array[..., self._first_component:stop]

    def _resident_buffer(self) -> Any:
        return self._owner._resident_buffer()


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

    def __init__(self,
                 function: Any,
                 *,
                 workgroup_size: tuple[int, int, int] = (1, 1, 1)):
        if (len(workgroup_size) != 3
                or any(not isinstance(value, int) or value <= 0
                       for value in workgroup_size)):
            raise ValueError(
                "workgroup_size must contain three positive integers")
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

    @staticmethod
    def _builtin_parameters(function: ast.FunctionDef) -> tuple[str, ...]:
        result: list[str] = []
        for argument in function.args.args:
            annotation = argument.annotation
            if annotation and any(
                    isinstance(node, ast.Call)
                    and Kernel._annotation_name(node.func) == "builtin"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "global_invocation_id"
                    for node in ast.walk(annotation)):
                result.append(argument.arg)
        return tuple(result)

    @staticmethod
    def _writable_parameters(function: ast.FunctionDef) -> tuple[str, ...]:
        parameters = {argument.arg for argument in function.args.args}
        writable: set[str] = set()
        for node in ast.walk(function):
            targets: list[ast.expr] = []
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (list(node.targets)
                           if isinstance(node, ast.Assign) else [node.target])
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
    ) -> tuple[FrontendCompileResult, ast.FunctionDef, tuple[str, ...],
               dict[str, Tensor]]:
        source = self._file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(self._file))
        function = next(
            (node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == self._entry
             ),
            None,
        )
        if function is None:
            raise RuntimeError(
                "kernel functions must be top-level definitions in files")
        builtins = self._builtin_parameters(function)
        user_parameters = [
            argument.arg for argument in function.args.args
            if argument.arg not in builtins
        ]
        if len(arguments) != len(user_parameters):
            raise TypeError(
                f"{self._entry} expects {len(user_parameters)} launch arguments"
            )
        tensors = {
            name: value
            for name, value in zip(user_parameters, arguments, strict=True)
            if isinstance(value, Tensor)
        }
        loaded_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
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
            tuple((name, value.dtype.str, value.shape)
                  for name, value in tensors.items()),
            constants,
            self._workgroup_size,
        )
        frontend = Compiler().compile_request(request)
        specialized_tree = ast.parse(frontend.specialized_source,
                                     filename=str(self._file))
        specialized_function = next(
            node for node in specialized_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == self._entry)
        return frontend, specialized_function, builtins, tensors

    def compile_artifact(self, *arguments: Any,
                         target: str) -> tuple[bytes, str]:
        if _native is None:
            raise RuntimeError("native artifact compilation requires "
                               "vernon_dsl._native")
        targets = {
            "cpu": _native.Target.CPU,
            "cuda": _native.Target.CUDA,
            "vulkan": _native.Target.VULKAN,
            "metal": _native.Target.METAL,
            "opengl": _native.Target.OPENGL,
            "opengles": _native.Target.OPENGL_ES,
        }
        if target not in targets:
            raise ValueError("target must be cpu, cuda, vulkan, metal, "
                             "opengl, or opengles")
        frontend, _, _, _ = self._lower(arguments)
        target_options = TargetOptions(
            target, {
                "glsl_version": 430
            } if target == "opengl" else {
                "glsl_version": 310
            } if target == "opengles" else {})
        program = _native.Compiler().compile_program_result(
            frontend.mlir, targets[target], **target_options.native_options)
        if not program.ok:
            raise RuntimeError(program.diagnostics)
        if len(program.artifacts) != 1:
            raise RuntimeError(
                "kernel compilation must produce exactly one artifact")
        return bytes(program.artifacts[0][1]), str(program.reflection)

    def _compile(
        self, arguments: tuple[Any, ...], features: tuple[str, ...] = ()
    ) -> _CompiledKernel:
        frontend, specialized_function, builtins, tensors = self._lower(
            arguments, features)
        target = {
            cpu: _native.Target.CPU,
            cuda: _native.Target.CUDA,
            vulkan: _native.Target.VULKAN,
            opengl: _native.Target.OPENGL,
            opengles: _native.Target.OPENGL_ES,
        }.get(_architecture) if _native is not None else None
        target_options = TargetOptions(
            _architecture.name, {
                "glsl_version": _interactive_glsl_version()
            } if _architecture in {opengl, opengles} else {})
        key_data = {
            "version": 2,
            "frontend": frontend.semantic_inputs,
            "target": target_options.target,
            "target_options": dict(target_options.options),
        }
        key = hashlib.sha256(
            canonical_json(key_data).encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if _native is None or _native_runtime is None:
            raise RuntimeError(
                f"{_architecture.name} kernel execution requires the native runtime"
            )
        if cached is None:
            assert target is not None
            program = _native.Compiler().compile_program_result(
                frontend.mlir, target, **target_options.native_options)
            if not program.ok:
                raise RuntimeError(program.diagnostics)
            cached = _CompiledKernel(
                frontend.mlir,
                specialized_function,
                builtins,
                self._writable_parameters(specialized_function),
                program,
            )
            self._cache[key] = cached
            self.compile_count += 1
        if (cached.native is None
                or cached.native_generation != _runtime_generation):
            if _architecture == cpu:
                cached.native = _native_runtime.load_cpu_entry(
                    cached.program, self._entry)
            else:
                if len(cached.program.artifacts) != 1:
                    raise RuntimeError(
                        "kernel compilation must produce exactly one artifact")
                cached.native = _native_runtime.load(
                    cached.program.artifacts[0][1],
                    cached.program.reflection,
                    self._entry,
                )
            cached.native_generation = _runtime_generation
        return cached

    def __call__(
        self,
        *arguments: Any,
        grid: tuple[int, int, int] | None = None,
        _pipeline_features: tuple[str, ...] = ()
    ) -> None:
        compiled = self._compile(arguments, _pipeline_features)
        user_parameters = [
            argument.arg for argument in compiled.function.args.args
            if argument.arg not in compiled.builtin_names
        ]
        if grid is None:
            writable_shapes = {
                value.shape
                for name, value in zip(user_parameters, arguments, strict=True)
                if name in compiled.writable_names
                and isinstance(value, Tensor)
            }
            if not writable_shapes:
                raise TypeError(
                    "grid is required when no writable Tensor domain can be inferred"
                )
            if len(writable_shapes) != 1:
                raise ValueError(
                    "all writable Tensor arguments must have the same shape")
            shape = next(iter(writable_shapes))
            if not 1 <= len(shape) <= 3:
                raise ValueError(
                    "inferred compute grids require Tensor rank one through three"
                )
            grid = tuple(reversed(shape)) + (1, ) * (3 - len(shape))
        if (len(grid) != 3 or any(not isinstance(value, int) or value <= 0
                                  for value in grid)):
            raise ValueError("grid must contain three positive integers")
        assert _native_runtime is not None and compiled.native is not None
        native_values: list[Any] = []
        for parameter, value in zip(
            (argument for argument in compiled.function.args.args
             if argument.arg not in compiled.builtin_names),
                arguments,
                strict=True,
        ):
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
        _native_runtime.synchronize()
        for name, value in zip(user_parameters, arguments, strict=True):
            if name in compiled.writable_names and isinstance(value, Tensor):
                value._mark_device_dirty()


atexit.register(Kernel.clear_cache)


class Texture:
    """RGBA8 two-dimensional runtime texture and annotation constructor."""

    def __init__(self, array: np.ndarray):
        if (not isinstance(array, np.ndarray) or array.dtype != np.uint8
                or array.ndim != 3 or array.shape[2] != 4
                or not array.flags.c_contiguous):
            raise ValueError(
                "Texture storage must be contiguous uint8 (height, width, 4)")
        self._array = np.array(array, copy=True, order="C")
        self._native_texture: Any | None = None
        self._native_generation = -1
        self._host_dirty = True
        self._device_dirty = False

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments, )
        return TypeExpr("Texture", arguments)

    @classmethod
    def zeros(cls, *, shape: tuple[int, int]) -> "Texture":
        return cls(np.zeros((*shape, 4), dtype=np.uint8))

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "Texture":
        return cls(array)

    @property
    def shape(self) -> tuple[int, int]:
        return self._array.shape[:2]

    def copy_from_numpy(self, array: np.ndarray) -> None:
        if (not isinstance(array, np.ndarray) or array.dtype != np.uint8
                or array.shape != self._array.shape
                or not array.flags.c_contiguous):
            raise ValueError(
                "texture upload requires matching uint8 shape and contiguity")
        np.copyto(self._array, array)
        self._host_dirty = True
        self._device_dirty = False

    def to_numpy(self) -> np.ndarray:
        if self._device_dirty:
            if self._native_texture is None:
                raise RuntimeError("device-dirty Texture has no allocation")
            downloaded = np.frombuffer(self._native_texture.download(),
                                       dtype=np.uint8).reshape(
                                           self._array.shape)
            np.copyto(self._array, downloaded)
            self._device_dirty = False
            self._host_dirty = False
        return self._array.copy(order="C")

    def _resident_texture(self) -> Any:
        if _native_runtime is None:
            raise RuntimeError(
                "Texture requires an initialized native runtime")
        if (self._native_texture is None
                or self._native_generation != _runtime_generation):
            height, width = self.shape
            self._native_texture = _native_runtime.create_texture(
                width, height)
            self._native_generation = _runtime_generation
            self._host_dirty = True
            self._device_dirty = False
        if self._host_dirty:
            self._native_texture.upload(self._array.tobytes(order="C"))
            self._host_dirty = False
        return self._native_texture

    def _mark_device_dirty(self) -> None:
        self._device_dirty = True
        self._host_dirty = False


@dataclass(frozen=True)
class _ReflectedParameter:
    stage: str
    entry: str
    name: str
    index: int
    interface: str
    location: int | None
    divisor: int
    value_shape: tuple[int, ...]
    varying: bool

    @property
    def components(self) -> int:
        return int(np.prod(self.value_shape,
                           dtype=np.int64)) if self.value_shape else 1


@dataclass(frozen=True)
class _ReflectedOutput:
    name: str | None
    location: int
    type_name: str


@dataclass
class _CompiledPipeline:
    native: Any
    parameters: tuple[_ReflectedParameter, ...]
    outputs: tuple[_ReflectedOutput, ...]
    key: str
    slots: Mapping[str, int]
    bundle: bytes
    target_identity: str
    native_generation: int


@dataclass(frozen=True)
class _PipelineStageRequest:
    frontend: FrontendCompileResult
    module: str
    module_manifest: str
    entry: str
    target: TargetOptions
    native_target: Any


def _annotation_parts(value: Any) -> tuple[Any, tuple[Annotation, ...]]:
    if get_origin(value) is Annotated:
        arguments = get_args(value)
        return arguments[0], tuple(item for item in arguments[1:]
                                   if isinstance(item, Annotation))
    return value, ()


class Pipeline:
    """Callable specialized compute/graphics composition."""

    _cache: ClassVar[dict[str, _CompiledPipeline]] = {}

    def __init__(self, *stages: Any, features: Iterable[str] = ()):
        kinds = tuple(
            getattr(stage, "__vernon_dsl__", (None, ))[0] for stage in stages)
        if kinds not in {("vertex", "fragment"),
                         ("compute", "vertex", "fragment")}:
            raise ValueError("pipeline stages must be (vertex, fragment) or "
                             "(compute, vertex, fragment)")
        feature_values = tuple(features)
        if any(not isinstance(value, str) or not value
               for value in feature_values):
            raise TypeError("pipeline features must be non-empty strings")
        self._features = tuple(sorted(set(feature_values)))
        self._stages = stages
        self._compute = stages[0] if kinds[0] == "compute" else None
        self._vertex = stages[-2]
        self._fragment = stages[-1]
        self._compiled: _CompiledPipeline | None = None
        self._compiled_generation = -1
        self.compile_count = 0

    @staticmethod
    def _planned_parameter(row: Mapping[str, Any]) -> _ReflectedParameter:
        uses = row.get("uses")
        if not isinstance(uses, list) or not uses:
            raise RuntimeError("planned pipeline parameter has no stage uses")
        use = next((value for value in uses
                    if value.get("stage") != "compute"), uses[0])
        location = use.get("vernon.location")
        return _ReflectedParameter(
            str(use["stage"]),
            str(use["entry"]),
            str(row["name"]),
            int(use["index"]),
            str(use["interface"]),
            int(location) if location is not None else None,
            int(use.get("vernon.instance_divisor", 0)),
            tuple(int(value) for value in use.get("shape", ())),
            False,
        )

    def _stage_request(
        self,
        stage_value: Any,
        call_arguments: Mapping[str, Any],
        target: TargetOptions,
        native_target: Any,
    ) -> _PipelineStageRequest:
        function = getattr(stage_value, "_function",
                           getattr(stage_value, "function", None))
        if function is None:
            raise RuntimeError("pipeline stage has no Python function")
        entry = function.__name__
        path = Path(inspect.getsourcefile(function) or "").resolve()
        if stage_value is self._compute:
            signature = inspect.signature(self._compute._function)
            hints = get_type_hints(self._compute._function,
                                   include_extras=True)
            values = []
            for parameter in signature.parameters.values():
                _, metadata = _annotation_parts(hints[parameter.name])
                if any(item.kind == "builtin" for item in metadata):
                    continue
                if parameter.name not in call_arguments:
                    raise TypeError(
                        f"missing pipeline argument {parameter.name!r}")
                values.append(call_arguments[parameter.name])
            frontend, _, _, _ = self._compute._lower(
                tuple(values), self._features)
        else:
            frontend = Compiler().compile_request(
                FrontendCompileRequest(path, entry, self._features))
        return _PipelineStageRequest(
            frontend,
            f"python/{path.stem}",
            canonical_json(frontend.semantic_inputs),
            entry,
            target,
            native_target,
        )

    def _compile_pipeline_bundle(
            self, call_arguments: Mapping[str, Any]) -> _CompiledPipeline:
        if _native is None or _native_runtime is None:
            raise RuntimeError("graphics requires the native pipeline runtime")
        targets = {
            vulkan: (_native.Target.VULKAN, "vulkan"),
            opengl: (_native.Target.OPENGL, "opengl"),
            opengles: (_native.Target.OPENGL_ES, "opengles"),
        }
        native_target, target_name = targets[_architecture]
        target_options = TargetOptions(
            target_name, {
                "glsl_version": _interactive_glsl_version()
            } if _architecture in {opengl, opengles} else {})
        stage_values = ([self._compute] if self._compute is not None else
                        []) + [self._vertex, self._fragment]
        requests = [
            self._stage_request(stage, call_arguments, target_options,
                                native_target) for stage in stage_values
        ]
        compiled_stages: list[CompiledStage] = []
        native_compiler = _native.Compiler()
        for request in requests:
            program = native_compiler.compile_program_result(
                request.frontend.mlir,
                request.native_target,
                **request.target.native_options,
            )
            try:
                compiled_stages.append(
                    compiled_stage_from_program(
                        program,
                        module=request.module,
                        module_manifest=request.module_manifest,
                        entry=request.entry,
                        target=request.target,
                    ))
            except PipelineCompileError as error:
                raise RuntimeError(str(error)) from None

        pipeline_id = "interactive/" + hashlib.sha256(canonical_json({
            "stages": [stage.id for stage in compiled_stages],
            "features": self._features,
            "target": target_name,
            "target_options": dict(target_options.options),
        }).encode("utf-8")).hexdigest()
        planned_by_kind = {
            stage.stage: stage
            for stage in compiled_stages
        }
        try:
            plan = build_bundle_plan(
                pipeline_id,
                target_options,
                self._features,
                [(self._features, planned_by_kind)],
            )
            bundle = materialize_bundle(
                plan, {
                    stage.id: inline_artifact_descriptor(stage.artifact)
                    for stage in compiled_stages
                })
        except PipelineCompileError as error:
            raise TypeError(str(error)) from None
        parameter_rows = plan.variants[0].parameters
        slots = {
            str(parameter["name"]): int(parameter["slot"])
            for parameter in parameter_rows
        }
        parameters = tuple(
            self._planned_parameter(parameter)
            for parameter in parameter_rows)
        outputs = tuple(
            _ReflectedOutput(
                None if output.get("name") == f"output_{output['location']}"
                else output.get("name"), int(output["location"]),
                             str(output["type"]))
            for output in plan.variants[0].outputs)
        bundle_bytes = serialize_bundle(bundle)
        key = hashlib.sha256(bundle_bytes).hexdigest()
        cached = self._cache.get(key)
        if cached is None:
            native = _native_runtime.load_pipeline(bundle_bytes,
                                                   list(self._features))
            cached = _CompiledPipeline(
                native, parameters, outputs, key, slots, bundle_bytes,
                canonical_json({
                    "target": target_name,
                    "target_options": dict(target_options.options),
                }), _runtime_generation)
            self._cache[key] = cached
            self.compile_count += 1
        elif cached.native_generation != _runtime_generation:
            cached.native = _native_runtime.load_pipeline(
                cached.bundle, list(self._features))
            cached.native_generation = _runtime_generation
        self._compiled = cached
        self._compiled_generation = _runtime_generation
        return cached

    def _compile(
            self,
            call_arguments: Mapping[str, Any] | None = None
    ) -> _CompiledPipeline:
        if (self._compiled is not None
                and self._compiled_generation == _runtime_generation):
            return self._compiled
        if _architecture in {vulkan, opengl, opengles}:
            target_identity = canonical_json({
                "target": _architecture.name,
                "target_options": ({
                    "glsl_version": _interactive_glsl_version()
                } if _architecture in {opengl, opengles} else {}),
            })
            if (self._compiled is not None
                    and self._compiled.target_identity == target_identity):
                assert _native_runtime is not None
                self._compiled.native = _native_runtime.load_pipeline(
                    self._compiled.bundle, list(self._features))
                self._compiled.native_generation = _runtime_generation
                self._compiled_generation = _runtime_generation
                return self._compiled
            return self._compile_pipeline_bundle(call_arguments or {})
        if _architecture == cpu:
            raise RuntimeError(
                "CPU graphics pipelines require a software rasterizer, which "
                "Vernon does not provide; CPU supports compute kernels only")
        raise RuntimeError("unsupported graphics backend")

    def __call__(self, **arguments: Any) -> None:
        target = arguments.pop("target", None)
        targets = arguments.pop("targets", None)
        indices = arguments.pop("indices", None)
        topology = arguments.pop("topology", triangles)
        if target is not None and targets is not None:
            raise TypeError("pipeline call cannot use both target and targets")
        compiled = self._compile(arguments)
        host_parameters = [
            value for value in compiled.parameters if not value.varying
        ]
        expected = {value.name for value in host_parameters}
        compute_names: set[str] = set()
        if self._compute is not None:
            compute_names = {
                parameter.name
                for parameter in inspect.signature(
                    self._compute._function).parameters.values() if
                not any(item.kind == "builtin" for item in _annotation_parts(
                    get_type_hints(self._compute._function,
                                   include_extras=True)[parameter.name])[1])
            }
        missing = expected - set(arguments)
        if missing:
            raise TypeError(
                f"missing pipeline argument(s): {', '.join(sorted(missing))}")
        unexpected = set(arguments) - expected - compute_names
        if unexpected:
            raise TypeError(
                f"unexpected pipeline argument(s): {', '.join(sorted(unexpected))}"
            )

        pipeline_arguments: list[tuple[Any, ...]] = []
        vertex_count: int | None = None
        instance_count: int | None = None
        for parameter in host_parameters:
            value = arguments[parameter.name]
            if parameter.interface == "uniform":
                if not isinstance(value, (Tensor, TensorView)):
                    raise TypeError(
                        f"uniform {parameter.name!r} must be a Tensor")
                values = value._borrowed_array()
                if (values.dtype != np.dtype(np.float32)
                        or values.size != parameter.components):
                    raise ValueError(
                        f"uniform {parameter.name!r} has incompatible dtype or shape"
                    )
                pipeline_arguments.append(
                    ("tensor", compiled.slots[parameter.name], values,
                     _native.DATA_F32, _native.ACCESS_READ))
                continue
            if not isinstance(value, (Tensor, TensorView)):
                scalar = np.asarray(value)
                if scalar.dtype.kind == "f":
                    scalar = np.asarray(value, dtype=np.float32)
                    dtype = _native.DATA_F32
                elif scalar.dtype.kind == "u":
                    scalar = np.asarray(value, dtype=np.uint32)
                    dtype = _native.DATA_U32
                else:
                    scalar = np.asarray(value, dtype=np.int32)
                    dtype = _native.DATA_I32
                pipeline_arguments.append(
                    ("tensor", compiled.slots[parameter.name], scalar, dtype,
                     _native.ACCESS_READ))
                continue
            if not isinstance(value, (Tensor, TensorView)):
                raise TypeError(
                    f"pipeline argument {parameter.name!r} must be a Tensor")
            if parameter.stage == "compute":
                dtype = {
                    np.dtype(np.float32): _native.DATA_F32,
                    np.dtype(np.uint32): _native.DATA_U32,
                    np.dtype(np.int32): _native.DATA_I32,
                    np.dtype(np.float64): _native.DATA_F64,
                }.get(value.dtype)
                if dtype is None:
                    raise TypeError(
                        f"Vulkan pipeline does not support dtype {value.dtype}"
                    )
                layout = value.layout
                pipeline_arguments.append(
                    ("tensor", compiled.slots[parameter.name],
                     value._resident_buffer(), dtype,
                     _native.ACCESS_READ_WRITE, list(value.shape),
                     list(layout.byte_strides), layout.byte_offset))
                continue
            if value.dtype != np.dtype(np.float32):
                raise TypeError("OpenGL vertex inputs require f32")
            if not value.shape or value.shape[1:] != parameter.value_shape:
                raise ValueError(
                    f"pipeline argument {parameter.name!r} has incompatible shape"
                )
            count = value.shape[0]
            if parameter.divisor:
                if instance_count is not None and instance_count != count:
                    raise ValueError(
                        "all instance inputs must share a leading dimension")
                instance_count = count
            else:
                if vertex_count is not None and vertex_count != count:
                    raise ValueError(
                        "all vertex inputs must share a leading dimension")
                vertex_count = count
            layout = value.layout
            buffer = value._resident_buffer()
            dtype = {
                np.dtype(np.float32): _native.DATA_F32,
                np.dtype(np.uint32): _native.DATA_U32,
                np.dtype(np.int32): _native.DATA_I32,
                np.dtype(np.float64): _native.DATA_F64,
            }.get(value.dtype)
            if dtype is None:
                raise TypeError(
                    f"Vulkan pipeline does not support dtype {value.dtype}")
            pipeline_arguments.append(
                ("tensor", compiled.slots[parameter.name], buffer, dtype,
                 _native.ACCESS_READ_WRITE, list(value.shape),
                 list(layout.byte_strides), layout.byte_offset))
        if vertex_count is None:
            raise ValueError("pipeline requires at least one vertex input")

        native_index = None
        index_count = 0
        if indices is not None:
            if (not isinstance(indices, Tensor)
                    or indices.dtype != np.dtype(np.uint32)
                    or len(indices.shape) != 1 or not indices.shape[0]):
                raise TypeError(
                    "indices must be a non-empty rank-one u32 Tensor")
            index_count = indices.shape[0]
            if int(np.max(indices.to_numpy())) >= vertex_count:
                raise ValueError("index Tensor references a missing vertex")
            native_index = indices._resident_buffer()
        draw_count = index_count or vertex_count
        if draw_count % topology.vertices_per_primitive:
            raise ValueError(
                f"draw count is incompatible with {topology.name} topology")

        native_target = None
        native_attachments: list[tuple[int, Any]] = []
        rendered_targets: list[Texture] = []
        if targets is not None:
            if not isinstance(targets, Mapping):
                raise TypeError("targets must be a mapping of output names")
            named_outputs = {
                output.name: output
                for output in compiled.outputs if output.name is not None
            }
            if len(named_outputs) != len(compiled.outputs):
                raise TypeError(
                    "targets= requires named fragment struct outputs")
            if set(targets) != set(named_outputs):
                raise ValueError(
                    "target keys must exactly match fragment output names")
            for name, output in sorted(named_outputs.items(),
                                       key=lambda item: item[1].location):
                texture = targets[name]
                if not isinstance(texture, Texture):
                    raise TypeError(f"target {name!r} must be a Texture")
                native_attachments.append(
                    (output.location, texture._resident_texture()))
                rendered_targets.append(texture)
            if len({texture.shape for texture in rendered_targets}) != 1:
                raise ValueError("all targets must have the same extent")
            # The native binding keeps the legacy target positional argument
            # non-null; ABI v3 ignores it when explicit attachments are set.
            native_target = native_attachments[0][1]
        else:
            if (not isinstance(target, Texture) or len(compiled.outputs) != 1
                    or compiled.outputs[0].location != 0
                    or compiled.outputs[0].name is not None):
                raise TypeError(
                    "target=Texture requires one unnamed fragment output at location zero"
                )
            native_target = target._resident_texture()
            rendered_targets.append(target)

        native_topology = {
            triangles: _native.TOPOLOGY_TRIANGLE_LIST,
            lines: _native.TOPOLOGY_LINE_LIST,
            points: _native.TOPOLOGY_POINT_LIST,
        }[topology]
        compiled.native.invoke(
            pipeline_arguments,
            native_index,
            index_count,
            0,
            native_attachments
            or ([(0, native_target)] if native_target is not None else []),
            native_topology,
            vertex_count,
            instance_count or 1,
        )
        assert _native_runtime is not None
        _native_runtime.synchronize()
        if self._compute is not None:
            for name in compute_names:
                value = arguments.get(name)
                if isinstance(value, Tensor):
                    value._mark_device_dirty()
        for texture in rendered_targets:
            texture._mark_device_dirty()


def pipeline(*stages: Any, features: Iterable[str] = ()) -> Pipeline:
    return Pipeline(*stages, features=features)
