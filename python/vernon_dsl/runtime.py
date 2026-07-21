from __future__ import annotations

import ast
import atexit
import hashlib
import importlib
import inspect
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, get_args, get_origin, get_type_hints

import numpy as np

from .compiler import Compiler
from .module_graph import load_project
from .shader_assets import (_dtype_and_shape, encode_runtime_stage,
                            serialize_runtime_pipeline_bundle)
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
        kernel_type.clear_cache()
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

    def _resident_buffer(self) -> Any:
        return self._owner._resident_buffer()


@dataclass(frozen=True)
class _CompiledKernel:
    mlir: str
    function: ast.FunctionDef
    builtin_names: tuple[str, ...]
    writable_names: tuple[str, ...]
    native: Any | None = None


class _CpuAotKernel:
    """Keep an ephemeral CPU bundle alive until its native library unloads."""

    def __init__(self, native: Any, temporary: Path):
        self._native = native
        self._temporary = temporary

    def launch(self, x: int, y: int, z: int, values: list[Any]) -> None:
        if self._native is None:
            raise RuntimeError("CPU AOT kernel is closed")
        self._native.launch(x, y, z, values)

    def close(self) -> None:
        if self._native is None:
            return
        native, self._native = self._native, None
        del native
        shutil.rmtree(self._temporary)

    def __del__(self) -> None:
        self.close()


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
        for compiled in cls._cache.values():
            if isinstance(compiled.native, _CpuAotKernel):
                compiled.native.close()
        cls._cache.clear()

    @staticmethod
    def _annotation_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _specialized_tree(self, source: str,
                          tensor_arguments: dict[str, Tensor]) -> ast.Module:
        tree = ast.parse(source, filename=str(self._file))
        selected: ast.FunctionDef | None = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == self._entry:
                selected = node
                break
        if selected is None:
            raise RuntimeError(
                f"kernel entry {self._entry!r} is not top-level")
        for argument in selected.args.args:
            tensor = tensor_arguments.get(argument.arg)
            if tensor is None or argument.annotation is None:
                continue
            annotation = argument.annotation
            if (isinstance(annotation, ast.Subscript) and
                    self._annotation_name(annotation.value) == "Annotated"):
                annotation = (annotation.slice.elts[0] if isinstance(
                    annotation.slice, ast.Tuple) else annotation.slice)
            if not (isinstance(annotation, ast.Subscript)
                    and self._annotation_name(annotation.value) == "Tensor"):
                continue
            items = (list(annotation.slice.elts) if isinstance(
                annotation.slice, ast.Tuple) else [annotation.slice])
            if len(items) < 2:
                continue
            shape_node = items[1]
            shape_nodes = (list(shape_node.elts) if isinstance(
                shape_node, ast.Tuple) else items[1:])
            if len(shape_nodes) != len(tensor.shape):
                raise ValueError(
                    f"Tensor argument {argument.arg!r} rank does not match annotation"
                )
            for index, (shape, concrete) in enumerate(
                    zip(shape_nodes, tensor.shape, strict=True)):
                if isinstance(shape, ast.Constant) and shape.value is None:
                    shape_nodes[index] = ast.copy_location(
                        ast.Constant(value=concrete), shape)
                elif not (isinstance(shape, ast.Constant)
                          and shape.value == concrete):
                    raise ValueError(
                        f"Tensor argument {argument.arg!r} shape does not match annotation"
                    )
            items[1] = ast.Tuple(elts=shape_nodes, ctx=ast.Load())
            annotation.slice = ast.Tuple(elts=items, ctx=ast.Load())
        ast.fix_missing_locations(tree)
        return tree

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
    ) -> tuple[str, ast.FunctionDef, tuple[str, ...], dict[str, Tensor], str]:
        original_source = self._file.read_text(encoding="utf-8")
        original_tree = ast.parse(original_source, filename=str(self._file))
        has_helpers = any(
            isinstance(node, ast.FunctionDef) and any(
                Kernel._annotation_name(decorator.func if isinstance(
                    decorator, ast.Call) else decorator) == "func"
                for decorator in node.decorator_list)
            for node in original_tree.body)
        project = (load_project(self._file, features, self._entry)
                   if features or has_helpers else None)
        source = project.source if project is not None else original_source
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
        specialized = self._specialized_tree(source, tensors)
        specialized_function = next(
            node for node in specialized.body
            if isinstance(node, ast.FunctionDef) and node.name == self._entry)
        specialized_function = _CapturedConstantSpecializer(
            self._globals).visit(specialized_function)
        assert isinstance(specialized_function, ast.FunctionDef)
        ast.fix_missing_locations(specialized_function)
        if project is not None:
            specialized.body = [
                specialized_function if isinstance(node, ast.FunctionDef)
                and node.name == self._entry else node
                for node in specialized.body
            ]
            specialized_source = ast.unparse(specialized)
        else:
            specialized_source = ast.unparse(
                ast.Module(body=[specialized_function], type_ignores=[]))
        mlir = Compiler().compile(
            specialized_source,
            str(self._file),
            project.dependencies if project is not None else (),
            project.features if project is not None else (),
            features,
        )
        return mlir, specialized_function, builtins, tensors, source

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
        mlir, _, _, _, _ = self._lower(arguments)
        return _native.Compiler().compile(mlir, targets[target])

    def _compile(
        self, arguments: tuple[Any, ...], features: tuple[str, ...] = ()
    ) -> _CompiledKernel:
        mlir, specialized_function, builtins, tensors, source = self._lower(
            arguments, features)
        key_data = {
            "version":
            1,
            "source":
            hashlib.sha256(source.encode()).hexdigest(),
            "entry":
            self._entry,
            "arch":
            _architecture.name,
            "workgroup":
            self._workgroup_size,
            "runtime_generation":
            _runtime_generation,
            "features":
            features,
            "tensors": [(name, value.dtype.str, value.shape)
                        for name, value in tensors.items()],
        }
        key = hashlib.sha256(repr(sorted(
            key_data.items())).encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if _native is None or _native_runtime is None:
            raise RuntimeError(
                f"{_architecture.name} kernel execution requires the native runtime"
            )
        if _architecture == cpu:
            native_kernel = self._compile_cpu_aot(mlir)
        else:
            target = {
                cuda: _native.Target.CUDA,
                vulkan: _native.Target.VULKAN,
                opengl: _native.Target.OPENGL,
                opengles: _native.Target.OPENGL_ES,
            }[_architecture]
            artifact, reflection = _native.Compiler().compile(mlir, target)
            native_kernel = _native_runtime.load(artifact, reflection,
                                                 self._entry)
        compiled = _CompiledKernel(
            mlir, specialized_function, builtins,
            self._writable_parameters(specialized_function), native_kernel)
        self._cache[key] = compiled
        self.compile_count += 1
        return compiled

    @staticmethod
    def _native_compiler_path() -> Path:
        executable = "vernon-compile.exe" if os.name == "nt" else "vernon-compile"
        configured = os.environ.get("VERNON_COMPILER")
        if configured:
            path = Path(configured).expanduser().resolve()
            if path.is_file():
                return path
            raise RuntimeError(f"VERNON_COMPILER does not name a file: {path}")
        assert _native is not None
        repository = Path(__file__).resolve().parents[2]
        source_build = repository / "build" / "source"
        candidates = [Path(_native.__file__).resolve().with_name(executable)]
        candidates.extend(
            source_build / configuration / executable
            for configuration in ("Release", "Debug", "RelWithDebInfo",
                                  "MinSizeRel"))
        candidates.append(source_build / executable)
        discovered = shutil.which(executable)
        if discovered:
            candidates.append(Path(discovered))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise RuntimeError(
            "CPU AOT requires vernon-compile; set VERNON_COMPILER to its path")

    def _compile_cpu_aot(self, mlir: str) -> _CpuAotKernel:
        assert _native_runtime is not None
        root = Path(tempfile.mkdtemp(prefix="vernon_cpu_kernel_"))
        mlir_path = root / "kernel.mlir"
        bundle_path = root / "bundle"
        mlir_path.write_text(mlir, encoding="utf-8", newline="\n")
        command = [
            str(self._native_compiler_path()), "--target", "cpu",
            str(mlir_path), "--compute-bundle",
            str(bundle_path), "--host-runtime-bundle"
        ]
        try:
            result = subprocess.run(command,
                                    capture_output=True,
                                    text=True,
                                    encoding="utf-8",
                                    errors="replace",
                                    check=False)
            if result.returncode != 0:
                diagnostics = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    f"CPU AOT compilation failed for {self._entry}: {diagnostics}"
                )
            native = _native_runtime.load_compute_bundle(str(bundle_path))
            return _CpuAotKernel(native, root)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise

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


@dataclass(frozen=True)
class _CompiledPipeline:
    native: Any
    parameters: tuple[_ReflectedParameter, ...]
    outputs: tuple[_ReflectedOutput, ...]
    key: str
    slots: Mapping[str, int]


def _annotation_parts(value: Any) -> tuple[Any, tuple[Annotation, ...]]:
    if get_origin(value) is Annotated:
        arguments = get_args(value)
        return arguments[0], tuple(item for item in arguments[1:]
                                   if isinstance(item, Annotation))
    return value, ()


def _reflection_shape(type_name: str) -> tuple[int, ...]:
    if not type_name.startswith("tensor<") or not type_name.endswith(">"):
        return ()
    dimensions = type_name[7:-1].split("x")[:-1]
    if not all(value.isdigit() for value in dimensions):
        return ()
    return tuple(int(value) for value in dimensions)


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
    def _entry(reflection: str, entry_name: str) -> dict[str, Any]:
        document = json.loads(reflection)
        for entry in document.get("entries", ()):
            if entry.get("name") == entry_name:
                return entry
        raise RuntimeError(f"compiler reflection has no entry {entry_name!r}")

    @staticmethod
    def _parameter(stage: str, entry: str,
                   row: Mapping[str, Any]) -> _ReflectedParameter | None:
        if "vernon.builtin" in row:
            return None
        name = row.get("vernon.source_name")
        interface = row.get("vernon.interface")
        if not isinstance(name, str) or not isinstance(interface, str):
            raise RuntimeError(
                "graphics reflection is missing source metadata")
        location = row.get("vernon.location")
        return _ReflectedParameter(
            stage,
            entry,
            name,
            int(row["index"]),
            interface,
            int(location) if location is not None else None,
            int(row.get("vernon.instance_divisor", 0)),
            _reflection_shape(str(row["type"])),
            bool(row.get("vernon.varying", False)),
        )

    def _compile_pipeline_bundle(
            self, call_arguments: Mapping[str, Any]) -> _CompiledPipeline:
        if _native is None or _native_runtime is None:
            raise RuntimeError("graphics requires the native pipeline runtime")
        targets = {
            vulkan: (_native.Target.VULKAN, "vulkan", 0),
            opengl: (_native.Target.OPENGL, "opengl", 330),
            opengles: (_native.Target.OPENGL_ES, "opengles", 310),
        }
        native_target, target_name, glsl_version = targets[_architecture]
        stage_values = ([self._compute] if self._compute is not None else
                        []) + [self._vertex, self._fragment]
        compiled_stages: list[dict[str, Any]] = []
        native_compiler = _native.Compiler()
        for stage_value in stage_values:
            function = getattr(stage_value, "_function",
                               getattr(stage_value, "function", None))
            if function is None:
                raise RuntimeError("pipeline stage has no Python function")
            entry_name = function.__name__
            path = Path(inspect.getsourcefile(function) or "")
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
                mlir, _, _, _, _ = self._compute._lower(
                    tuple(values), self._features)
            else:
                mlir = Compiler().compile_file(path,
                                               features=self._features,
                                               entry=entry_name)
            artifacts, reflection_text = native_compiler.compile_program(
                mlir, native_target, glsl_version)
            reflection = json.loads(reflection_text)
            entry = self._entry(reflection_text, entry_name)
            artifact_rows = [
                row for row in reflection.get("artifacts", ())
                if row.get("entry_point") == entry_name
                and row.get("stage") == entry["stage"]
            ]
            if len(artifact_rows) != 1:
                raise RuntimeError(
                    f"compiler reflection has no unique {entry['stage']} artifact"
                )
            reflected_filename = artifact_rows[0].get("filename")
            matches = [(name, bytes(data)) for name, data in artifacts
                       if name == reflected_filename]
            if len(matches) != 1:
                raise RuntimeError(
                    f"compiler produced no unique {entry['stage']} artifact")
            artifact_name, artifact = matches[0]
            stage_id = hashlib.sha256((entry_name + entry["stage"] +
                                       reflection_text).encode("utf-8") +
                                      artifact).hexdigest()
            record = {
                "id": stage_id,
                "entry": entry_name,
                "stage": entry["stage"],
                "target": target_name,
                "format": artifact_rows[0].get("format"),
                "filename": artifact_name,
                "reflection": reflection,
            }
            compiled_stages.append({
                "id": stage_id,
                "entry": entry,
                "record": record,
                "artifact": artifact,
                "mlir": mlir,
                "reflection": reflection_text,
                "path": str(path.resolve()),
            })

        uses_by_name: dict[str, list[dict[str, Any]]] = {}
        reflected_parameters: list[_ReflectedParameter] = []
        for stage in compiled_stages:
            entry = stage["entry"]
            for row in entry.get("arguments", ()):
                parameter = self._parameter(entry["stage"], entry["name"], row)
                if parameter is None:
                    continue
                reflected_parameters.append(parameter)
                if parameter.varying:
                    continue
                inferred_dtype, inferred_shape = _dtype_and_shape(
                    row.get("type"))
                use = {
                    "stage": entry["stage"],
                    "entry": entry["name"],
                    "index": int(row["index"]),
                    "kind": row.get("kind", "scalar"),
                    "type": row.get("type"),
                    "dtype": (row.get("dtype") or row.get("vernon.dtype")
                              or inferred_dtype),
                    "shape": row.get("shape", inferred_shape),
                    "interface": row["vernon.interface"],
                    "access": row.get("access", "read"),
                }
                for name in ("vernon.location", "vernon.instance_divisor",
                             "vernon.set", "vernon.binding"):
                    if name in row:
                        use[name] = row[name]
                uses_by_name.setdefault(parameter.name, []).append(use)

        slots = {name: slot for slot, name in enumerate(sorted(uses_by_name))}
        parameter_rows = []
        for name in sorted(uses_by_name):
            uses = uses_by_name[name]
            first = uses[0]
            kinds = {
                "inline" if use["interface"] == "uniform" or
                (use["stage"] == "compute"
                 and use["kind"] not in {"tensor", "texture"}) else
                ("texture" if use["kind"] == "texture" else "tensor")
                for use in uses
            }
            if len(kinds) != 1:
                raise TypeError(f"incompatible pipeline parameter {name!r}")
            accesses = {str(use.get("access", "read")) for use in uses}
            access = ("read_write" if "read_write" in accesses or accesses
                      == {"read", "write"} else next(iter(accesses)))
            parameter_rows.append({
                "name": name,
                "slot": slots[name],
                "kind": next(iter(kinds)),
                "type": first.get("type"),
                "dtype": first.get("dtype"),
                "shape": first.get("shape", []),
                "access": access,
                "uses": uses,
            })

        stage_by_kind = {
            stage["entry"]["stage"]: stage
            for stage in compiled_stages
        }
        fragment_entry = stage_by_kind["fragment"]["entry"]
        outputs = tuple(
            _ReflectedOutput(row.get("vernon.source_name"),
                             int(row["vernon.location"]), str(row["type"]))
            for row in fragment_entry.get("results", ())
            if "vernon.location" in row)
        steps = []
        if "compute" in stage_by_kind:
            steps.append({
                "kind": "dispatch",
                "stage": stage_by_kind["compute"]["id"],
            })
            steps.append({
                "kind": "barrier",
                "source": "compute_write",
                "destination": "vertex_read",
            })
        steps.append({
            "kind": "draw",
            "vertex": stage_by_kind["vertex"]["id"],
            "fragment": stage_by_kind["fragment"]["id"],
        })
        bundle = {
            "schema_version":
            2,
            "invocation_abi_version":
            1,
            "type":
            "pipeline",
            "id":
            "interactive/" + hashlib.sha256(
                repr((tuple(stage["id"] for stage in compiled_stages),
                      self._features)).encode("utf-8")).hexdigest(),
            "target":
            target_name,
            "target_options": {},
            "features":
            list(self._features),
            "variants": [{
                "key":
                list(self._features),
                "parameters":
                parameter_rows,
                "outputs": [{
                    "name": output.name or f"output_{output.location}",
                    "kind": "texture",
                    "dtype": _dtype_and_shape(output.type_name)[0],
                    "shape": _dtype_and_shape(output.type_name)[1],
                    "access": "write",
                    "location": output.location,
                    "type": output.type_name,
                } for output in outputs],
                "steps":
                steps,
            }],
            "stage_artifacts": {
                stage["id"]:
                encode_runtime_stage(stage["record"], stage["artifact"])
                for stage in compiled_stages
            },
        }
        bundle_bytes = serialize_runtime_pipeline_bundle(bundle)
        key = hashlib.sha256(bundle_bytes).hexdigest()
        cached = self._cache.get(key)
        if cached is None:
            native = _native_runtime.load_pipeline(bundle_bytes,
                                                   list(self._features))
            cached = _CompiledPipeline(native, tuple(reflected_parameters),
                                       outputs, key, slots)
            self._cache[key] = cached
            self.compile_count += 1
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
        uses_pipeline_runtime = _architecture in {vulkan, opengl, opengles}

        host_parameters = [
            value for value in compiled.parameters if not value.varying
        ]
        if uses_pipeline_runtime:
            merged_parameters: dict[str, _ReflectedParameter] = {}
            for value in host_parameters:
                previous = merged_parameters.get(value.name)
                if previous is None or previous.stage == "compute":
                    merged_parameters[value.name] = value
            host_parameters = [
                merged_parameters[name]
                for name in sorted(merged_parameters,
                                   key=lambda name: compiled.slots[name])
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

        bindings: list[tuple[Any, ...]] = []
        uniforms: list[tuple[str, bytes, int]] = []
        pipeline_arguments: list[tuple[Any, ...]] = []
        vertex_count: int | None = None
        instance_count: int | None = None
        for parameter in host_parameters:
            value = arguments[parameter.name]
            if parameter.interface == "uniform":
                if not isinstance(value, (Tensor, TensorView)):
                    raise TypeError(
                        f"uniform {parameter.name!r} must be a Tensor")
                values = value.to_numpy()
                if (values.dtype != np.dtype(np.float32)
                        or values.size != parameter.components):
                    raise ValueError(
                        f"uniform {parameter.name!r} has incompatible dtype or shape"
                    )
                uniforms.append(
                    (f"{parameter.entry}_arg_{parameter.index}._m0",
                     values.tobytes(order="C"), parameter.components))
                if uses_pipeline_runtime:
                    pipeline_arguments.append(
                        ("inline", compiled.slots[parameter.name],
                         _native.DATA_F32, values.tobytes(order="C"),
                         list(values.shape)))
                continue
            if uses_pipeline_runtime and not isinstance(
                    value, (Tensor, TensorView)):
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
                    ("inline", compiled.slots[parameter.name], dtype,
                     scalar.tobytes(), list(scalar.shape)))
                continue
            if not isinstance(value, (Tensor, TensorView)):
                raise TypeError(
                    f"pipeline argument {parameter.name!r} must be a Tensor")
            if uses_pipeline_runtime and parameter.stage == "compute":
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
            if parameter.location is None:
                raise RuntimeError(
                    f"vertex input {parameter.name!r} has no location")
            if parameter.divisor not in {0, 1}:
                raise RuntimeError(
                    "OpenGL runtime currently supports instance divisor 1 only"
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
            stride = layout.byte_strides[0]
            buffer = value._resident_buffer()
            if uses_pipeline_runtime:
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
                pipeline_arguments.append(
                    ("tensor", compiled.slots[parameter.name], buffer, dtype,
                     _native.ACCESS_READ_WRITE, list(value.shape),
                     list(layout.byte_strides), layout.byte_offset))
            if len(parameter.value_shape) == 2:
                columns, rows = parameter.value_shape
                for column in range(columns):
                    bindings.append(
                        (parameter.location + column, buffer, rows, stride,
                         layout.byte_offset + column * rows * 4,
                         parameter.divisor))
            else:
                bindings.append(
                    (parameter.location, buffer, parameter.components, stride,
                     layout.byte_offset, parameter.divisor))
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
            # non-null; ABI v2 ignores it when explicit attachments are set.
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

        if uses_pipeline_runtime:
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
        if uses_pipeline_runtime and self._compute is not None:
            for name in compute_names:
                value = arguments.get(name)
                if isinstance(value, Tensor):
                    value._mark_device_dirty()
        for texture in rendered_targets:
            texture._mark_device_dirty()


def pipeline(*stages: Any, features: Iterable[str] = ()) -> Pipeline:
    return Pipeline(*stages, features=features)


class _CapturedConstantSpecializer(ast.NodeTransformer):

    def __init__(self, globals_: dict[str, Any]):
        self.globals = globals_

    def visit_Name(self, node: ast.Name) -> ast.expr:
        value = self.globals.get(node.id)
        if isinstance(node.ctx, ast.Load) and isinstance(
                value, (int, float, bool)):
            return ast.copy_location(ast.Constant(value=value), node)
        return node
