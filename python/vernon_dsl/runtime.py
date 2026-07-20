from __future__ import annotations

import ast
import hashlib
import inspect
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .compiler import Compiler
from .types import TypeExpr, _Scalar

try:
    from . import _native
except ImportError:
    try:
        import importlib

        _native = importlib.import_module("_native")
    except ImportError:
        _native = None


@dataclass(frozen=True)
class _Architecture:
    name: str


cpu = _Architecture("cpu")
cuda = _Architecture("cuda")
_architecture = cpu
_native_runtime: Any | None = None
_runtime_generation = 0


def init(*, arch: _Architecture = cpu) -> None:
    global _architecture, _native_runtime, _runtime_generation
    if arch not in {cpu, cuda}:
        raise ValueError("arch must be vernon_dsl.cpu or vernon_dsl.cuda")
    if arch == cuda:
        if _native is None:
            raise RuntimeError(
                "CUDA requires the optional vernon_dsl._native module")
        if not _native.runtime_available(_native.RuntimeBackend.CUDA):
            raise RuntimeError(
                "CUDA Driver API or a usable CUDA device is unavailable")
        _native_runtime = _native.Runtime(_native.RuntimeBackend.CUDA)
    else:
        _native_runtime = (_native.Runtime(_native.RuntimeBackend.CPU)
                           if _native is not None else None)
    _architecture = arch
    _runtime_generation += 1


_NUMPY_DTYPES = {
    "bool": np.dtype(np.bool_),
    "i32": np.dtype(np.int32),
    "u32": np.dtype(np.uint32),
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}


class Tensor:
    """Contiguous row-major runtime Tensor and annotation constructor."""

    def __init__(self, array: np.ndarray):
        if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
            raise ValueError("Tensor storage must be a contiguous NumPy array")
        self._array = array

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

    def to_numpy(self) -> np.ndarray:
        self.synchronize()
        return self._array.copy(order="C")

    def copy_from_numpy(self, array: np.ndarray) -> None:
        if (not isinstance(array, np.ndarray) or array.dtype != self.dtype
                or array.shape != self.shape or not array.flags.c_contiguous):
            raise ValueError(
                "upload requires matching dtype, shape, and contiguity")
        np.copyto(self._array, array)

    def synchronize(self) -> None:
        return None


@dataclass(frozen=True)
class _CompiledKernel:
    mlir: str
    function: ast.FunctionDef
    builtin_names: tuple[str, ...]
    native: Any | None = None


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
        self._file = Path(inspect.getsourcefile(function) or "")
        self._entry = function.__name__
        self._workgroup_size = workgroup_size
        self._globals = function.__globals__
        self.compile_count = 0

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

    def _compile(self, arguments: tuple[Any, ...]) -> _CompiledKernel:
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
        specialized = self._specialized_tree(source, tensors)
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
            "tensors": [(name, value.dtype.str, value.shape)
                        for name, value in tensors.items()],
        }
        key = hashlib.sha256(repr(sorted(
            key_data.items())).encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        specialized_function = next(
            node for node in specialized.body
            if isinstance(node, ast.FunctionDef) and node.name == self._entry)
        specialized_function = _CapturedConstantSpecializer(
            self._globals).visit(specialized_function)
        assert isinstance(specialized_function, ast.FunctionDef)
        ast.fix_missing_locations(specialized_function)
        specialized_source = ast.unparse(
            ast.Module(body=[specialized_function], type_ignores=[]))
        mlir = Compiler().compile(specialized_source, str(self._file))
        native_kernel = None
        if _native is not None and _native_runtime is not None:
            target = (_native.Target.CUDA
                      if _architecture == cuda else _native.Target.CPU)
            try:
                artifact, reflection = _native.Compiler().compile(mlir, target)
                native_kernel = _native_runtime.load(artifact, reflection,
                                                     self._entry)
            except RuntimeError:
                if _architecture == cuda:
                    raise
        compiled = _CompiledKernel(mlir, specialized_function, builtins,
                                   native_kernel)
        self._cache[key] = compiled
        self.compile_count += 1
        return compiled

    def __call__(self, *arguments: Any, grid: tuple[int, int, int]) -> None:
        if (len(grid) != 3 or any(not isinstance(value, int) or value <= 0
                                  for value in grid)):
            raise ValueError("grid must contain three positive integers")
        compiled = self._compile(arguments)
        user_parameters = [
            argument.arg for argument in compiled.function.args.args
            if argument.arg not in compiled.builtin_names
        ]
        if compiled.native is not None:
            assert _native_runtime is not None
            native_values: list[Any] = []
            tensor_copies: list[tuple[Tensor, Any]] = []
            for parameter, value in zip(
                (argument for argument in compiled.function.args.args
                 if argument.arg not in compiled.builtin_names),
                    arguments,
                    strict=True,
            ):
                if isinstance(value, Tensor):
                    buffer = _native_runtime.allocate(value._array.nbytes,
                                                      value.dtype.itemsize)
                    buffer.upload(value._array.tobytes(order="C"))
                    native_values.append(buffer)
                    tensor_copies.append((value, buffer))
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
            for tensor, buffer in tensor_copies:
                downloaded = np.frombuffer(buffer.download(),
                                           dtype=tensor.dtype).reshape(
                                               tensor.shape)
                np.copyto(tensor._array, downloaded)
            return
        base = dict(zip(user_parameters, arguments, strict=True))
        for z in range(grid[2]):
            for y in range(grid[1]):
                for x in range(grid[0]):
                    environment = base.copy()
                    gid = np.array((x, y, z), dtype=np.uint32)
                    for name in compiled.builtin_names:
                        environment[name] = gid
                    _AstInterpreter(self._globals,
                                    environment).run(compiled.function.body)


class _AstInterpreter:

    def __init__(self, globals_: dict[str, Any], environment: dict[str, Any]):
        self.globals = globals_
        self.environment = environment

    def run(self, statements: list[ast.stmt]) -> Any:
        for statement in statements:
            result = self.statement(statement)
            if isinstance(result, _Return):
                return result.value
        return None

    def statement(self, node: ast.stmt) -> "_Return | None":
        if isinstance(node, ast.Pass):
            return None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            value = self.expression(node.value)
            self.assign(node.targets[0], value)
            return None
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            self.assign(node.target, self.expression(node.value))
            return None
        if isinstance(node, ast.AugAssign):
            current = self.expression(node.target)
            value = self.binary(node.op, current, self.expression(node.value))
            self.assign(node.target, value)
            return None
        if isinstance(node, ast.If):
            return self.run(
                node.body if self.expression(node.test) else node.orelse)
        if isinstance(node, ast.While):
            iterations = 0
            while self.expression(node.test):
                result = self.run(node.body)
                if isinstance(result, _Return):
                    return result
                iterations += 1
                if iterations > 10_000_000:
                    raise RuntimeError(
                        "kernel while loop exceeded safety limit")
            return None
        if isinstance(node, ast.Expr):
            self.expression(node.value)
            return None
        if isinstance(node, ast.Return):
            return _Return(self.expression(node.value) if node.value else None)
        raise RuntimeError(
            f"unsupported runtime statement {type(node).__name__}")

    def assign(self, target: ast.expr, value: Any) -> None:
        if isinstance(target, ast.Name):
            self.environment[target.id] = value
            return
        if isinstance(target, ast.Subscript):
            owner = self.expression(target.value)
            index = self.subscript(target.slice)
            array = owner._array if isinstance(owner, Tensor) else owner
            array[index] = value
            return
        raise RuntimeError("unsupported runtime assignment target")

    def expression(self, node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.environment:
                return self.environment[node.id]
            if node.id in self.globals and isinstance(self.globals[node.id],
                                                      (int, float, bool)):
                return self.globals[node.id]
            raise RuntimeError(f"unsupported captured value {node.id!r}")
        if isinstance(node, ast.Subscript):
            owner = self.expression(node.value)
            array = owner._array if isinstance(owner, Tensor) else owner
            return array[self.subscript(node.slice)]
        if isinstance(node, ast.BinOp):
            return self.binary(node.op, self.expression(node.left),
                               self.expression(node.right))
        if isinstance(node, ast.UnaryOp):
            value = self.expression(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.Not):
                return not value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left = self.expression(node.left)
            right = self.expression(node.comparators[0])
            operation = node.ops[0]
            if isinstance(operation, ast.Lt):
                return left < right
            if isinstance(operation, ast.LtE):
                return left <= right
            if isinstance(operation, ast.Gt):
                return left > right
            if isinstance(operation, ast.GtE):
                return left >= right
            if isinstance(operation, ast.Eq):
                return left == right
            if isinstance(operation, ast.NotEq):
                return left != right
        if isinstance(node, ast.BoolOp):
            values = [bool(self.expression(value)) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Call):
            name = Kernel._annotation_name(node.func)
            values = [self.expression(argument) for argument in node.args]
            if name in {"vec2", "vec3", "vec4"}:
                return np.asarray(
                    [
                        item for value in values
                        for item in np.asarray(value).flat
                    ],
                    dtype=np.float32,
                )
            operations = {
                "sin": np.sin,
                "cos": np.cos,
                "exp": np.exp,
                "log": np.log,
                "sqrt": np.sqrt,
                "abs": np.abs,
                "min": np.minimum,
                "max": np.maximum,
                "pow": np.power,
                "dot": np.dot,
                "norm": np.linalg.norm,
                "int": int,
                "float": float,
                "i32": np.int32,
                "u32": np.uint32,
                "f32": np.float32,
                "f64": np.float64,
            }
            if name in operations:
                return operations[name](*values)
        raise RuntimeError(f"unsupported runtime expression {ast.dump(node)}")

    @staticmethod
    def binary(operation: ast.operator, left: Any, right: Any) -> Any:
        if isinstance(operation, ast.Add):
            return left + right
        if isinstance(operation, ast.Sub):
            return left - right
        if isinstance(operation, ast.Mult):
            return left * right
        if isinstance(operation, ast.Div):
            return left / right
        if isinstance(operation, ast.Mod):
            return left % right
        if isinstance(operation, ast.Pow):
            return left**right
        raise RuntimeError(
            f"unsupported runtime binary operator {type(operation).__name__}")

    def subscript(self, node: ast.expr) -> Any:
        if isinstance(node, ast.Tuple):
            return tuple(self.expression(value) for value in node.elts)
        return self.expression(node)


@dataclass(frozen=True)
class _Return:
    value: Any


class _CapturedConstantSpecializer(ast.NodeTransformer):

    def __init__(self, globals_: dict[str, Any]):
        self.globals = globals_

    def visit_Name(self, node: ast.Name) -> ast.expr:
        value = self.globals.get(node.id)
        if isinstance(node.ctx, ast.Load) and isinstance(
                value, (int, float, bool)):
            return ast.copy_location(ast.Constant(value=value), node)
        return node
