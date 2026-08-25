"""Partial evaluation of Module.forward from AST, with Module instance as comptime."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import operator
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..module import Module
from ..operation_graph import GraphBuffer
from ..storage import TensorStorage, _logical_collection_shape
from ..storage import empty as storage_empty
from ..storage import empty_like as storage_empty_like
from ..storage import from_numpy as storage_from_numpy
from ..storage import from_values as storage_from_values
from ..storage import tangent_zeros as storage_tangent_zeros
from ..storage import zeros as storage_zeros
from ..storage import zeros_like as storage_zeros_like

_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
}

_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}

_PROGRAM_ALLOCS = {
    id(storage_empty): "empty",
    id(storage_empty_like): "empty_like",
    id(storage_zeros): "zeros",
    id(storage_zeros_like): "zeros_like",
    id(storage_from_values): "from_values",
}
_HOST_STORAGE = {
    id(storage_from_numpy),
    id(storage_tangent_zeros),
    id(TensorStorage.from_numpy),
    id(TensorStorage.tangent_zeros),
}


class _ForwardReturn(Exception):
    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value


@dataclass(frozen=True)
class ModuleParameterType:
    dtype: Any
    shape: tuple[int, ...]
    access: str = "read_write"
    as_view: bool = False


def interpret_module_forward(
    module: Module,
    parameter_types: Mapping[str, ModuleParameterType],
) -> tuple[Any, Any]:
    from ..program import ProgramCapture

    inputs = {
        name: GraphBuffer(parameter_type.dtype, parameter_type.shape, parameter_type.access, parameter_type.as_view)
        for name, parameter_type in parameter_types.items()
    }
    capture = ProgramCapture(module, inputs)
    interpreter = _ForwardInterpreter(capture, module)
    outputs = interpreter.interpret(module, dict(inputs))
    return capture, outputs


def _is_graph(value: Any) -> bool:
    return isinstance(value, GraphBuffer)


def _require_graph(value: Any, *, what: str) -> GraphBuffer:
    if not isinstance(value, GraphBuffer):
        raise TypeError(f"{what} requires a Program buffer")
    return value


def _contains_graph(value: Any) -> bool:
    if _is_graph(value):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_graph(member) for member in value)
    if isinstance(value, Mapping):
        return any(_contains_graph(member) for member in value.values())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(_contains_graph(getattr(value, field.name)) for field in dataclasses.fields(value))
    return False


def _require_comptime(value: Any, *, what: str) -> Any:
    if _contains_graph(value):
        raise TypeError(f"Module.forward() {what} must be host-static; got a Program Storage or TensorView")
    return value


def _forward_function(module: Module) -> Any:
    function = module.forward
    return getattr(function, "__func__", function)


def _parse_forward(module: Module) -> ast.FunctionDef:
    function = _forward_function(module)
    try:
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as error:
        raise TypeError(f"cannot parse {type(module).__name__}.forward: {error}") from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise TypeError(f"{type(module).__name__}.forward must be a function definition")
    return tree.body[0]


def _is_kernel(value: Any) -> bool:
    kind = getattr(value, "__vernon_dsl__", (None,))[0]
    return kind == "compute" and hasattr(value, "_lower") and hasattr(value, "_function")


class _ForwardInterpreter:
    def __init__(self, capture: Any, root: Module) -> None:
        self.capture = capture
        self.root = root

    def interpret(self, owner: Module, inputs: Mapping[str, Any]) -> Any:
        function = _parse_forward(owner)
        locals_: dict[str, Any] = dict(inputs)
        try:
            self._body(function.body, owner, locals_, _forward_function(owner).__globals__)
        except _ForwardReturn as returned:
            return returned.value
        raise TypeError(f"{type(owner).__name__}.forward must return a value")

    def _body(
        self,
        statements: list[ast.stmt],
        owner: Module,
        locals_: dict[str, Any],
        globals_: Mapping[str, Any],
    ) -> None:
        for statement in statements:
            self._statement(statement, owner, locals_, globals_)

    def _statement(
        self,
        statement: ast.stmt,
        owner: Module,
        locals_: dict[str, Any],
        globals_: Mapping[str, Any],
    ) -> None:
        if isinstance(statement, ast.Return):
            if statement.value is None:
                raise TypeError(f"{type(owner).__name__}.forward must return a value")
            raise _ForwardReturn(self._expr(statement.value, owner, locals_, globals_))
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            if isinstance(statement, ast.AnnAssign):
                if statement.value is None or not isinstance(statement.target, ast.Name):
                    raise TypeError("Module.forward() annotated assignment must bind a name")
                self._assign(statement.target, self._expr(statement.value, owner, locals_, globals_), locals_)
                return
            value = self._expr(statement.value, owner, locals_, globals_)
            for target in statement.targets:
                self._assign(target, value, locals_)
            return
        if isinstance(statement, ast.AugAssign):
            raise TypeError("Module.forward() does not support augmented assignment")
        if isinstance(statement, ast.If):
            test = _require_comptime(self._expr(statement.test, owner, locals_, globals_), what="if condition")
            branch = statement.body if test else statement.orelse
            self._body(branch, owner, locals_, globals_)
            return
        if isinstance(statement, ast.For):
            iterable = _require_comptime(self._expr(statement.iter, owner, locals_, globals_), what="for iterable")
            if statement.orelse:
                raise TypeError("Module.forward() for-loops cannot have else clauses")
            try:
                sequence = list(iterable)
            except TypeError as error:
                raise TypeError("Module.forward() for-loops require a host-static iterable") from error
            for item in sequence:
                self._assign(statement.target, item, locals_)
                self._body(statement.body, owner, locals_, globals_)
            return
        if isinstance(statement, ast.Expr):
            if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                return
            self._expr(statement.value, owner, locals_, globals_)
            return
        if isinstance(statement, ast.Pass):
            return
        if isinstance(statement, ast.Assert):
            test = _require_comptime(self._expr(statement.test, owner, locals_, globals_), what="assert")
            if test:
                return
            message = ""
            if statement.msg is not None:
                message = str(
                    _require_comptime(self._expr(statement.msg, owner, locals_, globals_), what="assert message")
                )
            raise AssertionError(message)
        if isinstance(statement, ast.Raise):
            exception = (
                self._expr(statement.exc, owner, locals_, globals_) if statement.exc is not None else RuntimeError()
            )
            _require_comptime(exception, what="raise")
            if isinstance(exception, BaseException):
                raise exception
            raise exception
        raise TypeError(f"Module.forward() does not support {type(statement).__name__} statements")

    def _assign(self, target: ast.expr, value: Any, locals_: dict[str, Any]) -> None:
        if isinstance(target, ast.Name):
            locals_[target.id] = value
            return
        if isinstance(target, ast.Tuple):
            if any(isinstance(element, ast.Starred) for element in target.elts):
                raise TypeError("Module.forward() does not support starred assignment")
            values = tuple(value)
            if len(target.elts) != len(values):
                raise TypeError("Module.forward() tuple assignment length mismatch")
            for element, member in zip(target.elts, values, strict=True):
                self._assign(element, member, locals_)
            return
        raise TypeError("Module.forward() can only assign to names and tuples")

    def _expr(
        self,
        expression: ast.expr,
        owner: Module,
        locals_: dict[str, Any],
        globals_: Mapping[str, Any],
    ) -> Any:
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, ast.Name):
            if expression.id in locals_:
                return locals_[expression.id]
            if expression.id == "self":
                return owner
            if expression.id in globals_:
                return globals_[expression.id]
            if expression.id in ("range", "len", "tuple", "list", "int", "bool", "float"):
                return getattr(__import__("builtins"), expression.id)
            raise NameError(f"Module.forward() name {expression.id!r} is not defined")
        if isinstance(expression, ast.Attribute):
            if isinstance(expression.value, ast.Name) and expression.value.id == "self":
                return getattr(owner, expression.attr)
            base = self._expr(expression.value, owner, locals_, globals_)
            if _is_graph(base):
                raise TypeError(
                    f"Module.forward() cannot read attribute {expression.attr!r} from Program Storage; "
                    "use vd.empty_like()/vd.zeros_like() or a host-static Module attribute"
                )
            return getattr(base, expression.attr)
        if isinstance(expression, ast.Tuple):
            return tuple(self._expr(element, owner, locals_, globals_) for element in expression.elts)
        if isinstance(expression, ast.List):
            return [self._expr(element, owner, locals_, globals_) for element in expression.elts]
        if isinstance(expression, ast.UnaryOp):
            operation = _UNARY.get(type(expression.op))
            if operation is None:
                raise TypeError(f"Module.forward() does not support {type(expression.op).__name__}")
            operand = _require_comptime(self._expr(expression.operand, owner, locals_, globals_), what="unary operand")
            return operation(operand)
        if isinstance(expression, ast.BinOp):
            operation = _BINOPS.get(type(expression.op))
            if operation is None:
                raise TypeError(f"Module.forward() does not support {type(expression.op).__name__}")
            left = _require_comptime(self._expr(expression.left, owner, locals_, globals_), what="binary operand")
            right = _require_comptime(self._expr(expression.right, owner, locals_, globals_), what="binary operand")
            return operation(left, right)
        if isinstance(expression, ast.Compare):
            left = _require_comptime(self._expr(expression.left, owner, locals_, globals_), what="comparison")
            for operation, comparator in zip(expression.ops, expression.comparators, strict=True):
                compare = _COMPARE.get(type(operation))
                if compare is None:
                    raise TypeError(f"Module.forward() does not support {type(operation).__name__} comparisons")
                right = _require_comptime(self._expr(comparator, owner, locals_, globals_), what="comparison")
                if not compare(left, right):
                    return False
                left = right
            return True
        if isinstance(expression, ast.BoolOp):
            values = [self._expr(value, owner, locals_, globals_) for value in expression.values]
            for value in values:
                _require_comptime(value, what="boolean operand")
            if isinstance(expression.op, ast.And):
                result: Any = True
                for value in values:
                    result = result and value
                    if not result:
                        return result
                return result
            result = False
            for value in values:
                result = result or value
                if result:
                    return result
            return result
        if isinstance(expression, ast.IfExp):
            test = _require_comptime(self._expr(expression.test, owner, locals_, globals_), what="ternary condition")
            return self._expr(expression.body if test else expression.orelse, owner, locals_, globals_)
        if isinstance(expression, ast.Subscript):
            value = self._expr(expression.value, owner, locals_, globals_)
            index = self._expr(expression.slice, owner, locals_, globals_)
            if _is_graph(value):
                raise TypeError("Module.forward() cannot index Program Storage; use a Kernel")
            _require_comptime(index, what="subscript")
            return value[index]
        if isinstance(expression, ast.Slice):
            lower = None if expression.lower is None else self._expr(expression.lower, owner, locals_, globals_)
            upper = None if expression.upper is None else self._expr(expression.upper, owner, locals_, globals_)
            step = None if expression.step is None else self._expr(expression.step, owner, locals_, globals_)
            return slice(lower, upper, step)
        if isinstance(expression, ast.Call):
            return self._call(expression, owner, locals_, globals_)
        raise TypeError(f"Module.forward() does not support {type(expression).__name__} expressions")

    def _call(
        self,
        expression: ast.Call,
        owner: Module,
        locals_: dict[str, Any],
        globals_: Mapping[str, Any],
    ) -> Any:
        if any(isinstance(argument, ast.Starred) for argument in expression.args) or any(
            keyword.arg is None for keyword in expression.keywords
        ):
            raise TypeError("Module.forward() does not support *args or **kwargs")
        attr_name = (
            expression.func.attr
            if isinstance(expression.func, ast.Attribute)
            and isinstance(expression.func.value, ast.Name)
            and expression.func.value.id == "self"
            else None
        )
        callee = self._expr(expression.func, owner, locals_, globals_)
        args = [self._expr(argument, owner, locals_, globals_) for argument in expression.args]
        keywords = {
            keyword.arg: self._expr(keyword.value, owner, locals_, globals_)
            for keyword in expression.keywords
            if keyword.arg is not None
        }
        if id(callee) in _HOST_STORAGE:
            raise TypeError(
                "vd.storage.from_numpy() and vd.storage.tangent_zeros() are host session operations and cannot "
                "appear in Module.forward(); pass Storage as a forward argument or use vd.from_values for constants"
            )
        allocator = _PROGRAM_ALLOCS.get(id(callee))
        if allocator is not None:
            return self._allocate(allocator, args, keywords)
        if isinstance(callee, Module):
            child_name = attr_name if attr_name is not None else type(callee).__name__
            bound = inspect.signature(callee.forward).bind(*args, **keywords)
            bound.apply_defaults()
            self.capture.enter_module(child_name)
            try:
                return self.interpret(callee, dict(bound.arguments))
            finally:
                self.capture.leave_module()
        if _is_kernel(callee):
            grid = keywords.pop("grid", None)
            features = keywords.pop("features", ())
            if keywords:
                raise TypeError(f"{callee.__name__} got unexpected Module.forward() keywords {sorted(keywords)}")
            grid_value = None if grid is None else _require_comptime(grid, what="grid")
            feature_value = _require_comptime(features, what="features")
            self.capture.capture_kernel(
                callee,
                tuple(args),
                grid_value,
                tuple(feature_value),
            )
            return None
        if dataclasses.is_dataclass(callee) and isinstance(callee, type):
            return callee(*args, **keywords)
        if any(_contains_graph(value) for value in (*args, *keywords.values())):
            raise TypeError(
                f"{getattr(callee, '__name__', type(callee).__name__)} cannot take Program Storage arguments "
                "in Module.forward(); call a Kernel, child Module, or vd.empty/vd.zeros/vd.from_values"
            )
        if not callable(callee):
            raise TypeError(f"Module.forward() tried to call non-callable {callee!r}")
        return callee(*args, **keywords)

    def _allocate(self, kind: str, args: list[Any], keywords: dict[str, Any]) -> GraphBuffer:
        if kind in {"empty_like", "zeros_like"}:
            if keywords:
                raise TypeError(f"vd.{kind}() does not take keywords")
            if len(args) != 1:
                raise TypeError(f"vd.{kind}() takes one Storage argument")
            source = _require_graph(args[0], what=f"vd.{kind}()")
            return self.capture.capture_allocation(kind, source.dtype, source.shape, like=source)
        if kind in {"empty", "zeros"}:
            if args:
                raise TypeError(f"vd.{kind}() takes only dtype= and shape=")
            dtype = _require_comptime(keywords.pop("dtype", None), what="dtype")
            shape = _require_comptime(keywords.pop("shape", None), what="shape")
            if keywords or dtype is None or shape is None:
                raise TypeError(f"vd.{kind}() requires dtype= and shape=")
            return self.capture.capture_allocation(kind, dtype, tuple(shape))
        if kind == "from_values":
            if len(args) != 1:
                raise TypeError("vd.from_values() takes values and dtype=")
            values = _require_comptime(args[0], what="from_values values")
            dtype = _require_comptime(keywords.pop("dtype", None), what="dtype")
            if keywords or dtype is None:
                raise TypeError("vd.from_values() requires dtype=")
            shape = tuple(_logical_collection_shape(values, dtype))
            return self.capture.capture_allocation("from_values", dtype, shape, values=values)
        raise RuntimeError(f"unknown Program allocation {kind!r}")


__all__ = ["ModuleParameterType", "interpret_module_forward"]
