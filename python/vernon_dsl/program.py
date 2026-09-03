"""Compilation and invocation of captured Module programs."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._runtime.tensor import TensorStorage, TensorView
from .operation_graph import GraphBuffer, GraphValueInput, KernelParameter, OperationGraph


def flatten_program_outputs(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, (GraphBuffer, TensorStorage, TensorView)):
        return {prefix or "output": value}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Any] = {}
        for field in dataclasses.fields(value):
            path = f"{prefix}.{field.name}" if prefix else field.name
            result.update(flatten_program_outputs(getattr(value, field.name), path))
        return result
    if isinstance(value, Mapping):
        result = {}
        for name, member in value.items():
            if not isinstance(name, str) or not name:
                raise TypeError("Module output mapping keys must be non-empty strings")
            path = f"{prefix}.{name}" if prefix else name
            result.update(flatten_program_outputs(member, path))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, member in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            result.update(flatten_program_outputs(member, path))
        return result
    raise TypeError(
        "Module.forward() outputs must contain only Program buffers, mappings, tuples, lists, or dataclasses"
    )


@dataclass(frozen=True)
class CapturedAlloc:
    kind: str
    result: GraphBuffer
    like: GraphBuffer | None = None
    values: Any = None


@dataclass(frozen=True)
class CapturedKernelCall:
    name: str
    kernel: Any
    arguments: tuple[Any, ...]
    grid: tuple[int, int, int]
    features: tuple[str, ...]


class ProgramCapture:
    def __init__(self, root: Any, inputs: Mapping[str, Any]):
        self.root = root
        self.inputs = dict(inputs)
        self.ops: list[CapturedAlloc | CapturedKernelCall] = []
        self._scopes = [type(root).__name__]
        self._name_counts: dict[str, int] = {}

    @property
    def calls(self) -> tuple[CapturedKernelCall, ...]:
        return tuple(op for op in self.ops if isinstance(op, CapturedKernelCall))

    @property
    def allocs(self) -> tuple[CapturedAlloc, ...]:
        return tuple(op for op in self.ops if isinstance(op, CapturedAlloc))

    def enter_module(self, name: str) -> None:
        self._scopes.append(name)

    def leave_module(self) -> None:
        self._scopes.pop()

    def capture_kernel(
        self,
        kernel: Any,
        arguments: tuple[Any, ...],
        grid: tuple[int, int, int] | None,
        features: tuple[str, ...],
    ) -> None:
        dispatch_grid = grid or (1, 1, 1)
        if len(dispatch_grid) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dispatch_grid
        ):
            raise ValueError("grid must contain three positive integers")
        base = ".".join((*self._scopes, kernel.__name__))
        ordinal = self._name_counts.get(base, 0)
        self._name_counts[base] = ordinal + 1
        name = base if ordinal == 0 else f"{base}.{ordinal}"
        self.ops.append(CapturedKernelCall(name, kernel, arguments, dispatch_grid, tuple(features)))

    def capture_allocation(
        self,
        kind: str,
        dtype: Any,
        shape: tuple[int, ...],
        *,
        like: GraphBuffer | None = None,
        values: Any = None,
    ) -> GraphBuffer:
        from .frontend.runtime_types import RuntimeParameterDescriptor

        logical = like.logical if like is not None else RuntimeParameterDescriptor.storage(dtype, tuple(shape)).logical
        result = GraphBuffer(dtype, tuple(shape), "read_write", logical=logical)
        self.ops.append(CapturedAlloc(kind, result, like, values))
        return result


@dataclass(frozen=True)
class KernelCallTemplate:
    name: str
    kernel: Any
    parameter_names: tuple[str, ...]
    parameters: tuple[KernelParameter, ...]
    grid: tuple[int, int, int]
    features: tuple[str, ...]


class ProgramTemplate:
    """Typed, reusable structure with no invocation resource references."""

    def __init__(self, calls: tuple[KernelCallTemplate, ...]):
        self.calls = calls

    @classmethod
    def compile(cls, capture: ProgramCapture) -> ProgramTemplate:
        calls: list[KernelCallTemplate] = []
        for captured in capture.calls:
            lowered = captured.kernel._lower(captured.features)
            frontend = lowered.frontend
            builtin_names = lowered.builtins
            parameter_names = tuple(
                name for name in inspect.signature(captured.kernel._function).parameters if name not in builtin_names
            )
            if len(parameter_names) != len(captured.arguments):
                raise TypeError(f"{captured.kernel.__name__} expects {len(parameter_names)} launch arguments")
            entry = next(
                (function for function in frontend.typed_functions if function.source.name == captured.kernel._entry),
                None,
            )
            if entry is None or entry.storage_activity is None:
                raise RuntimeError(f"compiled kernel {captured.kernel.__name__!r} has no typed storage activity")
            parameters: list[KernelParameter] = []
            for parameter in entry.parameters:
                if parameter.name not in parameter_names:
                    continue
                is_storage = parameter.type.kind == "tensor_view"
                access = (
                    str(parameter.type.arguments[3]) if parameter.type.kind == "texture" else parameter.access.value
                )
                parameters.append(
                    KernelParameter(
                        parameter.name,
                        access,
                        is_storage and parameter.name in entry.storage_activity.readable_roots,
                        bool(entry.storage_activity.dependencies_for(parameter.name)),
                    )
                )
            calls.append(
                KernelCallTemplate(
                    captured.name,
                    captured.kernel,
                    parameter_names,
                    tuple(parameters),
                    captured.grid,
                    captured.features,
                )
            )
        return cls(tuple(calls))

    def bind(self, capture: ProgramCapture, outputs: Any) -> ProgramInvocation:
        if len(capture.calls) != len(self.calls):
            raise RuntimeError("Module control flow changed for an existing Program specialization")
        graph = OperationGraph()
        slots: list[Any] = []
        for name, value in capture.inputs.items():
            graph.import_input(name, value)
        operation_bindings: dict[int, tuple[Any, ...]] = {}
        kernel_index = 0
        for captured in capture.ops:
            if isinstance(captured, CapturedAlloc):
                graph.append_alloc(
                    buffer=captured.result,
                    name=captured.kind,
                    like=captured.like,
                    values=captured.values,
                )
                continue
            template = self.calls[kernel_index]
            kernel_index += 1
            if (
                template.kernel is not captured.kernel
                or template.grid != captured.grid
                or template.features != captured.features
            ):
                raise RuntimeError("Module kernel sequence changed for an existing Program specialization")
            first_slot = len(slots)
            slots.extend(captured.arguments)
            bindings = dict(zip(template.parameter_names, captured.arguments, strict=True))
            binding_slots = {name: first_slot + index for index, name in enumerate(template.parameter_names)}
            operation = graph.append_kernel(
                name=template.name,
                kernel=template.kernel,
                bindings=bindings,
                binding_slots=binding_slots,
                parameters=template.parameters,
                grid=template.grid,
                features=template.features,
            )
            operation_bindings[operation.id] = captured.arguments
        graph.set_outputs(flatten_program_outputs(outputs))
        return ProgramInvocation(
            self,
            graph,
            tuple(slots),
            operation_bindings,
            dict(capture.inputs),
            {name: value.annotation for name, value in capture.inputs.items() if isinstance(value, GraphValueInput)},
            outputs,
        )


@dataclass(frozen=True)
class ProgramInvocation:
    template: ProgramTemplate
    graph: OperationGraph
    slots: tuple[Any, ...]
    operation_bindings: Mapping[int, tuple[Any, ...]]
    inputs: Mapping[str, Any]
    input_annotations: Mapping[str, Any]
    outputs: Any


class ModuleVjpExpression:
    def __init__(
        self,
        module: Any,
        *,
        wrt: tuple[str, ...],
        outputs: tuple[str, ...] | None,
        planning_policy: str,
    ):
        self.module = module
        self.wrt = wrt
        self.outputs = outputs
        self.planning_policy = planning_policy

    def __call__(self, *arguments: Any, **keywords: Any) -> tuple[Any, Any]:
        return execute_module_vjp(self, arguments, keywords)


def _capture_module(module: Any, parameter_types: Mapping[str, Any]) -> tuple[ProgramCapture, Any]:
    from .frontend.module_ast import interpret_module_forward

    return interpret_module_forward(module, parameter_types)


def _bind_forward_arguments(module: Any, arguments: tuple[Any, ...], keywords: Mapping[str, Any]) -> dict[str, Any]:
    bound = type(module)._module_definition.signature.bind(*arguments, **keywords)
    bound.apply_defaults()
    return dict(bound.arguments)


def _access_name(access: Any) -> str:
    return str(getattr(access, "name", access))


def _reflect_module_parameter_types(module: Any) -> dict[str, Any]:
    """Infer public resource types from nested typed Program calls."""

    from .module import Module
    from .types import TypeExpr, read_write

    cache: dict[int, dict[str, TypeExpr]] = {}
    active: set[int] = set()

    def merge(
        constraints: dict[str, TypeExpr],
        name: str,
        candidate: TypeExpr,
        owner: Any,
    ) -> None:
        existing = constraints.get(name)
        if existing is None:
            constraints[name] = candidate
            return
        existing_dtype, existing_shape, existing_access = existing.arguments
        dtype, shape, access = candidate.arguments
        if existing_dtype != dtype or existing_shape != shape:
            raise TypeError(
                f"conflicting reflected types for Module parameter {name!r} in {type(owner).__name__}.forward"
            )
        if existing_access != access:
            constraints[name] = TypeExpr("TensorView", (dtype, shape, read_write))

    def resolve(expression: ast.expr, owner: Any, globals_: Mapping[str, Any]) -> Any:
        if isinstance(expression, ast.Name):
            if expression.id == "self":
                return owner
            return globals_.get(expression.id)
        if isinstance(expression, ast.Attribute):
            base = resolve(expression.value, owner, globals_)
            return getattr(base, expression.attr, None) if base is not None else None
        return None

    def target_types(target: Any) -> tuple[inspect.Signature, dict[str, TypeExpr]] | None:
        if isinstance(target, Module):
            return type(target)._module_definition.signature, infer(target)
        function = getattr(target, "_function", None)
        if function is None:
            return None
        try:
            annotations = inspect.get_annotations(function, eval_str=True)
        except (NameError, TypeError):
            return None
        reflected = {
            name: annotation
            for name, annotation in annotations.items()
            if isinstance(annotation, TypeExpr) and annotation.name == "TensorView"
        }
        return inspect.signature(function), reflected

    def infer(owner: Any) -> dict[str, TypeExpr]:
        identity = id(owner)
        if identity in cache:
            return cache[identity]
        if identity in active:
            raise TypeError(f"recursive Module composition is not exportable: {type(owner).__name__}")
        active.add(identity)
        constraints: dict[str, TypeExpr] = {}
        definition = type(owner)._module_definition
        function = definition.original_function
        function_ast = definition.forward_ast
        if function_ast is None:
            active.remove(identity)
            cache[identity] = constraints
            return constraints
        parameters = set(definition.signature.parameters)
        globals_ = function.__globals__
        for call in (node for node in ast.walk(function_ast) if isinstance(node, ast.Call)):
            target = resolve(call.func, owner, globals_)
            reflected = target_types(target)
            if reflected is None:
                continue
            signature, types = reflected
            target_parameters = tuple(signature.parameters)
            bindings: list[tuple[ast.expr, str]] = [
                (argument, target_name) for argument, target_name in zip(call.args, target_parameters, strict=False)
            ]
            bindings.extend(
                (keyword.value, keyword.arg)
                for keyword in call.keywords
                if keyword.arg is not None and keyword.arg in signature.parameters
            )
            for expression, target_name in bindings:
                if not isinstance(expression, ast.Name) or expression.id not in parameters:
                    continue
                candidate = types.get(target_name)
                if candidate is not None:
                    merge(constraints, expression.id, candidate, owner)
        active.remove(identity)
        cache[identity] = constraints
        return constraints

    return infer(module)


def _module_parameter_types(module: Any) -> dict[str, Any]:
    from .frontend.model import SemanticCategory
    from .frontend.runtime_types import RuntimeParameterDescriptor, runtime_parameter_descriptor
    from .types import TypeExpr, dyn

    definition = type(module)._module_definition
    annotations = definition.resolved_annotations
    reflected = _reflect_module_parameter_types(module)
    types: dict[str, Any] = {}
    for parameter in definition.signature.parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        annotation = annotations.get(parameter.name)
        try:
            descriptor = runtime_parameter_descriptor(annotation)
        except TypeError:
            descriptor = None
        if descriptor is not None and descriptor.kind is not SemanticCategory.STORAGE:
            types[parameter.name] = descriptor
            continue
        inferred = annotation if isinstance(annotation, TypeExpr) and annotation.name == "TensorView" else None
        if inferred is None:
            inferred = reflected.get(parameter.name)
        if inferred is None:
            raise TypeError(
                f"cannot infer Module parameter {parameter.name!r}; annotate it as "
                "TensorView[element, (shape,), access], use it in a typed Program call, "
                "or supply an explicit export argument"
            )
        if len(inferred.arguments) != 3:
            raise TypeError(f"invalid TensorView annotation for Module parameter {parameter.name!r}")
        dtype, shape, access = inferred.arguments
        if not isinstance(shape, tuple) or any(extent is dyn for extent in shape):
            raise TypeError(
                f"cannot infer dynamic shape for Module parameter {parameter.name!r} without an export argument"
            )
        as_view = isinstance(annotation, TypeExpr) and annotation.name == "TensorView"
        types[parameter.name] = RuntimeParameterDescriptor.storage(
            dtype,
            tuple(shape),
            _access_name(access) if as_view else "read_write",
            as_view,
        )
    return types


def _parameter_types_from_values(
    module: Any,
    arguments: tuple[Any, ...],
    keywords: Mapping[str, Any],
) -> dict[str, Any]:
    from ._dtypes import scalar_name
    from .frontend.model import SemanticCategory
    from .frontend.runtime_types import RuntimeParameterDescriptor, runtime_parameter_descriptor, validate_host_value
    from .types import TypeExpr

    def tensor_dtype(value: TensorStorage) -> Any:
        if value._element_type is not None:
            return value._element_type
        name = scalar_name(value.dtype)
        if name is None:
            raise TypeError(f"cannot allocate a Module transient for dtype {value.dtype}")
        types = __import__("vernon_dsl.types", fromlist=[name])
        return getattr(types, name)

    def descriptor(annotation: Any, value: Any, name: str) -> RuntimeParameterDescriptor:
        as_view = isinstance(annotation, TypeExpr) and annotation.name == "TensorView"
        if isinstance(value, TensorView):
            owner = value.owner
            if not isinstance(owner, TensorStorage):
                raise TypeError("Module.forward() TensorView parameters must borrow TensorStorage")
            dtype = value.element_type if value.element_type is not None else tensor_dtype(owner)
            return RuntimeParameterDescriptor.storage(dtype, tuple(value.shape), str(value.access), True)
        if isinstance(value, TensorStorage):
            access = "read_write"
            if as_view and len(annotation.arguments) == 3:
                access = str(getattr(annotation.arguments[2], "name", annotation.arguments[2]))
            return RuntimeParameterDescriptor.storage(tensor_dtype(value), tuple(value.shape), access, as_view)
        if annotation is None:
            raise TypeError(
                f"Module.forward() argument {name!r} is a runtime Value or Resource and requires a DSL annotation"
            )
        result = runtime_parameter_descriptor(annotation)
        if result.kind is SemanticCategory.VALUE:
            validate_host_value(annotation, value, name)
        if result.kind is SemanticCategory.STORAGE:
            raise TypeError(f"Module.forward() argument {name!r} must be TensorStorage or TensorView")
        return result

    annotations = type(module)._module_definition.resolved_annotations
    types = {}
    for name, value in _bind_forward_arguments(module, arguments, keywords).items():
        types[name] = descriptor(annotations.get(name), value, name)
    return types


def _parse_module_program(
    module: Any,
    arguments: tuple[Any, ...] = (),
    keywords: Mapping[str, Any] | None = None,
    *,
    vjp_wrt: tuple[str, ...] | None = None,
    vjp_outputs: tuple[str, ...] | None = None,
    autodiff_planning_policy: str = "min_memory",
) -> Any:
    supplied_keywords = {} if keywords is None else dict(keywords)
    parameter_types = (
        _parameter_types_from_values(module, arguments, supplied_keywords)
        if arguments or supplied_keywords
        else _module_parameter_types(module)
    )
    capture, outputs = _capture_module(module, parameter_types)
    template = ProgramTemplate.compile(capture)
    invocation = template.bind(capture, outputs)
    from .program_frontend import parse_program

    selected_outputs = vjp_outputs
    if vjp_wrt is not None:
        selected_outputs = selected_outputs or tuple(invocation.graph.outputs)
        unknown_outputs = tuple(path for path in selected_outputs if path not in invocation.graph.outputs)
        if unknown_outputs:
            raise ValueError(f"Program autodiff outputs do not identify Module output paths: {unknown_outputs}")
    return parse_program(
        invocation,
        vjp_wrt=vjp_wrt,
        vjp_outputs=selected_outputs,
        autodiff_planning_policy=autodiff_planning_policy,
    )


@dataclass(frozen=True)
class _AllocationSpec:
    dtype: Any
    shape: tuple[int, ...]
    initializer: str
    values: Any = None

    def create(self) -> TensorStorage:
        if self.initializer == "from_values":
            return TensorStorage.from_values(self.values, dtype=self.dtype)
        if self.initializer in {"zeros", "zeros_like"}:
            return TensorStorage.zeros(dtype=self.dtype, shape=self.shape)
        return TensorStorage.empty(dtype=self.dtype, shape=self.shape)


@dataclass(frozen=True)
class _ValueRecipe:
    source: str
    key: str | int | None = None
    shape: tuple[int, ...] | None = None
    strides: tuple[int, ...] | None = None
    offset: int = 0
    access: str = "read_write"
    dtype: Any = None
    element_type: Any = None
    constant: Any = None
    as_view: bool = False

    def resolve(self, inputs: Mapping[str, Any], allocations: tuple[TensorStorage, ...]) -> Any:
        if self.source == "constant":
            return self.constant
        if self.source == "input":
            if not isinstance(self.key, str):
                raise ValueError("input tree recipe requires a parameter name")
            owner = inputs[self.key]
        else:
            if not isinstance(self.key, int):
                raise ValueError("allocation tree recipe requires an allocation index")
            owner = allocations[self.key]
        if isinstance(owner, TensorView):
            owner = owner.owner
        if self.as_view:
            view = getattr(owner, "view", None)
            if not callable(view):
                raise TypeError("tree recipe view owner does not support TensorView projection")
            return view(access=self.access)
        return owner


@dataclass(frozen=True)
class _TreeRecipe:
    kind: str
    value: Any

    def resolve(self, inputs: Mapping[str, Any], allocations: tuple[TensorStorage, ...]) -> Any:
        if self.kind == "leaf":
            return self.value.resolve(inputs, allocations)
        if self.kind == "mapping":
            return {name: member.resolve(inputs, allocations) for name, member in self.value}
        if self.kind == "tuple":
            return tuple(member.resolve(inputs, allocations) for member in self.value)
        if self.kind == "list":
            return [member.resolve(inputs, allocations) for member in self.value]
        if self.kind == "dataclass":
            cls, fields = self.value
            return cls(**{name: member.resolve(inputs, allocations) for name, member in fields})
        raise RuntimeError(f"unknown Program output recipe {self.kind!r}")


@dataclass(frozen=True)
class _PrimalSpecialization:
    program: Any
    template: ProgramTemplate
    invocation: ProgramInvocation
    signature: inspect.Signature
    allocations: tuple[_AllocationSpec, ...]
    calls: tuple[tuple[Any, tuple[_ValueRecipe, ...], tuple[int, int, int], tuple[str, ...]], ...]
    outputs: _TreeRecipe
    native_program: Any

    def invoke(self, arguments: tuple[Any, ...], keywords: Mapping[str, Any]) -> Any:
        inputs = self.signature.bind(*arguments, **keywords).arguments
        allocations = tuple(spec.create() for spec in self.allocations)
        outputs = self.outputs.resolve(inputs, allocations)
        return self.native_program.invoke(
            dataclasses.replace(
                self.invocation,
                inputs=dict(inputs),
                outputs=outputs,
            )
        )


def _capture_recipes(
    capture: ProgramCapture,
    outputs: Any,
) -> tuple[
    tuple[_AllocationSpec, ...],
    tuple[tuple[Any, tuple[_ValueRecipe, ...], tuple[int, int, int], tuple[str, ...]], ...],
    _TreeRecipe,
]:
    input_ids = {id(value): name for name, value in capture.inputs.items()}
    alloc_ids = {id(alloc.result): index for index, alloc in enumerate(capture.allocs)}

    def recipe(value: Any) -> _ValueRecipe:
        if isinstance(value, GraphBuffer):
            if id(value) in input_ids:
                source, key = "input", input_ids[id(value)]
            elif id(value) in alloc_ids:
                source, key = "allocation", alloc_ids[id(value)]
            else:
                raise TypeError(
                    "Module.forward() captured an unregistered Program buffer; "
                    "use vd.empty(), vd.zeros(), vd.from_values(), or an invocation parameter"
                )
            return _ValueRecipe(source, key, access=value.access, as_view=value.as_view)
        for name, supplied in capture.inputs.items():
            if value is supplied:
                return _ValueRecipe("input", name)
        return _ValueRecipe("constant", constant=value)

    def tree(value: Any) -> _TreeRecipe:
        if isinstance(value, (GraphBuffer, TensorStorage, TensorView)):
            return _TreeRecipe("leaf", recipe(value))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return _TreeRecipe(
                "dataclass",
                (
                    type(value),
                    tuple((field.name, tree(getattr(value, field.name))) for field in dataclasses.fields(value)),
                ),
            )
        if isinstance(value, Mapping):
            return _TreeRecipe("mapping", tuple((name, tree(member)) for name, member in value.items()))
        if isinstance(value, tuple):
            return _TreeRecipe("tuple", tuple(tree(member) for member in value))
        if isinstance(value, list):
            return _TreeRecipe("list", tuple(tree(member) for member in value))
        raise TypeError("Module output contains a value that cannot be reconstructed from Program IR")

    allocations = tuple(
        _AllocationSpec(alloc.result.dtype, tuple(alloc.result.shape), alloc.kind, alloc.values)
        for alloc in capture.allocs
    )
    calls = tuple(
        (
            call.kernel,
            tuple(recipe(value) for value in call.arguments),
            call.grid,
            call.features,
        )
        for call in capture.calls
    )
    return allocations, calls, tree(outputs)


def _primal_specialization(
    module: Any,
    capture: ProgramCapture,
    outputs: Any,
    invocation: ProgramInvocation,
) -> _PrimalSpecialization:
    allocations, calls, output_recipe = _capture_recipes(capture, outputs)
    from .program_frontend import parse_program

    parsed = parse_program(invocation)
    from ._runtime.program_autodiff import compile_program

    return _PrimalSpecialization(
        parsed,
        invocation.template,
        invocation,
        type(module)._module_definition.signature,
        allocations,
        calls,
        output_recipe,
        compile_program(parsed, invocation.template),
    )


@dataclass(frozen=True)
class _VjpSpecialization:
    compiled: Any
    invocation: ProgramInvocation
    signature: inspect.Signature
    allocations: tuple[_AllocationSpec, ...]
    outputs: _TreeRecipe

    @property
    def template(self) -> ProgramTemplate:
        return self.compiled.template

    @property
    def pipeline(self) -> Any:
        return self.compiled.pipeline

    def invoke(
        self,
        arguments: tuple[Any, ...],
        keywords: Mapping[str, Any],
        *,
        checkpoint_memory_budget: int | None,
        checkpoint_policy: str,
    ) -> tuple[Any, Any]:
        bound = self.signature.bind(*arguments, **keywords)
        bound.apply_defaults()
        inputs = dict(bound.arguments)
        allocations = tuple(spec.create() for spec in self.allocations)
        invocation = dataclasses.replace(
            self.invocation,
            inputs=inputs,
            outputs=self.outputs.resolve(inputs, allocations),
        )
        return self.compiled.invoke(
            invocation,
            checkpoint_memory_budget=checkpoint_memory_budget,
            checkpoint_policy=checkpoint_policy,
        )


def execute_module_primal(module: Any, arguments: tuple[Any, ...], keywords: Mapping[str, Any]) -> Any:
    key = module._program_specialization_key(arguments, keywords, variant="primal")
    specialization = module._program_cache.get(key)
    if not isinstance(specialization, _PrimalSpecialization):
        capture, outputs = _capture_module(module, _parameter_types_from_values(module, arguments, keywords))
        template = ProgramTemplate.compile(capture)
        invocation = template.bind(capture, outputs)
        specialization = _primal_specialization(module, capture, outputs, invocation)
        module._program_cache[key] = specialization
    return specialization.invoke(arguments, keywords)


def execute_module_vjp(
    expression: ModuleVjpExpression,
    arguments: tuple[Any, ...],
    keywords: Mapping[str, Any],
) -> tuple[Any, Any]:
    module = expression.module
    key = module._program_specialization_key(
        arguments,
        keywords,
        variant=(
            "vjp",
            expression.wrt,
            expression.outputs,
            expression.planning_policy,
        ),
    )
    specialization = module._program_cache.get(key)
    if not isinstance(specialization, _VjpSpecialization):
        capture, outputs = _capture_module(
            module,
            _parameter_types_from_values(module, arguments, keywords),
        )
        template = ProgramTemplate.compile(capture)
        invocation = template.bind(capture, outputs)
        selected_outputs = expression.outputs or tuple(invocation.graph.outputs)
        unknown_outputs = tuple(path for path in selected_outputs if path not in invocation.graph.outputs)
        if unknown_outputs:
            raise ValueError(f"Program autodiff outputs do not identify Module output paths: {unknown_outputs}")
        from .program_frontend import parse_program

        parsed_program = parse_program(
            invocation,
            vjp_wrt=expression.wrt,
            vjp_outputs=selected_outputs,
            autodiff_planning_policy=expression.planning_policy,
        )
        from ._runtime.program_autodiff import compile_program_autodiff

        compiled = compile_program_autodiff(parsed_program, template)
        allocations, _, output_recipe = _capture_recipes(capture, outputs)
        specialization = _VjpSpecialization(
            compiled,
            invocation,
            type(module)._module_definition.signature,
            allocations,
            output_recipe,
        )
        module._program_cache[key] = specialization
    budget = getattr(module, "checkpoint_memory_budget", None)
    if not isinstance(budget, int):
        budget = None
    return specialization.invoke(
        arguments,
        keywords,
        checkpoint_memory_budget=budget,
        checkpoint_policy=expression.planning_policy if budget is not None else "",
    )


__all__ = [
    "ModuleVjpExpression",
    "ProgramTemplate",
    "execute_module_primal",
]
