"""Compilation and invocation of captured Module programs."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._runtime.resources import TensorStorage, TensorView
from .frontend.capture import capture_scope
from .operation_graph import KernelParameter, OperationGraph


def flatten_program_outputs(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, (TensorStorage, TensorView)):
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
    raise TypeError("Module.forward() outputs must contain only tensors, mappings, tuples, lists, or dataclasses")


@dataclass(frozen=True)
class CapturedKernelCall:
    name: str
    kernel: Any
    arguments: tuple[Any, ...]
    grid: tuple[int, int, int]
    features: tuple[str, ...]


class ProgramCapture:
    def __init__(self, root: Any, inputs: Mapping[str, Any], *, consume_kernels: bool):
        self.root = root
        self.inputs = dict(inputs)
        self.consume_kernels = consume_kernels
        self.calls: list[CapturedKernelCall] = []
        self.allocations: dict[int, tuple[TensorStorage, str]] = {}
        self._scopes = [type(root).__name__]
        self._name_counts: dict[str, int] = {}

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
    ) -> bool:
        dispatch_grid = grid or (1, 1, 1)
        if len(dispatch_grid) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dispatch_grid
        ):
            raise ValueError("grid must contain three positive integers")
        base = ".".join((*self._scopes, kernel.__name__))
        ordinal = self._name_counts.get(base, 0)
        self._name_counts[base] = ordinal + 1
        name = base if ordinal == 0 else f"{base}.{ordinal}"
        self.calls.append(CapturedKernelCall(name, kernel, arguments, dispatch_grid, tuple(features)))
        return self.consume_kernels

    def capture_allocation(self, value: Any, initializer: str) -> None:
        if not isinstance(value, TensorStorage):
            raise TypeError("Program allocations must produce TensorStorage")
        self.allocations[id(value)] = (value, initializer)


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
            compiled = captured.kernel._compile(captured.arguments, captured.features)
            parameter_names = tuple(
                name
                for name in inspect.signature(captured.kernel._function).parameters
                if name not in compiled.builtin_names
            )
            if len(parameter_names) != len(captured.arguments):
                raise TypeError(f"{captured.kernel.__name__} expects {len(parameter_names)} launch arguments")
            entry = next(
                (
                    function
                    for function in compiled.frontend.typed_functions
                    if function.source.name == captured.kernel._entry
                ),
                None,
            )
            if entry is None or entry.storage_activity is None:
                raise RuntimeError(f"compiled kernel {captured.kernel.__name__!r} has no typed storage activity")
            parameters: list[KernelParameter] = []
            for parameter in entry.parameters:
                if parameter.name not in parameter_names:
                    continue
                is_storage = parameter.type.kind == "tensor_view"
                parameters.append(
                    KernelParameter(
                        parameter.name,
                        parameter.access.value,
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

    @classmethod
    def plan(cls, capture: ProgramCapture) -> ProgramTemplate:
        """Build Program operation metadata without producing target executables."""

        calls: list[KernelCallTemplate] = []
        for captured in capture.calls:
            frontend, _, builtin_names, _ = captured.kernel._lower(captured.arguments, captured.features)
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
            parameters = tuple(
                KernelParameter(
                    parameter.name,
                    parameter.access.value,
                    parameter.type.kind == "tensor_view" and parameter.name in entry.storage_activity.readable_roots,
                    bool(entry.storage_activity.dependencies_for(parameter.name)),
                )
                for parameter in entry.parameters
                if parameter.name in parameter_names
            )
            calls.append(
                KernelCallTemplate(
                    captured.name,
                    captured.kernel,
                    parameter_names,
                    parameters,
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
        for template, captured in zip(self.calls, capture.calls, strict=True):
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
            outputs,
        )


@dataclass(frozen=True)
class ProgramInvocation:
    template: ProgramTemplate
    graph: OperationGraph
    slots: tuple[Any, ...]
    operation_bindings: Mapping[int, tuple[Any, ...]]
    inputs: Mapping[str, Any]
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


def _capture_module(
    module: Any,
    arguments: tuple[Any, ...],
    keywords: Mapping[str, Any],
    *,
    consume_kernels: bool,
) -> tuple[ProgramCapture, Any]:
    inputs = dict(inspect.signature(module.forward).bind(*arguments, **keywords).arguments)
    capture = ProgramCapture(module, inputs, consume_kernels=consume_kernels)
    with capture_scope(capture):
        outputs = module.forward(*arguments, **keywords)
    return capture, outputs


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
            return inspect.signature(target.forward), infer(target)
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
        function = owner.forward.__func__
        try:
            source = textwrap.dedent(inspect.getsource(function))
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            active.remove(identity)
            cache[identity] = constraints
            return constraints
        parameters = set(inspect.signature(owner.forward).parameters)
        globals_ = function.__globals__
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
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


def _inferred_module_arguments(module: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    from .types import TypeExpr, dyn

    try:
        annotations = inspect.get_annotations(module.forward, eval_str=True)
    except (NameError, TypeError) as error:
        raise TypeError(f"cannot resolve {type(module).__name__}.forward annotations: {error}") from None
    positional: list[Any] = []
    keywords: dict[str, Any] = {}
    reflected = _reflect_module_parameter_types(module)
    for parameter in inspect.signature(module.forward).parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        annotation = annotations.get(parameter.name)
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
        access_name = getattr(access, "name", access)
        storage = TensorStorage.zeros(dtype=dtype, shape=tuple(shape))
        value = (
            storage.view(access=str(access_name))
            if isinstance(annotation, TypeExpr) and annotation.name == "TensorView"
            else storage
        )
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            keywords[parameter.name] = value
    return tuple(positional), keywords


def _parse_module_program(
    module: Any,
    arguments: tuple[Any, ...] = (),
    keywords: Mapping[str, Any] | None = None,
) -> Any:
    supplied_keywords = {} if keywords is None else dict(keywords)
    if not arguments and not supplied_keywords:
        arguments, supplied_keywords = _inferred_module_arguments(module)
    capture, outputs = _capture_module(
        module,
        arguments,
        supplied_keywords,
        consume_kernels=True,
    )
    template = ProgramTemplate.compile(capture)
    invocation = template.bind(capture, outputs)
    allocation_initializers = {
        invocation.graph.owner_id(value): initializer for value, initializer in capture.allocations.values()
    }
    from .program_frontend import parse_program

    return parse_program(invocation, allocation_initializers=allocation_initializers)


def _plan_module_program(module: Any) -> Any:
    arguments, keywords = _inferred_module_arguments(module)
    capture, outputs = _capture_module(module, arguments, keywords, consume_kernels=True)
    template = ProgramTemplate.plan(capture)
    invocation = template.bind(capture, outputs)
    allocation_initializers = {
        invocation.graph.owner_id(value): initializer for value, initializer in capture.allocations.values()
    }
    from .program_frontend import parse_program

    return parse_program(invocation, allocation_initializers=allocation_initializers)


@dataclass(frozen=True)
class _AllocationSpec:
    dtype: Any
    shape: tuple[int, ...]
    initializer: str

    def create(self) -> TensorStorage:
        factory = TensorStorage.zeros if self.initializer == "zeros" else TensorStorage.empty
        return factory(dtype=self.dtype, shape=self.shape)


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

    def resolve(self, inputs: Mapping[str, Any], allocations: tuple[TensorStorage, ...]) -> Any:
        if self.source == "constant":
            return self.constant
        owner = inputs[self.key] if self.source == "input" else allocations[int(self.key)]
        if self.shape is None:
            return owner
        if isinstance(owner, TensorView):
            owner = owner.owner
        return TensorView(
            owner,
            self.shape,
            self.strides or (),
            self.offset,
            self.access,
            dtype=self.dtype,
            element_type=self.element_type,
        )


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
class _NativePrimalPlan:
    executable: Any
    parameters: tuple[tuple[Any, ...], ...]
    compiled: tuple[Any, ...]
    user_parameters: tuple[tuple[str, ...], ...]
    binding_caches: tuple[Any, ...]

    def invoke(self, calls: tuple[tuple[Any, tuple[Any, ...]], ...]) -> None:
        from ._runtime import session
        from ._runtime.resources import TensorStorage, TensorView, _TextureResource

        prepared: dict[Any, Any] = {}
        borrows: list[tuple[str, Any, str]] = []
        access_names = {
            session._native.ACCESS_READ: "read",
            session._native.ACCESS_WRITE: "write",
            session._native.ACCESS_READ_WRITE: "read_write",
        }
        for index, ((kernel, arguments), parameters, compiled, user_names, cache) in enumerate(
            zip(
                calls,
                self.parameters,
                self.compiled,
                self.user_parameters,
                self.binding_caches,
                strict=True,
            )
        ):
            with cache.invocation(compiled.native) as builder:
                kernel._bind_direct_arguments(
                    compiled,
                    builder,
                    arguments,
                    list(user_names),
                    cache,
                )
            for graph_parameter, native_parameter in zip(parameters, compiled.native.parameters, strict=True):
                prepared[graph_parameter] = cache._prepared[native_parameter.slot][1]
                value = arguments[user_names.index(native_parameter.name)]
                if isinstance(value, (TensorStorage, TensorView, _TextureResource)):
                    borrows.append(
                        (
                            f"{index}.{native_parameter.name}",
                            value,
                            access_names[native_parameter.access],
                        )
                    )
        self.executable._submit_pipeline_bindings(prepared, borrows).wait()
        if session._architecture != session.cpu:
            for (_, arguments), compiled, user_names in zip(
                calls,
                self.compiled,
                self.user_parameters,
                strict=True,
            ):
                for native_parameter in compiled.native.parameters:
                    value = arguments[user_names.index(native_parameter.name)]
                    if native_parameter.access in {
                        session._native.ACCESS_WRITE,
                        session._native.ACCESS_READ_WRITE,
                    } and isinstance(value, (TensorStorage, TensorView, _TextureResource)):
                        value._mark_device_dirty()


@dataclass(frozen=True)
class _PrimalSpecialization:
    program: Any
    template: ProgramTemplate
    signature: inspect.Signature
    allocations: tuple[_AllocationSpec, ...]
    calls: tuple[tuple[Any, tuple[_ValueRecipe, ...], tuple[int, int, int], tuple[str, ...]], ...]
    outputs: _TreeRecipe
    native_plan: _NativePrimalPlan | None = None

    def invoke(self, arguments: tuple[Any, ...], keywords: Mapping[str, Any]) -> Any:
        inputs = self.signature.bind(*arguments, **keywords).arguments
        allocations = tuple(spec.create() for spec in self.allocations)
        resolved_calls = tuple(
            (
                kernel,
                tuple(recipe.resolve(inputs, allocations) for recipe in recipes),
            )
            for kernel, recipes, _, _ in self.calls
        )
        if self.native_plan is not None:
            self.native_plan.invoke(resolved_calls)
            return self.outputs.resolve(inputs, allocations)
        for kernel, recipes, grid, features in self.calls:
            kernel(
                *(recipe.resolve(inputs, allocations) for recipe in recipes),
                grid=grid,
                features=features,
            )
        return self.outputs.resolve(inputs, allocations)


def _lower_native_primal(capture: ProgramCapture) -> _NativePrimalPlan | None:
    from ._runtime import session

    if session._architecture == session.cpu:
        return None
    from ._runtime.execution_graph import ExecutionGraph, _NativePipelineComputePass
    from ._runtime.resources import _NativeBindingCache

    graph = ExecutionGraph()
    parameter_rows: list[tuple[Any, ...]] = []
    compiled_rows: list[Any] = []
    user_parameter_rows: list[tuple[str, ...]] = []
    caches: list[Any] = []
    previous = None
    for operation_index, call in enumerate(capture.calls):
        compiled = call.kernel._compile(call.arguments, call.features)
        if compiled.native is None:
            raise RuntimeError(f"KernelCallOp {call.name!r} did not produce a native pipeline")
        user_parameters = tuple(
            name for name in inspect.signature(call.kernel._function).parameters if name not in compiled.builtin_names
        )
        parameters = tuple(
            graph.parameter(f"{operation_index}.{parameter.name}") for parameter in compiled.native.parameters
        )
        execution_pass = _NativePipelineComputePass(
            call.name,
            compiled.native,
            parameters,
            call.grid,
        )
        if previous is not None:
            execution_pass.depends_on(previous)
        graph.add_pass(execution_pass)
        previous = execution_pass
        parameter_rows.append(parameters)
        compiled_rows.append(compiled)
        user_parameter_rows.append(user_parameters)
        caches.append(_NativeBindingCache())
    if not capture.calls:
        return None
    return _NativePrimalPlan(
        graph.compile(),
        tuple(parameter_rows),
        tuple(compiled_rows),
        tuple(user_parameter_rows),
        tuple(caches),
    )


def _primal_specialization(
    module: Any,
    capture: ProgramCapture,
    outputs: Any,
    invocation: ProgramInvocation,
) -> _PrimalSpecialization:
    from .module import _tensor_dtype

    input_owners: dict[int, str] = {}
    for name, value in capture.inputs.items():
        owner = value.owner if isinstance(value, TensorView) else value
        if isinstance(owner, TensorStorage):
            input_owners[id(owner)] = name
    allocation_rows = tuple(capture.allocations.values())
    allocation_indices = {id(value): index for index, (value, _) in enumerate(allocation_rows)}

    def recipe(value: Any) -> _ValueRecipe:
        owner = value.owner if isinstance(value, TensorView) else value
        if isinstance(owner, TensorStorage):
            identity = id(owner)
            if identity in input_owners:
                source, key = "input", input_owners[identity]
            elif identity in allocation_indices:
                source, key = "allocation", allocation_indices[identity]
            else:
                raise TypeError(
                    "Module.forward() captured an unregistered TensorStorage; "
                    "use Module.empty(), Module.zeros(), or an invocation parameter"
                )
            if isinstance(value, TensorView):
                return _ValueRecipe(
                    source,
                    key,
                    tuple(value.shape),
                    tuple(value.layout.element_strides),
                    int(value.layout.element_offset),
                    value.access,
                    value.dtype,
                    value.element_type,
                )
            return _ValueRecipe(source, key)
        for name, supplied in capture.inputs.items():
            if value is supplied:
                return _ValueRecipe("input", name)
        return _ValueRecipe("constant", constant=value)

    def tree(value: Any) -> _TreeRecipe:
        if isinstance(value, (TensorStorage, TensorView)):
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
        _AllocationSpec(_tensor_dtype(value), tuple(value.shape), initializer) for value, initializer in allocation_rows
    )
    allocation_initializers = {invocation.graph.owner_id(value): initializer for value, initializer in allocation_rows}
    calls = tuple(
        (
            call.kernel,
            tuple(recipe(value) for value in call.arguments),
            call.grid,
            call.features,
        )
        for call in capture.calls
    )
    from .program_frontend import parse_program

    return _PrimalSpecialization(
        parse_program(invocation, allocation_initializers=allocation_initializers),
        invocation.template,
        inspect.signature(module.forward),
        allocations,
        calls,
        tree(outputs),
        _lower_native_primal(capture),
    )


def execute_module_primal(module: Any, arguments: tuple[Any, ...], keywords: Mapping[str, Any]) -> Any:
    key = module._program_specialization_key(arguments, keywords, variant="primal")
    specialization = module._program_cache.get(key)
    if not isinstance(specialization, _PrimalSpecialization):
        capture, outputs = _capture_module(module, arguments, keywords, consume_kernels=True)
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
    capture, outputs = _capture_module(module, arguments, keywords, consume_kernels=True)
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
    from ._runtime.program_autodiff import (
        ProgramAutodiffSpecialization,
        compile_program_autodiff,
    )

    specialization = module._program_cache.get(key)
    template = specialization.template if isinstance(specialization, ProgramAutodiffSpecialization) else None
    if template is None:
        template = ProgramTemplate.compile(capture)
    invocation = template.bind(capture, outputs)
    selected_outputs = expression.outputs or tuple(invocation.graph.outputs)
    if set(selected_outputs) != set(invocation.graph.outputs):
        raise ValueError("Program autodiff currently requires all Module outputs")
    allocation_initializers = {
        invocation.graph.owner_id(value): initializer for value, initializer in capture.allocations.values()
    }
    from .program_frontend import parse_program

    parsed_program = parse_program(
        invocation,
        allocation_initializers=allocation_initializers,
        vjp_wrt=expression.wrt,
    )
    if not isinstance(specialization, ProgramAutodiffSpecialization):
        specialization = compile_program_autodiff(parsed_program, template)
        module._program_cache[key] = specialization
    return specialization.invoke(invocation)


__all__ = [
    "ModuleVjpExpression",
    "ProgramTemplate",
    "execute_module_primal",
]
