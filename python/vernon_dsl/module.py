"""User-facing Module composition over compiled Programs."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np

from ._runtime.tensor import TensorStorage, TensorView


@dataclass(frozen=True)
class ModuleDefinition:
    """Immutable frontend identity shared by every instance of a Module class."""

    original_function: Any
    signature: inspect.Signature
    resolved_annotations: Mapping[str, Any]
    source_identity: tuple[str, int, str]
    forward_ast: ast.FunctionDef | None
    forward_ast_error: str | None


def _module_definition(function: Any, owner: type[Any]) -> ModuleDefinition:
    owner_name = owner.__name__
    if not inspect.isfunction(function):
        raise TypeError(f"{owner_name}.forward must be an instance method")
    signature = inspect.signature(function)
    parameters = tuple(signature.parameters.values())
    if not parameters:
        raise TypeError(f"{owner_name}.forward must accept self")
    invocation_signature = signature.replace(parameters=parameters[1:])
    try:
        annotations = inspect.get_annotations(
            function,
            globals={**function.__globals__, owner_name: owner},
            eval_str=True,
        )
    except (NameError, TypeError) as error:
        raise TypeError(f"cannot resolve {owner_name}.forward annotations: {error}") from None
    code = function.__code__
    forward_ast = None
    forward_ast_error = None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            forward_ast_error = f"{owner_name}.forward must be a function definition"
        else:
            # Consumers treat this class-level tree as immutable frontend input.
            forward_ast = tree.body[0]
    except (OSError, TypeError, SyntaxError) as error:
        forward_ast_error = f"cannot parse {owner_name}.forward: {error}"
    return ModuleDefinition(
        function,
        invocation_signature,
        MappingProxyType(dict(annotations)),
        (code.co_filename, code.co_firstlineno, function.__qualname__),
        forward_ast,
        forward_ast_error,
    )


class Module:
    """Composable host program whose forward method is parsed into a Program."""

    _module_definition: ClassVar[ModuleDefinition]
    _module_initialized: bool
    _modules: dict[str, Module]
    _program_cache: dict[tuple[Any, ...], Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        function = cls.__dict__.get("forward")
        if function is None:
            cls._module_definition = next(
                base._module_definition for base in cls.__mro__[1:] if "_module_definition" in base.__dict__
            )
            return
        cls._module_definition = _module_definition(function, cls)

    def __init__(self) -> None:
        object.__setattr__(self, "_module_initialized", True)
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_program_cache", {})

    def __setattr__(self, name: str, value: Any) -> None:
        initialized = self.__dict__.get("_module_initialized", False)
        if initialized and not name.startswith("_"):
            if isinstance(value, (TensorStorage, TensorView)):
                raise TypeError(
                    "Module attributes cannot capture invocation TensorStorage or TensorView values; "
                    "pass them to forward() or allocate graph-owned transients there"
                )
            modules: dict[str, Module] = self.__dict__["_modules"]
            if isinstance(value, Module):
                modules[name] = value
            else:
                modules.pop(name, None)
        object.__setattr__(self, name, value)

    @property
    def modules(self) -> Mapping[str, Module]:
        self._require_initialized()
        return MappingProxyType(dict(self._modules))

    def forward(self, *arguments: Any, **keywords: Any) -> Any:
        raise NotImplementedError

    def backward(self, *arguments: Any, **keywords: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *arguments: Any, **keywords: Any) -> Any:
        self._require_initialized()
        from .program import execute_module_primal

        return execute_module_primal(self, arguments, keywords)

    def named_modules(self, prefix: str = "") -> tuple[tuple[str, Module], ...]:
        self._require_initialized()
        result: list[tuple[str, Module]] = [(prefix, self)]
        for name, child in self._modules.items():
            qualified = f"{prefix}.{name}" if prefix else name
            result.extend(child.named_modules(qualified))
        return tuple(result)

    def _program_specialization_key(
        self,
        arguments: tuple[Any, ...],
        keywords: Mapping[str, Any],
        *,
        variant: Any,
    ) -> tuple[Any, ...]:
        from ._runtime import session

        def signature(value: Any) -> Any:
            if isinstance(value, (TensorStorage, TensorView)):
                return (
                    type(value).__name__,
                    str(value.dtype),
                    tuple(value.shape),
                    repr(getattr(value, "_element_type", None)),
                    str(getattr(value, "access", "read_write")),
                )
            if isinstance(value, np.generic):
                return ("scalar", str(value.dtype))
            if isinstance(value, (str, int, float, bool, type(None))):
                return (type(value).__name__, value)
            if isinstance(value, tuple):
                return ("tuple", tuple(signature(member) for member in value))
            return (type(value).__module__, type(value).__qualname__)

        def module_configuration(module: Module) -> tuple[Any, ...]:
            runtime_fields = {
                "_module_initialized",
                "_modules",
                "_program_cache",
            }
            local = tuple(
                (name, signature(value))
                for name, value in sorted(module.__dict__.items())
                if name not in runtime_fields and not isinstance(value, Module)
            )
            children = tuple((name, module_configuration(child)) for name, child in sorted(module._modules.items()))
            return type(module)._module_definition.source_identity, local, children

        configuration = module_configuration(self)
        definition = type(self)._module_definition
        bound = definition.signature.bind(*arguments, **keywords)
        bound.apply_defaults()
        annotations = definition.resolved_annotations

        def invocation_signature(name: str, value: Any) -> Any:
            if isinstance(value, (TensorStorage, TensorView)):
                return signature(value)
            annotation = annotations.get(name)
            if annotation is not None:
                from .frontend.model import SemanticCategory
                from .frontend.runtime_types import runtime_parameter_descriptor

                try:
                    descriptor = runtime_parameter_descriptor(annotation)
                except TypeError:
                    pass
                else:
                    category = descriptor.kind
                    if category in {SemanticCategory.VALUE, SemanticCategory.RESOURCE}:
                        return (category.value, descriptor.logical)
            return signature(value)

        return (
            session._runtime_generation,
            session._architecture.name,
            definition.source_identity,
            configuration,
            tuple((name, invocation_signature(name, value)) for name, value in bound.arguments.items()),
            variant,
        )

    def _require_initialized(self) -> None:
        if not self.__dict__.get("_module_initialized", False):
            raise RuntimeError(f"{type(self).__name__}.__init__() must call super().__init__()")


Module._module_definition = _module_definition(Module.forward, Module)


__all__ = ["Module", "ModuleDefinition"]
