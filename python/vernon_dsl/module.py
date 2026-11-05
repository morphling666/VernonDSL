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

from ._params import assemble_params, is_params, params_leaf_items
from ._runtime.session import _SessionArtifactCache
from ._runtime.tensor import TensorStorage, TensorView

_RUNTIME_FIELDS = frozenset({"_module_initialized", "_modules", "_program_cache", "_configuration_cache"})
# Bumped by every Module attribute write; a child's write changes its parents' configurations too, so the counter is
# global rather than per instance.
_configuration_epoch = [0]


def _annotation_signature(annotation: Any) -> Any:
    """A DSL-annotated parameter's part of the specialization key, which its value does not change; None when the
    annotation is not a Storage, Value or Resource and the value's own signature is used."""
    if annotation is None:
        return None
    from .frontend.model import SemanticCategory
    from .frontend.runtime_types import runtime_parameter_descriptor

    try:
        descriptor = runtime_parameter_descriptor(annotation)
    except TypeError:
        return None
    if descriptor.kind not in {SemanticCategory.STORAGE, SemanticCategory.VALUE, SemanticCategory.RESOURCE}:
        return None
    return (
        descriptor.kind.value,
        descriptor.logical,
        getattr(descriptor, "access", None),
        getattr(descriptor, "as_view", None),
    )


@dataclass(frozen=True)
class ModuleDefinition:
    """Immutable frontend identity shared by every instance of a Module class."""

    original_function: Any
    signature: inspect.Signature
    resolved_annotations: Mapping[str, Any]
    source_identity: tuple[str, int, str]
    forward_ast: ast.FunctionDef | None
    forward_ast_error: str | None

    def bind_leaves(self, arguments: tuple[Any, ...], keywords: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
        """Binds forward() arguments as `(Program input name, annotation, value)`, one per params leaf."""
        bound = self.signature.bind(*arguments, **keywords)
        bound.apply_defaults()
        return [
            leaf
            for name, value in bound.arguments.items()
            for leaf in params_leaf_items(name, self.resolved_annotations.get(name), value)
        ]

    def fast_leaves(self) -> tuple[tuple[str, ...], tuple[Any, ...]] | None:
        """Parameter names and the leaf part of the specialization key when positional arguments bind one to one
        and the key does not depend on their values; None when calls must bind through the signature."""
        cached = self.__dict__.get("_fast_leaves", self)
        if cached is not self:
            return cached
        from .program import _program_control_kind

        names: list[str] = []
        leaves: list[tuple[str, Any]] = []
        plan: tuple[tuple[str, ...], tuple[Any, ...]] | None = None
        for parameter in self.signature.parameters.values():
            annotation = self.resolved_annotations.get(parameter.name)
            annotated = None
            if (
                parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                and not is_params(annotation)
                and _program_control_kind(annotation) is None
            ):
                annotated = _annotation_signature(annotation)
            if annotated is None:
                break
            names.append(parameter.name)
            leaves.append((parameter.name, annotated))
        else:
            plan = (tuple(names), tuple(leaves))
        object.__setattr__(self, "_fast_leaves", plan)
        return plan

    def assemble_arguments(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """The inverse of `bind_leaves`: rebuilds each params parameter from its dotted Program inputs."""
        arguments: dict[str, Any] = {}
        for name in self.signature.parameters:
            annotation = self.resolved_annotations.get(name)
            if is_params(annotation):
                arguments[name] = assemble_params(name, annotation, inputs)
            elif name in inputs:
                arguments[name] = inputs[name]
        return arguments


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
    _program_cache: _SessionArtifactCache

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
        object.__setattr__(self, "_program_cache", _SessionArtifactCache())

    def __delattr__(self, name: str) -> None:
        _configuration_epoch[0] += 1
        self.__dict__.get("_modules", {}).pop(name, None)
        object.__delattr__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        _configuration_epoch[0] += 1
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
        from ._runtime.session import _invocation_context
        from .program import execute_module_primal

        with _invocation_context():
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
        def signature(value: Any) -> Any:
            from .render import DrawCommand, DynamicState, RenderPass

            if isinstance(value, RenderPass):
                target = value.target
                colors = tuple(
                    (
                        location,
                        texture.dimension,
                        getattr(getattr(texture, "format", None), "name", None),
                        getattr(texture, "sample_count", 1),
                    )
                    for location, texture in target._color_attachments()
                )
                depth = target._depth_attachment()
                depth_signature = (
                    None
                    if depth is None
                    else (
                        depth.dimension,
                        getattr(getattr(depth, "format", None), "name", None),
                        getattr(depth, "sample_count", 1),
                    )
                )
                return ("RenderPass", colors, depth_signature)
            if isinstance(value, (DrawCommand, DynamicState)):
                # These are managed invocation controls, not specialization inputs.
                return type(value).__name__
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
            local = tuple(
                (name, signature(value))
                for name, value in sorted(module.__dict__.items())
                if name not in _RUNTIME_FIELDS and not isinstance(value, Module)
            )
            children = tuple((name, module_configuration(child)) for name, child in sorted(module._modules.items()))
            return type(module)._module_definition.source_identity, local, children

        definition = type(self)._module_definition

        def invocation_signature(annotation: Any, value: Any) -> Any:
            annotated = _annotation_signature(annotation)
            return signature(value) if annotated is None else annotated

        return (
            definition.source_identity,
            self._configuration_key(module_configuration),
            tuple(
                (leaf, invocation_signature(annotation, value))
                for leaf, annotation, value in definition.bind_leaves(arguments, keywords)
            ),
            variant,
        )

    def _configuration_key(self, compute: Any) -> tuple[Any, ...]:
        """The configuration part of the specialization key, recomputed only after some Module attribute changes;
        attribute values enter the key by type or by immutable scalar value, so a write is the only way it changes."""
        epoch = _configuration_epoch[0]
        cached = self.__dict__.get("_configuration_cache")
        if cached is not None and cached[0] == epoch:
            return cached[1]
        configuration = compute(self)
        # The epoch is read before computing, so a write racing with the walk only forces a recompute.
        object.__setattr__(self, "_configuration_cache", (epoch, configuration))
        return configuration

    def _fast_primal_key(self, arguments: tuple[Any, ...], variant: Any = "primal") -> tuple[Any, ...] | None:
        """The specialization key `_program_specialization_key` builds, without binding, when every forward()
        parameter is positional, DSL-annotated and supplied; None otherwise."""
        plan = type(self)._module_definition.fast_leaves()
        if plan is None or len(arguments) != len(plan[0]):
            return None
        cached = self.__dict__.get("_configuration_cache")
        if cached is None or cached[0] != _configuration_epoch[0]:
            return None
        return (type(self)._module_definition.source_identity, cached[1], plan[1], variant)

    def _require_initialized(self) -> None:
        if not self.__dict__.get("_module_initialized", False):
            raise RuntimeError(f"{type(self).__name__}.__init__() must call super().__init__()")


Module._module_definition = _module_definition(Module.forward, Module)


__all__ = ["Module", "ModuleDefinition"]
