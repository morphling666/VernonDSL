"""User-facing Module composition over compiled Programs."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from ._dtypes import scalar_name
from ._runtime.tensor import TensorStorage, TensorView


def _tensor_dtype(value: TensorStorage) -> Any:
    if value._element_type is not None:
        return value._element_type
    name = scalar_name(value.dtype)
    if name is None:
        raise TypeError(f"cannot allocate a Module transient for dtype {value.dtype}")
    types = __import__("vernon_dsl.types", fromlist=[name])
    return getattr(types, name)


class Module:
    """Composable host program whose forward method is parsed into a Program."""

    _module_initialized: bool
    _modules: dict[str, Module]
    _program_cache: dict[tuple[Any, ...], Any]

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
            return local, children

        configuration = module_configuration(self)
        return (
            session._runtime_generation,
            session._architecture.name,
            type(self),
            configuration,
            tuple(signature(value) for value in arguments),
            tuple((name, signature(value)) for name, value in sorted(keywords.items())),
            variant,
        )

    def _require_initialized(self) -> None:
        if not self.__dict__.get("_module_initialized", False):
            raise RuntimeError(f"{type(self).__name__}.__init__() must call super().__init__()")


__all__ = ["Module"]
