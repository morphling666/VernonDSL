"""The Host language: typed CPU code over C externs, with stateful classes and a reflection table. It runs as the
Python it is written in during development and is compiled ahead of time for packaged builds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .externs import ExternRegistry, HostBindingError, default_registry
from .model import HostEffect, HostProgram, HostStatus
from .values import HostExecutionError

if TYPE_CHECKING:
    from .module import HostModule


def load_host_program(source_path: Any, entry: str) -> HostProgram:
    from .frontend import load_host_program as load

    return load(source_path, entry)


def load_host_module(source_path: Any) -> HostProgram:
    from .frontend import load_host_module as load

    return load(source_path)


def address_of(instance: Any) -> int:
    """The address of a Host class instance's state, as C callers and the reflection table's methods take it."""

    from .native import address_of as address

    return address(instance)


def __getattr__(name: str) -> Any:
    if name == "HostModule":
        from . import module

        return module.HostModule
    raise AttributeError(name)


__all__ = [
    "ExternRegistry",
    "HostBindingError",
    "HostEffect",
    "HostExecutionError",
    "HostModule",
    "HostProgram",
    "HostStatus",
    "address_of",
    "default_registry",
    "load_host_module",
    "load_host_program",
]
