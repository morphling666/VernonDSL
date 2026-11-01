"""Binds the C symbols of `@extern` declarations.

A symbol binds to a native function pointer or to a Python callable. Host code running as Python calls either
directly; compiled code loaded into this process calls a Python callable through a ctypes callback with the
extern's C signature (`native_address`). Bindings never decide semantics: the extern declaration fixes the
signature, ABI, and effect, and the registry only checks that a binding exists and agrees with its schema.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..frontend.model import ConcreteType
from .layout import TRANSFERRED_KINDS, StructFields, layout, pack, transfer_item, unpack
from .model import ExternAbi, HostExtern
from .values import HostExecutionError, from_python, scalar_value, to_python

_CTYPES = {
    "bool": ctypes.c_bool,
    "i32": ctypes.c_int32,
    "u32": ctypes.c_uint32,
    "i64": ctypes.c_int64,
    "u64": ctypes.c_uint64,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
}


class HostStr(ctypes.Structure):
    """The by-value `{const char *data; uint64_t size;}` a str argument passes as."""

    _fields_ = [("data", ctypes.c_char_p), ("size", ctypes.c_uint64)]


class HostSpan(ctypes.Structure):
    """The by-value `VhSpan`, `{const void *data; uint64_t size;}`, a list or dict argument passes as."""

    _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_uint64)]


class HostBindingError(LookupError):
    pass


@dataclass(frozen=True)
class _Binding:
    native: int | None = None
    python: Callable[..., Any] | None = None
    schema: str | None = None
    # The message of the calling thread's last failed call of the library, for a native status extern.
    last_error: Callable[[], str] | None = None


def _ctype(value_type: ConcreteType) -> Any:
    if value_type.kind == "handle":
        return ctypes.c_uint64
    if value_type.kind == "enum":
        return ctypes.c_int32
    if value_type.kind == "scalar":
        return _CTYPES[value_type.name]
    if value_type.kind == "str":
        return HostStr
    if value_type.kind in {"list", "dict"}:
        return HostSpan
    return ctypes.c_void_p


class ExternRegistry:
    def __init__(self) -> None:
        self._bindings: dict[str, _Binding] = {}
        self._libraries: list[tuple[ctypes.CDLL, str | None]] = []
        # Compiled modules resolve symbols once, when loaded; a changed revision tells their owners to reload.
        self.revision = 0

    def bind(self, symbol: str, function: Callable[..., Any], *, schema: str | None = None) -> "ExternRegistry":
        """Binds `symbol` to a Python callable. It receives Python boundary Values (Structs as dicts, Tensors as
        tuples) and returns them or Host values; raising HostExecutionError fails the call with that status. A generic
        extern's callable also receives the bound Struct's `VhType` where Host code passed the type."""

        self._bindings[symbol] = _Binding(python=function, schema=schema)
        self.revision += 1
        return self

    def bind_address(
        self,
        symbol: str,
        address: int,
        *,
        schema: str | None = None,
        last_error: Callable[[], str] | None = None,
    ) -> "ExternRegistry":
        """Binds `symbol` to a C function address. `last_error` returns the library's message for the calling
        thread's last failed call, which Python mode adds to the HostExecutionError of a failing status extern."""

        if not address:
            raise HostBindingError(f"extern symbol '{symbol}' is bound to a null address")
        self._bindings[symbol] = _Binding(native=int(address), schema=schema, last_error=last_error)
        self.revision += 1
        return self

    def bind_symbols(
        self,
        symbols: Mapping[str, int],
        *,
        schema: str | None = None,
        last_error: Callable[[], str] | None = None,
    ) -> "ExternRegistry":
        """Binds every symbol of an address table, such as one a nanobind module exports for its C API."""

        for symbol, address in symbols.items():
            self.bind_address(symbol, address, schema=schema, last_error=last_error)
        return self

    def bind_module(self, module: Any) -> "ExternRegistry":
        """Binds a generated binding module, such as a nanobind module: its `vh_symbols()` table of C symbol
        addresses, under its `vh_schema` identity, with its `vh_last_error()` when it has one. Host code declared
        against another revision of the library then fails to load instead of calling functions of a different
        layout."""

        schema = getattr(module, "vh_schema", None)
        symbols = getattr(module, "vh_symbols", None)
        if not isinstance(schema, str) or not callable(symbols):
            raise HostBindingError(f"{module!r} is not a binding module: it needs vh_schema and vh_symbols()")
        last_error = getattr(module, "vh_last_error", None)
        return self.bind_symbols(symbols(), schema=schema, last_error=last_error if callable(last_error) else None)

    def bind_library(self, library: str | Path | ctypes.CDLL, *, schema: str | None = None) -> "ExternRegistry":
        """Resolves otherwise unbound symbols from the exports of a shared library."""

        loaded = library if isinstance(library, ctypes.CDLL) else ctypes.CDLL(str(library))
        self._libraries.append((loaded, schema))
        self.revision += 1
        return self

    def bound(self, symbol: str) -> bool:
        return self._lookup(symbol) is not None

    def _lookup(self, symbol: str) -> _Binding | None:
        binding = self._bindings.get(symbol)
        if binding is not None:
            return binding
        for library, schema in reversed(self._libraries):
            function = getattr(library, symbol, None)
            if function is not None:
                address = ctypes.cast(function, ctypes.c_void_p).value
                if address:
                    return _Binding(native=address, schema=schema)
        return None

    def resolve(self, symbol: str, schema: str | None = None) -> _Binding:
        """The binding of `symbol`, checked against the schema its declaration names."""

        binding = self._lookup(symbol)
        if binding is None:
            raise HostBindingError(f"extern symbol '{symbol}' has no binding")
        if schema is not None and binding.schema is not None and schema != binding.schema:
            raise HostBindingError(
                f"extern symbol '{symbol}' is declared by schema '{schema}' but bound from schema '{binding.schema}'"
            )
        return binding

    def _binding(self, extern: HostExtern) -> _Binding:
        return self.resolve(extern.symbol, extern.schema)

    def native_address(self, extern: HostExtern, fields: StructFields, errors: list[BaseException]) -> tuple[int, Any]:
        """The C function address compiled Host code calls for `extern`, and an object that must outlive it.

        Python bindings become ctypes callbacks with the extern's C signature. An exception they raise cannot unwind
        through native frames, so it is appended to `errors` and the call fails: a status extern returns its status
        or 1, and a direct extern returns zero.
        """

        binding = self._binding(extern)
        if binding.native is not None:
            return binding.native, None
        assert binding.python is not None
        callback = _python_callback(extern, binding.python, fields, errors)
        return ctypes.cast(callback, ctypes.c_void_p).value or 0, callback


def _prototype(extern: HostExtern) -> Any:
    argument_types = [_ctype(parameter) for _, parameter in extern.parameters]
    if extern.type_argument is not None:
        argument_types.insert(extern.type_argument[0], ctypes.c_void_p)
    if extern.abi == ExternAbi.STATUS:
        if extern.result is not None:
            argument_types.append(ctypes.c_void_p)
            if extern.result.kind in TRANSFERRED_KINDS:
                argument_types.append(ctypes.c_void_p)
        result_type: Any = ctypes.c_int32
    else:
        result_type = None if extern.result is None else _ctype(extern.result)
    return ctypes.CFUNCTYPE(result_type, *argument_types)


def _received(span: HostSpan, value_type: ConcreteType, fields: StructFields) -> Any:
    """The new Python list or dict a Python binding receives for a span."""

    item = transfer_item(value_type)
    stride = layout(item, fields).size
    data = ctypes.string_at(span.data, span.size * stride) if span.size else b""
    parts = [
        to_python(unpack(data[index * stride : (index + 1) * stride], item, fields), item, fields)
        for index in range(span.size)
    ]
    return parts if value_type.kind == "list" else {key: value for key, value in parts}


def _fill(out: int, allocator: int | None, returned: Any, value_type: ConcreteType, fields: StructFields) -> None:
    """Fills the empty VhList at `out` with what a Python binding returned, through the caller's allocator."""

    from .native import VhListRecord, reserve_record

    if value_type.kind == "str":
        data = str(returned).encode("utf-8")
        count, size, alignment = len(data), 1, 1
    else:
        item = transfer_item(value_type)
        parts = list(returned) if value_type.kind == "list" else list(dict(returned).items())
        data = b"".join(pack(from_python(part, item, fields), item, fields) for part in parts)
        item_layout = layout(item, fields)
        count, size, alignment = len(parts), item_layout.size, item_layout.alignment
    if count == 0:
        return
    record = VhListRecord.from_address(out)
    reserve_record(record, count, size, alignment, allocator)
    ctypes.memmove(record.data, data, len(data))
    record.size = count


def _python_callback(
    extern: HostExtern, function: Callable[..., Any], fields: StructFields, errors: list[BaseException]
) -> Any:
    def callback(*raw: Any) -> Any:
        descriptor = None
        parameters = [parameter for _, parameter in extern.parameters]
        result_type = extern.result
        if extern.type_argument is not None:
            from .reflect import VhType

            raw = list(raw)
            descriptor = VhType.from_address(raw.pop(extern.type_argument[0]))
            # One binding serves every specialization of the symbol, so values decode as the type passed in.
            declared = ConcreteType("struct", extern.type_argument[1])
            passed = ConcreteType("struct", descriptor.name.decode("utf-8"))
            parameters = [passed if parameter == declared else parameter for parameter in parameters]
            result_type = passed if result_type == declared else result_type
        try:
            values = []
            for value, parameter in zip(raw, parameters, strict=False):
                if parameter.kind == "str":
                    values.append(ctypes.string_at(value.data, value.size).decode("utf-8"))
                elif parameter.kind in {"list", "dict"}:
                    values.append(_received(value, parameter, fields))
                elif parameter.kind in {"handle", "enum"}:
                    values.append(int(value))
                elif parameter.kind == "scalar":
                    values.append(to_python(scalar_value(value, parameter.name), parameter, fields))
                else:
                    data = ctypes.string_at(value, layout(parameter, fields).size)
                    values.append(to_python(unpack(data, parameter, fields), parameter, fields))
            if extern.type_argument is not None:
                values.insert(extern.type_argument[0], descriptor)
            result = function(*values)
            if extern.abi == ExternAbi.STATUS:
                out = len(extern.parameters)
                if result_type is not None and result_type.kind in TRANSFERRED_KINDS:
                    _fill(raw[out], raw[out + 1], result, result_type, fields)
                elif result_type is not None:
                    packed = pack(from_python(result, result_type, fields), result_type, fields)
                    ctypes.memmove(raw[out], packed, len(packed))
                return 0
            if result_type is None:
                return None
            return from_python(result, result_type, fields)
        except HostExecutionError as error:
            if extern.abi == ExternAbi.STATUS:
                return error.status
            errors.append(error)
        except BaseException as error:  # noqa: BLE001 - re-raised by the Host call once native frames unwind.
            errors.append(error)
            if extern.abi == ExternAbi.STATUS:
                return 1
        if extern.result is None:
            return None
        return 0

    return _prototype(extern)(callback)


_default_registry = ExternRegistry()


def default_registry() -> ExternRegistry:
    """The registry Host functions called from Python use unless given another."""

    return _default_registry
