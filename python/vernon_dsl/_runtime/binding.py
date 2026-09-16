from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._dtypes import NUMPY_DTYPE_BY_SCALAR
from ..host_values import host_abi_layout, pack_host_value
from .residency import _ResourceTransaction
from .sampler import SamplerState
from .session import _InvocationContext, cpu
from .tensor import RawBuffer, TensorStorage, TensorView, _checked_access
from .texture import TextureView, _TextureResource


def _raise_invocation_error(outcome: Any, prefix: str, context: _InvocationContext) -> None:
    error_type = (
        ValueError if int(outcome.status) == context.session.native.Status.INVALID_ARGUMENT.value else RuntimeError
    )
    raise error_type(f"{prefix}: {outcome.error}")


def _program_tensor_layout(
    parameter: Any, value: TensorStorage | TensorView
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = tuple(value.shape)
    byte_strides = tuple(value.layout.byte_strides)
    rank = len(shape)
    element_size = value.dtype.itemsize
    if element_size != parameter.element_byte_size:
        while rank > len(parameter.shape):
            rank -= 1
            if byte_strides[rank] != element_size:
                raise ValueError(
                    f"bound storage for Program value {parameter.name!r} does not have contiguous aggregate elements"
                )
            element_size *= shape[rank]
        if element_size != parameter.element_byte_size:
            raise ValueError(
                f"bound storage element size for Program value {parameter.name!r} does not match its declared layout"
            )
    return shape[:rank], byte_strides[:rank]


@dataclass(frozen=True)
class _ActiveBindingTransaction:
    native: Any
    context: _InvocationContext


class _PersistentBindingTable:
    """Owns native transactional persistent bindings for one executable."""

    def __init__(self) -> None:
        self._native_program: Any | None = None
        self._instance: Any | None = None
        self._transactions = threading.local()
        self._lock = threading.RLock()

    def _native_instance(self, native_program: Any) -> Any:
        with self._lock:
            if self._instance is None or self._native_program is not native_program:
                self._native_program = native_program
                self._instance = native_program.program_instance()
            return self._instance

    def clear(self) -> None:
        with self._lock:
            self._native_program = None
            self._instance = None

    @contextmanager
    def invocation(self, native_program: Any, context: _InvocationContext) -> Iterator[Any]:
        native_transaction = self._native_instance(native_program).begin_invocation()
        if getattr(self._transactions, "current", None) is not None:
            raise RuntimeError("persistent binding invocations cannot be nested on one thread")
        transaction = _ActiveBindingTransaction(native_transaction, context)
        self._transactions.current = transaction
        try:
            yield native_transaction
        except BaseException:
            native_transaction.rollback()
            raise
        else:
            if not native_transaction.finished:
                native_transaction.rollback()
                raise RuntimeError("Program invocation exited without explicit commit or rollback")
        finally:
            self._transactions.current = None

    @property
    def telemetry(self) -> dict[str, int]:
        with self._lock:
            return {} if self._instance is None else dict(self._instance.telemetry)

    def _bind(
        self,
        builder: Any,
        parameter: Any,
        token: tuple[Any, ...],
        prepare: Any,
        *,
        upload_bytes: int = 0,
        upload_ranges: int = 0,
        upload_precedes_lookup: bool = False,
    ) -> None:
        transaction = getattr(self._transactions, "current", None)
        if transaction is None:
            raise RuntimeError("binding update requires an active invocation transaction")
        transaction.native.bind(
            parameter.slot,
            token,
            prepare,
            upload_bytes,
            upload_ranges,
            upload_precedes_lookup,
        )

    def bind_argument(
        self,
        builder: Any,
        native_program: Any,
        parameter: Any,
        value: Any,
        *,
        host_value: bool = False,
        annotation: Any | None = None,
        binding_token: int | None = None,
    ) -> Any:
        transaction = getattr(self._transactions, "current", None)
        if transaction is None:
            raise RuntimeError("argument binding requires an active invocation transaction")
        state = transaction.context.session
        if isinstance(value, (TensorStorage, TensorView)):
            declared_shape = tuple(parameter.shape)
            actual_shape, actual_byte_strides = _program_tensor_layout(parameter, value)
            if len(actual_shape) != len(declared_shape) or any(
                declared != 0 and declared != actual
                for declared, actual in zip(declared_shape, actual_shape, strict=True)
            ):
                raise ValueError(
                    f"bound shape for Program value {parameter.slot} conflicts with the declared Program shape: "
                    f"value {parameter.name!r} bound {list(actual_shape)}, declared {list(declared_shape)}"
                )
            if state.arch == cpu or host_value:
                owner = value.owner if isinstance(value, TensorView) else value
                owner.synchronize()
                array = value._native_host_array()
                token = (
                    "host-resource",
                    owner._control.owner_id,
                    array.dtype.str,
                    tuple(array.shape),
                    tuple(array.strides),
                )
                return self._bind(
                    builder,
                    parameter,
                    token,
                    lambda: builder.prepare_host_tensor(parameter.slot, array),
                )
            if state.rhi_host is None:
                raise RuntimeError("device Tensor arguments require a GPU RHI host")
            owner = value.owner if isinstance(value, TensorView) else value
            dirty_tracker = getattr(owner._control.coherence, "dirty_ranges", None)
            dirty_ranges = tuple(dirty_tracker.ranges) if dirty_tracker is not None else ()
            buffer = value._resident_buffer(transaction.context)
            layout = value.layout
            token = (
                "rhi-tensor",
                owner._control.owner_id,
                state.identity,
                parameter.access,
                actual_shape,
                actual_byte_strides,
                layout.byte_offset,
            )
            return self._bind(
                builder,
                parameter,
                token,
                lambda: builder.prepare_rhi_tensor(
                    parameter.slot,
                    buffer,
                    parameter.access,
                    list(actual_shape),
                    list(actual_byte_strides),
                    layout.byte_offset,
                ),
                upload_bytes=sum(end - begin for begin, end in dirty_ranges),
                upload_ranges=len(dirty_ranges),
                upload_precedes_lookup=True,
            )
        if isinstance(value, _TextureResource):
            if state.arch == cpu:
                raise TypeError("CPU kernels do not support Texture arguments")
            if state.rhi_host is None:
                raise RuntimeError("Texture arguments require a GPU RHI host")
            owner = value.owner if isinstance(value, TextureView) else value
            dirty_mips = owner._dirty_mips(owner._control.coherence.host_dirty_subresources)
            upload_bytes = sum(owner._mip_arrays[level].nbytes for level in dirty_mips)
            view = value._resident_view(transaction.context)
            return self._bind(
                builder,
                parameter,
                ("rhi-texture", owner._control.owner_id, state.identity, value._view_key),
                lambda: builder.prepare_rhi_texture(parameter.slot, view),
                upload_bytes=upload_bytes,
                upload_ranges=len(dirty_mips),
                upload_precedes_lookup=True,
            )

        execution_token = ("execution-value", binding_token) if binding_token is not None else None
        value_type = type(value)
        struct_type = (
            annotation
            if isinstance(annotation, type) and getattr(annotation, "__vernon_dsl__", (None, {}))[0] == "struct"
            else (value_type if getattr(value_type, "__vernon_dsl__", (None, {}))[0] == "struct" else None)
        )
        canonical_annotation = annotation if annotation is not None else struct_type
        if canonical_annotation is not None:
            layout = host_abi_layout(canonical_annotation)
            host_array = np.empty((), dtype=layout.dtype)
            host_array[()] = pack_host_value(canonical_annotation, value, parameter.name)
        else:
            numpy_dtypes = {
                state.native.DATA_BOOL: NUMPY_DTYPE_BY_SCALAR["bool"],
                state.native.DATA_I32: NUMPY_DTYPE_BY_SCALAR["i32"],
                state.native.DATA_U32: NUMPY_DTYPE_BY_SCALAR["u32"],
                state.native.DATA_F16: NUMPY_DTYPE_BY_SCALAR["f16"],
                state.native.DATA_F32: NUMPY_DTYPE_BY_SCALAR["f32"],
                state.native.DATA_F64: NUMPY_DTYPE_BY_SCALAR["f64"],
            }
            leaves = tuple(parameter.element_leaves)
            dtype = numpy_dtypes.get(leaves[0][0]) if len(leaves) == 1 and leaves[0][2] == 0 else None
            if dtype is None:
                packed_bytes = np.zeros(parameter.element_byte_size, dtype=np.uint8)
                paths = tuple(parameter.element_leaf_paths)
                if len(paths) != len(leaves):
                    raise RuntimeError(f"aggregate parameter {parameter.name!r} has an incomplete canonical ABI")
                for leaf, path in zip(leaves, paths, strict=True):
                    leaf_value = value
                    for component in path:
                        if isinstance(component, str):
                            if not isinstance(leaf_value, dict) or component not in leaf_value:
                                raise ValueError(f"parameter {parameter.name!r} missing field {component!r}")
                            leaf_value = leaf_value[component]
                        else:
                            try:
                                leaf_value = leaf_value[component]
                            except (IndexError, KeyError, TypeError) as error:
                                raise ValueError(f"parameter {parameter.name!r} missing index {component}") from error
                    leaf_dtype = numpy_dtypes.get(leaf[0])
                    if leaf_dtype is None:
                        raise TypeError(f"aggregate parameter {parameter.name!r} has an unsupported leaf dtype")
                    leaf_array = np.ascontiguousarray(np.asarray(leaf_value, dtype=leaf_dtype)).reshape(-1)
                    if leaf_array.size != leaf[1]:
                        raise ValueError(
                            f"parameter {parameter.name!r} leaf expects {leaf[1]} scalar values, got {leaf_array.size}"
                        )
                    encoded = leaf_array.view(np.uint8)
                    begin = leaf[2]
                    end = begin + encoded.size
                    if end > packed_bytes.size:
                        raise RuntimeError(f"aggregate parameter {parameter.name!r} exceeds its canonical ABI")
                    packed_bytes[begin:end] = encoded
                host_array = np.empty(
                    (),
                    dtype=np.dtype([("_bytes", np.uint8, (parameter.element_byte_size,))]),
                )
                host_array["_bytes"] = packed_bytes
            else:
                source = np.asarray(value, dtype=dtype)
                whole_value = not parameter.shape and leaves[0][1] > 1
                if whole_value:
                    packed = np.ascontiguousarray(source)
                    expected_shape = tuple(parameter.element_leaf_shapes[0])
                    if expected_shape and tuple(packed.shape) != expected_shape:
                        raise ValueError(
                            f"parameter {parameter.name!r} shape {list(packed.shape)} "
                            f"does not match reflection {list(expected_shape)}"
                        )
                    if packed.size != leaves[0][1] or packed.nbytes != parameter.element_byte_size:
                        raise ValueError(
                            f"parameter {parameter.name!r} expects {leaves[0][1]} scalar values, got {packed.size}"
                        )
                    host_array = np.empty(
                        (),
                        dtype=np.dtype([("_bytes", np.uint8, (parameter.element_byte_size,))]),
                    )
                    host_array["_bytes"] = packed.view(np.uint8).reshape(-1)
                else:
                    host_array = source
                if not whole_value and tuple(host_array.shape) != tuple(parameter.shape):
                    raise ValueError(
                        f"parameter {parameter.name!r} expects shape {tuple(parameter.shape)}, "
                        f"got {tuple(host_array.shape)}"
                    )
        token = execution_token or (
            "host-value",
            host_array.dtype.str,
            tuple(host_array.shape),
            host_array.tobytes(),
        )
        return self._bind(
            builder,
            parameter,
            token,
            lambda: builder.prepare_host_tensor(parameter.slot, host_array),
            upload_bytes=int(host_array.nbytes),
            upload_ranges=1,
        )

    def bind_sampler(self, builder: Any, native_program: Any, parameter: Any, sampler: SamplerState) -> None:
        transaction = getattr(self._transactions, "current", None)
        if transaction is None:
            raise RuntimeError("sampler binding requires an active invocation transaction")
        resident = sampler._resident_sampler(transaction.context)
        self._bind(
            builder,
            parameter,
            ("rhi-sampler", sampler._address, transaction.context.identity),
            lambda: builder.prepare_rhi_sampler(parameter.slot, resident),
        )

    def bind_render_pass_control(self, slot: int, token: tuple[Any, ...], control: Any) -> None:
        transaction = getattr(self._transactions, "current", None)
        if transaction is None:
            raise RuntimeError("Program control update requires an active invocation transaction")
        transaction.native.bind_render_pass(slot, token, control)

    def bind_draw_command_control(self, slot: int, token: tuple[Any, ...], control: Any) -> None:
        transaction = getattr(self._transactions, "current", None)
        if transaction is None:
            raise RuntimeError("Program control update requires an active invocation transaction")
        transaction.native.bind_draw_command(slot, token, control)

    def bind_dynamic_state_control(self, slot: int, token: tuple[Any, ...], control: Any) -> None:
        transaction = getattr(self._transactions, "current", None)
        if transaction is None:
            raise RuntimeError("Program control update requires an active invocation transaction")
        transaction.native.bind_dynamic_state(slot, token, control)


def _normalize_dispatch_borrows(
    borrows: list[tuple[str, TensorStorage | RawBuffer | TensorView | _TextureResource, str]],
) -> list[tuple[str, RawBuffer | TensorView | _TextureResource, str]]:
    normalized: list[tuple[str, RawBuffer | TensorView | _TextureResource, str]] = []
    for name, value, access in borrows:
        access = _checked_access(access)
        resource = value._full_view("read_write") if isinstance(value, TensorStorage) else value
        if isinstance(resource, TensorView):
            if access in {"read", "read_write"} and resource.access == "write":
                raise ValueError(f"TensorView argument '{name}' does not permit reads")
            if access in {"write", "read_write"} and resource.access == "read":
                raise ValueError(f"TensorView argument '{name}' does not permit writes")
        normalized.append((name, resource, access))
    return normalized


def _dispatch_borrow_owner(
    resource: TensorStorage | RawBuffer | TensorView | _TextureResource,
) -> TensorStorage | RawBuffer | _TextureResource:
    return resource.owner if isinstance(resource, (TensorView, TextureView)) else resource


class _DispatchBorrowLease(_ResourceTransaction):
    """An acquired resource borrow released only after dispatch completion."""

    def __init__(
        self,
        borrows: list[tuple[str, TensorStorage | RawBuffer | TensorView | _TextureResource, str]],
        context: _InvocationContext,
    ):
        normalized = _normalize_dispatch_borrows(borrows)
        super().__init__(
            [
                (name, _dispatch_borrow_owner(resource), resource._region, access)
                for name, resource, access in normalized
            ],
            context,
        )


@contextmanager
def _dispatch_borrow_scope(
    borrows: list[tuple[str, TensorStorage | RawBuffer | TensorView | _TextureResource, str]],
    context: _InvocationContext,
) -> Iterator[_DispatchBorrowLease]:
    lease = _DispatchBorrowLease(borrows, context)
    try:
        yield lease
    finally:
        lease.release()
