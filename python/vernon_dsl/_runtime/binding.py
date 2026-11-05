from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .._dtypes import NUMPY_DTYPE_BY_SCALAR
from .._markers import is_struct_class
from ..host_values import host_abi_layout, pack_host_value
from .residency import _ResourceTransaction
from .sampler import SamplerState
from .session import _InvocationContext, cpu
from .tensor import RawBuffer, TensorStorage, TensorView, _checked_access
from .texture import TextureView, _TextureResource


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


def _host_value_array(state: Any, parameter: Any, value: Any, annotation: Any | None) -> np.ndarray:
    value_type = type(value)
    struct_type = annotation if is_struct_class(annotation) else (value_type if is_struct_class(value_type) else None)
    canonical_annotation = annotation if annotation is not None else struct_type
    if canonical_annotation is not None:
        layout = host_abi_layout(canonical_annotation)
        host_array = np.empty((), dtype=layout.dtype)
        host_array[()] = pack_host_value(canonical_annotation, value, parameter.name)
        return host_array
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
        host_array = np.empty((), dtype=np.dtype([("_bytes", np.uint8, (parameter.element_byte_size,))]))
        host_array["_bytes"] = packed_bytes
        return host_array
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
            raise ValueError(f"parameter {parameter.name!r} expects {leaves[0][1]} scalar values, got {packed.size}")
        host_array = np.empty((), dtype=np.dtype([("_bytes", np.uint8, (parameter.element_byte_size,))]))
        host_array["_bytes"] = packed.view(np.uint8).reshape(-1)
        return host_array
    if tuple(source.shape) != tuple(parameter.shape):
        raise ValueError(
            f"parameter {parameter.name!r} expects shape {tuple(parameter.shape)}, got {tuple(source.shape)}"
        )
    return source


def _argument_preparation(
    builder: Any,
    context: _InvocationContext,
    parameter: Any,
    value: Any,
    *,
    host_value: bool = False,
    annotation: Any | None = None,
) -> Callable[[], Any]:
    """Prepares one Program argument on a builder when called."""
    state = context.session
    if isinstance(value, (TensorStorage, TensorView)):
        declared_shape = tuple(parameter.shape)
        actual_shape, actual_byte_strides = _program_tensor_layout(parameter, value)
        if len(actual_shape) != len(declared_shape) or any(
            declared != 0 and declared != actual for declared, actual in zip(declared_shape, actual_shape, strict=True)
        ):
            raise ValueError(
                f"bound shape for Program value {parameter.slot} conflicts with the declared Program shape: "
                f"value {parameter.name!r} bound {list(actual_shape)}, declared {list(declared_shape)}"
            )
        owner = value.owner if isinstance(value, TensorView) else value
        if state.arch == cpu or host_value:
            owner.synchronize()
            array = value._native_host_array()
            return lambda: builder.prepare_host_tensor(parameter.slot, array)
        buffer = value._resident_buffer(context)
        layout = value.layout
        return lambda: builder.prepare_device_tensor(
            parameter.slot,
            buffer,
            parameter.access,
            list(actual_shape),
            list(actual_byte_strides),
            layout.byte_offset,
        )
    if isinstance(value, _TextureResource):
        if state.arch == cpu:
            raise TypeError("CPU kernels do not support Texture arguments")
        view = value._resident_view(context)
        return lambda: builder.prepare_device_texture(parameter.slot, view)
    host_array = _host_value_array(state, parameter, value, annotation)
    return lambda: builder.prepare_host_tensor(parameter.slot, host_array)


def _sampler_preparation(
    builder: Any, context: _InvocationContext, parameter: Any, sampler: SamplerState
) -> Callable[[], Any]:
    resident = sampler._resident_sampler(context)
    return lambda: builder.prepare_device_sampler(parameter.slot, resident)


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
