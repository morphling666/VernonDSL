from __future__ import annotations

from typing import Any, cast

import numpy as np

from .. import session as state
from ..resources import TensorStorage, _DispatchBorrowLease
from .elementwise import OperatorLoweringUnavailable, append_add


def materialize(nodes: list[tuple[Any, ...]], command_sink: Any | None = None) -> TensorStorage | np.ndarray:
    if not nodes:
        raise ValueError("graph gradient expression is empty")
    leaves: list[Any] = []
    for index, node in enumerate(nodes):
        if isinstance(node, tuple) and len(node) == 2 and node[0] == "leaf":
            leaves.append(node[1])
        elif (
            not isinstance(node, tuple)
            or len(node) != 3
            or node[0] != "add"
            or not isinstance(node[1], int)
            or not isinstance(node[2], int)
            or isinstance(node[1], bool)
            or isinstance(node[2], bool)
            or not 0 <= node[1] < index
            or not 0 <= node[2] < index
        ):
            raise ValueError("graph gradient expression contains an invalid node")
    device = (
        state._architecture != state.cpu and bool(leaves) and all(isinstance(value, TensorStorage) for value in leaves)
    )
    if command_sink is not None and not device:
        raise RuntimeError("planned gradient expression requires device-resident TensorStorage leaves")
    template = leaves[0] if device else None
    if device:
        template = cast(TensorStorage, template)
        for value in leaves[1:]:
            value = cast(TensorStorage, value)
            try:
                TensorStorage._gradient_layout(template, value)
            except ValueError as error:
                if command_sink is not None:
                    raise RuntimeError("planned gradient expression has incompatible tangent layouts") from error
                device = False
                break
    if device:
        values: list[Any] = []
        outputs: list[TensorStorage] = []
        operations: list[list[tuple[Any, Any, Any]]] = []
        for node in nodes:
            if node[0] == "leaf":
                values.append(node[1])
                continue
            left = values[node[1]]
            right = values[node[2]]
            left = cast(TensorStorage, left)
            right = cast(TensorStorage, right)
            result = TensorStorage._empty_gradient_like(left)
            operations.append(TensorStorage._gradient_add_views(result, left, right))
            outputs.append(result)
            values.append(result)
        borrows = [
            (f"operator-{operation_index}-output", output, "write")
            if argument_index == 0
            else (f"operator-{operation_index}-input-{argument_index}", output, "read")
            for operation_index, operation in enumerate(operations)
            for views in operation
            for argument_index, output in enumerate(views)
        ]
        lease = _DispatchBorrowLease(borrows)
        retained = False
        transactions = []
        try:
            if command_sink is not None:
                for output in outputs:
                    transactions.append(output._begin_planned_device_write())
            operator_dag: Any | None = None
            try:
                for operation in operations:
                    for views in operation:
                        operator_dag = append_add(operator_dag, *views)
            except OperatorLoweringUnavailable as error:
                if command_sink is not None:
                    raise RuntimeError("planned gradient expression cannot be lowered to a device operator") from error
                device = False
            if device:
                if operator_dag is None:
                    raise RuntimeError("device gradient expression produced no operator commands")
                operator_dag.execute(command_sink, lease if command_sink is not None else None)
                retained = command_sink is not None
                if command_sink is None:
                    for output in outputs:
                        output._mark_device_dirty()
                else:
                    for transaction in transactions:
                        command_sink._retain_completion(transaction)
                return cast(TensorStorage, values[-1])
        except BaseException:
            for transaction in transactions:
                transaction._rollback_planned_state()
            raise
        finally:
            if not retained:
                lease.release()
    values = []
    for node in nodes:
        if node[0] == "leaf":
            values.append(node[1])
            continue
        left = values[node[1]]
        right = values[node[2]]
        if isinstance(left, TensorStorage) or isinstance(right, TensorStorage):
            storage = left if isinstance(left, TensorStorage) else right
            other = right if isinstance(left, TensorStorage) else left
            values.append(TensorStorage._add_gradients(storage, other))
        else:
            values.append(np.ascontiguousarray(np.asarray(left) + np.asarray(right)))
    return values[-1]
