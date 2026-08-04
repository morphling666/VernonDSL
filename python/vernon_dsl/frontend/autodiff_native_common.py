from __future__ import annotations

from dataclasses import dataclass

from .._versions import COMPILER_CONTRACT_VERSION, PIPELINE_VERSION
from .autodiff import BuiltinOp, ProgramGraphNode
from .autodiff_native_abi import native_type
from .autodiff_profiles import AutodiffProfilePlan


class AutodiffNativeLoweringError(ValueError):
    """Raised when a semantic autodiff graph cannot be lowered natively."""


@dataclass
class Names:
    index: int = 0

    def fresh(self) -> str:
        value = f"%ad{self.index}"
        self.index += 1
        return value


def module(function: list[str], profile: str, plan: AutodiffProfilePlan) -> str:
    attributes = (
        f"vernon.compiler_contract_version = {COMPILER_CONTRACT_VERSION} : i64, "
        f"vernon.pipeline_version = {PIPELINE_VERSION} : i64, "
        f'vernon.ad_profile = "{profile}", '
        f'vernon.ad_profiles_identity = "{plan.identity}"'
    )
    return "\n".join([f"module attributes {{{attributes}}} {{", *function, "}", ""])


def index_constants(indices: tuple[int, ...], names: Names, lines: list[str]) -> tuple[str, ...]:
    results = tuple(names.fresh() for _ in indices)
    for result, index in zip(results, indices, strict=True):
        lines.append(f"    {result} = arith.constant {index} : index")
    return results


def builtin_argument(node: ProgramGraphNode, index: int) -> str:
    if not isinstance(node.payload, BuiltinOp):
        raise ValueError("builtin node is missing its typed payload")
    builtin = node.payload.name
    return (
        f"%builtin{index}: {native_type(node.type)} "
        f'{{vernon.interface = "input", vernon.source_name = "{node.source_name or builtin}", '
        f'vernon.builtin = "{builtin}"}}'
    )


def invocation_indices(
    global_id: str,
    global_id_type: str,
    names: Names,
    lines: list[str],
) -> tuple[str, str, str]:
    results: list[str] = []
    for axis in range(3):
        index = names.fresh()
        component = names.fresh()
        result = names.fresh()
        lines.append(f"    {index} = arith.constant {axis} : index")
        lines.append(f"    {component} = tensor.extract {global_id}[{index}] : {global_id_type}")
        lines.append(f"    {result} = arith.index_castui {component} : i32 to index")
        results.append(result)
    return results[0], results[1], results[2]


def entry_header(
    symbol: str,
    arguments: str,
    workgroup_size: tuple[int, int, int],
    storage_effects: tuple[str, ...] = (),
) -> str:
    effects = ", ".join(storage_effects)
    return (
        f"  func.func @{symbol}({arguments}) "
        'attributes {vernon.entry, vernon.stage = "compute", '
        f"vernon.storage_effects = [{effects}], "
        f"vernon.workgroup_size = array<i32: {', '.join(str(value) for value in workgroup_size)}>}} {{"
    )


__all__ = [
    "AutodiffNativeLoweringError",
    "Names",
    "builtin_argument",
    "entry_header",
    "index_constants",
    "invocation_indices",
    "module",
]
