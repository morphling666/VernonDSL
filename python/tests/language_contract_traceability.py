from __future__ import annotations

import ast
import contextvars
import functools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeVar

from language_contract_cases import CONTRACT_CASE_GROUPS, LanguageContractCase

_TEST_LAYERS: Final = frozenset({"F", "I"})
_BINDINGS_ATTRIBUTE: Final = "__language_contract_bindings__"
_TestMethod = TypeVar("_TestMethod", bound=Callable[..., Any])
_OBSERVED_CASE_IDS: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "language_contract_observed_case_ids", default=None
)


@dataclass(frozen=True)
class ContractTestBinding:
    case_ids: tuple[str, ...]
    layers: frozenset[str]
    test_id: str


def contract_test_bindings(method: Callable[..., Any]) -> tuple[ContractTestBinding, ...]:
    bindings = getattr(method, _BINDINGS_ATTRIBUTE, ())
    if not isinstance(bindings, tuple) or not all(isinstance(binding, ContractTestBinding) for binding in bindings):
        raise TypeError("invalid language contract binding metadata")
    return bindings


def _attach_binding(
    traced: Callable[..., Any],
    method: Callable[..., Any],
    case_ids: tuple[str, ...],
    layers: frozenset[str],
) -> None:
    binding = ContractTestBinding(case_ids, layers, f"{method.__module__}.{method.__qualname__}")
    setattr(traced, _BINDINGS_ATTRIBUTE, contract_test_bindings(method) + (binding,))


def mark_case_observed(case_id: str) -> None:
    observed = _OBSERVED_CASE_IDS.get()
    if observed is None:
        raise AssertionError(f"{case_id} was observed outside a contract-bound test")
    observed.add(case_id)


def _run_traced(
    method: Callable[..., Any],
    case_ids: tuple[str, ...],
    layers: frozenset[str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    observed = _OBSERVED_CASE_IDS.get()
    owner = observed is None
    token = _OBSERVED_CASE_IDS.set(set()) if owner else None
    try:
        result = method(*args, **kwargs)
        current = _OBSERVED_CASE_IDS.get()
        assert current is not None
        missing = set(case_ids) - current
        if missing:
            raise AssertionError(f"{method.__qualname__} did not execute an oracle for cases: {sorted(missing)}")
        return result
    finally:
        if owner:
            assert token is not None
            _OBSERVED_CASE_IDS.reset(token)


def _validate_layers(layers: str) -> frozenset[str]:
    result = frozenset(layers)
    if not result or not result <= _TEST_LAYERS:
        raise ValueError(f"contract test layers must be a non-empty subset of FI: {layers!r}")
    return result


def covers_case(case_id: str, *, layers: str) -> Callable[[_TestMethod], _TestMethod]:
    validated_layers = _validate_layers(layers)

    def decorate(method: _TestMethod) -> _TestMethod:
        @functools.wraps(method)
        def traced(*args: Any, **kwargs: Any) -> Any:
            return _run_traced(method, (case_id,), validated_layers, args, kwargs)

        _attach_binding(traced, method, (case_id,), validated_layers)
        return traced  # type: ignore[return-value]

    return decorate


def covers_case_group(group_name: str, *, layers: str) -> Callable[[_TestMethod], _TestMethod]:
    if not group_name.isidentifier():
        raise ValueError(f"contract case group must be an identifier: {group_name!r}")
    validated_layers = _validate_layers(layers)

    def decorate(method: _TestMethod) -> _TestMethod:
        @functools.wraps(method)
        def traced(*args: Any, **kwargs: Any) -> Any:
            case_ids = tuple(case.id for case in CONTRACT_CASE_GROUPS[group_name])
            return _run_traced(method, case_ids, validated_layers, args, kwargs)

        case_ids = tuple(case.id for case in CONTRACT_CASE_GROUPS[group_name])
        _attach_binding(traced, method, case_ids, validated_layers)
        return traced  # type: ignore[return-value]

    return decorate


def _string_literal(node: ast.AST, *, field: str, test_id: str) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise AssertionError(f"{test_id} {field} must be a string literal")
    return node.value


def _decorator_name(decorator: ast.Call) -> str | None:
    return decorator.func.id if isinstance(decorator.func, ast.Name) else None


def _method_calls_case(method: ast.FunctionDef | ast.AsyncFunctionDef, case_id: str) -> bool:
    parents = {child: parent for parent in ast.walk(method) for child in ast.iter_child_nodes(parent)}
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "case_by_id"
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == case_id
        and not isinstance(parents.get(node), ast.Expr)
        for node in ast.walk(method)
    )


def _method_iterates_group(method: ast.FunctionDef | ast.AsyncFunctionDef, group_name: str) -> bool:
    return any(
        isinstance(node, (ast.For, ast.comprehension))
        and any(isinstance(child, ast.Name) and child.id == group_name for child in ast.walk(node.iter))
        for node in ast.walk(method)
    )


def _method_verifies_ir(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"assert_frontend_ir", "assert_verified_ir"}
        for node in ast.walk(method)
    )


def collect_test_bindings(
    test_files: Iterable[Path],
    *,
    case_groups: Mapping[str, tuple[LanguageContractCase, ...]],
) -> tuple[ContractTestBinding, ...]:
    bindings: list[ContractTestBinding] = []
    for path in test_files:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_decorators = {
            alias.asname or alias.name
            for node in module.body
            if isinstance(node, ast.ImportFrom) and node.module == "language_contract_traceability"
            for alias in node.names
            if alias.name in {"covers_case", "covers_case_group"}
        }
        for class_node in (node for node in module.body if isinstance(node, ast.ClassDef)):
            for method in (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
            ):
                test_id = f"{path.name}::{class_node.name}::{method.name}"
                for decorator in method.decorator_list:
                    if not isinstance(decorator, ast.Call):
                        continue
                    name = _decorator_name(decorator)
                    if name not in {"covers_case", "covers_case_group"}:
                        continue
                    if name not in imported_decorators:
                        raise AssertionError(f"{test_id} uses {name} without importing the canonical decorator")
                    if len(decorator.args) != 1:
                        raise AssertionError(f"{test_id} {name} requires exactly one positional argument")
                    keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
                    if set(keywords) != {"layers"}:
                        raise AssertionError(f"{test_id} {name} requires only the layers keyword")
                    layers = _validate_layers(_string_literal(keywords["layers"], field="layers", test_id=test_id))
                    if "I" in layers and not _method_verifies_ir(method):
                        raise AssertionError(f"{test_id} claims I without calling the canonical MLIR verifier")
                    reference = _string_literal(decorator.args[0], field="reference", test_id=test_id)
                    if name == "covers_case":
                        if not _method_calls_case(method, reference):
                            raise AssertionError(f"{test_id} declares {reference} without calling case_by_id")
                        case_ids = (reference,)
                    else:
                        if reference not in case_groups:
                            raise AssertionError(f"{test_id} references unknown case group {reference}")
                        if not _method_iterates_group(method, reference):
                            raise AssertionError(f"{test_id} declares {reference} without iterating that group")
                        case_ids = tuple(case.id for case in case_groups[reference])
                    bindings.append(ContractTestBinding(case_ids, layers, test_id))
    return tuple(bindings)
