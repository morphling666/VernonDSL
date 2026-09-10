from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from backend_test_matrix import CapabilityUnavailable
from language_contract_traceability import contract_test_bindings


class MutableTestReport(Protocol):
    outcome: str
    longrepr: Any

    @property
    def skipped(self) -> bool: ...


def reject_skipped_contract_test(
    report: MutableTestReport,
    test_method: Callable[..., Any],
    exception: BaseException | None = None,
) -> None:
    bindings = contract_test_bindings(test_method)
    if not report.skipped or not bindings:
        return
    if isinstance(exception, CapabilityUnavailable):
        return
    case_ids = sorted({case_id for binding in bindings for case_id in binding.case_ids})
    report.outcome = "failed"
    report.longrepr = (
        f"contract-bound tests may not be skipped; the following cases would lose required coverage: {case_ids}"
    )
