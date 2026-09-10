from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from language_contract_pytest import reject_skipped_contract_test


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> Generator[None, Any, Any]:
    report = yield
    reject_skipped_contract_test(report, item.obj)
    return report
