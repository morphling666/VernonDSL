from __future__ import annotations

import importlib
from typing import Any


def _session_state() -> Any:
    return importlib.import_module("vernon_dsl._runtime.session")
