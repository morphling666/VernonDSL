from __future__ import annotations

import threading
import weakref
from typing import Any

from .session import RuntimeSession, _InvocationContext


class SamplerState:
    """Immutable runtime sampler bound to a shader ``Sampler`` parameter."""

    def __init__(self, *, address: str = "repeat"):
        if address not in {"repeat", "clamp_to_edge", "mirrored_repeat"}:
            raise ValueError("sampler address must be 'repeat', 'clamp_to_edge', or 'mirrored_repeat'")
        self._address = address
        self._residencies: weakref.WeakKeyDictionary[RuntimeSession, Any] = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def _resident_sampler(self, context: _InvocationContext) -> Any:
        state = context.session
        if state.rhi_host is None:
            raise RuntimeError("SamplerState requires an initialized GPU RHI runtime")
        with self._lock:
            resident = self._residencies.get(state)
            if resident is not None:
                return resident
            address = {
                "repeat": state.native.SamplerAddressMode.REPEAT,
                "clamp_to_edge": state.native.SamplerAddressMode.CLAMP_TO_EDGE,
                "mirrored_repeat": state.native.SamplerAddressMode.MIRRORED_REPEAT,
            }[self._address]
            resident = state.rhi_host.create_sampler(address)
            self._residencies[state] = resident
            return resident


def sampler(*, address: str = "repeat") -> SamplerState:
    return SamplerState(address=address)
