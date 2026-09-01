from __future__ import annotations

from typing import Any

from .resource_common import _session_state


class SamplerState:
    """Immutable runtime sampler bound to a shader ``Sampler`` parameter."""

    def __init__(self, *, address: str = "repeat"):
        if address not in {"repeat", "clamp_to_edge", "mirrored_repeat"}:
            raise ValueError("sampler address must be 'repeat', 'clamp_to_edge', or 'mirrored_repeat'")
        self._address = address
        self._native_sampler: Any | None = None
        self._native_generation = -1
        _session_state()._runtime_children.add(self)

    def _release_runtime_native(self) -> None:
        self._native_sampler = None
        self._native_generation = -1

    def _resident_sampler(self) -> Any:
        state = _session_state()
        if state._native_runtime is None or state._rhi_host is None:
            raise RuntimeError("SamplerState requires an initialized GPU RHI runtime")
        if self._native_sampler is None or self._native_generation != state._runtime_generation:
            address = {
                "repeat": state._native.SamplerAddressMode.REPEAT,
                "clamp_to_edge": state._native.SamplerAddressMode.CLAMP_TO_EDGE,
                "mirrored_repeat": state._native.SamplerAddressMode.MIRRORED_REPEAT,
            }[self._address]
            self._native_sampler = state._rhi_host.create_sampler(address)
            self._native_generation = state._runtime_generation
        return self._native_sampler


def sampler(*, address: str = "repeat") -> SamplerState:
    return SamplerState(address=address)
