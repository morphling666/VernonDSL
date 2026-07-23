from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from ..types import Feature


@dataclass(frozen=True)
class PipelineAssetDeclaration:
    id: str
    stages: dict[str, Callable[..., Any]]
    variants: tuple[tuple[Feature, ...], ...]
    targets: dict[str, dict[str, Any]]


def pipeline_asset(
    *,
    id: str,
    compute: Callable[..., Any] | None = None,
    vertex: Callable[..., Any] | None = None,
    fragment: Callable[..., Any] | None = None,
    variants: Iterable[Iterable[Feature]] = ((),),
    targets: Mapping[str, Mapping[str, Any]],
) -> PipelineAssetDeclaration:
    """Declare a cookable pipeline without creating a runtime pipeline."""

    stages = {
        name: value
        for name, value in (("compute", compute), ("vertex", vertex), ("fragment", fragment))
        if value is not None
    }
    return PipelineAssetDeclaration(
        id=id,
        stages=stages,
        variants=tuple(tuple(key) for key in variants),
        targets={name: dict(options) for name, options in targets.items()},
    )
