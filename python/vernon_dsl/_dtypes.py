from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np

from .language.scalar_types import SCALAR_TYPES

NUMPY_DTYPE_BY_SCALAR = MappingProxyType({name: np.dtype(scalar.numpy) for name, scalar in SCALAR_TYPES.items()})
SCALAR_BY_NUMPY_DTYPE = MappingProxyType({dtype: name for name, dtype in NUMPY_DTYPE_BY_SCALAR.items()})


def numpy_dtype(scalar: str) -> np.dtype[Any]:
    try:
        return NUMPY_DTYPE_BY_SCALAR[scalar]
    except KeyError:
        raise TypeError(f"unsupported Vernon scalar dtype {scalar!r}") from None


def scalar_name(dtype: Any) -> str | None:
    return SCALAR_BY_NUMPY_DTYPE.get(np.dtype(dtype))


__all__ = [
    "NUMPY_DTYPE_BY_SCALAR",
    "SCALAR_BY_NUMPY_DTYPE",
    "numpy_dtype",
    "scalar_name",
]
