"""Program-level Module parsing and MLIR emission.

Kernel function compilation lives in :mod:`vernon_dsl.frontend`; this package
only models scheduling-level forward and backward graphs.
"""

from .model import MlirOperation, MlirValue, ParsedProgram, ProgramImplementation, ProgramType
from .parser import parse_program
from .providers import (
    BuiltinDslProvider,
    CapturedDslProvider,
    CapturedVjpDslProvider,
    ProviderChain,
)

__all__ = [
    "MlirOperation",
    "MlirValue",
    "BuiltinDslProvider",
    "CapturedDslProvider",
    "CapturedVjpDslProvider",
    "ParsedProgram",
    "ProgramImplementation",
    "ProgramType",
    "ProviderChain",
    "parse_program",
]
