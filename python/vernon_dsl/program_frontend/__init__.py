"""Program-level Module parsing and MLIR emission.

Kernel function compilation lives in :mod:`vernon_dsl.frontend`; this package
only models scheduling-level forward and backward graphs.
"""

from .model import GraphOperation, GraphValue, ParsedProgram, ProgramGraph, ProgramImplementation, ProgramType
from .parser import parse_program
from .providers import (
    BuiltinDslProvider,
    CapturedDslProvider,
    CapturedVjpDslProvider,
    DirectKernelDslProvider,
    ProviderChain,
)

__all__ = [
    "GraphOperation",
    "GraphValue",
    "BuiltinDslProvider",
    "CapturedDslProvider",
    "CapturedVjpDslProvider",
    "DirectKernelDslProvider",
    "ParsedProgram",
    "ProgramGraph",
    "ProgramImplementation",
    "ProgramType",
    "ProviderChain",
    "parse_program",
]
