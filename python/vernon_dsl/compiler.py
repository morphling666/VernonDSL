"""Public Python frontend facade.

Implementation phases live under :mod:`vernon_dsl.frontend`; this module keeps
the stable compile API without owning parsing, inference, or lowering logic.
"""

from .frontend.compiler import Compiler, compile_file, compile_source
from .frontend.request import FrontendCompileRequest, FrontendCompileResult

__all__ = [
    "Compiler",
    "FrontendCompileRequest",
    "FrontendCompileResult",
    "compile_file",
    "compile_source",
]
