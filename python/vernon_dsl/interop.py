"""Explicit low-level host/runtime interoperability APIs.

These objects are not Vernon source-language types and cannot appear in parsed
kernel or shader annotations.
"""

from ._runtime.resources import RawBuffer

__all__ = ["RawBuffer"]
