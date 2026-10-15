from __future__ import annotations

from vernon_dsl.program_assets import ProgramCompileError


def native_available() -> bool:
    try:
        from vernon_dsl._program_assets.compile_orchestration import _native_module

        _native_module()
        return True
    except ProgramCompileError:
        return False


__all__ = ["native_available"]
