from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Sequence

from ..bundle import PipelineCompileError

_C_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def write_cpu_static_registration(output: Path, symbols: Sequence[str]) -> dict[str, Any]:
    ordered = tuple(sorted(set(symbols)))
    if not ordered or any(_C_SYMBOL.fullmatch(symbol) is None for symbol in ordered):
        raise PipelineCompileError("CPU artifacts contain an invalid static entry symbol")

    identity = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
    stem = f"vernon_cpu_registration_{identity[:16]}"
    function = f"vernonRegisterCpuArtifacts_{identity[:16]}"
    header_name = f"{stem}.h"
    source_name = f"{stem}.c"
    guard = f"VERNON_CPU_REGISTRATION_{identity[:16].upper()}_H"

    header = "\n".join(
        [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            '#include "VernonRuntime.h"',
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            f"VernonStatus {function}(void);",
            "",
            "#ifdef __cplusplus",
            "}",
            "#endif",
            "",
            f"#endif /* {guard} */",
            "",
        ]
    )
    source_lines = [
        f'#include "{header_name}"',
        "",
        *(f"extern VernonStatus {symbol}(const VernonCpuInvocation *invocation);" for symbol in ordered),
        "",
        f"VernonStatus {function}(void) {{",
    ]
    for symbol in ordered:
        source_lines.extend(
            [
                "    {",
                f'        const VernonStringView name = {{"{symbol}", {len(symbol)}}};',
                f"        const VernonStatus status = vernonRuntimeRegisterStaticCpuEntry(name, &{symbol});",
                "        if (status != VERNON_STATUS_OK)",
                "            return status;",
                "    }",
            ]
        )
    source_lines.extend(["    return VERNON_STATUS_OK;", "}", ""])

    (output / header_name).write_text(header, encoding="utf-8", newline="\n")
    (output / source_name).write_text("\n".join(source_lines), encoding="utf-8", newline="\n")
    return {
        "identity": identity,
        "header": header_name,
        "source": source_name,
        "function": function,
        "symbols": list(ordered),
    }


__all__ = ["write_cpu_static_registration"]
