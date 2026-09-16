from __future__ import annotations

from vernon_dsl import _native as native
from vernon_dsl.frontend.compiler import compile_source

SOURCE = """
from typing import Annotated
from vernon_dsl import *

@kernel(workgroup_size=(8, 1, 1))
def cross_lane_device_epoch(
    scratch: TensorView[f32, (dyn,), read_write],
    output: TensorView[f32, (dyn,), write],
    gid: Annotated[Tensor[u32, (3,)], builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    scratch[index] = 1.0
    workgroup_barrier()
    neighbor = (index + 1) % 8
    output[index] = scratch[neighbor]
"""


def main() -> None:
    module = compile_source(SOURCE, "kernel_memory_phase_compile_fixture.py")
    result = native.Compiler().compile_program_result(module, native.Target.CPU)
    assert not result.ok, "cross-lane device communication unexpectedly compiled"
    assert "not a global completion boundary" in result.diagnostics, result.diagnostics
    assert "multi-kernel graph" in result.diagnostics, result.diagnostics


if __name__ == "__main__":
    main()
