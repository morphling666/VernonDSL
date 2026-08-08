from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    manifest = Path(sys.argv[1])
    sys.path[:0] = [sys.argv[2], sys.argv[3]]

    import numpy as np
    import vernon_dsl as vd
    import vernon_native_autodiff_f16_fixture  # noqa: F401
    from vernon_dsl import _native

    runtime = _native.Runtime(_native.RuntimeBackend.CPU)
    pipeline = runtime.load_pipeline_asset(manifest.read_bytes(), str(manifest.parent), [])

    objective = vd.storage.zeros(dtype=vd.f16, shape=(1,))
    output, pullback = pipeline.vjp({"x": np.float16(1.5), "output": objective}, (1, 1, 1))
    assert output is None
    np.testing.assert_array_equal(objective.to_numpy(), np.array([2.25], dtype=np.float16))

    implicit = pullback(None)
    np.testing.assert_array_equal(implicit["x"], np.float32(3.0))
    assert implicit["x"].dtype == np.dtype(np.float32)

    explicit = pullback(np.array([2.0], dtype=np.float32))
    np.testing.assert_array_equal(explicit["x"], np.float32(6.0))
    assert explicit["x"].dtype == np.dtype(np.float32)

    vd.init(arch=vd.cpu)
    cooked = vd.load_cooked_vjp_asset(manifest)
    wrapped_objective = vd.storage.zeros(dtype=vd.f16, shape=(1,))
    _, wrapped_pullback = cooked.vjp(
        {"x": np.float16(1.5), "output": wrapped_objective},
        (1, 1, 1),
    )
    cotangent = vd.storage.tangent_zeros(dtype=vd.f16, shape=(1,))
    cotangent.copy_from_numpy(np.array([2.0], dtype=np.float32))
    wrapped = wrapped_pullback(cotangent)
    np.testing.assert_array_equal(wrapped["x"], np.float32(6.0))


if __name__ == "__main__":
    main()
