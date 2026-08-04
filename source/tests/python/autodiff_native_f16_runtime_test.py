from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    manifest = Path(sys.argv[1])
    sys.path[:0] = [sys.argv[2], sys.argv[3]]

    import numpy as np
    import vernon_native_autodiff_f16_fixture  # noqa: F401
    from vernon_dsl import _native

    runtime = _native.Runtime(_native.RuntimeBackend.CPU)
    pipeline = runtime.load_pipeline_asset(manifest.read_bytes(), str(manifest.parent), [])

    output, pullback = pipeline.vjp({"x": np.float16(1.5)}, (1, 1, 1))
    np.testing.assert_array_equal(output, np.float16(2.25))
    assert output.dtype == np.dtype(np.float16)

    implicit = pullback()
    np.testing.assert_array_equal(implicit["x"], np.float32(3.0))
    assert implicit["x"].dtype == np.dtype(np.float32)

    explicit = pullback(np.float32(2.0))
    np.testing.assert_array_equal(explicit["x"], np.float32(6.0))
    assert explicit["x"].dtype == np.dtype(np.float32)


if __name__ == "__main__":
    main()
