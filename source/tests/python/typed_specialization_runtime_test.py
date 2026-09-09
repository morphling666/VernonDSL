from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    manifest = Path(sys.argv[1])
    sys.path[:0] = [sys.argv[2], sys.argv[3]]

    import vernon_dsl as vd
    import vernon_typed_specialization_fixture  # noqa: F401

    vd.init(arch=vd.cpu)
    extent_specialization = vd.specialization("extent", vd.u32)
    for extent in (1, 4):
        output = vd.storage.zeros(dtype=vd.f32, shape=(extent,))
        program = vd.load_program(manifest, specializations={extent_specialization: extent})
        program(output)
        assert float(output.to_numpy()[0]) == float(extent)


if __name__ == "__main__":
    main()
