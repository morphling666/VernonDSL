from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("object_suffix")
    arguments = parser.parse_args()

    document = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    output = arguments.manifest.parent
    artifacts = sorted(Path(record["artifact"]["path"]) for record in document["stage_artifacts"].values())
    if len(artifacts) != 3:
        raise ValueError("cooked autodiff fixture must contain primal, forward, and backward objects")
    for index, artifact in enumerate(artifacts):
        shutil.copyfile(output / artifact, output / f"autodiff_artifact_{index}{arguments.object_suffix}")

    registration = document["cpu_static_registration"]
    shutil.copyfile(output / registration["source"], output / "autodiff_registration.c")
    (output / "autodiff_registration_wrapper.c").write_text(
        "\n".join(
            [
                '#include "VernonRuntime.h"',
                "",
                f"extern VernonStatus {registration['function']}(void);",
                "",
                "VernonStatus vernonRegisterAutodiffFixture(void) {",
                f"    return {registration['function']}();",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
