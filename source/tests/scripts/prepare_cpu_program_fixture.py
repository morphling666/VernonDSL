from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("object_suffix")
    parser.add_argument("--wrapper-function", required=True)
    parser.add_argument("--archiver", required=True)
    arguments = parser.parse_args()

    document = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    output = arguments.manifest.parent
    variants = document.get("variants")
    blobs = document.get("blobs")
    if document.get("type") != "program" or not isinstance(variants, list) or not variants:
        raise ValueError("cooked CPU Program fixture must contain at least one variant")
    if not isinstance(blobs, dict):
        raise ValueError("cooked CPU Program fixture has no canonical artifact system")
    records: list[dict[str, Any]] = []
    for variant in variants:
        artifact_system = variant.get("artifact_system")
        variant_records = artifact_system.get("artifacts") if isinstance(artifact_system, dict) else None
        if not isinstance(variant_records, dict):
            raise ValueError("cooked CPU Program fixture has no canonical artifact system")
        records.append(variant_records)
    digests = {
        module["blob"]
        for variant_records in records
        for record in variant_records.values()
        for module in record.get("modules", ())
        if module.get("format") == "relocatable_object"
    }
    artifacts = sorted(
        Path(blobs[digest]["location"]["uri"])
        for digest in digests
        if digest in blobs and blobs[digest].get("location", {}).get("tag") == "external"
    )
    if not artifacts:
        raise ValueError("cooked CPU Program fixture contains no relocatable objects")
    artifact_archive = output / "program_artifacts.a"
    artifact_archive.unlink(missing_ok=True)
    subprocess.run(
        [
            arguments.archiver,
            "rcs",
            str(artifact_archive),
            *(str(output / artifact) for artifact in artifacts),
        ],
        check=True,
    )

    registration_sources = sorted(output.glob("vernon_cpu_registration_*.c"))
    if len(registration_sources) != 1:
        raise ValueError("cooked CPU Program fixture must contain exactly one CPU registration source")
    registration_source = registration_sources[0]
    identity_suffix = registration_source.stem.removeprefix("vernon_cpu_registration_")
    registration_function = f"vernonRegisterCpuArtifacts_{identity_suffix}"
    shutil.copyfile(registration_source, output / "program_registration.c")
    (output / "program_registration_wrapper.c").write_text(
        "\n".join(
            [
                '#include "VernonRuntime.h"',
                "",
                f"extern VernonStatus {registration_function}(void);",
                "",
                f"VernonStatus {arguments.wrapper_function}(void) {{",
                f"    return {registration_function}();",
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
