"""Cooking of Host module Program Assets: one native object that registers its reflection table under the asset id,
plus the C header and reflection JSON applications build against."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..bundle import CpuTargetOptions, ProgramCompileError, TargetOptions
from .artifact_io import prune_stale_artifacts, write_external_artifact
from .declaration import ProgramAssetDeclaration

HOST_MODULE_VERSION = 1


def cook_host_asset(declaration: ProgramAssetDeclaration, output: Path, target: TargetOptions) -> Path:
    """Writes `<dir>/<dir name>.program.json`, `<dir name>.h`, `<dir name>.json`, and the object under `artifacts/`.

    The manifest lists the object in `blobs` like a CPU Program's, so build rules archive and link it the same way.
    """

    if not isinstance(target, CpuTargetOptions):
        raise ProgramCompileError(f"Host module Program Assets cook only for the cpu target, not '{target.target}'")
    from ..diagnostics import CompileError
    from ..hostlang.aot import compile_host

    assert declaration.prefix is not None
    try:
        artifacts = compile_host(
            [Path(source) for source in declaration.host_sources],
            target=target.triple or None,
            prefix=declaration.prefix,
            cpu=target.processor,
            features=",".join(target.features),
            asset_id=declaration.id,
        )
    except (CompileError, ValueError) as error:
        raise ProgramCompileError(str(error)) from error

    output.mkdir(parents=True, exist_ok=True)
    name = output.name
    descriptor = write_external_artifact(
        output, artifacts.object, "relocatable_object", "host", f"{name}{artifacts.object_suffix}"
    )
    header = output / f"{name}.h"
    reflection = output / f"{name}.json"
    header.write_text(artifacts.header, encoding="utf-8", newline="\n")
    reflection_text = json.dumps(artifacts.reflection, indent=2) + "\n"
    reflection.write_text(reflection_text, encoding="utf-8", newline="\n")

    options: dict[str, Any] = {"triple": artifacts.triple}
    if target.processor:
        options["processor"] = target.processor
    if target.features:
        options["features"] = list(target.features)
    manifest: dict[str, Any] = {
        "type": "host_module",
        "host_module_version": HOST_MODULE_VERSION,
        "id": declaration.id,
        "target": {"kind": "cpu", "options": options},
        "blobs": {
            descriptor["sha256"]: {
                "byte_length": descriptor["size"],
                "location": {"tag": "external", "uri": descriptor["path"]},
                "sha256": descriptor["sha256"],
            }
        },
        "host_module": {
            "table": f"{declaration.prefix}module",
            "prefix": declaration.prefix,
            "object": descriptor["sha256"],
            "header": header.name,
            "reflection": reflection.name,
        },
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8"))
    digest.update(artifacts.header.encode("utf-8"))
    digest.update(reflection_text.encode("utf-8"))
    manifest["content_hash"] = digest.hexdigest()
    path = output / f"{name}.program.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    prune_stale_artifacts(output, {output / descriptor["path"]})
    return path


__all__ = ["HOST_MODULE_VERSION", "cook_host_asset"]
