from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def embed_text_artifacts(manifest: Path, output: Path) -> None:
    root = json.loads(manifest.read_text(encoding="utf-8"))
    bundle_root = manifest.parent.resolve()
    for stage in root["stage_artifacts"].values():
        descriptor = stage["artifact"]
        if descriptor.get("storage") != "external":
            continue
        if descriptor.get("format") not in {"glsl", "gles"}:
            raise ValueError(f"cannot embed non-text artifact format {descriptor.get('format')!r}")
        artifact = (bundle_root / descriptor["path"]).resolve()
        if bundle_root not in artifact.parents:
            raise ValueError(f"artifact escapes bundle directory: {descriptor['path']}")
        data = artifact.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != descriptor.get("size") or digest != descriptor.get("sha256"):
            raise ValueError(f"artifact integrity check failed: {descriptor['path']}")
        stage["artifact"] = {
            "format": descriptor["format"],
            "storage": "inline",
            "encoding": "utf8",
            "data": data.decode("utf-8"),
            "size": len(data),
            "sha256": digest,
        }

    root.pop("content_hash", None)
    root["content_hash"] = hashlib.sha256(_canonical_json(root).encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical_json(root) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    embed_text_artifacts(arguments.manifest, arguments.output)


if __name__ == "__main__":
    main()
