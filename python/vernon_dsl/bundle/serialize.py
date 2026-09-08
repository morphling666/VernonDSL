from __future__ import annotations

import base64
import hashlib
from typing import Any, Mapping

from .types import CompiledArtifact, ProgramCompileError, canonical_json


def content_hash(value: Mapping[str, Any]) -> str:
    unhashed = dict(value)
    unhashed.pop("content_hash", None)
    return hashlib.sha256(canonical_json(unhashed).encode("utf-8")).hexdigest()


def with_content_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_hash", None)
    result["content_hash"] = content_hash(result)
    return result


def inline_artifact_descriptor(artifact: CompiledArtifact) -> dict[str, Any]:
    encoding = "base64" if artifact.format in {"spirv", "dxil"} else "utf8"
    try:
        data = (
            base64.b64encode(artifact.data).decode("ascii") if encoding == "base64" else artifact.data.decode("utf-8")
        )
    except UnicodeDecodeError:
        raise ProgramCompileError(f"{artifact.format} runtime artifact is not UTF-8") from None
    return {
        "format": artifact.format,
        "storage": "inline",
        "encoding": encoding,
        "data": data,
        "size": len(artifact.data),
        "sha256": artifact.sha256,
    }


def serialize_bundle(bundle: Mapping[str, Any]) -> bytes:
    return (canonical_json(with_content_hash(bundle)) + "\n").encode("utf-8")


__all__ = [
    "canonical_json",
    "content_hash",
    "inline_artifact_descriptor",
    "serialize_bundle",
    "with_content_hash",
]
