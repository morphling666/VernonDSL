from __future__ import annotations

import hashlib
from pathlib import Path


def source_text_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def source_file_digest(path: Path) -> str | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return source_text_digest(source)
