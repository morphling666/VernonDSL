from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import vernon_dsl as vd
from vernon_dsl import _native as native
from vernon_dsl._formats import TEXTURE_FORMATS


def test_generated_formats_are_current() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, str(root / "tools/generate_formats.py"), "--check"],
        cwd=root,
        check=True,
    )


def test_native_texture_formats_match_the_table() -> None:
    assert set(native.TextureFormat.__members__) == {format._native_name for format in TEXTURE_FORMATS}


def test_every_table_format_is_public() -> None:
    for format in TEXTURE_FORMATS:
        assert getattr(vd, format.name) is format
        assert format.name in vd.__all__


def test_texel_layout_matches_pixel_size() -> None:
    for format in TEXTURE_FORMATS:
        assert format.dtype.itemsize * format.channels == format.pixel_bytes, format.name
