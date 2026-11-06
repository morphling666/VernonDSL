from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "source/include/VernonFormats.def"
OUTPUT = ROOT / "python/vernon_dsl/_formats.py"

_COLUMNS = (
    "name",
    "NAME",
    "texture",
    "component_class",
    "pixel_bytes",
    "data_format",
    "data_type",
    "baseline",
    "gl_internal_format",
    "gl_format",
    "gl_type",
    "vulkan",
    "dxgi",
    "metal",
    "spirv_storage",
)
_COMPONENT_CLASSES = frozenset({"float", "unorm", "snorm", "uint", "sint"})
_CAPABILITIES = ("renderable", "blendable", "filterable", "storage", "resolvable", "multisample", "copyable")
_ELEMENT_BYTES = {
    "UINT8": ("uint8", 1),
    "INT8": ("int8", 1),
    "UINT16": ("uint16", 2),
    "INT16": ("int16", 2),
    "FLOAT16": ("float16", 2),
    "UINT32": ("uint32", 4),
    "INT32": ("int32", 4),
    "FLOAT32": ("float32", 4),
}
_ASPECTS = {"DEPTH": "depth", "DEPTH_STENCIL": "depth stencil"}
_COLOR_COMPONENTS = {"RED": 1, "RG": 2, "RGB": 3, "RGBA": 4, "BGRA": 4}
# A packed integer texel holds several components in one element, so its component widths are not the element's.
_PACKED_INTEGER_BITS = {"rgb10a2_uint": (10, 10, 10, 2)}


@dataclass(frozen=True)
class _Row:
    name: str
    native_name: str
    texture: int
    component_class: str
    element: str
    channels: int
    components: int
    integer_bits: tuple[int, ...]
    pixel_bytes: int
    aspects: str
    capabilities: str


def _string(value: str, column: str, row: str) -> str:
    match = re.fullmatch(r'"([^"]*)"', value)
    if match is None:
        raise ValueError(f"{row}: {column} must be a string literal")
    return match.group(1)


def _parse_row(arguments: str) -> _Row:
    values = [value.strip() for value in re.split(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", arguments)]
    if len(values) != len(_COLUMNS):
        raise ValueError(f"VERNON_FORMAT({arguments.strip()[:40]}...) has {len(values)} columns, not {len(_COLUMNS)}")
    column = dict(zip(_COLUMNS, values, strict=True))
    name = column["name"]
    if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None or column["NAME"] != name.upper():
        raise ValueError(f"{name}: name must be lowercase and NAME its uppercase spelling")
    component_class = _string(column["component_class"], "component_class", name)
    if component_class not in _COMPONENT_CLASSES:
        raise ValueError(f"{name}: unknown component class {component_class!r}")
    words = _string(column["baseline"], "baseline", name).split()
    unknown = sorted(set(words) - set(_CAPABILITIES))
    if unknown or len(set(words)) != len(words):
        raise ValueError(f"{name}: baseline has unknown or repeated capabilities {unknown}")
    if ("storage" in words) != (column["spirv_storage"] != "Unknown"):
        raise ValueError(f"{name}: spirv_storage must be Unknown exactly when baseline lacks storage")
    if column["data_type"] not in _ELEMENT_BYTES:
        raise ValueError(f"{name}: unknown data type {column['data_type']}")
    element, element_bytes = _ELEMENT_BYTES[column["data_type"]]
    pixel_bytes = int(column["pixel_bytes"])
    data_format = column["data_format"]
    if data_format in _ASPECTS:
        channels = 1
    elif data_format in _COLOR_COMPONENTS and pixel_bytes % element_bytes == 0:
        channels = pixel_bytes // element_bytes
    else:
        raise ValueError(f"{name}: {pixel_bytes} bytes do not hold whole {data_format} {element} texels")
    components = _COLOR_COMPONENTS.get(data_format, 1)
    integer_bits: tuple[int, ...] = ()
    if component_class in ("uint", "sint"):
        integer_bits = _PACKED_INTEGER_BITS.get(name, (8 * element_bytes,) * components)
        if channels not in (1, components) or len(integer_bits) != components:
            raise ValueError(f"{name}: integer components do not fill {data_format} {element} texels")
    return _Row(
        name=name,
        native_name=column["NAME"],
        texture=int(column["texture"]),
        component_class=component_class,
        element=element,
        channels=channels,
        components=components,
        integer_bits=integer_bits,
        pixel_bytes=pixel_bytes,
        aspects=_ASPECTS.get(data_format, "color"),
        capabilities=" ".join(word for word in _CAPABILITIES if word in words),
    )


def _rows() -> list[_Row]:
    text = re.sub(r"/\*.*?\*/", "", TABLE.read_text(encoding="utf-8"), flags=re.DOTALL)
    rows = [_parse_row(arguments) for arguments in re.findall(r"^VERNON_FORMAT\(([^)]*)\)", text, re.MULTILINE)]
    if not rows:
        raise ValueError(f"{TABLE.relative_to(ROOT).as_posix()} has no VERNON_FORMAT rows")
    for key in ("name", "texture"):
        values = [getattr(row, key) for row in rows]
        if len(set(values)) != len(values):
            raise ValueError(f"VernonFormats.def repeats a {key}")
    if any(row.texture < 0 for row in rows):
        raise ValueError("VernonFormats.def texture values must be nonnegative")
    return rows


def _rendered() -> str:
    rows = _rows()
    lines = [
        "# Generated by tools/generate_formats.py from source/include/VernonFormats.def. Do not edit.",
        "",
        "from ._texture_format import TextureFormat",
        "",
        "",
        "def _words(text: str) -> frozenset[str]:",
        "    return frozenset(text.split())",
        "",
        "",
    ]
    for row in rows:
        lines += [
            f"{row.name} = TextureFormat(",
            f'    name="{row.name}",',
            f'    _native_name="{row.native_name}",',
            f'    component_class="{row.component_class}",',
            f'    element="{row.element}",',
            f"    channels={row.channels},",
            f"    components={row.components},",
            f"    integer_bits={row.integer_bits!r},",
            f"    pixel_bytes={row.pixel_bytes},",
            f'    aspects=_words("{row.aspects}"),',
            f'    capabilities=_words("{row.capabilities}"),',
            ")",
        ]
    lines += ["", "TEXTURE_FORMATS = ("]
    lines += [f"    {row.name}," for row in rows]
    lines += [")", "TEXTURE_FORMATS_BY_NAME = {format.name: format for format in TEXTURE_FORMATS}", "", "__all__ = ["]
    lines += [f'    "{row.name}",' for row in rows]
    lines += ["]", ""]
    return "\n".join(lines)


def _existing_newline(path: Path) -> str:
    if path.exists() and b"\r\n" in path.read_bytes():
        return "\r\n"
    return "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the generated format module is stale")
    args = parser.parse_args()

    expected = _rendered()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != expected:
            print(f"stale generated format module: {OUTPUT.relative_to(ROOT).as_posix()}", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(expected, encoding="utf-8", newline=_existing_newline(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
