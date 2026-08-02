from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

WHEEL_PATTERN = re.compile(
    r"^vernon_lang-(?P<version>[^-]+)-"
    r"(?P<python>cp\d+)-(?P<abi>cp\d+)-(?P<platform>[^-]+)\.whl$"
)
PYTHON_TAGS = {"cp311", "cp312", "cp313", "cp314"}
PLATFORMS = {
    "windows-x64": lambda value: value == "win_amd64",
    "linux-x64": lambda value: "manylinux" in value and value.endswith("_x86_64"),
    "macos-arm64": lambda value: value == "macosx_15_0_arm64",
}


def fail(message: str) -> None:
    raise SystemExit(message)


def wheel_identity(path: Path, expected_version: str) -> tuple[str, str]:
    match = WHEEL_PATTERN.fullmatch(path.name)
    if match is None:
        fail(f"unexpected wheel filename: {path.name}")
    if match["version"] != expected_version:
        fail(f"{path.name}: expected version {expected_version}")
    if match["python"] not in PYTHON_TAGS or match["abi"] != match["python"]:
        fail(f"{path.name}: expected one CPython 3.11-3.14 ABI-specific wheel")
    platform_name = next(
        (name for name, accepts in PLATFORMS.items() if accepts(match["platform"])),
        None,
    )
    if platform_name is None:
        fail(f"{path.name}: unsupported platform tag {match['platform']}")

    metadata_suffix = f"vernon_lang-{expected_version}.dist-info/METADATA"
    with zipfile.ZipFile(path) as wheel:
        candidates = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if candidates != [metadata_suffix]:
            fail(f"{path.name}: expected metadata {metadata_suffix}, found {candidates}")
        metadata = wheel.read(metadata_suffix).decode("utf-8")
    if f"\nVersion: {expected_version}\n" not in f"\n{metadata}":
        fail(f"{path.name}: embedded metadata version does not match {expected_version}")
    return match["python"], platform_name


def main() -> int:
    if len(sys.argv) != 3:
        fail("usage: validate_release_artifacts.py DIST_DIRECTORY VERSION")
    directory = Path(sys.argv[1])
    expected_version = sys.argv[2]
    wheels = sorted(directory.glob("*.whl"))
    identities = {wheel_identity(path, expected_version) for path in wheels}
    expected = {(python, platform) for python in PYTHON_TAGS for platform in PLATFORMS}
    if len(wheels) != len(expected) or identities != expected:
        missing = sorted(expected - identities)
        unexpected = sorted(identities - expected)
        fail(f"expected {len(expected)} unique wheels, found {len(wheels)}; missing={missing}, unexpected={unexpected}")
    print(f"validated {len(wheels)} VernonDSL {expected_version} release wheels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
