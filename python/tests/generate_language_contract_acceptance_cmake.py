from __future__ import annotations

import argparse
from pathlib import Path

from language_contract_cases import LANGUAGE_CONTRACT_REGISTRY, AcceptanceSuite


def _requirements(acceptance) -> str:
    requirements = set()
    if acceptance.requirements.compute:
        requirements.add("compute")
    if acceptance.requirements.graphics:
        requirements.add("graphics")
    if acceptance.requirements.texture_sampler_operations or acceptance.requirements.storage_texture:
        requirements.add("texture")
    if acceptance.requirements.program_vjp:
        requirements.add("autodiff")
    if acceptance.suite is AcceptanceSuite.GPU_AUTODIFF:
        requirements.add("gpu")
    if acceptance.suite is AcceptanceSuite.SYNCHRONIZATION:
        requirements.add("synchronization")
    if not requirements:
        raise ValueError(f"{acceptance.id} has no fixture requirements")
    return ",".join(sorted(requirements))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpp-output", type=Path, required=True)
    arguments = parser.parse_args()

    rows = []
    for acceptance in LANGUAGE_CONTRACT_REGISTRY.acceptances:
        source, separator, symbol = acceptance.asset_reference.partition(":")
        if separator != ":" or not symbol.isidentifier():
            raise ValueError(f"{acceptance.id} has an invalid asset reference")
        asset = (arguments.repository / source).resolve()
        if not asset.is_file():
            raise FileNotFoundError(asset)
        rows.append(f'    "{acceptance.suite.value}|{acceptance.id}|{asset}:{symbol}|{_requirements(acceptance)}"')

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "set(VERNON_LANGUAGE_CONTRACT_FIXTURE_SPECS\n" + "\n".join(rows) + "\n)\n",
        encoding="utf-8",
    )
    arguments.cpp_output.parent.mkdir(parents=True, exist_ok=True)
    expected_arrays = []
    descriptors = []
    runtime_acceptances = tuple(
        acceptance
        for acceptance in LANGUAGE_CONTRACT_REGISTRY.acceptances
        if acceptance.suite is AcceptanceSuite.LANGUAGE_CONTRACT
    )
    for index, acceptance in enumerate(runtime_acceptances):
        values = ", ".join(repr(float(value)) for value in acceptance.oracle.expected)
        expected_arrays.append(f"constexpr double acceptanceExpected{index}[]{{{values or '0.0'}}};\n")
        descriptors.append(
            "    {"
            f'"{acceptance.id}", '
            f"{str(acceptance.requirements.compute).lower()}, "
            f"{str(acceptance.requirements.graphics).lower()}, "
            f"{str(acceptance.requirements.storage_buffers).lower()}, "
            f"{str(acceptance.requirements.storage_texture).lower()}, "
            f"{str(acceptance.requirements.device_atomics).lower()}, "
            f"{str(acceptance.requirements.f32_atomic_add).lower()}, "
            f"{str(acceptance.requirements.texture_sampler_operations).lower()}, "
            f"{str(acceptance.requirements.program_vjp).lower()}, "
            f'"{acceptance.oracle.kind.value}", '
            f'"{acceptance.oracle.parameter}", '
            f"{acceptance.oracle.grid[0]}, {acceptance.oracle.grid[1]}, {acceptance.oracle.grid[2]}, "
            f"acceptanceExpected{index}, {len(acceptance.oracle.expected)}"
            "},\n"
        )
    arguments.cpp_output.write_text(
        "".join(expected_arrays)
        + "constexpr AcceptanceDescriptor acceptanceDescriptors[]{\n"
        + "".join(descriptors)
        + "};\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
