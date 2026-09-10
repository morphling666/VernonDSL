from __future__ import annotations

import re
from collections.abc import Mapping

from language_contract_cases import LanguageContractCase
from language_contract_traceability import ContractTestBinding

_HEADING = re.compile(r"^### `(?P<id>LANG-[A-Z0-9-]+)`", re.MULTILINE)
_LAYERS = re.compile(r"^- Layers: (?P<layers>.+)$", re.MULTILINE)

_GROUPED_LAYERS = {
    "LANG-BUILTIN-POSITION": frozenset({"F", "I", "C", "A", "R"}),
    "LANG-BUILTIN-VERTEX-INDEX": frozenset({"F", "I", "C", "A", "R"}),
    "LANG-BUILTIN-INSTANCE-INDEX": frozenset({"F", "I", "C", "A", "R"}),
    "LANG-BUILTIN-FRAG-COORD": frozenset({"F", "I", "C", "A", "R"}),
    "LANG-BUILTIN-FRONT-FACING": frozenset({"F", "I", "C", "A", "R"}),
    "LANG-BUILTIN-GLOBAL-ID": frozenset({"F", "I", "C", "A", "R"}),
    "LANG-BUILTIN-LOCAL-ID": frozenset({"F", "I", "C", "A", "R"}),
    "LANG-BUILTIN-WORKGROUP-ID": frozenset({"F", "I", "C", "A", "R"}),
    "LANG-GENERATED-RESOLUTION": frozenset({"F", "I", "C", "A", "R"}),
    "LANG-GENERATED-FRAGMENT-COORD": frozenset({"F", "I", "C", "A", "R"}),
    "LANG-GENERATED-FRONT-FACING": frozenset({"F", "I", "C", "A", "R"}),
    "LANG-GENERATED-VERTEX-ID": frozenset({"F", "I", "C", "A", "R"}),
    "LANG-GENERATED-INSTANCE-ID": frozenset({"F", "I", "C", "A", "R"}),
}

_PAIR_COMPONENTS = {
    "LANG-PAIR-001": ("LANG-TENSOR-002", "LANG-STRUCT-001"),
    "LANG-PAIR-002": ("LANG-VIEW-007", "LANG-VIEW-004"),
    "LANG-PAIR-003": ("LANG-VIEW-007", "LANG-VIEW-005"),
    "LANG-PAIR-004": ("LANG-HELPER-001", "LANG-SPECIALIZE-001"),
    "LANG-PAIR-005": ("LANG-EFFECT-001", "LANG-ENTRY-001"),
    "LANG-PAIR-006": ("LANG-WORKGROUP-001", "LANG-BARRIER-001", "LANG-VIEW-007"),
    "LANG-PAIR-007": ("LANG-ATOMIC-001", "LANG-ATOMIC-002"),
    "LANG-PAIR-008": (
        "LANG-TEXTURE-SAMPLE-IMPLICIT",
        "LANG-TEXTURE-SAMPLE-EXPLICIT",
        "LANG-TEXTURE-SAMPLE-LOD",
        "LANG-TEXTURE-SAMPLE-EXPLICIT-LOD",
    ),
    "LANG-PAIR-009": ("LANG-CONTROL-001", "LANG-TENSOR-001", "LANG-STRUCT-001"),
    "LANG-PAIR-010": ("LANG-AD-003", "LANG-CONTROL-004"),
    "LANG-PAIR-011": ("LANG-AD-002", "LANG-VIEW-004"),
    "LANG-PAIR-012": ("LANG-PROGRAM-001",),
    "LANG-PAIR-013": ("LANG-ABI-001",),
    "LANG-PAIR-014": ("LANG-DISPATCH-001", "LANG-VIEW-004"),
}


def inventory_layer_requirements(markdown: str) -> dict[str, frozenset[str]]:
    headings = tuple(_HEADING.finditer(markdown))
    requirements: dict[str, frozenset[str]] = {}
    for index, heading in enumerate(headings):
        section_end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        layer_match = _LAYERS.search(markdown, heading.end(), section_end)
        if layer_match:
            contract_id = heading["id"]
            if contract_id in requirements:
                raise ValueError(f"duplicate inventory contract heading: {contract_id}")
            declaration = layer_match["layers"]
            if not re.fullmatch(r"`[FICAR]`(?:, `[FICAR]`)*\.?", declaration):
                raise ValueError(f"{contract_id} has malformed layer declaration: {declaration!r}")
            parsed_layers = re.findall(r"`([FICAR])`", declaration)
            if len(parsed_layers) != len(set(parsed_layers)):
                raise ValueError(f"{contract_id} has duplicate layer declarations")
            requirements[contract_id] = frozenset(parsed_layers)
    requirements.update(_GROUPED_LAYERS)
    for pair_id, component_ids in _PAIR_COMPONENTS.items():
        missing_components = set(component_ids) - requirements.keys()
        if missing_components:
            raise ValueError(f"{pair_id} references unknown contracts: {sorted(missing_components)}")
        requirements[pair_id] = frozenset().union(*(requirements[component] for component in component_ids))
    inventory_ids = frozenset(re.findall(r"\bLANG-[A-Z0-9-]+\b", markdown))
    missing_ids = inventory_ids - requirements.keys()
    if missing_ids:
        raise ValueError(f"inventory contracts have no layer requirements: {sorted(missing_ids)}")
    return requirements


def observed_layers(
    cases: tuple[LanguageContractCase, ...],
    bindings: tuple[ContractTestBinding, ...],
) -> dict[tuple[str, str], frozenset[str]]:
    cases_by_id = {case.id: case for case in cases}
    result: dict[tuple[str, str], frozenset[str]] = {}
    for binding in bindings:
        for case_id in binding.case_ids:
            if case_id not in cases_by_id:
                raise AssertionError(f"{binding.test_id} references unknown language contract case {case_id}")
            case = cases_by_id[case_id]
            if case.expected_diagnostic is not None:
                continue
            for region in case.valid_regions:
                key = (case.contract_id, region)
                result[key] = result.get(key, frozenset()) | binding.layers
    return result


def coverage_gaps(
    requirements: Mapping[str, frozenset[str]],
    cases: tuple[LanguageContractCase, ...],
    bindings: tuple[ContractTestBinding, ...],
    *,
    layers: frozenset[str] | None = None,
) -> dict[tuple[str, str], frozenset[str]]:
    actual = observed_layers(cases, bindings)
    all_case_contracts = {case.contract_id for case in cases}
    unknown_contracts = all_case_contracts - requirements.keys()
    if unknown_contracts:
        raise AssertionError(f"canonical cases reference unknown inventory contracts: {sorted(unknown_contracts)}")
    regions_by_contract: dict[str, frozenset[str]] = {}
    for case in cases:
        if case.expected_diagnostic is not None:
            continue
        regions_by_contract[case.contract_id] = (
            regions_by_contract.get(case.contract_id, frozenset()) | case.valid_regions
        )
    missing_contracts = requirements.keys() - regions_by_contract.keys()
    if missing_contracts:
        raise AssertionError(f"inventory contracts have no canonical cases: {sorted(missing_contracts)}")
    unsupported_contracts = {contract_id for contract_id in requirements if not regions_by_contract[contract_id]}
    if unsupported_contracts:
        raise AssertionError(
            f"inventory contracts have no positive supported-region cases: {sorted(unsupported_contracts)}"
        )
    gaps: dict[tuple[str, str], frozenset[str]] = {}
    for contract_id, required in requirements.items():
        for region in regions_by_contract.get(contract_id, frozenset()):
            region_layers = required - {"I", "C"} if region == "host" else required
            required_layers = region_layers if layers is None else region_layers & layers
            missing = required_layers - actual.get((contract_id, region), frozenset())
            if missing:
                gaps[(contract_id, region)] = missing
    return gaps


def audit_coverage(
    requirements: Mapping[str, frozenset[str]],
    cases: tuple[LanguageContractCase, ...],
    bindings: tuple[ContractTestBinding, ...],
    *,
    layers: frozenset[str] | None = None,
) -> None:
    negative_case_ids = {case.id for case in cases if case.expected_diagnostic is not None}
    bound_negative_case_ids = {
        case_id
        for binding in bindings
        if "F" in binding.layers
        for case_id in binding.case_ids
        if case_id in negative_case_ids
    }
    missing_negative_cases = negative_case_ids - bound_negative_case_ids
    if missing_negative_cases:
        raise AssertionError(f"language contract diagnostic cases were not executed: {sorted(missing_negative_cases)}")
    gaps = coverage_gaps(requirements, cases, bindings, layers=layers)
    if gaps:
        details = ", ".join(
            f"{contract_id}/{region}: {''.join(sorted(missing))}"
            for (contract_id, region), missing in sorted(gaps.items())
        )
        raise AssertionError(f"language contract coverage gaps: {details}")
