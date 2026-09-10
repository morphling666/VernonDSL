from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from vernon_dsl.shader_contracts import (
    ATOMIC_OPERATION_NAMES,
    BUILTIN_CONTRACTS,
    GENERATED_INTERFACE_CONTRACTS,
)

RUNTIME_NOT_APPLICABLE: Final = "runtime_not_applicable"
DEVICE_REGIONS: Final = frozenset({"compute", "vertex", "fragment", "func"})
ENTRY_REGIONS: Final = frozenset({"compute", "vertex", "fragment"})
KNOWN_REGIONS: Final = DEVICE_REGIONS | {"host"}
KNOWN_CAPABILITIES: Final = frozenset(
    {
        "compute",
        "graphics",
        "storage_buffers",
        "device_storage_atomics",
        "f32_atomic_add",
        "f64_atomic_add",
        "texture_sampler",
        "storage_texture",
        "workgroup_memory",
        "program_vjp",
    }
)


@dataclass(frozen=True)
class MlirOracle:
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    counts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class LanguageContractCase:
    contract_id: str
    name: str
    source_construct: str
    source: str
    valid_regions: frozenset[str]
    invalid_regions: frozenset[str]
    capabilities: frozenset[str]
    expected_diagnostic: str | None
    runtime_oracle: str
    expected: object | None = None

    @property
    def id(self) -> str:
        return f"{self.contract_id}/{self.name}"


def _valid_type(
    contract_id: str,
    name: str,
    source: str,
    expected_mlir: str,
    *,
    regions: frozenset[str] = DEVICE_REGIONS,
    capabilities: frozenset[str] = frozenset(),
) -> LanguageContractCase:
    return LanguageContractCase(
        contract_id,
        name,
        source,
        source,
        regions,
        frozenset(),
        capabilities,
        None,
        RUNTIME_NOT_APPLICABLE,
        expected_mlir,
    )


def _invalid_type(
    contract_id: str,
    name: str,
    source: str,
    diagnostic: str,
) -> LanguageContractCase:
    return LanguageContractCase(
        contract_id,
        name,
        source,
        source,
        frozenset(),
        DEVICE_REGIONS,
        frozenset(),
        diagnostic,
        RUNTIME_NOT_APPLICABLE,
    )


def _invalid_source(
    contract_id: str,
    name: str,
    source_construct: str,
    source: str,
    diagnostic: str,
    *,
    valid_regions: frozenset[str] = frozenset(),
    invalid_regions: frozenset[str] = DEVICE_REGIONS,
    capabilities: frozenset[str] = frozenset(),
    runtime_oracle: str = RUNTIME_NOT_APPLICABLE,
) -> LanguageContractCase:
    return LanguageContractCase(
        contract_id,
        name,
        source_construct,
        source,
        valid_regions,
        invalid_regions,
        capabilities,
        diagnostic,
        runtime_oracle,
    )


def _valid_source(
    contract_id: str,
    name: str,
    source_construct: str,
    source: str,
    *,
    valid_regions: frozenset[str],
    capabilities: frozenset[str],
    runtime_oracle: str,
    expected: object | None = None,
) -> LanguageContractCase:
    return LanguageContractCase(
        contract_id,
        name,
        source_construct,
        source,
        valid_regions,
        DEVICE_REGIONS - valid_regions,
        capabilities,
        None,
        runtime_oracle,
        expected,
    )


def _invalid_body(
    contract_id: str,
    name: str,
    source_construct: str,
    body: str,
    diagnostic: str,
) -> LanguageContractCase:
    return _invalid_source(
        contract_id,
        name,
        source_construct,
        "from vernon_dsl import *\n" + body,
        diagnostic,
    )


TYPE_PARSER_VALID_CASES: Final = (
    _valid_type("LANG-SCALAR-001", "f32", "f32", "f32"),
    _valid_type("LANG-SCALAR-001", "float-alias", "float", "f32"),
    _valid_type("LANG-STRUCT-001", "nominal-record", "Record", '!vernon.struct<"Record">'),
    _valid_type(
        "LANG-RESOURCE-001",
        "sampler",
        "Sampler",
        "!vernon.sampler",
        capabilities=frozenset({"texture_sampler"}),
    ),
    _valid_type("LANG-TENSOR-001", "static-tensor", "Tensor[f32, (2, 4)]", "tensor<2x4xf32>"),
    _valid_type("LANG-TENSOR-004", "vector-alias", "Vector[f16, 4]", "tensor<4xf16>"),
    _valid_type("LANG-TENSOR-004", "matrix-alias", "Matrix[f32, 3, 3]", "tensor<3x3xf32>"),
    _valid_type(
        "LANG-VIEW-001",
        "dynamic-read-view",
        "TensorView[f32, (dyn,), read]",
        '!vernon.tensor_view<f32, [-1], "read", "device">',
        regions=frozenset({"compute", "func"}),
        capabilities=frozenset({"storage_buffers"}),
    ),
    _valid_type(
        "LANG-VIEW-001",
        "mixed-shape-read-write-view",
        "TensorView[f32, (4, dyn), read_write]",
        '!vernon.tensor_view<f32, [4, -1], "read_write", "device">',
        regions=frozenset({"compute", "func"}),
        capabilities=frozenset({"storage_buffers"}),
    ),
    _valid_type(
        "LANG-RESOURCE-002",
        "sampled-cube-type",
        'Texture["cube", f32]',
        '!vernon.texture<"cube", f32, "unknown", "sampled">',
        capabilities=frozenset({"texture_sampler"}),
    ),
)

TYPE_PARSER_INVALID_CASES: Final = (
    _invalid_type("LANG-ENTRY-002", "none-outside-result", "None", "None is only valid"),
    _invalid_type("LANG-INTERFACE-001", "annotated-arity", "Annotated[f32]", "Annotated requires"),
    _invalid_type("LANG-SCALAR-001", "unknown-type", "Missing", "unknown DSL type"),
    _invalid_type("LANG-TENSOR-001", "missing-shape", "Tensor[f32]", "Tensor requires"),
    _invalid_type("LANG-TENSOR-001", "zero-shape", "Tensor[f32, 0]", "positive integer"),
    _invalid_type("LANG-TENSOR-001", "boolean-shape", "Tensor[f32, True]", "positive integer"),
    _invalid_type("LANG-TENSOR-001", "none-shape", "Tensor[f32, (None, 4)]", "positive integer"),
    _invalid_type(
        "LANG-LEGACY-001",
        "vec-constructor",
        "vec[3, f32]",
        "unknown DSL type constructor",
    ),
    _invalid_type(
        "LANG-LEGACY-001",
        "mat-constructor",
        "mat[2, 3, f64]",
        "unknown DSL type constructor",
    ),
    _invalid_type(
        "LANG-LEGACY-001",
        "vec2-constructor",
        "vec2[f32]",
        "unknown DSL type constructor",
    ),
    _invalid_type(
        "LANG-LEGACY-001",
        "mat4-constructor",
        "mat4[f32]",
        "unknown DSL type constructor",
    ),
    _invalid_type("LANG-TENSOR-004", "vector-arity", "Vector[f32, f64, 2]", "Vector requires"),
    _invalid_type(
        "LANG-TUPLE-001",
        "resource-element",
        "Tuple[Sampler]",
        "Tuple elements must be ABI-stable",
    ),
    _invalid_type(
        "LANG-TENSOR-001",
        "resource-element",
        "Tensor[Sampler, 2]",
        "Tensor element type must be an ABI-stable",
    ),
    _invalid_type(
        "LANG-STORAGE-001",
        "storage-arity",
        "TensorStorage[f32, f32]",
        "TensorStorage requires one",
    ),
    _invalid_type(
        "LANG-STORAGE-001",
        "resource-storage-element",
        "TensorStorage[Sampler]",
        "TensorStorage element type must be an ABI-stable",
    ),
    _invalid_type(
        "LANG-LEGACY-001",
        "buffer-constructor",
        "Buffer[f32]",
        "unknown DSL type constructor 'Buffer'",
    ),
    _invalid_type("LANG-VIEW-001", "missing-shape", "TensorView[f32, read]", "TensorView requires"),
    _invalid_type("LANG-VIEW-001", "scalar-shape", "TensorView[f32, 1, read]", "TensorView shape"),
    _invalid_type(
        "LANG-VIEW-001",
        "unknown-access",
        "TensorView[f32, (dyn,), missing]",
        "TensorView access",
    ),
    _invalid_type("LANG-RESOURCE-002", "texture-arity", "Texture[f32]", "Texture requires"),
    _invalid_type(
        "LANG-RESOURCE-002",
        "unsupported-dimension",
        'Texture["1d", f32]',
        "texture dimension",
    ),
    _invalid_type(
        "LANG-RESOURCE-002",
        "dynamic-dimension",
        "Texture[value(), f32]",
        "string literal or name",
    ),
    _invalid_type(
        "LANG-SCALAR-001",
        "unknown-constructor",
        "Unknown[f32]",
        "unknown DSL type constructor",
    ),
)

METADATA_VALID_CASES: Final = tuple(
    LanguageContractCase(
        "LANG-INTERFACE-001",
        name,
        source,
        source,
        ENTRY_REGIONS,
        frozenset(),
        frozenset({"graphics"}),
        None,
        RUNTIME_NOT_APPLICABLE,
        expected,
    )
    for name, source, expected in (
        ("inferred-attribute", "Annotated[f32, attribute()]", ("attribute", (-1, 0))),
        ("located-attribute", "Annotated[f32, attribute(3)]", ("attribute", (3, 0))),
        (
            "divisor-attribute",
            "Annotated[f32, attribute(divisor=2)]",
            ("attribute", (-1, 2)),
        ),
        (
            "located-divisor-attribute",
            "Annotated[f32, attribute(location=3, divisor=2)]",
            ("attribute", (3, 2)),
        ),
        (
            "position-builtin",
            'Annotated[f32, builtin("position")]',
            ("builtin", ("position",)),
        ),
        ("inferred-uniform", "Annotated[f32, uniform()]", ("uniform", ())),
        (
            "bound-uniform",
            "Annotated[f32, uniform(set=1, binding=2)]",
            ("uniform", (1, 2)),
        ),
        ("varying", "Annotated[f32, varying()]", ("varying", ())),
        (
            "bound-resource",
            "Annotated[f32, resource(set=1, binding=2)]",
            ("resource", (1, 2)),
        ),
    )
)

METADATA_INVALID_CASES: Final = tuple(
    _invalid_type("LANG-INTERFACE-001", name, source, diagnostic)
    for name, source, diagnostic in (
        ("non-call", "Annotated[f32, marker]", "metadata must be a call"),
        ("unknown", "Annotated[f32, unknown()]", "unknown annotation metadata"),
        (
            "removed-location",
            "Annotated[f32, location(0)]",
            "unknown annotation metadata",
        ),
        (
            "removed-instance",
            "Annotated[f32, instance(location=0)]",
            "unknown annotation metadata",
        ),
        ("varying-arity", "Annotated[f32, varying(1)]", "wrong number"),
        (
            "negative-divisor",
            "Annotated[f32, attribute(divisor=-1)]",
            "integer or string",
        ),
        (
            "expanded-resource-options",
            "Annotated[f32, resource(**opts)]",
            r"\*\*kwargs",
        ),
        ("partial-resource-binding", "Annotated[f32, resource(set=1)]", "wrong number"),
        (
            "builtin-keyword",
            "Annotated[f32, builtin(value=0)]",
            "does not accept keyword",
        ),
        (
            "string-attribute-location",
            "Annotated[f32, attribute('position')]",
            "location must be non-negative",
        ),
        (
            "string-attribute-divisor",
            "Annotated[f32, attribute(0, 'instance')]",
            "divisor must be non-negative",
        ),
    )
)

BUILTIN_NAMES_BY_CONTRACT: Final = (
    ("LANG-BUILTIN-POSITION", "position"),
    ("LANG-BUILTIN-VERTEX-INDEX", "vertex_index"),
    ("LANG-BUILTIN-INSTANCE-INDEX", "instance_index"),
    ("LANG-BUILTIN-FRAG-COORD", "frag_coord"),
    ("LANG-BUILTIN-FRONT-FACING", "front_facing"),
    ("LANG-BUILTIN-GLOBAL-ID", "global_invocation_id"),
    ("LANG-BUILTIN-LOCAL-ID", "local_invocation_id"),
    ("LANG-BUILTIN-WORKGROUP-ID", "workgroup_id"),
)

BUILTIN_INVALID_CASES: Final = (
    LanguageContractCase(
        "LANG-BUILTIN-POSITION",
        "unknown-name",
        'builtin("mystery")',
        """
from vernon_dsl import *
@vertex
def main(value: Annotated[u32, builtin("mystery")]) -> Vector[f32, 4]:
    return Vector([0.0, 0.0, 0.0, 1.0])
""",
        frozenset(),
        DEVICE_REGIONS,
        frozenset({"graphics"}),
        "unknown builtin 'mystery'",
        RUNTIME_NOT_APPLICABLE,
    ),
    LanguageContractCase(
        "LANG-BUILTIN-VERTEX-INDEX",
        "fragment-input",
        'builtin("vertex_index")',
        """
from vernon_dsl import *
@fragment
def main(value: Annotated[u32, builtin("vertex_index")]) -> Vector[f32, 4]:
    return Vector([0.0, 0.0, 0.0, 1.0])
""",
        frozenset({"vertex"}),
        frozenset({"compute", "fragment", "func"}),
        frozenset({"graphics"}),
        "requires a vertex input",
        RUNTIME_NOT_APPLICABLE,
    ),
    LanguageContractCase(
        "LANG-BUILTIN-VERTEX-INDEX",
        "vertex-output",
        'builtin("vertex_index")',
        """
from vernon_dsl import *
@vertex
def main() -> Annotated[u32, builtin("vertex_index")]:
    return 0
""",
        frozenset({"vertex"}),
        frozenset({"compute", "fragment", "func"}),
        frozenset({"graphics"}),
        "requires a vertex input",
        RUNTIME_NOT_APPLICABLE,
    ),
    LanguageContractCase(
        "LANG-BUILTIN-GLOBAL-ID",
        "wrong-type",
        'builtin("global_invocation_id")',
        """
from vernon_dsl import *
@kernel(workgroup_size=(1, 1, 1))
def main(gid: Annotated[u32, builtin("global_invocation_id")]) -> None:
    pass
""",
        frozenset({"compute"}),
        frozenset({"vertex", "fragment", "func"}),
        frozenset({"compute"}),
        "requires type tensor<3xi32>",
        RUNTIME_NOT_APPLICABLE,
    ),
    LanguageContractCase(
        "LANG-BUILTIN-POSITION",
        "missing-output",
        "vertex position output",
        """
from vernon_dsl import *
@vertex
def main() -> Vector[f32, 3]:
    return Vector([0.0, 0.0, 0.0])
""",
        frozenset({"vertex"}),
        frozenset({"compute", "fragment", "func"}),
        frozenset({"graphics"}),
        "builtin 'position' requires type tensor<4xf32>",
        RUNTIME_NOT_APPLICABLE,
    ),
)

TEXTURE_OPERATION_INVALID_CASES: Final = (
    _invalid_source(
        "LANG-RESOURCE-002",
        "sampled-dimension",
        'Texture["1d", f32]',
        """
from vernon_dsl import *
@fragment
def sample(image: Texture["1d", f32]) -> f32:
    return 0.0
""",
        "texture dimension must be one of",
        capabilities=frozenset({"graphics", "texture_sampler"}),
    ),
    _invalid_source(
        "LANG-TEXTURE-SAMPLE-EXPLICIT",
        "cube-coordinate-rank",
        "texture_sample cube coordinates",
        """
from vernon_dsl import *
@fragment
def sample(image: Texture["cube", f32], sampler: Sampler,
           uv: Vector[f32, 2]) -> Vector[f32, 4]:
    return texture_sample(image, sampler, uv)
""",
        "3-component",
        capabilities=frozenset({"graphics", "texture_sampler"}),
    ),
    _invalid_source(
        "LANG-RESOURCE-002",
        "cube-storage",
        'Texture["cube", rgba32_float, write]',
        """
from vernon_dsl import *
@kernel
def bad(image: Texture["cube", rgba32_float, write]) -> None:
    pass
""",
        "storage Texture dimension",
        capabilities=frozenset({"compute", "storage_texture"}),
    ),
    _invalid_source(
        "LANG-TEXTURE-SAMPLE-IMPLICIT",
        "vertex-without-lod",
        "texture_sample(texture, coordinates)",
        """
from vernon_dsl import *
@vertex
def main(image: Texture["2d", f32], uv: Vector[f32, 2]) -> Vector[f32, 4]:
    return texture_sample(image, uv)
""",
        "without lod",
        valid_regions=frozenset({"fragment"}),
        invalid_regions=frozenset({"compute", "vertex", "func"}),
        capabilities=frozenset({"graphics", "texture_sampler"}),
    ),
    _invalid_source(
        "LANG-TEXTURE-SAMPLE-LOD",
        "integer-lod",
        "texture_sample(texture, coordinates, integer_lod)",
        """
from vernon_dsl import *
@fragment
def main(image: Texture["2d", f32], uv: Vector[f32, 2]) -> Vector[f32, 4]:
    return texture_sample(image, uv, 1)
""",
        "floating-point scalar",
        capabilities=frozenset({"graphics", "texture_sampler"}),
    ),
    _invalid_source(
        "LANG-TEXTURE-SIZE",
        "floating-lod",
        "texture_size(texture, floating_lod)",
        """
from vernon_dsl import *
@fragment
def main(image: Texture["2d", f32]) -> Vector[u32, 2]:
    return texture_size(image, 1.0)
""",
        "integer scalar",
        capabilities=frozenset({"graphics", "texture_sampler"}),
    ),
    _invalid_source(
        "LANG-TEXTURE-SAMPLE-EXPLICIT",
        "compute-without-lod",
        "texture_sample(texture, sampler, coordinates)",
        """
from vernon_dsl import *
@kernel(workgroup_size=(1, 1, 1))
def main(image: Texture["2d", f32], sampler: Sampler,
         uv: Vector[f32, 2]) -> None:
    color = texture_sample(image, sampler, uv)
""",
        "without lod.*fragment shaders",
        valid_regions=frozenset({"fragment"}),
        invalid_regions=frozenset({"compute", "vertex", "func"}),
        capabilities=frozenset({"compute", "texture_sampler"}),
    ),
    _invalid_source(
        "LANG-TEXTURE-SAMPLE-EXPLICIT",
        "mixed-sampler-mode",
        "mixed implicit and explicit texture_sample",
        """
from vernon_dsl import *
@fragment
def main(
    image: Annotated[Texture["2d", f32], resource(set=0, binding=0)],
    sampler: Annotated[Sampler, resource(set=0, binding=1)],
    uv: Vector[f32, 2],
) -> Vector[f32, 4]:
    implicit_value = texture_sample(image, uv)
    explicit_value = texture_sample(image, sampler, uv)
    return implicit_value + explicit_value
""",
        "both implicit and explicit sampler forms",
        capabilities=frozenset({"graphics", "texture_sampler"}),
    ),
)

TEXTURE_OPERATION_VALID_CASES: Final = tuple(
    _valid_source(
        "LANG-RESOURCE-002",
        f"sampled-{dimension}",
        f'Texture["{dimension}", f32]',
        f"""
from vernon_dsl import *

@fragment
def sample(
    image: Annotated[Texture["{dimension}", f32], resource(set=0, binding=0)],
    sampler: Annotated[Sampler, resource(set=0, binding=1)],
    uv: Vector[f32, {rank}],
) -> Vector[f32, 4]:
    return texture_sample(image, sampler, uv)
""",
        valid_regions=frozenset({"fragment"}),
        capabilities=frozenset({"graphics", "texture_sampler"}),
        runtime_oracle="filtered_texel",
        expected=f'!vernon.texture<"{dimension}", f32, "unknown", "sampled">',
    )
    for dimension, rank in (("2d", 2), ("3d", 3), ("cube", 3))
) + (
    _valid_source(
        "LANG-TEXTURE-STORAGE",
        "three-dimensional-store",
        "texture_store(write_texture, coordinate, value)",
        """
from vernon_dsl import *

@kernel
def store(
    image: Annotated[Texture["3d", rgba32_float, write], resource(set=0, binding=0)],
    coordinate: Vector[i32, 3],
    value: Vector[f32, 4],
) -> None:
    texture_store(image, coordinate, value)
""",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_texture"}),
        runtime_oracle="stored_texel",
        expected='!vernon.texture<"3d", f32, "rgba32_float", "write">',
    ),
    _valid_source(
        "LANG-TEXTURE-SAMPLE-EXPLICIT-LOD",
        "vertex",
        "texture_sample(texture, sampler, coordinates, lod)",
        """
from vernon_dsl import *
@vertex
def main(image: Texture["2d", f32], sampler: Sampler,
         uv: Vector[f32, 2]) -> Vector[f32, 4]:
    height = texture_sample(image, sampler, uv, 0.0)
    return Vector([uv, height.x, 1.0])
""",
        valid_regions=frozenset({"vertex"}),
        capabilities=frozenset({"graphics", "texture_sampler"}),
        runtime_oracle="selected_filtered_mip",
    ),
    _valid_source(
        "LANG-TEXTURE-SAMPLE-EXPLICIT-LOD",
        "compute",
        "texture_sample(texture, sampler, coordinates, lod)",
        """
from vernon_dsl import *
@kernel(workgroup_size=(1, 1, 1))
def main(image: Texture["2d", f32], sampler: Sampler) -> None:
    color = texture_sample(image, sampler, Vector([0.0, 0.0]), 0.0)
""",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "texture_sampler"}),
        runtime_oracle="selected_filtered_mip",
    ),
    _valid_source(
        "LANG-TEXTURE-SAMPLE-EXPLICIT-LOD",
        "fragment",
        "texture_sample(texture, sampler, coordinates, lod)",
        """
from vernon_dsl import *
@fragment
def main(image: Texture["2d", f32], sampler: Sampler,
         uv: Vector[f32, 2]) -> Vector[f32, 4]:
    return texture_sample(image, sampler, uv, 0.0)
""",
        valid_regions=frozenset({"fragment"}),
        capabilities=frozenset({"graphics", "texture_sampler"}),
        runtime_oracle="selected_filtered_mip",
    ),
)

ATOMIC_INVALID_CASES: Final = tuple(
    _invalid_source(
        "LANG-ATOMIC-001",
        f"readonly-{operation}",
        f"{operation}(read_only_view, index, value)",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def bad(values: TensorView[i32, (dyn,), read]) -> None:\n"
        f"    {operation}(values, 0, 1)\n",
        "requires a writable TensorView",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset({"vertex", "fragment", "func"}),
        capabilities=frozenset({"device_storage_atomics"}),
    )
    for operation in sorted(ATOMIC_OPERATION_NAMES)
) + (
    _invalid_source(
        "LANG-ATOMIC-001",
        "unnamed-owner",
        "atomic_add(conditional_view, index, value)",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def bad(values: TensorView[i32, (dyn,), read_write]) -> None:\n"
        "    atomic_add(values if True else values, 0, 1)\n",
        "requires a named storage owner",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset({"vertex", "fragment", "func"}),
        capabilities=frozenset({"device_storage_atomics"}),
    ),
)
READONLY_ATOMIC_INVALID_CASES: Final = tuple(case for case in ATOMIC_INVALID_CASES if case.name.startswith("readonly-"))

WORKGROUP_INVALID_CASES: Final = (
    _invalid_source(
        "LANG-WORKGROUP-001",
        "physical-limit",
        "aggregate workgroup_storage over 16 KiB",
        "from vernon_dsl import *\n"
        "@struct\n"
        "class Tiny:\n"
        "    first: bool\n"
        "    second: bool\n"
        "    third: bool\n"
        "@kernel\n"
        "def main() -> None:\n"
        "    values = workgroup_storage(Tiny, shape=(5458,))\n",
        "portable 16 KiB workgroup storage limit",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset({"vertex", "fragment", "func"}),
        capabilities=frozenset({"compute", "workgroup_memory"}),
    ),
    _invalid_source(
        "LANG-WORKGROUP-001",
        "combined-physical-limit",
        "combined workgroup_storage over 16 KiB",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def main(flag: bool) -> None:\n"
        "    first = workgroup_storage(i32, shape=(2049,))\n"
        "    if flag:\n"
        "        second = workgroup_storage(i32, shape=(2049,))\n",
        "portable 16 KiB workgroup storage limit",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset({"vertex", "fragment", "func"}),
        capabilities=frozenset({"compute", "workgroup_memory"}),
    ),
    _invalid_source(
        "LANG-BARRIER-001",
        "fragment-region",
        "workgroup_barrier() in fragment",
        "from vernon_dsl import *\n@fragment\ndef bad() -> None:\n    workgroup_barrier()\n",
        "only in compute kernels",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset({"vertex", "fragment", "func"}),
        capabilities=frozenset({"workgroup_memory"}),
    ),
    _invalid_source(
        "LANG-WORKGROUP-001",
        "zero-extent",
        "workgroup_storage zero extent",
        "from vernon_dsl import *\n@kernel\ndef bad() -> None:\n    values = workgroup_storage(f32, shape=(0,))\n",
        "positive compile-time integers",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset({"vertex", "fragment", "func"}),
        capabilities=frozenset({"compute", "workgroup_memory"}),
    ),
)

SYNCHRONIZATION_VALID_CASES: Final = (
    _valid_source(
        "LANG-PAIR-007",
        "workgroup-atomic-contention",
        "device/workgroup floating atomic_add contention",
        "from vernon_dsl import *\n"
        "@kernel(workgroup_size=(4, 1, 1))\n"
        "def main(\n"
        "    device: TensorView[f32, (dyn,), read_write],\n"
        "    local_id: Annotated[Vector[u32, 3], builtin('local_invocation_id')],\n"
        ") -> None:\n"
        "    shared = workgroup_storage(f32, shape=(1,))\n"
        "    if local_id[0] == 0:\n"
        "        shared[0] = 0.0\n"
        "    workgroup_barrier()\n"
        "    atomic_add(shared, 0, 1.0)\n"
        "    workgroup_barrier()\n"
        "    if local_id[0] == 0:\n"
        "        atomic_add(device, 1, shared[0])\n"
        "    atomic_add(device, 0, 1.0)\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "workgroup_memory", "f32_atomic_add", "device_storage_atomics"}),
        runtime_oracle="both_scopes_contended_sum",
        expected=(
            'atomic_kind = "add"',
            'scope = "workgroup"',
            'scope = "device"',
            "vernon.workgroup_size = array<i32: 4, 1, 1>",
        ),
    ),
    _valid_source(
        "LANG-ATOMIC-001",
        "workgroup-operations",
        "all integer atomic operations on workgroup storage",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def main(output: TensorView[i32, (dyn,), write]) -> None:\n"
        "    signed = workgroup_storage(i32, shape=(4,))\n"
        "    unsigned = workgroup_storage(u32, shape=(2,))\n"
        "    signed[0] = 4\n"
        "    unsigned[0] = u32(4)\n"
        "    workgroup_barrier()\n"
        "    added = atomic_add(signed, 0, 1)\n"
        "    minimum = atomic_min(signed, 1, 2)\n"
        "    maximum = atomic_max(signed, 2, 3)\n"
        "    exchanged = atomic_exchange(signed, 3, 4)\n"
        "    unsigned_minimum = atomic_min(unsigned, 0, 2)\n"
        "    unsigned_maximum = atomic_max(unsigned, 1, 3)\n"
        "    output[0] = added + minimum + maximum + exchanged\n"
        "    storage_barrier()\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "workgroup_memory"}),
        runtime_oracle="legal_workgroup_atomic_serialization",
        expected=(
            'atomic_kind = "add"',
            'atomic_kind = "min"',
            'atomic_kind = "max"',
            'atomic_kind = "exchange"',
            'atomic_kind = "umin"',
            'atomic_kind = "umax"',
        ),
    ),
    _valid_source(
        "LANG-WORKGROUP-001",
        "rank-zero",
        "rank-zero workgroup_storage",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def scalar_shared(output: TensorView[f32, (), write]) -> None:\n"
        "    value = workgroup_storage(f32, shape=())\n"
        "    value[()] = 3.0\n"
        "    workgroup_barrier()\n"
        "    output[()] = value[()]\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "workgroup_memory"}),
        runtime_oracle="independent_workgroup_value",
    ),
    _valid_source(
        "LANG-ATOMIC-001",
        "device-storage-operations",
        "integer TensorView atomic operations",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def main(\n"
        "    signed: TensorView[i32, (dyn,), read_write],\n"
        "    unsigned: TensorView[u32, (dyn,), read_write],\n"
        ") -> None:\n"
        "    added = atomic_add(signed, 0, 1)\n"
        "    minimum = atomic_min(signed, 1, 2)\n"
        "    maximum = atomic_max(signed, 2, 3)\n"
        "    exchanged = atomic_exchange(signed, 3, 4)\n"
        "    unsigned_minimum = atomic_min(unsigned, 0, 2)\n"
        "    unsigned_maximum = atomic_max(unsigned, 1, 3)\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "device_storage_atomics"}),
        runtime_oracle="legal_atomic_serialization",
    ),
)

TENSOR_VIEW_VALID_CASES: Final = (
    _valid_source(
        "LANG-VIEW-001",
        "shape-and-access-matrix",
        "TensorView rank and access declarations",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def shapes(\n"
        "    scalar: TensorView[f32, (), read_write],\n"
        "    static: TensorView[f32, (4, 8), read],\n"
        "    dynamic: TensorView[f32, (dyn,), read],\n"
        "    mixed: TensorView[f32, (dyn, 4), read],\n"
        ") -> None:\n"
        "    scalar[()] = scalar[()] + 1.0\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="bound_rank_shape_and_access",
    ),
    _valid_source(
        "LANG-VIEW-006",
        "rank-two-load",
        "ranked TensorView load",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def read(value: TensorView[f32, (dyn, dyn), read]) -> f32:\n"
        "    return value[0, 0]\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="projected_load_value",
    ),
    _valid_source(
        "LANG-TENSOR-005",
        "tensor-and-view-shape",
        "Tensor and TensorView .shape",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def extents(\n"
        "    output: TensorView[f32, (dyn, dyn), write],\n"
        "    tile: Tensor[f32, (2, 5)],\n"
        "    vec: Vector[f32, 3],\n"
        "    mat: Matrix[f32, 2, 4],\n"
        ") -> None:\n"
        "    output[output.shape[0], tile.shape[1]] = f32(vec.shape[0]) + f32(mat.shape[1])\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="dynamic_and_static_extents",
    ),
    _valid_source(
        "LANG-VIEW-004",
        "layout-not-specialization",
        "dynamic TensorView invocation layout",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def read(output: TensorView[f32, (1,), write], value: TensorView[f32, (2, dyn), read]) -> None:\n"
        "    output[0] = value[1, 2]\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="one_artifact_multiple_layouts",
    ),
)

TENSOR_VIEW_INVALID_CASES: Final = (
    _invalid_source(
        "LANG-VIEW-001",
        "non-tuple-shape",
        "TensorView scalar shape",
        "from vernon_dsl import *\n@kernel\ndef removed(value: TensorView[f32, 1, read]) -> None:\n    pass\n",
        "shape must be a tuple",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset({"vertex", "fragment", "func"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
    ),
    _invalid_source(
        "LANG-STORAGE-001",
        "device-annotation",
        "TensorStorage device annotation",
        "from vernon_dsl import *\n@kernel\ndef bad(value: TensorStorage[f32]) -> None:\n    pass\n",
        "host-runtime owner",
        valid_regions=frozenset({"host"}),
        invalid_regions=DEVICE_REGIONS,
    ),
    _invalid_source(
        "LANG-VIEW-006",
        "write-only-load",
        "load from write-only TensorView",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def bad(value: TensorView[f32, (dyn,), write]) -> f32:\n"
        "    return value[0]\n",
        "write-only TensorView",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset({"vertex", "fragment", "func"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
    ),
    _invalid_source(
        "LANG-TENSOR-005",
        "rank-zero-shape",
        "rank-zero TensorView .shape",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def bad(value: TensorView[f32, (), read_write]) -> None:\n"
        "    extent = value.shape\n",
        "shape requires rank >= 1",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset({"vertex", "fragment", "func"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
    ),
)

DISPATCH_CONTRACT_CASES: Final = (
    _valid_source(
        "LANG-DISPATCH-001",
        "dynamic-grid",
        "runtime grid=(x, y, z)",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def fill(output: TensorView[f32, (dyn,), write], value: f32) -> None:\n"
        "    output[0] = value\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute"}),
        runtime_oracle="exact_publication_for_legal_grid",
        expected=(3, 2, 1),
    ),
    _invalid_source(
        "LANG-DISPATCH-001",
        "zero-grid-axis",
        "runtime grid=(3, 0, 1)",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def fill(output: TensorView[f32, (dyn,), write], value: f32) -> None:\n"
        "    output[0] = value\n",
        "grid must contain three positive integers",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset(),
        capabilities=frozenset({"compute"}),
        runtime_oracle="pre_submission_dispatch_rejection",
    ),
)

_PROGRAM_VJP_ASSET_SOURCE: Final = """
import vernon_dsl as vd

@vd.kernel
def compute(
    value: vd.TensorView[vd.f32, (1,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss[0] = value[0] * value[0]

asset = vd.program_asset(
    id="compute/vjp",
    program=vd.ad.vjp(compute, wrt=("value",), outputs=("loss",)),
)
"""

PROGRAM_AUTODIFF_CASES: Final = (
    _valid_source(
        "LANG-PROGRAM-001",
        "module-composition",
        "initialized Module composition",
        """
import vernon_dsl as vd

@vd.kernel
def copy(source: vd.TensorView[vd.f32, (1,), vd.read],
         output: vd.TensorView[vd.f32, (1,), vd.write]) -> None:
    output[0] = source[0]

class Copy(vd.Module):
    def forward(self, source: vd.TensorStorage) -> vd.TensorStorage:
        output = vd.zeros_like(source)
        copy(source, output, grid=(1, 1, 1))
        return output

asset = vd.program_asset(id="module/copy", program=Copy())
""",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute"}),
        runtime_oracle="module_matches_direct_semantics",
    ),
    _valid_source(
        "LANG-AD-001",
        "canonical-capture",
        "vd.ad.vjp(program, wrt=..., outputs=...)",
        _PROGRAM_VJP_ASSET_SOURCE,
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "program_vjp"}),
        runtime_oracle="primal_and_finite_difference_gradient",
    ),
    _invalid_source(
        "LANG-AD-004",
        "graphics-tuple",
        "vd.ad.vjp((vertex, fragment), ...)",
        """
import vernon_dsl as vd

@vd.vertex
def vertex(value: vd.f32) -> vd.f32:
    return value

@vd.fragment
def fragment(value: vd.f32) -> vd.f32:
    return value

vd.ad.vjp((vertex, fragment), wrt=("value",))
""",
        r"must use vd\.pipeline",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset({"vertex", "fragment"}),
        capabilities=frozenset({"graphics"}),
    ),
    _invalid_source(
        "LANG-AD-001",
        "missing-storage-output",
        "VJP without writable Storage objective",
        """
import vernon_dsl as vd

@vd.kernel
def compute(value: vd.f32) -> None:
    pass

vd.ad.vjp(compute, wrt=("value",))
""",
        "requires non-empty writable Storage outputs",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset(),
        capabilities=frozenset({"compute", "program_vjp"}),
    ),
)

TENSOR_ELEMENT_INVALID_CASES: Final = (
    _invalid_source(
        "LANG-TENSOR-001",
        "storage-element",
        "Tensor[TensorView[...], ...]",
        "from vernon_dsl import *\n"
        "@func\n"
        "def bad(value: Tensor[TensorView[f32, (dyn,), read], (2,)]) -> None:\n"
        "    pass\n",
        "Storage 'TensorView'",
    ),
    _invalid_source(
        "LANG-TENSOR-001",
        "texture-element",
        "Tensor[Texture[...], ...]",
        'from vernon_dsl import *\n@func\ndef bad(value: Tensor[Texture["2d", f32], (2,)]) -> None:\n    pass\n',
        "Resource 'Texture'",
    ),
    _invalid_source(
        "LANG-TENSOR-001",
        "sampler-element",
        "Tensor[Sampler, ...]",
        "from vernon_dsl import *\n@func\ndef bad(value: Tensor[Sampler, (2,)]) -> None:\n    pass\n",
        "Resource 'Sampler'",
    ),
)

STRUCT_FIELD_INVALID_CASES: Final = (
    _invalid_source(
        "LANG-STRUCT-001",
        "storage-field",
        "Struct TensorView field",
        "from vernon_dsl import *\n@struct\nclass Bad:\n    data: TensorView[f32, (dyn,), read]\n",
        "Struct field 'Bad.data' must be an ABI-stable Value",
    ),
    _invalid_source(
        "LANG-STRUCT-001",
        "recursive-field",
        "recursive Struct field",
        "from vernon_dsl import *\n@struct\nclass Recursive:\n    next: Recursive\n",
        "Struct field 'Recursive.next' must be an ABI-stable Value",
    ),
)

SCALAR_CONVERSION_INVALID_CASES: Final = (
    _invalid_source(
        "LANG-SCALAR-002",
        "f64-to-f32",
        "implicit f64 to f32",
        "from vernon_dsl import *\n@func\ndef bad(value: f64) -> f32:\n    return value\n",
        "unsafe implicit conversion from f64 to f32",
    ),
    _invalid_source(
        "LANG-SCALAR-002",
        "f32-to-i32",
        "implicit f32 to i32",
        "from vernon_dsl import *\n@func\ndef bad(value: f32) -> i32:\n    return value\n",
        "unsafe implicit conversion from f32 to i32",
    ),
    _invalid_source(
        "LANG-SCALAR-002",
        "mixed-signedness",
        "dynamic i32/u32 arithmetic",
        "from vernon_dsl import *\n@func\ndef bad(left: i32, right: u32) -> i32:\n    return left + right\n",
        "no safe common type",
    ),
    _invalid_source(
        "LANG-SCALAR-003",
        "bool-arithmetic",
        "bool addition",
        "from vernon_dsl import *\n@func\ndef bad(left: bool, right: bool) -> bool:\n    return left + right\n",
        "unsupported binary operation",
    ),
)

HELPER_SPECIALIZATION_INVALID_CASES: Final = (
    _invalid_source(
        "LANG-HELPER-001",
        "annotated-argument",
        "partially annotated helper argument",
        "from vernon_dsl import *\n"
        "@func\n"
        "def helper(value: i32):\n"
        "    return value\n"
        "@fragment\n"
        "def main(value: f32) -> i32:\n"
        "    return helper(value)\n",
        "cannot pass f32 as i32",
        valid_regions=frozenset({"func", "fragment"}),
        invalid_regions=frozenset(),
        capabilities=frozenset({"graphics"}),
    ),
    _invalid_source(
        "LANG-HELPER-001",
        "annotated-result",
        "partially annotated helper result",
        "from vernon_dsl import *\n"
        "@func\n"
        "def helper(value) -> i32:\n"
        "    return value\n"
        "@fragment\n"
        "def main(value: f32) -> i32:\n"
        "    return helper(value)\n",
        "cannot return f32 as i32",
        valid_regions=frozenset({"func", "fragment"}),
        invalid_regions=frozenset(),
        capabilities=frozenset({"graphics"}),
    ),
)

TUPLE_INDEX_INVALID_CASES: Final = (
    _invalid_source(
        "LANG-TUPLE-001",
        "dynamic-index",
        "Tuple dynamic index",
        "from vernon_dsl import *\n@func\ndef bad(pair: Tuple[f32, i32], index: i32) -> f32:\n    return pair[index]\n",
        "Tuple indexing requires an integer literal",
    ),
    _invalid_source(
        "LANG-TUPLE-001",
        "out-of-bounds-index",
        "Tuple out-of-bounds index",
        "from vernon_dsl import *\n@func\ndef bad(pair: Tuple[f32, i32], index: i32) -> f32:\n    return pair[2]\n",
        "Tuple index is out of bounds",
    ),
)

ENTRY_SIGNATURE_INVALID_CASES: Final = (
    _invalid_source(
        "LANG-ENTRY-002",
        "unannotated-parameter",
        "entry parameter without annotation",
        "from vernon_dsl import *\n@kernel\ndef main(value) -> None:\n    pass\n",
        "entry argument 'value' requires a type annotation",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset({"vertex", "fragment"}),
        capabilities=frozenset({"compute"}),
    ),
    _invalid_source(
        "LANG-ENTRY-002",
        "unannotated-result",
        "entry result without annotation",
        "from vernon_dsl import *\n@kernel\ndef main(value: f32):\n    pass\n",
        "entry function 'main' requires a result annotation",
        valid_regions=frozenset({"compute"}),
        invalid_regions=frozenset({"vertex", "fragment"}),
        capabilities=frozenset({"compute"}),
    ),
)

INFERENCE_STATEMENT_INVALID_CASES: Final = (
    _invalid_body(
        "LANG-ENTRY-002",
        "void-helper-return-value",
        "value return from void helper",
        "@func\ndef bad(value: f32) -> None:\n    return value\n"
        "@fragment\ndef main(value: f32) -> f32:\n    bad(value)\n    return value\n",
        "void function.*returns a value",
    ),
    _invalid_body(
        "LANG-ENTRY-002",
        "missing-return",
        "value helper without return",
        "@func\ndef bad(value: f32) -> f32:\n    value + 1\n"
        "@fragment\ndef main(value: f32) -> f32:\n    return bad(value)\n",
        "requires a return value",
    ),
    _invalid_body(
        "LANG-ENTRY-002",
        "empty-value-return",
        "empty return from value helper",
        "@func\ndef bad(value: f32) -> f32:\n    return\n"
        "@fragment\ndef main(value: f32) -> f32:\n    return bad(value)\n",
        "requires a return value",
    ),
    _invalid_body(
        "LANG-CONTROL-001",
        "scalar-destructuring",
        "destructuring a scalar",
        "@fragment\ndef main(value: f32) -> f32:\n    left, right = value\n    return value\n",
        "assignment target",
    ),
    _invalid_body(
        "LANG-SEM-001",
        "value-store",
        "indexed assignment through a Value",
        "@fragment\ndef main(value: f32) -> f32:\n    value[0] = 1\n    return value\n",
        "writable Storage",
    ),
    _invalid_body(
        "LANG-SCALAR-002",
        "annotated-assignment",
        "unsafe annotated assignment",
        "@fragment\ndef main(value: f32) -> f32:\n    result: i32 = value\n    return value\n",
        "cannot infer assignment",
    ),
    _invalid_body(
        "LANG-SCALAR-003",
        "augmented-assignment",
        "incompatible augmented assignment",
        "@fragment\ndef main(value: f32) -> f32:\n    value += True\n    return value\n",
        "augmented assignment",
    ),
    _invalid_body(
        "LANG-CONTROL-001",
        "python-list",
        "arbitrary Python list expression",
        "@fragment\ndef main(value: f32) -> f32:\n    result = [value]\n    return value\n",
        "cannot infer expression syntax",
    ),
    _invalid_body(
        "LANG-TENSOR-005",
        "floating-index",
        "floating Tensor index",
        "@fragment\ndef main(value: Vector[f32, 2], index: f32) -> f32:\n    return value[index]\n",
        "index must be an integer",
    ),
)

INFERENCE_CALL_INVALID_CASES: Final = (
    _invalid_body(
        "LANG-ENTRY-002",
        "helper-arity",
        "helper call arity",
        "@func\ndef helper(value: f32) -> f32:\n    return value\n"
        "@fragment\ndef main(value: f32) -> f32:\n    return helper()\n",
        "expects 1 arguments",
    ),
    _invalid_body(
        "LANG-HELPER-001",
        "helper-argument",
        "incompatible helper argument",
        "@func\ndef helper(value: i32) -> i32:\n    return value\n"
        "@fragment\ndef main(value: f32) -> i32:\n    return helper(value)\n",
        "cannot pass",
    ),
    _invalid_body(
        "LANG-GENERATED-RESOLUTION",
        "arguments",
        "resolution(value)",
        "@fragment\ndef main(value: f32) -> f32:\n    return resolution(value)[0]\n",
        "does not accept arguments",
    ),
    _invalid_body(
        "LANG-STRUCT-001",
        "constructor-arity",
        "Struct constructor arity",
        "@struct\nclass Pair:\n    x: f32\n    y: f32\n"
        "@fragment\ndef main(value: f32) -> f32:\n    return Pair(value).x\n",
        "constructor requires 2",
    ),
    _invalid_body(
        "LANG-TENSOR-003",
        "empty-vector",
        "empty Vector constructor",
        "@fragment\ndef main(value: f32) -> f32:\n    return Vector([])[0]\n",
        "requires a non-empty sequence literal",
    ),
    _invalid_body(
        "LANG-TENSOR-003",
        "mixed-vector",
        "incompatible Vector elements",
        "@fragment\ndef main(value: f32) -> f32:\n    return Vector([value, True])[0]\n",
        "elements have incompatible types",
    ),
    _invalid_body(
        "LANG-MATMUL-001",
        "arity",
        "matmul(value)",
        "@fragment\ndef main(value: f32) -> f32:\n    return matmul(value)\n",
        "matmul requires two",
    ),
    _invalid_body(
        "LANG-MATMUL-001",
        "scalar-left",
        "matmul scalar left operand",
        "@fragment\ndef main(value: f32) -> f32:\n    return matmul(value, value)\n",
        "left operand must be a non-scalar Tensor",
    ),
    _invalid_body(
        "LANG-MATMUL-001",
        "scalar-right",
        "matmul scalar right operand",
        "@fragment\ndef main(value: Matrix[f32, 2, 2]) -> f32:\n    return matmul(value, 1)\n",
        "right operand must be a non-scalar Tensor",
    ),
    _invalid_body(
        "LANG-HELPER-001",
        "unknown-call",
        "unknown(value)",
        "@fragment\ndef main(value: f32) -> f32:\n    return unknown(value)\n",
        "cannot infer call",
    ),
    _invalid_body(
        "LANG-LEGACY-001",
        "vec2-call",
        "vec2(value, value)",
        "@fragment\ndef main(value: f32) -> f32:\n    return vec2(value, value)[0]\n",
        "cannot infer call to 'vec2'",
    ),
)

LEGACY_FRONTEND_INVALID_CASES: Final = (
    _invalid_source(
        "LANG-LEGACY-001",
        "array-constructor",
        "Array[f32, 4]",
        "from vernon_dsl import *\n@func\ndef main(value: Array[f32, 4]) -> f32:\n    return 0.0\n",
        "unknown DSL type constructor 'Array'",
    ),
    _invalid_source(
        "LANG-LEGACY-001",
        "compute-decorator",
        "@compute",
        "from vernon_dsl import *\n@compute\ndef main(value: f32) -> f32:\n    return value\n",
        "unknown DSL decorator 'compute'",
    ),
    _invalid_source(
        "LANG-LEGACY-001",
        "workgroup-array",
        "workgroup_array(4)",
        "from vernon_dsl import *\n@kernel\ndef main() -> None:\n    values = workgroup_array(4)\n",
        "cannot infer call to 'workgroup_array'",
    ),
)

REMOVED_PUBLIC_NAMES: Final = (
    "vec",
    "mat",
    "vec2",
    "vec3",
    "vec4",
    "mat2",
    "mat3",
    "mat4",
)

_RAW_BUILTIN_SOURCE = """
from vernon_dsl import *
@vertex
def vertex_main(
    vertex: Annotated[u32, builtin("vertex_index")],
    instance: Annotated[u32, builtin("instance_index")],
) -> Annotated[Vector[f32, 4], builtin("position")]:
    return Vector([0.0, 0.0, 0.0, 1.0])
@fragment
def fragment_main(
    coordinate: Annotated[Vector[f32, 4], builtin("frag_coord")],
    facing: Annotated[bool, builtin("front_facing")],
) -> Vector[f32, 4]:
    return coordinate
@kernel(workgroup_size=(1, 1, 1))
def compute_main(
    global_id: Annotated[Vector[u32, 3], builtin("global_invocation_id")],
    local_id: Annotated[Vector[u32, 3], builtin("local_invocation_id")],
    group_id: Annotated[Vector[u32, 3], builtin("workgroup_id")],
) -> None:
    pass
"""

RAW_BUILTIN_VALID_CASES: Final = tuple(
    _valid_source(
        contract_id,
        "registered-interface",
        f'builtin("{builtin_name}")',
        _RAW_BUILTIN_SOURCE,
        valid_regions=frozenset(stage for stage, _ in BUILTIN_CONTRACTS[builtin_name].uses),
        capabilities=frozenset({"compute", "graphics"}),
        runtime_oracle="backend_interface_value",
        expected=f'vernon.builtin = "{builtin_name}"',
    )
    for contract_id, builtin_name in BUILTIN_NAMES_BY_CONTRACT
)

_GENERATED_BUILTIN_SOURCE = """
from vernon_dsl import *
@vertex
def vertex_main(position: Vector[f32, 4]) -> Vector[f32, 4]:
    vertex = vertex_id()
    instance = instance_id()
    return position
@fragment
def fragment_main() -> Vector[f32, 4]:
    size = resolution()
    coordinate = fragment_coord()
    facing = front_facing()
    return coordinate
"""

GENERATED_BUILTIN_VALID_CASES: Final = tuple(
    _valid_source(
        contract_id,
        "registered-operation",
        operation,
        _GENERATED_BUILTIN_SOURCE,
        valid_regions=frozenset({GENERATED_INTERFACE_CONTRACTS[operation.removesuffix("()")].stage}),
        capabilities=frozenset({"graphics"}),
        runtime_oracle="backend_interface_value",
        expected=expected,
    )
    for contract_id, operation, expected in (
        ("LANG-GENERATED-RESOLUTION", "resolution()", 'vernon.implicit = "resolution"'),
        (
            "LANG-GENERATED-FRAGMENT-COORD",
            "fragment_coord()",
            'vernon.builtin = "frag_coord"',
        ),
        (
            "LANG-GENERATED-FRONT-FACING",
            "front_facing()",
            'vernon.builtin = "front_facing"',
        ),
        ("LANG-GENERATED-VERTEX-ID", "vertex_id()", 'vernon.builtin = "vertex_index"'),
        (
            "LANG-GENERATED-INSTANCE-ID",
            "instance_id()",
            'vernon.builtin = "instance_index"',
        ),
    )
)

CORE_FRONTEND_IR_SOURCE_CASES: Final = (
    _valid_source(
        "LANG-SEM-001",
        "category-model",
        "Tensor Value load and TensorView Storage write",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def main(output: TensorView[f32, (dyn,), write], value: Tensor[f32, (2,)]) -> None:\n"
        "    output[0] = value[0]\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="stored_value",
        expected='"vernon.store"',
    ),
    _valid_source(
        "LANG-TENSOR-002",
        "normalized-shape",
        "Tensor[Tensor[f32, (3,)], (8,)]",
        "from vernon_dsl import *\n"
        "@func\n"
        "def identity(value: Tensor[Tensor[f32, (3,)], (8,)]) -> Tensor[f32, (8, 3)]:\n"
        "    return value\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset(),
        runtime_oracle="equal_indexing",
        expected="tensor<8x3xf32>",
    ),
    _valid_source(
        "LANG-ABI-001",
        "portable-aggregate",
        "Struct and Tensor aggregate ABI leaves",
        "from vernon_dsl import *\n@struct\nclass Vertex:\n    position: Tensor[f32, (3,)]\n    weight: f64\n",
        valid_regions=DEVICE_REGIONS,
        capabilities=frozenset(),
        runtime_oracle="portable_leaf_round_trip",
        expected='abi_leaf_dtypes = ["f32", "f64"]',
    ),
    _valid_source(
        "LANG-VIEW-002",
        "inferred-address-space",
        "device parameter and workgroup-local TensorView",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def main(output: TensorView[f32, (dyn,), write]) -> None:\n"
        "    local = workgroup_storage(f32, shape=(1,))\n"
        "    local[0] = 1.0\n"
        "    output[0] = local[0]\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers", "workgroup_memory"}),
        runtime_oracle="stored_value",
        expected='"workgroup"',
    ),
    _valid_source(
        "LANG-VIEW-007",
        "aggregate-aos-projection",
        "TensorView[Vertex, (dyn,), read_write]",
        "from vernon_dsl import *\n"
        "@struct\n"
        "class Vertex:\n"
        "    position: Vector[f32, 3]\n"
        "    weight: f32\n"
        "@kernel\n"
        "def main(values: TensorView[Vertex, (dyn,), read_write]) -> None:\n"
        "    value = values[0]\n"
        "    values[0] = value\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="aggregate_round_trip",
        expected='!vernon.struct<"Vertex">',
    ),
    _valid_source(
        "LANG-EFFECT-001",
        "structured-effects",
        "parameter-owned read and write effects",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def main(output: TensorView[f32, (dyn,), write], source: TensorView[f32, (dyn,), read]) -> None:\n"
        "    output[0] = source[0]\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="effect_order",
        expected="vernon.storage_effects",
    ),
    _valid_source(
        "LANG-ATOMIC-002",
        "floating-add",
        "atomic_add(f32 TensorView)",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def main(values: TensorView[f32, (dyn,), read_write]) -> None:\n"
        "    previous = atomic_add(values, 0, 1.0)\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers", "f32_atomic_add"}),
        runtime_oracle="atomic_contention_sum",
        expected='atomic_kind = "add"',
    ),
    _valid_source(
        "LANG-ENTRY-001",
        "registered-stages",
        "@kernel, @vertex, and @fragment",
        "from vernon_dsl import *\n"
        "@kernel\ndef compute_main() -> None:\n    pass\n"
        "@vertex\n"
        "def vertex_main(value: Vector[f32, 4]) -> Annotated[Vector[f32, 4], builtin('position')]:\n"
        "    return value\n"
        "@fragment\ndef fragment_main(value: f32) -> f32:\n    return value\n",
        valid_regions=frozenset({"compute", "vertex", "fragment"}),
        capabilities=frozenset({"compute", "graphics"}),
        runtime_oracle="entry_invocation",
        expected="vernon.entry",
    ),
    _valid_source(
        "LANG-HELPER-002",
        "shared-pure-helper",
        "@func(shared=True)",
        "from vernon_dsl import *\n"
        "@func(shared=True)\n"
        "def square(value: f32) -> f32:\n"
        "    return value * value\n"
        "@kernel\n"
        "def main(output: TensorView[f32, (dyn,), write]) -> None:\n"
        "    output[0] = square(2.0)\n",
        valid_regions=frozenset({"host", "compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="host_device_equal",
        expected="vernon.shared",
    ),
    _valid_source(
        "LANG-CONTROL-002",
        "lazy-conditional",
        "and/or and conditional expression",
        "from vernon_dsl import *\n"
        "@func\n"
        "def choose(a: bool, b: bool, left: f32, right: f32) -> f32:\n"
        "    enabled = a and b\n"
        "    return left if enabled else right\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset(),
        runtime_oracle="selected_value",
        expected="scf.if",
    ),
    _valid_source(
        "LANG-CONTROL-003",
        "dynamic-range",
        "range(start, stop, step)",
        "from vernon_dsl import *\n"
        "@func\n"
        "def total(start: i32, stop: i32, step: i32) -> i32:\n"
        "    result = 0\n"
        "    for index in range(start, stop, step):\n"
        "        result += index\n"
        "    return result\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset(),
        runtime_oracle="python_range_equal",
        expected="cf.assert",
    ),
    _valid_source(
        "LANG-CONTROL-004",
        "structured-exits",
        "break, continue, and early return",
        "from vernon_dsl import *\n"
        "@func\n"
        "def search(limit: i32) -> i32:\n"
        "    result = 0\n"
        "    for index in range(limit):\n"
        "        if index == 2:\n"
        "            continue\n"
        "        if index > 4:\n"
        "            break\n"
        "        result = index\n"
        "    return result\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset(),
        runtime_oracle="structured_exit_result",
        expected="vernon.loop_control_index",
    ),
    _valid_source(
        "LANG-MATH-001",
        "portable-math",
        "acos, atan2, floor, dot, and norm",
        "from vernon_dsl import *\n"
        "@fragment\n"
        "def main(y: f32, x: f32, value: Vector[f32, 2]) -> f32:\n"
        "    return acos(clamp(x, -1.0, 1.0)) + atan2(y, x) + floor(y) + value.norm()\n",
        valid_regions=frozenset({"fragment"}),
        capabilities=frozenset({"graphics"}),
        runtime_oracle="host_math_reference",
        expected="math.atan2",
    ),
)

FRONTEND_BOUNDARY_CASES: Final = (
    _invalid_source(
        "LANG-INTEROP-001",
        "not-a-source-type",
        "RawBuffer in authored source",
        "from vernon_dsl import *\n@kernel\ndef main(value: RawBuffer) -> None:\n    pass\n",
        "unknown DSL type 'RawBuffer'",
        invalid_regions=DEVICE_REGIONS,
    ),
)

HOST_CONTRACT_CASES: Final = (
    _valid_source(
        "LANG-VIEW-003",
        "strict-direct-binding",
        "TensorStorage.view shape, dtype, access, and layout validation",
        "TensorStorage.view(shape=(4,), access=read_write)",
        valid_regions=frozenset({"host"}),
        capabilities=frozenset({"storage_buffers"}),
        runtime_oracle="binding_validation",
        expected="strict_binding",
    ),
    _valid_source(
        "LANG-VIEW-005",
        "alias-lifetime-proof",
        "overlapping writable TensorView dispatch borrows",
        "TensorStorage.view()[slice_a], TensorStorage.view()[slice_b]",
        valid_regions=frozenset({"host"}),
        capabilities=frozenset({"storage_buffers"}),
        runtime_oracle="alias_and_lifetime_validation",
        expected="overlap",
    ),
)

_VJP_STORAGE_SOURCE = """
import vernon_dsl as vd
@vd.kernel
def objective(
    value: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss[0] = value[0] * value[0]
"""

_VJP_CONTROL_SOURCE = """
import vernon_dsl as vd
@vd.func
def bounded(value: vd.f32, stop: vd.i32) -> vd.f32:
    result = value
    for index in range(2):
        if index == stop:
            return result
        result = result * value
    return result
@vd.kernel
def objective(
    value: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    stop: vd.i32,
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss[0] = bounded(value[0], stop)
"""

_MODULE_FAN_OUT_SOURCE = """
from dataclasses import dataclass
import vernon_dsl as vd

@vd.kernel
def square(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0]

@vd.kernel
def cube(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0] * source[0]

@dataclass
class Branches:
    square: vd.TensorStorage
    cube: vd.TensorStorage

class FanOut(vd.Module):
    def forward(self, source: vd.TensorStorage) -> Branches:
        squared = vd.empty_like(source)
        cubed = vd.empty_like(source)
        square(source, squared, grid=(1, 1, 1))
        cube(source, cubed, grid=(1, 1, 1))
        return Branches(squared, cubed)

asset = vd.program_asset(
    id="module/fan-out-vjp",
    program=vd.ad.vjp(FanOut(), wrt=("source",), outputs=("square", "cube")),
)
"""

SPECIALIZED_FRONTEND_IR_CASES: Final = (
    _valid_source(
        "LANG-SPECIALIZE-001",
        "feature-identity",
        'feature("DOUBLE")',
        "from vernon_dsl import *\n"
        'DOUBLE = feature("DOUBLE")\n'
        "@fragment\n"
        "def main(value: f32) -> f32:\n"
        "    if DOUBLE:\n"
        "        return value * 2.0\n"
        "    return value\n",
        valid_regions=frozenset({"fragment"}),
        capabilities=frozenset({"graphics"}),
        runtime_oracle="variant_identity",
        expected="vernon.features",
    ),
    _valid_source(
        "LANG-AD-002",
        "storage-leaves",
        "VJP of TensorView load and store leaves",
        _VJP_STORAGE_SOURCE,
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers", "program_vjp"}),
        runtime_oracle="storage_gradient",
        expected="owned_gradient_storage",
    ),
    _valid_source(
        "LANG-AD-003",
        "accepted-control-flow",
        "VJP through supported structured control flow",
        _VJP_CONTROL_SOURCE,
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers", "program_vjp"}),
        runtime_oracle="control_flow_gradient",
        expected="replayed_control_flow",
    ),
    _valid_source(
        "LANG-AD-004",
        "first-order-compute-vjp",
        "first-order compute Program VJP",
        _PROGRAM_VJP_ASSET_SOURCE,
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers", "program_vjp"}),
        runtime_oracle="supported_transform_boundary",
        expected="first_order_vjp",
    ),
)

PAIRWISE_IR_SOURCE_CASES: Final = (
    _valid_source(
        "LANG-PAIR-001",
        "nested-tensor-struct-boundary",
        "Tensor[StructWithTensor, (2,)]",
        "from vernon_dsl import *\n"
        "@struct\n"
        "class Box:\n"
        "    value: Tensor[f32, (3,)]\n"
        "@func\n"
        "def identity(value: Tensor[Box, (2,)]) -> Tensor[Box, (2,)]:\n"
        "    return value\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset(),
        runtime_oracle="struct_boundary_preserved",
        expected=(
            '!vernon.tensor<!vernon.struct<"Box">, [2]>',
            'fields = ["value:tensor<3xf32>"]',
        ),
    ),
    _valid_source(
        "LANG-PAIR-002",
        "aggregate-dynamic-descriptor",
        "dynamic TensorView of aggregate elements",
        "from vernon_dsl import *\n"
        "@struct\n"
        "class Pair:\n"
        "    left: f32\n"
        "    right: f32\n"
        "@kernel\n"
        "def main(values: TensorView[Pair, (dyn,), read_write]) -> None:\n"
        "    values[0] = values[1]\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="descriptor_layout_reuse",
        expected='!vernon.tensor_view<!vernon.struct<"Pair">, [-1], "read_write", "device">',
    ),
    _valid_source(
        "LANG-PAIR-005",
        "helper-entry-effects",
        "effectful helper called from compute entry",
        "from vernon_dsl import *\n"
        "@func\n"
        "def copy(output: TensorView[f32, (dyn,), write], source: TensorView[f32, (dyn,), read]) -> None:\n"
        "    output[0] = source[0]\n"
        "@kernel\n"
        "def main(output: TensorView[f32, (dyn,), write], source: TensorView[f32, (dyn,), read]) -> None:\n"
        "    copy(output, source)\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="propagated_entry_effects",
        expected=(
            "vernon.storage_effects",
            "func.call @copy",
            '"vernon.load"',
            '"vernon.store"',
        ),
    ),
    _valid_source(
        "LANG-PAIR-006",
        "aggregate-workgroup-barrier",
        "aggregate workgroup storage, barrier, and non-zero index",
        "from vernon_dsl import *\n"
        "@struct\n"
        "class Pair:\n"
        "    left: i32\n"
        "    right: f32\n"
        "@kernel\n"
        "def main(output: TensorView[f32, (dyn,), write]) -> None:\n"
        "    values = workgroup_storage(Pair, shape=(2, 3))\n"
        "    values[1, 2] = Pair(7, 2.5)\n"
        "    workgroup_barrier()\n"
        "    output[0] = values[1, 2].right\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers", "workgroup_memory"}),
        runtime_oracle="barrier_publication",
        expected=MlirOracle(
            required=(
                '!vernon.tensor_view<!vernon.struct<"Pair">, [2, 3], "read_write", "workgroup">',
                "arith.constant 1 : i32",
                "arith.constant 2 : i32",
                '"vernon.struct_get"',
            ),
            counts=(('"vernon.barrier"', 1),),
        ),
    ),
    _valid_source(
        "LANG-PAIR-008",
        "sampling-stage-sampler-mode",
        "implicit fragment sampling and explicit vertex LOD sampling",
        "from vernon_dsl import *\n"
        "@fragment\n"
        "def fragment_main(image: Texture['2d', f32], uv: Vector[f32, 2]) -> Vector[f32, 4]:\n"
        "    return texture_sample(image, uv)\n"
        "@vertex\n"
        "def vertex_main(image: Texture['2d', f32], sampler: Sampler, uv: Vector[f32, 2]) -> Vector[f32, 4]:\n"
        "    return texture_sample(image, sampler, uv, 0.0)\n",
        valid_regions=frozenset({"vertex", "fragment"}),
        capabilities=frozenset({"graphics", "texture_sampler"}),
        runtime_oracle="selected_filtered_texel",
        expected=MlirOracle(
            required=(
                "!vernon.sampler",
                'vernon.stage = "vertex"',
                'vernon.stage = "fragment"',
            ),
            counts=(
                ('name = "texture_sample"', 2),
                ('vernon.implicit = "sampler"', 1),
            ),
        ),
    ),
    _valid_source(
        "LANG-PAIR-009",
        "control-aggregate-value",
        "loop and conditional carrying Struct with Tensor field",
        "from vernon_dsl import *\n"
        "@struct\n"
        "class State:\n"
        "    value: Tensor[f32, (2,)]\n"
        "@func\n"
        "def update(state: State, enabled: bool) -> State:\n"
        "    result = state\n"
        "    for index in range(2):\n"
        "        if enabled:\n"
        "            result = State(result.value + Tensor([1.0, 1.0]))\n"
        "    return result\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset(),
        runtime_oracle="aggregate_control_result",
        expected=(
            '!vernon.struct<"State">',
            "scf.for",
            "scf.if",
            '"vernon.struct_create"',
        ),
    ),
    _valid_source(
        "LANG-PAIR-013",
        "carrier-portable-abi",
        "backend carrier packing of a portable aggregate ABI",
        "from vernon_dsl import *\n"
        "@struct\n"
        "class Payload:\n"
        "    value: f32\n"
        "    index: i32\n"
        "@kernel\n"
        "def main(values: TensorView[Payload, (dyn,), read_write]) -> None:\n"
        "    values[0] = values[1]\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="carrier_round_trip",
        expected='abi_leaf_dtypes = ["f32", "i32"]',
    ),
)

PAIRWISE_SPECIALIZED_CASES: Final = (
    _valid_source(
        "LANG-PAIR-003",
        "field-projection-alias-proof",
        "shared-owner writable field projections",
        "from vernon_dsl import *\n"
        "@struct\n"
        "class Vertex:\n"
        "    position: Vector[f32, 3]\n"
        "    weight: f64\n"
        "@kernel\n"
        "def main(values: TensorView[Vertex, (dyn,), read_write]) -> None:\n"
        "    values[0] = values[1]\n",
        valid_regions=frozenset({"host", "compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="field_projection_alias_validation",
        expected='!vernon.struct<"Vertex">',
    ),
    _valid_source(
        "LANG-PAIR-004",
        "helper-feature-identity",
        "generic helper specialized under captured feature",
        "from vernon_dsl import *\n"
        "import helpers\n"
        'DOUBLE = feature("DOUBLE")\n'
        "@fragment\n"
        "def main(value: f32) -> f32:\n"
        "    selected = value\n"
        "    if DOUBLE:\n"
        "        selected = f32(helpers.identity(f64(value)))\n"
        "    else:\n"
        "        selected = helpers.identity(value)\n"
        "    return selected\n",
        valid_regions=frozenset({"fragment"}),
        capabilities=frozenset({"graphics"}),
        runtime_oracle="specialization_identity",
        expected="func.func private @identity__",
    ),
    _valid_source(
        "LANG-PAIR-010",
        "vjp-structured-control",
        "VJP through bounded loop and early return",
        _VJP_CONTROL_SOURCE,
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers", "program_vjp"}),
        runtime_oracle="control_flow_gradient",
        expected="replayed_control_flow",
    ),
    _valid_source(
        "LANG-PAIR-011",
        "vjp-signed-stride-view",
        "VJP over a dynamic TensorView whose runtime layout may have signed strides",
        _VJP_STORAGE_SOURCE,
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers", "program_vjp"}),
        runtime_oracle="signed_stride_gradient",
        expected="runtime_layout_absent_from_vjp_ir",
    ),
    _valid_source(
        "LANG-PAIR-012",
        "module-fan-in-out-versions",
        "Module fan-out and fan-in across Value and Storage versions",
        _MODULE_FAN_OUT_SOURCE,
        valid_regions=frozenset({"host", "compute"}),
        capabilities=frozenset({"compute", "storage_buffers", "program_vjp"}),
        runtime_oracle="fan_in_accumulation",
        expected="captured_fan_out_vjp",
    ),
    _valid_source(
        "LANG-PAIR-014",
        "injectivity-dynamic-launch",
        "dynamic launch geometry with writable view injectivity",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def main(output: TensorView[f32, (dyn, dyn), write]) -> None:\n"
        "    output[0, 0] = 1.0\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="injectivity_before_dispatch",
        expected=("vernon.storage_effects", "indices = array<i64: 0, 0>"),
    ),
)

ADDITIONAL_FRONTEND_IR_SOURCE_CASES: Final = (
    _valid_source(
        "LANG-SCALAR-002",
        "contextual-conversion",
        "contextual literals and safe widening",
        "from vernon_dsl import *\n@func\ndef main(value: f64, count: i32) -> f64:\n    return value + count + 0.5\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset(),
        runtime_oracle="host_numeric_reference",
        expected="arith.sitofp",
    ),
    _valid_source(
        "LANG-SCALAR-003",
        "arithmetic-comparison-division",
        "typed arithmetic, comparison, and integer true division",
        "from vernon_dsl import *\n"
        "@func\n"
        "def main(left: i32, right: i32) -> f32:\n"
        "    value = left / right\n"
        "    if left < right:\n"
        "        value = value + 1.0\n"
        "    return value\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset(),
        runtime_oracle="host_numeric_reference",
        expected="arith.divf",
    ),
    _valid_source(
        "LANG-TENSOR-003",
        "rectangular-constructors",
        "Tensor, Vector, and Matrix constructors",
        "from vernon_dsl import *\n"
        "@func\n"
        "def main(value: f32) -> Tensor[f32, (2,)]:\n"
        "    vector = Vector([value, 1.0])\n"
        "    matrix = Matrix([[1.0, 0.0], [0.0, 1.0]])\n"
        "    tensor = Tensor([value, 1.0])\n"
        "    return tensor + matmul(matrix, vector)\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset(),
        runtime_oracle="constructor_values",
        expected='name = "construct"',
    ),
    _valid_source(
        "LANG-TUPLE-001",
        "construct-index-destructure",
        "Tuple construction, constant indexing, and destructuring",
        "from vernon_dsl import *\n"
        "@func\n"
        "def main(value: f32) -> f32:\n"
        "    pair = (value, i32(2))\n"
        "    first, second = pair\n"
        "    return first + f32(second) + pair[0]\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset(),
        runtime_oracle="tuple_values",
        expected='"vernon.tuple_create"',
    ),
    _valid_source(
        "LANG-CONTROL-001",
        "structured-statements",
        "if, while, for, and assignment",
        "from vernon_dsl import *\n"
        "@func\n"
        "def main(limit: i32) -> i32:\n"
        "    result = 0\n"
        "    for index in range(limit):\n"
        "        if index > 2:\n"
        "            result += index\n"
        "    return result\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset(),
        runtime_oracle="structured_control_result",
        expected="scf.for",
    ),
    _valid_source(
        "LANG-MATMUL-001",
        "matrix-vector",
        "matmul(Matrix, Vector)",
        "from vernon_dsl import *\n"
        "@func\n"
        "def main(matrix: Matrix[f32, 2, 2], vector: Vector[f32, 2]) -> Vector[f32, 2]:\n"
        "    return matmul(matrix, vector)\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset(),
        runtime_oracle="host_matmul_reference",
        expected='name = "matmul"',
    ),
    _valid_source(
        "LANG-TEXTURE-SAMPLE-LOD",
        "floating-lod",
        "texture_sample(texture, coordinates, floating_lod)",
        "from vernon_dsl import *\n"
        "@fragment\n"
        "def main(image: Texture['2d', f32], uv: Vector[f32, 2]) -> Vector[f32, 4]:\n"
        "    return texture_sample(image, uv, 1.0)\n",
        valid_regions=frozenset({"fragment"}),
        capabilities=frozenset({"graphics", "texture_sampler"}),
        runtime_oracle="selected_filtered_mip",
        expected='name = "texture_sample"',
    ),
    _valid_source(
        "LANG-TEXTURE-SIZE",
        "base-and-mip-size",
        "texture_size(texture) and texture_size(texture, lod)",
        "from vernon_dsl import *\n"
        "@fragment\n"
        "def main(image: Texture['2d', f32]) -> Vector[u32, 2]:\n"
        "    base = texture_size(image)\n"
        "    mip = texture_size(image, 1)\n"
        "    return base + mip\n",
        valid_regions=frozenset({"fragment"}),
        capabilities=frozenset({"graphics", "texture_sampler"}),
        runtime_oracle="mip_extent",
        expected='name = "texture_size"',
    ),
    _valid_source(
        "LANG-LEGACY-001",
        "canonical-replacements",
        "TensorStorage, TensorView, Tensor, Vector, Matrix, and workgroup_storage",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def main(output: TensorView[f32, (dyn,), write]) -> None:\n"
        "    local = workgroup_storage(f32, shape=(1,))\n"
        "    output[0] = Vector([local[0]])[0]\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers", "workgroup_memory"}),
        runtime_oracle="canonical_replacement_behavior",
        expected='"vernon.workgroup_alloc"',
    ),
    _valid_source(
        "LANG-BARRIER-001",
        "compute-barriers",
        "workgroup_barrier() and device_barrier()",
        "from vernon_dsl import *\n@kernel\ndef main() -> None:\n    workgroup_barrier()\n    storage_barrier()\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "workgroup_memory"}),
        runtime_oracle="barrier_publication",
        expected='"vernon.barrier"',
    ),
    _valid_source(
        "LANG-ENTRY-002",
        "annotated-signature",
        "fully annotated entry parameters and result",
        "from vernon_dsl import *\n"
        "@kernel\n"
        "def main(value: f32, output: TensorView[f32, (dyn,), write]) -> None:\n"
        "    output[0] = value\n",
        valid_regions=frozenset({"compute"}),
        capabilities=frozenset({"compute", "storage_buffers"}),
        runtime_oracle="entry_abi_binding",
        expected="func.func @main",
    ),
    _valid_source(
        "LANG-HELPER-001",
        "deterministic-specialization",
        "unannotated pure helper specialization",
        "from vernon_dsl import *\n"
        "@func\n"
        "def identity(value):\n"
        "    return value\n"
        "@fragment\n"
        "def main(value: f32) -> f32:\n"
        "    return identity(value)\n",
        valid_regions=frozenset({"func", "fragment"}),
        capabilities=frozenset({"graphics"}),
        runtime_oracle="specialized_result",
        expected="func.func private @identity__",
    ),
    _valid_source(
        "LANG-TEXTURE-SAMPLE-IMPLICIT",
        "fragment",
        "texture_sample(texture, coordinates)",
        "from vernon_dsl import *\n"
        "@fragment\n"
        "def main(image: Texture['2d', f32], uv: Vector[f32, 2]) -> Vector[f32, 4]:\n"
        "    return texture_sample(image, uv)\n",
        valid_regions=frozenset({"fragment"}),
        capabilities=frozenset({"graphics", "texture_sampler"}),
        runtime_oracle="filtered_texel",
        expected='name = "texture_sample"',
    ),
    _valid_source(
        "LANG-TEXTURE-SAMPLE-EXPLICIT",
        "fragment",
        "texture_sample(texture, sampler, coordinates)",
        "from vernon_dsl import *\n"
        "@fragment\n"
        "def main(image: Texture['2d', f32], sampler: Sampler, uv: Vector[f32, 2]) -> Vector[f32, 4]:\n"
        "    return texture_sample(image, sampler, uv)\n",
        valid_regions=frozenset({"fragment"}),
        capabilities=frozenset({"graphics", "texture_sampler"}),
        runtime_oracle="filtered_texel",
        expected='name = "texture_sample"',
    ),
)

HOST_API_CONTRACT_CASES: Final = (
    _valid_source(
        "LANG-STORAGE-001",
        "owning-storage",
        "TensorStorage owns mutable host/runtime storage",
        "import vernon_dsl as vd\nvalue = vd.TensorStorage.zeros(dtype=vd.f32, shape=(2,))\n",
        valid_regions=frozenset({"host"}),
        capabilities=frozenset({"storage_buffers"}),
        runtime_oracle="owner_round_trip",
        expected="TensorStorage",
    ),
    _valid_source(
        "LANG-INTEROP-001",
        "raw-buffer-boundary",
        "RawBuffer explicit byte-layout host interop",
        "import numpy as np\n"
        "import vernon_dsl as vd\n"
        "values = np.arange(12, dtype=np.float32).reshape(3, 4)\n"
        "backing = bytearray(values.tobytes())\n"
        "raw = vd.interop.RawBuffer.from_buffer(backing, alignment=4)\n"
        "value = raw.typed_view(\n"
        "    dtype=vd.f32,\n"
        "    shape=(3, 2),\n"
        "    byte_strides=(16, 4),\n"
        "    byte_offset=4,\n"
        "    access='read_write',\n"
        "    layout_units='bytes',\n"
        ")\n",
        valid_regions=frozenset({"host"}),
        capabilities=frozenset({"storage_buffers"}),
        runtime_oracle="typed_view_round_trip",
        expected="RawBuffer",
    ),
)

_CROSS_REGION_VALUE_SOURCE: Final = """
from vernon_dsl import *

@struct
class Payload:
    value: f32
    vector: Vector[f32, 4]

@func
def helper(value: f32, tensor: Tensor[f32, (2,)]) -> Payload:
    matrix = Matrix([[1.0, 0.0], [0.0, 1.0]])
    return Payload(value + tensor[0] + matrix[0, 0], Vector([value, 0.0, 0.0, 1.0]))

@kernel
def compute_main(output: TensorView[f32, (dyn,), write], value: f32) -> None:
    tensor = Tensor([value, 1.0])
    output[0] = helper(value, tensor).value

@vertex
def vertex_main(value: f32) -> Annotated[Vector[f32, 4], builtin("position")]:
    tensor = Tensor([value, 1.0])
    return helper(value, tensor).vector

@fragment
def fragment_main(value: f32) -> Vector[f32, 4]:
    tensor = Tensor([value, 1.0])
    return helper(value, tensor).vector
"""

_CROSS_REGION_RESOURCE_SOURCE: Final = """
from vernon_dsl import *

@func
def resource_helper(image: Texture["2d", f32], sampler: Sampler) -> None:
    pass

@kernel
def compute_main(
    output: TensorView[f32, (dyn,), write],
    image: Texture["2d", f32],
    sampler: Sampler,
) -> None:
    output[0] = texture_sample(image, sampler, Vector([0.0, 0.0]), 0.0)[0]

@vertex
def vertex_main(
    image: Texture["2d", f32],
    sampler: Sampler,
) -> Annotated[Vector[f32, 4], builtin("position")]:
    return texture_sample(image, sampler, Vector([0.0, 0.0]), 0.0)

@fragment
def fragment_main(image: Texture["2d", f32]) -> Vector[f32, 4]:
    return texture_sample(image, Vector([0.0, 0.0]))
"""

_CROSS_REGION_INTERFACE_SOURCE: Final = """
from vernon_dsl import *

@kernel
def compute_main(
    output: TensorView[u32, (dyn,), write],
    gid: Annotated[Vector[u32, 3], builtin("global_invocation_id")],
) -> None:
    output[0] = gid[0]

@vertex
def vertex_main(
    position: Annotated[Vector[f32, 4], attribute(location=0)],
    scale: Annotated[f32, uniform(set=0, binding=0)],
) -> Annotated[Vector[f32, 4], builtin("position")]:
    return position * scale

@fragment
def fragment_main(
    coordinate: Annotated[Vector[f32, 4], varying()],
    facing: Annotated[bool, builtin("front_facing")],
) -> Vector[f32, 4]:
    return coordinate
"""

CROSS_REGION_FRONTEND_IR_CASES: Final = (
    _valid_source(
        "LANG-SCALAR-001",
        "verified-all-regions",
        "f32 in helper and every entry region",
        _CROSS_REGION_VALUE_SOURCE,
        valid_regions=DEVICE_REGIONS,
        capabilities=frozenset({"compute", "graphics", "storage_buffers"}),
        runtime_oracle="canonical_scalar_ir",
        expected="f32",
    ),
    _valid_source(
        "LANG-TENSOR-001",
        "verified-all-regions",
        "static Tensor values in helper and every entry region",
        _CROSS_REGION_VALUE_SOURCE,
        valid_regions=DEVICE_REGIONS,
        capabilities=frozenset({"compute", "graphics", "storage_buffers"}),
        runtime_oracle="canonical_tensor_ir",
        expected="tensor<2xf32>",
    ),
    _valid_source(
        "LANG-TENSOR-004",
        "verified-all-regions",
        "Vector and Matrix aliases in helper and every entry region",
        _CROSS_REGION_VALUE_SOURCE,
        valid_regions=DEVICE_REGIONS,
        capabilities=frozenset({"compute", "graphics", "storage_buffers"}),
        runtime_oracle="canonical_alias_ir",
        expected=("tensor<4xf32>", "tensor<2x2xf32>"),
    ),
    _valid_source(
        "LANG-STRUCT-001",
        "verified-all-regions",
        "Struct values in helper and every entry region",
        _CROSS_REGION_VALUE_SOURCE,
        valid_regions=DEVICE_REGIONS,
        capabilities=frozenset({"compute", "graphics", "storage_buffers"}),
        runtime_oracle="canonical_struct_ir",
        expected='!vernon.struct<"Payload">',
    ),
    _valid_source(
        "LANG-RESOURCE-001",
        "verified-all-regions",
        "Texture and Sampler resources in helper and every entry region",
        _CROSS_REGION_RESOURCE_SOURCE,
        valid_regions=DEVICE_REGIONS,
        capabilities=frozenset({"compute", "graphics", "storage_buffers", "texture_sampler"}),
        runtime_oracle="canonical_resource_ir",
        expected=("!vernon.texture", "!vernon.sampler"),
    ),
    _valid_source(
        "LANG-RESOURCE-002",
        "verified-all-regions",
        "sampled Texture type in helper and every entry region",
        _CROSS_REGION_RESOURCE_SOURCE,
        valid_regions=DEVICE_REGIONS,
        capabilities=frozenset({"compute", "graphics", "storage_buffers", "texture_sampler"}),
        runtime_oracle="canonical_texture_ir",
        expected='!vernon.texture<"2d", f32, "unknown", "sampled">',
    ),
    _valid_source(
        "LANG-INTERFACE-001",
        "verified-entry-regions",
        "legal interface metadata in compute, vertex, and fragment entries",
        _CROSS_REGION_INTERFACE_SOURCE,
        valid_regions=ENTRY_REGIONS,
        capabilities=frozenset({"compute", "graphics", "storage_buffers"}),
        runtime_oracle="canonical_interface_ir",
        expected=("vernon.builtin", "vernon.interface"),
    ),
    _valid_source(
        "LANG-VIEW-001",
        "verified-helper-region",
        "TensorView parameter in a helper region",
        "from vernon_dsl import *\n"
        "@func\n"
        "def first(values: TensorView[f32, (dyn,), read]) -> f32:\n"
        "    return values[0]\n",
        valid_regions=frozenset({"func"}),
        capabilities=frozenset({"storage_buffers"}),
        runtime_oracle="canonical_view_ir",
        expected='!vernon.tensor_view<f32, [-1], "read", "device">',
    ),
)

FRONTEND_IR_SOURCE_CASES: Final = (
    CORE_FRONTEND_IR_SOURCE_CASES + ADDITIONAL_FRONTEND_IR_SOURCE_CASES + CROSS_REGION_FRONTEND_IR_CASES
)

LANGUAGE_CONTRACT_CASES: Final = (
    TYPE_PARSER_VALID_CASES
    + TYPE_PARSER_INVALID_CASES
    + METADATA_VALID_CASES
    + METADATA_INVALID_CASES
    + BUILTIN_INVALID_CASES
    + TEXTURE_OPERATION_VALID_CASES
    + TEXTURE_OPERATION_INVALID_CASES
    + ATOMIC_INVALID_CASES
    + WORKGROUP_INVALID_CASES
    + SYNCHRONIZATION_VALID_CASES
    + TENSOR_VIEW_VALID_CASES
    + TENSOR_VIEW_INVALID_CASES
    + DISPATCH_CONTRACT_CASES
    + PROGRAM_AUTODIFF_CASES
    + TENSOR_ELEMENT_INVALID_CASES
    + STRUCT_FIELD_INVALID_CASES
    + SCALAR_CONVERSION_INVALID_CASES
    + HELPER_SPECIALIZATION_INVALID_CASES
    + TUPLE_INDEX_INVALID_CASES
    + ENTRY_SIGNATURE_INVALID_CASES
    + INFERENCE_STATEMENT_INVALID_CASES
    + INFERENCE_CALL_INVALID_CASES
    + LEGACY_FRONTEND_INVALID_CASES
    + RAW_BUILTIN_VALID_CASES
    + GENERATED_BUILTIN_VALID_CASES
    + FRONTEND_IR_SOURCE_CASES
    + FRONTEND_BOUNDARY_CASES
    + HOST_CONTRACT_CASES
    + SPECIALIZED_FRONTEND_IR_CASES
    + PAIRWISE_IR_SOURCE_CASES
    + PAIRWISE_SPECIALIZED_CASES
    + HOST_API_CONTRACT_CASES
)

CONTRACT_CASE_GROUPS: Final = {
    "TYPE_PARSER_VALID_CASES": TYPE_PARSER_VALID_CASES,
    "TYPE_PARSER_INVALID_CASES": TYPE_PARSER_INVALID_CASES,
    "METADATA_VALID_CASES": METADATA_VALID_CASES,
    "METADATA_INVALID_CASES": METADATA_INVALID_CASES,
    "BUILTIN_INVALID_CASES": BUILTIN_INVALID_CASES,
    "READONLY_ATOMIC_INVALID_CASES": READONLY_ATOMIC_INVALID_CASES,
    "RAW_BUILTIN_VALID_CASES": RAW_BUILTIN_VALID_CASES,
    "GENERATED_BUILTIN_VALID_CASES": GENERATED_BUILTIN_VALID_CASES,
    "FRONTEND_IR_SOURCE_CASES": FRONTEND_IR_SOURCE_CASES,
    "FRONTEND_BOUNDARY_CASES": FRONTEND_BOUNDARY_CASES,
    "PAIRWISE_IR_SOURCE_CASES": PAIRWISE_IR_SOURCE_CASES,
    "TENSOR_ELEMENT_INVALID_CASES": TENSOR_ELEMENT_INVALID_CASES,
    "STRUCT_FIELD_INVALID_CASES": STRUCT_FIELD_INVALID_CASES,
    "SCALAR_CONVERSION_INVALID_CASES": SCALAR_CONVERSION_INVALID_CASES,
    "HELPER_SPECIALIZATION_INVALID_CASES": HELPER_SPECIALIZATION_INVALID_CASES,
    "TUPLE_INDEX_INVALID_CASES": TUPLE_INDEX_INVALID_CASES,
    "ENTRY_SIGNATURE_INVALID_CASES": ENTRY_SIGNATURE_INVALID_CASES,
    "INFERENCE_STATEMENT_INVALID_CASES": INFERENCE_STATEMENT_INVALID_CASES,
    "INFERENCE_CALL_INVALID_CASES": INFERENCE_CALL_INVALID_CASES,
    "LEGACY_FRONTEND_INVALID_CASES": LEGACY_FRONTEND_INVALID_CASES,
}


def cases_for_contract(contract_id: str) -> tuple[LanguageContractCase, ...]:
    return tuple(case for case in LANGUAGE_CONTRACT_CASES if case.contract_id == contract_id)


def case_by_id(case_id: str) -> LanguageContractCase:
    matches = tuple(case for case in LANGUAGE_CONTRACT_CASES if case.id == case_id)
    if len(matches) != 1:
        raise KeyError(case_id)
    return matches[0]


def audit_case_registry(inventory_contract_ids: frozenset[str] | None = None) -> None:
    ids = [case.id for case in LANGUAGE_CONTRACT_CASES]
    duplicates = sorted(case_id for case_id in set(ids) if ids.count(case_id) > 1)
    if duplicates:
        raise AssertionError(f"duplicate language contract case IDs: {duplicates}")
    builtin_names = {name for _, name in BUILTIN_NAMES_BY_CONTRACT}
    if builtin_names != BUILTIN_CONTRACTS.keys():
        raise AssertionError("language builtin cases do not match the canonical shader builtin registry")
    if inventory_contract_ids is not None:
        unknown_contract_ids = {case.contract_id for case in LANGUAGE_CONTRACT_CASES} - inventory_contract_ids
        if unknown_contract_ids:
            raise AssertionError(f"cases reference unknown contract IDs: {sorted(unknown_contract_ids)}")
    for case in LANGUAGE_CONTRACT_CASES:
        if not case.source_construct:
            raise AssertionError(f"{case.id} has no source construct")
        if not case.valid_regions and not case.invalid_regions:
            raise AssertionError(f"{case.id} has no applicable region")
        unknown_regions = (case.valid_regions | case.invalid_regions) - KNOWN_REGIONS
        if unknown_regions:
            raise AssertionError(f"{case.id} has unknown regions: {sorted(unknown_regions)}")
        unknown_capabilities = case.capabilities - KNOWN_CAPABILITIES
        if unknown_capabilities:
            raise AssertionError(f"{case.id} has unknown capabilities: {sorted(unknown_capabilities)}")
        if case.expected_diagnostic is None and not case.valid_regions:
            raise AssertionError(f"{case.id} has no expected diagnostic")
        if case.expected_diagnostic is not None:
            re.compile(case.expected_diagnostic)
        if not case.runtime_oracle:
            raise AssertionError(f"{case.id} has no runtime oracle disposition")
