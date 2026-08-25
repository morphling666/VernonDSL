# Program Execution Manifest

## 1. Status and contract selection

This document defines the normative deployment representation of a Vernon
Program. Every executable is a Program. A standalone compute kernel is a
one-node Program, and a standalone graphics pipeline is a one-node Program.

The only Program-bearing variant member is the strict `variant.program`
object. There is no sibling or nested execution object.

The Program representation has no independent schema or version field.
`COMPILER_CONTRACT_VERSION` and `PIPELINE_VERSION` jointly select exactly one
Program contract. This document does not change the current constants 12 and
16, and that pair does **not** map to this schema. A producer MUST NOT emit
this Program representation until a future coordinated compiler/pipeline
release assigns a new pair to it. After that release, a loader MUST require
the unique Program contract mapped by the selected pair and reject every
other shape. It MUST NOT translate another representation into this one.

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** are normative.

## 2. Architectural rules

A Program is a static directed acyclic graph of compute and graphics
operations. It contains these authorities:

1. `stages` declares portable logical stage contracts.
2. `parameters` declares instance-bound values.
3. `storages` is the sole byte-resource and resource-descriptor authority.
4. `values` defines the logical value ABI and immutable value versions.
5. `graphs` defines dataflow, resource access, and operation metadata.
6. `signature` defines the public call ABI.
7. `residual_contract`, when present, defines forward-to-backward captures.

The Program object and every ABI object it contains are platform-neutral.
ArtifactSystem is the only target-specific deployment layer: it selects code
format and target requirements, but it cannot inject native API binding
locations into Program or endpoint ABI. Descriptor sets, root parameters,
Metal indices, GL locations, register numbers, native handles, and native
pipeline-state objects are forbidden manifest data.

Names are diagnostic labels and never identity. IDs, artifact keys, Parameter
paths, Signature paths, and array position are identity where this document
says so.

The Program graph is semantic SSA. A write creates a new Value ID for the
same Storage ID. A backend MAY update physical memory in place only when the
observable behavior is identical to the declared versions and transactional
rules.

Barriers and backend synchronization are derived from the canonical DAG,
resource accesses, attachment transitions, and target rules. They are never
serialized as nodes.

Only direct compute dispatch and direct or indexed graphics draw are
supported. Transfer, presentation, boundary, barrier, indirect execution, and
control-flow nodes are invalid. Graph regions, branches, loops, calls, and
runtime-created nodes are invalid.

## 3. Canonical JSON profile

### 3.1 Syntax and primitive values

The manifest uses UTF-8 JSON with:

- no byte-order mark;
- no comments, trailing commas, duplicate object names, `NaN`, or infinity;
- shortest round-trippable decimal spelling for finite JSON numbers;
- integers serialized without a decimal point or exponent;
- Unicode strings in NFC;
- lowercase hexadecimal digests;
- base64 according to RFC 4648 with required padding.

All objects defined here are strict. Unknown members are errors. Optional
members are omitted when absent; they are not encoded as `null` unless
`null` is explicitly listed as a valid value.

IDs, indices, slots, alignments, offsets, lengths, counts, and static extents
are unsigned integers. Byte arithmetic and extent arithmetic use checked
uint64 operations. Fields documented as uint32 MUST fit `0..2^32-1`.

### 3.2 Canonical object and array order

Object member order is not semantic. Canonical bytes use lexicographically
ascending UTF-8 member names.

Array order is semantic and canonical as follows:

- `parameters`, `storages`, `values`, `shape_symbols`, and graph `nodes` are
  ordered by contiguous `id` equal to array index;
- `graphs` are ordered forward, then backward when present;
- graph `inputs` are ordered by boundary class in this exact order:
  `user_input`, `cotangent`, `parameter`, `allocation`, `constant`, `storage`,
  records within a class are ordered by slot or Value ID;
- graph `outputs` and signature arrays are public ABI order;
- node `operands` and `results` are ordered Value IDs with no duplicates;
- node `bindings` are ordered by module ordinal (`compute` = 0, `vertex` = 1,
  `fragment` = 2), interface ordinal (`argument` = 0, `result` = 1), then
  endpoint index;
- node `accesses` are ordered by the key defined in section 11;
- color attachments and blend records are ordered by location;
- backward `captures` and `residual_contract.captures` are ordered by captured
  Value ID; residual shape symbols and every replay `required_values` array
  are ordered by ascending ID without duplicates;
- ValueLayout leaves are ordered by canonical path, and reflected endpoints
  are ordered by module ordinal, interface ordinal, then endpoint index;
- usage, aspect, channel-mask, and capability sets are sorted lexical arrays
  without duplicates;
- constraints and alias preconditions are sorted by canonical JSON bytes.

No producer or loader may reorder an ABI-order array. Canonicalization MUST
fail rather than silently repair non-canonical input. Any array not assigned a
sort rule above or by its defining section preserves declared semantic order.

### 3.3 Shapes and expressions

A `ShapeDim` is a positive integer, `-1` for TensorView `vd.dyn`, or
`{"symbol": S}` when a ShapeSymbol is explicitly declared. An empty shape
denotes rank zero. `null` and bare strings are invalid. Ordinary TensorView
`vd.dyn` uses `-1` on Value `.shape` and does not allocate a ShapeSymbol.

A `ShapeExpr` is exactly one of:

```json
{"constant": 4}
{"symbol": 0}
{"op": "add", "args": [{"symbol": 0}, {"constant": 1}]}
{"op": "mul", "args": [{"symbol": 0}, {"constant": 4}]}
{"op": "align_up", "value": {"symbol": 0}, "alignment": 16}
```

`add` and `mul` have at least two operands. Their operands are flattened and
sorted by canonical JSON bytes. Every evaluated extent and byte length MUST
be positive; an offset MAY be zero.

### 3.4 Blob-backed artifacts

The enclosing bundle uses top-level `blobs`, whose values are strict Blob
objects. A Blob is exactly `{byte_length, sha256, location}`. `byte_length`
is uint64, `sha256` is the lowercase SHA-256 of the complete decoded bytes,
and `location` is one BlobLocation. Program value artifacts use `blob`, never
a buffer artifact field:

```json
{
  "blobs": {
    "constants": {
      "byte_length": 16,
      "sha256": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb",
      "location": {"tag": "external", "uri": "artifacts/constants.bin"}
    }
  },
  "artifacts": {
    "constant-0": {
      "tag": "value",
      "encoding": "value_abi",
      "blob": "constants",
      "offset": 0,
      "byte_length": 16,
      "sha256": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"
    }
  }
}
```

A BlobLocation is exactly one of:

- `{"tag":"inline","data":"..."}`;
- `{"tag":"external","uri":"..."}`.

Inline location is allowed only for non-code value payloads in an in-memory
non-persistent bundle. Every Blob referenced by a StageArtifact CodeModule in
a cooked bundle MUST use `external`. Code bytes are never base64-embedded in
the manifest.

For `inline`, strict base64 decoding MUST produce exactly the enclosing Blob
`byte_length`, and SHA-256 of those decoded bytes MUST equal the enclosing
Blob `sha256`. For `external`, the fetched object MUST have exactly that
length and hash before any artifact range is exposed.
An external URI is a normalized relative URI resolved beneath the bundle
root. Absolute, network, parent-traversing, and bundle-escaping URIs are
invalid.

A value artifact is exactly
`{tag:"value", encoding:"value_abi", blob, offset, byte_length, sha256}`.
Its checked half-open range `[offset, offset + byte_length)` MUST fit the
referenced Blob. Its `sha256` is over exactly the range bytes, independent of
the whole-Blob hash. Zero-length value artifacts are invalid. Offset,
alignment, range hash, encoding, and decoded ValueLayout size are validated
before materialization.

Runtime `Buffer` is not an independent logical Program type. Byte resources
are buffer Storage descriptors plus typed Values and views. In particular,
the removed compiler type `!vernon.buffer` has no manifest counterpart.

### 3.5 Executable artifact system

A cooked deployment bundle is exactly:

```json
{
  "compiler_contract_version": 13,
  "pipeline_version": 17,
  "artifact_system": {
    "target": {"kind": "vulkan", "options": {}},
    "blobs": {},
    "artifacts": {}
  },
  "variants": [
    {"key": [], "runtime_requirements": {}, "stage_bindings": {}, "program": {}}
  ],
  "content_hash": "0000000000000000000000000000000000000000000000000000000000000000"
}
```

Its persistent file layout is:

```text
bundle/
  program.pipeline.json
  artifacts/
    <blob-sha256>.o
    <blob-sha256>.obj
    <blob-sha256>.ptx
    <blob-sha256>.spv
    <blob-sha256>.glsl
    <blob-sha256>.gles
    <blob-sha256>.metal
    <blob-sha256>.dxil
```

Only applicable files required by emitted variants are present; the suffix
list above is illustrative rather than a requirement to emit every format.
Blob external URIs point
from `program.pipeline.json` into this `artifacts/` directory and remain
normalized relative paths beneath the bundle root.

The object above shows container structure only, not a valid bundle: version
numbers are illustrative future values, the digest is placeholder text, and
the nested objects are incomplete. Required root members are exactly those
shown.
`content_hash` is lowercase SHA-256 over canonical JSON bytes of the complete
bundle excluding `content_hash`. A variant contains exactly sorted unique
non-empty feature strings in `key`, one `stage_bindings` object, and one
`runtime_requirements` object, and one Program in `program`; variant keys are unique and variants are ordered by
canonical key bytes. `stage_bindings` maps every Program logical stage ID
exactly once to a StageArtifact ID in the one root `artifact_system`.
One cooked bundle has one target. A different backend is a different bundle,
not a fat-bundle branch or runtime fallback.

The cooker emits code and StageArtifacts for every explicitly requested
variant. Runtime selects exactly one variant from its canonical feature key
before artifact resolution and validates only that variant's aggregate Runtime
requirements. It then resolves, authenticates, loads, or JITs only
StageArtifacts, CodeModules, and external Blob ranges reachable from that
variant's `stage_bindings` and ConstantOrigins. It MUST NOT validate or load
code belonging only to an unselected variant. Selection or selected-variant
capability failure is reported before code I/O; Runtime never falls back to
another variant.

A deployable Program is resolved with exactly one ArtifactSystem:

```json
{
  "target": {"kind": "vulkan", "options": {}},
  "blobs": {
    "code": {
      "byte_length": 1024,
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "location": {
        "tag": "external",
        "uri": "artifacts/0000000000000000000000000000000000000000000000000000000000000000.spv"
      }
    }
  },
  "artifacts": {}
}
```

Program resolution never searches a filesystem by stage name, guesses an
extension, invokes another compiler to create a missing artifact, or selects
a different target.

The executable lookup path is exactly:

```text
Node.stage
  -> Program.stages[stage] portable contract
  -> variant.stage_bindings[stage]
  -> ArtifactSystem.artifacts[StageArtifactID]
  -> StageArtifact.modules[*]
  -> ArtifactSystem.blobs[module.blob]
  -> Blob.location + module [offset, offset + byte_length)
  -> module.entry_point
```

`target` is exactly `{kind, options}`. `kind` is `cpu`, `cuda`, `vulkan`,
`opengl`, `opengles`, `metal`, or `directx`. Options are strict:

- CPU requires non-empty normalized `triple` and optionally has non-empty
  `processor` and sorted unique `features`;
- OpenGL and OpenGL ES optionally have integer `version`;
- Metal requires `platform`, `macos` or `ios`;
- DirectX requires integer `shader_model`;
- CUDA and Vulkan options are empty.

Every StageArtifact and variant `runtime_requirements` contains required
`backend` equal to ArtifactSystem target kind and sorted unique `features`,
followed by the exact backend fields:

- CPU: non-empty `target_triple` and `object_format` (`elf`, `macho`, `coff`,
  or `wasm`);
- CUDA: `ptx_version`, `address_size`, and
  `minimum_compute_capability`;
- Vulkan: `api_version` and `spirv_version`;
- OpenGL/OpenGL ES: integer `glsl_version`, `api_version`, and `profile`;
- Metal: `apple_platform`, `msl_version`, and `minimum_os_version`;
- DirectX: `api_version`, `minimum_feature_level`, `shader_model`, and
  `root_signature_version`.

Every version or capability pair is exactly a two-element uint32 array.
`address_size` is 32 or 64. `profile` is `core`, `compatibility`, or `es` and
must agree with target kind. Backend-specific values and feature names are
defined by the same pipeline contract pair; unknown or extra members are
errors. Requirements are minimum execution requirements, not compilation
options, and Runtime validates them before loading any code.
Workgroup size is StageArtifact compute reflection authority and is not
duplicated in RuntimeRequirements; Runtime compares each reflected size and
limit directly with device capabilities.

The cooker derives each StageArtifact requirement object from that artifact's
modules and reflection. For a variant, it then derives one aggregate from only
the StageArtifacts reachable through that variant's `stage_bindings`.
`features` is the sorted union of their
`reflection.required_features`; minimum version,
capability, feature-level, shader-model, and OS pairs are maxima under the
backend's version ordering. Target triple, object format,
address size, API profile, and Apple platform must agree exactly across all
selected stages. A producer-supplied StageArtifact requirement or variant
aggregate that differs from recomputation is invalid. Adding an unrelated
variant does not change existing StageArtifact IDs or another variant's
requirements.

`artifacts` maps Artifact IDs to a strict union of the value artifact from
section 3.4 and StageArtifact. A value Artifact ID is the lowercase SHA-256 of
canonical
`{compiler_contract_version, pipeline_version, tag, encoding, byte_length,
sha256}` using the enclosing selected pair. A StageArtifact ID is the
lowercase SHA-256 of the canonical identity object containing:

- the selected compiler and pipeline contract versions;
- this ArtifactSystem's complete `target`;
- StageArtifact `tag`, `operation`, `contract_hash`,
  `runtime_requirements`, and `reflection`;
- for each module, `role`, `format`, `entry_point`, `byte_length`, and
  range `sha256`.

Artifact identity excludes module/value `blob` and `offset`, BlobLocation, and
whole-Blob hash. Repacking identical authenticated ranges therefore preserves
artifact identity, while code bytes, code selection, entry points, reflection,
target, or requirements cannot change without changing it. Blob IDs are local
map keys; Blob content is independently authenticated by its own hash.

The following StageArtifact is a structural schema example, not a resolvable
fixture because its Blob and digest are placeholders. A StageArtifact is:

```json
{
  "tag": "stage",
  "operation": "compute",
  "contract_hash": "0000000000000000000000000000000000000000000000000000000000000000",
  "runtime_requirements": {
    "backend": "vulkan",
    "features": [],
    "api_version": [1, 1],
    "spirv_version": [1, 3]
  },
  "modules": [
    {
      "role": "compute",
      "format": "spirv",
      "entry_point": "main",
      "blob": "code",
      "offset": 0,
      "byte_length": 1024,
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
    }
  ],
  "reflection": {
    "required_features": [],
    "endpoints": [],
    "compute": {
      "workgroup_size": [8, 8, 1],
      "subgroup": null,
      "capabilities": ["direct_dispatch"]
    }
  }
}
```

Required members are `tag`, `operation`, `contract_hash`,
`runtime_requirements`, `modules`, and `reflection`.
`operation` is `compute` or `graphics`. Each CodeModule contains exactly
`role`, `format`, `entry_point`, `blob`, `offset`, `byte_length`, and
`sha256`. Its checked non-empty range and range hash follow value-artifact
rules. Entry points are non-empty exact exported symbol or shader entry names;
Runtime never derives them from Program or diagnostic names.
The all-zero digest in the structural example is a placeholder, not a valid
digest unless it is the actual hash of the selected bytes.
`contract_hash` is SHA-256 of canonical `{operation, reflection}` and must
equal the portable StageContract selected by the Program.

Allowed target, format, and role combinations are:

- CPU: `relocatable_object` with role `compute`; the target object format
  determines `.o` or `.obj`, and `entry_point` is the exported ABI symbol
  resolved by static registration or the target linker;
- CUDA: `ptx` with role `compute`;
- Vulkan: `spirv` with role `compute`, `vertex`, or `fragment`;
- OpenGL: `glsl` with role `compute`, `vertex`, or `fragment`;
- OpenGL ES: `gles` with role `compute`, `vertex`, or `fragment`;
- Metal: `msl` with role `compute`, `vertex`, or `fragment`;
- DirectX: `dxil` with role `compute`, `vertex`, or `fragment`.

A compute artifact has exactly one compute module. A graphics artifact has
exactly one vertex and one fragment module in that order and no compute
module. Multiple module records may select different entry points in the same
Blob range. Source formats such as GLSL, GLES, MSL, and PTX are final cooked
inputs to their target Runtime compiler/JIT; Runtime may perform only the
target-defined load/JIT step and MUST NOT cross-compile them to another
backend. HLSL, LLVM IR, native shared libraries, host source, and fallback
modules are not deployment formats in this contract.

Before stage exposure, Runtime validates format identity independently of file
suffix: relocatable-object container and target, SPIR-V magic/version and
entry model, UTF-8 for PTX/GLSL/GLES/MSL, and DXIL container/version. It proves
that every declared entry point exists with the declared module role, either
from authenticated binary metadata or from the target load/JIT result.
Filename suffix is diagnostic only. A text parse, binary parse, link, or
entry-role mismatch is an artifact validation failure, not permission to try
another module.

Reflection is exactly `{required_features, endpoints, compute}` for compute or
`{required_features, endpoints, graphics}` for graphics. Endpoint and operation reflection are
defined in section 11.1. Reflection describes the complete StageArtifact,
including all graphics modules; it is not duplicated per module.
`required_features` is a sorted unique array from the contract pair's closed
feature vocabulary and is the sole source of ArtifactSystem feature
aggregation. StageArtifact `modules` array order is `[compute]` or
`[vertex, fragment]`; fixed sorting ordinals remain compute = 0, vertex = 1,
fragment = 2. The identity module sequence uses this same order. Artifact maps
are ordered by Artifact ID and Blob maps by Blob ID.
Every ArtifactSystem artifact is referenced by at least one variant
`stage_bindings` entry or ConstantOrigin, and every Blob is referenced by at
least one artifact range. Unreachable artifacts and Blobs are non-canonical.

All StageArtifacts selected by one variant belong to its one ArtifactSystem
target. Multi-target deployment is represented by separate cooked bundles.
A loader MUST NOT choose modules from another bundle or combine artifacts
from different targets.

## 4. Program object

`variant.program` is exactly:

```json
{
  "stages": {},
  "parameters": [],
  "storages": [],
  "values": [],
  "shape_symbols": [],
  "shape_constraints": [],
  "alias_preconditions": [],
  "graphs": [],
  "signature": {
    "inputs": [],
    "outputs": [],
    "cotangents": [],
    "gradients": []
  }
}
```

Required members are `stages`, `parameters`, `storages`, `values`,
`shape_symbols`, `shape_constraints`, `alias_preconditions`, `graphs`, and
`signature`. `residual_contract` is required exactly when a backward graph
exists and forbidden otherwise, including when its captures and shape-symbol
arrays would both be empty.

`stages` is an object from non-empty logical stage IDs to strict
StageContracts:

```json
{"operation":"compute","contract_hash":"0000000000000000000000000000000000000000000000000000000000000000"}
```

StageContract contains exactly `operation` and `contract_hash`; the hash is
SHA-256 of canonical `{operation, reflection}` for the portable StageArtifact
reflection required by this logical stage. Stage keys are sorted in canonical
bytes. Every StageContract is referenced by at least one node and every node
stage exists. A node operation tag MUST equal StageContract operation.
The enclosing variant `stage_bindings` maps each logical stage ID to one
StageArtifact with equal operation and contract hash. No node selects an
Artifact ID, target, module, or entry point directly.

Every Program has exactly one forward graph and at most one backward graph.
The backward graph is present exactly for a differentiated Program. The
empty objects and arrays in the structural example above show required
members, not a complete executable Program.

Examples use all-zero StageContract hashes as visible placeholders. A
serialized Program uses the recomputed lowercase contract digest.

## 5. Parameters and instance binding

A Parameter is:

```json
{"id": 0, "path": "scale", "value": 2}
```

`id`, `path`, and `value` are required. IDs are contiguous. `path` is a
unique non-empty public ABI identity string. It is not a diagnostic label.
`value` names exactly one Value with
`ParameterOrigin` for this parameter.

Parameters are supplied when a Program instance is created. They consume no
invocation argument slot and have no serialized or implicit default. Instance
creation fails if any parameter is absent, duplicated, or incompatible with
its Value, Storage, shape, or layout declaration. A parameter is immutable as
a binding even when it supplies mutable borrowed storage.

Every graph that uses a parameter lists a GraphInput with
`{"tag":"parameter","value":V,"parameter":P}`. The IDs MUST agree with the
Parameter and ParameterOrigin. Parameter Values are entry-available.

## 6. Storage

Storage is the sole descriptor authority:

```json
{
  "id": 0,
  "name": "input",
  "initial_value": 0,
  "ownership": "borrowed",
  "lifetime": "invocation",
  "mutability": "read_only",
  "descriptor": {
    "tag": "buffer",
    "byte_length": 1024,
    "alignment": 16,
    "memory": "device",
    "usage": ["storage"]
  }
}
```

Required common members are:

- `id`: contiguous uint32;
- optional `name`: diagnostic string;
- `initial_value`: REQUIRED Value ID for the unique initial root version;
- `ownership`: `owned` or `borrowed`;
- `lifetime`: `invocation`, `instance`, or `pullback`;
- `mutability`: `read_only` or `mutable`;
- `descriptor`: one descriptor variant.

The descriptor is exactly one of:

```json
{
  "tag": "buffer",
  "byte_length": 4096,
  "alignment": 16,
  "memory": "device",
  "usage": ["index", "storage"]
}
```

```json
{
  "tag": "image",
  "dimension": "2d",
  "extent": [64, 32, 1],
  "format": "rgba16_float",
  "sample_count": 1,
  "mip_levels": 1,
  "array_layers": 1,
  "aspects": ["color"],
  "usage": ["color_attachment", "sampled"]
}
```

```json
{
  "tag": "opaque",
  "contract": "vendor.example.handle",
  "contract_hash": "0000000000000000000000000000000000000000000000000000000000000000",
  "usage": ["stage_binding"]
}
```

Buffer `byte_length` is a non-negative integer. A positive value is a static
provider constraint or owned allocation size. `0` is legal only on borrowed
Storage: TensorView `vd.dyn` already records the dynamic rank/extent on the
Value `.shape` as `-1`, and bind takes the concrete byte length from the
provider buffer. Do not encode buffer size as a ShapeExpr, ShapeSymbol, or
control component, and do not bake a Python launch shape into the descriptor.
`alignment` is a positive power of two. `memory` and `usage` use enums defined
in Appendix B.

Image `dimension` is REQUIRED and is `1d`, `2d`, `3d`, or `cube`. `extent`
has exactly three non-negative components. `0` on an axis is legal only on
borrowed Storage and means that axis is TensorView dyn (same as Value `.shape`
`-1`). Canonical static shapes are `1d = [W,1,1]`, `2d = [W,H,1]`,
`cube = [W,H,1]` with `W = H`, and `3d = [W,H,D]`.
Format, sample count, mip levels, array layers, aspects, and usage are exact
creation or provider requirements. Opaque contract identity is exact; opaque
storage has no byte range, image subresource, or ValueLayout.

For owned Storage, the descriptor is exact runtime creation authority. The
runtime creates it at the start of its declared lifetime and MUST NOT replace
it with a merely compatible resource. For borrowed Storage, the descriptor is
a provider constraint checked before first use; the runtime MUST NOT recreate,
resize, reformat, migrate, or assume ownership of the provider resource.

`invocation` lifetime is fresh per call, `instance` lifetime belongs to one
Program instance, and `pullback` lifetime belongs to PullbackState. Borrowed
Storage is legal only when its provider lifetime covers the declared lifetime.

Each Storage has exactly one root Value equal to `initial_value`; no earlier
version of that Storage exists. That Value carries the Storage ID and is not a
view. An ArgumentOrigin, ParameterOrigin, or ConstantOrigin initial root is
readable at graph entry and appears in the matching GraphInput. An
AllocationOrigin initial root is entry-available allocation identity but has
no readable contents until initialized. A NodeResultOrigin initial root is
legal only when exactly one `initialize` access produces that same Value.
When an AllocationOrigin is the initial root, the first initializing access
is ordered after it and produces a distinct NodeResult version. Every later
root version is produced by exactly one write or attachment access and has
one predecessor. `read` never creates a root version.

No resource facts may be duplicated in ValueLayout. Values identify logical
contents and views; Storage defines physical resource requirements.

## 7. Values, layouts, and origins

### 7.1 Value

A Value is a strict object containing:

- required `id`, contiguous and equal to array index;
- optional diagnostic `name`;
- required canonical `type` using Appendix B grammar;
- optional `shape`, required for shaped logical values;
- required `origin`;
- optional `storage`, required for a storage-backed root or view;
- optional `view`, required exactly for a `ViewOrigin`;
- optional `value_layout`, allowed only for canonical value elements or the
  whole by-value ABI.

There is no external token or effect-token Value in this contract.

### 7.2 ValueLayout

ValueLayout is:

```json
{
  "scope": "element",
  "layout_hash": "95f54cae607cf7751c0ec4327f86b0982056823c49534e9d8dddd66fc98c5f07",
  "byte_size": 4,
  "alignment": 4,
  "leaves": [
    {
      "path": [],
      "dtype": "f32",
      "byte_offset": 0,
      "scalar_count": 1,
      "shape": []
    }
  ]
}
```

`scope` is `element` or `value`. The remaining fields are required.
`byte_size`, leaf offsets, scalar counts, and leaf shapes may use ShapeExpr
where dynamic. Leaf paths are arrays of non-empty field strings or unsigned
aggregate indices. Leaves are sorted by canonical path and do not overlap.

ValueLayout describes only canonical logical value bytes. It has no image,
opaque, resource, or token variant.

`layout_hash` is lowercase SHA-256 over canonical JSON bytes of the strict
object `{scope, byte_size, alignment, leaves}`, excluding `layout_hash`
itself. Expression objects and arrays use section 3 canonicalization. The
Appendix B defines canonical `value_abi` dtype, aggregate path, and shape
spellings. Payloads are little-endian; integers use two's complement, floats
use IEEE 754 binary interchange encoding, and boolean false/true are one byte
`0`/`1`. ValueLayout supplies every aggregate offset, size, and alignment.
Artifact reflection and Program Values MUST use these same rules.

### 7.3 Origin tagged union

Origin is exactly one of:

```json
{"tag": "argument", "graph": "forward", "slot": 0}
{"tag": "parameter", "parameter": 0}
{"tag": "constant", "payload": {"tag": "inline", "encoding": "value_abi", "data": "AAAAAA==", "byte_length": 4, "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"}}
{"tag": "constant", "payload": {"tag": "artifact", "artifact": "constant-0"}}
{"tag": "allocation", "graph": "forward"}
{"tag": "node_result", "graph": "forward", "node": 0}
{"tag": "view", "source": 3}
```

Argument slots are contiguous within each invocation boundary and agree with
GraphInput and Signature order. ParameterOrigin agrees with `parameters`.
AllocationOrigin is allowed only for owned Storage.

NodeResultOrigin names the unique producing node and is valid for ordinary
dataflow and resource versions. It is never valid as a control source.

ConstantOrigin payload bytes MUST match ValueLayout. ConstantOrigin MUST NOT
directly define a TensorView. A constant tensor view is represented by:

1. a read-only owned buffer Storage;
2. a constant-backed root Value containing its initializer payload; and
3. a second Value with ViewOrigin and a ViewDescriptor.

The backing Storage descriptor remains the resource authority. Runtime
initializes that Storage from the payload at the start of its declared
lifetime without adding a graph operation.

### 7.4 ViewDescriptor

A buffer view descriptor is:

```json
{
  "tag": "buffer_view",
  "byte_offset": 0,
  "extents": [{"control": {"parameter": 0}}, 4],
  "byte_strides": [16, 4]
}
```

An image view descriptor is:

```json
{
  "tag": "image_view",
  "aspects": ["color"],
  "base_mip": 0,
  "mip_count": 1,
  "base_layer": 0,
  "layer_count": 1
}
```

All descriptor components are static integers or ControlValueRef components.
A view refers to the same Storage as its source and is flattened to the
Storage base; views of views are not serialized. Buffer views MUST be in
bounds and writable views MUST be injective. Image views MUST select valid
declared subresources.

## 8. ControlValueRef

A ControlValueRef is exactly one of:

```json
{"argument": 0}
{"parameter": 0}
{"constant": 6}
{"capture": 9}
```

- `argument` names a Value with ArgumentOrigin available at this graph entry;
- `parameter` names a Parameter ID and thereby its ParameterOrigin Value;
- `constant` names a scalar or fixed aggregate Value with ConstantOrigin;
- `capture` names a forward Value ID listed in both the backward graph
  `captures` and `residual_contract.captures`. PullbackState supplies its
  retained concrete value at backward entry.

A control component is a static integer or `{"control": ControlValueRef}`.
Where a dimension is needed, use
`{"dimension":{"control":ControlValueRef,"axis":A}}`.

ControlValueRef is the only dynamic source for:

- owned allocation byte lengths and image extents;
- ShapeSymbol sources and shape-dependent layouts;
- every ViewDescriptor component;
- compute workgroup counts;
- attachment area, layer, clear, resolve, viewport, and scissor metadata;
- numeric render-state metadata;
- vertex, index, and instance draw metadata.

The referenced value MUST be available before any node starts. A value with
NodeResultOrigin is forbidden even when it dominates the use. Runtime MUST
NOT obtain control data by device readback, mapping, copying, transfer,
implicit synchronization, or speculative execution.

Non-integer render-state constants, including clear colors, clear depth,
viewport depth values, depth bias, line width, and blend constants, MUST be
Constant Program Values referenced through ControlValueRef. Such values are
never raw JSON floating-point fields in an operation.

## 9. Shape symbols and alias preconditions

TensorView `vd.dyn` is a type-level `-1` on Value `.shape`. It does not create
a ShapeSymbol. Phase-one compute keeps `shape_symbols`, `shape_constraints`,
and `alias_preconditions` empty.

A ShapeSymbol is:

```json
{
  "id": 0,
  "name": "width",
  "source": {"tag": "scalar", "control": {"argument": 0}},
  "min": 1,
  "max": 16384
}
```

Source is either:

- `{"tag":"scalar","control":R}`; or
- `{"tag":"dimension","control":R,"axis":A}`.

The source MUST satisfy ControlValueRef rules. Constraints are:

```json
{"tag": "equal", "lhs": {"symbol": 0}, "rhs": {"constant": 1024}}
{"tag": "less_equal", "lhs": {"symbol": 0}, "rhs": {"symbol": 1}}
{"tag": "divisible", "value": {"symbol": 0}, "divisor": 4}
```

All entry constraints are checked before allocation or external effects.
Backward symbols needed from forward are listed in `residual_contract` and
bound from retained capture witnesses.

Alias preconditions are strict records:

```json
{"tag": "disjoint", "lhs": 0, "rhs": 1}
{"tag": "may_alias_read_only", "lhs": 0, "rhs": 2}
```

They refer to borrowed Storage IDs. `may_alias_read_only` is legal only when
both Storage declarations are `read_only`. Writable aliasing is expressible
only through views carrying the same Storage ID. Two distinct Storage IDs
whose physical ranges overlap are invalid if either is mutable, regardless of
precondition. Failure is an entry validation error, not a request for copying.

## 10. Graph boundaries

A Graph is:

```json
{
  "name": "forward",
  "direction": "forward",
  "inputs": [],
  "captures": [],
  "outputs": [],
  "nodes": []
}
```

Name and direction are `forward` or `backward` and agree. Forward precedes
backward. `captures` is empty for forward. Backward captures are
`{"value":V}` records whose Value IDs refer directly to forward Values and
equal `residual_contract.captures[*].value`. Those original Value IDs become
entry-available in backward; no new Origin or GraphInput is serialized.

GraphInput is exactly one of:

```json
{"tag": "user_input", "value": 0, "slot": 0}
{"tag": "cotangent", "value": 8, "slot": 0, "primal": 4}
{"tag": "parameter", "value": 2, "parameter": 0}
{"tag": "constant", "value": 6}
{"tag": "storage", "value": 3, "storage": 2}
{"tag": "allocation", "value": 5, "storage": 4}
```

Only `user_input` and `cotangent` consume invocation slots. Every independently
available argument, parameter, constant, borrowed/instance state, allocation,
or initial storage value used by the graph appears once. Backward captures
are available from `captures`, not duplicated in `inputs`.

GraphOutput is exactly one of:

```json
{"tag": "user_output", "value": 4, "disposition": "borrow"}
{"tag": "state_update", "value": 7, "storage": 2}
{"tag": "gradient", "value": 10, "primal": 0}
```

`user_output` disposition is required for storage-backed values and forbidden
for by-value results. It is `transfer` or `borrow`. Transfer is solely an
ownership disposition, not an operation node. It is legal only for owned
Storage and changes ownership only after the entire invocation commits
successfully. On failure, ownership remains with the Program and no output is
published. Borrowed output lifetime MUST cover the caller-visible borrow.

All public resources are ordinary `user_output` values and ordinary Signature
outputs. There is no separate resource-output boundary class.

Nodes are in canonical topological order. Edges are derived from operands,
results, accesses, and attachment versions. Node IDs are graph-local,
contiguous, and equal array index.

## 11. Node and operation union

A Node consists of public fields plus one strict operation:

```json
{
  "id": 0,
  "name": "optional diagnostic name",
  "stage": "main",
  "operands": [0],
  "results": [1],
  "bindings": [],
  "accesses": [],
  "operation": {"tag": "compute", "workgroups": [1, 1, 1]}
}
```

`name` is optional; all other fields are required. `stage` selects
`Program.stages`. Operands and results are logical dataflow. Bindings map
stage endpoints. Accesses declare resource hazards and version transitions.

Operands are the exact sorted set of Program Values consumed by the node:

- every Value selected by a value binding;
- every access `value` or `before` root and every access `view`;
- the Storage initial root for an initialize access when that root has
  AllocationOrigin;
- the Parameter Value, constant Value, captured Value, or argument Value
  selected by every operation ControlValueRef;

Entry availability does not permit omission from `operands`. Access `after`
Values are not operands. Results are the exact sorted set of Values produced
by value-result bindings and access `after` members. A binding, access, view,
or operation control reference outside this closure is
`PROGRAM_OPERAND_CLOSURE`; an extra operand or result is the same error.
Attachment and index accesses remain outside EndpointBinding but remain in
operand/result closure.

An EndpointBinding is:

```json
{"module": "compute", "interface": "argument", "index": 0, "tag": "value", "value": 2}
{"module": "compute", "interface": "argument", "index": 1, "tag": "resource", "access": 0}
{"module": "compute", "interface": "result", "index": 0, "tag": "value", "value": 3}
```

`module`, `interface`, `index`, and `tag` are required. `module` selects one
CodeModule role in the StageArtifact. `interface` is `argument` or `result`;
endpoint indices are local to that module and interface. An optional `leaf`
selects a ValueLayout leaf. System, attachment, index, and control metadata
endpoints are not Program bindings.

### 11.1 Artifact reflection

Artifact reflection is selected by the same future compiler/pipeline contract
pair as Program; it has no independently selectable Program ABI. The
root contains required `required_features` and `endpoints` plus exactly one of
`compute` or `graphics`, matching StageArtifact operation.
The StageArtifact `reflection.endpoints` array contains strict endpoint records
ordered by the fixed module-role ordinal from section 3.2, then interface
ordinal and index. An endpoint is exactly one of:

- `value`: required `tag`, `module`, `interface`, `index`, canonical `type`,
  `layout_hash`, `transport`, `access`, and `abi`;
- `resource`: required `tag`, `module`, `interface`, `index`, resource `role`,
  canonical `type`, `layout`, `address_space`, `transport`, `access`, and
  `abi`;
- `system`: required `tag`, `module`, `interface`, `index`, and system
  `semantic` and `abi`; it forbids a Program EndpointBinding.

Endpoint `module` must name an actual CodeModule role. The tuple
`(module, interface, index)` is unique. A resource used by both vertex and
fragment modules has two reflected endpoints and two Program bindings, which
may select the same Value or ResourceAccess.

`transport` is `by_value`, `resource_handle`, or `device_address`; `access` is
`read`, `write`, or `read_write`. Reflection
layout includes every physical scalar/vector width, stride, alignment, and
resource-view requirement needed to validate a binding. Unknown endpoint
members, variants, roles, transports, or address spaces are rejected.
Canonical types, resource layouts, transports, address spaces, system
semantics, and portable ABI carriers are defined in Appendix B.

Compute reflection contains exactly `workgroup_size`, `subgroup`, and
`capabilities`. Workgroup size is a three-dimensional positive uint32 array,
and `subgroup` is null or the strict portable subgroup requirement in
Appendix B. Capabilities are a sorted unique array that MUST include
`direct_dispatch` and MUST NOT require indirect dispatch.
Device dispatch-count and invocation limits are Runtime capabilities, not
serialized StageContract fields.

Graphics reflection contains exactly `topology`, `vertex_inputs`,
`fragment_outputs`, `linkage`, `attachment_constraints`, `index_formats`, and
`capabilities`. `linkage` contains exactly `vertex_outputs` and
`fragment_inputs`; each record contains location or builtin identity,
canonical type, and interpolation. The two arrays must match exactly after
builtins that are consumed by fixed-function graphics are removed.
Location-bearing arrays are ordered by location; index formats and
capabilities are sorted unique arrays. These records declare vertex formats,
fragment types, attachment format/sample/aspect constraints, and required
render or dynamic state. Resource endpoint roles are `uniform`, `storage`,
`sampled`, `sampler`, or `vertex`. System endpoints include builtins such as
position, vertex/instance index, fragment coordinates, sample state, target
extent, and immutable default sampler; they are satisfied by the graphics
runtime, not Program bindings.
The strict graphics reflection record shapes and enum spellings are defined
in Appendix B.

Resolve rejects disagreement between reflection and Program rather than
rewriting either authority.

### 11.2 Binding and physical-value resolution

For a `value` binding without `leaf`, reflected canonical type and
`layout_hash` MUST equal the whole Program Value ABI and transport/access MUST
match the interface. With `leaf = L`, `L` must exist; reflection type, shape,
offset/stride requirements, and transport MUST equal that exact leaf.
Selecting multiple leaves is represented by multiple endpoints; duplicate
binding occurrence never implies leaf identity.

For a `resource` binding, `access` names exactly one ResourceAccess compatible
with reflected role, type, layout, address space, transport, and access mode.
The physical Value selected for binding is:

- `read`: `view` when present, otherwise `value`;
- `initialize`: `view` when present, otherwise `after`;
- `write`: `view` when present, otherwise `before`; `after` is only the new
  semantic SSA contents and is never a second physical argument;
- `attachment`: selected only by the GraphicsOperation attachment record and
  its required `view`, never by EndpointBinding.

The selected view contributes range/subresource and descriptor; its source
root contributes the exact contents version. A binding MUST NOT select a
different version, infer a view, or substitute the `after` Value as a second
allocation.

A ResourceAccess is exactly one of:

```json
{"tag": "read", "storage": 0, "value": 0, "view": 1}
{"tag": "initialize", "storage": 1, "after": 3, "view": 2}
{"tag": "write", "storage": 1, "before": 3, "after": 4, "view": 2, "access": "write"}
{"tag": "write", "storage": 1, "before": 4, "after": 5, "view": 2, "access": "read_write"}
{"tag": "attachment", "storage": 2, "before": 6, "after": 7, "view": 8}
```

`view` is required when the endpoint operates on a subrange or image
subresource. Read-only Storage cannot be initialized or written after its
constant initialization. Every `after` is a node result with the same Storage
ID; every predecessor is available. Access records and reflected stage access
must agree exactly.

Accesses are sorted lexicographically by `(storage, tag_ordinal, predecessor,
after)`, where ordinals are `read = 0`, `initialize = 1`, `write = 2`, and
`attachment = 3`; `predecessor` is `read.value`,
`Storage.initial_value`, `write.before`, or `attachment.before`
respectively; absent `after` is uint64 maximum. This key is total even though
different variants have different member names.

### 11.3 Derived dependency graph

Node-to-node dependency edges are not serialized. A `dependencies`,
`depends_on`, event, fence, token, or equivalent ordering member is invalid.
Resolve derives one exact graph from Value SSA and ResourceAccess records by
the following algorithm.

For one graph, define:

- `producer(V)` as the node named by a NodeResultOrigin for root Value `V`, or
  graph entry for ArgumentOrigin, ParameterOrigin, ConstantOrigin,
  AllocationOrigin, and a backward capture;
- `root(V)` as `V` itself for a root Value and the flattened ViewOrigin source
  for a view;
- `range(V)` as the complete Storage for a root, the checked byte interval for
  a buffer view, the selected aspects/mips/layers for an image view, and the
  complete indivisible resource for opaque Storage;
- `selected_range(A)` as `range(A.view)` when an access has a view and the
  complete range of its named root version otherwise;
- `overlap(A,B)` as byte-interval intersection for buffer ranges,
  aspect/mip/layer intersection for image ranges, and true for two opaque
  ranges of the same Storage. Distinct Storage IDs are separate alias domains
  except for declared `may_alias_read_only`; mutable physical overlap between
  distinct IDs is invalid at invocation entry.

Storages joined by `may_alias_read_only` may share physical ranges but all
their accesses are reads, so they form a shared read-only alias domain and
produce no hazard edge. Any write through either ID is already invalid under
section 9; ordering cannot legalize it.

Resolve first creates producer edges. For every node `N`, every Value in
`N.operands`, every `read.value`, `write.before`, and `attachment.before`, and
the root selected by every access view is a use. If `producer(root(V))` is a
node `P` distinct from `N`, add `P -> N`. A node result may name only its
declared producing node and that producer must list it in `results`.

Resolve then creates resource anti-dependencies. For each Storage version
`V`, let `readers(V)` contain:

- every read access whose selected root is `V`;
- every `write` access with `access = read_write` and `before = V`;
- every attachment access with `before = V` whose attachment load operation
  is `load`.

Attachment `clear` and `discard` do not add a reader; their `before` is
version identity and chain lineage only. A resolve destination is its own
attachment access and participates independently in closure, version-chain,
range-overlap, and edge derivation.

Let `successor(V)` be the unique node, if any, whose write or attachment
access has `before = V`. For an AllocationOrigin initial root, its first
initialize access is also its successor even though initialize has no
`before`. For each reader `R` of `V`, add `R -> successor(V)` when `R` is not
the successor itself and their selected ranges overlap. Allocation roots are
unreadable, so a conforming Program has no such reader before initialization;
the successor rule remains explicit for complete chain construction. Unknown
or dynamically controlled overlap is conservatively true. This is the
write-after-read edge required when a backend reuses physical storage for
semantic SSA versions.

Read-after-write and write-after-write edges are already producer edges from
the produced predecessor version. Initialization has no readable predecessor:
an AllocationOrigin initial root is allocation identity only, and an
initialize access produces its first readable successor. The special
NodeResultOrigin initial root form is produced by its one initialize access.

For each Storage, non-view root Values form exactly one non-branching version
chain:

1. the chain starts at `Storage.initial_value`;
2. each later root is the `after` of exactly one initialize, write, or
   attachment access;
3. each readable version has at most one write/attachment successor;
4. an initialize access occurs exactly once when required by, and only as
   permitted for, the initial-root forms in section 6;
5. every access names a version in this chain and every access view resolves
   to the same Storage;
6. no `after` Value is also an `after` of another access or a result of
   another node.

A forked version chain is invalid even when views are disjoint. Parallel
disjoint writes are represented in one node or as consecutive versions;
physical scheduling may still overlap backend work only when it preserves the
declared whole-Storage version semantics.

After all producer and anti-dependency edges are added, Resolve rejects a
self-edge or cycle. Serialized node order is valid exactly when every edge
`A -> B` has `A.id < B.id`; independent nodes retain producer-declared order.
Runtime may choose any schedule whose happens-before relation contains every
derived edge. It may add target synchronization but MUST NOT remove an edge or
invent a semantic dependency. The derived edge set, not array adjacency, is
the Program execution DAG.

### 11.4 ComputeOperation

ComputeOperation is:

```json
{"tag": "compute", "workgroups": [8, {"control": {"parameter": 0}}, 1]}
```

`workgroups` contains exactly three positive control components and specifies
direct dispatch counts. Workgroup size is artifact reflection. Zero counts,
indirect argument storage, backend command payloads, and runtime-computed
counts are invalid.

### 11.5 GraphicsOperation

GraphicsOperation directly contains `attachments`, `state`, and `draw`:

```json
{
  "tag": "graphics",
  "attachments": {
    "colors": [],
    "depth_stencil": null,
    "render_area": {"x": 0, "y": 0, "width": 640, "height": 480},
    "layer_count": 1
  },
  "state": {
    "raster": {
      "front_face": "counter_clockwise",
      "cull_mode": "none",
      "fill_mode": "fill"
    },
    "depth_stencil": {
      "depth_test": false,
      "depth_write": false,
      "depth_compare": "always",
      "stencil_test": false
    },
    "multisample": {
      "sample_mask": 4294967295,
      "alpha_to_coverage": false
    },
    "blend": [],
    "viewport": null,
    "scissor": null
  },
  "draw": {"tag": "direct", "vertex_count": 3, "instance_count": 1}
}
```

`attachments` contains required `colors`, `depth_stencil`, `render_area`, and
`layer_count`.

A ColorAttachment is exactly:

```json
{
  "location": 0,
  "access": 0,
  "load": {"tag": "clear", "value": {"constant": 5}},
  "store": "store",
  "resolve": {"access": 1, "mode": "average", "store": "store"}
}
```

`location` and `access` are required uint32. `load` is exactly
`{"tag":"load"}`, `{"tag":"discard"}`, or
`{"tag":"clear","value":ControlValueRef}`. `store` is `store` or `discard`.
`resolve` is optional and, when present, contains exactly `access`, `mode`,
and `store`; its access selects a compatible single-sample destination
attachment. Portable resolve mode is `average`; the active graphics target
contract admits no other resolve mode.
Resolve destination load is always implicit discard: its predecessor is
version lineage but its prior contents are not read. Source and destination
store policies are independent.

A DepthStencilAttachment contains exactly `access`, `read_only`,
`depth_load`, `stencil_load`, and `store`. The load members are AttachmentLoad
or `null` according to selected aspects. Writable use requires an attachment
access and `store`; read-only use requires a read access, null load/store
members, and disabled depth/stencil writes.

RenderArea contains exactly `x`, `y`, `width`, and `height`. Components are
non-negative integer control components; width and height are positive.
`layer_count` is a positive integer control component. All attachment views
have compatible extents, layer ranges, and sample counts, and the render area
fits each selected mip.

`state` contains required `raster`, `depth_stencil`, `multisample`, `blend`,
`viewport`, and `scissor`, plus optional `blend_constant`.

- Raster contains required `front_face`, `cull_mode`, and `fill_mode`, and
  optional `depth_bias`, `depth_bias_slope`, and `line_width` ControlValueRefs.
  Enums are front face `clockwise|counter_clockwise`, cull
  `none|front|back`, and fill `fill|line|point`.
- DepthStencil contains required `depth_test`, `depth_write`,
  `depth_compare`, and `stencil_test`. If stencil is enabled it also requires
  `front` and `back` StencilFace objects. A StencilFace contains `compare`,
  `fail`, `depth_fail`, `pass`, `read_mask`, `write_mask`, and `reference`;
  masks/reference are uint32 control components. `front` and `back` are
  forbidden when stencil is disabled.
- Multisample contains exactly uint32 control component `sample_mask` and
  boolean `alpha_to_coverage`.
- `blend` is ordered by location. Each entry contains `location`, `enabled`,
  and sorted `write_mask`; enabled entries additionally require `src_color`,
  `dst_color`, `color_op`, `src_alpha`, `dst_alpha`, and `alpha_op`.
  `blend_constant`, when needed by a factor, is a ControlValueRef to a
  four-component constant; it is otherwise forbidden. Disabled entries forbid
  factor and operation members.
- Viewport is `null` or exactly `{x,y,width,height,min_depth,max_depth}`;
  every member is an integer control component or a ControlValueRef,
  width/height are positive, and
  `0 <= min_depth <= max_depth <= 1`.
- Scissor is `null` or one RenderArea.

Compare enums are `never|less|equal|less_equal|greater|not_equal|
greater_equal|always`. Stencil operations are `keep|zero|replace|
increment_clamp|decrement_clamp|invert|increment_wrap|decrement_wrap`.
Blend operations are `add|subtract|reverse_subtract|min|max`. Blend factors
are `zero|one|src_color|one_minus_src_color|dst_color|
one_minus_dst_color|src_alpha|one_minus_src_alpha|dst_alpha|
one_minus_dst_alpha|constant_color|one_minus_constant_color|
constant_alpha|one_minus_constant_alpha`. Write-mask channels are
`r|g|b|a`. These spellings are authoritative manifest enums; the active
graphics target contract determines which declared combinations, formats,
sample counts, and dynamic-state capabilities are supported.

Draw is exactly:

```json
{"tag": "direct", "vertex_count": 3, "instance_count": 1, "first_vertex": 0, "first_instance": 0}
```

or:

```json
{"tag": "indexed", "index_access": 2, "index_count": 6, "instance_count": 1, "first_index": 0, "base_vertex": 0, "first_instance": 0}
```

Counts and first/base values are integer control components. Omitted optional
first values default to zero; omitted `instance_count` defaults to one.
`index_access` names a read access selecting a contiguous rank-one u16 or u32
typed view with buffer `index` usage. Vertex buffers remain ordinary resource
bindings. Primitive topology is artifact specialization.

For direct draw, `tag` and `vertex_count` are required; `instance_count`,
`first_vertex`, and `first_instance` are optional. For indexed draw, `tag`,
`index_access`, and `index_count` are required; `instance_count`,
`first_index`, `base_vertex`, and `first_instance` are optional.
`base_vertex` is a signed int32 literal or a ControlValueRef to an entry
signed integer; every other draw integer must fit uint32.

## 12. Signature and residual contract

Signature is:

```json
{
  "inputs": [{"path": "x", "value": 0}],
  "outputs": [{"path": "y", "value": 1, "disposition": "borrow"}],
  "cotangents": [],
  "gradients": []
}
```

Paths are non-empty public ABI strings and unique in each array. Signature
inputs correspond exactly to forward `user_input` boundaries. Outputs
correspond exactly to forward `user_output` boundaries, including resource
outputs and matching dispositions. Cotangents and gradients correspond
exactly to backward boundaries. Every cotangent and gradient entry contains
required `path`, `value`, and `primal`; `primal` names its paired forward
output or input respectively, and paths must match.

Parameters do not appear in Signature. They are bound through `parameters`.
State updates are not public values unless also listed as an explicit
user output.

ResidualContract is:

```json
{
  "captures": [
    {
      "value": 4,
      "replay": {
        "legal": true,
        "required_values": [2],
        "cost": 40
      }
    }
  ],
  "shape_symbols": [0]
}
```

Every capture contains required `value` and `replay`. Replay contains required
boolean `legal`, sorted unique forward `required_values`, and uint64 `cost`.
The backward graph capture array equals `residual_contract.captures[*].value`
exactly in order and content. Captured Values and concrete symbol witnesses
are retained in PullbackState and become backward-entry-available under their
original forward Value IDs.

When `legal` is true, `required_values` is the exact frontier of the unique
canonical backward slice from the capture producer. The slice is induced by
forward data/access predecessors, stops at that frontier, and executes in
canonical forward node order. It may contain compute nodes only and cannot
cross graphics, caller-visible mutation, state commit, ownership transfer, or
another external effect. Every `required_values` entry names another declared
capture. `cost` is a deterministic work estimate comparable within one target
variant only. When `legal` is false, `required_values` is empty and replay is
forbidden.

### 12.1 Derivative ABI

Tangent type construction is recursive:

- `f16` maps to `f32`, `f32` to `f32`, and `f64` to `f64`;
- integer, boolean, buffer/image/opaque resource, and other
  non-differentiable leaves map to Zero and are omitted from tangent storage;
- tuple, struct, fixed array, and tensor aggregates recurse in canonical leaf
  order and preserve aggregate paths;
- a public derivative root is invalid when all leaves are Zero.

A derivative inherits rank, every static extent, and the exact ShapeSymbol IDs
of its primal; it MUST NOT mint replacement symbols. Cotangent and gradient
ValueLayout is derived from tangent leaves under the same pipeline Value ABI.
Every public gradient uses fresh owned Storage that does not alias its primal,
another public gradient, or caller Storage.

PullbackState is reusable: a successful forward state may service multiple
backward applications with different cotangents. A backward failure leaves it
reusable only if all work and caller-visible writes were rolled back and its
captures remain intact. An unrecoverable backend failure, capture corruption,
or failed rollback poisons the PullbackState; later use fails deterministically.
Poisoning PullbackState does not silently invalidate an otherwise healthy
Program instance unless the same failure also makes instance state unsafe.

### 12.2 Tape and resolved derivative metadata

A kernel tape crossing a forward/backward stage boundary is an ordinary
storage-backed Value:

- Storage uses an opaque descriptor with contract `vernon.ad_tape`, the exact
  contract hash reflected by both endpoints, `ownership = owned`,
  `lifetime = pullback`, `mutability = mutable`, and
  `usage = ["stage_binding"]`;
- the tape root has no ValueLayout;
- the forward node produces the root through initialize or write access;
- the root is listed in both backward captures and ResidualContract captures;
- the backward endpoint consumes that exact captured version.

Capacity, overflow status, allocator identity, and destruction are defined by
the opaque contract and Runtime. They are not side-channel stage metadata.
Any contract-hash, version, status, or capture mismatch fails before backward
effects commit.

TangentLayout is not a second serialized layout kind. It is the ValueLayout
obtained by applying section 12.1 tangent recursion and the selected value ABI
to the primal's canonical leaves. Aggregate derivative packing, offsets,
alignment, and paths are therefore fully represented by the derivative
Value's ordinary ValueLayout.

ResolveProgram derives a canonical DerivativeGroup table rather than
serializing duplicate authority. Each record is exactly
`{kind, path, primal, derivative}`, where `kind` is `gradient` or `cotangent`;
records are ordered by Signature gradients first and then cotangents, each in
public ABI order. The table is a pure projection of Signature and is rejected
internally if any pairing differs.

ResolveProgram also materializes the validated replay plan from
ResidualContract: selected captures, replay slices, required capture frontier,
costs, retained symbol witnesses, and tape lifetimes. Planner policy may
choose retain versus a legal replay, but it cannot change a replay slice,
frontier, legality, or cost declared by the Program.

Sections 13 through 15 are complete structural Program examples but not
hash-valid cooked bundles. Their zero StageContract hashes are placeholders paired in the
conformance suite with generated ArtifactSystems containing real IDs, Blobs,
modules, entry points, and reflection. The graphics fixtures always pair one
graphics StageArtifact containing vertex and fragment modules; an empty
Program binding array is valid only when its paired reflection has no bindable
endpoint.

## 13. Structural one-node compute Program

This example has one invocation argument and one instance parameter. Dispatch
uses the entry parameter directly.

```json
{
  "stages": {"scale": {"operation": "compute", "contract_hash": "0000000000000000000000000000000000000000000000000000000000000000"}},
  "parameters": [{"id": 0, "path": "groups_x", "value": 2}],
  "storages": [
    {"id": 0, "name": "x", "initial_value": 0, "ownership": "borrowed", "lifetime": "invocation", "mutability": "read_only", "descriptor": {"tag": "buffer", "byte_length": 1024, "alignment": 16, "memory": "device", "usage": ["storage"]}},
    {"id": 1, "name": "y", "initial_value": 1, "ownership": "owned", "lifetime": "invocation", "mutability": "mutable", "descriptor": {"tag": "buffer", "byte_length": 1024, "alignment": 16, "memory": "device", "usage": ["storage"]}}
  ],
  "values": [
    {"id": 0, "name": "x", "type": "tensor<256xf32>", "shape": [256], "origin": {"tag": "argument", "graph": "forward", "slot": 0}, "storage": 0, "value_layout": {"scope": "element", "layout_hash": "95f54cae607cf7751c0ec4327f86b0982056823c49534e9d8dddd66fc98c5f07", "byte_size": 4, "alignment": 4, "leaves": [{"path": [], "dtype": "f32", "byte_offset": 0, "scalar_count": 1, "shape": []}]}},
    {"id": 1, "name": "y", "type": "tensor<256xf32>", "shape": [256], "origin": {"tag": "node_result", "graph": "forward", "node": 0}, "storage": 1, "value_layout": {"scope": "element", "layout_hash": "95f54cae607cf7751c0ec4327f86b0982056823c49534e9d8dddd66fc98c5f07", "byte_size": 4, "alignment": 4, "leaves": [{"path": [], "dtype": "f32", "byte_offset": 0, "scalar_count": 1, "shape": []}]}},
    {"id": 2, "name": "groups_x", "type": "u32", "shape": [], "origin": {"tag": "parameter", "parameter": 0}, "value_layout": {"scope": "value", "layout_hash": "280f13e115d9ccfbe2a5a33aaafefb004f2ad59b8312a2807f2dfaa7b0966bc6", "byte_size": 4, "alignment": 4, "leaves": [{"path": [], "dtype": "u32", "byte_offset": 0, "scalar_count": 1, "shape": []}]}}
  ],
  "shape_symbols": [],
  "shape_constraints": [],
  "alias_preconditions": [],
  "graphs": [{
    "name": "forward",
    "direction": "forward",
    "inputs": [{"tag": "user_input", "value": 0, "slot": 0}, {"tag": "parameter", "value": 2, "parameter": 0}],
    "captures": [],
    "outputs": [{"tag": "user_output", "value": 1, "disposition": "transfer"}],
    "nodes": [{
      "id": 0,
      "stage": "scale",
      "operands": [0, 2],
      "results": [1],
      "bindings": [{"module": "compute", "interface": "argument", "index": 0, "tag": "resource", "access": 0}, {"module": "compute", "interface": "argument", "index": 1, "tag": "resource", "access": 1}],
      "accesses": [{"tag": "read", "storage": 0, "value": 0}, {"tag": "initialize", "storage": 1, "after": 1}],
      "operation": {"tag": "compute", "workgroups": [{"control": {"parameter": 0}}, 1, 1]}
    }]
  }],
  "signature": {"inputs": [{"path": "x", "value": 0}], "outputs": [{"path": "y", "value": 1, "disposition": "transfer"}], "cotangents": [], "gradients": []}
}
```

## 14. Structural one-node graphics Program

Width and height are entry arguments. They authoritatively control owned image
allocation and render area. Clear color is a Constant Program Value. The
result resource transfers only after successful commit.

```json
{
  "stages": {"fullscreen": {"operation": "graphics", "contract_hash": "0000000000000000000000000000000000000000000000000000000000000000"}},
  "parameters": [],
  "storages": [
    {"id": 0, "name": "color", "initial_value": 2, "ownership": "owned", "lifetime": "invocation", "mutability": "mutable", "descriptor": {"tag": "image", "dimension": "2d", "extent": [{"control": {"argument": 0}}, {"control": {"argument": 1}}, 1], "format": "rgba16_float", "sample_count": 1, "mip_levels": 1, "array_layers": 1, "aspects": ["color"], "usage": ["color_attachment", "sampled"]}}
  ],
  "values": [
    {"id": 0, "name": "width", "type": "u32", "shape": [], "origin": {"tag": "argument", "graph": "forward", "slot": 0}, "value_layout": {"scope": "value", "layout_hash": "280f13e115d9ccfbe2a5a33aaafefb004f2ad59b8312a2807f2dfaa7b0966bc6", "byte_size": 4, "alignment": 4, "leaves": [{"path": [], "dtype": "u32", "byte_offset": 0, "scalar_count": 1, "shape": []}]}},
    {"id": 1, "name": "height", "type": "u32", "shape": [], "origin": {"tag": "argument", "graph": "forward", "slot": 1}, "value_layout": {"scope": "value", "layout_hash": "280f13e115d9ccfbe2a5a33aaafefb004f2ad59b8312a2807f2dfaa7b0966bc6", "byte_size": 4, "alignment": 4, "leaves": [{"path": [], "dtype": "u32", "byte_offset": 0, "scalar_count": 1, "shape": []}]}},
    {"id": 2, "name": "color.initial", "type": "image<rgba16_float>", "origin": {"tag": "allocation", "graph": "forward"}, "storage": 0},
    {"id": 3, "name": "color.view", "type": "image_view<rgba16_float>", "origin": {"tag": "view", "source": 2}, "storage": 0, "view": {"tag": "image_view", "aspects": ["color"], "base_mip": 0, "mip_count": 1, "base_layer": 0, "layer_count": 1}},
    {"id": 4, "name": "color.output", "type": "image<rgba16_float>", "origin": {"tag": "node_result", "graph": "forward", "node": 0}, "storage": 0},
    {"id": 5, "name": "clear_color", "type": "vector<4xf32>", "shape": [4], "origin": {"tag": "constant", "payload": {"tag": "inline", "encoding": "value_abi", "data": "AAAAAAAAAAAAAAAAAAAAAA==", "byte_length": 16, "sha256": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"}}, "value_layout": {"scope": "value", "layout_hash": "d477029ca3234046adda41757e10fe64396ed8d0233ee1c05aeca69e6e734d70", "byte_size": 16, "alignment": 4, "leaves": [{"path": [], "dtype": "f32", "byte_offset": 0, "scalar_count": 4, "shape": [4]}]}}
  ],
  "shape_symbols": [
    {"id": 0, "name": "width", "source": {"tag": "scalar", "control": {"argument": 0}}, "min": 1, "max": 16384},
    {"id": 1, "name": "height", "source": {"tag": "scalar", "control": {"argument": 1}}, "min": 1, "max": 16384}
  ],
  "shape_constraints": [],
  "alias_preconditions": [],
  "graphs": [{
    "name": "forward",
    "direction": "forward",
    "inputs": [{"tag": "user_input", "value": 0, "slot": 0}, {"tag": "user_input", "value": 1, "slot": 1}, {"tag": "allocation", "value": 2, "storage": 0}, {"tag": "constant", "value": 5}],
    "captures": [],
    "outputs": [{"tag": "user_output", "value": 4, "disposition": "transfer"}],
    "nodes": [{
      "id": 0,
      "stage": "fullscreen",
      "operands": [0, 1, 2, 3, 5],
      "results": [4],
      "bindings": [],
      "accesses": [{"tag": "attachment", "storage": 0, "before": 2, "after": 4, "view": 3}],
      "operation": {
        "tag": "graphics",
        "attachments": {"colors": [{"location": 0, "access": 0, "load": {"tag": "clear", "value": {"constant": 5}}, "store": "store"}], "depth_stencil": null, "render_area": {"x": 0, "y": 0, "width": {"control": {"argument": 0}}, "height": {"control": {"argument": 1}}}, "layer_count": 1},
        "state": {"raster": {"front_face": "counter_clockwise", "cull_mode": "none", "fill_mode": "fill"}, "depth_stencil": {"depth_test": false, "depth_write": false, "depth_compare": "always", "stencil_test": false}, "multisample": {"sample_mask": 4294967295, "alpha_to_coverage": false}, "blend": [{"location": 0, "enabled": false, "write_mask": ["a", "b", "g", "r"]}], "viewport": null, "scissor": null},
        "draw": {"tag": "direct", "vertex_count": 3, "instance_count": 1}
      }
    }]
  }],
  "signature": {"inputs": [{"path": "width", "value": 0}, {"path": "height", "value": 1}], "outputs": [{"path": "color", "value": 4, "disposition": "transfer"}], "cotangents": [], "gradients": []}
}
```

## 15. Structural mixed multi-node Program

This Program uses a parameter, a constant-backed read-only buffer view, a
compute node, and an indexed graphics node. The constant tensor view is not a
ConstantOrigin view.

```json
{
  "stages": {"build_vertices": {"operation": "compute", "contract_hash": "0000000000000000000000000000000000000000000000000000000000000000"}, "draw_mesh": {"operation": "graphics", "contract_hash": "0000000000000000000000000000000000000000000000000000000000000000"}},
  "parameters": [{"id": 0, "path": "instance_count", "value": 0}, {"id": 1, "path": "instances", "value": 1}],
  "storages": [
    {"id": 0, "name": "instances", "initial_value": 1, "ownership": "borrowed", "lifetime": "instance", "mutability": "read_only", "descriptor": {"tag": "buffer", "byte_length": 64, "alignment": 16, "memory": "device", "usage": ["storage"]}},
    {"id": 1, "name": "vertices", "initial_value": 2, "ownership": "owned", "lifetime": "invocation", "mutability": "mutable", "descriptor": {"tag": "buffer", "byte_length": 48, "alignment": 16, "memory": "device", "usage": ["storage", "vertex"]}},
    {"id": 2, "name": "indices.constant", "initial_value": 4, "ownership": "owned", "lifetime": "instance", "mutability": "read_only", "descriptor": {"tag": "buffer", "byte_length": 12, "alignment": 4, "memory": "device", "usage": ["index"]}},
    {"id": 3, "name": "color", "initial_value": 6, "ownership": "owned", "lifetime": "invocation", "mutability": "mutable", "descriptor": {"tag": "image", "dimension": "2d", "extent": [640, 480, 1], "format": "rgba8_unorm", "sample_count": 1, "mip_levels": 1, "array_layers": 1, "aspects": ["color"], "usage": ["color_attachment"]}}
  ],
  "values": [
    {"id": 0, "name": "instance_count", "type": "u32", "shape": [], "origin": {"tag": "parameter", "parameter": 0}, "value_layout": {"scope": "value", "layout_hash": "280f13e115d9ccfbe2a5a33aaafefb004f2ad59b8312a2807f2dfaa7b0966bc6", "byte_size": 4, "alignment": 4, "leaves": [{"path": [], "dtype": "u32", "byte_offset": 0, "scalar_count": 1, "shape": []}]}},
    {"id": 1, "name": "instances", "type": "tensor<4xvector<4xf32>>", "shape": [4, 4], "origin": {"tag": "parameter", "parameter": 1}, "storage": 0, "value_layout": {"scope": "element", "layout_hash": "95f54cae607cf7751c0ec4327f86b0982056823c49534e9d8dddd66fc98c5f07", "byte_size": 4, "alignment": 4, "leaves": [{"path": [], "dtype": "f32", "byte_offset": 0, "scalar_count": 1, "shape": []}]}},
    {"id": 2, "name": "vertices.initial", "type": "tensor<3xvector<4xf32>>", "shape": [3, 4], "origin": {"tag": "allocation", "graph": "forward"}, "storage": 1, "value_layout": {"scope": "element", "layout_hash": "95f54cae607cf7751c0ec4327f86b0982056823c49534e9d8dddd66fc98c5f07", "byte_size": 4, "alignment": 4, "leaves": [{"path": [], "dtype": "f32", "byte_offset": 0, "scalar_count": 1, "shape": []}]}},
    {"id": 3, "name": "vertices.ready", "type": "tensor<3xvector<4xf32>>", "shape": [3, 4], "origin": {"tag": "node_result", "graph": "forward", "node": 0}, "storage": 1, "value_layout": {"scope": "element", "layout_hash": "95f54cae607cf7751c0ec4327f86b0982056823c49534e9d8dddd66fc98c5f07", "byte_size": 4, "alignment": 4, "leaves": [{"path": [], "dtype": "f32", "byte_offset": 0, "scalar_count": 1, "shape": []}]}},
    {"id": 4, "name": "indices.backing", "type": "tensor<3xu32>", "shape": [3], "origin": {"tag": "constant", "payload": {"tag": "artifact", "artifact": "constant-indices"}}, "storage": 2, "value_layout": {"scope": "element", "layout_hash": "c46c9522633f23a233ac134abdcf8d3c0eaa33ffecbd7b4d730605745580c30e", "byte_size": 4, "alignment": 4, "leaves": [{"path": [], "dtype": "u32", "byte_offset": 0, "scalar_count": 1, "shape": []}]}},
    {"id": 5, "name": "indices.view", "type": "tensor_view<3xu32>", "shape": [3], "origin": {"tag": "view", "source": 4}, "storage": 2, "view": {"tag": "buffer_view", "byte_offset": 0, "extents": [3], "byte_strides": [4]}, "value_layout": {"scope": "element", "layout_hash": "c46c9522633f23a233ac134abdcf8d3c0eaa33ffecbd7b4d730605745580c30e", "byte_size": 4, "alignment": 4, "leaves": [{"path": [], "dtype": "u32", "byte_offset": 0, "scalar_count": 1, "shape": []}]}},
    {"id": 6, "name": "color.initial", "type": "image<rgba8_unorm>", "origin": {"tag": "allocation", "graph": "forward"}, "storage": 3},
    {"id": 7, "name": "color.view", "type": "image_view<rgba8_unorm>", "origin": {"tag": "view", "source": 6}, "storage": 3, "view": {"tag": "image_view", "aspects": ["color"], "base_mip": 0, "mip_count": 1, "base_layer": 0, "layer_count": 1}},
    {"id": 8, "name": "color.output", "type": "image<rgba8_unorm>", "origin": {"tag": "node_result", "graph": "forward", "node": 1}, "storage": 3}
  ],
  "shape_symbols": [],
  "shape_constraints": [],
  "alias_preconditions": [],
  "graphs": [{
    "name": "forward",
    "direction": "forward",
    "inputs": [{"tag": "parameter", "value": 0, "parameter": 0}, {"tag": "parameter", "value": 1, "parameter": 1}, {"tag": "allocation", "value": 2, "storage": 1}, {"tag": "allocation", "value": 6, "storage": 3}, {"tag": "constant", "value": 4}],
    "captures": [],
    "outputs": [{"tag": "user_output", "value": 8, "disposition": "transfer"}],
    "nodes": [
      {"id": 0, "stage": "build_vertices", "operands": [1, 2], "results": [3], "bindings": [{"module": "compute", "interface": "argument", "index": 0, "tag": "resource", "access": 0}, {"module": "compute", "interface": "argument", "index": 1, "tag": "resource", "access": 1}], "accesses": [{"tag": "read", "storage": 0, "value": 1}, {"tag": "initialize", "storage": 1, "after": 3}], "operation": {"tag": "compute", "workgroups": [1, 1, 1]}},
      {"id": 1, "stage": "draw_mesh", "operands": [0, 3, 4, 5, 6, 7], "results": [8], "bindings": [{"module": "vertex", "interface": "argument", "index": 0, "tag": "resource", "access": 0}], "accesses": [{"tag": "read", "storage": 1, "value": 3}, {"tag": "read", "storage": 2, "value": 4, "view": 5}, {"tag": "attachment", "storage": 3, "before": 6, "after": 8, "view": 7}], "operation": {"tag": "graphics", "attachments": {"colors": [{"location": 0, "access": 2, "load": {"tag": "discard"}, "store": "store"}], "depth_stencil": null, "render_area": {"x": 0, "y": 0, "width": 640, "height": 480}, "layer_count": 1}, "state": {"raster": {"front_face": "counter_clockwise", "cull_mode": "back", "fill_mode": "fill"}, "depth_stencil": {"depth_test": false, "depth_write": false, "depth_compare": "always", "stencil_test": false}, "multisample": {"sample_mask": 4294967295, "alpha_to_coverage": false}, "blend": [{"location": 0, "enabled": false, "write_mask": ["a", "b", "g", "r"]}], "viewport": null, "scissor": null}, "draw": {"tag": "indexed", "index_access": 1, "index_count": 3, "instance_count": {"control": {"parameter": 0}}}}}
    ]
  }],
  "signature": {"inputs": [], "outputs": [{"path": "color", "value": 8, "disposition": "transfer"}], "cotangents": [], "gradients": []}
}
```

Both instance-bound Values have distinct Parameter records and consume no
invocation slots. The index tensor is initialized from artifact bytes into
read-only owned Storage, then projected through ViewOrigin for indexed draw.
`instance_count` is consumed only by node 1 draw control and therefore appears
only in node 1 operands.
Node 0 produces Value 3 and node 1 consumes Value 3 in both `operands` and its
read access, so Resolve derives edge `0 -> 1`; no dependency member is
serialized.

## 16. Resolve

Resolve is pure with respect to caller resources and backend execution:

1. Parse strict canonical JSON and verify the selected contract pair.
2. Select exactly one variant and validate its aggregate Runtime requirements
   before external code I/O.
3. Validate IDs, array ordering, origins, Storage descriptors, ValueLayout,
   graph boundaries, exact operand/result closure, and Signature.
4. Resolve every logical stage ID through variant `stage_bindings` to one
   StageArtifact and validate its operation, contract hash, modules, and
   reflection.
5. Recompute selected Artifact IDs, authenticate only selected reachable Blob
   ranges, and select exact code modules and entry points.
6. Build exact endpoint bindings and resource-access plans.
7. Validate ControlValueRef provenance without reading device resources.
8. Derive the complete producer and resource dependency edge set.
9. Validate DAG topology, resource version chains, attachment compatibility,
   draw mode, and target capabilities.
10. Construct an immutable `ResolvedProgram` containing resolved code handles,
   entry points, bindings, accesses, dependency edges, canonical derivative
   groups, and the validated capture/replay plan.

Resolve never patches missing members, infers bindings from names, changes a
descriptor, inserts a node, converts a Program representation, or executes
backend work.

## 17. Runtime

Instance creation binds every parameter and creates owned instance Storage.
Invocation entry binds arguments, creates owned invocation Storage, restores
captured forward Values when applicable, evaluates control values, checks
borrowed providers, binds symbols, and checks entry constraints before
submitting work.

Before instance creation, Runtime loads or performs the target-defined JIT for
each authenticated CodeModule and resolves every declared entry point. A load,
JIT, link, reflection, or entry-point failure rejects the resolved Program;
Runtime does not try another format, module, symbol, artifact, or target.

Runtime executes nodes in any schedule equivalent to the canonical DAG.
Resource barriers and synchronization are derived after validation. Runtime
MUST NOT insert semantic resource copies or device-to-host reads to satisfy
metadata.

All externally observable changes are transactional. Outputs, state updates,
and transferred ownership become visible only after every submitted operation
completes successfully and all postconditions pass. Failure:

- publishes no output;
- performs no ownership transfer;
- preserves prior committed instance state;
- invalidates uncommitted owned invocation resources;
- retires submitted backend work before releasing referenced resources;
- emits one stable diagnostic.

A write to borrowed caller-visible Storage is transactional too. Runtime MUST
execute it against a private shadow or retain sufficient original contents to
restore the exact provider bytes/subresources before reporting failure. Merely
delaying publication of outputs is insufficient. If backend failure makes a
borrowed write, instance Storage, or ownership state impossible to restore
safely, the Program instance enters terminal-failed state; every later
instance bind, invocation, or pullback fails without backend execution.

A borrowed output is valid only for its declared provider lifetime. A
transferred owned output is removed from Program ownership at commit.

## 18. Diagnostics

Every rejection emits a strict Diagnostic:

```json
{
  "code": "PROGRAM_CONTROL_NODE_RESULT",
  "phase": "resolve",
  "path": "/variants/0/program/storages/0/descriptor/extent/0",
  "context": {"value": 7, "node": 2},
  "message": "control value 7 is not available at graph entry"
}
```

- `code` is a stable uppercase ASCII identifier and is the test authority;
- `phase` is `parse`, `canonicalize`, `resolve`, `instance_bind`,
  `invocation_entry`, `execute`, `commit`, or `pullback`;
- `path` is an RFC 6901 JSON Pointer to the primary failing member;
- `context` is a strict code-specific object containing stable IDs and values;
- `message` is non-empty explanatory English and is not test identity.

For a given invalid manifest and contract pair, validators MUST report the
same primary diagnostic code, phase, path, and context. Producers and tests
MUST NOT match message text. If multiple errors are present, the first is
selected by phase order above, then lexical JSON Pointer order, then code.

Required stable codes include:

- `PROGRAM_UNKNOWN_FIELD`
- `PROGRAM_CANONICAL_JSON`
- `PROGRAM_BUNDLE_HASH`
- `PROGRAM_NON_CANONICAL_ORDER`
- `PROGRAM_ID_SEQUENCE`
- `PROGRAM_STAGE_MISSING`
- `PROGRAM_ARTIFACT_ID`
- `PROGRAM_ARTIFACT_TARGET`
- `PROGRAM_ARTIFACT_MODULE`
- `PROGRAM_ARTIFACT_ENCODING`
- `PROGRAM_ARTIFACT_ENTRY_POINT`
- `PROGRAM_BLOB_AUTHENTICATION`
- `PROGRAM_RUNTIME_REQUIREMENTS`
- `PROGRAM_PARAMETER_BINDING`
- `PROGRAM_STORAGE_DESCRIPTOR`
- `PROGRAM_STORAGE_INITIAL_VALUE`
- `PROGRAM_VALUE_ORIGIN`
- `PROGRAM_OPERAND_CLOSURE`
- `PROGRAM_LAYOUT_HASH`
- `PROGRAM_CONSTANT_VIEW`
- `PROGRAM_CONTROL_NODE_RESULT`
- `PROGRAM_CONTROL_UNAVAILABLE`
- `PROGRAM_SHAPE_CONSTRAINT`
- `PROGRAM_ALIAS_PRECONDITION`
- `PROGRAM_GRAPH_CYCLE`
- `PROGRAM_DEPENDENCY_MEMBER`
- `PROGRAM_DEPENDENCY_ORDER`
- `PROGRAM_OPERATION_UNSUPPORTED`
- `PROGRAM_BINDING_MISMATCH`
- `PROGRAM_REFLECTION_MISMATCH`
- `PROGRAM_ACCESS_CHAIN`
- `PROGRAM_ACCESS_OVERLAP`
- `PROGRAM_ATTACHMENT_RESOLVE`
- `PROGRAM_RESOURCE_OUTPUT`
- `PROGRAM_SIGNATURE_MISMATCH`
- `PROGRAM_RESIDUAL_CONTRACT`
- `PROGRAM_REPLAY_INVALID`
- `PROGRAM_DERIVATIVE_ABI`
- `PROGRAM_TAPE_CONTRACT`
- `PROGRAM_RUNTIME_FAILURE`

`PROGRAM_DEPENDENCY_MEMBER` is the parse error reserved for a serialized order,
dependency, event, fence, or token member. `PROGRAM_DEPENDENCY_ORDER` is the
resolve error when an acyclic derived edge points backward in serialized node
order; `PROGRAM_GRAPH_CYCLE` is reserved for an actual cycle in the derived
edge set. `PROGRAM_ACCESS_OVERLAP` is the invocation-entry error for mutable
physical overlap between distinct borrowed Storage IDs. Storage chain,
producer, predecessor, or successor failures use `PROGRAM_ACCESS_CHAIN`.
Malformed canonical syntax or scalar spelling uses `PROGRAM_CANONICAL_JSON`;
bundle content-hash mismatch uses `PROGRAM_BUNDLE_HASH`;
Blob length, range, location, or hash failure uses
`PROGRAM_BLOB_AUTHENTICATION`. Failed shape constraints and borrowed alias
preconditions use `PROGRAM_SHAPE_CONSTRAINT` and
`PROGRAM_ALIAS_PRECONDITION`. Resolve/load disagreement uses
`PROGRAM_ATTACHMENT_RESOLVE`, `PROGRAM_RESIDUAL_CONTRACT`,
`PROGRAM_REPLAY_INVALID`, `PROGRAM_DERIVATIVE_ABI`, or
`PROGRAM_TAPE_CONTRACT` for the corresponding authority.

## 19. Required rejection cases

A conforming implementation rejects at least:

- a cooked bundle with missing/unknown root members, wrong content hash,
  duplicate/non-canonical variant keys, or a missing/extra variant
  `stage_bindings` entry;
- an unreachable Artifact or Blob;
- a Program object with missing or unknown members;
- any second Program/executable object adjacent to or inside
  `variant.program`;
- any Program-specific schema/version field;
- non-canonical array order, duplicate IDs, or non-contiguous IDs;
- a missing parameter, parameter default, parameter invocation slot, or
  Parameter/GraphInput/ParameterOrigin disagreement;
- a stage binding selecting a missing, non-stage, wrong-operation,
  wrong-contract, wrong-target, or incorrectly content-addressed artifact;
- an unsupported code format/role/target combination, wrong module count or
  order, empty entry point, invalid code range/hash, or missing Runtime
  requirement;
- a Storage without ownership, lifetime, mutability, or one strict descriptor;
- a Storage without exactly one valid `initial_value`, with a view as its
  initial root, or with an invalid initial origin/access relation;
- a resource descriptor duplicated or contradicted by a Value;
- owned Storage whose runtime creation differs from its descriptor;
- borrowed Storage whose provider fails any descriptor constraint;
- image, opaque, resource, or token-shaped ValueLayout;
- an external or effect token;
- ConstantOrigin applied directly to a TensorView;
- a view outside its Storage descriptor or writable non-injective view;
- an operand/result set that omits or adds a binding, access, view, control,
  or produced Value;
- a ControlValueRef to NodeResultOrigin or any non-entry value;
- metadata that would require readback, mapping, copying, or synchronization;
- raw non-integer numeric render-state metadata;
- a graph cycle or non-static graph construct;
- a serialized dependency/order/event/token member, a derived edge that points
  backward in node order, or a missing producer relation;
- any operation other than compute direct dispatch or graphics direct/indexed
  draw;
- serialized transfer, presentation, boundary, barrier, indirect, branch,
  loop, call, or other control operation;
- compute workgroups with wrong rank, zero, or unsupported control source;
- graphics operation missing direct attachments, state, or draw;
- attachment/storage/view/access disagreement;
- indirect draw, unsupported index format, or index access represented as a
  shader endpoint;
- binding/reflection/access disagreement;
- reflection with an unknown endpoint variant, incomplete portable ABI,
  unsupported workgroup/dispatch limit, graphics role/capability mismatch, or
  Program binding for a system endpoint;
- a write to read-only Storage;
- `may_alias_read_only` involving mutable Storage, or mutable physical overlap
  between distinct Storage IDs;
- a Storage version without exactly one producer and valid predecessor;
- a forked Storage version chain, duplicate successor, invalid access view, or
  omitted overlapping-reader anti-dependency in the resolved execution plan;
- a public resource not represented as ordinary user output and Signature
  output;
- resource output without `borrow` or `transfer`, or transfer of borrowed
  Storage;
- ownership becoming visible before successful commit;
- Signature and graph-boundary disagreement;
- backward captures or symbol witnesses differing from ResidualContract.

### 19.1 Key negative fixtures

Node-produced dispatch metadata:

```json
{"control": {"constant": 7}}
```

The `constant` ControlValueRef tag can denote only ConstantOrigin. If Value 7
instead has NodeResultOrigin, the record is syntactically expressible but has
a tag/provenance mismatch and is rejected with
`PROGRAM_CONTROL_NODE_RESULT` during resolve.

Direct constant tensor view:

```json
{"type": "tensor_view<4xf32>", "origin": {"tag": "constant", "payload": {"tag": "artifact", "artifact": "weights"}}}
```

Reject with `PROGRAM_CONSTANT_VIEW`; the producer must emit constant-backed
read-only buffer Storage and a ViewOrigin Value.

Serialized synchronization operation:

```json
{"operation": {"tag": "barrier"}}
```

Reject with `PROGRAM_OPERATION_UNSUPPORTED`; synchronization is derived.

Serialized dependency member:

```json
{"id": 1, "dependencies": [0]}
```

Reject with `PROGRAM_DEPENDENCY_MEMBER`. If the same order is required by SSA
or ResourceAccess, Resolve derives it; otherwise it is not Program semantics.

Operand closure omission:

```json
{"operands": [], "accesses": [{"tag": "read", "storage": 0, "value": 3}]}
```

Reject with `PROGRAM_OPERAND_CLOSURE` because Value 3 is consumed but omitted
from `operands`.

Early ownership publication:

```json
{"tag": "user_output", "value": 4, "disposition": "transfer"}
```

If Storage for Value 4 is borrowed, or runtime publishes it before commit,
reject or fail with `PROGRAM_RESOURCE_OUTPUT`.

## 20. Conformance suite

Compiler and runtime repositories MUST share content-addressed complete
Program fixtures covering:

1. rejection of this schema under current pair 12/16 and acceptance only
   under the future coordinated pair assigned to it;
2. exact bundle root, variant ordering, content hash, every target, Runtime
   requirement, code format, module role, Blob range, entry point, and
   StageArtifact content hash;
   paired backend bundles must preserve identical Program and StageContract
   hashes while changing only variant stage bindings and ArtifactSystem;
   all requested variants are emitted, while Runtime opens only the selected
   variant's reachable external code and value Blobs;
3. one-node direct compute dispatch;
4. one-node direct graphics draw, including distinct vertex and fragment
   modules;
5. mixed compute plus indexed graphics DAG;
6. Parameter paths with no defaults or invocation slots;
7. Blob/BlobLocation decoded length, whole hash, range hash, and bounds;
8. constant-backed read-only buffer Storage and ViewOrigin;
9. every Storage initial-root origin and initialization relation;
10. dynamic buffer length and image extent from each ControlValueRef class;
11. rejection of every NodeResult control source;
12. dynamic view, dispatch, render, and draw metadata without readback;
13. owned and borrowed Storage for buffer, image, and opaque descriptors;
14. resource user outputs with borrow, successful transfer, failed transfer,
    caller-visible borrowed-write rollback, and terminal instance failure;
15. strict reflected endpoint variants, whole-root/leaf matching, compute
    limits, graphics roles/capabilities, and system endpoints;
16. every Attachment, RenderState, and Draw field and enum combination;
17. derivative recursion, inherited ShapeSymbols, fresh gradient Storage,
    capture equality, canonical replay, reuse, and poison validation;
18. alias preconditions and resource-version hazards;
19. exact operand/result closure, producer/version edges, overlapping-reader
    write-after-read anti-dependencies, disjoint and unknown-overlap behavior,
    fork rejection, cycles, and canonical topological order;
20. layout-hash recomputation, canonical JSON bytes, and all array orders;
21. stable diagnostic code, phase, path, and context;
22. rejection of every unsupported operation listed in section 19.

Each positive fixture includes artifact reflection, blobs, expected public
values, expected committed ownership/state, and canonical content hashes.
Each negative fixture asserts exact diagnostic code, phase, path, and context.
Backend suites additionally prove that derived barriers preserve semantics
without serialized synchronization nodes.

## Appendix A. Implementation invariants

An implementation is conforming only when all of these invariants hold:

- every executable variant has exactly one Program and it is
  `variant.program`;
- Program and StageContract hashes are identical across target bundles for the
  same executable semantics;
- every executable path is represented by a Program graph;
- every stage resolves to authenticated target code and exact entry points;
- Storage is the only physical descriptor authority;
- ValueLayout describes only logical value bytes;
- all operation metadata is static or entry-controlled;
- no runtime metadata dependency requires a device observation;
- operation selection is a closed compute/graphics union;
- the execution DAG is the exact derived producer/resource edge set and every
  Storage has one non-branching version chain;
- resource ownership changes only at successful commit;
- canonical bytes and stable diagnostics are reproducible across processes.

This appendix is normative. It introduces no alternate representation,
translation path, or release mode.

## Appendix B. Closed ABI vocabulary

This appendix closes spellings used by the strict objects above. A coordinated
contract release may add spellings only by assigning a new compiler/pipeline
pair.

Canonical type strings contain no whitespace. Scalars are `bool`, `i8`, `u8`,
`i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f16`, `f32`, and `f64`.
Recursive forms are:

```text
vector<NxT>
matrix<RxCxT>
array<NxT>
tuple<T0,T1,...>
struct<field0:T0,field1:T1,...>
tensor<D0xD1x...xT>
tensor_view<D0xD1x...xT>
image<FORMAT>
image_view<FORMAT>
sampler
opaque<CONTRACT>
```

`N`, `R`, `C`, and static `D` are positive canonical decimal integers; a
dynamic `D` is `?` and must correspond positionally to Value `shape`.
Aggregate field order is semantic, field names are unique NFC identifiers,
and nesting is recursive. Tensor/type dimensions MUST equal Value shape
dimensions after symbol substitution. Resource formats are
`r8_unorm`, `r8_snorm`, `r16_float`, `r32_float`, `r32_uint`, `r32_sint`,
`rg8_unorm`, `rg16_float`, `rg32_float`, `rgba8_unorm`, `rgba8_srgb`,
`bgra8_unorm`, `bgra8_srgb`, `rgba16_float`, `rgba32_float`,
`depth16_unorm`, `depth24_plus`, `depth24_plus_stencil8`,
`depth32_float`, `depth32_float_stencil8`, or `unknown`. `unknown` is allowed
only for sampled image logical types; Storage descriptors always use concrete
formats.

Buffer memory is `host`, `device`, or `shared`. Buffer usage members are
`uniform`, `storage`, `vertex`, `index`, or `stage_binding`. Image usage
members are `sampled`, `storage`, `color_attachment`,
`depth_stencil_attachment`, or `stage_binding`. Opaque usage contains only
`stage_binding`. Image aspects are `color`, `depth`, or `stencil`. Usage and
aspect arrays are sorted and non-empty.

Endpoint address space is `host`, `device`, `workgroup`, `constant`, or
`opaque`. Resource role is `uniform`, `storage`, `sampled`, `sampler`, or
`vertex`. Transport is `by_value`, `resource_handle`, or `device_address`.

A resource endpoint `layout` is exactly one of:

```json
{"tag":"buffer","view_rank":2,"element_layout_hash":"0000000000000000000000000000000000000000000000000000000000000000","minimum_alignment":16}
{"tag":"image","dimension":"2d","format":"rgba16_float","sample_count":1,"aspects":["color"]}
{"tag":"sampler"}
{"tag":"opaque","contract_hash":"0000000000000000000000000000000000000000000000000000000000000000"}
```

Hashes must be real lowercase digests; zeros above are structural
placeholders. Buffer `view_rank` is uint32, and image fields use Storage
descriptor vocabulary. Endpoint constraints may be less specific only by
using image format `"any"` or sample count `0`; Storage remains the concrete
resource authority. An image layout for role `sampled` additionally permits
optional
`sampler_endpoint:{"module":"fragment","interface":"argument","index":0}`;
it is forbidden
for every other role.

Every endpoint `abi` object is exactly `{"bindings":[...]}`. It is
backend-independent. Bindings are ordered by semantic ordinal and then
carrier. Each binding contains exactly `semantic` and `carrier`. Semantic is
one of:

```json
"value"
"resource"
"sampler"
"byte_offset"
"byte_length"
{"storage_leaf": 0}
{"extent": 0}
{"byte_stride": 0}
```

Semantic ordinals are `value`, `resource`, `sampler`, `byte_offset`,
`byte_length`, storage leaves by canonical ValueLayout leaf index, then extents
by axis, then byte strides by axis. Scalar semantics occur at most once; leaf
and axis records use contiguous indices. A
carrier is exactly one of:

```json
{"tag":"value_slot","slot":0,"byte_offset":0,"byte_size":16,"alignment":16}
{"tag":"resource_slot","slot":0}
{"tag":"constant_region","slot":0,"byte_offset":0,"byte_size":16,"alignment":16}
{"tag":"device_address_slot","slot":0}
```

Carrier tag ordinals are listed order above; equal-tag carriers are ordered by
their numeric tuple. Numeric fields are uint32, sizes/alignments are positive,
and alignment is a power of two. Endpoint transport constrains its primary
`value`, `resource`, or `sampler` semantic: `by_value` uses a value slot or
constant region; `resource_handle` uses a resource slot; `device_address` uses
a device-address slot or value slot. Auxiliary offset, length, extent, and
stride semantics use value slots or constant regions independently of the
primary transport. TensorView endpoints list one `storage_leaf` resource
carrier for every canonical element-layout leaf, followed by every extent,
stride, byte offset, and byte length carrier consumed by emitted code.
`sampler_endpoint` is required when image and sampler are one logical combined
binding.

Native descriptor sets/bindings, root parameters, argument-buffer indices,
uniform locations, texture units, register numbers, and host frame symbols
are never serialized in Program or StageArtifact reflection. Each target
format has a deterministic lowering from portable ABI slots to its native
locations. Artifact validation proves that emitted code implements that
lowering; Runtime does not recover native locations from Program metadata.

Portable slots are global within one stage and contiguous from zero. A slot is
lowered as follows; this table is target ABI, not serialized reflection:

- CPU: endpoints are ordered by reflected endpoint order and each endpoint's
  carriers are ordered by semantic ordinal. Those carriers are concatenated
  into a pointer-aligned frame; resource and device-address carriers contain
  their address and value carriers contain their declared bytes. This
  endpoint-major frame rule is independent of graphics slot numbering and has
  no hidden TensorView fields because every frame word has a declared carrier.
- Vulkan: slot `S` is descriptor set 0, binding `S`. Resource carriers use the
  descriptor type implied by endpoint role. Value slots are storage-buffer
  descriptors containing the declared bytes. No push-constant substitution is
  permitted.
- Metal: slot `S` is member ID `S` of argument buffer index 0. Resource and
  value carriers are represented by the corresponding argument-buffer pointer;
  a value slot points at bytes with the declared size and alignment.
- CUDA: slot `S` is kernel parameter `S`. Resources and device addresses are
  pointer parameters; value slots are pointer parameters to the declared bytes.
- DirectX: slot `S` is register index `S`, space 0, in the register class
  implied by endpoint role. Value slots are SRV byte-address buffers. The root
  signature is derived from this sequence.
- OpenGL and OpenGL ES: slot `S` is interface binding `S`; resources use the
  endpoint-implied interface class and value slots use shader-storage blocks.

The compiler MUST emit the exact locations above and validate them against its
own artifact parser before returning a successful result. Runtime derives the
same mapping from portable slots and rejects an artifact whose target
reflection disagrees. Compiler-private sidecars may report native locations,
but neither cooker nor Runtime may require those sidecars.

System semantics are `position`, `vertex_index`, `instance_index`,
`fragment_coordinates`, `sample_index`, `sample_mask`, `target_extent`, or
`default_sampler`. Fixed-function semantics may use an empty ABI binding
array; generated target extent and default sampler require concrete portable
ABI bindings.

Compute subgroup is `null` or exactly:

```json
{"minimum_size": 4, "maximum_size": 32, "required_size": 32}
```

`required_size` is optional; all sizes are positive powers of two and, when
present, lie in the inclusive range. Required feature names are
`atomics`, `barriers`, `compute`, `instancing`, `samplers`, `tensor_views`,
`textures`, and `workgroup_storage`.
Compute capabilities is exactly `["direct_dispatch"]`.

Graphics topology is `point_list`, `line_list`, `line_strip`,
`triangle_list`, or `triangle_strip`. Graphics capabilities are
`direct_draw`, `indexed_draw`, `instancing`, `multisample`, `depth_stencil`,
`blend`, `dynamic_viewport`, `dynamic_scissor`, and `resolve`.
Interpolation is `smooth`, `flat`, or `noperspective`.

A linkage record is exactly one of:

```json
{"location":0,"type":"vector<4xf32>","interpolation":"smooth"}
{"builtin":"position","type":"vector<4xf32>"}
```

Location records require interpolation; builtin records forbid it. Builtins
use the system-semantic vocabulary applicable to the module direction.
Vertex-input records are exactly
`{location, endpoint_index, format, byte_offset, byte_stride, step, divisor}`;
`step` is `vertex` or `instance`, and divisor is uint32. Fragment-output
records are exactly `{location,type}`. Attachment constraints are exactly
`{location,formats,sample_counts,aspects}` with sorted unique arrays.
Vertex format is `r32_float`, `rg32_float`, `rgb32_float`, `rgba32_float`,
`r32_uint`, `rg32_uint`, `rgb32_uint`, `rgba32_uint`, `r32_sint`,
`rg32_sint`, `rgb32_sint`, or `rgba32_sint`. Index formats are `u16` or
`u32`.

Multi-module loading is delegated to the ArtifactSystem target adapter. The
portable contract requires it to consume the declared modules as one graphics
stage and validate cross-module linkage, resource layout, attachment
constraints, entry roles, and portable ABI slots before exposure. Native API
objects and binding locations never become Program fields. CPU registration
source is not a deployment artifact: cooker-generated registration code is a
build sidecar derived from the authenticated relocatable-object entry symbol
and is never loaded or interpreted by Runtime.
