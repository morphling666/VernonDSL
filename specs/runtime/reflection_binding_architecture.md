# Reflection and backend binding architecture

Status: implemented architecture record and design assessment.

This document records how typed compiler reflection becomes executable
bindings for CPU, CUDA, Vulkan, DirectX 12, Metal, OpenGL, and OpenGL ES. It
describes the implemented `PreparedComputeBindingPlan` and
`PreparedGraphicsBindingPlan` architecture and its remaining schema work.

Normative Program fields are defined by
[`../program/execution_manifest.md`](../program/execution_manifest.md).
Runtime ownership and execution are defined by [`design.md`](design.md).
The remaining physical-binding migration is tracked by
[`binding_and_tape_contract_plan.md`](binding_and_tape_contract_plan.md).
This document does not introduce a second manifest schema.

## 1. Decision summary

The correct architecture has four independent authorities:

1. the Program owns logical Values, Storages, endpoint identity, and portable
   ABI slots;
2. the compiler owns target code and the physical ABI expected by that code;
3. RuntimeCore owns Program-to-endpoint projection and invocation
   materialization;
4. the provider owns native resource creation, command encoding, submission,
   completion, and native limit validation.

These authorities must not be collapsed. In particular:

- logical layout is not a target storage layout;
- source representation is not a physical carrier;
- a Program slot is not a provider slot or a native binding;
- a Tensor value is not a TensorView resource;
- dynamic descriptor values are not part of artifact identity;
- a backend may encode a reflected physical ABI but may not reconstruct one.

The compiler-selected `InterfacePlan` and `TargetBindingPlan` are the
foundations. Runtime now expands them once into immutable typed compute or
graphics prepared plans. Backend pipeline files retain only genuinely native
resolution policy and consume the resulting ordered physical sequence.

## 2. End-to-end data flow

The current flow is:

```text
typed Vernon entry IR
  -> compiler logical reflection
  -> target lowering and target physical-layout reflection
  -> Program StageArtifact endpoints + compiled_abi
  -> strict Program manifest parsing
  -> TargetBindingPlan
  -> StageBindingPlan + ReflectedEntry
  -> PreparedComputeBindingPlan or PreparedGraphicsBindingPlan
  -> PlannedComputeLaunch or PlannedGraphicsInvocation
  -> shared invocation binding materialization
  -> RuntimeCore binding revision
  -> RHI adapter / CPU provider
  -> native API encoding
```

The implemented flow removes backend-local physical expansion:

```text
StageArtifact + resolved Program projections
  -> TargetBindingPlan
  -> PreparedBindingPlan
  -> invocation-specific PreparedBinding values
  -> provider-native encoding
```

The prepared plan is a resolve-time product; provider binding values are
invocation-time products. Neither target code nor reflection is reinterpreted
while commands are encoded.

## 3. Reflection production

### 3.1 Logical reflection

The compiler reflects entry arguments and results from retained typed Vernon
IR. Logical types remain the authority for:

- scalar identity and signedness;
- Tensor rank and static shape;
- Tuple and Struct structure;
- TensorView access and declared shape constraints;
- image dimension and resource role;
- builtin and stage interface identity;
- canonical Value layout and layout hash.

Logical reflection must not be reverse-inferred from LLVM, SPIR-V, PTX, GLSL,
MSL, HLSL, or DXIL. Target lowering may erase language semantics after the
logical model has been retained.

The main implementation is `source/lib/compiler/compiler_reflection.cpp`.
Canonical Program assembly and compiled endpoint projection are implemented in
`source/lib/compiler/compiler_program_stage.cpp`.

The Value ABI implementation in
`source/lib/Dialect/Vernon/IR/VernonValueAbi.cpp` is the layout oracle. It
deliberately exposes two different layout classes:

- `getValueAbiLayout()` produces the language-dtyped canonical Value layout.
  Its hash includes the explicit language leaf dtype, including the distinction
  between `i32` and `u32`;
- `getValueStorageLayout()` produces the signless physical storage layout used
  when aggregate TensorView elements are expanded into storage leaves.

These hashes are not interchangeable. MLIR signless `i32` is not sufficient to
recover a language dtype, and a storage-layout hash must never be compared to a
Program Value layout hash. Reflection of integer aggregate leaves therefore
requires explicit language dtype provenance such as
`vernon.abi_leaf_dtypes`.

### 3.2 Physical-layout reflection

For every compiled argument or result, the compiler emits one or more named
physical profiles. A selected profile becomes an `InterfacePlan`:

- `CpuCall` describes a CPU packed call frame;
- `KernelParameter` describes a direct kernel parameter ABI;
- `ByteTransport` describes a byte-addressed buffer transport;
- `NativeUniform` describes a target-native uniform value.

Each plan contains a recursive `TransportNode`. Nodes record representation,
offset, byte size, alignment, shape, byte strides, and children. This is the
complete compiler-owned description needed to transform canonical Value bytes
into the bytes consumed by target code.

Target reflection pruning in `compiler_reflection.cpp` retains only profiles
legal for the selected target. Representative profiles are:

- `host_value`;
- `cuda_kernel_parameter`;
- `vulkan_std430_storage_buffer`;
- `vulkan_std140_uniform_buffer`;
- `vulkan_push_constant`;
- `opengl_native_uniform`;
- `directx_constant_buffer`;
- `metal_constant_buffer`.

The selected profile is target code ABI. Runtime must not replace it with a
layout derived only from dtype and shape.

### 3.3 Compiled endpoint ABI

`compiledProgramEndpointAbi()` in
`source/lib/compiler/compiler_program_stage.cpp` copies the target facts needed
at deployment into each StageArtifact `compiled_abi` row:

- module, interface kind, and argument index;
- builtin identity;
- source-derived native uniform name;
- value transport;
- descriptor set and binding;
- selected interface plan;
- packed CPU frame offset;
- element layout;
- sampled-image binding provenance.

`source/lib/runtime/program_execution_manifest.cpp` parses this object
fail-closed into `CompiledEndpointAbi`, declared in
`source/lib/runtime/program_execution_manifest.h`. Unknown, missing, empty, or
type-invalid required fields reject the bundle before backend mutation.

The source-derived `uniform_name` is part of this ABI projection. OpenGL must
not recreate it from a Program parameter name: optimizer activity and
SPIRV-Cross naming can otherwise make a valid native uniform appear inactive.

### 3.4 Two reflection products

The repository has two intentionally different reflection products:

1. raw kernel/stage reflection uses `entries[].arguments[]`,
   `entries[].results[]`, and each row's `physical_layouts`; it is parsed by
   `pipeline_metadata.cpp` into `ReflectedEntry`;
2. Program StageArtifact reflection uses portable `endpoints[].abi.bindings`
   plus target-selected `compiled_abi`; it is parsed by
   `program_execution_manifest.cpp`, merged with Program graph projections,
   and only then projected into `TargetBindingPlan`.

They share compiler layout authorities but are not interchangeable JSON
schemas. Feeding a raw entry row to the Program endpoint parser, or treating a
portable endpoint as if it already contained the compiled target ABI, is a
contract error. The final prepared-binding migration may share typed internal
builders, but it must not add a permissive compatibility parser between these
schemas.

## 4. The four binding namespaces

Four numeric namespaces coexist and must stay explicit.

### 4.1 Program boundary slot

The public Program parameter slot identifies a stable logical boundary.
Invocation binds this slot through `VernonProgramArgument`. Names are lookup
and diagnostic conveniences, not transport identity.

### 4.2 Portable endpoint slot

Stage endpoint `abiBindings` assign portable slots to value, resource,
storage-leaf, offset, extent, stride, sampler, and other endpoint semantics.
These slots are stable inside the Program contract.

### 4.3 Provider slot and physical ordinal

The provider slot identifies one entry in a prepared provider binding
sequence. A physical ordinal identifies one actual target function parameter
or descriptor entry. Both must be unique and deterministic. They need not
equal the Program or portable endpoint slot.

The shared prepared-plan builders allocate additional internal provider slots
for storage leaves and TensorView descriptor fields and sort the sequence once.
Backends do not create a second ordinal namespace.

### 4.4 Native location

The compiled target chooses native locations:

- CPU frame byte offset;
- CUDA kernel parameter ordinal;
- Vulkan descriptor set/binding or push-constant offset;
- DirectX root parameter, descriptor, or constant range;
- Metal argument-buffer index/member or buffer/texture/sampler index;
- OpenGL uniform location/name, buffer binding, texture unit, or attribute
  location.

Native locations come from compiled reflection and target resource metadata.
They are never assigned by the Program cooker.

## 5. Resolve-time projection

`buildTargetBindingPlan()` in
`source/lib/runtime/target_binding_plan.cpp` joins:

- the resolved Program Node binding;
- canonical Program Value and Storage metadata;
- portable Stage endpoints and ABI bindings;
- `CompiledEndpointAbi`;
- selected Runtime backend.

Each `TargetBinding` records independent dimensions:

- `ProgramProjection`: supplying Value, optional logical leaf, physical leaf,
  and input/output direction;
- `SourceRepresentation`: whole Value bytes, element stream, TensorView
  descriptor, resource handle, system value, or implicit sampler;
- `TargetCarrier`: inline value, uniform buffer, storage buffer, vertex/index
  buffer, image, sampler, or attachment;
- `CarrierSemantic`: Value, resource, or tape;
- canonical whole-Value and element layouts;
- compiler-selected target transport and native location;
- storage-leaf expansion;
- TensorView descriptor field bindings;
- graphics attribute leaves and sampled-image provenance;
- access, role, view transform, shape, and write footprint.

This separation is essential. A ranked Tensor supplied as canonical whole
Value bytes may use a storage buffer as its target carrier. Reclassifying that
source as an element stream skips compiler-directed packing and is incorrect
for matrix and aggregate layouts.

`buildStageBindingPlan()` currently projects `TargetBindingPlan` into the
older `StageBindingPlan` and `ReflectedEntry` views. `ParameterUse` retains the
stage interface, argument index, native binding, transport, packing mode,
`InterfacePlan`, TensorView descriptor bindings, attribute leaves, and sampled
image bindings.

Compiler reflection shapes use signed extents with `-1` for a dynamic axis.
Runtime contract shapes use unsigned extents with `0` for a dynamic axis.
`shape_layout.cpp` is the required translation boundary. Code must not compare
or copy the two encodings without passing through the typed shape helpers.

## 6. Invocation materialization

### 6.1 Compute

`planComputeInvocation()` in
`source/lib/runtime/compute_launch_planner.cpp` indexes arguments by reflected
compute argument index and creates `PlannedComputeLaunch`.

Materialization performs these operations:

- validate Program argument kind and resource range;
- preserve an existing RHI Tensor resource for true storage;
- preserve host Tensor storage for the CPU provider;
- pack canonical Value bytes with `compileWholeValueCopyPlan()` or
  `compileElementStreamCopyPlan()` and `packTensor()`;
- retain packed bytes in invocation-owned `hostTensorStorage`;
- register host result unpacking in `resultCommits`;
- attach image and sampler resource references;
- validate and expose concrete TensorView offset, extents, and strides;
- retain the command encoder and concrete dispatch grid.

`commitComputeResults()` unpacks host Value results only after successful
execution. Device Storage writes remain device-resident and are published by
the normal resource transaction.

### 6.2 Graphics

`planGraphicsInvocation()` in
`source/lib/runtime/graphics_invocation_planner.cpp` resolves:

- Program arguments by boundary slot;
- color and depth/stencil attachments;
- attachment formats and extents;
- sampled image/sampler pairs by reflected descriptor identity;
- vertex input resources and draw counts;
- viewport, scissor, topology, and generated resolution.

Graphics uniform bytes are packed against each `ParameterUse.interfacePlan`
while the backend binding values are prepared. Attachments are render-scope
controls and resource transitions, not shader parameters.

## 7. Current backend mapping

### 7.1 CPU

`runtime_pipeline_cpu.cpp` and `backend_cpu.cpp` consume compiler-reflected
packed frame offsets and sizes. Storage tensors are host pointers. Values and
results occupy inline packed-frame fields. A CPU TensorView is passed as one
packed descriptor structure, including pointer-sized fields. CPU autodiff adds
typed tape and reduction behavior but does not change public Program binding
identity.

CPU has no graphics rasterizer. Graphics semantics are validated and compiled
for shader reference behavior only where explicitly supported.

### 7.2 CUDA

`runtime_pipeline_cuda.cpp` creates the ordered `cuLaunchKernel` argument
sequence. True storage and ranked Value carriers use storage-buffer provider
bindings; scalar kernel values remain inline. TensorView offset, extent, and
stride fields are currently emitted as separate 64-bit inline values.

`rhi_adapter/adapter_cuda.cpp` converts storage bindings to the five-word
memref descriptor expected by lowered kernels. Packed host Values whose
physical carrier is storage use
`VERNON_RUNTIME_PROVIDER_BINDING_HOST_STORAGE`: the adapter owns aligned device
storage, uploads the compiler-packed bytes, and passes the resulting memref.
It does not reinterpret canonical layout.

CUDA is compute-only.

### 7.3 Vulkan

`runtime_pipeline_vulkan.cpp` maps reflected descriptor sets/bindings to
storage buffers, uniform buffers, images, samplers, and push constants. Static
Value tensors transported through buffers are packed with their
`InterfacePlan`. TensorView descriptor fields are separate 32-bit values at
the reflected bindings.

Graphics uses reflected vertex leaves, descriptor-backed resources, packed
uniforms, generated resolution, and explicit sampler provenance.
`adapter_vulkan.cpp` owns descriptor sets, pipeline layouts, barriers, and
native command encoding.

### 7.4 DirectX 12

`runtime_pipeline_directx12.cpp` maps the provider binding plan to root
constants, constant buffers, SRV/UAV descriptors, vertex buffers, images, and
samplers. The Runtime consumes compiler-generated DXIL and never runs DXC.
TensorView fields are currently 32-bit values.

Graphics resources currently require descriptor set zero and a stricter
single-use parameter shape than the general Stage model. Depth-only graphics
pipelines are valid and must produce a complete PSO even when there are no
color attachments.

### 7.5 Metal

`runtime_pipeline_metal.cpp` resolves portable `(stage, kind, set, binding)`
facts through `NativeResourceSlot` into Metal argument-buffer indices and
member IDs. It maps Values to inline constants or reflected buffers, vertex
inputs to Metal vertex buffer indices, and images/samplers to their native
indices. TensorView descriptor fields are currently separate 32-bit values.

`adapter_metal.mm` owns `MTLArgumentEncoder`, pipeline state, render/compute
encoders, and completion lifetime.

### 7.6 OpenGL and OpenGL ES

`runtime_pipeline_opengl.cpp` requires reflected binding records and descriptor
set zero. It maps buffers, images, samplers, attributes, and native uniforms to
the provider layout. Native uniforms use the reflected `uniform_name` and
physical scalar/vector/matrix shape. Buffered Values are packed from their
`InterfacePlan`. TensorView descriptor fields are currently 32-bit values.

`adapter_opengl.cpp` performs `glUniform*`, `glBindBufferBase`, texture/image
unit, sampler, vertex attribute, framebuffer, and draw encoding after making
the associated context current. OpenGL ES uses the same architecture with its
stricter storage-image and language-version constraints.

## 8. Cross-backend invariants

The following invariants are required before provider pipeline creation:

- every compiled physical parameter has exactly one prepared entry;
- physical ordinals are unique and contiguous;
- provider slots are unique;
- every external entry resolves to exactly one Program projection;
- internal bindings are explicitly marked and have no Program parameter;
- canonical and physical layout hashes agree with the selected
  `InterfacePlan`;
- language Value layout hashes are never compared with signless storage layout
  hashes;
- packed byte size and alignment exactly match the physical root node;
- storage-leaf offset and size fit the carrier stride;
- TensorView rank and descriptor field count match reflection;
- descriptor values use checked conversion to the compiler-reflected width;
- dynamic extents and strides change values, never ordinals;
- access does not weaken while crossing planning layers;
- image dimension, format class, and sampler provenance are preserved;
- runtime-created transient storage is retained through command completion;
- results are committed only after successful completion.

Any missing or contradictory fact rejects the Program. No backend fallback,
default layout, guessed name, inferred binding, or legacy reflection
interpretation is allowed.

## 9. Design assessment

### 9.1 What is already optimal

The following choices should be retained:

- canonical Value layout is backend-independent;
- target physical ABI is compiler-owned and hash-covered;
- Program projection is resolved once, before invocation;
- dynamic TensorView data is excluded from artifact identity;
- resource lifetime is enforced below binding materialization;
- native target APIs remain specialized instead of being forced into a fake
  universal descriptor model;
- OpenGL source names and sampler provenance are reflected rather than
  rediscovered;
- host packing is driven by recursive `InterfacePlan`, not handwritten
  matrix/Struct cases.

These choices minimize semantic duplication without hiding genuine native API
differences.

### 9.2 Remaining type-system work

The binding migration removed backend-local candidate construction, sorting,
and invocation filling. These broader schema limitations remain:

1. `ParameterUse.interfaceKind` and `transport` are strings. The shared
   prepared-plan builder currently interprets them when selecting a physical
   binding kind.
2. CUDA descriptor fields are 64-bit while the SPIR-V-derived backends use
   32-bit fields. A target difference may be valid, but it must be represented
   explicitly in compiler reflection rather than selected from backend identity.
3. Host result commit and backend result publication are represented by
   separate mechanisms rather than one typed prepared output variant.
4. A binding crosses MLIR attributes, logical reflection models, JSON,
   StageArtifact endpoints, `TargetBinding`, `ParameterUse`, and sometimes
   `ReflectedArgument`. No generated typed schema currently guarantees that a
   field addition reaches every representation.
5. Raw kernel reflection and Program StageArtifact reflection are separate
    schemas but use structurally similar field names, increasing the risk of
    accidental cross-consumption.
6. Logical Value and signless storage layouts are distinct C++ values but
    their hashes are represented as ordinary strings, so the type system does
    not prevent an invalid comparison.
7. Compiler and runtime dynamic-shape sentinels differ and rely on explicit
    conversion rather than distinct encoded types.

These are maintenance and correctness risks, not reasons to discard the
layered design.

### 9.3 Optimal end state

Runtime produces immutable compute and graphics prepared plans containing
ordered typed physical entries. Each entry directly contains the applicable
subset of:

- logical endpoint and Program projection;
- physical ordinal and provider slot;
- typed carrier and access;
- typed source selector;
- native location;
- complete physical layout;
- optional storage-leaf projection;
- optional TensorView descriptor field kind and integer width;
- optional image, sampler, attribute, attachment, or system metadata;
- transient ownership and lifetime class;
- output commit policy.

Invocation materialization should produce the corresponding typed
`PreparedBinding` values: inline bytes, resource reference, transient uploaded
storage, descriptor scalar, image view, sampler, attachment, or output
destination.

Backend resolution may validate native limits and cache native objects.
Backend invocation should only iterate the prepared sequence and encode it. It
must not inspect Stage reflection, infer a binding kind from strings, allocate
a new ordinal namespace, or choose a packing plan.

This is the optimal design for Vernon because it centralizes policy while
retaining the native distinctions needed for CPU call frames, CUDA kernel
parameters, descriptor APIs, Metal argument buffers, and OpenGL uniforms.

## 10. Migration and verification

Implemented:

1. typed compute and graphics prepared binding entries;
2. shared physical expansion and structural validation;
3. explicit descriptor scalar widths in prepared compute sources;
4. one ordered compute sequence consumed by CUDA, Vulkan, DirectX 12, Metal,
   OpenGL, and OpenGL ES;
5. one ordered graphics sequence consumed by Vulkan, DirectX 12, Metal,
   OpenGL, and OpenGL ES;
6. CPU packed-frame offsets, result policy, and tape builtins represented in
   the compute plan;
7. deletion of backend-local candidate sorting and duplicated invocation
   filling.

Release acceptance still requires rebuilding all fixtures and passing the
complete C++ and Python suites on each supported platform.

Regression tests must include:

- scalar values beside ranked Value tensors;
- row-major canonical matrices requiring target physical repacking;
- nested Struct/Tuple/Tensor Values;
- dynamic TensorView shapes, offsets, negative/legal strides, and repeated
  invocation with stable ordinals;
- aggregate storage-leaf expansion;
- sampled image/sampler provenance;
- native and spilled graphics uniforms;
- depth-only graphics;
- host-packed outputs and device-resident outputs;
- autodiff tape, replay, and non-power-of-two reductions.

## 11. Implementation index

Compiler reflection and Program aggregation:

- `source/lib/compiler/compiler_reflection.cpp`
- `source/lib/compiler/compiler_program_stage.cpp`
- `source/lib/compiler/compiler_program_abi.cpp`
- `source/lib/compiler/compiler_program_aggregation.cpp`
- `source/lib/compiler/compiler_program_finalization.cpp`

Runtime parsing and resolve-time projection:

- `source/lib/runtime/program_execution_manifest.h`
- `source/lib/runtime/program_execution_manifest.cpp`
- `source/lib/runtime/target_binding_plan.h`
- `source/lib/runtime/target_binding_plan.cpp`
- `source/lib/runtime/stage_binding_plan.h`
- `source/lib/runtime/stage_binding_plan.cpp`
- `source/lib/runtime/pipeline_metadata.h`
- `source/lib/runtime/pipeline_metadata.cpp`

Invocation materialization:

- `source/lib/runtime/compute_launch_planner.h`
- `source/lib/runtime/compute_launch_planner.cpp`
- `source/lib/runtime/graphics_invocation_planner.h`
- `source/lib/runtime/graphics_invocation_planner.cpp`
- `source/lib/runtime/tensor_bridge.h`
- `source/lib/runtime/tensor_bridge.cpp`

Backend and provider mapping:

- `source/lib/runtime/runtime_pipeline_dispatch.cpp`
- `source/lib/runtime/runtime_pipeline_cpu.cpp`
- `source/lib/runtime/runtime_pipeline_cuda.cpp`
- `source/lib/runtime/runtime_pipeline_vulkan.cpp`
- `source/lib/runtime/runtime_pipeline_directx12.cpp`
- `source/lib/runtime/runtime_pipeline_metal.cpp`
- `source/lib/runtime/runtime_pipeline_opengl.cpp`
- `source/lib/runtime/runtime_core.cpp`
- `source/lib/runtime/backend_cpu.cpp`
- `source/lib/runtime/rhi_adapter/adapter_*.cpp`

## 12. Build and binary boundaries

The compiler and Runtime reuse implementation objects without changing their
Windows visibility according to the final library type:

- `VernonCompilerEngine` is an object library containing the compiler
  implementation and directly owning its LLVM, MLIR, and SPIRV-Cross link
  requirements. `VernonDSLCompiler` compiles only the public C façade with
  export visibility. Compiler tests compile that façade with static visibility
  and reuse the engine objects.
- `VernonRuntimeImplementation` contains reusable implementation sources and
  always uses static visibility. `VernonRuntime` compiles
  `VernonRuntime.cpp` and `autodiff/runtime_autodiff.cpp` with export
  visibility; `VernonRuntimeTestHost` compiles the same two public translation
  units with static visibility.
- `VernonRuntimeUtilities` remains a separate static target because both the
  Runtime and Python native module consume it. Static Runtime packages export
  this target with the other transitive static dependencies.
- Internal C++ telemetry and planning types are not production DLL ABI.
  Python-only observations cross a private C/POD callback bridge; test-only
  C++ callers link `VernonRuntimeTestHost`.

The installed Runtime C API and compiler C API retain `dllimport` for ordinary
Windows consumers. Extensible capability data uses caller-sized queries;
legacy by-value capability structures remain unchanged.
