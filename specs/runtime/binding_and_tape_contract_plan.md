# Prepared binding and autodiff tape contract plan

Status: active architecture plan.

This plan completes two related runtime/compiler boundaries:

1. one typed physical binding plan shared by every backend; and
2. a compiler-owned autodiff tape sizing contract.

It does not preserve compatibility with intermediate, unpublished Program
bundles. Existing bundles must be recompiled after the contract is finalized.
The implementation must remain fail-closed and must not add backend-specific
fallbacks for old reflection shapes.

Normative Program fields remain owned by
[`../program/execution_manifest.md`](../program/execution_manifest.md).
This document defines the intended architecture and migration gates.

## 1. Goals

- Make the compiler and RuntimeCore the only authorities for physical ABI
  decisions.
- Materialize every logical endpoint into an immutable, typed physical binding
  sequence before entering a backend.
- Give CUDA, Vulkan, DirectX 12, Metal, OpenGL, OpenGL ES, and CPU the same
  binding identities and ordering rules.
- Remove backend reconstruction of parameter kind, ordering, offsets, strides,
  descriptors, and inline layouts.
- Derive tape capacity requirements from compiled autodiff code and carry them
  through artifacts as ABI, never as a user-authored hint.
- Preserve existing language semantics, resource ownership, publication,
  replay, and cross-platform behavior.

## 2. Binding architecture

The canonical pipeline is:

```text
Stage reflection + resolved Program Value projections
  -> TargetBindingPlan
  -> PreparedBindingPlan
  -> provider/backend encoding
```

### 2.1 TargetBindingPlan

`TargetBindingPlan` owns semantic target decisions:

- which Program Value and projection supplies an endpoint;
- source representation;
- target carrier;
- value versus resource versus tape semantics;
- logical shape and view transform;
- canonical value and element layouts;
- access and resource role;
- portable and native endpoint identity.

It must not depend on live invocation data or allocate provider resources.

### 2.2 PreparedBindingPlan

`PreparedBindingPlan` is the immutable physical ABI projection of one resolved
Stage. It expands every target binding into an explicitly ordered sequence of
typed physical bindings.

Each prepared binding must contain, directly or through typed variants:

- stable logical endpoint identity;
- physical parameter ordinal;
- provider slot and native location;
- physical binding kind;
- access and resource role;
- source projection;
- byte offset, byte size, and required alignment;
- element stride and storage-leaf projection where applicable;
- TensorView descriptor field identity and integer width;
- inline-value transport layout;
- image view, sampler, vertex, index, or attachment metadata;
- runtime-managed versus invocation-supplied ownership;
- lifetime class for temporary materialization.

The physical parameter ordinal is authoritative. A backend must never sort
parameters independently by a different slot namespace.

TensorView expansion must produce typed entries for the resource carrier and
for each required descriptor field. Descriptor offset, extents, and strides
must have one compiler-defined order and width. Aggregate tensors must use
explicit storage leaves and element strides rather than inferred packing.

### 2.3 Invocation materialization

Invocation preparation combines a `PreparedBindingPlan` with bound Values and
creates typed `PreparedBinding` values. This step may:

- validate shapes, ranges, formats, and alignments;
- project an existing provider resource;
- pack canonical host bytes;
- create a transient uniform/storage carrier;
- emit TensorView descriptor scalars;
- retain the owners required through submission completion.

It must not reinterpret Stage reflection. Invalid or incomplete inputs fail
before command encoding.

### 2.4 Backend responsibility

Backends consume the prepared sequence without rebuilding it:

- CUDA emits `cuLaunchKernel` parameters in prepared ordinal order.
- Descriptor-based APIs write descriptors from prepared native locations.
- OpenGL binds prepared buffers, images, uniforms, and vertex attributes.
- CPU builds its call frame from prepared inline/resource entries.

Backends may validate native limits and encode native objects. They may not
change tensor packing, invent descriptor fields, infer storage-leaf offsets,
or reclassify endpoint semantics.

### 2.5 Required invariants

- Every reflected physical ABI parameter is represented exactly once.
- Every prepared ordinal is unique and the ordinals are contiguous.
- Every non-internal prepared binding resolves to one Program projection.
- Resource byte ranges fit the bound resource.
- Storage-leaf offset plus size fits its carrier element stride.
- TensorView descriptor rank and field count match reflection exactly.
- Inline bytes match the canonical interface plan exactly.
- Read/write intent remains unchanged through all planning layers.
- A plan is immutable and reusable across dynamic invocation shapes.
- Dynamic shape values affect materialized descriptor data, not parameter
  identity or ordering.

## 3. Compiler-owned tape sizing

`minimum_tape_stride_bytes` is an ABI requirement, not an optimization hint.
It is the minimum valid initial byte stride for one logical tape lane.
Runtime may grow the stride after a checked overflow and replay, so the value
is a lower bound rather than the final allocation size.

### 3.1 Single authority

The compiler derives the value from the finalized autodiff memory plan and GPU
tape ABI:

```text
required payload
  + invocation/region ABI overhead
  + required padding
  -> alignment rounding
  -> minimum_tape_stride_bytes
```

The computation must use checked arithmetic. The resulting value must cover
every unconditional access performed before overflow can be reported.

The compiler writes the value directly into compiled Stage reflection.
Program aggregation copies the producer contract into the tape Value and
typed TapePlan and verifies that all participating Stage contracts agree.
Runtime consumes the TapePlan value without guessing a default.

### 3.2 Remove frontend plumbing

The finalized design removes `minimum_tape_stride_bytes` as a writable field
from Python `ProgramImplementation`. Python providers must not calculate,
override, or manually transport tape sizing.

`tape_bytes` remains compiler planning telemetry. It is not a deployment ABI
input. It may currently equal the minimum stride payload, but the two concepts
can diverge because the physical ABI can add headers, alignment, and
backend-independent replay metadata.

### 3.3 Runtime behavior

- Initial allocation uses the compiled minimum stride.
- There is no implicit 16-byte, 64-byte, or backend-specific default.
- Every tape address is checked against the lane capacity before access.
- Overflow status reports the required capacity without first performing an
  out-of-bounds load or store.
- Replay grows capacity with checked arithmetic and restores only authoritative
  graph inputs.
- Missing, zero, inconsistent, or overflowing contracts reject the bundle.

## 4. Implementation sequence

1. Define typed `PreparedBindingPlan` and `PreparedBinding` variants beside the
   existing target binding types.
2. Centralize physical expansion in RuntimeCore.
3. Add structural validation for ordinals, slots, descriptor fields, resource
   ranges, and interface layouts.
4. Convert CUDA first because its launch parameter ABI directly exposes
   ordering and width errors.
5. Convert DirectX 12, Vulkan, Metal, OpenGL/OpenGL ES, and CPU without keeping
   dual interpretation paths.
6. Delete backend reflection reconstruction and parallel slot/offset arrays.
7. Move tape sizing emission into native compiler Stage reflection and remove
   Python `ProgramImplementation` tape sizing.
8. Recompile every fixture and bundle under the new contract.
9. Run the complete C++ and Python cross-backend matrix on Windows, macOS, and
   Linux.

Each backend conversion must preserve the same provider SPI and native resource
ownership rules. A backend is complete only after the old reconstruction path
is deleted.

## 5. Remaining issues and evidence required

### 5.1 Windows CPU workgroup autodiff crash

Observed test:

`RuntimeWorkgroupAutodiff.ReplaysBarriersInReverseForEveryLaneGradient`

Current evidence:

- phase 0 reaches coroutine suspension;
- the process crashes while entering/resuming phase 1;
- a CPU thread budget of one still crashes, excluding an ordinary scheduler
  race;
- the corresponding path succeeds on macOS.

The failure is outside `PreparedBindingPlan`. The unresolved boundary is the
custom coroutine frame/handle generated by `VernonCpuAbiWrapper.cpp` and the
Windows LLVM coroutine split/resume ABI.

Required next evidence:

1. capture LLVM IR before and after coroutine splitting for Windows and macOS;
2. compare frame type, alignment, handle storage, calling convention,
   parameter attributes, resume thunk, and destruction path;
3. obtain the first failing Windows stack and faulting address with symbols;
4. identify the invalid frame field or ABI mismatch before changing scheduler
   behavior.

Do not add a Windows-only phase path, disable barriers, or replace coroutine
resumption with a test-specific state machine.

### 5.2 CUDA dynamic shape Program failure

Observed test:

`RuntimeModuleProgramComputeGpuCApi.ReusesLoadedProgramAcrossDynamicShapesAndGrids/cuda`

This non-autodiff failure is expected to be the first validation target for
`PreparedBindingPlan`. Its dynamic TensorView resource and descriptor fields
must retain stable physical ordinals while their materialized shape values
change between invocations.

The plan may fix the failure if its cause is parameter ordering, descriptor
width, slot mapping, offset, or stride. This is not yet proven. Record the
prepared sequence and compare it with the finalized CUDA kernel signature.
If they agree and the illegal address remains, inspect generated address
arithmetic and resource bounds separately.

### 5.3 CUDA non-power-of-two autodiff failure

Observed test:

`RuntimeGpuAutodiffMatrix.ExecutesRegisteredAcceptanceOracles/cuda`

The first observed illegal address occurs in the non-power-of-two reduction
oracle. Later CUDA error 700 results in the same process may be consequences of
the poisoned CUDA context and must not be counted as independent failures.

Re-run this test in a fresh process after the prepared-binding conversion.
If it still fails, investigate independently:

- lane and workgroup index bounds for the non-power-of-two tail;
- tape lane base and stride arithmetic;
- the first unconditional tape access;
- overflow reporting before memory access;
- reduction scratch and cotangent carrier bounds.

`PreparedBindingPlan` is a diagnostic prerequisite, not a predetermined fix
for this failure.

### 5.4 Local LLVM WebAssembly target

The local installed LLVM lacks the WebAssembly target, while the compiler test
expects `wasm32-unknown-emscripten`. Reconfigure and rebuild LLVM with
`WebAssembly` in `LLVM_TARGETS_TO_BUILD`, then verify the installed
`LLVMConfig.cmake`. This is a toolchain installation issue, not a Vernon target
routing fallback.

### 5.5 Standalone Python OpenGL/OpenGL ES context

Standalone Python RHI creation has failed when no valid current GL context is
owned or registered. Confirm whether the current `RuntimeSession` context
ownership implementation resolves this in a fresh process. If not, trace
context construction and current-context lifetime; do not weaken RHI device
creation or silently substitute another backend.

### 5.6 Frontend invocation-index ownership

The frontend smoke module has been rejected because an ordinary device
TensorView store was not proven lane-exclusive. Determine whether
`global_invocation_id` is lost during frontend lowering or omitted by the
injectivity analysis. Fix the canonical index-expression proof. Do not
whitelist the fixture or disable exclusive-write validation.

### 5.7 Verification state

Previous local runs established that the Windows full build completes and that
CUDA aggregate tensors, OpenGL/DirectX aggregate paths, External GL tests, and
the structured scalar autodiff failures were repaired. Because the working
tree and LLVM installation continue to change, the exact remaining CTest
count is not authoritative until a clean rebuild and fresh-process test run.

The completion gate is:

- all C++ and Python tests pass on supported Windows backends;
- CUDA failures are isolated in fresh processes so context poisoning cannot
  hide the first error;
- macOS Metal/Vulkan/OpenGL/OpenGL ES/CPU regression tests pass;
- Linux Vulkan/OpenGL/OpenGL ES/CPU and available CUDA tests pass;
- fixture bundles are regenerated and no legacy compatibility path remains.

## 6. Non-goals

- No compatibility support for unpublished old bundles.
- No backend-specific tape stride constants.
- No reflection parsing inside backend dispatch loops.
- No test relaxation that changes language or memory-safety semantics.
- No platform-specific workaround without a demonstrated platform ABI
  requirement.
