# Runtime design

Status: current Runtime architecture. Programs run as linked schedules over
the HAL ([`architecture_v2.md`](architecture_v2.md)). The Python claims and
residencies of section 1 remain until buffers have one home.

Program semantics and deployment are defined by
[`../program/architecture.md`](../program/architecture.md) and
[`../program/execution_manifest.md`](../program/execution_manifest.md).
Graphics-specific semantics are defined by
[`../program/graphics_execution.md`](../program/graphics_execution.md).

## 1. Ownership layers

Runtime execution is split into two layers:

- **Runtime API** owns Program bundles, executables, manifest resolution,
  Stage binding plans, linked schedules and VJPs, validation diagnostics,
  and public resource references (`VernonRuntime.h`, `VernonSchedule.h`).
- **HAL** owns devices, resources, surfaces, pipelines, recorded command
  buffers, submission, and timeline points (`VernonHal.h`). A HAL device
  owns the runtime context its schedules link on.

A schedule names resources by opaque references (`vernonHalReference*`). The
Runtime API does not own native textures, framebuffers, descriptor heaps,
queues, command buffers, or fences.

### Python RuntimeSession ownership

`RuntimeSession` selects one fully constructed native Runtime. The native
binding stores the HAL device, whose runtime context it uses, in one shared
`RuntimeState`; an internally created OpenGL context is owned by the device's
state rather than by a Python helper. Destruction is ordered Runtime → HAL
device → owned GL context. Runtime, executable, schedule, VJP, argument-record, residual, and
timeline wrappers retain the shared state directly. Teardown therefore does not depend on
Python garbage-collection order or binding-layer parent/child keep-alive
edges. Its
`RuntimeConfiguration` is immutable and canonical: non-OpenGL backends have no
API version, while OpenGL configurations include the effective API version and
the registered external-context identity.

`vd.init(...)` transactionally selects the process default. An identical
canonical configuration returns the existing session without reconstruction.
Concurrent initialization of one configuration joins one construction flight;
different configurations may probe concurrently. A candidate is fully
constructed and capability-probed before a short registry critical section
publishes it. Probe failure leaves the current default unchanged. Replacement
retires the old default but does not invalidate its native objects; Runtime,
device, executable, schedule, resource, and residual owner leases determine
final teardown.

`RuntimeSession` is also a context manager. Its selection uses `ContextVar`, so
nested and concurrent scopes do not mutate the process default. Every public
Program, Kernel, pipeline, cooked-asset, and VJP entry creates one
immutable invocation context for the complete operation. Binding transactions,
resource claims, residency, compilation, and execution receive that context
and never rediscover the process default.

Executable caches are session-local weak-key partitions. A cache miss is
published through a transient per-session, per-key construction flight, so
unrelated sessions and specializations compile concurrently while one native
executable is created for a shared key. Hits read an immutable snapshot.
Epoch-based clearing cancels stale publication without detaching active
waiters. A completed flight remains only until its final participant leaves
and retains no permanent per-key lock. Kernel and pipeline frontend lowering
uses the same publication protocol in a backend-independent semantic cache.
Artifact hashes remain pure semantic/target identities; a
Runtime session identity selects only the native executable partition and
never enters capture, compile, cook, manifest, or artifact identity. Retiring
a session therefore requires neither global cache clearing nor a child
weak-set invalidation pass.

Every mutable Python resource owner has one `ResourceControlBlock`. It owns
the stable monotonic owner identity, owner-local lock, active host/device
claims, authority version, poisoned/unknown state, adapter coherence state,
and session-partitioned native residencies. Buffer coherence records host and
device byte ranges; image coherence records exact aspect/mip/layer
subresources. Tensor, RawBuffer, and Texture views cache only projections and
never own independent authority, dirty, version, or residency state.

A call locks all participating control blocks in stable owner order,
validates every claim, publishes all claims atomically, and releases the locks
before residency preparation, backend calls, or waits. Same-session reads and
non-overlapping projected accesses may coexist. Cross-session overlap fails
before materialization. Host APIs publish a projected claim before
synchronization and retain it until copying or mutation finishes. Buffer
readback downloads only dirty ranges intersecting that claim, so a disjoint
device operation is never covered by an incidental full-allocation readback.

Allocation, migration, upload, download, and wait run through explicit
reservation tickets outside the owner lock. A first allocation or
cross-session migration reserves the full backing before doing I/O. Concurrent
creation for one `(owner, session)` joins one materialization flight; existing
residencies permit non-overlapping tickets to proceed independently. Publishing
a ticket revalidates its exact dirty snapshot and preserves unrelated dirty
ranges or subresources created concurrently.

A linked call writes in place (`architecture_v2.md` section 10). Its
resource transaction resolves the call while its write claims are still
held: `resolve_linked_call` publishes every claimed write and records the
call's timeline point on each written residency, so a later host read waits
for that point. A texture has no readback that waits for a point, so a call
that writes one waits for it at once, and a failed wait poisons every owner
the call writes. The general `resolve_mutations` takes one state per write:
unchanged, committed, written in place, or indeterminate, which poisons the
owner. Host reads and new device claims of a poisoned owner fail until a
complete authoritative host replacement recovers it. Python never infers
mutation from an exception or aggregate success flag.

Fragmented host writes may conservatively expand their claim to a contiguous
span, synchronize device-written gaps under that expanded claim, and issue one
preserving upload; they never touch a gap outside the acquired claim.
Fragmented readback remains a set of exact ranges and uses one native batched
transfer. Texture transfer descriptors carry exact box, mip, array layer, and
aspect. A residency operation submits all selected ranges or subresources as
one native batch; cube faces and depth/stencil planes are not widened in
Python.

The pullback of a linked VJP call is its backward schedule over the
residuals the forward returned. Residuals point into the memory of the
call's inputs, so an input written while its pullback lives first leaves the
memory the pullback reads to a snapshot, which the backward reads instead.
No Python resource claim survives the forward call. A
backward call claims its cotangents and gradients like any other call; one
whose values do not pack natively, or that takes a carrier, runs through the
general pullback wrapper over the same residuals.

Mutable resources retain one residency per session identity, allowing
authority to migrate without discarding live native owners. Samplers remain
immutable session-local residencies and need no mutable access claims.

## 2. Canonical Program lifecycle

The only deployment lifecycle is:

```text
manifest bytes
  -> VernonProgramBundle
  -> resolve typed specialization variant
  -> VernonProgramExecutable
  -> link: VernonLinkedSchedule or VernonScheduleVjp
  -> write the argument record, starting from the record template
  -> vernonScheduleCall (or VjpForward, VjpBackward) -> timeline point
  -> vernonScheduleWait(point)
```

`VernonProgramBundle` is immutable deployment data.
`VernonProgramExecutable` owns one immutable `ResolvedExecutionPlan`. Linking
plans the executable once into a schedule (`architecture_v2.md` section 4):
its slots, fields and the extents each slot must have describe the argument
record, whose bytes hold every argument, control and grid of one call. A call
writes nothing but that record. The caller checks its arguments against the
slots; the call checks only extents and the grid, and returns a point on the
device timeline once the work is submitted. Calls of one schedule run one at
a time. Waiting for a point waits for every earlier submission of the device.

Runtime has no pipeline-bundle compatibility loader, profile executor, direct
Stage asset submit path, unbound Program execution path, or per-call binding
state. A schedule linked from a ProgramGraph takes only the exported
parameters of its executable; linking maps them to the nodes' slots, which
`vernonScheduleNodeSlot` and `vernonScheduleSlotBoundary` report. A
single-kernel schedule is linked from a Stage directly.

## 3. Resolve boundary

Resolution has two explicit results:

1. `ResolvedProgram` validates logical Program structure, Value SSA, Storage
   versions, controls, Stage contracts, boundaries, and derived dependencies.
2. `ResolvedExecutionPlan` selects exact physical Stage implementations,
   endpoint projections, carriers, residency, transfers, hazards, backend
   barriers, graphics scopes, and residual plans.

Linking consumes the physical plan. The linker and the HAL devices do not
rediscover policy from names, carrier shapes, resource aliases, or sentinel
values.

The resolver validates, before loading target code:

- Program and compiler contract versions;
- typed specialization key selection;
- target identity and runtime requirements;
- artifact size, digest, format, and entry point;
- reflection and Program endpoint compatibility;
- device/context capabilities;
- workgroup, graphics, resource, and ABI limits.

A `type` key entry is selected by bytewise identity comparison; Runtime never
interprets the identity as a module path. The C++ wrapper sets it through
`ProgramVariant::setType` rather than a `set` overload, because a
`set(name, "literal")` call would bind the `const char *` to the `bool`
overload.

## 4. Binding and materialization

Public boundary reflection assigns stable slots. Binding uses
`VernonProgramArgument`; names are diagnostics and lookup conveniences, not
transport identity.

Program Values and Storages are canonical logical state. Linking applies the
resolved endpoint projections once: each node slot reads its argument's
record bytes or an internal allocation of the schedule, packing and unpacking
physical aggregate carriers without changing canonical Value layout. Recorded
graphics draws still materialize their Stage-local arguments through a
`MaterializedNodeFrame` until draws are data (`architecture_v2.md` section 11).

TensorView bindings carry:

- owner/resource identity;
- element layout and access;
- rank and concrete shape;
- signed byte strides;
- byte offset and extent.

Dynamic shape, stride, offset, byte length, and compute grid are call data,
written into the argument record. They are not artifact specialization inputs. Zero values are never used
as “infer this later” sentinels.

## 5. Transfers and residency

A linked schedule executes only transfers present in the resolved plan:

- host upload;
- device-to-device buffer or image copy;
- host readback;
- provider-native transition.

Transfers are passes of the schedule, ordered with its Stage executions by
the same hazards. Compatible intermediate Values remain device-resident. Runtime does
not synchronously read a device result to host merely to feed a later device
Node.

A Value's residency at a Node follows the Stage parameter, not only its
target carrier. By-value parameters (`hostProjected`: uniforms, values,
results and native uniforms) are packed by the Stage from host memory, even
when their carrier is a uniform or storage buffer, because the HAL device
places those bytes in a push constant, a native uniform, or a buffer it
fills, such as its recording's arena. Such a Value therefore needs host
residency there, and a host argument gets no device copy. Only resource
parameters require device residency. Linking uses the same
predicate, so the plan never uploads a Value that the Stage then reads from
the host.

Checked arithmetic is required for every address, offset, extent, stride,
shape product, and transfer range.

## 6. Publication

A linked schedule writes every output in place and rolls nothing back
(`architecture_v2.md` sections 9 and 10): a failed check writes nothing, and
a failed submission poisons what it writes. The manifest's per-boundary
`publication` (`commit_after_success` or `in_place`) is still emitted but
not read; removing it changes the manifest version.

The compiler marks an output in place when its borrowed Storage is written
only by render-pass attachments. Staging those attachments would cost every
pass a per-call image, a full copy and a host wait before the copy, which
keeps frames from pipelining. The price is that a failed pass leaves its
attachments indeterminate, not unchanged.

## 7. Linked VJPs

`vernonScheduleLinkVjp` links a resolved Program's forward and backward
graphs as two schedules (`architecture_v2.md` section 13). A forward call
returns `VernonScheduleResiduals`, which own the Values the backward reads
other than the forward arguments, and nothing else: no executable state,
scratch or command buffer. The forward arguments the backward reads stay
unchanged until it runs. A backward call takes the residuals plus a record of
cotangents and gradient destinations, and overwrites the gradients. A
composed ProgramGraph's backward runs its nodes' backward graphs in reverse,
each node's input gradient feeding the cotangent of the output it reads.
Residual and checkpoint
selection is planned once over the resolved forward graph when the VJP is
linked. Derivative leaf/group reflection is metadata, not another execution
ABI.

A linked VJP's GPU autodiff tape is sized optimistically and checked at the
pullback (`architecture_v2.md` section 16.3): the forward never waits for the
tape's status, the first backward call reads it, and an overflow reruns the
forward with the reported stride into scratch outputs and a block of its own,
until every lane fits, since a lane reports only what it needed when it stopped.
Chosen over a wait in the forward because the backward waits for the forward
anyway, so a step that fits its tape costs no stall; the VJP keeps the largest
stride it has seen, so a step overflows at most once per growth.

A linked VJP with a memory budget checkpoints at link time
(`architecture_v2.md` section 16.2). Each forward call keeps the residuals of
its last nodes that fit the budget. Its pullback reruns the prefix of forward
nodes it did not keep, which always includes every node that records a tape.
This was chosen over per-residual selection because one truncated forward
schedule per prefix length is all the link adds, and the backward schedule
stays the same.

## 8. Schedule execution

A linked schedule owns its hazards, barriers, render-scope formation and
fusion, transfers, and the recorded command buffers of its passes; recording
and the call path follow `architecture_v2.md` section 7. Its graph is not
serialized as Program topology and is not a public Python authoring API.
Debug groups and scope timings follow [`gpu_profiling.md`](gpu_profiling.md)
sections 2 and 3: a timed scope is one render scope or one compute Node, and
a fused scope's label joins its members' labels with `+`.

## 9. CPU execution

CPU cooking emits:

- one canonical `*.program.json`;
- one relocatable `.o` or `.obj` per CPU module.

A CPU Program deploys like a PTX or GLSL Program: manifest plus modules, with
no generated sources. The compiler emits, into the object only (never the JIT
module), a private `VernonStaticCpuEntry` table and an `llvm.global_ctors`
constructor that passes it to `vernonRuntimeRegisterStaticCpuEntries`.
Applications link the objects and load the manifest; Runtime resolves the
manifest `symbol` through the static registry and does not parse or relocate
object files. A matching `llvm.global_dtors` destructor passes the same table
to `vernonRuntimeUnregisterStaticCpuEntries`. Registrations are reference
counted per symbol and pointer: copies folded within one image share a
pointer, while copies in separate shared libraries stay distinct, so unloading
one library withdraws only its own definitions and lookups fall back to the
oldest live one. The registry is created on first use and intentionally
leaked, so constructors may run before any other Runtime static initializer
and destructors may run after Runtime statics are destroyed.

Host module assets ([`../host/design.md`](../host/design.md#build-integration))
register the same way, through `vernonRuntimeRegisterStaticHostModule` and
`vernonRuntimeUnregisterStaticHostModule`, with `{asset id, VhModule table}`
records. That registry is also created on first use, leaked, and reference
counted per id and table. Its lookup, `vernonRuntimeFindHostModule`, returns
the newest live registration rather than the oldest. A table registered at
run time, such as a Python-mode table during development or a reloaded one,
must shadow the linked object, and withdrawing it restores the earlier one.

Chosen over cooked registration `.c`/`.h` glue because that glue had to be
compiled per Program and invoked by the application, which no other backend
requires. The cost is the usual constructor rule: objects in a static library
must be linked whole-archive, or the linker drops them with their constructor.

CMake projects cook through one helper, `vernon_add_program_asset` in
`cmake/VernonProgramAsset.cmake`. The VernonRuntime CMake project includes it,
and the wheel ships it with the Runtime sources. It cooks at build time into
the binary tree and returns an INTERFACE target. For CPU, that target links the
cooked objects as one archive through `$<LINK_LIBRARY:WHOLE_ARCHIVE,...>`.
Cooked artifacts are specific to one target (CPU objects are native code for
one triple), so repositories keep only DSL sources. The engine, the
external-engine example, the benchmarks and the Runtime test fixtures all use
this helper instead of per-project cook rules. Including it also defines
`Vernon::HostHeaders`, an INTERFACE target of the `vernon_host/` headers
(`reflect.h`, `bindgen.hpp`), from a checkout or the wheel; a library that
implements externs, such as the engine's C API, links it for `VhList`,
`VhAllocator` and `VhType` without linking the Runtime.

Interactive CPU execution may use compiler-owned LLJIT. LLVM IR and ORC state
are not persistent deployment formats.

### Range-phase scheduler

CPU compute and CPU VJP use one range-phase entry ABI. A bounded worker pool
executes contiguous lane ranges rather than creating one task or OS thread per
lane.

Each workgroup owns its shared arena, lane-private state, current phase,
barrier site, outstanding range count, and latched diagnostic. A lowered
barrier yields a range. A phase advances only when all ranges complete or
yield at the same barrier site.

Mixed completion/yield, mismatched barrier sites, allocation errors, invalid
outcomes, and entry exceptions fail the dispatch and release group state.
Workers never block at a workgroup barrier.

The web profile uses the same scheduler with calling-thread execution and no
worker pool.

## 10. GPU backend loading

The HAL backends discover the CUDA and Vulkan loaders at run time
(`hal/cuda/cuda_driver.cpp`, `hal/vulkan/vulkan_driver.cpp`). Vulkan headers
are compile-time dependencies only. Vulkan loader discovery uses the standard
Khronos loader and preserves ordinary ICD discovery.

Backend inclusion at build time does not imply runtime availability. Runtime
availability requires a loadable API, usable device/context, and all artifact
requirements.

The internal backend dispatch table separates its required core from optional
capabilities. Required callbacks are ordinary function pointers and must be
present for every included backend. A callback is wrapped in `Option` only when
a valid backend can lack that operation, such as image, sampler, rendering, or
timeline support. Absence maps to `Unsupported`; failure after invoking a
present callback is a typed `Result` error. Dispatch registration names each
field explicitly and does not use positional null sentinels.

Compute and graphics resolve are shared by every backend: a backend supplies
only its diagnostic name and artifact kind, and whether a device receives the
artifact's binary or source follows from which one the Stage loader filled.
Resolve validates the artifact and prepares the Stage's binding plan and group
layouts, and creates no native object. The HAL device that creates a linked
schedule creates its pipelines and lowers its group layouts then
(`architecture_v2.md` section 16.3), so a resolve that fails halfway leaks
nothing. Sharing keeps the backends from drifting: they report identical
resolve diagnostics.

### CUDA

CUDA is compute-only. It loads PTX through the Driver API and validates
address size, PTX requirements, compute capability, workgroup limits, and
required features before pipeline creation.

Every operation runs on one stream, so stream order is submission order and
device uploads are queue-ordered without barriers. Kernel launches enter the
stream as they are recorded; a submission records an event, which completion
polls (`cuEventQuery`) and waits on (`cuEventSynchronize`) and which covers
every launch enqueued before it. An upload's pinned staging is reused once an
event recorded after its copy completes. Each call makes the primary context
current on the calling thread, so a CUDA device accepts calls from any thread.

### Vulkan

Vulkan supports compute and offscreen graphics. Instance and device extensions
are enabled only when advertised. SPIR-V modules, descriptor layouts, pipeline
layouts, graphics/compute pipelines, barriers, and resource transfers are
owned by the HAL device (`hal/vulkan/`).

Submissions do not wait. Each transfer recording owns a command frame (pool,
command buffer, fence, and the staging chunks it bumps through) that is reused
once the device's submission serial passes the serial the frame was submitted
under; at most 64 frames are in flight. A retired frame returns its staging
chunks to a pool the device's frames share (`hal/vulkan/staging_pool.h`), so
staging memory follows the data in flight rather than the frame count; a
frame that outgrows its chunk takes the smallest free chunk that fits or a new
one twice as large. Every recording begins with a global memory barrier, so
it observes all work submitted before it without a host wait and individual
transfers need no barriers of their own. Native objects the caller destroys
are destroyed once the queue reaches the serial submitted before the destroy.

Descriptor sets belong to a recording's frame slot, which keeps its
descriptor pool, sized up to powers of two, while the slot is pooled
(`architecture_v2.md` section 16.3).

Native Vulkan rasterizes normalized device y = -1 to image row 0, the
opposite of the shared convention (`graphics_execution.md` section 6). The
HAL device therefore sets a negative-height viewport (origin y + height, height
-height), core since Vulkan 1.1, which the backend requires. Front face and
scissor pass through unchanged, because the flipped viewport keeps winding and
framebuffer rows in the shared convention. Negating `gl_Position.y` in the
shader, as on GL, was rejected: GL gets the negation from SPIRV-Cross, but
Vulkan consumes SPIR-V directly and would need a SPIR-V rewrite of every
vertex stage, while the viewport is dynamic state the device already sets.

### DirectX 12

The DirectX 12 HAL device (`hal/d3d12/`) runs on Windows. The compiler cooks
DXIL and a deployment never invokes DXC; the rules below fix the artifacts the
device consumes.

Inline uniforms lower to the same SPIR-V push-constant block as on Vulkan, in
the `vulkan_push_constant` layout. SPIRV-Cross emits that block as an explicit
`cbuffer` at b0 space0 with `packoffset` taken from the SPIR-V member offsets
(`kDirectXRootConstantRegister` in `runtime/interface_names.h`). The device
places each value at its running offset aligned to the reflected element
alignment, sorted by entry argument order, the rule the SPIR-V lowering uses,
so every byte offset is decided by the compiler-owned layout. The root
signature binds that register as one root constant buffer view per graphics
stage, pointing at 256-byte blocks a call composes in the recording's upload
arena. Root constants were rejected: their values are recorded into the command
list, so every value change would force a new recording, and two stages of
values could exceed the 64-DWORD root signature budget.
Loose HLSL globals were rejected: DXC packs them into an implicit `$Globals`
buffer with its own packing rules and auto-assigns its register, so the
runtime would have to mirror HLSL packing and the binding would depend on
which registers were free. Push-constant matrix columns use a 16-byte stride on
every target, because an HLSL `cbuffer` starts each matrix column at a register
and `packoffset` cannot express the tighter std430 stride of 2-row matrices.

Descriptor set N uses register space N + 1
(`kDirectXDescriptorSetRegisterSpaceOffset`): the compiler shifts every
resource before emitting HLSL, and the device shifts every descriptor range,
so space 0 holds only the root constants. Otherwise the uniform block planned
at set 0 binding 0 would also take b0 space0. A constant-buffer block takes its
layout from the std140 SPIR-V interface decorations, as on every other backend.

Compute and graphics root signatures have the same descriptor tables: a
CBV/SRV/UAV table and a sampler table, each in layout entry order. They
differ only in inline values, which are read storage buffers in compute and
per-stage root constant buffer views in graphics. A shader-visible sampler
heap holds at most 2048 descriptors, so a recording shares sampler tables by
content: binding sets that differ only in resources reuse one table rather
than each writing its own, which a frame of a few hundred material draws would
otherwise exhaust.

SPIRV-Cross emits a struct or array value as an HLSL initializer list, which is
valid only in a declaration. A value whose temporary it hoists out of the
block that builds it (one read after a branch in a loop body) becomes an
assignment from an initializer list, which DXC rejects; SPIRV-Cross has no
fix upstream. The compiler therefore reads each member of a composite built
from its members directly (`ForwardCompositeMembersPass` in
`compiler_spirv.cpp`, before serialization), so a struct built only to be read,
as a helper's returned struct is, never reaches SPIRV-Cross. Patching
SPIRV-Cross was rejected because the patch would have to be carried across
every upgrade. `python/tests/loop_struct_shader.py` reproduces the shape.

Limit: a struct or array that stays a value still reaches SPIRV-Cross, so the
DirectX target fails to cook when SPIRV-Cross hoists one. That is a struct
carried from one loop iteration to the next, merged from the arms of a
branch, stored to a function variable, or passed to a function that is not
inlined, when it is read after a branch in the block that builds it. The
failure is a DXC "expected expression" error at cook time, never wrong
output, and the other targets are unaffected. A shader avoids it by carrying
the struct's members as separate values. Splitting such structs into
per-member values in the compiler, through loop-carried values and branch
results, would remove the limit.

`SV_VertexID` and `SV_InstanceID` exclude a draw's base vertex and first
instance, which Vulkan's indices include. SPIRV-Cross
(`support_nonzero_base_vertex_base_instance`) adds them from an auxiliary
constant buffer at b1 space0 (`kDirectXDrawOffsetsRegister`), which the
graphics root signature declares as two root constants {base vertex, first
instance} the device sets before each draw. Indirect draws use a command
signature that sets the same constants and then draws, over 28-byte arguments
(the two offsets and `D3D12_DRAW_INDEXED_ARGUMENTS`): a host stream's records
are expanded into the arena by the call, a device stream's by a built-in
compute shader (`hal/d3d12/d3d12_draw_stream.hlsl`) into the recording's
device-memory argument buffer before the render scope. Shader model 6.8's
`SV_StartVertexLocation` would avoid the expansion but needs drivers the
baseline does not require.

Host-visible buffers are in custom heaps in memory pool L0 (write-combined for
upload, write-back for readback) rather than the upload and readback heaps, whose buffers cannot be
UAVs nor leave their one state; every buffer therefore begins each submission
in `COMMON`, to which buffers decay, and reads its mapping once the fence
passes. Raw buffer views start at 16-byte offsets
(`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`). An image whose views reinterpret it is
created in its family's typeless format, so 8-bit RGBA has UNORM and sRGB views
and a sampled depth image has a color view. Depth-stencil transfers copy whole
subresources, as D3D12 requires. Mipmap generation and draw-stream expansion
are compute shaders compiled by DXC at build time and embedded; a build
without DXC generates no mipmaps and draws no device streams. A render scope binds a null render target
view at color locations without an attachment, so the others keep their
locations. Sampler anisotropy above 1 selects the anisotropic filter.
Presentation needs a window station with a visible desktop, which services and
SSH sessions lack.

A storage buffer whose layout access has no write is a raw SRV
(`ByteAddressBuffer`, a `t` register); only written ones are raw UAVs. Reads
then need no UAV state, so a buffer one dispatch writes and the next reads
takes an ordinary state transition, and a recording's packed values can be
read in place from upload staging, which cannot be a UAV. The compiler matches
the layout access: VernonToGPU records each texture's and storage argument's
access (`vernon.resource_access`; by-value storage is `read`), and since the
generic SPIR-V ABI lowering builds globals without argument attributes, the
read bindings ride on the entry function until a pass after that lowering
gives their globals a block type whose members are `NonWritable`, as glslang
emits `readonly` blocks. A `NonWritable` variable decoration is not enough:
the lowering shares one block type between equal leaves, and a Vulkan driver
then served stale reads of the leaves a kernel wrote. Graphics storage reads
are `NonWritable` blocks too.

### Metal

The Metal HAL device (`source/lib/hal/metal`, Objective-C++ under ARC) needs
macOS 12 or iOS 15 and Tier 2 argument buffers, and compiles the cooked MSL 2.4
with fast math off, so floating-point results match the SPIR-V targets.

- Bindings: set g is one argument buffer at `[[buffer(g)]]`, encoded per
  binding slot through the function's own argument encoder
  (`newArgumentEncoderWithBufferIndex:`), so member offsets are the
  compiler's rather than a layout the device reconstructs. Each stage of a
  graphics pipeline has its own encoders, because SPIRV-Cross lays out each
  stage's argument buffer from the members that stage reads. Push constants
  are `setBytes` at buffer 15, plain uniforms direct buffers from 8, and
  vertex binding b is buffer 16 + b.
- Residency: argument buffers reference resources indirectly, so every
  dispatch calls `useResource` for what its sets reference, and every render
  scope does once for the union of its draws, with the stages that read each
  resource on macOS 13 and iOS 16.
- Ordering: resources use Metal's automatic hazard tracking, so recorded
  barriers encode nothing. A submit encodes a fresh `MTLCommandBuffer` from
  the recorded command list, since Metal command buffers are single-use.
- Render scopes clear whole attachments only, because a load action has no
  region; a cleared scope over part of an attachment fails to record.
  Pipeline variants are keyed by the scope's attachment formats, because a
  pipeline whose formats differ from the pass is a Metal validation error.
- Device draw streams: a counted stream expands through a built-in compute
  kernel that zeroes the instance count of records past the count; an
  uncounted stream draws each record indirectly from the record buffer,
  whose 20-byte record is `MTLDrawIndexedPrimitivesIndirectArguments`.
- Transfers: macOS copies and fills whole 4-byte words, so the device pads
  uploads and readbacks to 4 bytes. Depth-stencil writes cover whole mip
  levels.
- Present copies the RGBA8 ring image into the layer's BGRA8 drawable with a
  fullscreen-triangle render pass, because `CAMetalLayer` has no RGBA8 format.
- The device reports no submission timings yet.

### OpenGL and OpenGL ES

OpenGL and OpenGL ES require a host-owned current context. The owner supplies
`make_current` and `get_proc_address` callbacks. Runtime does not link a
window-system library.

Each backend accepts only artifacts for its matching GLSL profile. Compute
requires OpenGL 4.3+ or OpenGL ES 3.1+. Capabilities that depend on API version
are queried after context creation.

Graphics GLSL targets 3.30, which cannot declare `layout(binding)` on a uniform
block, so every block links at binding point 0. The compiler names each GLSL
uniform block from its SPIR-V binding (`openGLUniformBlockName`, shared through
`runtime/interface_names.h`). After linking, the HAL device looks up each
uniform-buffer entry's block by its planned binding and calls
`glUniformBlockBinding`. Pipeline creation fails if any active block is left
unassigned, because that block would silently read binding point 0. Names come
from the binding rather than the source name because source names are
sanitized, uniqued per stage, or generated (TensorView metadata carriers). The
`GL_ARB_shading_language_420pack` alternative was rejected because macOS
OpenGL 4.1 does not expose it.

GL rasterizes normalized device y = +1 to the last image row, as native
Vulkan does (`graphics_execution.md` section 6). Every GL vertex stage therefore
negates `gl_Position.y` (SPIRV-Cross `flip_vert_y`), and the HAL device inverts
`glFrontFace`, because the negation mirrors window-space winding. Window
coordinates then count rows from row 0, so `gl_FragCoord`, `glViewport` and
`glScissor` need no change. wgpu and Dawn emulate the same convention on GL.
`glClipControl(GL_UPPER_LEFT)` was rejected because it requires OpenGL 4.5,
which macOS OpenGL 4.1 and OpenGL ES 3.1 lack.

GL also clips depth to [-w, w] where the other targets clip to [0, w]. Every
GL vertex stage therefore rewrites `gl_Position.z` to `2z - w` (SPIRV-Cross
`fixup_clipspace`), so the window transform stores z / w as elsewhere. The
same OpenGL 4.5 requirement rules out `glClipControl(GL_ZERO_TO_ONE)`; the
remap gives up the precision of reversed depth on GL, which the [-1, 1]
range lost anyway.

Writes and clears to an `srgb` color attachment store encoded values on
every backend. OpenGL ES always encodes, but desktop GL encodes only while
`GL_FRAMEBUFFER_SRGB` is enabled, so the desktop HAL device enables it for
each render scope. It is not left enabled across the device:
2D image copies and present are framebuffer blits, which would then decode and
re-encode `srgb` texels instead of copying them unchanged, and the host's own
draws would see the state. Every blit also disables it, because a scope whose
native begin fails is never ended.

GL surfaces (`presentable_surfaces.md` section 5) map acquire to handing out
the next HAL image of the configuration, after comparing `drawable_size`
with the configured extent, and present to: make the device's context current
on the window, set the swap interval, blit the texture into framebuffer 0 with
its rows reversed, swap, and make the device's own target current again. The
rows are reversed because the negated vertex y above puts image row 0 at the
top while framebuffer 0 counts rows from the bottom. The blit goes through a
framebuffer object created and deleted per present, because the cached
framebuffers belong to rendering scopes and per-present names cost nothing
next to the blit.

On every backend the HAL keeps the serials of acquired images whose contents
are still undefined in one process-wide set (`hal/native_images.h`). Acquire
adds the image; a host write, or the submission of a recording that writes
it, removes it, and so does destroying the image. A host read, or the
submission of a recording that reads the image, fails with
`VERNON_HAL_INVALID_ARGUMENT` while it is in the set. Removal happens at
submission rather than at recording, so a read cannot pass on a write that has
not been submitted and may never be. A flag that is clear while the set is
empty keeps the check off the hot path of processes that never acquire.

The GL draw state that Runtime sets on a host context, including the inverted
front face, persists after Runtime returns. The host re-establishes any state
its own draws depend on, as with Skia and Qt RHI external commands. Runtime
in turn reapplies its own state whenever the host may have changed it.
Restoring host state after each draw was rejected because it would need a
query of every state Runtime touches.

The context owner outlives the HAL device and Runtime. All GL allocation,
transfer, deletion, and dispatch operations first make the associated
context current. A context is current on one thread at a time, so a GL device
is bound to the thread that owns its context; other backends advertise
concurrent submission instead. A submission inserts a fence sync, which
completion polls and waits on; GL itself orders later commands after it and
defers deleting objects that pending commands use.

The packed uniform bytes of a recording, and the records of its host draw
streams, go to the recording's arena rather than a buffer per group. A call
uploads the arena into one of three buffers the recording uses in turn, and
only when the bytes differ from what that buffer holds, so a call rarely
rewrites bytes the previous calls' commands read; GL orders the write after
them either way, so no call waits. Every draw uses one vertex array per
device, made when the device is created, so calls create no GL objects of
their own.

GL reports invalid transfers only through `glGetError`, leaving the
destination untouched, so downloads, and mipmap generation, which reports an
unsupported format the same way, check their own errors on every device; an
unchecked download error would otherwise surface as zeroed data. Uploads run
inside an error scope on a validation device and otherwise fail the next
completion ([`hot_path_diagnostics.md`](hot_path_diagnostics.md) section 2). Core OpenGL ES cannot
`glReadPixels` depth or stencil, so on OpenGL ES 3.1+ a compute pass copies
one aspect of a depth image region into a buffer, one 32-bit word per texel:
`texelFetch` for 2D and 2D-array images, and for cube images a `textureLod` of
the texel's center through a nearest-mipmap-nearest sampler, which selects
exactly that texel and level; stencil is read through
`GL_DEPTH_STENCIL_TEXTURE_MODE`. Downloads map that buffer; the same copy is
the OpenGL ES depth and stencil path of the image-to-buffer copy command that
asynchronous readbacks need. Dawn
does the same on OpenGL ES, where its `UseBlitForDepth32FloatTextureToBufferCopy`
and `UseBlitForStencilTextureToBufferCopy` toggles replace those copies with
compute blits. Declaring the downloads unsupported was
rejected because it would put an OpenGL ES driver limit into the language's
host access to depth textures. GL has no transfer that writes one aspect of a
packed depth-stencil texture, so such uploads are unsupported there.

## 11. Runtime requirements

Every Stage artifact carries hash-covered `runtime_requirements` derived from
the emitted target code and reflection. Requirements describe the minimum
execution environment; target options describe compilation inputs.

Runtime validates requirements before code loading. Examples include:

- CPU target triple, object format, address size, and calling ABI;
- CUDA PTX version, address size, compute capability, and workgroup limits;
- Vulkan API/SPIR-V versions, limits, and features;
- DirectX API, feature level, Shader Model, root signature, and limits;
- Metal platform, MSL version, minimum OS, and features;
- OpenGL/OpenGL ES API version, profile, and required extensions.

Compiler target availability and runtime device availability are independent.

## 12. Images and graphics

Program image bindings are image-view references. Provider descriptor queries
return immutable parent allocation metadata, selected view metadata, hazard
identity, and ownership kind.

Graphics Nodes bind normalized render-pass, draw, dynamic-state, and shader
resource inputs. Color/depth attachments are resource transitions rather than
shader arguments. Runtime plans image subresource hazards by parent image plus
aspect, mip, and layer ranges.

Compiler-generated resolution and implicit Sampler endpoints are system
endpoints. They are resolved from validated graphics state and do not become
public Program parameters.

Every sampled Texture binds as a combined image-sampler, so a Sampler has no
descriptor of its own, in compute and graphics alike. The prepared binding
plan gives a Sampler one layout entry at the native descriptor of each
Texture it samples (the reflected relation is in the
[compiler design](../compiler/design.md)), and the HAL device merges each pair
at that descriptor. Vulkan writes one combined descriptor. D3D12 places the
sampler in a sampler table at the Texture's register. OpenGL binds the
sampler object to the Texture's unit. Metal resolves the sampler member that
reflection records beside the Texture. OpenGL compute GLSL (4.30, and ES
3.10 as for every OpenGL ES stage) declares each unit through
`layout(binding)`, so compute assigns no unit uniform, unlike desktop
graphics GLSL 3.30.

Render scopes:

- A render scope belongs to a pass fragment: the HAL device begins and ends
  it when it records the pass (`architecture_v2.md` section 11). Draws carry
  no attachments, operations, or render area; a draw renders into its scope.
  A scope without draws still begins and ends, so its clears and stores
  apply; that is how a graphics Node without draws clears.
- A scope declares the images its draws sample. `checkRenderScope`
  (`hal/native_images.h`) rejects a read without sampled usage or
  overlapping an attachment (same parent, overlapping mips, layers and
  aspects), and Runtime reports a Node sampling what it renders to. The
  device makes the reads shader-readable before it begins the native scope.
  This is required because a Vulkan render pass cannot change layouts, and
  the first draw of a fused scope cannot transition the images that later
  draws sample. Splitting the native scope at such a draw was rejected:
  Vulkan fixes store operations when the scope begins, so the first half
  would have to store contents that the scope's DISCARD stores drop.
- Vulkan tracks image layouts per subresource on the image only. A view
  points to its image and covers the subresources its descriptor selects,
  so transitions through a view and through its image, or through two views
  of one image, see the same state.
- Vulkan image layouts follow submission order, because independent
  recordings may be submitted in another order than they were recorded in. A
  recording keeps, per subresource it uses, the layout it needs first and the
  layout it leaves, and never changes the image's layouts. A subresource's
  first use records no barrier. Submission, under the device lock, records
  the transitions from the image's layouts to those first layouts into a
  prologue command buffer of the same frame, submits the prologue and the
  recording in one batch, and then sets the image's layouts to the ones the
  recording leaves. Abandoning a recording has nothing to restore. This is
  the Dawn and wgpu model; a journal with rollback was rejected because it
  cannot order two recordings that are both still open.
- The HAL has no borrowed Vulkan device, whose host would execute the
  commands: a linked schedule submits its own recordings.
- Scope objects do not depend on a pipeline. Vulkan without dynamic rendering
  caches render passes in the HAL device, and a recording makes the
  framebuffers of its scopes. OpenGL caches one
  complete framebuffer per attachment set (texture names plus the stencil
  flag), attached and completeness-checked when the set is first rendered, so
  a scope opening only binds it. Re-attaching one shared framebuffer whenever
  the set changed cost about 6 µs per scope, mostly in
  `glCheckFramebufferStatus`, because frames alternate their targets.
  Destroying an image deletes the framebuffers that attach it, imported
  images included: GL may reuse the texture name, and an unbound framebuffer
  keeps the texture storage alive. The HAL device makes the framebuffer when
  a schedule is built, and the builder tracks the draw state it has recorded,
  so it records a bind only when that state changes.
- `GraphicsScopeMaterializer` owns the fused scope: it takes the first Node's
  load operations, updates the stores from each fused Node, and collects the
  reads of every draw of every fused Node; the pass fragment records exactly
  that scope. A depth-only attachment uses
  DISCARD stencil operations, because a stencil load or store of a format
  without stencil is invalid on D3D12 and Vulkan.
- The Metal device keys each pipeline variant by the scope's attachment
  formats, because a mismatch is a Metal validation error. Other backends do
  not check this at draw time.
- Format facts live in one public X-macro table,
  `source/include/VernonFormats.def`, read by the texture format enum, every
  backend mapping, layout and aspect checks, compiler storage checks and
  Python. It is public because that enum is public ABI and its values are
  generated from the table; rows are append-only so the enum values never
  move. Backend native formats are expanded in each backend's own switch, so
  the shared accessor `VernonFormats.hpp` needs no Vulkan, D3D12 or Metal
  headers. Python reads a checked-in generated module, like `_versions.py`,
  so the pure-Python package and type checkers see the formats without a
  build step; `VernonGeneratedFormatsCheck` fails when it is stale.
- `vernonHalDeviceGetFormatCapabilities` returns the table baseline
  intersected with what the backend reports, so a device never claims more than
  every backend allows. Vulkan queries per-format features, counts a format
  copyable only if it can also be sampled, and reports storage for formats
  outside the core SPIR-V set only when it enables
  `shaderStorageImageExtendedFormats`; every SPIR-V module declares
  `StorageImageExtendedFormats`, because only formats the device reports are
  bound. D3D12 reports nothing for a format that cannot be a `TEXTURE2D`.
  OpenGL applies version and extension rules: OpenGL ES float color rendering
  needs ES 3.2 or `GL_EXT_color_buffer_float`, 32-bit float filtering and
  blending need `GL_OES_texture_float_linear` and `GL_EXT_float_blend`, 16-bit
  unorm and snorm need `GL_EXT_texture_norm16`, snorm rendering needs
  `GL_EXT_render_snorm`, ES has no BGRA formats, ES never linearly filters
  depth, and ES storage images take only the ES 3.1 formats. OpenGL storage
  needs compute, multisampling comes from `GL_NUM_SAMPLE_COUNTS`, and copies
  need a color-renderable format because 2D copies are framebuffer blits.
- The table holds the uncompressed formats every backend shares. Block
  compression has a different host layout and per-platform families (BC on
  desktop, ETC2 and ASTC on mobile), so it is separate work.
  `d24_unorm_s8_uint` is left out because Apple GPUs and AMD's Vulkan driver
  lack it, so a portable Program could not rely on it, and its packed
  depth-stencil layout differs from the float-plus-stencil layout the
  depth-stencil paths assume. `s8_uint` is left out because D3D12 has no
  stencil-only format and image aspects cannot name stencil alone.
- Program load checks every image storage and graphics attachment against
  those capabilities, so an unsupported target fails at load with
  `PROGRAM_OPERATION_UNSUPPORTED` rather than at its first call. An
  attachment lists alternative formats, so one usable format suffices;
  blending an integer-only attachment reports `integer color targets cannot
  blend`.
- Color clears are a `VernonClearColor` union read by the attachment format's
  component class, so integer clears are exact; Runtime rejects non-finite
  clear values only for non-integer formats, where the bits are floats.
  D3D12 `ClearRenderTargetView` takes floats, which cannot carry every 32-bit
  integer, so the D3D12 device creates integer color attachments with
  unordered access and clears them with `ClearUnorderedAccessViewUint`
  (preferred over a clear draw, which needs a pipeline per format). That
  call saturates each value to the component's unsigned range, so values of
  8- and 16-bit signed formats are masked to their two's-complement bits
  first.
- Every HAL backend checks the image and view descriptor rules that hold on
  every backend (`validImage` and `validView` in `hal/native_images.h`, over
  `runtime/image_rules.h`: shape, mip chain length, single sample, known
  usages, attachment usage matching the format's aspects, view ranges) before
  it creates anything, so every backend rejects them alike with
  `VERNON_HAL_INVALID_ARGUMENT`. Backends check only their own limits, such as
  native format support and cube arrays.
The full image contract is [`image_resources.md`](image_resources.md).

## 13. Resource lifetime and failure

Destroying a HAL resource frees it after the device's latest submitted
point, so the submissions that name it finish first; a recording that names
it is evicted with it (`architecture_v2.md` section 7). Residual blocks are
buffers of the device the schedule links on and live as long as the residual
that holds them. Host writes are ordered on the device's timeline rather than
tracked per owner: a write takes effect after every submission made before
it and before every later one, and its source bytes are free when it
returns.

Each `RuntimeContext` has the same owner state machine and a separate local
operation control. Bundles, ProgramGraphs, resolved Stages, Program
executables, linked schedules and VJPs hold committed context-child leases.
Context close therefore rejects live direct or transitive work without relying
on raw pointer counts. A HAL device owns its context and refuses destruction,
returning `VERNON_HAL_INVALID_ARGUMENT` and staying valid, while it has
objects or its context has children; the owner releases them and destroys
again. An image with views and a surface with an acquired image refuse
destruction alike.

Backend implementation boundaries return allocation-free `Result` errors.
Public C ABI entry points are the only layer that maps those errors to status
values and cold diagnostics.

Authored Runtime and HAL expected control flow does not throw. Standard-library and
third-party operations may still throw; each public C entry point contains
those exceptions in one shared cold boundary around an internal
`Result`-returning operation. Fallible handle construction additionally exposes
a status-bearing out-handle form so the C++ wrapper can return `Result` without
discarding parse, verification, unsupported-target, resource, lifecycle, or
internal failure identity. The C++ wrapper is `noexcept`; Python raises only
after the native result reaches the binding boundary. No failure silently
changes backend, narrows dtype, serializes an invalid dispatch, or falls back
to a compatibility path.

The native Runtime and HAL support layer provides move-aware, non-sentinel
`Option<T>`, `Result<T, E>`, and `Result<void, E>` values. Inactive-alternative
access is a contract violation. Lifecycle, HAL, Runtime, and resource-access
errors are allocation-free values
with stable codes and bounded numeric/static context; explicit adapters map
them to the existing C statuses. Checked arithmetic, checked atomic
retain/release, and non-throwing allocation helpers use these results.
Emergency diagnostic rendering is bounded and does not allocate.

Runtime diagnostics are thread-local and keyed by both context address and a
monotonic context generation, so context-address reuse cannot expose stale
errors. A bounded table avoids allocation and process-global lookup during
diagnostic admission. Nested boundaries publish only the completed primary
operation: successful cleanup does not erase its failure, while the next
primary operation replaces it. If rich diagnostic storage fails, the original
typed error is rendered into bounded emergency storage without allocating.

Backend deferred actions catch exceptions
at the immediate callback boundary. Deferred drains preserve the first typed
failure, continue required cleanup, and retain failed resource-release work for
retry. A command-recording allocation failure makes pending-write state
conservatively unknown rather than losing a required synchronization.
The source policy rejects authored Runtime and HAL `throw` expressions and requires
each remaining `catch` site to have an exact reviewed fingerprint, category,
and reason; moving or substituting a catch invalidates that review.

The Python package declares its public surface for static analysis but resolves
native-backed exports lazily. Importing shader-contract helpers during
configuration therefore does not load `_native`; direct import, star import,
and `dir()` still expose the canonical public names.

The shared lifecycle support layer uses one owner-local atomic word for
`Open`, `Closing`, or `Closed` together with its admitted child count. Child
admission and the `Open -> Closing` transition therefore have one atomic
linearization order: an admission that wins blocks close, while close that
wins rejects the admission before native mutation. A move-only child
reservation owns a checked parent reference and rolls back unless publication
commits it into a child lease. Close and destruction attempts are move-only
rollback guards; failure or abandonment restores `Open`, while successful
commit is the only route to `Closed`.

Already admitted children use their own local operation control block.
Operation pins increment its checked atomic pin count without consulting a
process-global registry. Destruction first enters `Closing`; outstanding pins
reject that attempt with a stable count snapshot, and no new pin can enter
until rollback restores `Open`. A zero-pin attempt may destroy native state
and commit `Closed`. Operation controls are created fallibly and destroyed only
through checked intrusive references. Each pin and destruction attempt owns
such a reference, so dropping the registry or creator reference cannot
invalidate an in-flight guard. Explicit guard completion releases its retained
control-block reference immediately rather than keeping counters artificially
saturated until lexical scope exit.
