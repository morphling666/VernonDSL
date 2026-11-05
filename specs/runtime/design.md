# Runtime design

Status: current Runtime architecture. Programs run as linked schedules over
the HAL ([`architecture_v2.md`](architecture_v2.md)). RuntimeCore and the RHI
remain for Stage resolution and pipeline preparation until step 5 of
`architecture_v2.md` section 16.3 removes them; the Python claims and
residencies of section 1 remain until buffers have one home.

Program semantics and deployment are defined by
[`../program/architecture.md`](../program/architecture.md) and
[`../program/execution_manifest.md`](../program/execution_manifest.md).
Graphics-specific semantics are defined by
[`../program/graphics_execution.md`](../program/graphics_execution.md).

## 1. Ownership layers

Runtime execution is split into four layers:

- **Runtime API** owns Program bundles, executables, linked schedules and
  VJPs, timeline points, validation diagnostics, and public resource
  references.
- **RuntimeCore** owns manifest resolution, Stage binding plans, pipeline
  preparation, and provider-neutral caches.
- **HAL** owns devices, resources, surfaces, recorded command buffers,
  submission, and timeline points (`VernonHal.h`).
- **VernonRHI** is the provider that RuntimeCore prepares pipelines through.
  It still backs DirectX 12 and Metal, which have no HAL device yet.

RuntimeCore stores opaque provider resource references. It does not own native
textures, framebuffers, descriptor heaps, queues, command encoders, or fences.

### Python RuntimeSession ownership

`RuntimeSession` selects one fully constructed native Runtime. The native
binding stores the Runtime and optional RHI host in one shared `RuntimeState`;
an internally created OpenGL context is owned directly by that RHI host rather
than by a Python helper. Destruction is ordered Runtime → RHI device → owned GL
context. Runtime, executable, schedule, VJP, argument-record, residual, and
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
RHI, executable, schedule, resource, and residual owner leases determine
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
when their carrier is a uniform or storage buffer, because the provider
places those bytes in a push constant, a native uniform, or a buffer it
fills, such as its encoder's transient memory. Such a Value therefore needs host
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

VernonRHI owns runtime discovery of CUDA and Vulkan loaders. Vulkan headers are
compile-time dependencies only. Vulkan loader discovery uses the standard
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

Compute and graphics resolve are shared by every RuntimeCore provider backend
(Vulkan, DirectX 12, Metal, OpenGL/ES, CUDA) in `rhi_provider_pipeline`. A
backend supplies only a target: its diagnostic name and artifact kind.
Whether the provider receives the artifact's binary or source follows from
which one the Stage loader filled. Metal also supplies native binding
resolvers that remap layouts to argument-buffer locations. Resolve prepares a
Stage's binding plan and group layouts but creates no groups: the HAL device
that runs a linked schedule makes them under the owner rule of
`binding_groups.md` section 6. The shared pipeline objects own their provider
handles and release them on destruction, so a resolve that fails halfway leaks
nothing and backends need no teardown hook. Sharing keeps the backends from
drifting: they report identical diagnostics, each including the provider's
last error.

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
owned by VernonRHI.

Submissions do not wait. Each recording owns a command frame (pool, command
buffer, fence, and the staging chunks it bumps through) that is reused once the
device's submission serial passes the serial the frame was submitted under; at
most 64 frames are in flight. A retired frame returns its staging chunks to a
pool the device's frames share (`rhi/staging_pool.h`, also used by DirectX 12),
so staging memory follows the data in flight rather than the frame count; a
frame that outgrows its chunk takes the smallest free chunk that fits or a new
one twice as large. Every recording begins with a global memory barrier, so
it observes all work submitted before it without a host wait and individual
transfers need no barriers of their own. Native objects the caller destroys
are destroyed once the queue reaches the serial submitted before the destroy.

Descriptor sets belong to binding snapshots and live as long as them, not as
long as a submission, so they come from a chain of free-capable pools rather
than per-frame pools. Allocation tries the pool that last had room, then every
other pool (freed sets return space to their own pool), and only then creates
a pool twice the size of the last, up to sixteen times the first. Pools are
created on first use, so borrowed devices allocate sets the same way.

Native Vulkan rasterizes normalized device y = -1 to image row 0, the
opposite of the shared convention (`graphics_execution.md` section 6). The
adapter therefore sets a negative-height viewport (origin y + height, height
-height), core since Vulkan 1.1, which the backend requires. Front face and
scissor pass through unchanged, because the flipped viewport keeps winding and
framebuffer rows in the shared convention. Negating `gl_Position.y` in the
shader, as on GL, was rejected: GL gets the negation from SPIRV-Cross, but
Vulkan consumes SPIR-V directly and would need a SPIR-V rewrite of every
vertex stage, while the viewport is dynamic state the adapter already sets.

### DirectX 12

DirectX 12 loads pre-cooked DXIL and never invokes DXC at deployment. The
provider owns device, queue, allocators, fences, resources, descriptor heaps,
and transient upload/readback storage. Tests may select WARP through an
internal facility; normal creation selects hardware adapters.

Submissions do not wait. Each recording owns a command frame (allocator,
command list, pooled staging chunks, descriptor heaps) that is reused once the
queue fence reaches the frame's value; a device records one list at a time and
keeps at most 64 frames in flight. Recordings need no queue-order barrier: separate
`ExecuteCommandLists` calls on one queue complete in submission order and
flush caches at their boundaries. Owned resources the caller destroys are
released once the fence reaches the value submitted before the destroy.

Inline uniforms lower to the same SPIR-V push-constant block as on Vulkan, in
the `vulkan_push_constant` layout. SPIRV-Cross emits that block as an explicit
`cbuffer` at b0 space0 with `packoffset` taken from the SPIR-V member offsets,
and the root signature declares the same register as one 32-bit root-constant
parameter per stage (`kDirectXRootConstantRegister` in
`runtime/interface_names.h`). The adapter places each value at its running
offset aligned to the reflected element alignment, the rule the SPIR-V
lowering uses, so every byte offset is decided by the compiler-owned layout.
Loose HLSL globals were rejected: DXC packs them into an implicit `$Globals`
buffer with its own packing rules and auto-assigns its register, so the
runtime would have to mirror HLSL packing and the binding would depend on
which registers were free. Push-constant matrix columns use a 16-byte stride on
every target, because an HLSL `cbuffer` starts each matrix column at a register
and `packoffset` cannot express the tighter std430 stride of 2-row matrices.

Descriptor set N uses register space N + 1
(`kDirectXDescriptorSetRegisterSpaceOffset`): the compiler shifts every
resource before emitting HLSL, and the adapter shifts every descriptor range,
so space 0 holds only the root constants. Otherwise the uniform block planned
at set 0 binding 0 would also take b0 space0. A constant-buffer block takes its
layout from the std140 SPIR-V interface decorations, as on every other backend.

Compute and graphics root signatures share one descriptor-table builder: a
CBV/SRV/UAV table and a sampler table, each in layout entry order. They
differ only in inline values, which are read storage buffers in compute and
root constants in graphics.

A storage buffer whose layout access has no write is a raw SRV
(`ByteAddressBuffer`, a `t` register); only written ones are raw UAVs. Reads
then need no UAV state, so a buffer one dispatch writes and the next reads
takes an ordinary state transition, and an encoder's transient values can be
read in place from upload staging, which cannot be a UAV. The compiler matches
the layout access: VernonToGPU records each texture's and storage argument's
access (`vernon.resource_access`; by-value storage is `read`), and since the
generic SPIR-V ABI lowering builds globals without argument attributes, the
read bindings ride on the entry function until a pass after that lowering
gives their globals a block type whose members are `NonWritable`, as glslang
emits `readonly` blocks. A `NonWritable` variable decoration is not enough:
the lowering shares one block type between equal leaves, and a Vulkan driver
then served stale reads of the leaves a kernel wrote. Graphics storage reads
are `NonWritable` blocks too. Dispatches and draws write storage buffers, images
and samplers through one descriptor writer, so both report the same errors.
A dispatch or draw that reuses the descriptors an earlier one of its binding
set wrote in the same encoder still transitions every resource, because a
command in between may have left a resource in another state.

### Metal

Metal consumes cooked MSL on supported Apple platforms. Runtime validates MSL,
OS, device feature, argument-buffer, and workgroup requirements before
creating compute or graphics pipelines.

### OpenGL and OpenGL ES

OpenGL and OpenGL ES require a host-owned current context. The owner supplies
`make_current` and `get_proc_address` callbacks. Runtime and VernonRHI do not
link a window-system library.

Each backend accepts only artifacts for its matching GLSL profile. Compute
requires OpenGL 4.3+ or OpenGL ES 3.1+. Capabilities that depend on API version
are queried after context creation.

Graphics GLSL targets 3.30, which cannot declare `layout(binding)` on a uniform
block, so every block links at binding point 0. The compiler names each GLSL
uniform block from its SPIR-V binding (`openGLUniformBlockName`, shared through
`runtime/interface_names.h`). After linking, the adapter looks up each
uniform-buffer entry's block by its planned binding and calls
`glUniformBlockBinding`. Pipeline creation fails if any active block is left
unassigned, because that block would silently read binding point 0. Names come
from the binding rather than the source name because source names are
sanitized, uniqued per stage, or generated (TensorView metadata carriers). The
`GL_ARB_shading_language_420pack` alternative was rejected because macOS
OpenGL 4.1 does not expose it.

GL rasterizes normalized device y = +1 to the last image row, as native
Vulkan does (`graphics_execution.md` section 6). Every GL vertex stage therefore
negates `gl_Position.y` (SPIRV-Cross `flip_vert_y`), and the adapter inverts
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
`GL_FRAMEBUFFER_SRGB` is enabled, so the desktop adapter enables it from
`beginRendering` to `endRendering`. It is not left enabled across the device:
2D image copies and present are framebuffer blits, which would then decode and
re-encode `srgb` texels instead of copying them unchanged, and the host's own
draws would see the state. Every blit also disables it, because a scope whose
native begin fails is never ended.

GL surfaces (`presentable_surfaces.md` section 5) map acquire to handing out
the next RHI texture of the configuration, after comparing `drawable_size`
with the configured extent, and present to: make the device's context current
on the window, set the swap interval, blit the texture into framebuffer 0 with
its rows reversed, swap, and make the device's own target current again. The
rows are reversed because the negated vertex y above puts image row 0 at the
top while framebuffer 0 counts rows from the bottom. The blit goes through a
framebuffer object created and deleted per present, because the cached
framebuffers belong to rendering scopes and per-present names cost nothing
next to the blit.

On every backend the command layer keeps the keys of acquired images whose
contents are still undefined in a per-device list. Each surface reserves one
entry when it is created, and holds at most one acquired image, so marking at
acquire never allocates and an acquire cannot fail after the backend has
handed out the image. Present removes the entry even when nothing wrote the image, which
keeps the list within its reservation. An encoder lists the undefined images
it writes and removes them from the device list only when it is submitted, so
a read in another encoder cannot pass on a write that has not been submitted
and may never be. The encoder reserves its list entry before it records the
write, so a write the backend has recorded is always listed.

The GL draw state that Runtime sets on a host context, including the inverted
front face, persists after Runtime returns. The host re-establishes any state
its own draws depend on, as with Skia and Qt RHI external commands. Runtime
in turn reapplies its own state whenever the host may have changed it.
Restoring host state after each draw was rejected because it would need a
query of every state Runtime touches.

The context owner outlives RHI and Runtime children. All GL allocation,
transfer, deletion, and dispatch operations first make the associated
context current. A context is current on one thread at a time, so a GL device
is bound to the thread that owns its context; other backends advertise
concurrent submission instead. A submission inserts a fence sync, which
completion polls and waits on; GL itself orders later commands after it and
defers deleting objects that pending commands use.

The spilled uniform bytes of a transient group go to its encoder's
transient memory (`binding_groups.md` section 8) rather than a buffer per
group, and those of a persistent group to a block of the device's persistent
block pool, written once with the group. Every draw uses one vertex array per adapter, made
when the first graphics pipeline is prepared, so calls create no GL
objects of their own.

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
[compiler design](../compiler/design.md)), and the adapter merges each pair
at that descriptor. Vulkan writes one combined descriptor. D3D12 places the
sampler in a sampler table at the Texture's register. OpenGL binds the
sampler object to the Texture's unit. Metal resolves the sampler member that
reflection records beside the Texture. OpenGL compute GLSL (4.30, and ES
3.10 as for every OpenGL ES stage) declares each unit through
`layout(binding)`, so compute assigns no unit uniform, unlike desktop
graphics GLSL 3.30.

Render scopes:

- A scope begins and ends explicitly: RHI `BeginRendering`/`EndRendering`, or
  the provider `begin_rendering`/`end_rendering` callbacks that Runtime reaches
  through `vernonRuntimeCoreBeginRendering`/`EndRendering`. The RHI adapter
  forwards both callbacks to the RHI with view resource keys. Draw descriptors
  carry no attachments, operations, or render area; a draw renders into the
  open scope. Keeping attachments in one place removed the draw-versus-scope
  consistency checks that every adapter needed.
- The RHI owns the native scope. Each backend implements `beginRendering` and
  the RHI opens it lazily, on the first draw or at `EndRendering`. A scope
  without draws is opened at `EndRendering` so its clears and stores still
  apply; that is how a graphics Node without draws clears. Clears requested
  before the native scope opens become load operations.
- A scope declares the images its draws sample (`sampled_images` of the
  provider rendering descriptor). Core rejects a read without sampled usage
  or overlapping an attachment (same parent, overlapping mips, layers and
  aspects), and Runtime reports a Node sampling what it renders to. The
  backend makes the reads shader-readable before it opens the native scope.
  This is required because a Vulkan render pass cannot change layouts, and
  the first draw of a fused scope cannot transition the images that later
  draws sample. Splitting the native scope at such a draw was rejected:
  Vulkan fixes store operations when the scope begins, so the first half
  would have to store contents that the scope's DISCARD stores drop. The
  Vulkan draw only checks that its sampled images are readable and fails
  clearly when a read was not declared. D3D12, OpenGL and Metal can make
  images readable inside a scope, so their draws do not depend on the
  declaration.
- Vulkan and D3D12 track image layouts and states per subresource on the
  image only. A view points to its image and covers the subresources its
  descriptor selects, so transitions through a view and through its image,
  or through two views of one image, see the same state.
- Vulkan image layouts follow submission order, because independent
  recordings may be submitted in another order than they were recorded in. A
  recording keeps, per subresource it uses, the layout it needs first and the
  layout it leaves, and never changes the image's layouts. A subresource's
  first use records no barrier. Submission, under the device lock, records
  the transitions from the image's layouts to those first layouts into a
  prologue command buffer of the same frame, submits the prologue and the
  recording in one batch, and then sets the image's layouts to the ones the
  recording leaves. Abandoning a recording has nothing to restore. A first
  use under an explicit barrier keeps the barrier as a same-layout memory
  dependency, since the recording's earlier untracked accesses still need
  it. Recordings on a borrowed device execute in the order they are
  recorded, inside the host's command buffer, so a first use transitions
  inline from the image's layouts. An abandoned borrowed recording commits
  its layouts, because the host still executes what it recorded. This is the
  Dawn and wgpu model; a journal with rollback was rejected because it cannot
  order two recordings that are both still open. D3D12 keeps its journal:
  it records one encoder at a time, so recording order is submission order.
- A borrowed Vulkan or D3D12 device only records; its host executes the
  commands. Device-side immediate operations (uploads, mipmaps) record into
  the host's command buffer in call order, and their staging is retired when
  the host signals completion. Each borrowed submission takes a serial, and
  the completion of serial s retires only the storage recorded up to s, since
  the host may not have executed later submissions yet. An abandoned borrowed
  recording keeps its storage until the next submission retires it, because
  the host still executes what it recorded. Downloads are rejected with
  `UNSUPPORTED` before anything is recorded, because they would return before
  the copy ran.
  The device advertises `BackendCommandExternalExecution`, and the RHI front
  end rejects every operation whose results the host reads, so the rule
  holds for every backend that borrows. Submitting into a private command
  buffer on the borrowed queue was rejected: the host owns that queue and
  would have to promise never to use it concurrently.
- Scope objects do not depend on a pipeline. Vulkan without dynamic rendering
  caches render passes and framebuffers in the RHI device. OpenGL caches one
  complete framebuffer per attachment set (texture names plus the stencil
  flag), attached and completeness-checked when the set is first rendered, so
  a scope opening only binds it. Re-attaching one shared framebuffer whenever
  the set changed cost about 6 µs per scope, mostly in
  `glCheckFramebufferStatus`, because frames alternate their targets.
  Destroying an image deletes the framebuffers that attach it, imported
  images included: GL may reuse the texture name, and an unbound framebuffer
  keeps the texture storage alive. The device's `drawStateGeneration` advances
  whenever the RHI binds a scope framebuffer or changes write masks or the
  scissor, so the adapter's cached draw state is reapplied.
- `GraphicsScopeMaterializer` owns the fused scope: it takes the first Node's
  load operations, updates the stores from each fused Node, and collects the
  reads of every draw of every fused Node; the managed batch begins exactly
  that scope. A depth-only attachment uses
  DISCARD stencil operations, because a stencil load or store of a format
  without stencil is invalid on D3D12 and Vulkan.
- Metal checks pipeline attachment formats against the open scope at every
  draw because a mismatch is a Metal validation error. Other backends do not
  check this at draw time.
- Format facts live in one public X-macro table,
  `source/include/VernonFormats.def`, read by the RHI and texture enums, every
  backend mapping, layout and aspect checks, compiler storage checks and
  Python. It is public because the RHI enum is public ABI and its values are
  generated from the table; rows are append-only so the enum values never
  move. Backend native formats are expanded in each backend's own switch, so
  the shared accessor `VernonFormats.hpp` needs no Vulkan, D3D12 or Metal
  headers. Python reads a checked-in generated module, like `_versions.py`,
  so the pure-Python package and type checkers see the formats without a
  build step; `VernonGeneratedFormatsCheck` fails when it is stale.
- `vernonRhiDeviceGetFormatCapabilities` returns the table baseline
  intersected with what the backend reports, so a device never claims more than
  every backend allows. Vulkan and D3D12 query per-format features; Metal and
  OpenGL apply version and extension rules: OpenGL ES float color rendering
  needs ES 3.2 or `GL_EXT_color_buffer_float`, 32-bit float filtering and
  blending need `GL_OES_texture_float_linear` and `GL_EXT_float_blend`, ES never
  linearly filters depth, and ES storage images take only the ES 3.1 formats.
  OpenGL storage needs compute, multisampling comes from `GL_NUM_SAMPLE_COUNTS`,
  and copies need a color-renderable format because 2D copies are framebuffer
  blits.
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
  integer, so integer color attachments are created with unordered access and
  cleared with `ClearUnorderedAccessViewUint` (preferred over a clear draw,
  which needs a pipeline per format). The clear needs a non-shader-visible
  CPU descriptor, which comes from a separate staged descriptor ring. The
  clear transitions render target to unordered access and back inside the
  callback, so the tracked state is unchanged and nothing is journaled.
- The RHI front end checks the image descriptor rules that hold on every
  backend (shape, mip chain length, single sample, known usages, attachment
  usage matching the format's aspects) before it dispatches, so every
  backend rejects them alike with `INVALID_ARGUMENT`. Backends check only
  their own limits, such as native format support and cube arrays.
The full image contract is [`image_resources.md`](image_resources.md).

## 13. Resource lifetime and failure

Program bindings, provider references, transient descriptor storage, command
resources, and completion objects remain alive until the provider reports
completion.

Destroying a public owner does not release a resource still retained by an
in-flight call or by residuals. Caller-owned RHI resources are not tracked as
pending owners; caller writes to them are ordered by memory class instead.
A host upload into `VERNON_RHI_MEMORY_DEVICE` is queue-ordered after every
earlier submission on the device, without a CPU wait. A write into
`VERNON_RHI_MEMORY_UPLOAD` or Metal shared memory is immediate, so the
caller keeps one region per call in flight. Runtime generation changes invalidate
cached native handles.

Each RHI device registry publishes an immutable lookup snapshot. Device-local
resource and command slots have stable addresses, are allocated fallibly
before native creation, and publish only after their device-child reservation
and native payload are complete. Buffer, Image, ImageView, Sampler,
NativeDescriptorRange, CommandEncoder, and Completion therefore have one
lifecycle authority; there is no parallel binding-reference record.

Persistent binding, graph, encoder, and completion ownership is represented by
move-only typed retained leases. A retained lease delays native recycle, while
an operation pin protects only one admitted operation. Native resolution and
descriptor access validate the exact device, generation, resource kind, key,
and retained lease before pinning. Public destroy hides a retained resource
immediately and tears it down only when its final retained lease is released.

Command state belongs to its device. Encoder and completion lookup uses
acquire-published stable slot pages rather than a process-global command
registry or hot-path registry mutex. Submit transactionally moves retained
resource leases and cleanup state from the encoder to the completion; failed
native submit rolls the encoder destruction attempt back.

Each `RuntimeContext` has the same owner state machine and a separate local
operation control. Bundles, ProgramGraphs, resolved Stages, Program
executables, linked schedules and VJPs hold committed context-child leases.
Context close therefore rejects live direct or transitive work without relying on raw
pointer counts. Device-backed contexts and their RHI adapters each retain an
explicit RHI device-child lease, so an attempted device destroy cannot
invalidate adapter state. `vernonRhiDestroyDevice` then returns
`VERNON_RHI_STATUS_INVALID_ARGUMENT` and leaves the device published; the owner
must release its children and destroy again, because a device still published
at process exit aborts in the device registry's destructor. Every destroy
failure is reported rather than aborting. A backend teardown failure rolls the
registry's close and destroy guards back, returns
`VERNON_RHI_STATUS_INTERNAL_ERROR`, and also leaves the device published and
retryable.

RuntimeCore publishes an immutable binding revision containing the provider
object, retained resources, and canonical value snapshot. Replacement
validates and constructs a complete candidate before swapping one retained
revision reference under the binding-set-local lock. Encode retains the
published revision under that short lock, releases the lock, and only then
calls the provider; an old revision is destroyed after its final in-flight
encode reference is released. Failure releases only candidate state and leaves
the published revision unchanged. Unchanged canonical bindings are recognized
without allocation or provider mutation. Providers expose creation and
destruction only; in-place binding mutation is not part of the provider
contract.

A provider with `VERNON_RUNTIME_PROVIDER_PARTIAL_BINDING_SETS` accepts binding
sets that cover part of a layout (`vernonRuntimeCoreCreateBindingSubset`) and
draws that compose several of them through `partial_bindings`. Composition,
not a provider-side merge, is the contract because immutable sets must stay
shareable between draws and pipelines of one layout; RuntimeCore checks that
the composed sets cover every slot once before the provider sees the draw.
Recorded draws use it to bind a sealed set's native state once
([`../program/recorded_draws.md`](../program/recorded_draws.md) section 13).
Only OpenGL implements it; set-based backends map it onto per-group
descriptor sets with the group partitions of the compiler cut.

Backend and built-in provider implementation boundaries return allocation-free
`Result` errors. Public C ABI and provider-vtable callbacks are the only layers
that map those errors to status values and cold diagnostics.

Authored Runtime/RHI expected control flow does not throw. Standard-library and
third-party operations may still throw; each public C entry point contains
those exceptions in one shared cold boundary around an internal
`Result`-returning operation. Fallible handle construction additionally exposes
a status-bearing out-handle form so the C++ wrapper can return `Result` without
discarding parse, verification, unsupported-target, resource, lifecycle, or
internal failure identity. The C++ wrapper is `noexcept`; Python raises only
after the native result reaches the binding boundary. No failure silently
changes backend, narrows dtype, serializes an invalid dispatch, or falls back
to a compatibility path.

The native Runtime/RHI support layer provides move-aware, non-sentinel
`Option<T>`, `Result<T, E>`, and `Result<void, E>` values. Inactive-alternative
access is a contract violation. Lifecycle, RHI, Runtime, and resource-access
errors, together with provider-boundary errors, are allocation-free values
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

Foreign provider lease callbacks and backend deferred actions catch exceptions
at the immediate callback boundary. Deferred drains preserve the first typed
failure, continue required cleanup, and retain failed resource-release work for
retry. A command-recording allocation failure makes pending-write state
conservatively unknown rather than losing a required synchronization.
The source policy rejects authored Runtime/RHI `throw` expressions and requires
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
