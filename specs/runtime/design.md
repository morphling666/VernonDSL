# Runtime design

Status: current Runtime, provider, RHI, and Program execution architecture.

Program semantics and deployment are defined by
[`../program/architecture.md`](../program/architecture.md) and
[`../program/execution_manifest.md`](../program/execution_manifest.md).
Graphics-specific semantics are defined by
[`../program/graphics_execution.md`](../program/graphics_execution.md).

## 1. Ownership layers

Runtime execution is split into four layers:

- **Runtime API** owns Program bundles, executables, instances, invocations,
  pullbacks, validation diagnostics, and public resource references.
- **RuntimeCore** owns manifest resolution, Stage binding plans, Program
  invocation planning, provider-neutral caches, and publication policy.
- **Runtime provider** owns physical resources, command construction,
  submission, completion, barriers, and transient binding storage.
- **VernonRHI** implements the built-in GPU provider over CUDA, Vulkan,
  DirectX 12, Metal, OpenGL, and OpenGL ES.

The CPU provider implements compute directly and does not link VernonRHI.
Foreign engines may implement the provider SPI without depending on
VernonRHI.

RuntimeCore stores opaque provider resource references. It does not own native
textures, framebuffers, descriptor heaps, queues, command encoders, or fences.

### Python RuntimeSession ownership

`RuntimeSession` selects one fully constructed native Runtime. The native
binding stores the Runtime and optional RHI host in one shared `RuntimeState`;
an internally created OpenGL context is owned directly by that RHI host rather
than by a Python helper. Destruction is ordered Runtime → RHI device → owned GL
context. Runtime, executable, instance, invocation, and pullback wrappers
retain the shared state directly; argument and graphics-control marshalling
objects are owned by their invocation. Teardown therefore does not depend on
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
RHI, executable, invocation, resource, and pullback owner leases determine
final teardown.

`RuntimeSession` is also a context manager. Its selection uses `ContextVar`, so
nested and concurrent scopes do not mutate the process default. Every public
Program, Kernel, pipeline, cooked-asset, VJP, and pullback entry creates one
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

An invocation locks all participating control blocks in stable owner order,
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

Every invocation returns one canonical mutation outcome. Its overall
submission is `NotSubmitted`, `Completed`, or `Indeterminate`; each writable
Program boundary or RenderPass control independently reports `Unchanged`,
`Committed`, `InPlaceCommitted`, or `Indeterminate`. An asynchronous Execute
reports `Submitted` for the invocation and for each boundary its queued work
writes; the outcome of its submission, read once the submission completes,
replaces each with the completion state
([`pipelined_submission.md`](pipelined_submission.md) section 2). Commit-after-success
destinations remain unchanged after planning, allocation, transfer, execution,
readback, or pre-commit failure. Only an in-place boundary whose submitted work
has unknown completion poisons its owner. Host reads and new device claims then
fail until a complete authoritative host replacement recovers that owner.
Python resolves this outcome while its write claims are still held and never
infers mutation from an exception or aggregate success flag.

Fragmented host writes may conservatively expand their claim to a contiguous
span, synchronize device-written gaps under that expanded claim, and issue one
preserving upload; they never touch a gap outside the acquired claim.
Fragmented readback remains a set of exact ranges and uses one native batched
transfer. Texture transfer descriptors carry exact box, mip, array layer, and
aspect. A residency operation submits all selected ranges or subresources as
one native batch; cube faces and depth/stencil planes are not widened in
Python.

Pullback construction retains only runtime-owned host bytes, device buffers,
and tape carriers selected by the resolved residual plan. No Python resource
claim survives the forward invocation. Pullback apply first prepares a typed
cotangent/gradient boundary plan without residency work, admits every exact
claim in stable owner order, then materializes and executes. Device and
packed-host publication must both succeed before gradient write claims are
released.

If runtime tape validation requires a larger carrier, forward replay restores
only the resolved graph inputs before executing again, using each input's
authoritative host snapshot or retained device source. Intermediate Values are
outputs of the replay and are never restored from host allocation bytes; this
keeps retry semantics independent of uninitialized transient storage and
allocator history.

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
  -> VernonProgramInstance
  -> VernonProgramInvocation
  -> bind Values and controls
  -> optional completion mode
  -> execute
  -> commit or rollback
  -> optional VernonPullback
  -> optional VernonSubmission (wait, state, completion outcome)
```

`VernonProgramBundle` is immutable deployment data.
`VernonProgramExecutable` owns one immutable `ResolvedExecutionPlan`.
`VernonProgramInstance` owns one persistent binding state for both arguments
and graphics controls. `VernonProgramInvocation` owns one transaction over
that state, its immutable execution snapshot, concrete invocation descriptors,
invocation-local autodiff options, and temporary physical state. Persistent
state and frozen snapshots share immutable binding maps; dirty updates create a
copy-on-write revision, while a clean invocation reuses its snapshot without a
map clone. Execute never mutates a published binding or control payload and
performs the resolved plan without publishing staged persistent bindings. A
successful execute must be followed by commit; every failed or abandoned
invocation must roll back. Commit is the only operation that publishes staged
binding revisions and transfers a retained pullback. In both completion modes
a submitting Execute hands out its `VernonSubmission` once, before the
invocation is destroyed; the submission may outlive the invocation, and an
invocation destroyed with its submission untaken waits for it.

Runtime has no pipeline-bundle compatibility loader, profile executor, direct
Stage asset submit path, unbound Program execution path, temporary instance,
node-local direct-binding path, or executable-owned mutable invocation state.
A ProgramGraph invocation binds only the canonical exported parameters and
resolved graphics-control slots of its executable; Runtime maps those
parameters to physical alias slots through indexes precomputed at resolution.
Control binding validates both slot and control kind. Resource-backed argument
and graphics-control revisions retain caller-provided owner leases for as long
as the persistent revision or a frozen invocation references them. An optimized
single-node schedule remains a Program execution optimization.

## 3. Resolve boundary

Resolution has two explicit results:

1. `ResolvedProgram` validates logical Program structure, Value SSA, Storage
   versions, controls, Stage contracts, boundaries, and derived dependencies.
2. `ResolvedExecutionPlan` selects exact physical Stage implementations,
   endpoint projections, carriers, residency, transfers, hazards, backend
   barriers, graphics scopes, publication transactions, and residual plans.

Every execution path consumes the physical plan. Invocation code and backends
do not rediscover policy from names, carrier shapes, resource aliases, or
sentinel values.

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

Program Values and Storages are canonical logical state. A
`MaterializedNodeFrame` creates transient Stage-local arguments according to
the resolved endpoint projections. It packs and unpacks physical aggregate
carriers without changing canonical Value layout.

TensorView bindings carry:

- owner/resource identity;
- element layout and access;
- rank and concrete shape;
- signed byte strides;
- byte offset and extent.

Dynamic shape, stride, offset, byte length, and compute grid are invocation
data. They are not artifact specialization inputs. Zero values are never used
as “infer this later” sentinels.

## 5. Transfers and residency

`ResolvedTransferExecutor` executes only transfers present in the resolved
plan:

- host upload;
- device-to-device buffer or image copy;
- host readback;
- provider-native transition.

Transfer ordering participates in the same command dependency graph as Stage
execution. Compatible intermediate Values remain device-resident. Runtime does
not synchronously read a device result to host merely to feed a later device
Node.

Checked arithmetic is required for every address, offset, extent, stride,
shape product, and transfer range.

## 6. Publication

`PublicationTransaction` implements Program boundary publication:

- `commit_after_success` stages output and commits only after successful
  completion;
- `in_place` binds caller-visible storage directly.

The compiler publishes an output in place when its borrowed Storage is written
only by render-pass attachments. Staging those attachments would cost every
pass a per-invocation image, a full copy and a host wait before the copy,
which keeps frames from pipelining
([`pipelined_submission.md`](pipelined_submission.md) section 2). The price
is that a failed pass leaves its attachments indeterminate, not unchanged.
Outputs written by compute keep commit-after-success value semantics.

An asynchronous invocation publishes only in place: its writes land in
caller-visible storage in queue order, and its submission reports whether
they completed. A commit-after-success boundary still needs a host-observed
completion before its copy, so its plan requires `WAIT` until guarded
deferred copies exist (`pipelined_submission.md` sections 3.2 and 4).

Rollback discards staged buffer/image publications. Device-resident
publication uses provider copy commands. Host upload/download is used only
when the resolved boundary residency requires host transport.

Invocation failure poisons only state whose contract cannot be restored.
Reusable Program instances and pullbacks preserve their documented state after
recoverable transactional failures.

## 7. Pullbacks

Forward-with-VJP returns a pullback that owns:

- the immutable executable plan;
- immutable retained Value and Storage snapshots;
- retained provider resources and exact versions;
- immutable tape;
- replay and checkpoint descriptions.

Mutable invocation scratch and command encoders are not retained inside the
pullback. Each apply creates a fresh `ProgramInvocationState`, binds canonical
cotangent and gradient slots, executes the backward plan, and transactionally
publishes fresh gradients.

Residual and checkpoint selection is Program-owned planning over the resolved
forward graph. The shared internal DAG planner supplies a
`ProgramResidualPlan`; it is not a differentiable CommandGraph execution API.

Pullback application uses only `VernonProgramArgument` boundary bindings.
Derivative leaf/group reflection is metadata, not another execution ABI.

A ProgramGraph executable has no composite pullback or derivative ABI.
Executing with pullback retention enabled prepares independent retained state
for each differentiated child but exposes none of it before commit. Commit
returns a null composite pullback. After commit and before destroying the
invocation, the caller transfers each node's retained handle exactly once with
`vernonRuntimeProgramInvocationGetNodePullback`. Applying that handle executes
only the child Program's resolved backward plan in its original boundary-slot
namespace. Runtime does not reverse ProgramGraph connections or accumulate
cotangents between nodes. Executing with retention disabled performs
primal-only graph execution and retains no node pullbacks.

## 8. Private Command DAG

The private Command DAG owns:

- Value and resource hazards;
- upload, device-copy, Stage execution, and readback ordering;
- native barriers and resource transitions;
- render-scope formation and fusion;
- execution of resolved replay and checkpoint commands from the Program
  residual plan;
- submission, completion, and transient resource lifetime.

It is not serialized as Program topology and is not a public Python authoring
API. Runtime records and submits commands internally. Engine-owned execution
graph embedding does not change Program binding semantics.

Debug groups and timings ([`gpu_profiling.md`](gpu_profiling.md) sections 2
and 3) are a property of DAG encoders, not commands: every encoder of a Node,
including its transfers and barriers, shares the Node's scope (label, first
Node, Node count), and submission opens one group, and with timings one timer
just inside it, for each run of consecutive encoders sharing a scope. A scope
is shared rather than copied because fusing a later Node extends its label
(`a+b`) and Node count after the first member's encoders exist. The
submission record keeps each device submission's scopes in timer order and
reads their spans when it retires the completion, before destroying it. A
graphics Node's transfers are
planned apart and end the open render batch, which is appended before them;
appending them straight into the plan would place them ahead of the earlier
Node's draws that the open batch still holds. Label and timer order are tested
through the native calls a fake OpenGL driver receives, not through an RHI trace,
because a trace inside the RHI would cost every shipping draw a lock; the
real backends are only checked to render the same pixels with labels.

Every GPU step of one Execute, pullback apply, or Stage submission goes
through one submission record: publication initialization and commit copies,
graph flushes, tape-retry restores, and the immediate Stage encoder. The record
owns each step's completion and the contexts that step retains until it
completes. A blocking step is `submit` followed by `wait` on that same record,
not a separate wait-and-execute path. Submit-time failure injection fires
before the native submit and completion-time injection after a successful
wait, so step states stay `NotSubmitted`, `Indeterminate`, or `Completed` with
the same meaning on every path. Recording and submission stay in one call
because devices without independent recording serialize both under one
execution session; submit order must match record order. Destroying a public
submission waits for its record before the context-child lease commits,
because the context may destroy its device once that destruction commits.
Device wait time is measured by the record, so control-plane telemetry
reports it for every path that waits.

An invocation in `VERNON_COMPLETION_ASYNCHRONOUS` mode leaves its steps
`Submitted` and hands the record out through `TakeSubmission`; the record
then also retains the invocation snapshot and the graphics recordings its
commands read. Until deferred effects exist (phase 2), asynchronous plans
are those whose writes are all in place. Resolve records the first other
effect as the plan's wait requirement: a commit-after-success publication,
then an autodiff tape. Retaining a pullback also needs a wait, because the
pullback is published from host-validated state. Execute rejects either with
`PROGRAM_SUBMISSION_REQUIRES_WAIT` before it submits anything.

The record also retains the invocation's execution workspace lease. A
workspace returns to its instance's pool only when both the invocation and
the record release it, because preparing a workspace for another invocation
frees the storage that the pending commands read. Invocations of one instance
running at the same time lease different workspaces.

Each context admits submissions through one queue. The queue holds weak
references to the records, oldest first, with the bytes each record charged.
Before an asynchronous Execute or a Stage submission records anything,
`admit` retires finished records. It then waits for the oldest record while
the pending count is at `max_pending_submissions`, while the instance's
estimate would push the pending bytes past `transient_budget_bytes`, or while
it would push the RHI submissions the pending records hold past the device's
`max_in_flight_submissions`. The estimate is the bytes and RHI submissions
of the instance's previous Execute; a Stage submission is one RHI submission.
The RHI rejects a submission over its limit instead of waiting, and frees a
slot only when a completion is observed, so without this bound a slow GPU
turns a within-bounds frame into a failed submission. Waiting for
the oldest record first keeps retirement in submission order and never blocks
on work submitted later. Settling a record releases what it retains, and that
can call back into Runtime, so it runs outside the queue lock. The queue
never keeps a record alive. Destroying a submission handle still waits for
its record, so destroying a context or executable needs no drain. Telemetry
reports the pending count and bytes, plus the wait counts and durations for
each bound.

Runtime-owned storage that one Execute creates is charged through a
thread-local meter scoped to the executing context. The charges come from
device buffers, invocation-owned images and publication staging images.
Storage of another context, or created on another thread, is not charged.
A phase-1 graphics frame charges only the device copies of host-memory
Tensors its draws read: uniforms are bound at Bind time, and inline uniforms
and descriptors live in RHI rings that retire by completion value. A frame
whose inputs are all device resources charges 0 bytes, so only the
pending-count bound limits it until plans own intermediate storage (phase 2).

When an asynchronous submission fails, the record reports `Indeterminate`
for every boundary it writes. That is all the poisoning phase 1 needs:
asynchronous plans write only in-place, caller-owned storage, and no
runtime-owned mutable instance storage exists. Asynchronous Python resource
owners are phase 4.

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
this helper instead of per-project cook rules.

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

Compute and graphics resolve and invoke are shared by every RuntimeCore
provider backend (Vulkan, DirectX 12, Metal, OpenGL/ES, CUDA) in
`rhi_provider_pipeline`. A backend supplies only a target: its diagnostic name
and artifact kind, plus the attachment-format mapping for graphics and the
value fill for compute. Whether the provider receives the artifact's binary or
source follows from which one the Stage loader filled. Metal also supplies
native binding resolvers that remap layouts to argument-buffer locations, and
CUDA supplies its own value fill, since packed host values become host-storage
buffers there. The binding set is created at the first invocation and updated
in place afterwards; resolve never creates placeholder bindings, because every
invocation rewrites all values. The shared pipeline objects own their provider
handles and release them on destruction, so a resolve that fails halfway leaks
nothing and backends need no teardown hook. Sharing keeps the backends from
drifting: they report identical diagnostics, each including the provider's
last error. CPU stays separate: it packs a call frame and dispatches through
the CPU provider with push constants.

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
differ only in inline values, which are raw UAV buffers in compute and root
constants in graphics. Dispatches and draws write storage buffers, images
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

The GL draw state that Runtime sets on a host context, including the inverted
front face, persists after Runtime returns. The host re-establishes any state
its own draws depend on, as with Skia and Qt RHI external commands. Runtime
in turn reapplies its own state whenever the host may have changed it.
Restoring host state after each draw was rejected because it would need a
query of every state Runtime touches.

The context owner outlives RHI and Runtime children. All GL allocation,
transfer, deletion, and invocation operations first make the associated
context current. A context is current on one thread at a time, so a GL device
is bound to the thread that owns its context; other backends advertise
concurrent submission instead. A submission inserts a fence sync, which
completion polls and waits on; GL itself orders later commands after it and
defers deleting objects that pending commands use.

GL reports invalid transfers only through `glGetError`, leaving the
destination untouched, so image uploads and downloads run inside an error
scope; an unchecked error would otherwise surface as zeroed data and then fail
whichever operation next checks the context. Core OpenGL ES cannot
`glReadPixels` depth or stencil, so on OpenGL ES 3.1+ a compute pass copies
one aspect of a depth image region into a buffer, one 32-bit word per texel:
`texelFetch` for 2D and 2D-array images, and for cube images a `textureLod` of
the texel's center through a nearest-mipmap-nearest sampler, which selects
exactly that texel and level; stencil is read through
`GL_DEPTH_STENCIL_TEXTURE_MODE`. Downloads map that buffer; the same copy is
the OpenGL ES depth and stencil path of the image-to-buffer copy command that
asynchronous readbacks need (`pipelined_submission.md` section 3.3). Dawn
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
  caches render passes and framebuffers in the RHI device. OpenGL keeps one
  scope framebuffer per device, re-attached only when the attachments change;
  the device's `drawStateGeneration` advances whenever the RHI rebinds it or
  changes write masks or the scissor, so the adapter's cached draw state is
  reapplied.
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
  `PROGRAM_OPERATION_UNSUPPORTED` rather than at its first invocation. An
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
in-flight invocation, submission, or pullback. A pending asynchronous
submission therefore lets its caller release or rebind everything it bound
as soon as Execute returns. Caller-owned RHI resources are not tracked as
pending owners; caller writes to them are ordered by memory class instead.
A host upload into `VERNON_RHI_MEMORY_DEVICE` is queue-ordered after every
earlier submission on the device, without a CPU wait. A write into
`VERNON_RHI_MEMORY_UPLOAD` or Metal shared memory is immediate, so the
caller keeps one region per submission in flight
(`pipelined_submission.md` section 5). Runtime generation changes invalidate
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
executables, submissions, and retained pullbacks hold committed context-child
leases. Executables own instance admission; instance state owns invocation
admission and remains alive until its final invocation is destroyed. Context
close therefore rejects live direct or transitive work without relying on raw
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
