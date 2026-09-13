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
`Committed`, `InPlaceCommitted`, or `Indeterminate`. Commit-after-success
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
  -> execute
  -> commit or rollback
  -> optional VernonPullback
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
binding revisions and transfers a retained pullback.

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

## 9. CPU execution

CPU cooking emits:

- one canonical `*.program.json`;
- a relocatable `.o` or `.obj`;
- generated static-registration `.c` and `.h` sources.

Applications link the object and registration source before loading the
manifest. Runtime resolves symbols through the static CPU entry registry and
does not parse or relocate object files.

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

### CUDA

CUDA is compute-only. It loads PTX through the Driver API and validates
address size, PTX requirements, compute capability, workgroup limits, and
required features before pipeline creation.

### Vulkan

Vulkan supports compute and offscreen graphics. Instance and device extensions
are enabled only when advertised. SPIR-V modules, descriptor layouts, pipeline
layouts, graphics/compute pipelines, barriers, and resource transfers are
owned by VernonRHI.

### DirectX 12

DirectX 12 loads pre-cooked DXIL and never invokes DXC at deployment. The
provider owns device, queue, allocators, fences, resources, descriptor heaps,
and transient upload/readback storage. Tests may select WARP through an
internal facility; normal creation selects hardware adapters.

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

The context owner outlives RHI and Runtime children. All GL allocation,
transfer, deletion, and invocation operations first make the associated
context current.

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

The full image contract is [`image_resources.md`](image_resources.md).

## 13. Resource lifetime and failure

Program bindings, provider references, transient descriptor storage, command
resources, and completion objects remain alive until the provider reports
completion.

Destroying a public owner does not release a resource still retained by an
in-flight invocation or pullback. Runtime generation changes invalidate
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
invalidate adapter state.

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
