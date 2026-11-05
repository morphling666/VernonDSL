# Runtime architecture: linked schedules over an asynchronous HAL

Status: accepted design, migrating as section 16 lists; linked schedules are
the only execution path ([`design.md`](design.md) sections 2 to 8). It
replaces the per-Execute interpreter, the stored plans of
[`stored_plans.md`](stored_plans.md), the binding groups of
[`binding_groups.md`](binding_groups.md), transactional publication
([`../program/architecture.md`](../program/architecture.md) section 9) and
the per-submission completion effects of pipelined submission, as section
15 lists. It keeps the goals of
[`execution_model.md`](execution_model.md) and makes them the only path
rather than a cache. The compiler and engine parts of the performance work
are [`../performance.md`](../performance.md) sections 4 to 6.

## 1. Goal

**A call or a frame costs the bytes that changed plus one submission per
queue, whatever changed.** No input value, shape,
tensor identity, visible set or streamed object takes a slower path. Only a
new graph structure links, and only a new linked schedule or a new pipeline
in a pass records commands, and then only for what it changes. A new
capacity class reallocates memory; it records again only on the backends of
section 7 that lack indirect dispatch.

This is the floor for any system that submits work: the host must write
what changed and tell the device to run. The device side runs at the
kernels' speed, with barriers and memory the compiler chose.

## 2. Why the current runtime cannot reach it

Every Execute interprets the Program: it freezes bindings in a transaction,
materializes Node frames and carriers, plans transfers and publication,
builds the Command DAG and its barriers, validates recordings, submits,
then commits or rolls back. Stored plans cache that work behind a signature
in which shape, resource identity and draw structure are structural. The
cost is therefore a cliff, not a bound: `PlanetWalker` takes 15.2 ms a frame
standing and 36.8 ms walking, 80% of it re-planning
(`reports/benchmarks/planet_walker_frame.md`), and a compute launch takes
600 µs where numba takes 7 µs (`reports/benchmarks/numba_comparison.md`).

Three semantic choices force the cost, however well it is implemented:

- **Transactional publication.** Commit-after-success outputs need staging,
  host waits and transaction state on every call, to protect against a
  failed submission, which is almost always device loss and unrecoverable.
- **Host-canonical tensors.** Tensors are host bytes with device residencies
  as caches, so the runtime needs dirty ranges, owner control blocks, claims
  locked in owner order, reservation tickets and cross-session migration.
- **Layers that convert.** A Program Value reaches the device through a Node
  endpoint, a Stage endpoint, a carrier, a provider argument, a packet and a
  native call; carriers exist because a Stage's layout differs from the
  Program's.

## 3. Overview

```text
cook   (compiler, per Program variant):  device code + schedule fragment + argument record layout
load   (runtime, per artifact):          validate, create executables
link   (runtime, per graph structure):   fragments + connections -> linked schedule
record (runtime, per linked schedule):   linked schedule -> native command buffers
call   (stub, every call or frame):      write changed argument bytes, submit
```

Work is placed by **rate**, not by side. Each Program is cooked alone,
often one per node or asset (a material, a pass, a kernel), and Vernon
combines them into a `ProgramGraph` at load or on a settings change. The
compiler therefore decides everything that depends only on its own Program,
and emits it as a relocatable **schedule fragment**, like an object file.
Whatever depends on the neighbors (buffer identity across edges, aliasing
between nodes, barriers at boundaries, queues, render-pass merging) is
decided by the runtime's **linker** when the graph's structure changes,
never on a call.

The runtime is three pieces: the **linker** (section 4.2), the **HAL**
(section 6), a thin layer over the devices, and the **executor**
(section 7), which records a linked schedule and replays it. Everything else
the runtime does today moves into the compiler or the linker, or
disappears.

## 4. The schedule

### 4.1 Fragments: what the compiler decides

Program IR lowers to a schedule fragment, which states everything that
depends only on the Program:

- **Operations in order**: dispatches, passes and copies. A pass names its
  attachments with their load and store intent. Each operation lists its
  accesses: which local buffer, read or write, and as what (storage,
  uniform, sampled, attachment, indirect argument, copy).
- **Local buffers.** Each is a boundary (an import or export the graph
  connects, like an object file's undefined and defined symbols), a constant,
  or a transient with its lifetime as an interval of operations.
- **Sizes are arithmetic on shape symbols.** Each dynamic extent is a symbol
  bound by the call. An allocation's size is rounded up to its **capacity
  class** (the next power of two, or a maximum the Program declares)
  (`execution_model.md` section 8).
- **Grids and sizes are data.** Every dispatch reads its grid, and every
  copy its size, from the argument arena by indirect dispatch and indirect
  copy; a kernel reads its extents from its TensorViews. So an operation's
  commands do not depend on shapes, and capacity classes size memory only.
  CUDA graphs have no indirect launch: a call whose grid changed updates
  those kernel nodes of the instantiated graph in place, O(changed nodes),
  without recording.
- **Stages are compiled against the canonical layout.** A kernel or shader
  reads the strides and offsets of the TensorViews it is given, so no
  carrier repacks its arguments. Where a layout truly differs, as for a
  uniform block's std140 packing, the fragment contains an explicit
  conversion kernel or copy, visible and costed like any other operation.
- **Device control flow.** The operation set reserves a conditional and a
  loop whose predicate is a device value, so data-dependent control flow
  (an iterative solver until convergence, adaptive refinement) runs without
  a host wait per iteration. They record into CUDA graph conditional nodes
  and Vulkan device-generated commands; a backend without them, and the
  first migration phases, run them on the host, waiting on the predicate.
- **Transfers are operations.** Uploads, copies and readbacks are ordered
  with the kernels.
- **The argument record layout** is a flat record of the Program's
  arguments: scalars, shape symbols, buffer addresses or bindless handles,
  and per-pass draw streams (section 11), at offsets the compiler fixes.
  The record is the backend's own kernel ABI, so a submission passes its
  bytes without repacking. On the CPU it is the kernel's packed call frame.
  On CUDA it is the kernel parameter list in order, each parameter at its
  natural alignment: a TensorView passes as a memref descriptor whose
  address appears twice and whose size is the buffer's length in elements,
  a scalar as itself, and the TensorView metadata as one struct after them.
  Constants of the ABI, such as a memref's zero offset and unit stride, sit
  in the record's **template**, which a caller copies once and no slot
  writes.

A fragment has no barriers, queues, arena offsets or addresses: those depend
on the neighbors and the device. Kernel fusion and operation order inside
the Program are the compiler's alone. Program-level control flow other than
the device operations above stays host-static, as today
([`../program/architecture.md`](../program/architecture.md) section 5).

### 4.2 The linker: what the graph decides

The linker takes the fragments of a `ProgramGraph`'s nodes and its
connections, and produces one linked schedule:

- **Buffers.** A Value edge makes the producer's export and the consumer's
  import one buffer. A Storage chain is one buffer with ordered versions.
  Where homes or layouts differ, the linker inserts the explicit copy or
  conversion, never a carrier.
- **Shapes.** Connected boundaries unify their shape symbols, so the graph
  has one set of symbols and one vector of capacity classes.
- **Order and lifetimes.** It orders operations topologically, keeping each
  fragment's order, and extends lifetimes to the whole graph. Graph-owned
  transients and every fragment's transients share one arena per memory
  class, aliased by interval colouring, so nodes reuse each other's memory.
- **Barriers** come from the access lists over the whole order, with one
  algorithm inside fragments and across them, so a boundary costs only the
  barrier its accesses need, never a full one.
- **Queues.** It assigns each operation a queue (graphics, async compute,
  copy); cross-queue edges become semaphore waits.
- **Render scopes.** Adjacent passes whose attachments match merge into one
  render pass, or into subpasses where the device supports them.
- **Arguments.** The graph's arena is each node's argument record at a fixed
  offset, plus the graph's exported slots. A node-local binding token is a
  node and an offset, so the typed stubs generated for a Program
  (section 12) bind the same Program as a graph node unchanged.

A standalone Program links as a one-node graph by the same code.

### 4.3 Link rate and limits

- **Linking is O(operations + edges)**, plus O(t log t) for aliasing t
  transients. It runs at load, on a settings change, or when the Program set
  changes, as Vernon's `FrameGraph` already relinks
  (Vernon `specs/render/architecture.md` section 9). It never runs because
  values, shapes, tensors, visibility or draws changed.
- **Counts are data, not nodes.** A graph has a node per kind of work, never
  per object, view or tile: a variable number of shadow views is a view
  count and a capacity in a buffer, read by one node's indirect dispatch,
  and streamed objects are rows in tables (Vernon
  `specs/render/architecture.md` section 2, principle 4). A graph whose
  topology follows the scene would link every frame, so graph authors must
  not build one.
- **Linked schedules are cached** by the fragments' digests, the
  connections, the exports and the device, in memory and optionally on disk
  like a pipeline cache. Relinking may run off the frame thread and swap at a
  frame boundary.
- **The linker is a library.** A cook tool that knows the graph (an AOT
  application, a test) runs it offline and stores the linked schedule, so
  offline and runtime linking cannot diverge.
- **The linker never generates code** and never invokes a compiler, as
  `ProgramGraph` selection already requires
  ([`../program/architecture.md`](../program/architecture.md) section 8).
  Kernel fusion across nodes and cross-node autodiff belong to a Module
  cooked as one Program.
- **The linker never merges Program IR.** Today's graph linker builds one
  canonical Program and resolves it; fragments make that unnecessary, so the
  runtime holds no Program IR.

## 5. The artifact

An artifact holds the device code of every Stage, the schedule fragment and
the argument record layout, with the reflection the loader validates. It is
still a Program: kernels, pipelines, Modules and VJP transforms produce the
same artifact form, and the manifest contract
([`../program/execution_manifest.md`](../program/execution_manifest.md))
describes these three parts instead of the physical-resolve inputs. An
artifact may also carry linked schedules a cook tool produced (section 4.3).

Loading validates versions, digests, capabilities and limits, fail-closed,
then creates the HAL executables. It plans nothing; linking plans the graph.

## 6. The HAL

| Object | Role |
|---|---|
| Device | one GPU, or the CPU |
| Queue | in-order execution with a timeline semaphore |
| Buffer | memory with one home: device, host-visible or unified |
| Image, Sampler | textures and their views |
| Executable | a compiled kernel or graphics pipeline |
| Command buffer | recorded once, submitted many times, reading its arguments from an arena |
| Allocator | sub-allocation, arenas and pools per memory class |

- **Backends.** The RHI backends become HAL backends: Vulkan (with
  descriptor indexing and buffer device addresses, core in 1.2), D3D12,
  Metal, CUDA, WebGPU, OpenGL and OpenGL ES.
- **The CPU is a device.** Its memory is unified, its command buffer is a
  tape of chunk calls, and its queue is the worker pool (`performance.md`
  section 3.2). There is no separate CPU execution path.
- **A command buffer's native form** is the one of `execution_model.md`
  section 5: a Vulkan primary command buffer, a D3D12 command list, a CUDA
  graph built node by node, Metal indirect command buffers, WebGPU render
  bundles, and a pre-resolved tape on OpenGL and the CPU.

## 7. The executor

- **Recording.** The first call of a linked schedule runs it once: it
  allocates the arena at the call's capacity classes, and records the native
  command buffers. A call at a new capacity class reallocates the buffers
  that grow and writes their new addresses into the arena, without
  recording. OpenGL, OpenGL ES and WebGPU, whose indirect dispatch and
  addressing cannot carry every grid and buffer, record once per capacity
  class instead. An image whose extent changes, such as an attachment
  resized with the window, records the passes that use it again. A
  recording is keyed by the linked schedule, on those backends the capacity
  classes, and,
  for graphics, the attachment set and each pass's structure revision
  (section 11). Resource identity is not part of the key: kernels and
  shaders reach buffers through addresses or bindless handles read from the
  arena, so a new tensor is payload.
- **A call.** Write the argument record's changed bytes into the arena,
  write indirect counts and grids, then submit the recorded command buffers
  with the call's wait points and signal one point per queue. On OpenGL and
  the CPU, submitting decodes the tape, which also rebinds resources there.
- **No per-call planning.** There is no Command DAG, Node frame, transfer
  plan, publication, binding revision, signature or validation on the call
  path.

Recordings are kept per linked schedule in a small cache by key; a recording
whose buffers are destroyed is evicted with them.

## 8. Data model

- **A buffer has one home**, chosen when it is created. A tensor is a view
  over a buffer: offset, shape, strides and dtype. Moving data between
  homes is an explicit copy or a schedule's transfer.
- **Ordering is timeline points.** Each buffer records the points of its last
  write and of the reads since. A submission waits for the points of the
  buffers it writes and of the writes it reads, an O(1) check per buffer
  argument, and nothing on its own queue.
- **Host access waits, then maps or copies.** Reading a tensor waits for its
  last write point, then maps unified or host-visible memory or downloads
  with one copy into the destination array. Writing waits for the pending
  reads and writes, then maps or uploads through pinned staging, one copy.
  On the CPU device numpy memory is the buffer. `wait()` waits for the last
  write point and neither maps nor copies; until buffers have one home,
  linked calls record that point on the residency they write.
- **Untracked host memory makes a call synchronous.** A call that borrows a
  numpy array returns after its submission completes, since writes through
  numpy cannot be observed. A call over VernonDSL tensors returns after
  submission.
- **Sessions select devices.** `RuntimeSession` chooses the HAL device and
  queues; buffers belong to one device. Claims, dirty ranges, residency
  migration and owner control blocks are removed.

## 9. Errors and validation

- **Synchronous errors happen before anything is written.** Artifact and
  capability checks run at load; argument checks (dtypes, ranks, extents
  against declared bounds, homes, capacity) run at the argument boundary in
  O(arguments); a draw record is checked when it is written (section 11).
- **An asynchronous failure poisons what it writes.** The submission reports
  failure, and every buffer it writes is poisoned: host access and later
  calls reading it fail until it is overwritten from the host. Nothing is
  rolled back.
- **A debug validation layer** checks the schedule's hazards and every
  command against the device's state, like Vulkan's validation layers. It is
  a device option, never on the call path of a release build.

## 10. Value semantics without transactions

Outputs are written in place. A Program keeps value semantics because the
compiler decides where each output lives:

- an output the caller supplies is a destination the call writes;
- an output the caller does not supply is a fresh buffer the call returns;
- an input the Program writes is updated in place only when it is declared
  donated or in place, as JAX donation and XLA aliasing declare it; else the
  schedule writes a fresh buffer.

So a failed call never needs undoing: a synchronous failure wrote nothing,
and an asynchronous one wrote only buffers that are then poisoned.

## 11. Graphics: draws are data

- **A pass's structure is its table** of pipelines, materials and geometry
  kinds, with its attachments. Adding or removing an entry advances the
  pass's structure revision and records that pass's commands again; the
  other passes keep theirs.
- **Its draws are a draw stream**: a buffer of draw records, each a table
  index, an index or vertex range in a shared geometry buffer, an instance
  range and an offset into the instance list. On the high tier a culling
  kernel writes the stream and its counts, and the pass draws it with
  multi-draw indirect count. Immediate-mode drawing (UI, debug) writes the
  stream from the host through a typed writer.
- **A record is valid by construction.** The writer checks the table index
  and ranges against the pass's table and capacities when it writes, in
  O(1); a culling kernel clamps the same way. The runtime never walks the
  draws.
- **On OpenGL and OpenGL ES** the tape decodes the host-written stream into
  draw calls, one per record, without validation. Where the device lacks
  compute shaders or device streams, the culling kernel's CPU cook writes
  the host stream, so one kernel culls on every backend (Vernon
  `specs/render/gpu_scene.md` section 8.2).
- **Swapchain images** are attachments whose identity is part of the
  recording key, so a pass that renders into the presented image keeps one
  recording per image.

## 12. One calling convention

From each argument layout a generator, in the style of `vernon_dsl.bindgen`,
produces typed call stubs:

- **Python**: one native call that fills the record from the arguments
  through the buffer protocol and submits, returning the fresh outputs and a
  future over the signalled points.
- **Host code and AOT**: a C function with the same record, called directly.
- **C++**: a header with the record as a struct.

A warm call's cost is the stub's packing plus the submission: under 5 µs
from Python, under 1 µs from Host code (`performance.md` section 1). The
frontend's source checks move off the call to a process-wide source epoch
(`performance.md` section 3.1).

## 13. Autodiff

A VJP transform produces two executables: the forward one, which returns the
residual buffers as ordinary outputs, and the backward one, which takes them
with the cotangents. A pullback is those buffers and the backward
executable. A residual whose size depends on data has a capacity class and
an overflow flag the forward writes; the host checks the flag when it waits
for the forward and calls again with the next class. The runtime has no
pullback, tape or replay machinery of its own.

## 14. Queues and devices

The linker assigns operations to queues, and cross-queue edges are
semaphore waits, so async compute, copy queues for streaming and, later,
several devices are more queues and transfers in the same model. The
distributed compiler of [`../future/README.md`](../future/README.md) emits
fragments with device placement, which the linker honours.

## 15. What it replaces

| Today | Fate |
|---|---|
| `ResolvedExecutionPlan` physical resolve at load | split: the Program's part into its fragment, the graph's part into the linker |
| `program_graph_linker`, merging child Programs into one canonical Program | the fragment linker (section 4.2) |
| `MaterializedNodeFrame`, endpoint carriers | removed; Stages read the canonical layout, explicit conversions where needed |
| `ResolvedTransferExecutor` | schedule transfers |
| Command DAG (`source/lib/execution_graph`) at run time | barriers and queues in the linked schedule |
| `PublicationTransaction`, commit and rollback | removed (section 10) |
| Instance binding revisions, invocation transactions | the argument record |
| Stored plans, signatures, stored graphics scopes | one recording per linked schedule (section 7) |
| `GroupStore`, `InstanceGroups`, binding groups | bindless handles in the arena; bound groups only in the OpenGL tape |
| Recording validation per Execute | draw records valid by construction (section 11) |
| `ResourceControlBlock`, claims, dirty ranges, residency migration | buffers with one home and timeline points (section 8) |
| `SubmissionRecord`, `VernonSubmission` | HAL submissions and timeline points |
| Python `_PersistentBindingTable`, binding tokens | generated call stubs |
| RHI backends | HAL backends |
| Packets | the OpenGL and CPU tape format |

## 16. Migration

Each phase builds beside the old runtime, moves workloads, deletes what they
no longer use, and ends with the benchmarks of `performance.md` section 7.
The whole migration ships under one `PROGRAM_VERSION` and one
`COMPILER_CONTRACT_VERSION` cut, without compatibility code, as the spec
index allows before publication.

1. **HAL on CPU and CUDA, one-kernel fragments.** Fragments for single
   kernels, the linker for one-node graphs, the executor, generated Python
   stubs. Gate: the numba benchmark's launch, `axpy` and `relief` targets on
   CPU and CUDA.
   Linked calls share the old runtime's tensors in this phase. A call takes
   the linked path only when each buffer owner is unborrowed and current
   where it runs: on the CPU it has no device residency; on CUDA its
   residency in the session holds its latest version with no host writes
   pending. Any other call runs as a Program, which uploads or migrates;
   the next call then links. A linked device write publishes as a
   Program's does: the owner's version advances, the residency holds it,
   and the host downloads before it reads. A CUDA call returns once its
   launch is enqueued on the device's stream, which the RHI's copies share,
   so a readback orders after it.
2. **Multi-kernel fragments, compute graphs, Modules and VJP.** Linking
   across nodes, arena aliasing, capacity classes, the two-executable VJP.
   Gate: autodiff acceptance tests and a training step whose CPU time does
   not grow with its layer count. Section 16.1 states how.
3. **Graphics on Vulkan and the OpenGL tape.** Pass fragments, render-scope
   merging, pass tables, draw streams, swapchain recordings; Vernon's
   `FrameGraph` links its nodes' fragments. Gate: `PlanetWalker` standing
   and walking (`performance.md` section 1), which
  `reports/benchmarks/planet_walker_frame.md` measures, and the Editor
  scenes (`reports/benchmarks/editor_scenes.md`). Section 16.3 moves the
  graphics backends onto the HAL and deletes the RHI.
4. **Remaining backends** (D3D12, Metal, WebGPU, OpenGL ES, wasm32 CPU), then
   deletion of the old runtime.

### 16.1 Phase 2: Programs as fragments

- **A Program's fragment is built from its manifest.** `programFragment`
  takes a Program variant's graph, its ABI and the reflection of each Stage
  its nodes run, and lays the nodes' kernel frames one after another in the
  record, in node order, in O(nodes + bindings). It is the library function
  of section 4.3: the loader calls it until the cook tool stores its result
  in the artifact, so both produce the same fragment.
- **Local buffers come from Storages.** A Storage that an input boundary
  slot borrows is an import, and one that an output slot publishes is an
  export; the caller supplies both, so a fresh output is allocated by the
  stub from its extents (below), as `_AllocationSpec` does today. Every
  other owned Storage is a transient. A transient the Program creates with
  zeros starts with a fill operation. So does an export or a gradient that
  starts as zeros: the caller allocates it for the call and does not upload
  its zeros, and the fills of such slots come first in the fragment, their
  arguments side by side in the record.
- **A boundary argument has a site per binding.** A slot's first binding is
  its offset; each further binding writes the same bytes again, a buffer's
  address as an `ADDRESS` field and a scalar's bytes as a `VALUE` field, and
  each binding of a TensorView has its own offset, extent and stride
  fields. A stub writes every field it is given, so it packs one node and a
  hundred alike.
- **Extents are static or an import's.** A Storage's byte length names, per
  axis, a constant or an axis of a Value; the fragment resolves that Value
  to the import slot it came from. The linked schedule exposes every slot's
  extents in that form (`vernonScheduleSlotExtents`), so a stub sizes fresh
  outputs without the Program, and the executor checks a supplied export
  against them in O(arguments).
- **Transients live in the arena.** Each has the interval of operations
  that access it; the linker packs transients whose intervals overlap at
  distinct offsets, largest first at the lowest offset that fits, and lets
  disjoint ones share bytes. Sizes are capacity classes, the next power of
  two of the bytes the call's extents need, so a call whose classes did not
  change reuses the arena and its addresses; a new class packs and grows the
  arena again. The executor writes the transients' addresses and metadata
  into its copy of the record on each call; the caller never sees them.
- **Order is the node order.** The CPU tape completes each dispatch before
  the next, and CUDA runs one stream, so neither needs barriers. The linker
  still gives each operation the earlier ones it must follow, from the
  accesses over the whole linked order: a read follows the buffer's last
  write, and a write follows that write and every read since. The Vulkan
  recording of phase 3 turns these into barriers. Arena aliasing adds none,
  since transients that share bytes have disjoint intervals; a recording
  that packs the arena orders one's last access before the next one's first.
- **Graphs of Programs link into one schedule.** `vernonScheduleLinkGraph`
  takes each node's Program descriptor in run order and connections, each
  a set of node slots that are one buffer or scalar: a Value edge's
  producer and consumers, or a Storage chain's versions. Connected slots
  must have one kind, element type and rank, and agree on static extents.
  A connection becomes one slot of the schedule at its first site; each
  other site repeats the slot's bytes as an `ADDRESS` or `VALUE` field. An
  internal connection, a Value the graph does not export, becomes an arena
  transient instead, sized from the extents its sites declare, and the
  first operation that accesses it must write it. An extent naming an
  internal connection resolves to that connection's extents, so a graph's
  outputs are sized from its inputs through any chain of nodes. Each
  node's record, executables and constants follow the previous node's, and
  `vernonScheduleNodeSlot` maps node slots to schedule slots. Linking is
  O(operations + fields + sites). Autodiff tapes and caller grids do not
  connect across nodes.
- **CUDA graphs launch a schedule once.** The compiler gives every kernel a
  twin entry, `<name>__vernon_indirect(ptr)`, that loads the kernel's
  parameters from device memory at the offsets the record packs them at
  (each at the next offset its NVPTX ABI alignment allows) and runs the
  kernel's body. A command buffer of several commands whose kernels all
  have twins replays as a CUDA graph: one kernel node per dispatch, each
  pointing at the record in device memory, and one node per run of fills
  whose arguments lie side by side, a fill kernel whose grid row `y`
  zeroes the `y`th fill. A call copies its record into the next of four
  record slots, each with its own graph instance, and launches that
  instance; a slot is rewritten once the event after its last launch has
  completed. A node changes only when its dispatch's grid does, so a step
  that binds new buffers costs a record copy and one launch, not a
  parameter update per node. The runtime checks the record layout against
  the direct entry's PTX parameters before it uses a twin.
- **A buffer's memory is allocated once per size.** The CUDA device caches
  the allocations freed buffers release, by size rounded to a power of two
  below 1 MiB and to whole MiB above, up to 1 GiB in all, so a step's fresh
  outputs, gradients and residual blocks reuse memory without a driver
  call, and `cuMemFree`, which waits for the device, stays off the step.
  Reuse needs no fence: all device work is ordered on one stream.
- **Creating a buffer costs the same with thousands alive.** The CUDA
  device finds a free buffer handle next-fit: the search starts after the
  handle it last gave out and wraps to the start only while a destroyed
  handle waits to be reused, so a step's fresh gradients and residual
  blocks do not scan every live buffer. Vulkan's slot reservation still
  scans from the start; phase 3 gives it the same search.
- **The VJP links two fragments.** The forward fragment is the forward
  graph with the Values the backward graph captures as extra exports; the
  backward fragment imports those residuals and the cotangents and exports
  the gradients, which start as zeros. When every residual's extents are
  static or another forward argument's, a forward call writes its residuals
  into one native residual block, each at an aligned offset, and returns
  it; the pullback holds the block, and the backward call binds each
  residual from it. Otherwise each residual is a tensor the call creates.
  A step therefore creates no tensor per residual. The backward fragment
  imports only the captured Storages its nodes bind and takes each scalar
  it binds as a `SCALAR` slot, filled from the forward call's argument. The
  residual block copies each captured argument's descriptor (extents,
  strides, element layout) and a scalar's bytes, since the call's argument
  memory is gone when the pullback runs. A captured Storage is read where
  it lives, copy-on-write: while a pullback that reads it lives, its owner's
  control block lists the pullback, and the first write admitted to it
  (host or device) moves the owner to a copy of its memory and leaves the
  original host array and residencies to a snapshot owner the pullback
  holds; a native call refuses to write such an owner, so the write takes
  that admission. An unwritten input costs one list entry. A Module pullback
  binds the snapshot in place of a written input; a Program pullback's
  record already points into the snapshot's memory. A GPU backward kernel
  whose forward kernel keeps no tape binds the tape only for the launch
  metadata, which the schedule holds as a constant.
- **A CPU autodiff tape is a residual the block owns.** A tape Storage
  (`vernon.ad_tape`) is a `TAPE` slot of both schedules: written by the
  forward node that initializes it, read by the backward node that reads
  it. Its record bytes hold a `VernonScheduleTape` pointer, which the
  residual block supplies after its buffer residuals. A dispatch that keeps
  a tape runs one frame per lane, each naming its lane's allocator and root
  region. The forward dispatch fills a host batch of one lane per
  invocation, with the tape Value's `minimum_tape_stride_bytes` static
  stride, so a dynamic loop promotes the lane rather than overflowing. It
  fails the call when any lane reports an allocator error, and otherwise
  seals the batch. The backward dispatch reads the sealed batch through a
  reader per lane. A forward call rewrites the batch of its own block, so
  each pullback replays its own forward call. The process's tapes share one
  `AutodiffMemoryPolicy` (64 MiB per dispatch, 256 MiB live). A tape that
  is not a residual of a VJP, or that two nodes use, keeps the Program
  path.
- **A CUDA autodiff tape is device buffers the block owns, checked at the
  pullback.** A GPU kernel keeps its tape in i32 carriers it reflects with
  the roles `tape`, `replay_segment` and `replay_status`; they bind the
  tape Storage's `TAPE` slot, and their fields go in the dispatch's tape
  use rather than in a slot. A forward call allocates per tape a data
  buffer of stride × workgroup volume × groups bytes, one
  `gpu::Segment` per group and a cleared `gpu::BatchSummary`, uploaded
  stream-ordered, and writes the carriers' address and extent into the
  call's record; the backward dispatch binds the same buffers. The stride
  is the tape Value's `minimum_tape_stride_bytes`, or the largest one an
  earlier pullback of the same VJP found its forward needed. The forward
  never waits for its lanes: the first backward call on a block, which
  waits for the forward anyway, reads each status. A lane that outgrew its
  stride reports the bytes it needed when it stopped, a lower bound; the
  backward call then raises the VJP's learned stride and runs the forward
  again into a block the first one owns, from the inputs its record names,
  which copy-on-write kept, with its outputs in scratch buffers, until
  every lane fits (each report exceeds the stride it ran with, and a tape
  is at most 4 GiB). The backward runs against the last such block. The
  caller's outputs are never rewritten, so a forward that writes one of
  its inputs fails instead of running again. Rejected: waiting in the forward call to size the tape
  (a stall per VJP step), and rerunning into the caller's outputs (a
  pullback that rewrites what the caller already read). A forward that
  keeps device tapes keeps every argument's shape and strides in its
  block, not only those the backward reads, since a replay rereads them.
- **A bound VJP's step is flat in Python.** `step.bind` (language
  contract section 8) gives each of the two schedules a bound view: the
  slots of bound inputs and bound gradients hold their objects, and a call
  passes only the rest. Every per-layer check stays, in native code: each
  bound buffer is validated and bound as an argument is, a forward call
  stores the versions of the bound buffers it reads in its residual block,
  and the backward call fails when one has moved. The shapes of a call's
  fresh outputs are kept per shape of its unbound arguments. A bound step
  therefore costs Python work only for its unbound arguments and outputs,
  and native work of about one binding per bound buffer. A bound buffer
  keeps its validated binding, with the instance dicts of its owner and
  control block, its residency set and, for a buffer the schedule writes,
  its coherence and region ranges. A call revalidates the binding by
  reading those dicts: the version, poison, borrows, I/O reservations, the
  owner's array and the residency. It binds again through attributes when
  any of them changed or a dict lacks an entry. A call that writes the
  buffer publishes through the same objects. The residual block's slots,
  which every call passes None, are filled by the bound view itself. When a bound
  buffer is not current on the device the call uploads every bound input
  once and retries; a call that still does not link runs as a Program and
  copies the gradients into the bound Storages.
- **A public gradient is the kernel's destination.** The Program VJP
  returns an adjoint as the public gradient without a copy when its Storage
  traces to a zeroed allocation of the backward graph, by following each
  compute result to the operand it writes (`result_resource_sources`). Every
  value along that chain must be used only as the next compute's write
  destination, and no other gradient may share the allocation. The chain
  then takes the gradient's write-only public type. Any other adjoint is
  copied, since it may be a caller's cotangent or be read. This removes one
  copy node per layer from a chain's backward schedule. The cotangent's
  seed copy stays, one per output.
- **A linked call uploads a host-only input.** A Module's or a VJP's
  linked call gives an input only the host holds a device buffer and
  uploads it, as a Program call would, so a fresh batch does not fall back.
  The call's fresh outputs, gradients and residuals get their device
  buffers without an upload.
- **Built-in copies run one lane per element.** The compiler's
  `_program_copy_*` and `_program_add_*` kernels have no grid of their own;
  their dispatch carries the extents of their first TensorView, and the
  executor writes the product as the grid on each call.
- **A kernel that takes one leaf of a Struct Storage takes a dense
  carrier.** The fragment gives the binding a transient of the leaf's
  type, gathered from the Storage before the dispatch when the Storage
  may be read and scattered back after it when the kernel writes the
  leaf, as `LeafCopy` commands. The CPU copies on the host; CUDA runs a
  built-in PTX kernel, one lane per element, with the copy's addresses
  and extents passed by value, so a command buffer with copies launches on
  the stream rather than as a graph.
- **What does not link falls back.** Struct values, grids computed from
  Values, Instance storages and Module parameters make
  `programFragment` fail, and the call runs as a Program, as phase 1
  kernels do. Graphics nodes link on graphics devices (section 16.2).

### 16.2 Phase 3: passes, tables and streams

- **Graphics devices are Core devices until section 16.3 replaces them.**
  The HAL device of Vulkan, OpenGL and OpenGL ES is one implementation
  over RuntimeCore's provider (`VernonRuntimeCore.h`). This bullet and the
  Core-specific ones after it (plans, stored scopes, packet batches, the
  OpenGL adapter, scope fusion in `GraphicsScopeMaterializer`) describe that
  interim device; slots, tables, streams, labels and the compute and VJP
  links carry over to the HAL backends unchanged. An
  executable is a Core pipeline with its binding plan, prepared from the
  Stage artifacts as Program resolution prepares them; a command buffer is
  a Core packet batch; a submission begins commands on the RHI encoder,
  writes the call's value bytes into updatable groups, encodes the batch
  with the call's payload (draw counts, grids, viewports, clear values) and
  submits. The batch is the pre-resolved tape of section 6 on every Core
  backend: on Vulkan a call encodes it into a fresh primary command buffer
  instead of resubmitting a native one, since the RHI tracks image layouts
  and staging per command frame. Encoding is O(commands), and with draw
  streams a pass has a command per table entry, not per draw.
- **A graphics node is a pass.** The descriptor names the resolved Program
  executable (`VernonScheduleProgramDescriptor::executable`) whose Stages
  prepared the pipelines, so linking prepares nothing. The fragment's slots
  are the graph's input boundary slots as `ARGUMENT` slots, each a
  `VernonProgramArgument` (an RHI buffer view, image view, sampler or host
  value bytes), then per node a `PASS` slot (a `const VernonRenderPass *`:
  attachment views, load and store operations, clear values, render area),
  a `DRAW` slot (a `VernonDrawCommand`, whose template is the node's default
  draw), a `DYNAMIC` slot (a `VernonDynamicState`; zero `struct_size` draws
  with the render area as viewport and scissor) and a `STREAM` slot (a
  `VernonScheduleDrawStream`; zero `struct_size` draws the `DRAW` slot
  once, and a stream replaces that draw with its records, indexing the
  whole buffer of the `DRAW` slot's index binding) and a `TABLE` slot (a
  `const VernonSchedulePassTable *`; null draws the node's own draw).
  Each Stage parameter reads the `ARGUMENT` slot of the boundary Value it
  projects, or, for a Value an earlier node wrote into a caller's view, the
  slot of the boundary Value with the same Storage; a node whose parameter
  reads part of a Value, an internal Value or a view-transformed endpoint
  does not link, and its Program runs as a Program. A call gives each
  argument the Stage's slot, endpoint access and element (or whole-Value)
  layout, as an invocation materializes it. Attachments come from the
  `PASS` slot, so output image slots are not slots of the fragment.
- **A compute node of the graph is a dispatch on the same device.** It has
  only its `ARGUMENT` slots. Its grid is the node's static workgroup
  counts, or per axis the host scalar of the boundary Value a control names
  (32- or 64-bit, negative rejected), or for a linearized dispatch the
  element count of its first shaped Tensor argument, all resolved at call
  time as an invocation resolves them. It encodes outside every render
  scope (`encodeResolvedStage`), so it ends the scope before it and no pass
  after it fuses with one before it; Core orders its writes before later
  draws that read them. A frame that culls on the GPU and then draws links
  as one schedule this way.
- **A pass is planned once per structure on Core devices.** Without
  buffer addresses in shaders, a bound resource is a group the batch names,
  so planning a draw command (the node's draw, its stream, or one per table
  entry) costs O(bindings). A call reuses the plan an earlier call made of
  the same table and revision, or for a pass without a table of the same
  draw, when the words of everything else the draws depend on match: the
  node's arguments but for the bytes of host Tensors, the attachments'
  formats and extent, the render area and the dynamic state. A reused plan
  takes the call's host bytes again.
- **A render scope of reused plans replays.** The first call that encodes
  a scope whose passes are all reused plans stores its packet batch, keyed
  by the plans and the attachments. A later call that matches the key
  rewrites only the updatable groups whose bytes changed and encodes the
  stored batch with its payload; stream records are read when the batch is
  decoded. A steady frame therefore plans, validates and creates nothing.
  Each command keeps up to four stored scopes, the oldest evicted first,
  so a call never evicts a scope it replays later; swapchain images are
  attachments, so each presented image has its own stored scope.
- **A packet batch prepares its render scopes.** Building a batch checks
  each render scope against the device once: attachment formats, the
  stencil aspect, and the subresources its reads sample and their overlap
  with the attachments (`prepareRendering`). The batch leases the
  attachment and sampled images, so destroying one hides it at once and
  frees it when the last batch naming it retires, as stored plans do.
  Beginning a prepared scope on an encoder checks only the encoder's state:
  no image the scope loads or samples is still undefined; then it records
  the attachments defined and takes the clear values from the payload.
- **The OpenGL adapter applies only what changed.** A draw binds its
  program only when another is bound, and a group's bindings only when the
  group differs or the program reads an entry the last application skipped.
  Each program keeps a read mask of the group entries it reads (active
  uniform blocks and sampler uniforms, and the samplers of their units), so
  entries a program never reads are never bound. Named uniforms upload per
  program when its group changes. With separate attribute formats (OpenGL
  4.3, OpenGL ES 3.1) a vertex buffer change is one call per binding;
  otherwise the attribute pointers are specified again.
- **Adjacent passes share a render scope.** A submission plans every pass
  before it begins a scope, since a scope names the images all its passes
  sample when it begins; then `GraphicsScopeMaterializer` fuses a pass whose
  attachments are the previous pass's views and which loads rather than
  clears them into that pass's scope. Image layouts and buffer barriers
  between operations follow from the accesses the linker ordered, as Core
  emits them for the resources each command binds.
  `vernonScheduleGetGraphicsTelemetry` reports the scopes and draw commands
  of the last call; a stream is one draw command.
- **A pass table is a pass's structure.** `VernonSchedulePassTable` holds
  the entries a `TABLE` slot's pass draws in order inside the pass's scope,
  each a pipeline, its arguments, a draw, an optional stream and an optional
  dynamic state. A pipeline must conform to the node
  (`vernonRuntimeGraphicsPipelineCreate` with this executable and node); a
  null pipeline draws with the node's own Stage. An entry's arguments are
  keyed by boundary slot of the pipeline's Program. A pipeline's Stage
  parameter follows its endpoint classification, as in a recording: an
  endpoint the node fixes reads the node's `ARGUMENT` slot, and a per-draw
  endpoint reads the entry's argument for its parameter, or the node's
  binding where the pipeline conforms to it. With a null pipeline each
  parameter reads the entry's argument for the Value it projects, else the
  node's `ARGUMENT` slot. Without a stream an entry draws its draw
  once; with one, the stream's records index the draw's index binding. A
  viewport or scissor the entry leaves empty is the pass's. A recording
  translates to a table with an entry per run of draws that share a
  pipeline, arguments, dynamic state and index range, the run's draws
  becoming its host stream. The table belongs to the caller and is read
  during the call.
- **Labels and timings are an invocation's.** A call labels each render
  scope with its nodes' labels joined by `+`, gives each node of a fused
  scope its own group and each dispatch one, and encodes the debug commands
  of a table's entries before their draws and its trailing ones after the
  last entry (`VernonScheduleDebugCommand`); a recording's debug commands
  end a run, so they go with the entry after them. They are packets, so a
  stored scope replays them. `vernonScheduleSetTimings` makes later calls
  bracket each scope and dispatch with RHI timers, which flushes the
  packet stream at each edge; `vernonScheduleGetScopeTimings` reports the
  spans of a waited call as `VernonScopeTiming`, so a caller maps them to
  its passes as it maps an Execute's (`gpu_profiling.md` section 3). The
  device keeps the spans of its latest 32 timed calls. Per-node counts
  stay with the caller: a table is built from a recording, which knows its
  binds and drawn draws, while the schedule reports scopes and draw
  commands.
- **A draw stream is data.** A stream is indexed draw records
  (`VernonRuntimeProviderDrawRecord`, in `VkDrawIndexedIndirectCommand`
  layout: index count, instance count, first index, base vertex, first
  instance; 20 bytes, also OpenGL's `DrawElementsIndirectCommand`) and
  `uint32_t` counts; a stream draws the first `min(counts[count_index],
  max_records)` records from `first_record` on. It is one Core packet,
  `DRAW_INDEXED_STREAM`, whose payload carries the range and, for a host
  stream, the host addresses, so a stored batch draws another stream by
  patching payload words. A device stream names buffers and is one
  `vkCmdDrawIndexedIndirectCount` on Vulkan (`VK_KHR_draw_indirect_count`,
  which the 1.1 instance admits where core 1.2 would not; providers
  advertise `VERNON_RUNTIME_PROVIDER_DEVICE_DRAW_STREAMS`). A host stream
  lives in host memory (`VERNON_RUNTIME_PROVIDER_HOST_DRAW_STREAMS`, on
  Vulkan, OpenGL and OpenGL ES): the provider reads it while it encodes,
  one native draw per record, and Core checks each drawn record's indices
  against the index buffer first, O(drawn records), which the per-record
  native draws cost anyway. A kernel that writes a device stream clamps to
  the same bounds. Writers zero the records past their count, so a stale
  count draws nothing.
- **First instance is portable on OpenGL.** SPIRV-Cross emits
  `gl_InstanceID + SPIRV_Cross_BaseInstance`, defined as
  `gl_BaseInstanceARB` under `GL_ARB_shader_draw_parameters` and a uniform
  otherwise. The adapter draws with base-instance entry points where the
  driver has them and sets the uniform where the program has it; without
  base-instance draws it offsets each per-instance attribute pointer by
  `first / divisor` records, and a divisor that does not divide `first` is
  unsupported. Records with nonzero `first_instance` therefore draw on every
  Core backend except D3D12, whose `SV_InstanceID` lacks a base constant
  yet.
- **A compute Program links from its executable on the CPU and CUDA.** The
  fragment is built from the variant's `program` JSON and the Stage kernel
  fragments prepared at load, and every owned Invocation buffer Storage is
  zeroed, as an invocation's allocation is. The caller's record is one
  `ARGUMENT` slot (a `VernonProgramArgument`, 16-byte aligned) per bindable
  boundary slot in boundary slot order, as a graphics node's is, so Vernon
  writes compute and graphics records alike; each call copies the linked
  template and writes the arguments' TensorViews into the slots and fields
  of the linked record (`ProgramArguments`), so the slot queries describe
  the caller's record and report no fields or grid. An output that
  republishes an input's Storage is an alias row after the inputs, and its
  argument is ignored. A value grid on a unit axis is the constant 1 and the
  value keeps its own slot. Struct and Tensor value arguments link when
  their physical leaves sit at the canonical layout's offsets.
  `vernonScheduleSlotBoundary` names the boundary slot of every `ARGUMENT`
  slot, on graphics devices too.
- **A composed Program graph links per node on the CPU and CUDA.** Resolving
  a composed graph there loads each node's bundle executable once and keeps
  the graph's connections and exports (`ProgramGraphComposition`); a link
  builds each node's fragment from its own executable and links them with
  `vernon::schedule::link`, a Value edge or Storage chain as one connection.
  A connection no export names is internal, as is a boundary row no
  connection or export covers, such as an output no node reads. Rejected:
  serializing the composed Program to JSON for one fragment, which would
  duplicate the graph linker's renumbering. Graphics devices keep linking
  the composed Program itself, whose passes are already per node, and keep
  the composition only when every node has a backward graph, for the VJP:
  its forward and backward schedules link the nodes' pass fragments as the
  CPU links their kernel fragments.
- **A linked VJP is two schedules and a residual block.** `planVjp` takes
  from the variant JSON the forward allocations the backward graph reads
  (residuals), the tapes that are slots (every tape on the CPU, one a
  forward kernel binds on CUDA), the Storages the backward graph reads and the
  boundary slot each comes from, and its captured scalars. The forward
  schedule exposes residuals and tapes as block slots; the backward exposes
  its captured Storages, tapes and scalars. A forward call copies its record
  into the block and allocates the residuals there, 256-byte aligned, in a
  host buffer on the CPU and an RHI device buffer on CUDA, sized from the
  extents the call's arguments resolve; tapes are owned by the block. A
  backward call rereads captured arguments from that record at their
  forward offsets and binds its block slots to the block's entries. The
  backward schedule's `ARGUMENT` slots name the primal boundary slot each
  differentiates (`DerivativeProjection`), and `vernonScheduleVjpSlotRole`
  whether it takes a cotangent or a gradient, so a bundle VJP and a composed
  one have one convention. A derivative of a struct field does not link yet.
  A backward call overwrites its gradient arguments, as the autodiff
  contract's fresh tangent owner requires: Stages that sum lanes into a
  gradient, such as a GPU kernel's one-element buffer for a scalar
  gradient, find it zeroed by the call first, on every device.
- **A residual block outlives the calls that read it.** Each call through
  a block records its timeline point there; destroying the block waits on
  the HAL device for the last one before freeing its buffers and tapes.
  That wait reports nothing: a failed call is reported only by
  `vernonScheduleWait` of its own point, so a pullback object can be
  dropped without consuming the error its caller's read is owed.
- **A memory budget checkpoints a VJP by a recomputed prefix.**
  `vernonScheduleLinkCheckpointedVjp` links a bundle VJP whose forward
  calls hold at most the budget until their pullback: the forward inputs
  and the residuals kept. Each block entry is credited to the forward node
  that produces it (the last node when none is). A forward call keeps the
  residuals of nodes k and later and writes the rest to the VJP's discard
  buffer, where nothing reads them. Its pullback runs the forward's
  operations of nodes 0 through k - 1 again, linked once per k as a
  truncated copy of the forward schedule, with outputs in scratch. The
  backward then reads the recomputed entries from that run and the kept
  ones from the forward call's block. The fragment records where each
  graph node's operations end, so a prefix is an operation count. k is
  chosen per call, since extents are known only once a call's arguments
  are written. It covers every node that records a tape, whose size is
  known only after the node ran. `balanced` and `min_runtime` take the
  smallest k that fits the budget and `min_memory` takes every node. A
  call that no k fits fails with the budget in its error. The block
  reports the plan (`vernonScheduleResidualsGetCheckpoint`): held,
  initial, kept and recomputed bytes and k of the node count. The
  recomputation factor is 1 + k / nodes. A recompute writes the entries
  the forward kept to the discard buffer rather than over them, since an
  earlier node may write an earlier version of a Storage that a later
  node finalizes. Discarded tapes are tapes the VJP shares across calls.
  A device tape whose carriers are large enough is rewritten in place in
  submission order, and carriers it outgrows are retired rather than
  freed. A forward that reads and writes one argument does not link
  checkpointed, since a recompute would read what it wrote, and neither
  does a composed VJP yet. A recompute writes the forward's write-only
  arguments, like its outputs, to scratch buffers, never the caller's. Rejected: choosing residuals one by
  one, as the Program planner's checkpoint DAG did, which needs a
  backward schedule per selection rather than per prefix length, and
  rerunning the whole forward, which recomputes the nodes whose residuals
  the budget could keep.
- **Cooked Programs call the same schedules.** A cooked Program loaded by
  Python links its schedule and VJP from its executable on first use, per
  session state, and calls them with its boundary bindings split by role;
  it uses the Program path only when the link fails. Its pullback keeps
  the copy-on-write owners of the inputs the backward reads.
- **A composed graph has one VJP.** Its forward schedule is its nodes'
  forward graphs; a connection some backward graph reads is kept as a
  residual of the block instead of being internal. Its backward schedule is
  the nodes' backward graphs in reverse, a node's input gradient internally
  connected to the cotangent of the output feeding it. A Value read by
  several nodes, or by a node and the caller, needs its cotangents summed
  and does not link yet, nor does a Storage chain or an output whose
  cotangent would be zero. Rejected: per-node pullbacks that the caller
  chains, which made every intermediate cotangent a caller buffer and a
  call.
- **Vernon's `FrameGraph` links one schedule.** Its passes are graph nodes
  linked by `vernonScheduleLinkGraph`; scene passes are `STREAM` passes
  whose tables hold Vernon's buckets (`gpu_scene.md`). On the high tier a
  culling kernel node writes the streams and counts; on the compatibility
  tier the CPU cook of the same kernel writes them into host streams. `Runtime::initialize`
  admits Vulkan and presents through an RHI surface.

### 16.3 The HAL replaces the RHI

The Core device of section 16.2 encodes every call again through four
layers: RuntimeCore packets, the RHI adapter's decoder, the RHI's command
layer (encoder slots, leases, registry lookups, layout tracking) and the
RHI backend. That per-call cost is what held Vulkan above the phase 3
gate, and it is the design sections 6 and 7 replace, so it is not tuned
further. The CPU, CUDA, Vulkan, OpenGL and OpenGL ES devices are HAL
backends, and the Core device is deleted. This section ends with
`VernonRHI.h` and RuntimeCore deleted and every backend a HAL backend.

- **Layout.** `source/lib/hal/` holds the interface (`hal.h`, moved from
  `runtime/schedule`) and a directory per backend: `cpu/`, `cuda/`,
  `vulkan/`, `opengl/` (OpenGL 3.3 and 4.x, OpenGL ES 3.x), and in phase 4
  `d3d12/`, `metal/` and `webgpu/`. The RHI backends' device, memory,
  staging, format and surface code moves there; their command encoders,
  registry and validation do not. The linker and executor stay in
  `runtime/schedule`.
- **One public device API.** `VernonHal.h` replaces `VernonRHI.h`: devices
  (created on an adapter, or adopting a caller's OpenGL context), buffers
  with one home, images and views, samplers, surfaces (configure, acquire,
  present), uploads and readbacks ordered on the queue's timeline, and
  timeline points. It has no command encoder: commands are recorded only
  from linked schedules, so Vernon's UI draws through a linked graphics
  Program whose table is ImGui's draw lists.
- **Handles are objects.** A handle is a pointer to the backend's object,
  with no registry, lease or lookup on the call path. Destroying a resource
  frees it after the device's latest submitted point, O(1); a recording
  that names it is evicted with it (section 7).
- **Vulkan records once.** The baseline is the RHI's: a 1.1 instance,
  dynamic rendering where present and render passes otherwise,
  `VK_KHR_draw_indirect_count` for counted streams. A recording is a primary
  command buffer per frame slot (two or three in flight), keyed as section
  7 states, with the swapchain image an attachment of its key. Pipelines
  are created when a schedule or pass table is created, never in a call.
  Barriers and image layouts are recorded from the linker's accesses: a
  sampled image rests in the read-only layout between uses, and an
  attachment returns to it at the end of its scope. Each recording owns the
  descriptor sets its pipelines' argument tables lower to (step 0) and an
  arena block of the frame slot; a call writes value
  bytes into the block and compares each argument's resource with the one
  its set holds, updating only the descriptors that changed, so a steady
  frame makes no Vulkan call but the submit. Host streams are copied into
  the block and drawn as device streams, so stream contents never change a
  recording; a table's structure revision does. A call is one
  `vkQueueSubmit` that signals the queue's timeline semaphore, after one
  batched transfer submission for the uploads queued since the last.
- **The Vulkan device as built in step 1** (`hal/vulkan/vulkan_device.cpp`)
  differs from that end state where the RHI device code or the pass
  planner still decides:
  - A recording covers every pass of a command buffer. Its key is each
    graphics pass's plan id, each compute pass's executable, resources and
    whether its grid is empty, the attachments of every render scope, and
    whether the call is timed. A command buffer keeps its eight most
    recently used recordings, so one per swapchain image fits.
  - Compute group counts are arena bytes read by `vkCmdDispatchIndirect`,
    since a grid that follows an instance count would otherwise key a new
    recording every frame.
  - A changed resource changes the pass's plan and so builds a new
    recording, rather than updating one descriptor. A pass whose plan
    changes every call, such as the UI's table, builds a recording every
    call; the slots of dropped recordings are pooled with their command
    pool, arena and descriptor pool, rounded up to powers of two, so such a
    build makes no Vulkan object.
  - Graphics inline values are push constants, recorded into the command
    buffer: a call whose push bytes differ from the slot's records the
    slot again. Other inline values are arena bytes.
  - Pipelines are made on a pass's first build, per variant of its state
    and attachment formats.
  - Image layouts are tracked per recording from the first use of each
    subresource; submission records the transitions from the layouts the
    RHI tracks into a prologue, or after the pending uploads, which go in
    the same `vkQueueSubmit` ahead of the recording. Completion is the RHI
    frame's fence and serial until step 4.
  - A device whose host owns the native objects (a borrowed RHI device)
    links no schedule, since its host submits; `VernonHal.h` has no
    borrowed Vulkan device.
- **OpenGL executes a tape.** Recording resolves each command to GL names,
  drops state the previous command already set (the Core OpenGL adapter's
  change tracking moves here), and stores a flat array of operations; a
  call patches the names of arguments whose resources changed, reads host
  streams, and runs the array. OpenGL 4.3 and OpenGL ES 3.1 dispatch
  compute nodes; on OpenGL 3.3 a compute node runs on the CPU device and
  writes host streams, from the same DSL kernel.
- **The OpenGL device as built in step 2** (`hal/opengl/opengl_device.cpp`),
  where the RHI device code or the pass planner still decides:
  - It drives the RHI device's context, under the RHI device's mutex. The
    device's own mutex is taken before that one, never after, so building
    a scope's framebuffer (which takes it) happens only while a recording
    is built.
  - Recordings are keyed and kept as on Vulkan: eight per command buffer,
    a new one whenever a pass's plan changes.
  - Each pipeline links its own GL program from the pipeline's GLSL, and
    fixes its uniform block bindings, sampler units and uniform locations
    then, so the tape binds units and binding points and never queries the
    program.
  - Building tracks the state each operation leaves and drops an operation
    whose state the tape already holds; at the start of a tape nothing is
    known. Vertex attribute enables are diffed at execution against the
    device's enabled mask instead, since it outlives tapes.
  - A recording's arena is a host image of its value bytes, host draw
    records and dispatch group counts. A call writes it, and uploads it
    into one of three buffers, round robin and without waiting, only when
    it differs from the bytes that buffer last received.
  - Host streams draw as multi-draws from the arena on OpenGL 4.3;
    otherwise an operation loops over the records, setting the base
    instance uniform where the driver has no base-instance draws. Device
    streams are multi-draws on 4.3 and counted on 4.6; OpenGL ES has none.
  - A compute dispatch is followed by a barrier of every bit, as the next
    pass's reads are not known to the tape.
  - A render scope binds a framebuffer for its attachments from the
    context's cache, which drops a framebuffer when a texture it attaches
    is destroyed, since GL reuses texture names. It enables sRGB writes on
    desktop OpenGL, which the tape's end disables again, and invalidates
    discarded attachments the render area covers.
  - A call is one fence; scope timings are timestamp queries from a pool,
    read when the call retires.
- **CPU and CUDA move unchanged** to `hal/cpu/` and `hal/cuda/`.
- **A CUDA buffer is a HAL allocation** (`hal/cuda/cuda_resources.cpp`)
  in the CUDA context of the device's RHI device, outside the RHI's
  registry. Its reference carries its device address as the resource
  value, with an identity whose low bits (kind 5) no RHI adapter identity
  uses, so a schedule binds it without a lookup and the Program invocation
  runtime rejects it. Residual blocks are buffers of the device the
  schedule links on. Freeing one returns it to the context's allocation
  cache, which the device's single stream orders after every submission
  that read it. Rejected: keeping RHI buffers and caching the registry
  lookup, which costs a lookup per argument per call on the RHI path this
  section removes.
- **A Vulkan or OpenGL buffer is a HAL allocation**, outside the RHI's
  registry:
  - On Vulkan (`hal/vulkan/vulkan_resources.cpp`) it is made by the RHI's
    unregistered-buffer functions of `rhi/vulkan_backend.h`. These share the
    device's lock, staging and queue order with registered buffers, and a
    destroyed buffer lives on until every submission made so far completes.
  - On OpenGL and OpenGL ES (`hal/opengl/opengl_resources.cpp`) it is an
    OpenGL buffer made under the device's mutex through
    `rhi/opengl_access.h`.
  - Its reference has identity kind 5 over the device state's address
    (`hal/native_buffers.h`). Its resource value is a serial no later buffer
    reuses, never the native handle. Recordings and plan keys compare
    resource values, and drivers reuse handles, so a handle value would let
    a cached recording bind a new buffer through stale descriptors. The
    device maps the serial to the buffer once per build through a
    process-wide registry.
- **A Vulkan or OpenGL image, view, sampler or surface is a HAL object**
  (`hal/vulkan/vulkan_resources.cpp`, `hal/opengl/opengl_resources.cpp`,
  `hal/native_images.h`), outside the RHI's registry:
  - A view's reference has identity kind 6 and a sampler's kind 7, over the
    device state's address, with a serial as the resource value, for the
    reason buffers use one. The view's registry entry holds its image's
    descriptor, its own and its image's serial, and the backend's object,
    so the planner describes a view without asking the device.
  - The render scope checks the planner leaves to the device are shared
    (`checkRenderScope`): live views of the device, attachment usage, one
    mip and layer covering the format's aspects, and sampled reads.
  - A surface image's contents are undefined from its acquire until a
    submitted pass writes it, and a submission that reads it before then is
    rejected. The set is process-wide, keyed by image serial, with a flag
    that skips its lock while it is empty.
  - A surface is a ring of its own images. Present blits one to the window:
    on Vulkan to an acquired swapchain image, on OpenGL to the default
    framebuffer under the window's context, with the rows reversed.
  - On OpenGL a view that covers its whole image in the image's format binds
    the image's texture; any other view is a texture view, which OpenGL ES
    and contexts without texture views cannot make.
  - Vulkan samples isotropically, so anisotropy above 1 is unsupported
    there; OpenGL ignores it.
- **DirectX 12 and Metal have no HAL device until phase 4.** Creating one
  returns `VERNON_HAL_UNSUPPORTED`, and Python sessions report both
  unavailable. Their compiler targets remain.
- **`VernonHal.h` as built in step 3** (`hal/capi.cpp`, `hal/resources.h`),
  where the RHI device code still decides:
  - It is a C header exported by `VernonRuntime`. Images, views, formats,
    usage and aspects are the `VernonTextureTypes.h` types schedules
    already take, so a HAL image's descriptor is the one a schedule
    argument declares. Status, buffer, sampler and surface types are the
    HAL's own; values shared with `VernonRHI.h` are equal, so the RHI
    device code the backends still call takes them by cast until it moves
    into `hal/`.
  - A HAL device owns its runtime context; schedules link on
    `vernonHalDeviceGetRuntime`, and waiting on the device waits on its
    schedules. A device, an image with views, and a surface with an
    acquired image refuse destruction rather than leave dangling objects.
  - Failures return a status and leave a message in a thread-local
    `vernonHalGetLastError`, as the call path has no context argument to
    hold one.
  - The C layer checks what is the same on every backend: struct sizes,
    transfer ranges inside the resource, view ranges, sampler enums and
    anisotropy in [0, 16], and one acquired surface image at a time.
    Backends check only what differs between them.
  - The CPU device's resources are native host memory: buffers 256-byte
    aligned and zero-filled, image transfers in the format's whole-texel
    layout, no mipmap generation, attachments, multisampling or surfaces.
  - The CUDA, Vulkan and OpenGL devices run on an RHI device that
    `hal/capi.cpp` creates for them, for its context, queue and staging
    only. The CUDA device has no images, samplers or surfaces.
- **Programs as schedules in step 4**, the cases a Program fragment links
  that section 16.1 does not name:
  - A TensorView of Structs is one BUFFER slot. Its storage leaves index
    from the view's buffer start in their own scalars, so they share its
    address, and its metadata is the view's, in whole elements; on CUDA
    each further leaf is another memref descriptor of that address.
  - On CUDA a Struct, Tuple or ranked Tensor Value is a kernel parameter
    by pointer. Its bytes are record bytes, and the device writes their
    device address into the descriptor before the launch (a `RecordAddress`
    of `hal.h`), staging the record through a ring of device records.
  - A Value links where its physical bytes are its canonical bytes: one
    scalar, or a transport of the canonical size whose copy plan moves no
    byte. Other layouts stay unlinked until a fragment can repack.
  - Written Program parameters whose element spans overlap are rejected
    before anything is written, as the writable-overlap rule requires.
  - A boundary slot no node binds still takes its argument: it gets a
    record site no operation reads, typed from its Value.
  - A pass parameter that reads an earlier pass's color or depth output
    reads it through the RenderPass slot of the first pass rendering to
    it, as an image argument built when the pass is planned.
  - An owned Storage whose extents follow an input (`vd.empty_like`) is
    an internal buffer sized when its pass is planned, from the input's
    ARGUMENT slot. Internal buffers, carriers and tape launch counts are
    buffers of the device's own pool, never the RHI's. An internal buffer
    grows to the largest call so far; the buffer it
    outgrows is freed when the device is idle, since a running submission
    may read it.
  - A VJP, a composed graph's included, links on Vulkan and OpenGL as on
    the CPU and CUDA. A residual
    the forward graph allocates is an ARGUMENT slot of both graphs, which
    the residual block allocates and the call zeroes, as the Program would
    its owned Storage. A tape is a TAPE slot, written by the graph whose
    node initializes it and read by the graph that captures it. Each tape
    carrier a pass reads (tape words, replay segments, replay status) is a
    `VernonProgramArgument` in record bytes the executor writes on every
    call from the block's device tape. A tape's launch metadata is the
    forward dispatch's group counts, read from a device buffer per
    distinct count triple.
  - A Stage parameter whose endpoint transforms its view (a constant axis,
    a reordered axis of the Value, an axis over the dispatch's invocations)
    takes the extents and byte strides the planner derives on each call.
    An axis over the invocations needs a grid the arguments do not size.
  - A Stage parameter that reads one leaf of a Struct Value reads a dense
    carrier: an internal buffer of the view's extents. A Copy before the
    pass gathers the leaf when the Stage reads it, and one after the pass
    scatters it back when the Stage writes it. The planner turns each copy
    into byte ranges on every call, merging adjacent elements' ranges.
    Vulkan records them as one `vkCmdCopyBuffer` between barriers, OpenGL
    as one `glCopyBufferSubData` per range.
  - A pullback whose tape overflowed replays the forward call with each
    output the caller passes replaced by scratch of the caller's view, so
    the replay does not overwrite the caller's outputs.
- **Steps, each gated.** Every step keeps PlanetWalker at or under its last
  measurement on each device (`reports/benchmarks/planet_walker_frame.md`)
  with CTest, pytest and the
  Vernon GPU suites green.
  0. The HAL interface, written against every backend before any is
     built: each operation of section 6 mapped to Vulkan, D3D12, Metal,
     WebGPU, OpenGL and OpenGL ES, the CPU and CUDA, with the backend that
     constrains it named. It fixes four things a later backend must not
     have to change:
     - **An argument table.** Stage artifacts name each resource a shader
       reads as an index into its pipeline's argument table, never as a
       backend binding. A backend lowers the table when it creates the
       pipeline: descriptor sets on Vulkan until it is bindless, descriptor
       heap ranges on D3D12, argument buffers on Metal, bind groups on
       WebGPU, units and binding points on OpenGL. Moving a backend to
       bindless handles or buffer addresses then changes that backend only.
     - **Queues.** A device has one or more queues; a submission waits on
       points of other queues' timelines and signals its own, so a copy or
       async compute queue, or a second device, adds queues and not an
       interface.
     - **Threads.** Recording is safe from any thread; submission is
       ordered per queue.
     - **Resource kinds.** Buffers, images and views, samplers, surfaces,
       and room for acceleration structures, sparse residency and aliased
       memory, so adding them adds kinds and not calls.
     Each recording key (section 7) is listed with what invalidates it,
     since every invalidation is a frame that records again.
  1. The Vulkan HAL backend beside the Core device. Until step 4 it adopts
     the RHI device's native device and resolves an RHI resource to its
     native handle only when an argument's resource changes. Gate: Vulkan
     PlanetWalker under 1 ms standing and 2 ms walking.
  2. The OpenGL tape backend on the same bridge. Gate: OpenGL at or under
     the numbers above, OpenGL 3.3 included.
  3. CPU and CUDA under `source/lib/hal/`; `VernonHal.h` with the resource
     API on every HAL backend.
  4. Vernon and the Python frontend on `VernonHal.h` and linked schedules
     only: textures, readback, device buffers, samplers, surfaces, the UI
     renderer, and every Program invocation and recording replaced by a
     schedule call. The bridge of step 1 is removed.
  5. Deletion, then the benchmarks of `performance.md` section 7: the
     Core device (`hal_core`, `pass_fragment`'s Core parts), `rhi/`,
     `runtime/rhi_adapter/`, RuntimeCore and the provider
     (`VernonRuntimeCore.h`, `VernonRuntimeProvider.h`,
     `VernonRuntimeRHIAdapter.h`), `VernonRHI.h`, `execution_graph/`,
     `runtime/program_execution/`, and the Program invocation runtime of
     section 15, with their tests.

## 17. Risks

- **Capacity classes cost memory**, up to twice an allocation; a Program may
  declare maxima to bound it, and the arena aliases transients.
- **OpenGL keeps a cost per draw**, the tape's decode and the driver call,
  without validation or planning.
- **A graphics leaf copy costs a range per element** when the leaf is one
  of several in its Struct, so a large view of such a Struct costs one
  OpenGL call per element, and one Vulkan copy region per element, on
  every call that reads or writes that leaf.
- **Untracked host memory** makes CPU calls over numpy synchronous; code that
  needs asynchrony uses VernonDSL tensors.
- **The migration is large.** Each phase is gated on benchmarks and deletes
  the code it replaces, so two runtimes coexist only for the workloads not
  yet moved.

## 18. Rejected alternatives

- **Evolving the interpreter with caches** (stored plans, argument arenas,
  native programs on top of today's Execute). Each cache keeps a cliff where
  its key changes, and walking, new tensors and new shapes change it.
- **Commit-after-success as an opt-in per output.** It keeps the staging,
  host wait and transaction state in the runtime for every Program that may
  use it; destination passing and donation give value semantics without them.
- **Host-canonical tensors with device caches.** They need a coherence
  protocol in one process; one home per buffer with timeline points needs
  none.
- **A Command DAG at run time.** Its hazards and barriers are known when the
  graph is linked.
- **The whole schedule in the compiler.** Programs are cooked one per node
  or asset and combined at load, so a whole-graph schedule would need the
  compiler at graph construction, one artifact per combination, and a
  recook when any node changes.
- **Complete schedules per Program, submitted in sequence.** Each would own
  its arena and fence its boundaries: no aliasing between nodes, a full
  barrier at every edge, and no render-pass merging, which tile GPUs pay for
  most.
- **Linking Program IR, then resolving it** (today's graph linker). The
  runtime would keep Program IR and a resolver; fragments link in
  O(operations + edges).
- **Validating draws per Execute.** Draw records checked when written cost
  O(1) each once, not every frame.
- **CUDA stream capture** to build graphs: it captures runtime-internal work;
  recording from the schedule is deterministic (as `execution_model.md`
  section 16 already rejects).
- **Shape as structure.** Every new extent would record again; indirect
  grids make extents payload, and capacity classes bound reallocation.
- **Recording per capacity class on every backend.** A tensor crossing a
  power of two would record again; indirect grids and addresses in the arena
  make the class a memory decision only, except where the backend cannot
  carry them (section 7).
- **Fusion at link time.** It would put a kernel compiler (MLIR and LLVM)
  in the runtime, and fusing kernels is the compiler's job when a Program,
  or a Module of several, is cooked. The linker only merges graphics passes
  into one render scope where their attachments allow (section 4.2).
- **A graphics HAL device over RuntimeCore** (section 16.2's Core device).
  It reaches every backend at once, but encodes each call again through
  packets, the RHI adapter and the RHI's validation; on Vulkan that held
  PlanetWalker near 1.5 ms of host time, against OpenGL's 0.8 ms through the
  same path and the 1 ms gate.
- **Keeping the RHI as the resource layer below the HAL.** It would keep
  two public device APIs and the registry and leases the call path must not
  touch; its backend code moves into the HAL instead.
- **Graphs that follow the scene**, a node per object or view. They link
  every frame; counts belong in buffers (section 4.3).
- **A built-in compute gather for graphics leaf copies**, as CUDA's PTX
  kernel. The runtime has no shader compiler, so Vulkan would need
  SPIR-V written by hand, and OpenGL 3.3 has no compute. Buffer copies
  run on every graphics device, and leaf reads are VJP paths rather than
  frame paths. A gather becomes worth it when the compiler emits one per
  Program.
- **Rewriting a tape's launch counts on every call.** A recording in
  flight may still read the last counts; a buffer per distinct count
  triple, written once, keeps them valid, and a schedule launches few
  distinct grids.
- **Runtime guards**: a device status word that device-side checks set and
  later dispatches are predicated on. A failed call is almost always device
  loss, which no predicate recovers; predication needs indirect draws, which
  OpenGL 3.3, OpenGL ES 3.0 and WebGL2 lack, and CUDA conditional graph
  nodes, which need the CUDA device runtime. A conditional update, such as
  skipping an optimizer step whose gradients are not finite, is dataflow in
  the Program (`where(all_finite(g), params - lr * g, params)`).
- **Retrying a failed call in the runtime.** Later calls that read its
  outputs are already queued; running it again after them breaks queue
  order.
- **Frame or step pacing in the runtime.** How many calls run ahead is the
  application's latency policy; the runtime bounds only its own resources,
  and callers pace themselves on timeline points.
