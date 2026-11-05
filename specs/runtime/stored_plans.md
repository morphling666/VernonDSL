# Stored execution plans

Status: superseded by [`architecture_v2.md`](architecture_v2.md), whose
one recording per linked schedule replaces stored plans (its sections 7 and 15);
this document describes the code until that migration removes it. Sections
4 and 12 implemented; section 11 in progress; sections 3 and 5 to 10 will
not be implemented. Amends
`binding_groups.md` sections 4, 6 and 13 and the recording API of
`../program/recorded_draws.md`. A stored plan holds resolved provider command
packets (section 11) from its first implementation; there is no interim
replay through the per-command encode path. Diagnostics on the hot path are
specified in [`hot_path_diagnostics.md`](hot_path_diagnostics.md) and GPU
batching in [`../program/indirect_draws.md`](../program/indirect_draws.md).

Every Execute rebuilds the physical work of its invocation: it validates the
recordings, builds the invocation values, materializes every Node frame,
plans transfers, render targets and draws, builds the Command DAG and its
barriers, remakes groups and encodes the stream. In the Editor (OpenGL, cave
and acrobot, 255 draws) that is about 1.41 ms per frame on the main thread,
of which the GL driver is 0.26 ms; heap traffic (0.21 ms), reference-count
atomics (0.16 ms), mutexes (0.12 ms) and registry pins (0.08 ms) come mostly
from rebuilding and per-command validation. Nothing structural changed
between those frames.

With recordings persisted (section 4), the Editor drawing 1000 boxes
(OpenGL, about 5,700 draws) spends 5.7 ms per frame in Execute, of which the
driver calls that draw, bind and set state take about 0.1 ms. The rest:

| Share | Cause | Removed by |
|---|---|---|
| 45% | device-registry lookup, lock, reference counts and device-ownership check on every draw and group bind | packets with resolved resources (section 11) |
| 23% | planning, group binding, recording validation and hazard sets | the stored plan (sections 5 to 7) |
| 15% | driver-thread synchronization from `glGetError` scopes, per-edge query creation and debug groups | `hot_path_diagnostics.md` |
| 15% | heap allocation and other per-Execute work | section 7.1 |

Removing all of it leaves the driver calls, the packet decode and the
signature comparison.

`ResolvedExecutionPlan` is already resolved once per executable. This
document keeps the next layer, the physical plan of one invocation, across
Executes: an Execute whose structure matches a stored plan patches its
payload and replays it.

## 1. Scope

The mechanism is shared by every workload, because every workload goes
through the same layer:

| Workload | Stored | Patched per Execute |
|---|---|---|
| Compute Stages | dispatch order, transfers, barriers, groups | by-value bytes, grids |
| Graphics, unrecorded | scopes, render targets, draws, barriers, groups | by-value bytes, clear values, viewport |
| Graphics, recorded | the validated recording and its plan | draw counts, dynamic state, by-value bytes |
| Transfers and publication | upload, copy, readback and commit nodes | upload bytes |
| Pullbacks | the backward plan, stored on the executable | cotangent bytes, by-value bytes |
| CPU execution | Node frames and the Node order | by-value bytes, grids |

Standalone Stage invocations are one-command plans and use the same cache.
A workload whose structure changes every Execute takes the rebuild path,
which stays what it is today plus one signature comparison.

## 2. Structure and payload

Every input the plan builder reads is either structural or payload. The
split follows the resolved plan, never names or values:

- **Payload**: the bytes of host-projected parameters
  (`stage_binding_plan.cpp` `hostProjected`; `design.md` section 5), compute
  grids, draw, instance and index counts and first elements, viewport,
  scissor, stencil reference, clear values, and the bytes and length of an
  upload whose destination is unchanged and whose length stays within the
  destination's extent; the slot's transient memory is sized for the
  extent.
- **Structural**: everything else, including resource identities (RHI
  handle and generation), carriers, rank, shape, strides, offset and extent
  of every argument (host-projected included, since they decide packed
  sizes), attachment identities and load and store operations, the pipelines
  and sets of a recording and its command order, the completion mode, and
  whether pullbacks are retained.

Shape is structural although it is invocation data in the ABI: it decides
transfer ranges, residency, carrier packing and Node frames. Treating it as
payload would need a second, shape-polymorphic plan, which no workload has
asked for.

The **signature** of an Execute is the tuple of its structural inputs. Each
contributor exposes a structure revision, so the signature is a short array
of integers compared in full; it is never a hash of values.

## 3. Bindings

A Program instance binding gets two revisions:

- Binding an argument whose structure equals the current one keeps the
  binding's structure object and installs new bytes as a new payload
  revision. Copy-on-write snapshots keep their bytes as today.
- Binding an argument with a different structure installs a new structure
  object and advances the structure revision.

The binding token keeps its meaning for host callers: a reused token still
skips the bind. It no longer decides group identity.

**Group identity** (amends `binding_groups.md` section 6): an instance
binding owns its group through its structure object, so a payload revision
remakes nothing. A slot holding bytes (an inline value, a spilled uniform, a
host-storage value) is not filled into the persistent group. The group holds
the slot's layout and a dynamic offset; each Execute writes the bytes into
the transient memory of its encoder and binds the group with that offset:

| Backend | Dynamic offset |
|---|---|
| Vulkan | `UNIFORM_BUFFER_DYNAMIC` / `STORAGE_BUFFER_DYNAMIC` descriptors, offsets at `vkCmdBindDescriptorSets` |
| D3D12 | root CBV or SRV at the transient address |
| Metal | buffer offset at bind |
| OpenGL/ES | `glBindBufferRange` on the transient buffer, or `glUniform*` for native uniforms |
| CUDA, CPU | the packed parameter fragment of the launch |

The provider ABI's `bind group` takes the group's dynamic offsets. Because a
group's by-value leaves pack into one block (`binding_groups.md` section
3.4), a group writes one range and takes one offset for them per Execute.
Groups of sealed sets keep their bytes, because a sealed set never changes.

## 4. Recordings

A recording is a persistent object owned by a Program instance and one
graphics Node, replacing `vernonRuntimeProgramInvocationRecordGraphics`:

```c
vernonRuntimeGraphicsRecordingCreate(instance, node, &recording);
vernonRuntimeGraphicsRecordingReset(recording);                /* structure */
vernonRuntimeGraphicsRecordingAppend(recording, commands, count, leases, lease_count);
vernonRuntimeGraphicsRecordingSetCommand(recording, index, &command);  /* payload */
vernonRuntimeProgramInvocationBindRecording(invocation, recording);
vernonRuntimeGraphicsRecordingDestroy(recording);
```

`Reset` and `Append` edit the structure. `SetCommand` edits the payload of
one command: it replaces a viewport, scissor, stencil reference or draw with
one of the same kind, and a draw keeps whether it is indexed and its index
buffer, so pipelines, sets and command kinds never change through it. One
entry point covers draws and dynamic state because both are payload
commands addressed by position.

Validation (`recorded_draws.md`) runs at the first Execute after an edit and
is kept until the next one; it also keys on the invocation's initial dynamic
state, which it reads. The hazard check depends on what each invocation
writes, so every Execute repeats it over the distinct sets and index ranges
the validation kept. Execute freezes the validated recording, as it freezes
bindings: a submission keeps what it drew while the host edits the
recording for the next frame. Until the stored plan (section 5) exists, any
edit discards the validation; the plan is where payload edits stop
invalidating structure.

A host that re-records an unchanged draw list every frame gets no reuse;
the Engine keeps one recording per pass and edits it.


## 5. The stored plan

A stored plan holds what the builder produced for one signature:

- the Node frames, transfer plans and managed graphics Nodes;
- the merged Command DAG and its interior barriers;
- the persistent groups it binds, and the layout of its transient writes;
- the command stream as provider packet batches (section 11) whose objects
  and resources are resolved once at build, with **patch sites**: where each payload input
  lands (a transient write, a dispatch grid, a draw count, a dynamic state,
  a clear value). A patch site is an index into the Execute's payload array;
  packets read their payload from it at replay, so the stored packets are
  never written.

A stored plan is immutable once built and shared by every submission that
replays it. It contains no per-submission state. It retains every
object its packets name, once, for its own lifetime, so neither a replay
nor a submission retains objects one by one.

**Per-submission state** stays per Execute: the transient encoder memory,
retained leases, timers, completion effects, the `SubmissionRecord`, and the
publication transaction.

**Submission slots.** Invocation-owned resources (owned images, device alias
buffers, Runtime carriers, retained pullback storages) would be written by
two submissions of one plan in flight. A stored plan therefore has one slot
per submission it may have in flight, up to the instance's submission bound.
A slot owns its resources and the
persistent groups that reference them; commands name such resources by plan
index and the replay resolves them through the slot. A slot is reused only
after its previous submission retires, which `SubmissionQueue::admit`
already waits for.

**Entry barriers.** Interior barriers are stored. The transitions from the
state a resource has when the submission starts are computed at replay from
the RHI's resource state, because another plan may have used the resource
in between.

## 6. Cache

- Graph and standalone invocations store plans on their Program instance;
  backward plans are stored on the executable, because every training step
  creates a fresh pullback. A pullback's signature includes the slot of the
  forward plan whose retained storages it reads, so steps reuse it with the
  period of the slot ring.
- Each owner keeps at most four plans, evicting the least recently used.
  An evicted plan is destroyed when its last in-flight submission retires.
- A build that fails stores nothing.
- A plan retains the objects it names, so destroying a buffer, image or
  group that a stored plan names only drops the caller's handle. The
  destroyed handle's generation changes the signature of the next Execute;
  that rebuild also evicts every stored plan of the owner that names a
  destroyed handle, so the object is freed at the next rebuild, or with its
  owner, after the plan's submissions retire.

## 7. Execute

1. Freeze bindings, controls and recordings; form the signature.
2. On a match, take the plan's next free slot. Otherwise build as today,
   store the result and take its first slot.
3. Write the payload array from the frozen payload (grids, counts, dynamic
   state, clear values) and the transient bytes into the slot's transient
   memory.
4. Compute the entry barriers, replay the packet batches on the encoder with
   one provider call per batch, submit.

A recorded pass whose recording was edited only through `SetCommand` keeps
its plan: the edit changes payload. `Reset` and `Append` change structure
and rebuild the plan of the Node's invocation at the next Execute.

### 7.1 Allocation

A steady-state Execute allocates nothing on the heap:

- Transient bytes, payload arrays, entry-barrier lists and the
  `SubmissionRecord` live in storage of the slot, sized when the plan is
  built and reused when the slot is reused.
- Completion effects, retained leases and timer storage come from pools of
  the Program instance and the device, which only grow.
- Diagnostic paths and messages are built only on failure, as recording
  validation already does.

A rebuild allocates freely; the counters of section 8 report it.

## 8. Verification

The stored plan is a structure whose parts must stay consistent with a
fresh build. A test option builds a fresh plan on every Execute and compares
it with the stored one (commands, barriers, group identities modulo slots,
patch sites), and runs on every backend in the runtime test suite and in a
libFuzzer target that interleaves binds, recording edits and Executes.
Counters of plan builds, group creations, transient writes, heap
allocations (a test allocator hook) and RHI device lookups let tests assert
that a steady-state Execute builds and allocates nothing and looks the
device up once per packet batch. `StoredPlan::validate()` checks that every
packet's resource index names a retained or slot object, that patch sites
point at packets taking payload, and that interior barriers cover every
hazard between the plan's commands.

## 9. Acceptance

- Every structural input of section 2 has a test that changes it alone and
  observes a rebuild; every payload input has one that observes none and
  checks the GPU result.
- Two submissions of one plan in flight write different slots.
- A pullback applied in a training loop reuses its backward plan.
- A steady-state Execute makes no heap allocation and one device lookup per
  packet batch, on every backend.
- A stored plan naming a destroyed buffer still replays correctly for its
  in-flight submissions, and the buffer is freed after the next rebuild.
- The Editor profile shows no plan builds and no group creations after the
  first frames, with a before and after measurement.

## 10. Rejected alternatives

- **Hashing a re-recorded draw list every frame**: no API change, but the
  host still records and the Runtime still walks and hashes every command.
- **One stored plan per owner**: an Engine that alternates two passes or
  two viewports through one instance would rebuild every Execute.
- **Storing backward plans on the pullback**: a pullback lives for one step.
- **Finding groups by values or treating payload bytes as group identity**:
  rejected in `binding_groups.md` section 13; per-frame uniforms would remake
  their groups every frame, which is the churn measured above.
- **Per-group buffers updated in place**: a submission in flight still reads
  the previous bytes.
- **Replaying stored commands through the per-command encode path first**,
  with packets later: it keeps the per-command registry lookups and
  reference counts, the largest cost measured, and builds a replay path the
  packets would delete.
- **Weak, generation-checked references from a plan with per-submission
  pins**: every Execute would again touch every resource it names. The plan
  retains each object once instead and is evicted when one is destroyed.
- **Native retained command buffers** (Vulkan secondary buffers, D3D12
  bundles, Metal indirect command buffers): not on OpenGL, CUDA or CPU, and
  they freeze state that patch sites must change. Replaying stored commands
  serves every backend.

## 11. Batched provider command packets

Today every provider command resolves the objects it names, and the RHI
adapters then call into the RHI for each draw and group bind: a registry
lookup, the encoder lock, a device-ownership check and a deferred cleanup
entry that retains the pipeline, group or buffer. That is correct per call
and is the largest cost measured above. A packet batch pays it once per
batch and once per object at batch creation.

Packets live in the provider ABI, not in the RHI. Pipelines and groups are
provider objects: the RHI adapters own them and issue the native calls, the
CPU backend and foreign providers implement the same ABI, and the RHI owns
only resources, encoders, scopes and barriers. Decoding in the adapter costs
the same per packet as decoding in the RHI would (a switch and the native
call); the layers above it are crossed once per batch.

### 11.1 Recording and batches

```c
vernonRuntimeCorePacketsCreate(provider, &packets);
vernonRuntimeCorePacketsBindGraphicsVariant(packets, variant);   /* and the other commands */
vernonRuntimeCorePacketBatchCreate(packets, &batch);             /* provider create_packet_batch */
vernonRuntimeCoreEncodePackets(commands, batch, payload, payload_count);  /* provider encode_packets */
vernonRuntimeCorePacketBatchDestroy(batch);
vernonRuntimeCorePacketsReset(packets);
```

- Core records packets with the validation the per-command functions did,
  at the call that appends them: compatibility of pipelines and groups,
  complete groups and vertex buffers at each draw and dispatch, scope
  balance and attachment checks. A bind of what is already bound appends
  nothing. A recording retains the Core pipelines and groups it names and
  keeps one entry per distinct object and per distinct resource reference.
- A batch is self-contained: it starts with nothing bound and ends outside a
  rendering scope with balanced debug groups. Between batches the encoder's
  owner records its own RHI commands and calls `resume_commands`, as before.
- Payload values are `u32` words in the recording's payload array, named by
  packets through payload indices: draw counts, first elements, viewport,
  scissor and stencil reference, dispatch grids, clear values. A one-use
  batch is encoded with the recording's payload; a stored plan patches a
  copy. `EncodePackets` checks each payload site against the limits the
  recording established (non-zero counts, index ranges within the index
  buffer, stencil reference, finite clear values), once per site.
- A recording that binds a transient group is encodable only on that
  group's command encoder.

### 11.2 Packets

Each packet is a `u32` header (opcode in the low 16 bits, word count
including the header in the high 16 bits) followed by its operands. Object
operands index the batch's object table (pipelines and groups), resource
operands its resource table (vertex and index buffers), and rendering
operands its rendering table (attachments, load and store operations).

| Opcode | Operands | Payload |
|---|---|---|
| `BEGIN_RENDERING` | rendering | clear colors, depth, stencil |
| `END_RENDERING` | | |
| `BIND_PIPELINE` | object | |
| `BIND_GROUP` | group index, object | |
| `BIND_VERTEX_BUFFERS` | first binding, count, resources | |
| `DRAW` | topology | vertex and instance counts, first instance, viewport, scissor, stencil reference |
| `DRAW_INDEXED` | topology, index resource | index and instance counts, first instance, base vertex, first index, viewport, scissor, stencil reference |
| `DRAW_INDEXED_STREAM` | topology, index resource, records and counts resources (both none for a host stream) | host addresses of records and counts, first record, max records, count index, viewport, scissor, stencil reference |
| `DISPATCH` | | grid |
| `PUSH_DEBUG_GROUP`, `INSERT_DEBUG_MARKER` | label bytes | |
| `POP_DEBUG_GROUP` | | |

`DRAW_INDEXED_STREAM` is the draw stream of `architecture_v2.md` section
16.2; it serves indirect draws (`../program/indirect_draws.md`) too.
Dynamic offsets (section 3) add payload to
`BIND_GROUP`. Barriers, copies and timer edges stay RHI commands recorded
between batches.

### 11.3 Providers

`create_packet_batch` may allocate and look objects up; `encode_packets` is
the hot path and must not. An RHI adapter meets that with four rules, each
checked by a counter test:

1. **Objects.** Create retains each distinct pipeline and group once and
   resolves each distinct resource once, holding an RHI lease, so the batch
   keeps them alive and knows their native handles. Encode retains the batch
   once for the encoder's deferred cleanup and names nothing else.
2. **Commands.** The draw and dispatch counts and the rendering-scope
   bookkeeping reach the RHI once per batch and once per scope, never per
   draw.
3. **Writes.** The resources the batch's groups write are collected at
   create and recorded on the encoder once per encode.
4. **Labels.** Debug packets are dropped at create when the device does not
   emit labels (`hot_path_diagnostics.md` section 4); otherwise each one is
   an RHI debug command.

| Provider | Decode |
|---|---|
| OpenGL/ES | GL calls with the existing state cache; `BIND_GROUP` applies the group's resolved bindings, diffed against the context |
| Vulkan | `vkCmd*` on the encoder's command buffer |
| D3D12 | command-list calls |
| Metal | encoder calls |
| CUDA | launches |
| CPU | the Node's kernel |

### 11.4 Scope

Packet batches are the provider ABI's only encoding path: the per-command
encode callbacks and Core functions are removed, and the Runtime records
one batch per encode callback of the RHI command executor, a one-use batch
on the rebuild path and a stored one in a stored plan.

## 12. Stored graphics scopes

The first stored form, per rendering scope, of the stored plan of section
5. It removes planning, group making, packet recording and validation from
the rendering scopes of a steady-state Execute; transfers, compute Nodes and
the Command DAG are still built per Execute.

**Keys.** Each graphics Node has a key: the recording's validated structure
(or the draw and dynamic-state controls of a Node without one), the bound
RenderPass control,
the Node, the attachment references, and per frame parameter its structure
(kind, slot, value layout, rank, shape, strides, offset and extent) and
resources. Host bytes are payload (section 2) and are not keyed. A Node
whose frame needs copies or carriers, or whose binding sets are lowered per
invocation, has no key; a scope with an unkeyed member is encoded without
being stored.

**Scopes.** A scope is stored on the Program instance under the Node that
opens it, with the keys of its members, what fusing and hazard tracking read
of each member (targets, sampled reads, buffer reads), its packet batch, and
weak references to the objects its keys name. A scope replays only while
they all live, since an object made after one died may take its address. An
instance keeps the four most recently published scopes per opening Node. A
Node opening a scope whose stored first member matches its key replays it as
long as each following Node matches the next stored member; a Node that does
not, or a scope that ends early, plans the Nodes matched so far after all.
Matched Nodes skip lowering their binding sets, planning and recording.

**Draw values.** A recording's draw values (`recorded_draws.md` section 7)
are not keyed. Each stored recorded draw keeps what planning its values
reads (its vertex input records, topology and resolution) and the position
of its words in the batch's payload. A matched Node whose draw values are
not the stored ones plans only the draws whose draw command or dynamic
state differs, as a fresh plan would, and the replay encodes a copy of the
stored payload with their words replaced. A draw whose values fail to plan,
or plan to another resolution, which the stored groups were filled for,
makes the Node not match. The read set stays valid because a draw's index
read is its whole index view, which `SetCommand` cannot change. Culling by
instance count, as a moving view does every frame, thus replays the stored
scope.

Each stored member also keeps the values it last matched and the words
planned for their draws that differ from the stored ones, so a match plans
only the draws whose values differ from both; matching the same values
again plans nothing. Planning reads only the stored member and the values,
so what a match planned stays right whether or not its replay was encoded,
and concurrent invocations replace it atomically. Without it, a draw culled
since the scope was built was planned again in every replay; for 1000
orbiting boxes matching fell from 4.4% to 1.7% of the main thread.

**Argument arena.** The arena is a scope's own groups: every group of a
keyed member that no binding set owns alone belongs to the scope, made
`UPDATABLE` (`binding_groups.md` section 8) and found within the scope by
the arguments it holds. Each such group keeps a site: its binding plan and
prepared layouts, the resolution, and per Stage parameter the index of the
argument in its member's frame or, for a binding set's argument, the
argument itself. A replay fills each site from the frames of the Nodes it
matched and writes the groups whose inline bytes changed, after flushing and
before encoding the stored batch, so a moving camera rewrites one view
uniform and replays every draw unchanged. Replays of one scope are
serialized, because its CPU-side values (OpenGL named uniforms, Vulkan push
constants, D3D12 root constants) are read when a batch is encoded.

The arena is device memory written by copies in command order rather than
per-slot host-visible memory (as `execution_model.md` section 4 first
proposed), so one stored batch and one set of groups serve every submission
in flight: each submission's copy is ordered after the earlier submissions'
reads by the barrier that precedes it. Per-slot memory would need one batch
and one set of groups per slot. Providers without `update_group` key host
bytes, and an argument that a Program instance binding owns by the binding,
as before.
