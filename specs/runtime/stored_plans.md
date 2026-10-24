# Stored execution plans

Status: proposed. Amends `binding_groups.md` sections 4, 6 and 13 and the
recording API of `../program/recorded_draws.md`. Followed by batched RHI
command packets (section 11), which replay the stored commands.

Every Execute rebuilds the physical work of its invocation: it validates the
recordings, builds the invocation values, materializes every Node frame,
plans transfers, render targets and draws, builds the Command DAG and its
barriers, remakes groups and encodes the stream. In the Editor (OpenGL, cave
and acrobot, 255 draws) that is about 1.41 ms per frame on the main thread,
of which the GL driver is 0.26 ms; heap traffic (0.21 ms), reference-count
atomics (0.16 ms), mutexes (0.12 ms) and registry pins (0.08 ms) come mostly
from rebuilding and per-command validation. Nothing structural changed
between those frames.

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
  scissor, stencil reference, clear values, and upload bytes of a transfer
  whose range is unchanged.
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

A recording becomes a persistent object owned by a Program instance and one
graphics Node, replacing `vernonRuntimeProgramInvocationRecordGraphics`:

```c
vernonRuntimeGraphicsRecordingCreate(instance, node, &recording);
vernonRuntimeGraphicsRecordingReset(recording);           /* structure edit */
vernonRuntimeGraphicsRecordingAppend(recording, commands, count);
vernonRuntimeGraphicsRecordingSetDraw(recording, draw, &command);  /* payload */
vernonRuntimeGraphicsRecordingSetDynamicState(recording, draw, &state);
vernonRuntimeProgramInvocationBindRecording(invocation, recording);
vernonRuntimeGraphicsRecordingDestroy(recording);
```

`Reset` and `Append` advance the structure revision; `SetDraw` and
`SetDynamicState` advance the payload revision of one draw and are rejected
if they would change its pipeline, sets or kind. Validation (`recorded_draws.md`)
runs when a plan is built for a new structure revision. Execute freezes the
recording's current revisions, as it freezes bindings.

A host that re-records an unchanged draw list every frame gets no reuse;
the Engine keeps one recording per pass and edits it.

## 5. The stored plan

A stored plan holds what the builder produced for one signature:

- the Node frames, transfer plans and managed graphics Nodes;
- the merged Command DAG and its interior barriers;
- the persistent groups it binds, and the layout of its transient writes;
- the command stream as an ordered list of commands with **patch sites**:
  where each payload input lands (a transient write, a dispatch grid, a draw
  count, a dynamic state, a clear value).

A stored plan is immutable once built and shared by every submission that
replays it. It contains no per-submission state.

**Per-submission state** stays per Execute: the transient encoder memory,
retained leases, timers, completion effects, the `SubmissionRecord`, and the
publication transaction.

**Submission slots.** Invocation-owned resources (owned images, device alias
buffers, Runtime carriers, retained pullback storages) would be written by
two submissions of one plan in flight. A stored plan therefore has one slot
per submission it may have in flight, up to the instance's submission bound
(`pipelined_submission.md` section 7). A slot owns its resources and the
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

## 7. Execute

1. Freeze bindings, controls and recordings; form the signature.
2. On a match, take the plan's next free slot. Otherwise build as today,
   store the result and take its first slot.
3. Write every patch site from the frozen payload: transient bytes, grids,
   counts, dynamic state.
4. Compute the entry barriers, replay the stream on the encoder, submit.

Replaying before section 11 calls the existing encode path with the stored
commands; it already skips planning, validation, group lookup and DAG
construction.

## 8. Verification

The stored plan is a structure whose parts must stay consistent with a
fresh build. A test option builds a fresh plan on every Execute and compares
it with the stored one (commands, barriers, group identities modulo slots,
patch sites), and runs on every backend in the runtime test suite and in a
libFuzzer target that interleaves binds, recording edits and Executes.
Counters of plan builds, group creations and transient writes let tests
assert that a steady-state Execute builds nothing.

## 9. Acceptance

- Every structural input of section 2 has a test that changes it alone and
  observes a rebuild; every payload input has one that observes none and
  checks the GPU result.
- Two submissions of one plan in flight write different slots.
- A pullback applied in a training loop reuses its backward plan.
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
- **Native retained command buffers** (Vulkan secondary buffers, D3D12
  bundles, Metal indirect command buffers): not on OpenGL, CUDA or CPU, and
  they freeze state that patch sites must change. Replaying stored commands
  serves every backend.

## 11. Batched RHI command packets

Stored commands replay through the public RHI C ABI as packet batches: one
call validates and pins the device, the encoder and the batch's resources,
then the backend decodes plain packets (bind pipeline, bind group with
dynamic offsets, dynamic state, draw, dispatch, copy, barrier, debug label,
timer). This replaces one validated RHI call per command. Its packet layout
and ABI are specified separately once the stored plan lands, since the
stored command list is what it encodes.
