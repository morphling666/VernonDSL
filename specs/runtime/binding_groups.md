# Binding groups and command streams

Status: superseded by [`architecture_v2.md`](architecture_v2.md): bindless
handles in the argument arena replace groups, and bound groups remain only
in the OpenGL tape (its section 15). Not to be implemented further. Was:
proposed. One `COMPILER_CONTRACT_VERSION` and `PROGRAM_VERSION` cut.
It replaces the per-command binding sets of the provider ABI for compute and
graphics on every backend, completes the group partitions of
[`../program/recorded_draws.md`](../program/recorded_draws.md) section 6, and
supersedes the uncommitted partial binding sets
(`VERNON_RUNTIME_PROVIDER_PARTIAL_BINDING_SETS`,
`vernonRuntimeCoreCreateBindingSubset`, `RhiBindingSetCache`), which are
removed. Amended contracts are listed in section 12.

## 1. Problem

Every dispatch and draw is encoded as a complete Stage invocation. Each one
fills every binding value from Program arguments, creates or updates a
binding object that covers the whole layout, hands it to the provider, and
the provider reapplies all of it and retains every object for the command
again. The OpenGL provider additionally reapplies the program, vertex array,
fixed-function state and vertex attributes and reads `glGetError` four times
per draw. Compute pays the same per dispatch; set-based backends allocate and
write descriptors per command.

Measured on OpenGL (Vernon frame benchmark, about 3.9 draws per box): 1000
boxes take 50 ms of host time per frame, about 11 µs per draw, against 3.4 ms
for the legacy GL renderer, which binds only what changed.

The cost does not come from missing caches. It comes from treating bindings
as a per-command argument instead of encoder state: work is proportional to
the layout size times the command count, not to what changed.

## 2. Principles

1. **Bind by update frequency.** Bindings are partitioned into groups that
   change together. The source states the partition: each leaf's `set` is its
   group (section 3).
2. **Bindings are encoder state.** A command stream binds a pipeline, binds
   groups, and dispatches or draws. Bound state is sticky, as on every native
   API.
3. **An immutable group is one native object, made once.** It is the group's
   native form, not a cache: a descriptor set, a descriptor-table range, an
   argument buffer, or on OpenGL and CUDA its resolved values.
4. **Per-command work is proportional to what changed.** A command that
   changes nothing binds nothing, validates nothing, and retains nothing.

## 3. Parameters, placement and groups (language and compiler)

### 3.1 `@vd.params`

`@vd.params` replaces `@vd.binding_group`. A params class has no physical
meaning: an entry parameter of a params type is exactly its leaves, declared
in field order under their dotted paths. Packing and partitioning never look
at the class.

```python
@vd.params
class Geometry:
    position: Annotated[Vec3, vd.attribute(location=0)]
    normal: Annotated[Vec3, vd.attribute(location=1)]
    palette: Annotated[TensorView[Mat4, 1], vd.resource(set=2)]
    model: Annotated[Mat4, vd.uniform(set=2)]
```

- A field takes every annotation an entry parameter of its type takes:
  `attribute`, `builtin`, `varying`, `uniform` and `resource`, with their
  locations and placement. Placement is written on fields, so a params
  class places its leaves the same way wherever it is used.
- A params parameter is valid in a stage when each of its leaves would be
  valid there as a parameter: a class with an `attribute` leaf only in a
  vertex stage, one with a `varying` leaf only in a fragment stage, one
  with a `builtin` leaf where that builtin is an input.
- A params-typed parameter takes no annotation at its use site except
  `When[condition, T]`, which enables or disables all of its leaves; any
  other fails with `params parameter cannot declare`.
- Pinned leaves (section 3.2) of two occurrences of one class claim the same
  bindings or locations and fail as two parameters would. Unpinned leaves
  never conflict.
- The rest of [`../language/contract.md`](../language/contract.md) section
  6.4 is unchanged: a params value is not a Value, helpers and methods take
  it, and autodiff treats it leaf by leaf.

### 3.2 Placement

Placement is per leaf, in every stage kind, Kernels included:

| Annotation | Placement |
|---|---|
| `uniform(set=None, binding=None)` | a by-value leaf |
| `resource(set=None, binding=None)` | a TensorView, Texture or Sampler leaf |

`attribute(location=None, divisor=0)` is unchanged. A vertex attribute is a
vertex-input location, as on every native API, and belongs to no group
(section 3.3).

- `set` is the leaf's group, its update-frequency partition. A leaf without
  `set` is in group 0, as are the values the Runtime derives (resolution).
- `binding` pins a leaf to that binding of its group. It exists for
  interoperation with authored native layouts; the compiler assigns every
  unpinned binding after reserving the pinned ones. A pinned by-value leaf
  keeps a uniform block of its own and is not packed.
- Only what the source states reaches MLIR: `vernon.set` when `set` is
  written and `vernon.binding` when `binding` is written. The compiler maps
  whatever is missing. Its current mapping puts a leaf without `set` in set 0
  and assigns every unpinned binding at the pipeline link; a smarter
  heuristic may replace it without a source change. A `vernon.binding`
  without `vernon.set` is in set 0. No attribute names a group: the group is
  the set.
- A graphics TensorView cannot take `binding`: it spans one binding per
  storage leaf plus the metadata carrier.
- A Kernel Value parameter is by-value without `uniform()`; `uniform(set=...)`
  only places it. This replaces "Kernel Resources bind in argument order"
  in [`../language/contract.md`](../language/contract.md) section 6. Kernel
  placement lands with packing (section 3.4), which rebuilds the Kernel
  layout per group; until then a Kernel leaf with `set` or `binding` fails
  with `kernel parameters bind in argument order`.
- The same leaf in two stages of a pipeline has one placement; authored
  placements that differ fail at the pipeline link.

### 3.3 Groups

- A Stage layout has one group per `set` index its leaves use, and empty
  layouts for unused indices below the highest. No group is made or bound
  at an empty index: no native API binds a set the shaders never read, and
  a convention such as pass, object, geometry, material leaves holes for
  the Stages that skip a frequency. Each group owns its
  TensorView metadata carrier per module, so no carrier mixes groups. The
  carrier split lands with packing; until then one carrier per module sits
  in set 0.
- Set-based targets place group `g` in descriptor set `g` (Vulkan), root
  descriptor table `g` (D3D12) and argument buffer `g` (Metal). OpenGL, CUDA
  and CPU keep flat bindings and record the group of every binding. A Stage
  with more groups than the target allows fails with
  `PROGRAM_GRAPHICS_GROUP_SET_LIMIT`, for Kernels too.
- Each group has a **group layout**: its entries in binding order, with kind,
  stages, and value layout. Two Stages with equal group layouts at the same
  index are compatible at that index, which is what keeps a bound group
  valid across a pipeline switch.
- Vertex inputs belong to no group. They are the pipeline's vertex-input
  state, bound by slot as every native API binds them
  (`vkCmdBindVertexBuffers`, `IASetVertexBuffers`, `setVertexBuffer`,
  `glBindVertexBuffer`), with their stride in the graphics variant key on
  every backend. Vertex-buffer bindings are numbered from 0 across the
  pipeline, at most 16 (`VERNON_RUNTIME_PROVIDER_MAX_VERTEX_BINDINGS`), and
  at most 15 on Metal, which maps binding `b` to buffer index `16 + b` so
  that indices below 16 stay with the groups. Each binding carries its
  instance divisor in the pipeline layout. Vulkan 1.1 fixes strides at
  pipeline creation and dynamic stride needs 1.3 or
  `VK_EXT_extended_dynamic_state`; one rule on every backend keeps Vulkan
  1.1 supported. A stride change selects another variant, made once; it
  never rebinds groups or vertex inputs. A base pipeline has no strides, and
  RuntimeCore rejects a variant without a nonzero stride for every binding.
- Implicit samplers belong to the group of the sampled image they pair with.
  A params leaf stays an endpoint of every Stage that declares its
  parameter, so a sampled Texture leaf a Stage never reads still binds
  there; pairing is
  per Program ([`../program/integer_targets.md`](../program/integer_targets.md)
  section 4): the leaf gets an implicit sampler unless another Stage of the
  pipeline reads it, which pairs it in that Stage.

### 3.4 Packing

The compiler packs by-value leaves; the host never chooses where a value
lives.

- Push constants, root constants and native uniforms belong to the
  pipeline, so only group 0 uses them: group 0's unpinned by-value leaves
  stay there in flattened declaration order while they fit the target's
  inline budget.
- Every other unpinned by-value leaf of a group, in group 0 the ones that
  did not fit, packs into one uniform block per group, shared by the
  pipeline's stages like a binding (pipeline link,
  [`../program/graphics_storage_reads.md`](../program/graphics_storage_reads.md)
  section 3). Leaves are laid out in flattened declaration order under the
  target's uniform-block rules; a leaf only one stage reads keeps its offset
  in both.
- Reflection reports each packed leaf's block and byte offset, and its
  group's block size. A persistent group holds its block in one buffer, a
  transient group in one range of its encoder's transient memory, so a group
  costs at most one uniform binding and one dynamic offset for its values.
- CUDA and CPU pass a group's by-value leaves as its parameter fragment,
  laid out by the same rule.

## 4. Group objects (Runtime)

| Source | Object | Made | Lifetime |
|---|---|---|---|
| Sealed binding set covering one group | persistent | at its first match with a pipeline, per distinct group layout | the set |
| Arguments of a linked schedule's graphics node or pass-table entry | transient | at each call | the call's command stream |

- A persistent group retains its resources once, when it is made. Binding it
  retains the group object for the command encoder; the RHI deduplicates per
  encoder, so a group bound by many commands is retained once.
- A set covers at most one group of each pipeline it matches, plus any
  vertex inputs: it binds every per-draw leaf of that group and no leaf of
  another group, and one set serves every pipeline whose group layout at
  that index is equal. A match that covers part of a group or several
  groups fails with `PROGRAM_BINDING_SET_MISMATCH`. A set's vertex inputs
  are lowered with it (buffers, offsets, strides) and bound as vertex-input
  state, not through its group. A group that holds both per-draw and fixed
  endpoints fails pipeline conformance (`PROGRAM_PIPELINE_CONFORMANCE`),
  because no set could cover it; fixed endpoints belong in group 0.
- Values that stay fixed across calls belong in a sealed binding set, whose
  group is made once; a schedule's record carries only the arguments that
  change, so a post-processing pass called each frame packs only those.

## 5. Command streams

Every Node lowers to a stream on its command encoder:

| Command | Meaning |
|---|---|
| `bind pipeline` | compute pipeline or graphics variant |
| `bind group` | group index and group object |
| `bind vertex buffers` | first binding and buffer references, each with its offset |
| `set dynamic state` | viewport, scissor, stencil reference, index buffer |
| `dispatch` / `draw` | counts only |

- A graphics Node is one render scope; a recorded pass is its recording
  lowered command by command; a standalone graphics or compute Node is the
  one-command stream.
- Group `g` stays bound across a pipeline switch when the two layouts are
  equal at `g`; otherwise the Runtime binds the object made for the new
  layout. A vertex input stays bound at its slot when the new pipeline's
  input there has the same format. Stickiness of recordings follows: a
  set whose group layouts and vertex formats correspond stays bound.
- Each command validates only what it changes: `bind group` checks layout
  compatibility at that index; `dispatch` or `draw` checks that every group
  and vertex input of the bound pipeline is bound.
- The stream is built at invocation entry by the existing recording
  validation walk; draw counts of 0 are inferred from the leading
  dimensions of the bound vertex inputs.
- An encoder carries one stream, shared by every Node of the submitted plan
  that encodes on it: graphics batches, compute and derivative Nodes. Its
  transient groups live as long as the stream, so Nodes sharing an owner
  identity share one group, and a pipeline or group the previous Node left
  bound is not bound again.
- The plan records barriers, copies, timers and debug groups on the native
  encoder between Nodes. The stream is therefore a sequence of segments: the
  first Node to encode begins it, and every later Node resumes it, so the
  provider reapplies only the native state those commands may have changed
  (section 8). A segment boundary never falls inside a rendering scope.
- The plan ends the stream with a finisher that runs once after its last Node
  and before the encoder finishes, also when a Node failed, so no stream
  outlives its encoder. A standalone Stage invocation and a CPU Node, which
  have no plan, begin and end their own stream.

## 6. Work by frequency

| When | Work |
|---|---|
| Pipeline creation | prepare shaders, group layouts and the pipeline layout |
| Set seal or first match | validate, lower its values, make its group object |
| Instance rebind | remake the persistent groups whose parameters changed |
| Invocation | validate the recording, make transient groups, plan render targets once per scope, build the stream |
| Command | apply what changed, then dispatch or draw |

Compute and graphics lower the same way; a dispatch is a command like a
draw, and a standalone Node is a one-command stream through the same
encoder.

**Owners.** Every Stage parameter of a command takes its argument from one
owner, and every slot of a group depends on the owners of the parameters
that decide it:

| Owner | Identity | Changes |
|---|---|---|
| Binding set matched to a pipeline | the match | never; a set is sealed, and seal copies its host bytes |
| Program instance binding | the binding's payload | when a bind installs a new payload; a reused token keeps it |
| Invocation | the invocation | every invocation: Node outputs, Runtime carriers, dispatch-derived views |
| Dynamic state | the recorded dynamic state | at `set_viewport`, `set_scissor` and `set_stencil_reference` |

An argument belongs to its binding only when materialization is a function
of the binding and the plan alone; an argument that needs a Runtime carrier,
a copy, or a shape the invocation decides belongs to the invocation. The
resolution belongs to the dynamic state; an implicit sampler belongs to the
owner of the sampled image it pairs with.

**Group identity.** A group is identified by its layout and the identities
of its slots' owners, never by its values:

- If one binding-set match or one instance payload owns every slot, the
  group is persistent and kept by that owner, found by layout. Several
  instance payloads keep their group on the instance, found by the tuple of
  payloads. Either way it is made when first needed and remade only when an
  identity changes.
- Otherwise the group is transient and kept by the command stream, found by
  the identity tuple, so draws that share a pass or a dynamic state share
  one group.
- Binding a group found this way fills, packs, hashes and validates nothing.
  Values are filled only for a group being made.

**Lowering an owner** happens when its group is made. It validates the
owner's arguments against the parameters, describes its images, fills and
packs its slots, and records what later commands read: on the owner, the
buffers, offsets, leading dimensions and strides of its vertex inputs; on
the group, the buffer accesses it adds to the Node and the images it
samples. A sealed set's matched
argument frame is lowered once per match and reused by every invocation; a
match whose arguments need a Runtime carrier is materialized per invocation
instead.

**The stream** is built by one walk at invocation entry, which is also the
recording validation: a set's hazard check and frame run once per distinct
set in the invocation, and render targets, attachment formats and the
default viewport once per scope. A command holds its pipeline or variant,
its group objects per index, its vertex inputs, its dynamic state and its
counts. The encoder binds a pipeline, variant, group or vertex input only
when it differs from the bound one; the variant is chosen from the pipeline
and the strides of the bound vertex inputs, and draw counts of 0 are
inferred from their leading dimensions.

## 7. RuntimeCore

RuntimeCore keeps the command state of an encoder and validates at the
provider boundary:

```c
vernonRuntimeCorePrepareGroupLayout(provider, descriptor, &group_layout);
vernonRuntimeCoreCreateGroup(group_layout, values, value_count, lifetime, command_encoder, &group);
vernonRuntimeCoreBeginCommands(provider, command_encoder, &commands);
vernonRuntimeCoreBindPipeline(commands, pipeline_or_variant);
vernonRuntimeCoreBindGroup(commands, index, group);
vernonRuntimeCoreBindVertexBuffers(commands, first_binding, buffers, count);
vernonRuntimeCoreDispatch(commands, groups);
vernonRuntimeCoreBeginRendering(commands, rendering);
vernonRuntimeCoreDraw(commands, draw);
vernonRuntimeCoreEndRendering(commands);
vernonRuntimeCoreResumeCommands(commands);
vernonRuntimeCoreUpdateGroup(commands, group, values, value_count);
vernonRuntimeCoreEndCommands(commands);
```

`UpdateGroup` writes the inline values of an updatable group between
batches: batches encoded before it read the previous values and batches
encoded after it, including commands recorded but not yet encoded, the new
ones. Core checks the lifetime, the provider and the values against the
group layout.

`ResumeCommands` starts a segment outside a rendering scope after other
commands were recorded on the native encoder. The bound pipeline, groups
and vertex inputs stay bound in RuntimeCore, which therefore still skips binding them again.
Vertex buffers stay bound across pipeline binds; `BindVertexBuffers`
forwards only the run of bindings whose buffer changed.

Group objects keep the immutable-revision lifetime of today's binding sets
([`design.md`](design.md) section 13): an encoded command pins the group it
binds until the encoder completes. `CreateBindings`, `UpdateBindings` and the
per-command `EncodeDispatch`/`EncodeDrawInvocation` are removed.

## 8. Provider ABI

```c
prepare_group_layout(user_data, descriptor, &group_layout);
prepare_pipeline_layout(user_data, {group_layouts, count, push_constant_size}, &layout);
create_group(user_data, group_layout, values, count, lifetime, encoder, &group);
destroy_group(user_data, group);
encode_bind_pipeline(user_data, encoder, pipeline);
encode_bind_group(user_data, encoder, index, group);
encode_bind_vertex_buffers(user_data, encoder, first_binding, buffers, count);
encode_dispatch(user_data, encoder, groups[3]);
encode_draw(user_data, encoder, draw);
resume_commands(user_data, encoder);
update_group(user_data, commands, group, values, count);  /* optional */
```

`lifetime` is persistent, transient or updatable; a transient group belongs
to `encoder`, may use encoder-scoped memory, and is released with it. An
updatable group is persistent, and its inline values are undefined until
`update_group` writes them; only a provider with `update_group` makes one.
The provider skips native calls whose state is already current.

`resume_commands` reapplies only what the RHI's own commands can change:

| Backend | Reapplied at resume | Why |
|---|---|---|
| OpenGL/ES | all cached context state is invalidated | copies, clears and mipmap generation rebind buffers, textures and framebuffers |
| D3D12 | root signatures, root tables, descriptor heaps and the bound pipeline state | mipmap generation and clears set heaps, a compute root signature and a pipeline state; barriers change the resource states that decide when root tables are rewritten |
| Vulkan | nothing | the RHI records barriers, copies and debug labels, which change no bound state |
| Metal | nothing | compute encoders are made per dispatch, and a render encoder change is already tracked |
| CUDA, CPU | nothing | no native binding state |

Encoder-scoped memory belongs to the RHI recording, not to the adapter. The
RHI already owns the memory that lives exactly as long as a recording, such
as Vulkan and D3D12 frame upload staging, and only the RHI knows when the
recording's work has completed. An adapter asks for it with
`writeCommandTransient(device, encoder, bytes, size, alignment)`, which
returns a native buffer and offset that stay valid until the encoder
completes or is abandoned. Backends whose transient memory is host-visible
also offer `reserveCommandTransient(device, encoder, size, alignment)`,
which returns the same memory with its mapped address and copies nothing,
for data the adapter encodes in place. Each backend suballocates from 64 KiB chunks, or
one the size of a larger value, that its device pools across recordings
(`rhi/staging_pool.h`), so steady-state encoding makes no native buffers.
OpenGL keeps one arena per open encoder and returns its chunks to the pool
from the encoder's deferred cleanup; GL orders the later
`glBufferSubData` that reuses a chunk after the commands that read it.
Vulkan serves it from the command frame's upload staging, whose chunks
also carry the uniform and storage buffer usages. A transient group's
Vulkan descriptor set likewise comes from a descriptor pool of the
command frame (`vulkan::allocateCommandDescriptorSet`), made without
`FREE_DESCRIPTOR_SET` and reset when the frame retires, so transient
groups neither free sets one by one nor grow the device's pool chain.
D3D12 serves it from the command frame's upload staging, which stays in
`GENERIC_READ`; a transient group's constant-buffer and raw-SRV views point
into it, so the group owns no default-heap buffer and records no copy.
CUDA keeps one device-memory arena for the open recording, since a device
records one encoder at a time; writes go through the stream's pinned
staging, and submitting or abandoning the recording retires its chunks
behind a stream event. A transient group's host values bound as storage
point into it instead of a device allocation of their own.
Metal keeps one arena of shared-storage chunks per open recording, keyed by
its command buffer, and returns it to the pool when the recording completes
or is abandoned. A transient group reserves its inline values and argument
buffers there; its argument encoders encode into the reserved range, which
`MTLArgumentEncoder` requires to be an `MTLBuffer`, so a copy-only call
could not serve them.

A persistent group's memory (its inline values, CUDA host storage, Metal
argument buffers) comes from blocks of the device's persistent memory,
which the RHI also owns: `placePersistentBlock(device, bytes, size,
alignment)` copies the bytes in, or, with null bytes on host-visible
backends, returns the block's mapped address for the adapter to fill. A
block takes the power-of-two size class of its size and alignment, at least
256 bytes, from a 64 KiB chunk (or one block, when larger) that serves only
that class (`rhi/block_pool.h`). The group holds its blocks and returns them
when it is destroyed. Binding a group retains it until its encoder
completes, so no command still reads a returned block, and the next group of
that class reuses it at once. Remaking a group on every rebind therefore
copies its bytes and writes its descriptors but makes no native buffer, on
every backend. The memory is the kind transient memory already uses: D3D12
upload-heap chunks in `GENERIC_READ`, whose views need no copy or
transition; Vulkan host-visible, coherent chunks with the uniform and
storage buffer usages; OpenGL buffers written with `glBufferSubData`, which
GL orders after the commands that read a reused block; CUDA device memory
uploaded on the stream that its kernels read on; Metal shared-storage
buffers. Chunks are kept until the device is destroyed, so the pool holds
the peak number of live blocks of each class.

An updatable group's inline values live in an updatable block
(`placeUpdatableBlock`), which only `writeUpdatableBlock(device, encoder,
block, bytes, size)` writes, in the encoder's command order outside
rendering scopes. Updatable blocks are pooled by size class as persistent
blocks are:

| Backend | Updatable memory | Write |
|---|---|---|
| Vulkan | device-local chunks with transfer-destination, uniform and storage usages | encoder upload staging, a barrier from shader reads to the transfer, `vkCmdCopyBuffer`, a barrier back |
| D3D12 | default-heap chunks, whose tracked state the write moves to `COPY_DEST` and back | encoder upload staging and `CopyBufferRegion` |
| Metal | private-storage chunks | encoder transient memory and a blit copy |
| OpenGL/ES | the persistent blocks' buffers | `glBufferSubData`, which GL orders after earlier reads |
| CUDA, CPU | none | no `update_group` |

`update_group` also rewrites the values the provider keeps on the CPU:
OpenGL named uniforms, Vulkan push constants, D3D12 root constants and Metal
direct values, giving the group a new serial so commands apply them again.
It leaves `commands` ready for their next batch, as `resume_commands` would
where its writes change bound state (OpenGL rebinds the array buffer).

| Backend | Group object | `bind group` | Transient memory |
|---|---|---|---|
| Vulkan | descriptor set from a pool per group layout | `vkCmdBindDescriptorSets` at set `g` | descriptor pools of the encoder's command frame; RHI transient memory of the encoder |
| D3D12 | descriptor range in a shader-visible heap | `SetGraphicsRootDescriptorTable`/`SetComputeRootDescriptorTable` at table `g` | descriptor rings of the encoder's command frame; RHI transient memory of the encoder |
| Metal | argument buffer (tier 2) | `setVertexBuffer`/`setFragmentBuffer`/`setBuffer` at index `g`; `useResource` for its resources | RHI transient memory of the encoder, reserved in place |
| OpenGL/ES | resolved bindings; inline uniform-buffer data in persistent blocks of the device | `glBindBufferRange`, texture units, samplers, `glUniform*`, attribute pointers, each only when it differs | RHI transient memory of the encoder |
| CUDA | packed parameter fragment | records the fragment | RHI transient memory of the encoder |
| CPU | packed call-frame fragment | records the fragment | none |

On OpenGL, buffer, texture and sampler bindings are context state and
survive a program switch; `glUniform*` values and vertex-array attributes
are program and vertex-array state and are reapplied after a pipeline bind
only where the new pipeline needs them. Fixed-function state is compared with
the last applied pipeline. On a validation device
([`hot_path_diagnostics.md`](hot_path_diagnostics.md) section 1)
`glGetError` is read once per render scope and encoder end, which reports
the failure for that scope; otherwise a GL error fails the submission's
completion (section 2 there).

## 9. Interaction with other contracts

- Linked schedules ([`architecture_v2.md`](architecture_v2.md) section 7):
  the groups of schedule arguments are transient to the call (section 4).
- Hazards stay static: groups hold only borrowed, read-only resources for
  recorded passes, and Node accesses still decide barriers.
- Autodiff is unchanged: derivative Programs are ordinary Programs and
  lower the same way.
- Push constants serve group 0 only (section 3.4).

## 10. Rollout

The compiler cut, RuntimeCore, the provider ABI and every provider land in
one version cut, with no emulation path. Vulkan, D3D12, OpenGL, OpenGL ES and
CUDA are verified on Windows; Metal needs the macOS suite before the cut
merges. The executor's per-draw contexts, `RhiGraphicsPipeline::bindings`,
`RhiComputePipeline::bindings`, and the partial-binding-set code are deleted.

## 11. Acceptance

1. Groups: a Kernel and a Pipeline whose leaves use sets 0, 1 and 2 report
   three groups with their own carriers and one packed block each on every
   target; a params parameter is reflected identically to its leaves
   declared one by one; too many groups fail with
   `PROGRAM_GRAPHICS_GROUP_SET_LIMIT`.
2. A recorded pass with N draws that bind K distinct sets makes K group
   objects once across frames and issues K group binds and N draws; a mock
   driver shows no rebinding of unchanged state on OpenGL and Vulkan.
3. A pipeline switch keeps compatible groups bound and rebinds the others;
   pixels match sequential invocations on every backend.
4. A compute Node invoked repeatedly with unchanged arguments makes no group
   after its first invocation; rebinding one parameter remakes only its group.
5. Recorded-graphics, compute and autodiff suites pass on every backend.
6. The Vernon frame benchmark at 1000 boxes reaches at most 2 µs of host time
   per draw on OpenGL and Vulkan.

## 12. Contract changes

Status: the specs below describe phases 1 and 2 as implemented. Their parts
that depend on uniform packing (section 3.4), namely per-set carriers, packed
by-value leaves and Kernel placement by `set`, and the per-group command
state of phases 3 to 5, are amended as those phases land. `When[...]` on a
params parameter's use site is still rejected, unlike section 3.1.

- [`../language/contract.md`](../language/contract.md): section 6 takes
  `set` and `binding` on every stage kind and drops "Kernel Resources bind
  in argument order"; section 6.4 becomes `@vd.params` (section 3.1), whose
  fields take placement and whose leaves follow their own `set` instead of
  descriptor set 0, and whose fields take every entry annotation, so
  `binding group field cannot declare` is removed; the
  vertex group's "one bound group object serves a draw" rationale is
  dropped, since vertex inputs bind outside groups; section 6.6 and every
  other spec rename binding groups to params.
- [`../language/static_specialization.md`](../language/static_specialization.md)
  section 4: a params class no longer implies its own descriptor set.
- [`../program/recorded_draws.md`](../program/recorded_draws.md): section 6
  is replaced by section 3 here, and a set covers exactly one group
  (section 4); section 7's per-parameter stickiness becomes per-group;
  section 8 lowers through section 5; section 13 drops the pending group
  partitions and the per-parameter native state.
- [`../program/graphics_storage_reads.md`](../program/graphics_storage_reads.md)
  sections 2 to 4: carriers per group; the link allocates per group from
  `set` instead of per occurrence in set 0, and packs by-value leaves
  (section 3.4).
- [`../compiler/design.md`](../compiler/design.md): Kernel bindings follow
  section 3.2 instead of argument order in set 0.
- [`design.md`](design.md): sections 10 and 13 describe groups and command
  state instead of per-command binding sets.
- [`reflection_binding_architecture.md`](reflection_binding_architecture.md)
  and [`tensor_view_metadata_abi_plan.md`](tensor_view_metadata_abi_plan.md):
  group indices in reflection and carriers per group.

## 13. Rejected alternatives

- **Native binding objects per set, pipeline and slot subset** (the partial
  binding sets): they fake objects OpenGL does not have, depend on slot
  subsets instead of a layout unit, and still reapply all state per command.
- **Slot-granular `bind values` as the ABI**: OpenGL handles it, but
  set-based backends would rebuild a descriptor set on every change and every
  command would validate values.
- **Emulating command state on set-based backends**: it keeps their
  per-command descriptor work; every backend has native group binding.
- **Bindless resource tables now**: optimal for GPU-driven passes, but it
  needs resource indexing in the language and descriptor indexing on every
  target. Groups stay compatible with it: a group can later hold table
  indices instead of resources.
- **One complete invocation per command**.
- **Finding groups by their values**: every command fills, packs and hashes
  all its values to find groups that did not change, which kept per-draw
  work proportional to the layout (39 ms at 1000 boxes).
- **Push-constant ranges per group** for its by-value leaves: every group
  bind pushes its bytes again, so per-command work grows with the group, and
  all groups share the 128-byte minimum push budget of Vulkan.
- **A group per top-level binding-group occurrence**: it gives a
  convenience class physical meaning, so moving a field into or out of a
  class changes the layout, a class cannot mix frequencies, and packing
  served groups only, not group 0.
- **Vertex buffers as group entries** (an earlier form of this design): no
  native API binds vertex buffers through descriptor sets, tables or
  argument buffers, so every group bind also issued a vertex-buffer bind
  outside its object, and attributes would need a `set` that means nothing
  to their location.
- **A layout declared on the Pipeline or Kernel, or as a specialization
  variant**: the partition decides packing and must be known when the
  module compiles, and a variant multiplies compiled forms for what is a
  property of each leaf.
- **Placement at the use site of a params parameter**: it would be a second
  way to place leaves, beside field annotations, with precedence rules
  between them.
- **One spilled uniform buffer per by-value leaf**: each costs a binding;
  OpenGL and OpenGL ES guarantee 12 uniform blocks per stage, and D3D12
  sizes each constant buffer in 256-byte steps.
- **Composite groups** for sets that cover part of a group or several
  groups: every such recording makes a transient group per invocation, and
  it keeps a second binding path beside the one-object group.
- **A stream per Node or per graphics batch**: every Node began and ended its
  own stream, so it rebound its pipeline and groups, remade transient groups
  that an earlier Node had made for the same owners, and drained `glGetError`
  at each begin and end.
- **Ending the stream at every RHI command instead of resuming it**: correct
  without a provider callback, but it costs the same as a stream per Node,
  since the plan records a barrier or debug group between almost every pair
  of Nodes.
- **A native buffer per persistent group**: a uniform rebound every frame
  remakes its group, so every frame created a buffer on every backend, and
  D3D12 also copied the group's default-heap buffer once per encoder.
- **Persistent blocks bumped out of staging-pool arenas**
  (`rhi/staging_pool.h`): an arena retires whole chunks at once, so one
  long-lived group pins its chunk, and the blocks of short-lived groups
  around it are never reused.
- **Treating host values as invocation-owned**, so that groups holding only
  host values are transient: it loses "a Node invoked with unchanged
  arguments makes no group" and re-uploads unchanged uniforms on every
  invocation.

Future: [`stored_plans.md`](stored_plans.md) keeps a plan whose structure is
unchanged between Executes and replays its commands with patched payload,
without retained command buffers. It changes the group identity of instance
bindings (section 6) from the payload to the binding's structure.

Outside this design: Vernon's scene synchronization (about 13% of host time
in the benchmark).
