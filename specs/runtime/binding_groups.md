# Binding groups and command streams

Status: proposed. One `COMPILER_CONTRACT_VERSION` and `PROGRAM_VERSION` cut.
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
   change together: what the pass or invocation fixes, and each top-level
   binding-group occurrence, which is how sources already express per-object
   and per-material data.
2. **Bindings are encoder state.** A command stream binds a pipeline, binds
   groups, and dispatches or draws. Bound state is sticky, as on every native
   API.
3. **An immutable group is one native object, made once.** It is the group's
   native form, not a cache: a descriptor set, a descriptor-table range, an
   argument buffer, or on OpenGL and CUDA its resolved values.
4. **Per-command work is proportional to what changed.** A command that
   changes nothing binds nothing, validates nothing, and retains nothing.

## 3. Groups (compiler)

- A Stage layout has group 0, holding the ungrouped endpoints and the
  values the Runtime derives (resolution), and group `k + 1` for top-level
  binding-group occurrence `k` in declaration order. Each group owns its own
  TensorView metadata carrier per module, so no carrier mixes groups. This
  replaces "all ranges are in descriptor set 0" in
  [`../language/contract.md`](../language/contract.md) section 6.4.
- Set-based targets place group `g` in descriptor set `g` (Vulkan), root
  descriptor table `g` (D3D12) and argument buffer `g` (Metal). OpenGL, CUDA
  and CPU keep flat bindings and record the group of every binding. A Stage
  with more groups than the target allows fails with
  `PROGRAM_GRAPHICS_GROUP_SET_LIMIT`, for Kernels too.
- Each group has a **group layout**: its entries in binding order, with kind,
  stages, and value layout. Two Stages with equal group layouts at the same
  index are compatible at that index, which is what keeps a bound group
  valid across a pipeline switch.
- Vertex buffers are group entries but bind as vertex inputs, with their
  stride in the graphics variant key on every backend. An ungrouped
  attribute is an entry of group 0; an attribute leaf of a vertex group
  (`../language/contract.md` section 6.4) is an entry of that group, so a
  set covering the group binds the draw's vertex buffers with the rest of
  its geometry in one persistent group. Vertex-buffer slots (Vulkan and
  D3D12 bindings, Metal buffer indices, OpenGL attribute bindings) are
  numbered across the pipeline in entry order, so a group's slots are
  fixed by its pipeline layout like its descriptor bindings. Vulkan 1.1 fixes
  strides at pipeline creation and dynamic stride needs 1.3 or
  `VK_EXT_extended_dynamic_state`; one rule on every backend keeps Vulkan 1.1
  supported. A stride change selects another variant, made once; it never
  rebinds the groups, whose layouts do not include strides.
- Implicit samplers belong to the group of the sampled image they pair with.
  A group leaf stays an endpoint of every Stage that declares the group, so
  a sampled Texture leaf a Stage never reads still binds there; pairing is
  per Program ([`../program/integer_targets.md`](../program/integer_targets.md)
  section 4): the leaf gets an implicit sampler unless another Stage of the
  pipeline reads it, which pairs it in that Stage.
- A binding group is only a combination of flattened leaves: each leaf is
  the endpoint it would be ungrouped, placed in its group. A by-value leaf
  therefore spills to its own uniform buffer in the group's set, as an
  ungrouped value that cannot stay inline does; push constants and native
  uniforms belong to the pipeline, so they stay group 0's. The group object
  holds the spilled data: a persistent group uploads it when it is made, and
  a transient group writes it to its encoder's transient memory. CUDA and CPU
  pass the leaves as parameters of the group's fragment.
- Packing several spilled values into one buffer is a separate, later design
  that applies to every set, group 0 included. Until it lands, each spilled
  leaf costs a binding: OpenGL and OpenGL ES guarantee 12 uniform blocks per
  stage, and D3D12 sizes each constant buffer in 256-byte steps.

## 4. Group objects (Runtime)

| Source | Object | Made | Lifetime |
|---|---|---|---|
| Sealed binding set covering one group | persistent | at its first match with a pipeline, per distinct group layout | the set |
| Program instance values of a Node's group | persistent | when a binding token of one of the group's parameters changes | the instance |
| Values fixed for one invocation (pass values, resolution after `set_viewport`) | transient | at invocation encode | the command encoder |
| Sets that together cover one group in a recording | composite, transient | once per distinct combination in the invocation | the command encoder |

- A persistent group retains its resources once, when it is made. Binding it
  retains the group object for the command encoder; the RHI deduplicates per
  encoder, so a group bound by many commands is retained once.
- A set normally covers exactly one group, and one set serves every pipeline
  whose group layout at that index is equal. A set that covers part of a
  group or several groups stays valid (`recorded_draws.md` section 6) through
  composite groups.
- Standalone invocations that rebind nothing reuse every persistent group,
  so a post-processing Node invoked each frame with unchanged arguments makes
  no group at all after its first frame.

## 5. Command streams

Every Node lowers to a stream on its command encoder:

| Command | Meaning |
|---|---|
| `bind pipeline` | compute pipeline or graphics variant |
| `bind group` | group index and group object |
| `set dynamic state` | viewport, scissor, stencil reference, index buffer |
| `dispatch` / `draw` | counts only |

- A graphics Node is one render scope; a recorded pass is its recording
  lowered command by command; a standalone graphics or compute Node is the
  one-command stream.
- Group `g` stays bound across a pipeline switch when the two layouts are
  equal at `g`; otherwise the Runtime binds the object made for the new
  layout. Per-parameter stickiness of recordings follows from groups: a set
  whose group layouts correspond stays bound, as section 7 of
  `recorded_draws.md` requires.
- Each command validates only what it changes: `bind group` checks layout
  compatibility at that index; `dispatch` or `draw` checks that every group of
  the bound pipeline is bound.
- The stream is built at invocation entry by the existing recording
  validation walk; draw counts of 0 are inferred from the vertex-input leading
  dimensions recorded on the bound groups.

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
packs its slots, and records on the group what later commands read: the
leading dimensions and strides of its vertex inputs, the buffer accesses
it adds to the Node, and the images it samples. A sealed set's matched
argument frame is lowered once per match and reused by every invocation; a
match whose arguments need a Runtime carrier is materialized per invocation
instead.

**The stream** is built by one walk at invocation entry, which is also the
recording validation: a set's hazard check and frame run once per distinct
set in the invocation, and render targets, attachment formats and the
default viewport once per scope. A command holds its pipeline or variant,
its group objects per index, its dynamic state and its counts. The encoder
binds a pipeline, variant or group only when it differs from the bound one;
the variant is chosen from the pipeline and the vertex strides recorded on
the bound groups, and draw counts of 0 are inferred from their vertex
leading dimensions.

## 7. RuntimeCore

RuntimeCore keeps the command state of an encoder and validates at the
provider boundary:

```c
vernonRuntimeCorePrepareGroupLayout(provider, descriptor, &group_layout);
vernonRuntimeCoreCreateGroup(group_layout, values, value_count, lifetime, &group);
vernonRuntimeCoreBeginCommands(provider, command_encoder, &commands);
vernonRuntimeCoreBindPipeline(commands, pipeline_or_variant);
vernonRuntimeCoreBindGroup(commands, index, group);
vernonRuntimeCoreDispatch(commands, groups);
vernonRuntimeCoreBeginRendering(commands, rendering);
vernonRuntimeCoreDraw(commands, draw);
vernonRuntimeCoreEndRendering(commands);
vernonRuntimeCoreEndCommands(commands);
```

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
encode_dispatch(user_data, encoder, groups[3]);
encode_draw(user_data, encoder, draw);
```

`lifetime` is persistent or transient; a transient group belongs to
`encoder`, may use encoder-scoped memory, and is released with it. The
provider skips native calls whose state is already current.

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
into it, so the group owns no default-heap buffer and records no copy, while
a persistent group copies its inline values into its own default-heap
buffers once per encoder.
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

| Backend | Group object | `bind group` | Transient memory |
|---|---|---|---|
| Vulkan | descriptor set from a pool per group layout | `vkCmdBindDescriptorSets` at set `g`; vertex buffers with `vkCmdBindVertexBuffers` | descriptor pools of the encoder's command frame; RHI transient memory of the encoder |
| D3D12 | descriptor range in a shader-visible heap | `SetGraphicsRootDescriptorTable`/`SetComputeRootDescriptorTable` at table `g` | descriptor rings of the encoder's command frame; RHI transient memory of the encoder |
| Metal | argument buffer (tier 2) | `setVertexBuffer`/`setFragmentBuffer`/`setBuffer` at index `g`; `useResource` for its resources | RHI transient memory of the encoder, reserved in place |
| OpenGL/ES | resolved bindings; inline uniform-buffer data in a buffer made with the group | `glBindBufferRange`, texture units, samplers, `glUniform*`, attribute pointers, each only when it differs | RHI transient memory of the encoder |
| CUDA | packed parameter fragment | records the fragment | RHI transient memory of the encoder |
| CPU | packed call-frame fragment | records the fragment | none |

On OpenGL, buffer, texture and sampler bindings are context state and
survive a program switch; `glUniform*` values and vertex-array attributes
are program and vertex-array state and are reapplied after a pipeline bind
only where the new pipeline needs them. Fixed-function state is compared with
the last applied pipeline. `glGetError` is read once per render scope and
encoder end, which reports the failure for that scope; per-command checks
remain behind the adapter's debug option.

## 9. Interaction with other contracts

- Pipelined submission ([`pipelined_submission.md`](pipelined_submission.md)):
  transient groups are charged to the transient budget of their encoder.
- Hazards stay static: groups hold only borrowed, read-only resources for
  recorded passes, and Node accesses still decide barriers.
- Autodiff is unchanged: derivative Programs are ordinary Programs and
  lower the same way.
- Push constants keep their current use; per-draw scalars beyond them are
  outside this design.

## 10. Rollout

The compiler cut, RuntimeCore, the provider ABI and every provider land in
one version cut, with no emulation path. Vulkan, D3D12, OpenGL, OpenGL ES and
CUDA are verified on Windows; Metal needs the macOS suite before the cut
merges. The executor's per-draw contexts, `RhiGraphicsPipeline::bindings`,
`RhiComputePipeline::bindings`, and the partial-binding-set code are deleted.

## 11. Acceptance

1. Groups: a Kernel and a Pipeline with two binding-group parameters report
   three groups with their own carriers and spilled values on every target, and
   too many groups fail with `PROGRAM_GRAPHICS_GROUP_SET_LIMIT`.
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

- [`../language/contract.md`](../language/contract.md) section 6.4: group
  occurrences take their own partition instead of descriptor set 0.
- [`../program/recorded_draws.md`](../program/recorded_draws.md): section 6
  is implemented by section 3 here; section 8 lowers through section 5;
  section 13 drops the pending group partitions and the per-parameter native
  state.
- [`../program/graphics_storage_reads.md`](../program/graphics_storage_reads.md)
  sections 2 to 4: carriers per group.
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
- **Push-constant ranges per group** for binding-group values: every group
  bind pushes its bytes again, so per-command work grows with the group, and
  all groups share the 128-byte minimum push budget of Vulkan.
- **A uniform block per binding group** for its by-value leaves: it ties
  packing to groups, although group 0 spills the same way, and it needs
  per-leaf block offsets in reflection and multi-parameter packing in the
  Runtime for groups only. Packing spilled values is a separate design that
  serves every set alike (section 3).
- **Binding-group values in group 0**: a set holding material values would
  make a composite transient group per invocation instead of a persistent
  one.

Outside this design: Vernon's scene synchronization (about 13% of host time
in the benchmark).
