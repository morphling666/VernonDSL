# Recorded draws

Status: the Runtime slice (sections 2, 4, 5, 7, 8 and 9) is implemented; the
compiler partitions of section 6 and the base constants of section 7 are
pending (section 13). On acceptance it
amends the contracts listed in section 12 under one coordinated
`COMPILER_CONTRACT_VERSION` and `PROGRAM_VERSION` cut.

A graphics Node draws a fixed number of times with one Stage. A renderer pass
draws a data-dependent set of objects with several pipelines, per-object
resources, and, for translucency, a sorted order across pipelines. This
document lets one graphics Node be such a pass: the Program fixes the pass,
and each invocation records the pass content.

## 1. Rejected alternatives

- **One Node per draw.** Topology would follow scene data, so the Program
  would be relinked every frame.
- **One Program invocation per draw.** Nothing plans hazards, render scopes,
  or publication across draws; a pass becomes N native render scopes and N
  transactions.
- **One Node per pipeline, fused by render-scope planning.** Sorted
  translucency interleaves pipelines on one attachment, which contradicts the
  attachment version order of separate Nodes; the graph grows with content.
- **Draw lists, pipeline tables, or binding tables as Program Values or
  language types.** Draw content is never differentiated, partitioned, or
  analyzed by the compiler, so exposing it to Program IR adds types without
  adding meaning.
- **A source declaration of recorded draws or per-draw groups**, such as
  `draw=vd.recorded("geometry", "material")`. Whether draws are recorded is an
  invocation choice, and which parameters may vary follows from how they are
  bound (section 3). A declaration would duplicate those facts and tie
  ordinary pipelines, such as material Programs that are also drawn
  standalone, to one use.
- **Binding-group records in portable reflection.** Group leaves are ordinary
  reflected parameters. Per-draw binding works on parameters, so a group
  identity would be a second, redundant compatibility rule.
- **Host callbacks that record native commands.** They are opaque to the
  Runtime, which owns command recording.
- **Recorded compute dispatches.** Batched compute belongs inside one dispatch
  that indexes an offset table through a TensorView. Per-dispatch writes to
  distinct resources would hide hazards from the resolved plan.

## 2. Model

A graphics Node whose draw control is bound to a recording is one pass:

- The Program fixes everything the resolved plan needs: attachments,
  load/store/clear/resolve, render area, fixed endpoints, and resource
  accesses. Hazards, barriers, and publication stay static.
- The Node's own Stage defines the pass interface and is pipeline 0.
- Each invocation records the pass content in order: pipeline binds, binding
  set binds, dynamic state, and draws.
- A **graphics pipeline** is a prepared Runtime object made from a resolved
  one-node graphics Program that conforms to the Node (section 4). Creating
  one never changes the executable.
- A **binding set** is an immutable, pre-validated binding of some parameters
  of a pipeline (section 5).

Semantics:

- The Node produces one version of each attachment, as any graphics Node does.
- Recorded commands execute in recording order. Runtime never reorders them;
  a caller that wants state-sorted order sorts before recording.
- A Node with no recorded draws performs its attachment load and store only,
  so a clear still happens.
- Graphics remains primal-only; recorded content never enters a derivative
  path.

## 3. Program representation

Recorded draws add no source annotation, draw tag, reflection record, or
ProgramGraph call. Every fact a pass needs is already in the Program:

- A graphics Node whose `draw` is an invocation control accepts either one
  draw command or a recording. A recording supersedes the DrawCommand for its
  invocation, including one that persists in instance state from an earlier
  invocation; the persisted DrawCommand applies again to the next invocation
  that records nothing. A Node with a static draw in its manifest accepts
  neither, as today.
- How each endpoint of the Node's Stage is bound decides whether a recording
  may rebind it:
  - an endpoint bound to a Program parameter or invocation argument that no
    Node writes (no Node result, Result projection, or non-read access on its
    Storage) is **per-draw**. Its parameter binding is the initial binding
    of the recording, so draws that never rebind it use it;
  - every other endpoint, including one bound to a Value produced or written
    inside the Program or to a constant, is **fixed** for the pass.
- `state` is pipeline 0's static state. `viewport` and `scissor` are the
  initial dynamic state of the pass.

Endpoints of different pipelines correspond when they have the same module
role and `source_name`, the stage-local identity reflection already carries.
A binding-group leaf's `source_name` is its dotted path, so `material.albedo`
in two pipelines is one parameter name.

## 4. Graphics pipelines and conformance

```c
vernonRuntimeGraphicsPipelineCreate(executable, node, source, &pipeline);
```

`source` is a resolved one-node graphics Program on the same Runtime context.
It conforms to Node `node` of `executable` when:

1. its attachment signature equals the Node's: color locations, formats,
   sample counts, and depth/stencil aspects;
2. every endpoint of `source` that corresponds to a fixed endpoint of the
   Node's Stage reflects the same type, layout, role, transport, and access.
   `source` may omit endpoints of the Node's Stage;
3. its required features and graphics capabilities are supported by the
   context.

Every other endpoint of `source` is per-draw. The pipeline keeps `source`'s
own static state and topology. Creation prepares the native pipeline object
and may run off the rendering thread. Pipelines retain their source and pass
executables, so destroying either fails while a pipeline or a set made from it
lives. Pipeline 0, the Node's own Stage, needs no creation; a 0 `bind_pipeline`
names it, and `source = NULL` returns a handle to it so sets can be made for
it.

Because conformance is checked when a pipeline is created, loading a new
surface type adds pipelines without resolving the executable again. The
executable changes only when the pass structure changes.

## 5. Binding sets

```c
vernonRuntimeBindingSetCreate(pipeline, &set);
vernonRuntimeBindingSetBind(set, argument, lease);
vernonRuntimeBindingSetSeal(set);
```

`argument.slot` is the parameter slot reported by
`vernonRuntimeProgramExecutableFindParameter` on the pipeline's source
executable.

A binding set is the existing parameter binding made into a value. A
non-recorded invocation binds one value per slot into instance state through
`vernonRuntimeProgramInvocationBind`; a recorded pass binds the same parameter
once per draw, so the binding cannot live in slot state and must be a value
the recording refers to. Nothing else about binding changes:

- A set binds any subset of a pipeline's per-draw parameters, by parameter
  name. Each binding takes the same `VernonProgramArgument`, lease, and
  validation as `vernonRuntimeProgramInvocationBind`, run once against that
  pipeline's reflection.
- Sealing makes the set immutable, so Runtime builds its native state once.
  The C++ binding-set wrapper that binds a Program today seals the parameters
  it has bound into a Runtime set; there is no second binding API.
- A sealed set binds to another pipeline, of any executable on the same
  context, when each of its parameters corresponds to a per-draw parameter of
  that pipeline with equal reflection. Runtime caches the result per
  (set, pipeline), so one geometry set serves depth, shadow, and forward
  passes without revalidation.
- Resources in a set are borrowed and read-only for every invocation that
  records the set (section 7).
- A set never needs a Runtime upload: host data binds only endpoints carried
  inline or in a uniform buffer; anything else binds a device resource. An
  upload would be a per-draw transfer outside the static hazard plan.

## 6. Physical layout

Semantics never refer to binding groups. Groups are the unit the compiler uses
to partition native bindings, so that the common per-draw switch touches only
the switched native state:

- Each top-level binding-group occurrence owns its own TensorView metadata
  carrier per module. Ungrouped TensorViews share the module's remaining
  carrier. A carrier that mixed pass and per-draw fields would be rewritten on
  every draw.
- Set-based targets give ungrouped endpoints and their carriers descriptor
  set 0 and top-level occurrence `k`, in declaration order, set `k + 1`.
  D3D12 uses one descriptor table per occurrence and Metal one argument buffer
  per occurrence. A Stage with more occurrences than the target's set limit
  minus one fails compilation.
- OpenGL keeps flat bindings.

Runtime reads these partitions from the target implementation records it
already validates. A sealed set that covers exactly one partition binds as
one native object. A set that covers part of a partition, or several
partitions, is still valid; Runtime composes the native state for it at a
higher cost.

## 7. Recording

```c
vernonRuntimeProgramInvocationRecordGraphics(invocation, node, commands, count, leases, lease_count);
```

`commands` is an array of tagged records appended to the Node's recording:

| Command | Members |
|---|---|
| `bind_pipeline` | pipeline, or 0 for the Node's Stage |
| `bind_set` | binding set |
| `set_viewport`, `set_scissor`, `set_stencil_reference` | dynamic state |
| `draw` | a `VernonDrawCommand`: `vertex_count`, `instance_count`, `first_instance`, and for an indexed draw an index binding (buffer reference, index type, byte offset of the first index, `index_count`, `base_vertex`) |
| `push_debug_group`, `insert_debug_marker` | UTF-8 label, copied at record |
| `pop_debug_group` | none |

A recorded draw is the same `VernonDrawCommand` the draw control binds, with
the same validation and the same meaning of 0 counts (inferred from vertex
inputs), so one draw description serves both forms.

Blend constants are absent because no blend factor reads one; they join the
dynamic state together with constant blend factors.

The C++ wrapper exposes the same commands as chained calls
(`pass.bind(pipeline).bind(set).draw(...)`) that build this array.

Rules:

- Bindings are sticky per parameter: a later set overrides the parameters it
  binds and leaves the others. `bind_set` requires every parameter of the set
  to match the bound pipeline. On a pipeline switch, each bound parameter is
  matched alone: it stays bound when it corresponds to a per-draw parameter of
  the new pipeline with equal reflection; otherwise it becomes unbound.
- Binding a fixed endpoint is invalid.
- Every draw requires a bound pipeline, every per-draw parameter of that
  pipeline bound (by the recording or by its initial parameter binding), and
  only dynamic state the pipeline supports.
- `instance_index` equals `first_instance` plus the instance ordinal on every
  backend; a backend whose native value excludes the base adds it through a
  Runtime-supplied constant. For `draw_indexed`, `vertex_index` equals the
  index value plus `base_vertex`. Until the compiler emits that constant,
  D3D12 and OpenGL fail a draw with a nonzero `first_instance`, and D3D12
  one with a nonzero `base_vertex`, instead of drawing with a different
  index.
- Resources reached through recorded binding sets and index buffers must not
  be written by any Node of the same invocation. Anything a Node writes is a
  fixed endpoint bound through ordinary accesses, which keeps the hazard plan
  static.
- Validation runs once at invocation entry, linear in the command count,
  before any native command is recorded. Recording belongs to the invocation
  transaction; failure follows the ordinary rollback rules, and recorded
  leases are retained until completion.
- Recordings are invocation-local. A new invocation starts with an empty
  recording.
- Debug groups and markers name the draws between them in capture tools and
  have no effect on rendering. Groups nest inside the Node's own group
  ([`../runtime/gpu_profiling.md`](../runtime/gpu_profiling.md) section 2),
  and every push needs a later pop in the same recording.

## 8. Render scopes and lowering

- A recorded Node is exactly one native render scope regardless of its command
  count. Fusion with neighbouring graphics Nodes follows
  [`graphics_execution.md`](graphics_execution.md) section 9 unchanged.
- OpenGL: one framebuffer scope; a pipeline switch applies the program and a
  diff of cached state; a set bind rebinds its parameters.
- Vulkan: one dynamic-rendering scope; one descriptor set per partition a set
  covers.
- D3D12: one render-target scope; one descriptor table per partition.
- Metal: one render encoder; one argument buffer per partition.
- Runtime telemetry reports per Node: native render scopes, pipeline binds,
  set binds, and draws (`vernonRuntimeProgramInvocationGetGraphicsTelemetry`,
  valid after execute).

## 9. Diagnostics

| Code | Phase | Cause |
|---|---|---|
| `PROGRAM_PIPELINE_CONFORMANCE` | pipeline creation | a rule of section 4 fails |
| `PROGRAM_BINDING_SET_MISMATCH` | invocation_entry | a set's parameter does not correspond, with equal reflection, to a per-draw parameter of the bound pipeline |
| `PROGRAM_RECORDING_INVALID` | invocation_entry | a draw without a pipeline, an unbound per-draw parameter, a binding of a fixed endpoint, unsupported dynamic state, or unbalanced debug groups |
| `PROGRAM_BINDING_SET_HAZARD` | invocation_entry | a recorded resource is written by a Node of the same invocation |
| `PROGRAM_GRAPHICS_GROUP_SET_LIMIT` | compile | more group occurrences than the target's set limit allows |

## 10. Outside this design

- **GPU-produced draws.** A later extension adds indirect draw commands whose
  argument and count buffers are static Node accesses. Indirect draws switch
  no binding sets; per-object resources come from resource tables.
- **GPU pipeline switching** is not planned: GPU-driven passes issue one
  indirect draw per pipeline.
- **Graph-level gradient fan-in** is independent of this design.

## 11. Acceptance

Run on every backend with graphics support:

1. A recorded Node with N draws produces the same pixels as N one-node
   graphics Programs drawn in sequence.
2. Draws interleaving two pipelines with different blend state follow
   recording order; permuting the recording changes pixels as predicted.
3. Two surface types bind sets whose parameters share names but differ in
   type; one geometry set is reused by two pipelines and by two executables.
4. A pipeline created after the executable resolved draws without resolving it
   again.
5. `instance_index` includes `first_instance`; indexed draws honour the
   index offset and `base_vertex`.
6. A Node with no recorded draws clears its attachments; telemetry reports one
   render scope per recorded Node.
7. A compute Node writes a fixed TensorView endpoint that the recorded draws
   read, and the write is ordered first.
8. A set covering one binding group binds as one native object on set-based
   backends; a set covering part of a group draws the same pixels.
9. The same cooked pipeline draws standalone and as a recorded pass, with no
   recook.
10. Every diagnostic in section 9 is produced with its code, phase, path, and
    context.

## 12. Contract changes on acceptance

- [`execution_manifest.md`](execution_manifest.md): section 2 admits
  recorded draws; section 11.5 states that a draw control accepts a
  recording; section 18 adds the diagnostics; section 15.1 stops requiring
  one specialization key for all ProgramGraph children, which the linker
  already does not require.
- [`graphics_storage_reads.md`](graphics_storage_reads.md) sections 2 to 4
  and [`../language/contract.md`](../language/contract.md) section 6.4:
  metadata carriers and native binding partitions follow top-level group
  occurrences instead of one descriptor set 0.
- [`graphics_execution.md`](graphics_execution.md): section 5 lists recorded
  draws as a DrawCommand form.
- [`../runtime/design.md`](../runtime/design.md): graphics pipelines and
  binding sets join the owner model; recordings are invocation-local; binding
  sets share the validation path of `vernonRuntimeProgramInvocationBind`.

## 13. Implementation status and decisions

Pending:

- **Base constants (section 7)** and **group partitions (section 6, acceptance
  8)** belong to the compiler cut, which is deferred until an Engine renderer
  needs them. Until then every set composes per-parameter native state, and
  D3D12 and OpenGL reject draws with a nonzero `first_instance` (D3D12 also a
  nonzero `base_vertex`).
- A Program parameter typed as a TensorView that feeds a `uniform()` Stage
  parameter cooks with a descriptor TensorView reflection but a push-constant
  implementation and no metadata carrier, so the bundle fails to load
  (`PROGRAM_REFLECTION_MISMATCH`). Per-draw uniforms in the recorded-draws
  fixture are therefore host values. Whether a TensorView may feed a uniform
  is a language-contract question that the compiler cut must settle.

Decisions:

- A recorded Node without draws plans its render targets like a drawing Node
  and opens its scope at RHI `EndRendering`
  ([`../runtime/design.md`](../runtime/design.md) section 12), so clear-only
  passes need no Runtime special case.

- The pipeline-0 state of a Node is cached on the executable and holds no
  lease, because a lease on its own executable would keep the executable alive
  forever. Handles and sets made from it hold their own lease.
- A set's match against a pipeline is cached per (set, pipeline serial) and
  records a result per parameter, which is what per-parameter stickiness on a
  pipeline switch reads. Entries of destroyed pipelines are pruned on insert.
- Endpoint classification compares Storage, not Value identity: a Node that
  writes a parameter in place makes the endpoint read a later Value of the
  same Storage, so that endpoint is fixed, and a set that binds the parameter
  reports `PROGRAM_RECORDING_INVALID` instead of a mismatch.
- Hazards compare buffers by identity, value and byte range, and images by
  their parent image, against every non-read boundary binding and every
  attachment of the invocation's forward graphics Nodes.
