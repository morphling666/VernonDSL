# Recorded draws

Status: the Runtime slice (sections 2, 4, 5, 7, 8 and 9) is implemented over
linked schedules; the compiler partitions of section 6 and the base constants
of section 7 are pending (section 13). On acceptance it
amends the contracts listed in section 12 under one coordinated
`COMPILER_CONTRACT_VERSION` and `PROGRAM_VERSION` cut. A pass table is a
pass's structure; its draws may be a draw stream of records
([`../runtime/architecture_v2.md`](../runtime/architecture_v2.md) section 11).

A graphics Node draws a fixed number of times with one Stage. A renderer pass
draws a data-dependent set of objects with several pipelines, per-object
resources, and, for translucency, a sorted order across pipelines. This
document lets one graphics Node be such a pass: the Program fixes the pass,
and each call supplies the pass content in a pass table.

## 1. Rejected alternatives

- **One Node per draw.** Topology would follow scene data, so the Program
  would be relinked every frame.
- **One call per draw.** Nothing plans hazards or render scopes across draws;
  a pass becomes N native render scopes.
- **One Node per pipeline, fused by render-scope planning.** Sorted
  translucency interleaves pipelines on one attachment, which contradicts the
  attachment version order of separate Nodes; the graph grows with content.
- **Draw lists, pipeline tables, or binding tables as Program Values or
  language types.** Draw content is never differentiated, partitioned, or
  analyzed by the compiler, so exposing it to Program IR adds types without
  adding meaning.
- **A source declaration of recorded draws or per-draw groups**, such as
  `draw=vd.recorded("geometry", "material")`. Whether a pass draws a table is
  a call choice, and which parameters may vary follows from how they are
  bound (section 3). A declaration would duplicate those facts and tie
  ordinary pipelines, such as material Programs that are also drawn
  standalone, to one use.
- **Params records in portable reflection.** Params leaves are ordinary
  reflected parameters. Per-draw binding works on parameters, so a params
  identity would be a second, redundant compatibility rule.
- **Host callbacks that record native commands.** They are opaque to the
  Runtime, which owns command recording.
- **Recorded compute dispatches.** Batched compute belongs inside one dispatch
  that indexes an offset table through a TensorView. Per-dispatch writes to
  distinct resources would hide hazards from the resolved plan.

## 2. Model

A graphics Node whose call supplies a pass table is one pass:

- The Program fixes everything the resolved plan needs: attachments,
  load/store/clear/resolve, render area, fixed endpoints, and resource
  accesses. Hazards and barriers stay static.
- The Node's own Stage defines the pass interface and is pipeline 0.
- Each call's table lists the pass content in order: entries, each with a
  pipeline, the binding sets bound since the previous entry, its own
  arguments, dynamic state, a draw or a draw stream, and debug commands.
- A **graphics pipeline** is a prepared Runtime object made from a resolved
  one-node graphics Program that conforms to the Node (section 4). Creating
  one never changes the executable.
- A **binding set** is an immutable, pre-validated binding of some parameters
  of a pipeline (section 5).

Semantics:

- The Node produces one version of each attachment, as any graphics Node does.
- Entries draw in table order. Runtime never reorders them; a caller that
  wants state-sorted order sorts before building the table.
- A table with no entries performs the attachment load and store only, so a
  clear still happens.
- Graphics remains primal-only; table content never enters a derivative
  path.

## 3. Program representation

Pass tables add no source annotation, draw tag, reflection record, or
ProgramGraph call. Every fact a pass needs is already in the Program:

- A graphics Node of a linked schedule has a DRAW, a STREAM and a TABLE slot
  (`VernonSchedule.h`). A null table draws the DRAW command once, or the
  STREAM's records; a table draws its entries instead.
- How each endpoint of the Node's Stage is bound decides whether an entry
  may rebind it:
  - an endpoint bound to a Program parameter or call argument that no Node
    writes (no Node result, Result projection, or non-read access on its
    Storage) is **per-draw**. Its parameter binding is the initial binding
    of the table, so entries that never rebind it use it;
  - every other endpoint, including one bound to a Value produced or written
    inside the Program or to a constant, is **fixed** for the pass.
- `state` is pipeline 0's static state. The DYNAMIC slot's `viewport` and
  `scissor` are the initial dynamic state of the pass.

Endpoints of different pipelines correspond when they have the same module
role and `source_name`, the stage-local identity reflection already carries.
A params leaf's `source_name` is its dotted path, so `material.albedo`
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
lives. Pipeline 0, the Node's own Stage, needs no creation; an entry with a
null `pipeline` draws with it, and `source = NULL` returns a handle to it so
sets can be made for it.

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

A binding set is a parameter binding made into a value. A pass binds the same
parameter once per draw, so the binding cannot live in the schedule's
argument record and must be a value the table refers to:

- A set binds any subset of a pipeline's per-draw parameters, by parameter
  name. Each binding takes a `VernonProgramArgument` and lease, validated
  once against that pipeline's reflection.
- Sealing makes the set immutable, so Runtime builds its native state once.
- A sealed set binds to another pipeline, of any executable on the same
  context, when each of its parameters corresponds to a per-draw parameter of
  that pipeline with equal reflection. Runtime caches the result per
  (set, pipeline), so one geometry set serves depth, shadow, and forward
  passes without revalidation.
- Resources in a set are borrowed and read-only for every call whose table
  binds the set (section 7). Host data is copied when it is bound, so a
  sealed set's values never change and its native state is built once.
- A set never needs a Runtime upload: host data binds only endpoints carried
  inline or in a uniform buffer; anything else binds a device resource. An
  upload would be a per-draw transfer outside the static hazard plan.

## 6. Physical layout

Semantics never refer to groups. A group is the set of leaves that declare
one `set` ([`../runtime/binding_groups.md`](../runtime/binding_groups.md)
section 3), the unit the compiler uses to partition native bindings, so that
the common per-draw switch touches only the switched native state:

- Set-based targets give each set its own native object: a descriptor set on
  Vulkan, a descriptor table on D3D12, an argument buffer on Metal. A set at
  or above the target's set limit fails compilation.
- Every TensorView metadata carrier is in set 0, one per module, until
  per-set carriers land with uniform packing
  ([`../runtime/binding_groups.md`](../runtime/binding_groups.md) section
  3.4).
- OpenGL keeps flat bindings, unique across sets.

Runtime reads these partitions from the target implementation records it
already validates. A sealed set that covers exactly one partition binds as
one native object. A set that covers part of a partition, or several
partitions, is still valid; Runtime composes the native state for it at a
higher cost.

## 7. Pass tables

A call writes a `const VernonSchedulePassTable *` into the TABLE slot of a
graphics Node (`VernonSchedule.h`). The table is read during the call only.
Each `VernonSchedulePassEntry` holds:

| Member | Meaning |
|---|---|
| `pipeline` | a graphics pipeline of the Node, or null for the Node's Stage |
| `sets` | the sealed binding sets bound since the previous entry, in bind order |
| `arguments` | arguments of the pipeline's Program by boundary slot, for this entry only |
| `draw` | a `VernonDrawCommand`: `vertex_count`, `instance_count`, `first_instance`, and for an indexed draw an index binding (buffer reference, index type, byte offset of the first index, `index_count`, `base_vertex`) |
| `stream` | a draw stream, whose records index the whole buffer of `draw.index_binding` |
| `dynamic` | viewport, scissor and stencil reference; an empty rectangle keeps the pass's |
| `debug_commands` | push and pop debug group, insert debug marker, encoded before the entry draws |

The table's own `debug_commands` are encoded after the last entry.

An entry's draw is the same `VernonDrawCommand` the DRAW slot takes, with the
same validation and the same meaning of 0 counts, so one draw description
serves both forms: a `vertex_count` of 0 is inferred from the per-vertex
inputs, and an `instance_count` of 0 draws nothing and reads no instance
records. A renderer culls a draw by setting its instance count to 0. An
instance input must hold at least the records the draw reads and may hold
more, so a draw that binds a fixed range of instance records changes how
many it draws through its counts alone.

Blend constants are absent because no blend factor reads one; they join the
dynamic state together with constant blend factors.

Rules:

- Bindings are sticky per parameter: a later set overrides the parameters it
  binds and leaves the others. A set requires every parameter it binds to
  match the entry's pipeline. On a pipeline switch, each bound parameter is
  matched alone: it stays bound when it corresponds to a per-draw parameter of
  the new pipeline with equal reflection; otherwise it becomes unbound. An
  entry's `arguments` take precedence over the sets for that entry only.
- Binding a fixed endpoint is invalid.
- Every entry requires every per-draw parameter of its pipeline supplied (by
  its arguments, a bound set, or the Node's initial binding), and only dynamic
  state the pipeline supports.
- `instance_index` equals `first_instance` plus the instance ordinal on every
  backend; a backend whose native value excludes the base adds it through a
  Runtime-supplied constant. For `draw_indexed`, `vertex_index` equals the
  index value plus `base_vertex`. Until the compiler emits that constant,
  D3D12 and OpenGL fail a draw with a nonzero `first_instance`, and D3D12
  one with a nonzero `base_vertex`, instead of drawing with a different
  index.
- Resources reached through binding sets, entry arguments and index buffers
  must not be written by any Node of the same call. Anything a Node writes is
  a fixed endpoint bound through ordinary accesses, which keeps the hazard
  plan static. The call does not walk the draws to check this
  (`architecture_v2.md` section 11); a sampled image the pass also renders to
  is rejected.
- A nonzero `revision` names the table's contents: a later call may reuse
  what an earlier call planned for a table at the same address and revision,
  as long as the Node's arguments keep their resources and shapes and its
  attachments their formats and extent. While a caller keeps one revision,
  only the host stream records and counts, and the host bytes of the Node's
  arguments, may change. Zero plans the table at every call. Per-frame
  culling therefore costs no replanning.
- Debug groups and markers name the draws between them in capture tools and
  have no effect on rendering. Groups nest inside the Node's own group
  ([`../runtime/gpu_profiling.md`](../runtime/gpu_profiling.md) section 2),
  and every push needs a later pop in the same table.

## 8. Render scopes and lowering

- A table's Node is exactly one native render scope regardless of its entry
  count. Fusion with neighbouring graphics Nodes follows
  [`graphics_execution.md`](graphics_execution.md) section 9 unchanged.
- OpenGL: one framebuffer scope; a pipeline switch applies the program and a
  diff of cached state; a set bind rebinds its parameters.
- Vulkan: one dynamic-rendering scope; one descriptor set per partition a set
  covers.
- D3D12: one render-target scope; one descriptor table per partition.
- Metal: one render encoder; one argument buffer per partition.
- `vernonScheduleGetGraphicsTelemetry` reports the native render scopes and
  the draw commands of the schedule's last call.

## 9. Diagnostics

| Code | Phase | Cause |
|---|---|---|
| `PROGRAM_PIPELINE_CONFORMANCE` | pipeline creation | a rule of section 4 fails |
| `PROGRAM_BINDING_SET_MISMATCH` | invocation_entry | a set's parameter does not correspond, with equal reflection, to a per-draw parameter of the entry's pipeline |
| `PROGRAM_RECORDING_INVALID` | invocation_entry | a binding of a fixed endpoint, or host data for a device endpoint |
| `PROGRAM_GRAPHICS_GROUP_SET_LIMIT` | compile | more group occurrences than the target's set limit allows |

A malformed table, an entry with no argument for a Stage parameter, or
unbalanced debug groups fail the call with the context's last error naming
the pass and entry.

## 10. Outside this design

- **GPU-produced draws** are specified in
  [`indirect_draws.md`](indirect_draws.md): indirect draw commands whose
  argument and count buffers are static Node accesses. Indirect draws switch
  no binding sets; per-object data comes from tables indexed by
  `draw_index`.
- **GPU pipeline switching** is not planned: GPU-driven passes issue one
  indirect draw per pipeline.
- **Graph-level gradient fan-in** is independent of this design.

## 11. Acceptance

Run on every backend with graphics support:

1. A table with N entries produces the same pixels as N one-node graphics
   Programs drawn in sequence.
2. Entries interleaving two pipelines with different blend state follow
   table order; permuting the table changes pixels as predicted.
3. Two surface types bind sets whose parameters share names but differ in
   type; one geometry set is reused by two pipelines and by two executables.
4. A pipeline created after the executable resolved draws without resolving it
   again.
5. `instance_index` includes `first_instance`; indexed draws honour the
   index offset and `base_vertex`.
6. A table with no entries clears its attachments; telemetry reports one
   render scope per pass.
7. A compute Node writes a fixed TensorView endpoint that the table's draws
   read, and the write is ordered first.
8. A binding set covering one group binds as one native object on set-based
   backends; a binding set covering part of a group draws the same pixels.
9. The same cooked pipeline draws standalone and in a pass table, with no
   recook.
10. Every diagnostic in section 9 is produced with its code, phase, path, and
    context.
11. A table kept at one revision draws new host stream counts (an instance
    count of 0, fewer records) in every following call without replanning.

## 12. Contract changes on acceptance

- [`execution_manifest.md`](execution_manifest.md): section 2 admits
  pass tables; section 11.5 states that a graphics Node draws a pass table;
  section 18 adds the diagnostics; section 15.1 stops requiring one
  specialization key for all ProgramGraph children, which the linker already
  does not require.
- [`graphics_storage_reads.md`](graphics_storage_reads.md) sections 2 to 4
  and [`../language/contract.md`](../language/contract.md) section 6.4:
  metadata carriers and native binding partitions follow top-level group
  occurrences instead of one descriptor set 0.
- [`graphics_execution.md`](graphics_execution.md): section 5 lists pass
  tables as a DrawCommand form.
- [`../runtime/design.md`](../runtime/design.md): graphics pipelines and
  binding sets join the owner model.

## 13. Implementation status and decisions

Pending:

- **Base constants (section 7)** and **group partitions (section 6, acceptance
  8)** belong to the compiler cut, which is deferred until an Engine renderer
  needs them. Until then a set's native state is per-parameter: it is built
  once per set, pipeline and slot group. D3D12 and OpenGL reject draws with
  a nonzero `first_instance` (D3D12 also a nonzero `base_vertex`).
- A Program parameter typed as a TensorView that feeds a `uniform()` Stage
  parameter cooks with a descriptor TensorView reflection but a push-constant
  implementation and no metadata carrier, so the bundle fails to load
  (`PROGRAM_REFLECTION_MISMATCH`). Per-draw uniforms in the pass-table
  fixture are therefore host values. Whether a TensorView may feed a uniform
  is a language-contract question that the compiler cut must settle.

Decisions:

- A table with no entries plans its render targets like a drawing Node, so
  clear-only passes need no Runtime special case.
- The pipeline-0 state of a Node is cached on the executable and holds no
  lease, because a lease on its own executable would keep the executable alive
  forever. Handles and sets made from it hold their own lease.
- A set's match against a pipeline is cached per (set, pipeline serial) and
  records a result per parameter, which is what per-parameter stickiness on a
  pipeline switch reads. Entries of destroyed pipelines are pruned on insert.
- The match also owns the set's native state for that pipeline: a binding
  subset per distinct group of slots that draws take from the set, since a
  later set can override part of it. A slot belongs to a set when every
  parameter its value depends on comes from that set; the pass Node's values,
  the resolution, implicit samplers and slots mixing sources go through the
  pipeline's own subset, which updates only when its values change. Recreating
  native bindings per draw cost about 30 µs per draw on OpenGL.
- Endpoint classification compares Storage, not Value identity: a Node that
  writes a parameter in place makes the endpoint read a later Value of the
  same Storage, so that endpoint is fixed, and a set that binds the parameter
  reports `PROGRAM_RECORDING_INVALID` instead of a mismatch.
