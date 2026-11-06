# Indirect draws

Status: proposed. Extends [`recorded_draws.md`](recorded_draws.md) section 10
("GPU-produced draws") into a design; amends its sections 7, 9 and 11.

A renderer that draws each object with its own draw pays a per-draw cost on
the CPU, in the driver and on the GPU: the Editor's 1000 boxes take 4000
shadow draws and 1.5 ms of shadow GPU time for a few thousand triangles.
Objects that share a pipeline, a material set and a geometry buffer need
only one draw command carrying many draws, with per-object data read from a
table indexed per draw. The same command lets a compute Node produce the
draws (GPU culling) without a CPU round trip.

## 1. Language

- A vertex stage may declare `draw_index: Annotated[vd.u32,
  vd.builtin("draw_index")]`: the ordinal of the draw within the
  multi-draw command that issued it, 0 for a direct draw.
- Per-draw data comes from TensorView tables indexed by `draw_index`
  ([`graphics_storage_reads.md`](graphics_storage_reads.md)). A table row
  usually holds an object index into an instance table, so one argument
  record can draw many instances that read their row through
  `draw_index` and `instance_index`.
- `vd.DrawIndirect` and `vd.DrawIndexedIndirect` are record types for
  TensorViews of argument records. Their members are the draw's counts and
  first elements: `vertex_count`, `instance_count`, `first_vertex`; and
  `index_count`, `instance_count`, `first_index`, `base_vertex`. A compute
  stage writes them like any TensorView element. Their byte layout belongs
  to the target (section 3) and is reported by reflection, so host writers
  use it from reflection instead of assuming one.
- `first_instance` is not a member. Every backend draws indirect records
  with base instance 0, because OpenGL ES requires it and D3D12's
  `SV_InstanceID` excludes it; `draw_index` replaces it as the per-draw row.

## 2. Recording commands

| Command | Members |
|---|---|
| `draw_indirect` | argument endpoint, record offset, `draw_count`, optional count endpoint |
| `draw_indexed_indirect` | the same, plus the index binding of `draw` without counts (buffer, index type) |

- The argument and count buffers are **fixed endpoints** of the Node's Stage
  declared with indirect access, so the resolved plan orders the Node that
  writes them before the draw and emits the indirect-read barrier, as any
  fixed access. They are never in a binding set.
- `draw_count` is the maximum number of records drawn. With a count
  endpoint, the draw uses `min(count, draw_count)` records; the records from
  the count up to `draw_count` must have `instance_count` 0, so backends
  that cannot read the count draw them as empty draws (section 3).
- The record offset and `draw_count` are payload: `SetCommand` replaces them
  without a plan rebuild.
- Validation (`recorded_draws.md` section 7) checks that the endpoint holds
  at least `offset + draw_count` records and the count endpoint one `u32`.
  Record contents are device data and are not validated; an out-of-range
  index count reads out of bounds, as an out-of-bounds TensorView read does.

## 3. Targets

| Backend | Multi-draw | `draw_index` | Count |
|---|---|---|---|
| Vulkan | `vkCmdDraw(Indexed)Indirect` | `DrawIndex` (`shaderDrawParameters`) | `vkCmdDraw(Indexed)IndirectCount` (1.2) |
| D3D12 | `ExecuteIndirect`, command signature with one root constant and the draw arguments | the root constant, written by the record type's lowering as the record's ordinal | count buffer argument |
| OpenGL 4.6, or 4.3 with `ARB_shader_draw_parameters` | `glMultiDraw(Elements)Indirect` | `gl_DrawID` | `glMultiDraw(Elements)IndirectCount` with `ARB_indirect_parameters` |
| OpenGL 4.3 without it, OpenGL ES 3.1+ | one `glDraw(Elements)Indirect` per record | a Runtime uniform set per record | empty records |
| Metal | one `drawPrimitives`/`drawIndexedPrimitives` with an indirect buffer per record | `setVertexBytes` per record | empty records |

- On D3D12 the record type's layout carries the root constant before the
  arguments, so a compute stage that writes record `i` also writes `i`.
  Other targets use the native argument layout, which is the same order on
  Vulkan, OpenGL and Metal.
- The per-record loop is the backend's lowering of a multi-draw, not a
  second path: it is one operation of the device's recording
  (`../runtime/architecture_v2.md` section 16.3).
  The compiler emits the `draw_index` source for the target (built-in or
  Runtime uniform) from the target version and extensions it cooks for, and
  the Runtime checks the device against the Stage's runtime requirements.
- CUDA and CPU have no graphics.

## 4. Diagnostics

| Code | Phase | Cause |
|---|---|---|
| `PROGRAM_RECORDING_INVALID` | invocation_entry | an indirect draw whose endpoint is not declared with indirect access, or too small for `offset + draw_count` records |
| `PROGRAM_INDIRECT_DRAW_UNSUPPORTED` | compile | `draw_index` or an indirect-record type on a target without indirect drawing (GLSL before 4.30 or ES 3.10, CUDA, CPU) |

## 5. Rejected alternatives

- **Base instance as the per-draw row.** OpenGL ES fixes it at 0 in indirect
  records and D3D12's `SV_InstanceID` excludes it, so neither could read it.
- **Bindless resource tables in this extension.** Per-object buffers and
  textures would let one draw cover objects of different materials, but
  need descriptor indexing that OpenGL ES and older OpenGL lack. Draws
  are batched per material set; bindless tables are a later extension.
- **Layered rendering for shadow cascades.** With one multi-draw per pipeline
  and cascade, cascades cost a few commands, and writing the layer from the
  vertex stage is missing on OpenGL ES and needs optional features
  elsewhere.
- **Native indirect command buffers** (Metal ICB, Vulkan device-generated
  commands): they also switch pipelines and sets from the GPU, which
  `recorded_draws.md` section 10 rules out, and they exist on two backends.

## 6. Acceptance

Run on every backend with graphics support:

1. A multi-draw of N records draws the same pixels as N direct draws, with
   `draw_index` selecting each draw's row.
2. A compute Node writes the records and the count in the same invocation;
   the draw is ordered after it and draws only the counted records.
3. Records past the count with `instance_count` 0 draw nothing on backends
   without count support.
4. Changing the record offset or `draw_count` through `SetCommand` rebuilds
   no plan.
5. Every diagnostic in section 4 is produced with its code, phase and path.
