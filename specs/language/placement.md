# Parameters, placement and groups

Status: implemented, except where a rule names what is still to land.

Entry parameters reach native bindings through three source-level rules:
`@vd.params` aggregates leaves without physical meaning, placement puts each
leaf in a group (its update-frequency partition) and optionally a binding,
and the compiler packs by-value leaves per group. Semantics never refer to
groups; they decide only native layout.

## 1. `@vd.params`

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
- Pinned leaves (section 2) of two occurrences of one class claim the same
  bindings or locations and fail as two parameters would. Unpinned leaves
  never conflict.
- The rest of [`contract.md`](contract.md) section 6.4 is unchanged: a
  params value is not a Value, helpers and methods take it, and autodiff
  treats it leaf by leaf.

## 2. Placement

Placement is per leaf, in every stage kind, Kernels included:

| Annotation | Placement |
|---|---|
| `uniform(set=None, binding=None)` | a by-value leaf |
| `resource(set=None, binding=None)` | a TensorView, Texture or Sampler leaf |

`attribute(location=None, divisor=0)` is unchanged. A vertex attribute is a
vertex-input location, as on every native API, and belongs to no group
(section 3).

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
  only places it. Kernel placement lands with packing (section 4), which
  rebuilds the Kernel layout per group; until then Kernel parameters bind in
  argument order ([`contract.md`](contract.md) section 6) and a Kernel leaf
  with `set` or `binding` fails with
  `kernel parameters bind in argument order`.
- The same leaf in two stages of a pipeline has one placement; authored
  placements that differ fail at the pipeline link.

## 3. Groups

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
  index are compatible at that index, so a pass table's entries that differ
  only in a higher group share the lower groups' native state.
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
  never rebinds groups or vertex inputs. A variant needs a nonzero stride
  for every binding.
- Implicit samplers belong to the group of the sampled image they pair with.
  A params leaf stays an endpoint of every Stage that declares its
  parameter, so a sampled Texture leaf a Stage never reads still binds
  there; pairing is
  per Program ([`../program/integer_targets.md`](../program/integer_targets.md)
  section 4): the leaf gets an implicit sampler unless another Stage of the
  pipeline reads it, which pairs it in that Stage.

## 4. Packing

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
  group's block size, so a group costs at most one uniform binding for its
  values.
- CUDA and CPU pass a group's by-value leaves as its parameter fragment,
  laid out by the same rule.
