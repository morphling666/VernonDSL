# Graphics execution

Status: current architecture.

This document defines graphics-specific behavior inside the canonical Program
model. Program lifecycle and layering are defined in
[`architecture.md`](architecture.md); exact serialized fields are defined in
[`execution_manifest.md`](execution_manifest.md).

## 1. Invariants

1. A graphics pipeline is a one-node Program. Graphics inside a Module uses
   the same capture, compiler, manifest, loader, resolver, and invocation path.
2. Compute and graphics are distinct typed Node operations but share Program
   Values, Storages, endpoint projections, hazards, and publication.
3. Graphics pipeline state is immutable Stage specialization.
4. Render pass, draw command, and dynamic state are typed invocation controls.
5. Attachments are Program image subresource accesses and versions.
6. Runtime and backend code receive fully parsed and resolved graphics plans.
   They do not parse source-language JSON or infer semantic defaults.
7. Unsupported graphics capabilities fail during resolve or invocation; a
   graphics Node never degrades to a no-op.

## 2. Authoring and normalization

`vd.pipeline(vertex, fragment, ...)` is the graphics authoring object.
`program_asset(program=...)` accepts that pipeline directly. A tuple of entry
functions is not a graphics Program declaration.

Source and Program IR may use typed render conveniences, but deployment
normalizes them into one `GraphicsOperation` containing:

- immutable pipeline state;
- attachment signatures and Program resource accesses;
- a render-pass control;
- a draw-command control and optional static default;
- a dynamic-state control;
- explicit endpoint projections for shader Values and Resources.

There is no second graphics-only manifest, loader, or binding ABI.

## 3. Typed operation

Program Nodes use a closed operation union:

```cpp
using NodeOperation = std::variant<ComputeOperation, GraphicsOperation>;
```

Generic Program logic dispatches by typed operation kind. Backend selection is
independent from operation-kind selection.

The parser converts the serialized operation tag exactly once. Code after
parsing uses typed visitation and never selects semantics through string
comparisons, empty payloads, or legacy mode flags.

## 4. Static pipeline state

Static graphics state includes:

- primitive topology;
- rasterization;
- depth/stencil state;
- multisample state;
- per-color-target blend and write-mask state;
- attachment format/sample compatibility required for native pipeline
  creation.

Clear values and attachment load/store operations are render-pass behavior,
not pipeline state.

Primitive topology participates in Stage specialization and artifact identity.
It is not an invocation Value.

## 5. Invocation controls

Every graphics Node references three canonical Program controls:

- **RenderPass**: concrete image views, load/store/clear/resolve operations,
  render area, and attachment geometry;
- **DrawCommand**: direct or indexed draw counts, offsets, instance range, and
  index-buffer binding;
- **DynamicState**: viewport, scissor, stencil reference, blend constants, and
  other supported dynamic backend state.

Controls are bound on `VernonProgramInvocation`. They are not shader
parameters and do not use Stage endpoint slots.

Programs may declare static defaults where the manifest contract permits.
Runtime never infers vertex or index counts from shader inputs.

## 6. Attachments and image versions

Each attachment use identifies:

- exact Program Storage and image-view descriptor;
- aspect, mip, and layer range;
- load operation and required clear value;
- store operation;
- optional resolve target;
- format and sample-count compatibility;
- input and output Value versions.

Load/store/resolve semantics create Program resource transitions. Runtime
derives RAW, WAR, and WAW dependencies from concrete image/view identity and
subresource overlap.

Resolve destination load is implicitly discard. Source and destination store
availability are independent.

Shader-visible sampled images, storage images, typed byte storage, and
Samplers bind through ordinary Program endpoints. They are not nested inside
ABI-stable aggregate Values.

ImageView is a descriptor-bearing alias of image Storage. Program Value
origins retain parent version and exact subresource range; the physical
allocation format belongs to the Storage descriptor.

## 7. Render-scope planning

The private Command DAG may fuse adjacent graphics Nodes into one native
render pass when all of the following hold:

- attachment views and geometry are compatible;
- version continuity is preserved;
- load, store, clear, and resolve operations compose exactly;
- no intervening resource hazard requires a scope boundary;
- backend capabilities permit the fused form.

Pipeline-state equality is not required: compatible native pipelines and
dynamic state may change between draws in one render pass.

Fusion is a physical optimization. Program semantics and publication must be
identical with or without fusion.

## 8. Backend boundary

Physical resolution produces a `ResolvedGraphicsInvocation` containing:

- selected native graphics Stage executable;
- exact endpoint carriers and resource bindings;
- normalized static pipeline state;
- concrete render-pass, draw, and dynamic controls;
- planned barriers and image transitions;
- publication actions.

Backends consume this normalized object. They do not:

- choose between old and current schemas;
- parse graphics state at execution time;
- infer omitted required controls;
- reconstruct Program Value bindings by name;
- own Program-level resource versioning.

## 9. Program composition

Compute and graphics Nodes may coexist in one static Program DAG. Shared
buffers and images flow through ordinary Program Values and Storage versions.
The resolved transfer and hazard plan orders producer writes, graphics reads,
attachment transitions, and publications.

Graphics remains outside active VJP. A VJP request fails closed when its
selected derivative path traverses a graphics Node. Graphics Nodes unrelated
to requested derivatives may remain in primal execution without introducing
captures or gradient boundaries.

## 10. Acceptance

The cross-backend suite must verify:

- standalone and Module-captured graphics produce equivalent pixels and
  reflection;
- compute-to-graphics data flow preserves numerical results and command order;
- attachment clear/load/store/discard/resolve behavior;
- indexed and direct draws with explicit counts;
- persistent graphics controls across Program instances;
- transactional publication on submission failure;
- image subresource hazard ordering;
- capability-based execution across every applicable graphics backend.

Backend-specific native interop and ABI tests remain separate from these
backend-independent semantic tests.
