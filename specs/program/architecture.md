# Program architecture

Status: current architecture.

This document defines the stable boundaries shared by standalone compute
kernels, graphics pipelines, initialized `Module` objects, and explicit
Program transforms such as VJP. Field-level deployment syntax belongs only to
[`execution_manifest.md`](execution_manifest.md).

## 1. One executable model

Every public executable is a Program:

- a standalone compute kernel is a one-node compute Program;
- a standalone graphics pipeline is a one-node graphics Program;
- an initialized `Module` is a Program with one or more nodes;
- an explicit transform produces another Program.

Node count, stage kind, and differentiation do not select another asset,
loader, binding API, or runtime architecture. Optimized one-node execution may
begin only after canonical resolution.

The public asset declaration is `program_asset`. It accepts a compute kernel,
a `vd.pipeline(...)` graphics pipeline, an initialized Module, or a supported
Program transform. Graphics entry tuples are not Program assets by themselves.

## 2. Vocabulary

- **Program** is the callable semantic graph. It owns public boundaries,
  Values, Storages, Nodes, effects, forward/backward graphs, and Program ABI.
- **Stage** is a reusable implementation contract selected for one or more
  Nodes. It does not own invocation-specific Program Value bindings.
- **Node** is one invocation of a Stage in a Program graph. It owns endpoint
  projections and resource-version effects.
- **ArtifactSystem** contains authenticated target code and reflection for the
  selected target.
- **Program bundle** is an immutable deployment description containing one
  Program and its variants.
- **Program executable** owns one immutable `ResolvedExecutionPlan`.
- **Linked schedule** is an executable planned once into recorded passes; a
  call writes one argument record and returns a timeline point.
- **Linked VJP** is a forward and a backward schedule; a forward call returns
  the residuals its backward calls read.
- **ProgramGraph** is an optional public pre-resolution composition builder.
  Its nodes are loaded Programs and it finalizes into one ordinary Program.

`Pipeline` is reserved for the `vd.pipeline(...)` graphics authoring object or
a native backend pipeline. It is not a deployment version axis or an
alternative executable model.

## 3. Layered flow

```text
source kernel / vd.pipeline / initialized Module
  -> ProgramAssetDeclaration
  -> CapturedProgram
  -> MLIR Program IR
  -> optional Program transforms
  -> implementation selection and Stage compilation
  -> Program + ArtifactSystem
  -> VernonProgramBundle
  -> ResolveProgram
  -> ResolvedProgram
  -> ResolvedExecutionPlan
  -> VernonProgramExecutable
  -> linked schedule or VJP
  -> HAL device
```

The boundaries are strict:

1. Declaration selects a source executable and variants.
2. Capture creates canonical semantic Program input without executing device
   work.
3. Program IR owns semantic graph transforms, including VJP and cotangent
   fan-in.
4. Stage compilation owns target code and physical endpoint reflection.
5. Deployment serializes immutable Program and artifact descriptions.
6. Logical resolve validates Program topology, Value SSA, Storage versions,
   controls, and portable Stage contracts into `ResolvedProgram`.
7. Physical resolve selects exact Stage implementations, carriers, transfers,
   residency, hazards, and residual plans into `ResolvedExecutionPlan`.
8. Linking plans the resolved plan once into a schedule; each call writes its
   Values and controls into one argument record.

No lower layer reconstructs semantic information owned by a higher layer.
Names, endpoint order, dtype guesses, zero sentinels, and runtime heuristics
must not replace explicit IDs and validated projections.

## 4. Compute Stage contract

A compute Stage has three distinct phases:

```text
source + annotations + typed specialization key -> lower -> portable IR
portable IR + target                         -> compile -> Stage artifact
artifact + runtime bindings                  -> invoke -> dispatch
```

- Lowering and Stage compilation do not receive invocation tensors, launch
  extents, concrete dynamic shapes, strides, offsets, or byte lengths.
- `vd.dyn` remains dynamic in Program and Stage contracts.
- One artifact is reusable across compatible runtime shapes and launch sizes.
- C++ binding is the authority for concrete TensorView descriptors.
- Borrowed dynamic Storage descriptors omit unknown extents. They never encode
  zero placeholders.
- Compute grid axes are ordinary Program Values referenced by controls.
- Different launch sizes do not trigger Stage recompilation.

## 5. Program IR and transforms

MLIR Program IR is the semantic compiler authority. It contains stable Value
and Storage IDs, typed operations, effects, resource versions, source
provenance, public boundaries, and forward/backward graphs.

Kernel IR remains a separate implementation level:

- Program IR owns graph composition, reverse scheduling, residual boundaries,
  cotangent fan-in, and unsupported-path rejection.
- Kernel IR owns thread semantics, structured control flow, workgroup behavior,
  kernel ABI, and kernel-local tape generation.

Program-level Python control flow is host-static. It may depend on initialized
Module state, constants, and typed specializations. It may not depend on
invocation Values or device data. Dynamic per-element control flow belongs
inside a kernel or shader.

Initialized Module instances are compiler inputs, not serialized objects.
Artifacts contain no Module instance, `__dict__`, pickle payload, Python
callback, or constructor configuration tree.

Interactive Module capture and Program reuse key on the annotation-static
forward signature and host-static Module configuration, not concrete dynamic
TensorView extents or launch values. A transient allocation declared relative
to a dynamic input or prior allocation derives its concrete shape from that
source for each invocation. The descriptor and artifact-identity boundary is
defined by [`language/tensor_view.md`](../language/tensor_view.md).

## 6. Stage and Node binding

Program Values and Stage-local physical endpoints are different ABI layers.
Each Node carries explicit projections between:

- one Program Value or resource transition;
- one Stage endpoint;
- an optional logical aggregate leaf;
- an optional physical carrier leaf.

A Stage may be reused by multiple Nodes. Each Node retains its own projection
and dependency context.

`StorageDescriptor` is the sole authority for allocation identity, ownership,
layout, extent, and lifetime. Value IDs represent semantic SSA versions.
Storage versions form a non-branching transition chain.

Resolve derives dependencies from:

- producer/consumer Value SSA;
- overlapping resource RAW, WAR, and WAW transitions;
- explicit control dependencies required by the Program contract.

The manifest does not serialize a second independently authoritative edge
graph.

## 7. Resolution and execution

`ResolveProgram` first produces a logical `ResolvedProgram`, then an immutable
physical `ResolvedExecutionPlan`.

The physical plan contains:

- selected target Stage implementations and entry points;
- per-Node endpoint projections;
- host/device carrier and residency decisions;
- ordered upload, device-copy, and readback transfers;
- resource hazards and command ordering;
- forward, backward, replay, checkpoint, and tape plans;
- canonical public and derivative boundaries.

Linking consumes this plan without re-deriving physical policy.

The public C lifecycle is:

```text
load bundle -> resolve executable -> link schedule
            -> write record -> call -> wait for the point
```

The Runtime records and submits backend commands internally
([`../runtime/design.md`](../runtime/design.md) sections 2 and 8).

## 8. ProgramGraph composition

`ProgramGraph` composes independently cooked Programs. It does not accept
resolved executables, Stages, backend resources, command encoders, or callback
passes. Resolving it loads each child once and composes only their ABI: the
composed executable's boundary slots are the children's that stay visible,
and its schedule links the children's fragments
([`../runtime/architecture_v2.md`](../runtime/architecture_v2.md) sections 4
and 16.2). No merged Program is built or resolved.

Each graph node has an opaque identity and a local boundary/control namespace.
Equal child boundary names never collide. Names enter the composite public ABI
only when the graph explicitly exports a boundary; frame updates use stable
node-local binding tokens resolved during composition.

Adding a node selects one exact child variant through canonical typed
specialization assignments. Selection is node-local: two nodes may instantiate
the same bundle at different pre-cooked specialization constants. Boolean capabilities and
non-Boolean specialization values share the typed variant-key representation;
rank and other constants must not be encoded as feature-name strings.
Selection never generates a missing signature or artifact and never invokes a
compiler.

Graph construction creates explicit graph Value handles from child outputs,
connects those Values to child inputs, and creates explicit graph Storage
handles whose appended child boundaries form ordered version chains. A graph
Storage is bound once per call; node-local tokens in that chain resolve to
the same composite boundary. Concrete tensors, image views, shapes, strides,
offsets, controls, resources, and launch grids remain call arguments. In particular, two independently supplied native
image handles do not establish a compositional alias. Graphics Programs that
share a framebuffer connect compatible attachment boundaries into one ordered
symbolic graph Image Storage-version chain; this is storage composition, not a
producer-to-consumer Value edge.

A child Program owns its semantic subgraph but never owns an independently
submitted physical command graph. The schedule linker orders the children's
operations, places their barriers and fuses adjacent passes into native render
scopes where their attachments allow it.

ProgramGraph composition adds no backward graph, derivative boundary, or
accumulation operation to the composite Program. A linked VJP of a graph of
differentiated children runs the children's backward graphs in reverse, each
child's input gradient feeding the cotangent of the output it reads
([`../autodiff/contract.md`](../autodiff/contract.md) section 9). Cotangent
fan-in belongs in a Module transformed and cooked as one VJP Program;
deployment-time scheduling never synthesizes it.

Standalone Program execution is the same model with an implicit one-node root
composition.

External engine objects may contain graph node handles and handle arbitrary
events. The engine updates node-local bindings explicitly; Vernon does not own
or call external polymorphic objects. Only the call of the enclosing linked
schedule submits work.

## 9. Publication

Outputs are written in place, and value semantics come from fresh outputs and
declared donation ([`../runtime/architecture_v2.md`](../runtime/architecture_v2.md)
section 10). A failed check writes nothing; a failed submission poisons what
it writes, and nothing is rolled back. The manifest's per-output
`publication` mode is still emitted but not read
([`../runtime/design.md`](../runtime/design.md) section 6).

## 10. Differentiated Programs

An explicit VJP transform produces one Program with:

```text
ForwardGraph(X) -> (Y, R)
BackwardGraph(R, dY) -> dX
```

- `X` and `Y` are primal public boundaries;
- `dY` and `dX` are cotangent and gradient boundaries;
- `R` is internal residual state.

Program VJP is the only authority that constructs reverse graph topology and
cotangent accumulation. Kernel VJP supplies differentiated compute Stage
implementations and opaque tape ABI. Tape is one residual storage strategy,
not another executable model.

A forward call returns residuals: the Values, tapes, or checkpoint prefix the
linked VJP selected. Each backward call over them writes its gradients in
place and may be repeated with other cotangents.

Graphics execution is currently primal-only. Capture rejects Program or Module
VJP when graphics Nodes are present; path-selective pruning of unrelated
graphics Nodes is not part of the current contract.

The normative differentiation contract is
[`../autodiff/contract.md`](../autodiff/contract.md).

## 11. Graphics normalization

Compute and graphics use the same Program Value and Storage model.
Source-level render conveniences normalize before deployment into graphics
Node fields and ordinary Program Values:

- attachments and their load/store/clear/resolve behavior;
- immutable graphics state;
- draw controls;
- independent shader resource bindings.

Render state is static Stage specialization. Render pass, draw command, and
dynamic state are call controls. Draw counts are explicit and are not
inferred from vertex inputs.

Image subresource versions and attachment continuity participate in the same
resource-hazard model as compute. Native render-pass fusion is a linking
optimization and must preserve attachment semantics.

Details are in [`graphics_execution.md`](graphics_execution.md).

## 12. Variant and artifact rules

One Program bundle may contain multiple canonical typed specialization variants. Each
variant:

- identifies one exact target ArtifactSystem;
- includes only reachable Stage artifacts;
- maps every portable Stage ID to exactly one implementing Stage artifact;
- validates aggregate compiler and runtime requirements before code loading.

Stage artifact identity covers target, requirements, authenticated modules,
entry points, reflection, and the current compiler/Program contract pair.
Artifacts from unsupported Program versions are rejected, not normalized.

## 13. Invariants

- Every executable is a Program.
- ProgramGraph composition links its children's fragments and merges no
  Program IR.
- Program, logical resolve, physical resolve, and a linked schedule form the
  only deployment path.
- Node count and Stage kind never select another architecture.
- Program semantic state is immutable after cooking.
- Runtime data never leaks into compiled artifacts or manifests.
- Program and Stage ABIs are connected only by explicit validated projections.
- Compute and graphics share Program Value binding and resource transitions.
- Unsupported mappings and derivative paths fail closed.
- A primal call has no AD work unless the caller explicitly links a VJP.
- Residuals are reusable: a backward call does not consume or mutate them.
- Transfers, residency, and hazards are resolved plans rather than
  execution-time guesses.
- Legacy schemas, loaders, direct Stage asset execution, and compatibility
  normalization are not accepted.
