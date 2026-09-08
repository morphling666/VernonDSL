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
- **Program instance** owns persistent binding state.
- **Program invocation** owns one concrete set of bindings and controls.
- **Pullback** owns immutable retained forward state for reusable backward
  applications.
- **ExecutionGraph/Command DAG** is private runtime scheduling machinery, not a
  public authoring or deployment model.

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
  -> ProgramInstance
  -> ProgramInvocation
  -> private Command DAG and backend
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
   residency, hazards, publication transactions, and residual plans into
   `ResolvedExecutionPlan`.
8. Invocation binds runtime Values and controls and executes the resolved
   plan.

No lower layer reconstructs semantic information owned by a higher layer.
Names, endpoint order, dtype guesses, zero sentinels, and runtime heuristics
must not replace explicit IDs and validated projections.

## 4. Compute Stage contract

A compute Stage has three distinct phases:

```text
source + annotations + features -> lower -> portable IR
portable IR + target            -> compile -> Stage artifact
artifact + runtime bindings     -> invoke -> dispatch
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
Module state, constants, and Features. It may not depend on invocation Values
or device data. Dynamic per-element control flow belongs inside a kernel or
shader.

Initialized Module instances are compiler inputs, not serialized objects.
Artifacts contain no Module instance, `__dict__`, pickle payload, Python
callback, or constructor configuration tree.

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
- publication transactions;
- forward, backward, replay, checkpoint, and tape plans;
- canonical public and derivative boundaries.

Execution consumes this plan without re-deriving physical policy.

The public C++ lifecycle is:

```text
load bundle -> resolve executable -> create instance
            -> begin invocation -> bind -> invoke
```

The Runtime records and submits backend commands internally. External encoders
are an optional private embedding concern and do not define another Program
API.

## 8. Publication

Program outputs declare one publication mode:

- `commit_after_success` stages externally visible results and commits them
  only after successful execution;
- `in_place` permits direct externally visible mutation and cannot promise
  rollback of already executed writes.

Publication is planned during physical resolution. A failed
`commit_after_success` invocation rolls back staged publications. Image and
buffer publication use device transfers when residency permits; synchronous
host round-trips are not a publication strategy.

## 9. Differentiated Programs

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

Forward returns a reusable pullback retaining immutable snapshots, tapes,
resource versions, or replay/checkpoint descriptions selected by the resolved
plan. Each pullback application creates fresh invocation scratch and fresh
gradient publications.

Graphics execution is currently primal-only. Differentiation fails closed only
when the requested derivative path traverses a graphics Node.

The normative differentiation contract is
[`../autodiff/contract.md`](../autodiff/contract.md).

## 10. Graphics normalization

Compute and graphics use the same Program Value and Storage model.
Source-level render conveniences normalize before deployment into graphics
Node fields and ordinary Program Values:

- attachments and their load/store/clear/resolve behavior;
- immutable graphics state;
- draw controls;
- independent shader resource bindings.

Render state is static Stage specialization. Render pass, draw command, and
dynamic state are invocation controls. Draw counts are explicit and are not
inferred from vertex inputs.

Image subresource versions and attachment continuity participate in the same
resource-hazard model as compute. Native render-pass fusion is a private
Command DAG optimization and must preserve attachment semantics.

Details are in [`graphics_execution.md`](graphics_execution.md).

## 11. Variant and artifact rules

One Program bundle may contain multiple canonical feature variants. Each
variant:

- identifies one exact target ArtifactSystem;
- includes only reachable Stage artifacts;
- maps every portable Stage ID to exactly one implementing Stage artifact;
- validates aggregate compiler and runtime requirements before code loading.

Stage artifact identity covers target, requirements, authenticated modules,
entry points, reflection, and the current compiler/Program contract pair.
Artifacts from unsupported Program versions are rejected, not normalized.

## 12. Invariants

- Every executable is a Program.
- Program, logical resolve, physical resolve, and Program invocation form the
  only deployment path.
- Node count and Stage kind never select another architecture.
- Program semantic state is immutable after cooking.
- Runtime data never leaks into compiled artifacts or manifests.
- Program and Stage ABIs are connected only by explicit validated projections.
- Compute and graphics share Program Value binding and resource transitions.
- Unsupported mappings and derivative paths fail closed.
- Primal invocation has no AD work unless the caller explicitly requests VJP.
- Pullbacks are reusable and do not share mutable invocation scratch.
- Publication, transfers, residency, and hazards are resolved plans rather
  than execution-time guesses.
- Legacy schemas, loaders, direct Stage asset execution, and compatibility
  normalization are not accepted.
