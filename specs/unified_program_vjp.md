# Unified Program and VJP Architecture

## 1. Status and intent

This document defines the target architecture for unifying:

- standalone compute kernels;
- graphics pipelines;
- multi-node `Module` Programs;
- primal execution;
- explicit VJP execution;
- native `ExecutionGraph` scheduling.

Every standalone compute executable, standalone graphics executable, and
initialized `Module` is one Program. A standalone compute kernel or graphics
pipeline is a one-node Program; a single shader is likewise a one-node Program,
not a separate runtime architecture. Kernel and shader IR remain implementation
IR inside Program nodes.

This is the breaking target for a coordinated compiler/pipeline contract
release. The current compiler contract and pipeline contract versions must not
be changed before that release. Once released, loaders reject artifacts from
the old pipeline contract; there is no legacy normalizer or compatibility
manifest input.

## 2. Canonical layers

```text
Python kernel / graphics pipeline / initialized Module
    ↓
Typed semantic Program IR
    ↓
Optional Program transforms, including VJP
    ↓
Implementation selection and fusion
    ↓
KernelCompileRequest[]
    ↓
Compiled kernel/graphics artifacts
    ↓
Program object (`stages`, `parameters`, `storages`, `values`, `graphs`,
                `signature`)
    ↓
ResolveProgram
    ↓
Resolved Program
    ↓
ExecuteProgram / Native ExecutionGraph / Command DAG / backend
```

Responsibilities are separated as follows:

- **Program IR** owns public signatures, semantic operations, cross-node value
  flow, resource effects, forward/backward graphs, fusion boundaries and
  differentiation.
- **Kernel IR** owns scalar/thread semantics, kernel ABI, workgroup behavior,
  local structured control flow and kernel-level tape generation.
- **ExecutionGraph** owns resource hazards, barriers, checkpoint scheduling,
  replay, command coalescing, submission and resource lifetime.
- **RHI/backend code** owns native pipelines, queues, command encoders and
  completion.

No layer reconstructs semantic AD from a lower layer. `ResolveProgram` is the
only deployment validation and binding boundary. Every execution path,
including any optimized single-node path, consumes its resolved result.

## 3. Single-kernel normalization

All public executable inputs enter the Program pipeline.

### 3.1 Primal kernel

```text
Program {
  ForwardGraph {
    node 0 = compute(kernel implementation)
  }
}
```

The compiler/runtime may optimize a single-node dispatch only after
`ResolveProgram` has produced the canonical resolved Program. That fast path is
an execution optimization, not a separate manifest, loader, name-binding path,
or runtime model.

### 3.2 Differentiated kernel

```text
DifferentiatedProgram {
  ForwardGraph {
    node 0 = compute(kernel forward implementation)
  }
  BackwardGraph {
    node 0 = compute(kernel backward implementation)
  }
  ResidualContract {
    kernel tape and/or saved primal values
  }
}
```

Kernel structured VJP still generates the kernel forward/backward
implementations and owns the opaque tape ABI. The top-level execution topology
is expressed only by Program graphs.

### 3.3 Multi-node Module

A Module differs only in node count. Program VJP constructs one backward graph
for the whole semantic Program. Each selected compute node may delegate its
local derivative implementation to kernel structured VJP.

This removes the long-term distinction between:

- “kernel pipeline execution”;
- “Module Program execution”;
- “single-pipeline autodiff execution”;
- “graph autodiff execution”.

## 4. Differentiated Program

An explicit VJP transform produces one logical object:

```text
DifferentiatedProgram
  - ForwardGraph
  - BackwardGraph
  - ResidualContract
  - public primal/cotangent/gradient signature
```

Its mathematical interface is:

```text
ForwardGraph(X) -> (Y, R)
BackwardGraph(R, dY) -> dX
```

where:

- `X` is the public primal input;
- `Y` is the public primal output;
- `dY` is the public cotangent input;
- `dX` is the public gradient output;
- `R` is internal pullback state described by the residual contract.

Forward and backward remain separate logical graphs because they execute at
different times and a pullback owns state between them. Logical separation does
not prevent the native command planner from coalescing compatible static
forward, replay and backward work into one submission.

The callable public API remains:

```python
outputs, pullback = vd.ad.vjp(program, wrt=...)(inputs)
gradients = pullback(cotangents)
```

The pullback is reusable for sequential calls. Each application returns fresh
gradients and does not consume retained state. Concurrent calls on one
pullback are not promised.

An ordinary primal call uses a separately identified primal Program and must
not compile backward implementations, allocate tape or retain pullback state.

## 5. Residual contract

Tape is not a second graph model. It is one physical realization of residual
state.

At the executable Program boundary, kernel tape is a first-class
`!vernon.ad_tape` Program value with an opaque-resource ABI. Kernel VJP owns
the tape contract and physical contents; Program topology owns the handle's
producer, backward consumer, stable value identity and capture lifetime. There
is no stage-local residual side channel.

Residual state may contain:

- retained primal tensors or value versions;
- retained input/resource references;
- kernel-local opaque tape;
- structured control history;
- checkpoint snapshots;
- replay recipes and launch geometry;
- dynamic tape status.

The logical residual contract links backward Capture records to stable forward
value-version identities and carries:

- type and shape;
- producer and consumer;
- size/alignment or an opaque tape layout;
- liveness;
- retain/replay legality;
- replay and recomputation cost;
- determinism constraints;
- required resource versions.

Storage IDs identify allocation identity. Program value IDs identify SSA
version positions, instantiated with one concrete invocation and storage
generation at runtime. A mutation that backward may distinguish is an explicit
before/after ResourceTransition within one Storage. Residual replay metadata records deterministic replay
legality, required captured versions and target-relative recomputation cost;
the runtime policy chooses retention, checkpoint or replay.

The runtime creates a `PullbackState` as the concrete realization of this
contract. A capture may be satisfied by a retained value, retained kernel tape,
checkpoint restoration or replay. The backward graph must not encode one fixed
global policy such as “always replay the whole forward graph”.

Only dynamic status and requested public results may require host readback.
Compatible residuals, cotangents and gradients stay device-resident.

## 6. Program VJP authority

The MLIR Program VJP transform is the single reverse-construction authority.

`source/lib/Dialect/VernonProgram/Transforms/VernonProgramVjp.cpp` must:

1. consume one typed primal Program graph;
2. create the differentiated forward graph;
3. create a separate backward graph;
4. insert semantic cotangent accumulation operations at fan-in;
5. collect only primal values required by derivative rules;
6. expose those values as internal backward captures linked to forward value
   identities;
7. delegate custom compute-node derivatives to kernel structured VJP;
8. reject unsupported graphics differentiation.

It must stop cloning the complete primal computation into the backward graph.
Rematerialization is selected by the native checkpoint planner.

The existing Python reverse graph and `LoweredReverseProgram.execute` are
transitional. Python may capture and request VJP, but it must not remain a
second reverse-construction or reverse-execution authority.

## 7. Reflection and manifest

The field-level schema currently recorded in
[`program_execution_manifest.md`](program_execution_manifest.md). It defines
value origins, payloads, TensorView descriptors, canonical ValueLayout,
root-to-stage endpoint/leaf bindings, signature groups, resolution, and
fail-closed validation. At the coordinated breaking release it is emitted as
one Program object with the following direct fields; old nested/profile
artifacts are not accepted as alternate inputs.

The Program object has one execution topology:

```text
Program
  - stages             # logical stage ID -> portable StageContract
  - parameters[]       # canonical public/entry binding declarations
  - storages[]
  - values[]
  - shape_symbols[]
  - shape_constraints[]
  - alias_preconditions[]
  - graphs[]          # forward and optional backward
  - signature
  - residual_contract # required exactly with backward
```

Descriptor shape/extent information is carried by the owning Storage.
`StorageDescriptor` is the sole authority for allocation identity,
layout, extent, lifetime, and ownership. Stage control metadata is entry-only:
it may provide launch/draw values to a node entry but is not a resource binding
authority and cannot override a `StorageDescriptor`. Ordinary resources,
including textures and typed byte storage, use normal public output signature entries;
there is no separate resource-output channel.

The target Program contract admits only static DAGs of compute and graphics
nodes. Transfer nodes and deployment control-flow nodes (`if`, `loop`, `scan`,
or nested regions) are outside this contract. Kernel-local structured control flow remains an
implementation detail of a compute node.

Per-artifact reflection remains necessary for:

- exact target code modules, formats, Blob ranges, and entry points;
- stage kind;
- parameter ABI;
- workgroup size;
- descriptor/resource binding;
- dispatch contract;
- backend artifact metadata.

Per-artifact reflection must not duplicate the Program graph topology.
The ArtifactSystem has one exact target. Each StageArtifact carries its own
requirements, and each variant carries only its reachable aggregate.
StageArtifact IDs cover target, requirements, modules, entry points, and
reflection. Each cooked variant `stage_bindings` maps portable Program
StageContracts to those target artifacts. Compute has one compute module;
graphics has ordered vertex and fragment modules. Separate targets are never
mixed at resolution.

Program values and stage-local artifact endpoints are different ABI layers.
Each node explicitly binds an artifact argument/result index to a Program
value or ResourceAccess and, when decomposing an aggregate, a canonical `leaf`
index. A physical read-write endpoint maps to distinct semantic before/after
values in one storage transition.
Runtime resolves and validates this 1:N mapping once. Names, dtypes, duplicate
occurrence, or execution-time temporary variants must not be used to infer it.

Node dependencies are not a second serialized topology. ResolveProgram derives
producer edges from Value SSA and derives overlapping-resource RAW/WAR/WAW
edges from ResourceAccess version transitions and view ranges. Storage versions
form one non-branching chain, and serialized node order must be a topological
order of the complete derived edge set.

Compute and graphics share this same Program/Node/EndpointBinding model and one
public `bind(ProgramValues)` operation. Source and Program IR may contain
typed `RenderTarget`, `Attachment`, and `RenderState` builtins, but final
deployment normalization lowers them to graphics-node `attachments`, `state`,
and `draw` fields plus ordinary Program values. They are not serialized as a
resource-aggregate public ABI.

An `Attachment` owns one image view/use, `load` (`load`, `clear`, or `discard`),
its clear value when required, `store` (`store` or `discard`), and an optional
resolve target. `RenderState` owns blend, depth/stencil, raster, color write
mask, and multisample state; clear is never pipeline state. `RenderTarget` is
source convenience that internally allocates textures and aggregates
attachments. A draw carries an optional `index_buffer: TensorView`,
`instance_count` defaulting to one, and an explicit required `vertex_count`
for direct draws. Deployment never infers draw count from vertex inputs.
There is no `DrawArgs` aggregate.

Resolve destination load is implicit discard, while source and destination
store availability are independent.

Primitive topology is static pipeline/artifact specialization, not an
invocation value. Shader-visible Texture/ImageView/typed-byte-storage/Sampler resources
bind independently through ordinary endpoint bindings and cannot be nested in
ABI-stable structs. Stable texture identity lowers to one Storage ID; ordered
attachment uses lower to exact internal Value versions. There is no separate
GraphResourceHandle type, and graphics calls do not return replacement Texture
handles.

ImageView is a source/runtime descriptor-bearing alias, not a distinct Program
dialect type. Final Program Values for roots and views both use
`!vernon.texture`; view Origin metadata carries the exact parent version and
aspect/mip/layer descriptor. Sampled Texture types use `format = "unknown"`;
the image `StorageDescriptor` carries the concrete allocation format. Concrete Texture
formats apply only to storage access (`read`, `write`, or `read_write`).

Graphics attachment transitions are explicit Program version edges but remain
ExecutionGraph-owned effects for load/store visibility, subresource hazards,
native render-pass fusion, and submission failure. Fusion follows attachment
continuity, compatible views/geometry and load/store/resolve semantics, and the
absence of intervening hazards; it does not require `RenderState` equality
because state may change between draws in one native render pass. Graphics
remains outside the active VJP: any requested derivative path through a
graphics node fails closed, while unrelated graphics nodes do not create
captures or derivatives.

Public outputs are identified independently from internal residual results.
Forward values consumed by backward appear in the backward graph's explicit
capture boundary, exactly matching ResidualContract records. Backward
cotangents remain external tagged inputs; residual captures are internal
bindings supplied by `PullbackState`.

The breaking loader accepts only this Program topology. It does not load or
normalize root `autodiff.profiles`, stage-name binding tables, copied
`parameters`/`internal_parameters`/`uses` tracks, or direct stage topology.

## 8. Compilation flow

Compilation remains two-phase:

1. C++/MLIR transforms, fuses and partitions Program graphs.
2. Planning returns `KernelCompileRequest[]`.
3. Python DSL providers compile requested implementation regions.
4. C++ finalization validates exact request coverage and ABI compatibility.
5. Bundle planning writes exactly one Program object and its artifact-system
   stage endpoints.

Program VJP runs before implementation selection and fusion. Kernel structured
VJP is selected as the implementation of differentiated compute requests; it
does not create another top-level Program.

## 9. Runtime model

`ResolveProgram(Program, StageBindings, ArtifactSystem, Target)` produces one immutable
resolved Program owner containing:

- authenticated target code modules and exact entry points;
- resolved forward node stages and derived dependency edges;
- resolved backward node stages when differentiated;
- public signature and derivative groups;
- residual contract;
- checkpoint/replay planning metadata;
- backend-independent value/resource mapping.

`ExecuteProgram` accepts only this owner. Interactive standalone execution,
cooked execution, graphics execution, and Module execution do not bypass
resolution.

Primal invocation:

1. binds all public Program Values through one binding operation;
2. materializes Program values and normalized compute/graphics nodes;
3. submits `ForwardGraph` through ExecutionGraph.

VJP forward:

1. submits `ForwardGraph`;
2. retains the resources and planning state required by the selected residual
   plan;
3. returns public outputs and a reusable `PullbackState`.

Pullback application:

1. binds public cotangents;
2. resolves each backward capture from retention, tape, checkpoint or replay;
3. materializes and submits `BackwardGraph`;
4. publishes fresh public gradients transactionally.

The ExecutionGraph checkpoint planner receives the complete primal/reverse DAG.
Runtime AD emits replay, checkpoint and derivative command nodes; it does not
perform hidden nested submit/wait operations behind the graph scheduler.

## 10. Reuse and retirement

Reuse:

- tape allocator ABI and host tape storage;
- derivative `Signature`/`ValueAbi` validation;
- derivative group metadata;
- memory policies and accounting;
- GPU replay/tape command construction;
- `PassPullback`, `GraphPullback` and checkpoint planner;
- failure, transaction and deterministic accumulation semantics.

Remove at the breaking release:

- `VernonLoadedAutodiff` as a separate single-pipeline owner;
- monolithic three-profile `ad::Executable`;
- variant-level autodiff topology as the primary model;
- Python `LoweredReverseProgram.execute`;
- Python `_RematerializedPullback` prefix replay;
- duplicate direct/cooked/graph VJP orchestration;
- artifact compatibility normalizers, name binding, and copied
  `parameters`/`internal_parameters`/`uses` authority;
- direct stage topology and profile-manifest loading.

## 11. Current implementation state

Already present:

- MLIR forward/backward Program graph representation;
- Python parser residual forward results and backward arguments;
- Program reflection and `KernelCompileRequest[]`;
- C++ request finalization and ABI checks;
- Program bundle planning;
- manifest parsing and validation;
- CPU forward Program materialization into native `ExecutionGraph`;
- multi-node dependency and internal tensor materialization.

Still required:

- one reverse-construction authority in MLIR;
- capture-based MLIR backward instead of full primal cloning;
- native backward graph resolution and execution;
- reusable Program `PullbackState`;
- checkpoint planner integration with Program residual captures;
- standalone kernel normalization to one-node Program;
- GPU Program resource materialization;
- removal of transitional Python reverse execution;
- unified graphics Program-value binding and normalized graphics-node
  materialization; graphics differentiation remains outside the active VJP.

## 12. Migration order

1. Normalize standalone compute and graphics executables to one-node Programs.
2. Define one residual/capture linkage accepted by MLIR, reflection, manifest
   validation and runtime.
3. Make MLIR Program VJP authoritative and remove unconditional primal cloning
   from backward.
4. Normalize standalone kernel VJP to one-node forward/backward Program graphs;
   kernel structured VJP remains only the compute-node implementation.
5. Materialize native backward Program graphs and reusable pullback state on
   CPU.
6. Route retention, checkpoint and replay through the native ExecutionGraph
   planner.
7. Add GPU Program value/resource materialization and device-resident backward.
8. Route interactive, cooked and Module VJP APIs through the Program owner.
9. Delete old artifact/profile loading and transitional runtime/Python
   orchestration in the coordinated breaking release.
10. Normalize source/IR graphics builtins into graphics-node attachments,
    state and draw fields, bind shader resources independently as Program
    Values, and materialize stable Texture identities with explicit internal
    versions. Graphics remains outside the active VJP.

## 13. Core invariants

- Every executable is a Program; node count does not select architecture.
- Program, ResolveProgram, and ExecuteProgram are the only deployment path.
- The target Program contract is a static compute/graphics DAG; transfer and
  deployment control flow are excluded.
- Kernel IR and Program IR remain separate levels.
- Normal primal invocation has no AD overhead.
- VJP is explicit and has a separate deterministic identity.
- Forward and backward are separate logical graphs linked by residual state.
- Tape is a residual storage strategy, not a top-level execution model.
- Pullbacks are reusable and own their retained state.
- Program VJP is the only layer that discovers cotangent fan-in.
- Kernel VJP owns kernel-local tape ABI only.
- ExecutionGraph owns checkpoint, replay, hazards and submission.
- Per-artifact reflection describes ABI, not Program topology.
- Program root ABI and stage-local ABI are linked only by validated endpoint
  bindings defined in `program_execution_manifest.md`.
- Compute and graphics use one Program Value binding path; normalized graphics
  node fields are not a second public resource-aggregate ABI.
- Graphics topology is static specialization, and shader resources cannot hide
  inside ABI-stable structs.
- Unsupported differentiation and resource mappings fail closed.
- `StorageDescriptor` is resource authority; control metadata is entry-only and
  resources use ordinary public outputs.
- Existing contract versions are not bumped before the coordinated release.
- Artifacts from before that release are rejected, not normalized.

## 14. Acceptance criteria

- A standalone kernel and an equivalent one-node Module produce equivalent
  Program reflection and bundle topology.
- Primal standalone kernels and Modules use the same loader and Program runtime.
- Differentiated standalone kernels and Modules use the same forward/backward
  graph and pullback path.
- Ordinary primal calls allocate no tape and compile no backward stage.
- Backward does not unconditionally replay the complete forward graph.
- Retain and rematerialize policies produce equal gradients.
- Kernel opaque tape crosses the Program residual boundary without host payload
  readback on compatible GPU paths.
- Repeated pullback calls with different cotangents produce fresh correct
  gradients.
- Fan-out accumulation is explicit Program IR.
- CPU and supported GPU paths preserve deterministic, failure-transaction and
  memory-budget behavior.
- Full C++ and Python test suites pass throughout migration.
