# Autodiff design

> **Status:** accepted contract and implemented compute VJP surface. Compiler
> contract 12 and pipeline contract 16 provide deterministic
> `primal`/`forward_with_tape`/`backward` profiles, typed derivative groups,
> checked dynamic tape, explicit accumulation plans, and invocation-time
> `(x,y,z)` workgroup grids with physical invocation carriers. Structured CPU VJP supports
> direct and cooked void Kernels with explicit Storage objectives, recursive
> Scalar/Tensor/Tuple/Struct Storage elements, dynamic and signed-stride
> TensorViews, mutable scratch versioning, structured branches and loops,
> runtime gather/scatter accumulation, reusable pullbacks, and fresh packed
> tangent owners. Direct and cooked execution normalize the same native
> derivative metadata and share one structured CPU executable and pullback
> implementation.
>
> Structured CPU VJP is the only supported autodiff execution and cooking path.
> Execution graphs compose structured CPU pipeline pullbacks in the native C++
> graph scheduler. Browser wasm32 graph VJP, GPU and graphics autodiff,
> graphics backward lowering, custom compute VJPs,
> higher-order AD, persistent `.grad`, and unrestricted temporal
> differentiation remain deferred. GPU targets may still compile and run
> ordinary non-AD compute and graphics pipelines.

> **Breaking architecture target:** The profile and direct/cooked descriptions
> below are historical facts about the currently implemented structured-kernel
> VJP path, not future deployment alternatives. The coordinated release makes
> every standalone compute executable a one-node Program and expresses VJP only
> as Program forward/backward graphs linked by residual state. Kernel structured
> VJP remains a compute-node implementation detail. After that release, Runtime
> does not load an old profile manifest or normalize it into a Program. See
> [`unified_program_vjp.md`](unified_program_vjp.md).

This document defines Vernon's first public automatic-differentiation model.
The design uses reverse-mode vector-Jacobian products (VJPs) and preserves the
Value/Storage/Resource split. The implemented surface differentiates CPU
Kernels; differentiated graphics stages remain future work.

## 1. Design principles

- Source `@kernel`, `@vertex`, and `@fragment` signatures never gain a
  `grad_or_not` parameter.
- `vd.ad.vjp(program, ...)` is the only initial public reverse-mode transform.
  `grad`, `value_and_grad`, an implicit Tape, and a combined `apply` shortcut
  are not separate initial APIs.
- `wrt` is the only declaration of differentiated inputs. There is no
  `requires_grad` flag, implicit `.grad` state, global gradient clearing, or
  context exit that silently executes a backward pass.
- One execution returns `(outputs, pullback)`. Applying the pullback to output
  cotangents returns gradients for the declared `wrt` inputs.
- Primal and differentiated artifacts have separate deterministic identities.
  An ordinary primal invocation never pays tape or backward costs.
- Unsupported differentiation is rejected before artifacts or Runtime state
  are produced. The compiler never substitutes a zero gradient silently.

The initial public surface is deliberately VJP-only. JVP, full-Jacobian
materialization, higher-order differentiation, and convenience aliases may be
added later without changing the VJP contract.

## 2. Mathematical contract

For a specialized program

```text
Y = F(X)
```

the pullback computes

```text
dX = J_F(X)^T dY
```

where `dY` has the same differentiable Value/Storage structure as `Y`, and
`dX` has the structure selected by `wrt`.

Aggregate Storage cotangents and gradients preserve this logical structure at
the public API boundary. Compute-node ABIs may transport canonical leaves
independently, but the host reconstructs them through one structural
`TangentLayout` and one packed tangent owner per primal Storage owner.

Autodiff does not require `Y` to be scalar. For
`Y.shape = (2, 2, 2)` and `X.shape = (2, 2)`, the full Jacobian has logical
shape `(2, 2, 2, 2, 2, 2)`. The initial API does not materialize that matrix; it
computes its product with a supplied output cotangent.

If the output is one floating Scalar, `pullback()` may omit the cotangent and
uses the unique natural seed `1`. Tensor, Tuple, Struct, Storage, and
multi-output results require an explicitly matching cotangent. There is no
implicit Tensor `sum` or `mean`.

A downstream scalar objective commonly supplies a non-scalar cotangent. For
example,

```text
L = mean((image - target)^2)
d_image = 2 * (image - target) / element_count
```

and the renderer pullback maps `d_image` to scene gradients. Graph-level
reverse traversal is not currently supported; callers compose intermediate
cotangents explicitly.

## 3. Value, Storage, and Resource gradients

Floating Scalar leaves are differentiable Values. Tensor, Tuple, and Struct
Values derive gradient structure recursively from floating leaves.

- A floating Scalar gradient is an ordinary Scalar Value.
- An immutable Tensor gradient is an ordinary Tensor Value with the same
  logical shape.
- A TensorView or mutable Storage gradient is newly owned Storage at the public
  pullback boundary. In the compute-node ABI it is a writable TensorView
  endpoint of the backward stage, not a fixed-shape SSA result.
- Vector, Matrix, Tensor, Tuple, and Struct Storage elements derive one
  structural tangent schema recursively. Their public gradient is one packed
  tangent `TensorStorage`, not one owner per ABI leaf.
- Integer and Boolean leaves are non-differentiable unless an explicit custom
  rule consumes them without requesting a gradient.
- Resource handles and sampler state are non-differentiable.

The initial gradient accumulation types are:

```text
f16 primal -> f32 gradient
f32 primal -> f32 gradient
f64 primal -> f64 gradient
```

Gradient Storage is separate from primal Storage and never changes primal
type, identity, ownership, or layout. The first API does not expose
`accumulate_into` or persistent `.grad` state. Explicit reusable accumulation
buffers are a later, independent Runtime feature.

Dynamic Storage extents remain dynamic through differentiation. Backward
compute-node implementations receive descriptor-backed shape-source TensorViews
for primal Storage identities, descriptor-backed cotangent TensorViews, and writable gradient
TensorViews. Invocation-local adjoint buffers use those runtime extents;
compiler analysis and reflection must not specialize `dyn` dimensions. This
does not change the tape allocator ABI.

External gradient destinations carry mandatory ownership metadata derived by
autodiff analysis: `invocation_private`, `workgroup_shared`, `atomic_shared`,
or `none`. Static and dynamic TensorViews use the same rule; shape size never
selects ownership. Missing metadata fails Program resolution, and unsupported
shared accumulation fails lowering instead of selecting shared staging or
serial execution.

Forward and backward compute stages each reflect the same internal dispatch
contract used by ordinary compute. `ResolveProgram` validates both contracts
against the requested primal grid before constructing an effect transaction,
allocating tape, or staging gradients, and validates the backward contract
again when a reusable pullback is applied.

The tangent layout is independent from the primal layout. Dtype promotion may
change element size, alignment, product offsets, and element stride.
Non-differentiable product children remain logical `Zero` nodes and occupy no
tangent bytes. A tangent layout is structural metadata, not a generated
nominal DSL Struct declaration, and it must never reinterpret primal bytes.

The recursive tangent schema is:

```text
T(f16) = Scalar(f32)
T(f32) = Scalar(f32)
T(f64) = Scalar(f64)
T(bool | i32 | u32) = Zero
T(Tensor[S, shape]) = Tensor(shape, T(S))
T(Vector[S, n]) = Tensor((n,), T(S))
T(Matrix[S, m, n]) = Tensor((m, n), T(S))
T(Tuple[A, ...]) = Product(0:T(A), ...)
T(Struct{a:A, ...}) = Product(a:T(A), ...)
```

`TangentLayout` adapts this schema to the same canonical host ABI planner used
for ordinary Values. It does not maintain an independent alignment or offset
algorithm. Product-valued Tensor elements retain their element coordinates in
canonical derivative paths; `Zero` projections fail deterministically.

All compatible `wrt` views of one primal allocation accumulate into one fresh
tangent owner. Separately bound overlapping read/write aliases remain invalid.
A legal in-place `read_write` binding is differentiated through explicit
Storage versions.

## 4. Source and asset declaration

`pipeline_asset()` remains the only cookable declaration. For supported
autodiff cooking, its `program=` operand is a VJP `ProgramExpression`. A CPU
Kernel operand normalizes to a one-node Program before VJP construction.

```python
loss_asset = vd.pipeline_asset(
    id="pipeline/loss",
    program=vd.ad.vjp(
        loss_kernel,
        wrt=("parameters",),
        outputs=("loss",),
    ),
    variants=((),),
)
```

`vd.ad.vjp(...)` in this position is a statically parsed program transform,
not an execution call and not another asset type.

An ordinary `program=(vertex, fragment)` or `program=kernel` declaration cooks
one primal Program. A VJP program expression cooks one differentiated Program
containing forward and backward graphs, residual state, and versioned
cotangent/gradient signature metadata. The cooker does not emit a three-profile
topology.

The cooker CLI and descriptor reference remain unchanged:

```text
vernon-cook-pipeline model.py:loss_asset --target cpu -o build/loss
```

There is no `ad_pipeline_asset()` and no `autodiff=True` Boolean.

## 5. Python execution

After Runtime resolves the cooked Program:

```python
program = vd.resolve_program(
    "build/render/render.program.json",
    features=(),
)
image, pullback = program.vjp(bindings, grid=(grid_x, grid_y, grid_z))
gradients = pullback(d_image)

d_roughness = gradients["material.roughness"]
d_vertices = gradients["vertices.position"]
```

The binding names and paths accepted by `wrt` are canonical reflected source
paths. They are fixed at cook time and cannot be changed by a Runtime call.
Program signature groups transport structured output cotangents and
aggregate gradients through flattened, fully qualified leaf paths such as
`output.color` and `material.roughness`. The public pullback API groups those
leaves by the declared output or `wrt` path and packs aggregate Storage through
its reflected `TangentLayout`; leaf paths are not separate public Storage
owners. Every Program variant must expose identical groups and leaf paths.

The serialized Program stores derivative authority only in Signature.
`ResolveProgram` projects one canonical typed derivative-group table in
gradient-then-cotangent order, validates canonical paths, unique group
ownership, and the executable signature once, and stores the table in
ResolvedProgram. Interactive and cooked APIs read groups from ResolvedProgram;
neither has a second direct/cooked grouping or execution model.

The public C Runtime exposes indexed reflection rather than a scalar-output
special case:

- `vernonRuntimeLoadedPipelineGetAdOutputCount` and
  `vernonRuntimeLoadedPipelineGetAdOutputByIndex`;
- the equivalent cotangent and gradient count/index pairs;
- derivative-group count/index queries and indexed leaf-path queries.

Each value query returns one `VernonAdValueMetadataView` containing the
canonical path, tangent dtype, rank, and logical shape. This is the only public
output/cotangent/gradient reflection API and supports aggregate and independent
multi-output objectives without reinterpretation.

For launches larger than one invocation, packed aggregate cotangents prepend the
physical `(grid.z*workgroup.z, grid.y*workgroup.y, grid.x*workgroup.x)` carrier
dimensions to the primal owner shape. TensorView
offsets and signed strides apply to the trailing owner dimensions, so
carrier packing does not erase subview descriptors.

The CPU static tape stride is the fixed compiler/manifest estimate. The Runtime
does not calibrate later dispatches from maximum observed per-lane usage;
runtime calibration was removed because it inflated subsequent reservations.

The positive three-dimensional workgroup grid is supplied for each forward invocation;
it is not part of Program semantic identity or cooked artifact identity. Each
workgroup contains the reflected
`workgroup_size`; invocation carriers use physical `(z, y, x)` order, with X
fastest. A pullback retains its forward grid and
always runs backward over that same logical invocation domain.

A no-Tape or bounded-replay pullback owns immutable primal inputs/versions and
launch geometry rather than whole-dispatch Tape. A retained static pullback owns
immutable CPU Tape only when its selected `balanced`/`min_runtime` plan fits the
hard context budget. Every form may be called sequentially with
multiple cotangents; each call returns fresh gradients and does not consume or
mutate retained state. Destroying the pullback releases that state. The synchronous
Runtime contract does not promise concurrent calls on one pullback.

### CPU range-phase execution

CPU primal, forward-with-tape, and backward compute implementations all use the
ordinary range-phase scheduler described by
[`runtime/design.md`](runtime/design.md#cpu-range-phase-execution). There is no
AD scalar-entry adapter, worker-local lane identity, fixed-tape execution
branch, or serial fallback.

Runtime stores packed argument/result frames in dispatch-level contiguous
slabs and supplies flattened-global pointer tables to each scheduler range.
The generated wrapper invokes the lowered program for every lane in the
contiguous interval without a Runtime callback per lane. Frame storage persists
while a lane is yielded; completion clears its pointer-table entries without a
per-lane heap allocation. Pointer-table bounds are explicit in
`VernonCpuRangeV1`.

One `HostStaticTapeBatch` owns either an admitted complete static dispatch or one
complete-workgroup replay segment. Straight-line lanes write compiler-assigned
fixed residual offsets. A lane that opens nested
control flow promotes into the batch's shared `HostDynamicTapeBatch`, whose
POD lane state appends to chunked payload, child, region, and record arenas.
Descriptors carry their batch identity directly, so semantic callbacks do not
acquire a process-global ownership-registry lock. Final completion validates
every lane and compacts dynamic metadata into page-layout-v1 arrays; immutable
readers are range-local views over the retained batch.
Compiler-generated code sees only the versioned semantic allocator callbacks
and opaque handles; Runtime alone owns payload, region, record, and index
representation. CPU autodiff lowering also inspects function argument and result
types, so a logical Tape handle is lowered even when canonicalization leaves no
operation-level use from which to rediscover it; logical handles never reach the
CPU ABI wrapper.

Backward execution creates fresh mutable lane/workgroup phase state for each
pullback application. The original bounded forward uses the Tape-free primal
implementation and retains one immutable prepared input shadow; it does not construct
and discard per-workgroup Tape. Bounded replay restores that shadow, replays one
complete workgroup with its original virtual IDs, applies backward into shared
transactional gradient destinations, and releases that segment's Tape. Reverse
barriers are lowered through the same coroutine/yield machinery as primal
barriers. Invocation-private gradients follow logical lanes across workers;
workgroup-shared gradients remain in the group arena; atomic-shared updates use
the declared target capability. External gradients are accumulated and
published only after every backward workgroup completes successfully.

Before dispatch, checked policy accounting covers tape reservation,
cotangent carriers, invocation-private or workgroup-shared gradient staging,
and arithmetic overflow. Forward Storage/output writes use
`HostEffectTransaction` shadows and commit exactly once only after every lane
has completed and the dispatch batch is valid. Failure discards shadows,
construction arenas, compact metadata, frame slabs, group arenas, staging, and
dispatch reservations. Parallel range failures collect diagnostics without
writing shared invocation state from worker threads.

## 6. Typed transform and cache identity

Autodiff runs after source loading, feature/constant specialization,
monomorphization, type inference, and effect validation, but before target
lowering.

An immutable `ProgramTransformSpec` records:

- transform kind (`vjp`);
- canonical `wrt` leaf paths;
- output cotangent paths;
- gradient element-type policy;
- semantic accumulation operations and determinism requirements;
- bounded tape/checkpoint policy;
- derivative-rule and custom-rule-set versions.

The transform spec participates in frontend semantic identity, compiler cache
identity, symbols, reflection, cooked stage identity, and the pipeline
manifest. It does not alter primal helper specialization keys.

The compiler's typed Program IR represents one specialized program's Value
flow, structured kernel-local control semantics, Storage effects, alias
regions, differentiability boundaries, saved Values, and reverse dependencies.
Program VJP is the sole authority that constructs deployment forward,
backward, and residual topology. Kernel structured VJP only supplies a selected
compute-node implementation and its tape ABI.

## 7. Stateful Kernel differentiation

Stateful reverse mode requires all of the following:

- functionalize local mutation and TensorView writes where semantics permit;
- preserve typed read/write/atomic/barrier effects;
- prove alias, injectivity, and owner-lifetime constraints;
- generate `ReduceSum` and `ScatterAdd` adjoints, with `DisjointScatter`
  evidence only when injectivity is proven independently of one invocation's
  grid;
- define bounded branch/loop tape layouts and reject unbounded tape;
- keep primal, tape, cotangent, and gradient bindings explicit in reflection.

A target lowers semantic accumulation to disjoint direct stores, a supported
atomic-add fast path, or canonical X-fastest deterministic serial reduction.
Backend capability does not change the frontend derivative graph.

## 8. Deferred graphics and custom VJP rules

Compute GPU autodiff supports no-Tape and captured static/dynamic Storage
pullbacks on CUDA, Vulkan, DirectX 12, Metal, and OpenGL. Captured implementations use
complete-workgroup bounded replay with original virtual IDs, device-local Tape,
fixed lane-status readback, transactional gradient publication, and no
GPU-to-host Tape payload readback. RHI graph resources provide checkpoint
snapshots for graph replay. Inputs and final gradients cross the host API
boundary; derivative execution and temporary Tape/gradient storage remain
backend-local. Graphics autodiff remains unsupported. Graphics support must
account for the fact that
differentiating vertex and fragment functions independently does not define a
differentiable graphics pipeline. The Pipeline ProgramGraph would model varying
interpolation, rasterization, visibility, depth, blending, and texture
sampling as versioned stage-boundary primitives.

Ordinary differentiable arithmetic uses built-in versioned rules.
Rasterization, visibility, depth, blend, and texture sampling require a named
custom VJP rule set. Each rule declares:

- primal inputs and outputs;
- saved Values and tape bounds;
- accepted cotangent and produced-gradient ABI;
- supported formats, topology, sampling modes, and backend capabilities;
- behavior at discontinuities.

Missing or incompatible rules are compile errors. Resource handles and sampler
state remain non-differentiable. A texture rule may produce gradients for
coordinates and for texel data explicitly exposed as differentiable Storage;
it does not differentiate the Texture handle.

Rule-set identity participates in compiler and pipeline contracts, cache keys,
reflection, and manifests.

## 9. ExecutionGraph composition

The following is the shipped pre-breaking composition behavior, retained here
as implementation history. It is removed as a second AD topology by the
coordinated Program release.

Python `ExecutionGraph` builders may name differentiable resource or execution
parameter inputs and Storage objective resources. A `VjpComputePass` binds one
directly compiled or cooked structured CPU VJP to graph resources. Compilation
snapshots the derivative signature with the immutable execution plan.
Automatic checkpoint planning currently requires directly compiled passes,
because the frozen pipeline contract does not expose cooked tape-size and
replay-cost metadata; cooked graph VJPs remain available without checkpoint
planning.

In the target architecture, Program forward/backward/residual graphs are the
only top-level AD topology. Native ExecutionGraph consumes that topology for
hazards, checkpointing, replay, and submission; it does not discover a second
reverse graph from passes or compose profile pullbacks.

`CompiledExecutionGraph.vjp()` calls the native graph VJP entry and returns a
`GraphPullback` retaining that submission, the plan, resources, checkpoints,
and required pipeline pullbacks. The C++ graph core traverses scheduled
differentiable passes in reverse order, replays bounded segments through the
compiled scheduler, supplies zero cotangents to inactive local outputs, and
deterministically accumulates contributions by logical graph identity. One
scalar objective may omit its cotangent; multiple or structured objectives
must provide an exact name-to-cotangent mapping. A non-differentiable write on
an active reverse path is an error.

The canonical backward operation is `GraphPullback.submit()`, which returns a
submission carrying named gradients. Native CPU execution may complete inline;
calling the pullback directly is the synchronous shorthand. Pullbacks are
reusable and outlive the builder and caller's compiled-plan handle.

The orchestration core and type-erased cotangent accumulation are native C++.
Python adapts values at the binding boundary but does not traverse the schedule
or snapshot graph resources. Statically linked wasm32 structured VJP entries
remain required before CPU/WebAssembly graph VJP is complete.

## 10. C and C++ deployment API

C++ resolves a differentiated Program and invokes its VJP:

```cpp
auto program = bundle.resolveProgram("pipeline/render");
auto [image, pullback] = program.vjp(bindings, {gridX, gridY, gridZ});
auto gradients = pullback({{"color", dImage}});
```

The application-facing C ABI uses an opaque pullback handle:

```c
VernonPullback *pullback = NULL;
VernonLaunchSize grid = {grid_x, grid_y, grid_z};
vernonProgramVjpForward(program, grid, &inputs, &outputs, &pullback);
vernonPullbackApply(pullback, &cotangents, &gradients);
vernonPullbackDestroy(pullback);
```

The C structures use `struct_size` and reserved fields. C++ provides RAII over
the same ownership rules. Deployment never runs Python or performs the
autodiff transform.

## 11. Versioning and acceptance

Compiler contract 12 and pipeline contract 16 remain the current shipped CPU
VJP boundary until the coordinated release; their profile behavior above is
historical implementation fact. The breaking Program release updates the
contracts together and intentionally rejects all older profile manifests. It
does not reinterpret, normalize, or retain them as a parallel loading path.
The target CPU VJP has one Program/ResolveProgram/ExecuteProgram path.
Differentiated graphics remains unsupported.

Current pre-breaking CPU acceptance covers direct and cooked structured Storage VJP, recursive
aggregate tangents, multiple outputs, owner aliases, signed-stride descriptors,
multi-invocation cotangent carriers, dynamic control flow, scratch overwrite,
finite differences, and deterministic reusable pullbacks. Runtime acceptance
also freezes tape allocator ABI, ownership, memory charging, failure latching,
and public exception containment. GPU acceptance covers only ordinary non-AD
compute and graphics pipelines.

Any missing derivative or custom rule, unbounded tape, unsupported target
capability, uncertain write conflict, malformed group/cotangent/signature,
or stale contract version fails before execution.
